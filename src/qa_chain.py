"""qa_chain.py - grounded QA over the Phase 1 retriever (Phase 4: optional source filter).

Flow: retrieve(filter_source) -> score gate -> prompt -> Gemini -> parse/validate (1 retry)
      -> citation post-validation on (source, page) -> QAResult.

Importable with no side effects: the LLM is built lazily inside answer_question().
Backward compatible: filter_source is the LAST parameter and defaults to None.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import ValidationError

import config
from prompts import build_prompt, format_context
from recipe4_retrieve import clean_filter, list_sources, retrieve  # reuse Phase 1/4 retrieval
from schemas import LLMAnswer, QAResult, Source

# [NOOK-P2-12-FIX] Model name is overridable; 'latest' alias survives model deprecations.
LLM_MODEL: str = os.getenv("RAG_LLM_MODEL", "gemini-flash-latest")
LLM_TEMPERATURE: float = 0.2  # [NOOK-P2-8-FIX]
LLM_MAX_ATTEMPTS: int = 4
_TRANSIENT = ("429", "resource_exhausted", "resourceexhausted", "quota",
              "rate limit", "503", "unavailable", "timeout", "deadline")

ValidKey = Tuple[str, int]


def build_llm():
    """Construct the Gemini chat model. Called lazily, never at import."""
    # [NOOK-P2-7-FIX] No network/model construction at import time.
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        google_api_key=config.get_api_key(),  # fails loudly if key missing
    )


@lru_cache(maxsize=1)
def _chunk_id_lookup() -> Dict[str, str]:
    """text -> chunk_id from index/chunks_meta.json. Empty dict if unavailable."""
    # [NOOK-P2-5-FIX] Optional enrichment; citation validation never depends on it.
    try:
        rows = json.loads(Path(config.CHUNKS_META_FILE).read_text(encoding="utf-8"))
        return {r["text"]: r["chunk_id"] for r in rows if "text" in r and "chunk_id" in r}
    except Exception:
        return {}


def _enrich(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Copy results, adding chunk_id (or None) without mutating retrieve() output."""
    lookup = _chunk_id_lookup()
    return [{**r, "chunk_id": r.get("chunk_id") or lookup.get(r["text"])} for r in results]


def _norm_source(name: str) -> str:
    """'D:\\x\\Report.PDF' / 'x/Report.pdf' -> 'report.pdf'."""
    return Path(name.replace("\\", "/")).name.strip().lower()


def _valid_pairs(results: List[Dict[str, Any]]) -> Dict[ValidKey, Tuple[str, Optional[str]]]:
    """(normalised source, page) -> (original source name, first chunk_id)."""
    # [NOOK-P2-4-FIX] [NOOK-P2-5-FIX] Ground truth for citations = what retrieval returned.
    # [NOOK-P4-3-FIX] With a source filter this set contains ONLY that file, so a citation
    # to any other file is impossible to validate and gets dropped.
    valid: Dict[ValidKey, Tuple[str, Optional[str]]] = {}
    for r in results:
        key = (_norm_source(str(r["source"])), int(r["page"]))
        valid.setdefault(key, (str(r["source"]), r.get("chunk_id")))
    return valid


def _validate_sources(cited, valid) -> Tuple[List[Source], List[str]]:
    """Keep only citations that exist in retrieved results; dedupe; report dropped."""
    kept: List[Source] = []
    dropped: List[str] = []
    seen = set()
    for c in cited:
        key = (_norm_source(c.source), c.page)
        if key not in valid:
            dropped.append(f"{c.source} p.{c.page}")  # [NOOK-P2-4-FIX] fake citation dropped
            continue
        if key in seen:
            continue
        seen.add(key)
        name, cid = valid[key]
        kept.append(Source(source=name, page=c.page, chunk_id=cid))
    return kept, dropped


def _to_text(content: Any) -> str:
    """AIMessage.content may be str or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
        return "".join(parts)
    return str(content)


def _invoke_llm(llm: Any, messages: List[BaseMessage]) -> str:
    """llm.invoke with exponential backoff on 429/5xx-style errors."""
    delay = 2.0
    for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
        try:
            return _to_text(llm.invoke(messages).content)
        except Exception as e:
            transient = any(m in str(e).lower() for m in _TRANSIENT)
            if not transient or attempt == LLM_MAX_ATTEMPTS:
                raise
            wait = delay + random.uniform(0, 1)
            # [NOOK-P2-10-FIX] Rate limits do not kill the query.
            print(f"[qa] transient LLM error ({attempt}/{LLM_MAX_ATTEMPTS}), "
                  f"sleeping {wait:.1f}s: {str(e)[:100]}", file=sys.stderr)
            time.sleep(wait)
            delay = min(delay * 2, 30.0)
    raise RuntimeError("unreachable")


def answer_question(
    question: str,
    top_k: int = config.TOP_K,
    threshold: Optional[float] = None,
    llm: Any = None,
    filter_source: Optional[str] = None,
) -> QAResult:
    """Answer `question` from the indexed PDFs, with validated citations.

    filter_source: optional PDF file name (e.g. 'report.pdf') to restrict retrieval to.
    None / '' / 'none' / 'all' search every document (Phase-2 behaviour).
    Pass `llm` to inject a stub/mock (testing); otherwise Gemini is built lazily.
    """
    q = question.strip()
    if not q:
        raise ValueError("question is empty")
    thr = config.SCORE_THRESHOLD if threshold is None else float(threshold)

    # [NOOK-P4-8-FIX] 'None' / 'all' / '' from a caller or an LLM tool-call means no filter
    flt = clean_filter(filter_source)
    flt_label = str(filter_source).strip().strip("\"'") if flt is not None else None
    scope_warn: List[str] = [f"search restricted to source '{flt_label}'"] if flt else []

    results = _enrich(retrieve(q, top_k, filter_source=flt))  # Phase 1/4 retriever, reused as-is

    # [NOOK-P4-3-FIX] retrieve() returned [] under a filter: that file is not indexed.
    # Answer without calling the LLM and tell the caller which sources DO exist.
    if not results and flt is not None:
        known = list_sources()
        listing = ", ".join(known) if known else "none"
        return QAResult(
            question=q,
            answer=(f"No indexed document matches the source filter '{flt_label}'. "
                    f"Indexed sources: {listing}."),
            answerable=False, used_context=False, confidence="low", sources=[],
            top_score=None, threshold=thr, num_retrieved=0, warnings=scope_warn,
        )

    top = max((float(r["score"]) for r in results), default=None)

    # [NOOK-P2-1-FIX] Score gate: weak retrieval => refuse WITHOUT calling the LLM.
    if top is None or top < thr:
        shown = "n/a" if top is None else f"{top:.3f}"
        scope = f" within '{flt_label}'" if flt else ""
        return QAResult(
            question=q,
            answer=(f"I could not find relevant content{scope} in the indexed documents "
                    f"(best similarity {shown} is below the threshold {thr})."),
            answerable=False, used_context=False, confidence="low", sources=[],
            top_score=top, threshold=thr, num_retrieved=len(results), warnings=scope_warn,
        )

    parser = PydanticOutputParser(pydantic_object=LLMAnswer)
    messages: List[BaseMessage] = build_prompt().format_messages(
        context=format_context(results),
        question=q,
        format_instructions=parser.get_format_instructions(),  # [NOOK-P2-6-FIX]
    )
    model = llm if llm is not None else build_llm()
    warnings: List[str] = list(scope_warn)

    raw1 = _invoke_llm(model, messages)
    parsed: Optional[LLMAnswer] = None
    raw_last = raw1
    try:
        parsed = parser.parse(raw1)
    except (OutputParserException, ValidationError, ValueError) as err:
        # [NOOK-P2-3-FIX] Exactly one retry, telling the model what was wrong.
        warnings.append("first LLM output was invalid; retried once")
        retry_msgs = messages + [
            AIMessage(content=raw1),
            HumanMessage(content=(
                f"Your previous reply was invalid: {str(err)[:400]}\n"
                "Reply again with ONLY one valid JSON object matching the OUTPUT FORMAT. "
                "No prose, no code fences.")),
        ]
        raw_last = _invoke_llm(model, retry_msgs)
        try:
            parsed = parser.parse(raw_last)
        except (OutputParserException, ValidationError, ValueError):
            parsed = None

    if parsed is None:
        # [NOOK-P2-3-FIX] Raw fallback: return the text, flagged, with NO citations.
        warnings.append("LLM output failed validation twice; raw text returned, citations unverified")
        return QAResult(
            question=q, answer=(raw_last or raw1).strip()[:2000] or "(empty model output)",
            answerable=True, used_context=True, confidence="low", sources=[],
            top_score=top, threshold=thr, num_retrieved=len(results),
            fallback_raw=True, warnings=warnings,
        )

    sources, dropped = _validate_sources(parsed.sources, _valid_pairs(results))
    if dropped:
        warnings.append("dropped citations not present in retrieved results: " + "; ".join(dropped))
    confidence = parsed.confidence
    if parsed.answerable and not sources:
        # Answer claims content but nothing verifiable backs it: downgrade, never invent citations.
        confidence = "low"
        warnings.append("answer has no verifiable citations")

    return QAResult(
        question=q, answer=parsed.answer, answerable=parsed.answerable, used_context=True,
        confidence=confidence, sources=sources, top_score=top, threshold=thr,
        num_retrieved=len(results), fallback_raw=False, warnings=warnings,
    )


if __name__ == "__main__":
    import argparse

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Grounded QA over indexed PDFs")
    ap.add_argument("question", nargs="+")
    ap.add_argument("--top-k", type=int, default=config.TOP_K)
    ap.add_argument("--source", default=None, help="restrict to one indexed PDF, e.g. report.pdf")
    cli = ap.parse_args()
    print(answer_question(" ".join(cli.question), top_k=cli.top_k,
                          filter_source=cli.source).model_dump_json(indent=2))
    sys.exit(0)