"""recipe4_retrieve.py - top-k cosine retrieval over the FAISS index (Phase 4: source filter).

Usage:
  python src\\recipe4_retrieve.py "What does this PDF say about X?" --top-k 3
  python src\\recipe4_retrieve.py "question" --source report.pdf
  python src\\recipe4_retrieve.py --selftest            (offline, no API)

retrieve(query, top_k, filter_source=None): filter_source=None searches everything
(identical to Phase 1). With a filter, FAISS is oversampled (top_k * 5) and filtered in Python.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import faiss
import numpy as np

import config

OVERSAMPLE_FACTOR = 5  # filtered search first asks FAISS for top_k * 5 candidates
# [NOOK-P4-8-FIX] values an LLM or a user may pass meaning "no filter"
_NO_FILTER = {"", "none", "null", "nil", "all", "any", "*", "*.pdf", "n/a"}


def normalize_source(name: str) -> str:
    """'D:\\\\docs\\\\Report.PDF' or 'docs/Report.pdf' -> 'report.pdf'."""
    # [NOOK-P4-1-FIX] compare by lower-cased base name: ignores path, case, stray whitespace
    return Path(str(name).replace("\\", "/")).name.strip().lower()


def clean_filter(filter_source: Optional[str]) -> Optional[str]:
    """Normalise a user/LLM supplied filter. Returns None when it means 'no filter'."""
    if filter_source is None:
        return None
    s = str(filter_source).strip().strip("\"'").strip()
    # [NOOK-P4-8-FIX] 'None' / 'all' / '' must never become a literal file name
    if s.lower() in _NO_FILTER:
        return None
    return normalize_source(s)


def list_sources() -> Dict[str, int]:
    """Indexed source file names -> chunk count (from chunks_meta.json). {} if no index yet."""
    p = config.CHUNKS_META_FILE
    if not p.exists():
        return {}
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    counts: Dict[str, int] = {}
    for c in rows:
        counts[c["source"]] = counts.get(c["source"], 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[0].lower()))


def load_index():
    """Load index + chunk metadata, hard-failing on any model/dim mismatch."""
    for p in (config.FAISS_INDEX_FILE, config.INDEX_META_FILE, config.CHUNKS_META_FILE):
        if not p.exists():
            sys.exit(f"ERROR: {p} missing. Run recipes 1-3 first.")
    meta = json.loads(config.INDEX_META_FILE.read_text(encoding="utf-8"))

    # [NOOK-N6-FIX] [NOOK-N2-FIX] query-embedding mismatch guard: same model + dim as build
    if meta.get("embed_model") != config.EMBED_MODEL:
        sys.exit(f"ERROR: index built with '{meta.get('embed_model')}' but current model is "
                 f"'{config.EMBED_MODEL}'. Rebuild: python src\\recipe2_embed.py --force")
    if meta.get("dim") != config.EMBED_DIM:
        sys.exit(f"ERROR: index dim {meta.get('dim')} != {config.EMBED_DIM}.")
    if meta.get("normalization") != "l2":  # [NOOK-N8-FIX]
        sys.exit("ERROR: index_meta says vectors are not L2-normalized; rebuild with recipe3.")

    index = faiss.read_index(str(config.FAISS_INDEX_FILE))  # [NOOK-N3-FIX]
    chunks = json.loads(config.CHUNKS_META_FILE.read_text(encoding="utf-8"))
    if index.d != config.EMBED_DIM or index.ntotal != len(chunks):
        sys.exit("ERROR: faiss.index and chunks_meta.json are out of sync. Rebuild recipes 2-3.")
    return index, chunks


def _allowed_ids(chunks: List[Dict[str, Any]], want: str) -> Set[int]:
    """Row ids whose source matches `want` ('report.pdf'; the bare stem 'report' also matches)."""
    # [NOOK-P4-1-FIX] accept 'report' and 'report.pdf', any case, any path prefix
    wanted = {want} if want.endswith(".pdf") else {want, want + ".pdf"}
    return {i for i, c in enumerate(chunks) if normalize_source(c["source"]) in wanted}


def _filtered_search(index, q: np.ndarray, allowed: Set[int], top_k: int, k: int
                     ) -> List[Tuple[float, int]]:
    """Search k candidates, keep only allowed ids, return up to top_k (score, id) in rank order."""
    scores, ids = index.search(q, k)
    hits: List[Tuple[float, int]] = []
    for s, i in zip(scores[0], ids[0]):
        if i >= 0 and int(i) in allowed:
            hits.append((float(s), int(i)))
            if len(hits) == top_k:
                break
    return hits


def retrieve(query: str, top_k: int = config.TOP_K,
             filter_source: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return top_k chunks ranked by cosine similarity, optionally restricted to one source file.

    filter_source=None (or 'none'/'all'/'') searches the whole corpus, exactly as in Phase 1.
    Returns [] when the filter matches no indexed source. No embedding call is spent in that case.
    """
    if not query.strip():
        sys.exit("ERROR: empty query.")
    want = clean_filter(filter_source)
    index, chunks = load_index()

    allowed: Optional[Set[int]] = None
    if want is not None:
        allowed = _allowed_ids(chunks, want)
        if not allowed:
            # [NOOK-P4-3-FIX] unknown source: return [] immediately; caller explains it,
            # and no Gemini query-embedding call is wasted.
            return []

    # [NOOK-N9-FIX] queries ALWAYS use RETRIEVAL_QUERY (asymmetric with documents)
    assert config.QUERY_TASK == "RETRIEVAL_QUERY"
    emb = config.build_embedder(config.QUERY_TASK)
    q = np.asarray([emb.embed_query(query)], dtype=np.float32)
    if q.shape != (1, config.EMBED_DIM):
        sys.exit(f"ERROR: query vector shape {q.shape}, expected (1, {config.EMBED_DIM}).")
    faiss.normalize_L2(q)  # [NOOK-N8-FIX] normalize the query too

    top_k = max(1, int(top_k))  # [NOOK-N7-FIX]
    if allowed is None:
        # [NOOK-P4-4-FIX] unfiltered path is byte-for-byte the Phase-1 behaviour
        k = max(1, min(top_k, index.ntotal))
        scores, ids = index.search(q, k)
        hits = [(float(s), int(i)) for s, i in zip(scores[0], ids[0]) if i >= 0]
    else:
        # filter via oversample: ask FAISS for top_k * 5, keep only matching rows
        k = min(index.ntotal, top_k * OVERSAMPLE_FACTOR)
        hits = _filtered_search(index, q, allowed, top_k, k)
        if len(hits) < top_k and k < index.ntotal:
            # [NOOK-P4-2-FIX] oversample too shallow (the wanted file ranks low globally):
            # widen to the whole index. IndexFlatIP is exact, so this is still correct, just wider.
            hits = _filtered_search(index, q, allowed, top_k, index.ntotal)

    results: List[Dict[str, Any]] = []
    for rank, (s, i) in enumerate(hits, start=1):
        c = chunks[i]
        results.append({"rank": rank, "score": round(s, 4),
                        "source": c["source"], "page": c["page"], "text": c["text"]})
    return results


def _selftest() -> int:
    """Offline checks for filter normalisation. No index, no API."""
    checks = [
        (normalize_source("D:\\docs\\Report.PDF") == "report.pdf"),
        (normalize_source("docs/Report.pdf") == "report.pdf"),
        (clean_filter(None) is None),
        (clean_filter("") is None),
        (clean_filter(" None ") is None),
        (clean_filter("ALL") is None),
        (clean_filter('"My File.pdf"') == "my file.pdf"),
        (clean_filter("C:\\x\\Policy.PDF") == "policy.pdf"),
    ]
    ok = all(checks)
    print("recipe4 selftest:", "OK" if ok else f"FAILED {checks}")
    return 0 if ok else 1


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows console safety
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--top-k", type=int, default=config.TOP_K)
    ap.add_argument("--source", default=None,
                    help="restrict search to one indexed PDF file name, e.g. report.pdf")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if not args.query:
        ap.error("query is required (or use --selftest)")
    results = retrieve(args.query, args.top_k, filter_source=args.source)
    if not results and clean_filter(args.source) is not None:
        known = ", ".join(list_sources()) or "none"
        print(f"WARN: no indexed source matches '{args.source}'. Indexed sources: {known}",
              file=sys.stderr)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())