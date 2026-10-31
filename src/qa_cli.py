"""qa_cli.py - command-line front end for grounded QA.

Usage:
  python src\\qa_cli.py "What does this PDF say about X?"
  python src\\qa_cli.py "..." --top-k 3 --threshold 0.3 --json
  python src\\qa_cli.py            (interactive; type 'exit' to quit)
Exit codes: 0 ok (including 'not found' answers), 1 error.
"""
from __future__ import annotations

import argparse
import sys

import config
from qa_chain import answer_question
from schemas import QAResult


def _setup_stdio() -> None:
    # [NOOK-P2-9-FIX] Windows consoles (cp1252) crash on non-ASCII PDF text; force UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def render(res: QAResult) -> str:
    lines = [f"Q: {res.question}", "", f"A: {res.answer}", ""]
    if res.sources:
        lines.append("Sources:")
        for s in res.sources:
            cid = f"  [{s.chunk_id}]" if s.chunk_id else ""
            lines.append(f"  - {s.source}, p.{s.page}{cid}")
    else:
        lines.append("Sources: none")
    top = "n/a" if res.top_score is None else f"{res.top_score:.3f}"
    lines.append("")
    lines.append(f"[used_context={res.used_context} answerable={res.answerable} "
                 f"confidence={res.confidence} top_score={top} "
                 f"threshold={res.threshold} retrieved={res.num_retrieved}]")
    if res.fallback_raw:
        lines.append("[WARNING: raw fallback - model output was not valid JSON; citations unverified]")
    return "\n".join(lines)


def _ask(question: str, args: argparse.Namespace) -> int:
    try:
        res = answer_question(question, top_k=args.top_k, threshold=args.threshold)
    except SystemExit:
        raise  # retrieve()/config already printed a clear message
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(res.model_dump_json(indent=2) if args.json else render(res))
    for w in res.warnings:
        print(f"WARN: {w}", file=sys.stderr)
    return 0


def main() -> int:
    _setup_stdio()
    ap = argparse.ArgumentParser(description="Grounded QA over indexed PDFs")
    ap.add_argument("question", nargs="*", help="question text (omit for interactive mode)")
    ap.add_argument("--top-k", type=int, default=config.TOP_K)
    ap.add_argument("--threshold", type=float, default=config.SCORE_THRESHOLD)
    ap.add_argument("--json", action="store_true", help="print the full QAResult as JSON")
    args = ap.parse_args()

    if args.question:
        return _ask(" ".join(args.question), args)

    print("Interactive mode. Type 'exit' to quit.")
    while True:
        try:
            q = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if q.lower() in ("exit", "quit"):
            return 0
        if q:
            _ask(q, args)


if __name__ == "__main__":
    sys.exit(main())