"""agent_cli.py - command-line chat with the Phase-3 agent.

Usage:
  python src\\agent_cli.py --query "What does the document say about X?"
  python src\\agent_cli.py --query "File an expense of 42.50 at Rourkela Station on 2026-10-30" --verbose
  python src\\agent_cli.py                 (interactive; type 'exit' to quit)
Exit codes: 0 ok, 1 error.
"""
import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _setup_stdio() -> None:
    # Windows consoles (cp1252) crash on non-ASCII text from PDFs: force UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _log_step(line: str) -> None:
    print(line, file=sys.stderr)


def _ask(query: str, verbose: bool) -> int:
    # [NOOK-P3-3-FIX] Global catch: missing key/index (SystemExit), import/version problems,
    # API failures, anything. The user gets "Agent Error: ..." instead of a traceback.
    try:
        from agent import run_agent  # imported here so --help works and import errors are caught
        answer = run_agent(query, verbose=verbose, on_step=_log_step if verbose else None)
    except SystemExit as e:
        if e.code in (0, None):
            raise
        print(f"Agent Error: {e.code}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Agent Error: interrupted by user", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Agent Error: {e}", file=sys.stderr)
        return 1
    print(answer)
    return 0


def main() -> int:
    _setup_stdio()
    ap = argparse.ArgumentParser(description="Chat with the RAG Agent (documents + expense tool)")
    ap.add_argument("--query", type=str, default=None, help="your request (omit for interactive mode)")
    ap.add_argument("--verbose", action="store_true",
                    help="print each agent step (tool calls and tool results) to stderr")
    args = ap.parse_args()

    if args.query is not None:
        if not args.query.strip():
            print("Agent Error: --query is empty", file=sys.stderr)
            return 1
        return _ask(args.query, args.verbose)

    print("Interactive mode. Type 'exit' to quit.")
    while True:
        try:
            q = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if q.lower() in ("exit", "quit"):
            return 0
        if q:
            _ask(q, args.verbose)  # keep the session alive after a failed turn


if __name__ == "__main__":
    sys.exit(main())