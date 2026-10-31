"""recipe5_milestone.py - end-to-end Phase 1 milestone runner (idempotent)."""
from __future__ import annotations

import argparse
import subprocess
import sys  # <-- Make sure sys is imported cleanly at the very top globally

import config

def run(script: str, *extra: str) -> int:
    # [NOOK-N3-FIX] list-form subprocess + str(Path): no shell quoting, space-safe
    cmd = [sys.executable, str(config.SRC_DIR / script), *extra]
    print(f"\n>>> {script} {' '.join(extra)}".rstrip())
    return subprocess.run(cmd, cwd=str(config.PROJECT_ROOT)).returncode


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="What does this PDF say about its main topic?")
    ap.add_argument("--top-k", type=int, default=config.TOP_K)
    ap.add_argument("--threshold", type=float, default=config.SCORE_THRESHOLD)
    ap.add_argument("--force", action="store_true", help="re-embed even if corpus unchanged")
    args = ap.parse_args()

    config.ensure_dirs()
    config.get_api_key()  # [NOOK-N1-FIX] fail fast before any work

    pdfs = list(config.PDF_DIR.glob("*.pdf"))
    if not pdfs:  # [NOOK-N10-FIX] graceful: nothing to do, nothing spent
        print(f"ERROR: {config.PDF_DIR} has no PDFs. Add one and re-run.", file=sys.stderr)
        return 1

    # Step 1: ingest only if chunks missing or older than newest PDF
    newest_pdf = max(p.stat().st_mtime for p in pdfs)
    if (not config.CHUNKS_FILE.exists()) or config.CHUNKS_FILE.stat().st_mtime < newest_pdf:
        if run("recipe1_ingest.py") != 0:
            return 1
    else:
        print("[skip] recipe1: chunks.jsonl is up to date")

    # Step 2: embed (recipe2 has its own sha256 guard => no-op when unchanged)
    if run("recipe2_embed.py", *(["--force"] if args.force else [])) != 0:
        return 1

    # Step 3: index only if missing or older than embeddings
    if (not config.FAISS_INDEX_FILE.exists() or not config.INDEX_META_FILE.exists()
            or config.FAISS_INDEX_FILE.stat().st_mtime < config.EMBEDDINGS_FILE.stat().st_mtime):
        if run("recipe3_build_index.py") != 0:
            return 1
    else:
        print("[skip] recipe3: faiss.index is up to date")

    # Step 4: retrieve (direct import)
    # Step 4: retrieve (dynamic import to support both direct execution and module invocation)
    import importlib.util, sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("recipe4_retrieve", str(config.SRC_DIR / "recipe4_retrieve.py"))
    recipe4_retrieve = importlib.util.module_from_spec(spec)
    sys.modules["recipe4_retrieve"] = recipe4_retrieve
    spec.loader.exec_module(recipe4_retrieve)
    retrieve = recipe4_retrieve.retrieve
    results = retrieve(args.query, args.top_k)

    print(f"\nQUERY: {args.query}\n" + "=" * 70)
    for r in results:  # [NOOK-N7-FIX] scores printed so top-k can be judged manually
        print(f"#{r['rank']}  score={r['score']}  {r['source']}  p.{r['page']}")
        print(r["text"][:500].replace("\n", " "))
        print("-" * 70)

    best = max((r["score"] for r in results), default=0.0)
    if best > args.threshold:
        print(f"PHASE-1 MILESTONE: PASS  (best score {best} > {args.threshold})")
        print('Next: run your own question ->  python src\\recipe4_retrieve.py "your question" --top-k 3')
        print("Phase 2 (generation layer): paste E1's Phase-2 meta prompt to E2.")
        return 0
    print(f"PHASE-1 MILESTONE: FAIL  (best score {best} <= {args.threshold}). "
          "Check PDF text extraction and the query wording.")
    return 2


if __name__ == "__main__":
    sys.exit(main())