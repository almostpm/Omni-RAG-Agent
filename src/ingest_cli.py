"""ingest_cli.py - add PDFs to the corpus and keep the FAISS index in sync.

Usage:
  python src\\ingest_cli.py add "D:\\downloads\\policy.pdf" [--overwrite] [--no-pipeline] [--max-mb 50]
  python src\\ingest_cli.py list
  python src\\ingest_cli.py sync [--force]

add   : validate -> copy into data\\pdfs -> recipe1 (chunk) -> recipe2 (embed) -> recipe3 (index) -> verify
list  : indexed sources with chunk counts, plus PDFs not yet indexed
sync  : re-run recipes 1-3 (resumes after a failed embed; no-op when nothing changed)
Exit codes: 0 ok, 1 error.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config

DEFAULT_MAX_MB = 50.0
_ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {f"lpt{i}" for i in range(1, 10)}


def _setup_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sanitize_filename(name: str) -> str:
    """Make a Windows-safe '<name>.pdf' (illegal chars, reserved names, trailing dots/spaces)."""
    # [NOOK-P4-7-FIX] file names become the 'source' key: they must be valid on Windows
    stem = _ILLEGAL_CHARS.sub("_", Path(name).stem).strip(" .")[:120]
    if not stem:
        stem = "document"
    if stem.lower() in _RESERVED:
        stem += "_"
    return stem + ".pdf"


def validate_pdf(path: Path, max_mb: float) -> Optional[str]:
    """Return an error string, or None if the file is an acceptable, readable PDF."""
    # [NOOK-P4-7-FIX] reject non-PDFs, empty/huge files, fake extensions, encrypted PDFs up front
    if not path.exists() or not path.is_file():
        return f"file not found: {path}"
    if path.suffix.lower() != ".pdf":
        return "only .pdf files are accepted"
    size = path.stat().st_size
    if size == 0:
        return "file is empty"
    if size > max_mb * 1024 * 1024:
        return f"file is {size / 1048576:.1f} MB, over the {max_mb:g} MB limit (see --max-mb)"
    with open(path, "rb") as f:
        head = f.read(1024)
    if b"%PDF-" not in head:
        return "not a real PDF (missing %PDF- header)"
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        if reader.is_encrypted and reader.decrypt("") == 0:
            return "PDF is password-protected"
        if len(reader.pages) == 0:
            return "PDF has no pages"
    except Exception as e:
        return f"PDF cannot be parsed ({type(e).__name__}: {str(e)[:100]})"
    return None


def run_script(script: str, *extra: str) -> int:
    cmd = [sys.executable, str(config.SRC_DIR / script), *extra]
    print(f"\n>>> {script} {' '.join(extra)}".rstrip())
    return subprocess.run(cmd, cwd=str(config.PROJECT_ROOT)).returncode


def _count_by_source_jsonl(path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not path.exists():
        return counts
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                src = json.loads(line)["source"]
                counts[src] = counts.get(src, 0) + 1
    return counts


def _count_by_source_meta() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not config.CHUNKS_META_FILE.exists():
        return counts
    try:
        rows = json.loads(config.CHUNKS_META_FILE.read_text(encoding="utf-8"))
    except Exception:
        return counts
    for r in rows:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    return counts


def verify_sync(expect_source: Optional[str]) -> Optional[str]:
    """Check the index is consistent with chunks_meta.json (and contains expect_source)."""
    # [NOOK-P4-6-FIX] never report success unless the index really covers the new file
    if not config.INDEX_META_FILE.exists() or not config.CHUNKS_META_FILE.exists():
        return "index files missing after pipeline"
    idx_meta = json.loads(config.INDEX_META_FILE.read_text(encoding="utf-8"))
    meta_counts = _count_by_source_meta()
    total = sum(meta_counts.values())
    if idx_meta.get("num_vectors") != total:
        return f"index has {idx_meta.get('num_vectors')} vectors but chunks_meta has {total}"
    jsonl_counts = _count_by_source_jsonl(config.CHUNKS_FILE)
    if jsonl_counts != meta_counts:
        return "chunks.jsonl and the embedded chunks differ; run: python src\\ingest_cli.py sync"
    if expect_source and expect_source.lower() not in {s.lower() for s in meta_counts}:
        return f"'{expect_source}' is not in the index"
    return None


def pipeline(force: bool) -> int:
    """recipe1 -> recipe2 -> recipe3. Returns 0 on success."""
    if run_script("recipe1_ingest.py") != 0:
        return 1
    chunks = sum(_count_by_source_jsonl(config.CHUNKS_FILE).values())
    # [NOOK-P4-6-FIX] recipe2 hashes the WHOLE corpus: any change re-embeds every chunk. Say so.
    print(f"[ingest] corpus = {chunks} chunks. recipe2 has no per-chunk cache: if the corpus "
          "changed, ALL chunks are re-embedded (unchanged corpus = zero API calls).")
    if run_script("recipe2_embed.py", *(["--force"] if force else [])) != 0:
        return 1
    if run_script("recipe3_build_index.py") != 0:
        return 1
    return 0


def _rollback(target: Path, backup: Optional[Path]) -> None:
    """Undo a copy: delete the new file, restore the backup, rebuild chunks.jsonl."""
    # [NOOK-P4-5-FIX] a failed add must leave the corpus exactly as it was
    try:
        if target.exists():
            target.unlink()
        if backup is not None and backup.exists():
            backup.replace(target)
    except OSError as e:
        print(f"WARN: rollback could not fully restore files: {e}", file=sys.stderr)
    run_script("recipe1_ingest.py")


def cmd_add(args: argparse.Namespace) -> int:
    src = Path(args.path).expanduser()
    err = validate_pdf(src, args.max_mb)
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1

    config.ensure_dirs()
    if not args.no_pipeline:
        config.get_api_key()  # fail loudly BEFORE copying anything

    clean = sanitize_filename(src.name)
    if clean != src.name:
        print(f"[ingest] file name sanitised for Windows: '{src.name}' -> '{clean}'")
    target = config.PDF_DIR / clean
    new_hash = sha256_file(src)

    # [NOOK-P4-5-FIX] duplicate content under a different name would double-count every answer
    for existing in sorted(config.PDF_DIR.glob("*.pdf")):
        if existing.name.lower() == target.name.lower():
            continue
        if sha256_file(existing) == new_hash:
            print(f"ERROR: identical content is already in the corpus as '{existing.name}'. "
                  "Not adding a duplicate.", file=sys.stderr)
            return 1

    backup: Optional[Path] = None
    if target.exists():
        if sha256_file(target) == new_hash:
            print(f"[ingest] '{target.name}' is already in the corpus (same content); skipping copy.")
        elif not args.overwrite:
            # [NOOK-P4-5-FIX] never silently replace a different document with the same name
            print(f"ERROR: '{target.name}' already exists with DIFFERENT content. "
                  "Use --overwrite to replace it.", file=sys.stderr)
            return 1
        else:
            backup = target.with_name(target.name + ".bak")  # .bak is ignored by *.pdf globs
            target.replace(backup)
            shutil.copy2(str(src), str(target))
    else:
        shutil.copy2(str(src), str(target))
    print(f"[ingest] corpus file: {target}")

    if args.no_pipeline:
        print("[ingest] copied only. Index NOT updated; run: python src\\ingest_cli.py sync")
        return 0

    rc = run_script("recipe1_ingest.py")
    counts = _count_by_source_jsonl(config.CHUNKS_FILE) if rc == 0 else {}
    if rc != 0 or target.name not in counts:
        print(f"ERROR: '{target.name}' produced no text chunks (scanned/empty/unreadable). "
              "Rolled back; the corpus is unchanged.", file=sys.stderr)
        _rollback(target, backup)
        return 1
    print(f"[ingest] '{target.name}': {counts[target.name]} chunks "
          f"(corpus: {sum(counts.values())} chunks in {len(counts)} sources)")

    if pipeline(force=False) != 0:
        print("ERROR: pipeline failed. The PDF and its chunks are in place; fix the cause "
              "(API key, quota, network) and run: python src\\ingest_cli.py sync", file=sys.stderr)
        return 1
    err = verify_sync(target.name)
    if err:
        print(f"ERROR: verification failed: {err}", file=sys.stderr)
        return 1
    if backup is not None and backup.exists():
        backup.unlink()
    print(f"\nOK: '{target.name}' is indexed. Try: "
          f'python src\\recipe4_retrieve.py "your question" --source "{target.name}"')
    return 0


def cmd_list(_args: argparse.Namespace) -> int:
    indexed = _count_by_source_meta()
    on_disk = sorted((p.name for p in config.PDF_DIR.glob("*.pdf")), key=str.lower)
    print("Indexed sources (index\\chunks_meta.json):")
    if indexed:
        width = max(len(n) for n in indexed)
        for name, n in sorted(indexed.items(), key=lambda kv: kv[0].lower()):
            print(f"  {name.ljust(width)}  {n:>5} chunks")
    else:
        print("  (none)")
    indexed_lower = {n.lower() for n in indexed}
    missing = [n for n in on_disk if n.lower() not in indexed_lower]
    if missing:
        print("\nIn data\\pdfs but NOT indexed (run: python src\\ingest_cli.py sync):")
        for n in missing:
            print(f"  {n}")
    jsonl_counts = _count_by_source_jsonl(config.CHUNKS_FILE)
    if jsonl_counts and jsonl_counts != indexed:
        print("\nWARN: chunks.jsonl differs from the embedded index (stale). "
              "Run: python src\\ingest_cli.py sync")
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    config.get_api_key()
    if pipeline(force=args.force) != 0:
        print("ERROR: sync failed; re-run to resume (embedding batches are checkpointed).",
              file=sys.stderr)
        return 1
    err = verify_sync(None)
    if err:
        print(f"ERROR: verification failed: {err}", file=sys.stderr)
        return 1
    print("\nOK: index is in sync.")
    return cmd_list(args)


def main() -> int:
    _setup_stdio()
    ap = argparse.ArgumentParser(description="Add PDFs to the RAG corpus and keep the index in sync")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="validate, copy and index one PDF")
    p_add.add_argument("path", help="path to the .pdf file (quote it if it has spaces)")
    p_add.add_argument("--overwrite", action="store_true",
                       help="replace an existing corpus file with the same name")
    p_add.add_argument("--no-pipeline", action="store_true",
                       help="only copy the file; run 'sync' later")
    p_add.add_argument("--max-mb", type=float, default=DEFAULT_MAX_MB)
    p_add.set_defaults(func=cmd_add)

    p_list = sub.add_parser("list", help="show indexed sources and unindexed PDFs")
    p_list.set_defaults(func=cmd_list)

    p_sync = sub.add_parser("sync", help="re-run recipes 1-3 and verify")
    p_sync.add_argument("--force", action="store_true", help="re-embed even if the corpus is unchanged")
    p_sync.set_defaults(func=cmd_sync)

    args = ap.parse_args()
    try:
        return args.func(args)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("ERROR: interrupted", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())