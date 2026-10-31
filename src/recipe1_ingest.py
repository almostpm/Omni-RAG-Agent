"""recipe1_ingest.py - PDF ingestion + chunking (Phase 4 update).

CLI behaviour is unchanged:  python src\\recipe1_ingest.py
New in Phase 4:
  * ingest_all() returns a report dict (per-source chunk counts + skip reasons), used by ingest_cli.py
  * chunks.jsonl is written atomically (tmp file + os.replace)
  * unreadable PDFs are reported with a reason instead of only a warning
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.document_loaders import PyPDFLoader

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # older langchain
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

import config


def build_splitter() -> RecursiveCharacterTextSplitter:
    """800-char chunks, 150 overlap (Phase-1 settings, unchanged)."""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_pdf(pdf: Path, splitter: RecursiveCharacterTextSplitter
              ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Chunk one PDF. Returns (rows, skip_reason); skip_reason is None on success."""
    try:
        pages = PyPDFLoader(str(pdf)).load()
    except Exception as e:  # corrupt / encrypted PDF
        # [NOOK-P4-7-FIX] a bad PDF is skipped WITH a reason the CLI can show; it never aborts the run
        return [], f"unreadable: {type(e).__name__}: {str(e)[:120]}"

    total_chars = sum(len((p.page_content or "").strip()) for p in pages)
    print(f"[ingest] {pdf.name}: {len(pages)} pages, {total_chars} chars extracted")
    if total_chars < config.MIN_PDF_CHARS:
        return [], (f"only {total_chars} extractable chars (<{config.MIN_PDF_CHARS}); "
                    "scanned/empty PDF, OCR is out of scope")

    pages = [p for p in pages if (p.page_content or "").strip()]
    per_page: Dict[int, int] = defaultdict(int)
    rows: List[Dict[str, Any]] = []
    for ch in splitter.split_documents(pages):
        text = ch.page_content.strip()
        if not text:
            continue
        page = int(ch.metadata.get("page", 0)) + 1
        n = per_page[page]
        per_page[page] += 1
        rows.append({
            "chunk_id": f"{pdf.stem}_p{page}_c{n}",
            "source": pdf.name,
            "page": page,
            "text": text,
        })
    if not rows:
        return [], "no chunks produced"
    return rows, None


def _write_jsonl_atomic(rows: List[Dict[str, Any]]) -> None:
    """Write chunks.jsonl via a temp file so a crash never leaves a half-written corpus."""
    # [NOOK-P4-6-FIX] atomic replace: downstream recipes never see a truncated chunks.jsonl
    tmp = config.CHUNKS_FILE.with_name(config.CHUNKS_FILE.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, config.CHUNKS_FILE)


def ingest_all(pdf_dir: Path = config.PDF_DIR) -> Dict[str, Any]:
    """Chunk every PDF in pdf_dir and rewrite chunks.jsonl.

    Returns {"pdfs": int, "chunks": int, "per_source": {name: n}, "skipped": {name: reason}}.
    chunks.jsonl is only rewritten when at least one chunk was produced.
    """
    config.ensure_dirs()
    # case-insensitive, deterministic order (Windows file names are case-insensitive)
    pdfs = sorted(pdf_dir.glob("*.pdf"), key=lambda p: p.name.lower())
    report: Dict[str, Any] = {"pdfs": len(pdfs), "chunks": 0, "per_source": {}, "skipped": {}}
    if not pdfs:
        return report

    splitter = build_splitter()
    rows: List[Dict[str, Any]] = []
    for pdf in pdfs:
        got, reason = chunk_pdf(pdf, splitter)
        if reason is not None:
            report["skipped"][pdf.name] = reason
            continue
        report["per_source"][pdf.name] = len(got)
        rows.extend(got)

    if rows:
        _write_jsonl_atomic(rows)
    report["chunks"] = len(rows)
    return report


def main() -> int:
    report = ingest_all()
    if report["pdfs"] == 0:
        print(f"ERROR: no PDFs found in {config.PDF_DIR}. Drop at least one .pdf there.",
              file=sys.stderr)
        return 1
    for name, reason in report["skipped"].items():
        print(f"WARN: {name} SKIPPED: {reason}", file=sys.stderr)
    if report["chunks"] == 0:
        print("ERROR: zero chunks produced (all PDFs empty/unreadable).", file=sys.stderr)
        return 1
    for name, n in report["per_source"].items():
        print(f"[ingest] {name}: {n} chunks")
    print(f"[ingest] OK: {report['chunks']} chunks -> {config.CHUNKS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())