"""recipe2_embed.py - Gemini embeddings for all chunks (task_type=RETRIEVAL_DOCUMENT).

Flags: --force (ignore hash guard), --sleep SECONDS (pause between batches).
Resumable: completed batches are checkpointed to index/.embed_ckpt/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from typing import Dict, List

import numpy as np

import config

MAX_RETRIES = 6
TRANSIENT_MARKERS = ("429", "resource_exhausted", "resourceexhausted", "quota",
                     "rate limit", "503", "unavailable", "timeout", "deadline")
CKPT_DIR = config.INDEX_DIR / ".embed_ckpt"


def corpus_hash(texts: List[str]) -> str:
    """sha256 over model|dim + sorted chunk texts."""
    h = hashlib.sha256()
    h.update(f"{config.EMBED_MODEL}|{config.EMBED_DIM}\n".encode("utf-8"))
    for t in sorted(texts):
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def embed_with_backoff(emb, texts: List[str]) -> List[List[float]]:
    """Call embed_documents with exponential backoff on rate-limit/transient errors."""
    delay = 2.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return emb.embed_documents(texts)
        except Exception as e:
            msg = str(e).lower()
            transient = any(m in msg for m in TRANSIENT_MARKERS)
            if not transient or attempt == MAX_RETRIES:
                raise
            wait = delay + random.uniform(0, 1)
            # [NOOK-N1-FIX] exponential backoff on 429 instead of crashing mid-run
            print(f"[embed] transient error (attempt {attempt}/{MAX_RETRIES}), "
                  f"sleeping {wait:.1f}s: {str(e)[:120]}", file=sys.stderr)
            time.sleep(wait)
            delay = min(delay * 2, 60.0)
    raise RuntimeError("unreachable")


def atomic_save_npy(path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()

    config.ensure_dirs()
    if not config.CHUNKS_FILE.exists():
        print(f"ERROR: {config.CHUNKS_FILE} missing. Run recipe1 first.", file=sys.stderr)
        return 1
    with open(config.CHUNKS_FILE, encoding="utf-8") as f:
        chunks: List[Dict] = [json.loads(l) for l in f if l.strip()]
    if not chunks:
        print("ERROR: chunks.jsonl is empty.", file=sys.stderr)
        return 1
    texts = [c["text"] for c in chunks]

    # [NOOK-N10-FIX] re-embedding guard: unchanged corpus => zero API calls
    h = corpus_hash(texts)
    if (not args.force and config.EMBED_HASH_FILE.exists()
            and config.EMBED_HASH_FILE.read_text().strip() == h
            and config.EMBEDDINGS_FILE.exists()
            and config.CHUNKS_META_FILE.exists()):
        existing = np.load(config.EMBEDDINGS_FILE, mmap_mode="r")
        if existing.shape == (len(texts), config.EMBED_DIM):
            print(f"[embed] corpus unchanged (hash {h[:10]}...), skipping. Use --force to override.")
            return 0

    # [NOOK-N9-FIX] documents ALWAYS use RETRIEVAL_DOCUMENT
    assert config.DOC_TASK == "RETRIEVAL_DOCUMENT"
    emb = config.build_embedder(config.DOC_TASK)  # fails loudly if key missing
    print(f"[embed] key={config.mask_key(config.get_api_key())} model={config.EMBED_MODEL} "
          f"chunks={len(texts)}")

    # [NOOK-N1-FIX] write-ahead checkpoint: a crash/429 never loses finished batches
    manifest = CKPT_DIR / "manifest.json"
    if CKPT_DIR.exists():
        stale = True
        if manifest.exists():
            stale = json.loads(manifest.read_text()).get("hash") != h
        if stale or args.force:
            shutil.rmtree(CKPT_DIR)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"hash": h, "n": len(texts)}))

    parts: List[np.ndarray] = []
    n_batches = (len(texts) + config.BATCH_SIZE - 1) // config.BATCH_SIZE
    try:
        for b in range(n_batches):
            lo, hi = b * config.BATCH_SIZE, min((b + 1) * config.BATCH_SIZE, len(texts))
            ck = CKPT_DIR / f"batch_{b:05d}.npy"
            if ck.exists():
                arr = np.load(ck)
                print(f"[embed] batch {b+1}/{n_batches} loaded from checkpoint")
            else:
                vecs = embed_with_backoff(emb, texts[lo:hi])
                arr = np.asarray(vecs, dtype=np.float32)
                # [NOOK-N2-FIX] catch silent 3072-dim output (old langchain-google-genai
                # ignoring output_dimensionality) BEFORE it touches disk.
                if arr.shape != (hi - lo, config.EMBED_DIM):
                    print(f"ERROR: got shape {arr.shape}, expected ({hi-lo}, {config.EMBED_DIM}). "
                          "Upgrade: pip install -U langchain-google-genai", file=sys.stderr)
                    return 1
                atomic_save_npy(ck, arr)
                print(f"[embed] batch {b+1}/{n_batches} done ({hi-lo} vectors)")
                time.sleep(args.sleep)
            parts.append(arr)
    except Exception as e:
        print(f"ERROR: embedding failed: {e}\nCheckpoints kept; just re-run to resume.",
              file=sys.stderr)
        return 1

    matrix = np.concatenate(parts, axis=0).astype(np.float32)
    assert matrix.shape == (len(texts), config.EMBED_DIM), matrix.shape

    atomic_save_npy(config.EMBEDDINGS_FILE, matrix)
    # [NOOK-N6-FIX] chunk metadata saved in the SAME order as the matrix rows
    config.CHUNKS_META_FILE.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    config.EMBED_HASH_FILE.write_text(h)  # written LAST: hash present => artifacts complete
    shutil.rmtree(CKPT_DIR, ignore_errors=True)
    print(f"[embed] OK: {matrix.shape} float32 -> {config.EMBEDDINGS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())