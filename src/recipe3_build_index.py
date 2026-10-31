"""recipe3_build_index.py - build + persist a cosine-similarity FAISS index.

IndexFlatIP over L2-normalized vectors == cosine similarity.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import faiss
import numpy as np

import config


def main() -> int:
    config.ensure_dirs()
    for p in (config.EMBEDDINGS_FILE, config.CHUNKS_META_FILE):
        if not p.exists():
            print(f"ERROR: {p} missing. Run recipe2 first.", file=sys.stderr)
            return 1

    vecs = np.load(config.EMBEDDINGS_FILE)
    chunks = json.loads(config.CHUNKS_META_FILE.read_text(encoding="utf-8"))

    if vecs.ndim != 2 or vecs.shape[0] != len(chunks):
        print(f"ERROR: embeddings {vecs.shape} vs {len(chunks)} chunks - row mismatch.",
              file=sys.stderr)
        return 1
    # [NOOK-N2-FIX] assert dim before anything is written
    if vecs.shape[1] != config.EMBED_DIM:
        print(f"ERROR: vector dim {vecs.shape[1]} != {config.EMBED_DIM}.", file=sys.stderr)
        return 1

    # faiss needs C-contiguous float32; normalize_L2 works in place.
    vecs = np.ascontiguousarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1)
    if (norms == 0).any():
        print(f"ERROR: {(norms == 0).sum()} zero-norm vectors (degenerate input chunk).",
              file=sys.stderr)
        return 1
    # [NOOK-N8-FIX] normalize BEFORE adding to IndexFlatIP, else scores are not cosine
    faiss.normalize_L2(vecs)

    index = faiss.IndexFlatIP(config.EMBED_DIM)
    index.add(vecs)
    assert index.ntotal == len(chunks)

    # [NOOK-N3-FIX] faiss C++ I/O wants a plain str; Path objects / odd quoting break
    # on 'D:\projects\RAG Agent'. Round-trip read proves the path works.
    faiss.write_index(index, str(config.FAISS_INDEX_FILE))
    check = faiss.read_index(str(config.FAISS_INDEX_FILE))
    if check.ntotal != index.ntotal or check.d != config.EMBED_DIM:
        print("ERROR: round-trip read of faiss.index does not match.", file=sys.stderr)
        return 1

    meta = {
        "embed_model": config.EMBED_MODEL,   # [NOOK-N2-FIX] [NOOK-N6-FIX] checked at query time
        "dim": config.EMBED_DIM,
        "normalization": "l2",               # [NOOK-N8-FIX]
        "metric": "inner_product",
        "index_type": "IndexFlatIP",
        "doc_task_type": config.DOC_TASK,
        "query_task_type": config.QUERY_TASK,
        "num_vectors": int(index.ntotal),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    config.INDEX_META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[index] OK: {index.ntotal} vectors -> {config.FAISS_INDEX_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())