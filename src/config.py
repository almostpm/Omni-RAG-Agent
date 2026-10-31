"""config.py - single source of truth for the RAG Agent project (Phase 1).

Never hardcode the project path anywhere else: PROJECT_ROOT is resolved from
this file's location, so the space in 'D:\\projects\\RAG Agent' is harmless.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# [NOOK-N3-FIX] Anchor everything to __file__, never a hardcoded Windows path.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Load .env FIRST before checking for keys
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True, encoding="utf-8")

# Now check if the key is available
if not os.getenv("RAG_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    print("WARNING: RAG_API_KEY is not set yet! .env file check again.")

# [NOOK-N3-FIX] Anchor everything to __file__, never a hardcoded Windows path.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
DATA_DIR: Path = PROJECT_ROOT / "data"
PDF_DIR: Path = DATA_DIR / "pdfs"
CHUNKS_DIR: Path = DATA_DIR / "chunks"
INDEX_DIR: Path = PROJECT_ROOT / "index"

CHUNKS_FILE: Path = CHUNKS_DIR / "chunks.jsonl"
EMBEDDINGS_FILE: Path = INDEX_DIR / "embeddings.npy"
CHUNKS_META_FILE: Path = INDEX_DIR / "chunks_meta.json"
EMBED_HASH_FILE: Path = INDEX_DIR / ".embed_hash"
FAISS_INDEX_FILE: Path = INDEX_DIR / "faiss.index"
INDEX_META_FILE: Path = INDEX_DIR / "index_meta.json"

# text-embedding-004 was shut down (2026-01-14). gemini-embedding-001 defaults
# to 3072 dims, so we force Matryoshka truncation to 768 via output_dimensionality.
EMBED_MODEL: str = "models/gemini-embedding-001"
EMBED_DIM: int = 768
# [NOOK-N2-FIX] Hard lock on dimensionality.
assert EMBED_DIM == 768, "EMBED_DIM must stay 768 for Phase 1 (index portability lock)."

# [NOOK-N9-FIX] The two task types are constants so they cannot be mixed up.
DOC_TASK: str = "RETRIEVAL_DOCUMENT"
QUERY_TASK: str = "RETRIEVAL_QUERY"

CHUNK_SIZE: int = 800      # [NOOK-N4-FIX] chars per chunk
CHUNK_OVERLAP: int = 150   # [NOOK-N4-FIX] overlap so boundary answers survive
TOP_K: int = 3             # [NOOK-N7-FIX] default, overridable via --top-k
BATCH_SIZE: int = 100
SCORE_THRESHOLD: float = 0.3
MIN_PDF_CHARS: int = 50

load_dotenv(PROJECT_ROOT / ".env")
# Force override and enforce UTF-8 encoding so Windows doesn't mess it up
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True, encoding="utf-8")

def mask_key(key: str) -> str:
    """Show only the first 6 chars of a key."""
    return key[:6] + "..."


def get_api_key() -> str:
    """Return the Gemini API key or exit loudly. Mirrors into GOOGLE_API_KEY."""
    primary = os.getenv("RAG_API_KEY")
    mirror = os.getenv("GOOGLE_API_KEY")
    key = primary or mirror
    if not key:
        # [NOOK-N1-FIX] No silent fallback: fail loudly, non-zero exit.
        sys.exit(
            "ERROR: neither RAG_API_KEY nor GOOGLE_API_KEY is set.\n"
            "Fix (cmd):  setx RAG_API_KEY \"your_key\"  then open a NEW terminal,\n"
            f"or put RAG_API_KEY=your_key in {PROJECT_ROOT / '.env'}"
        )
    if primary and not mirror:
        # LangChain auto-loaders look for GOOGLE_API_KEY, so mirror it for this process.
        print("WARN: RAG_API_KEY set but GOOGLE_API_KEY missing; mirroring for this process.",
              file=sys.stderr)
        os.environ["GOOGLE_API_KEY"] = primary
    return key


def ensure_dirs() -> None:
    """Create data/index directories if missing."""
    for d in (PDF_DIR, CHUNKS_DIR, INDEX_DIR):
        d.mkdir(parents=True, exist_ok=True)


def build_embedder(task_type: str):
    """Build a Gemini embedder locked to EMBED_MODEL / EMBED_DIM and one task type."""
    assert task_type in (DOC_TASK, QUERY_TASK), f"bad task_type: {task_type}"  # [NOOK-N9-FIX]
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    key = get_api_key()
    return GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=key,
        task_type=task_type,
        output_dimensionality=EMBED_DIM,
    )