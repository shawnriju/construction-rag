"""Central configuration: paths, model names and tunable constants.

Everything that a reviewer might want to change lives here (or can be overridden
with an environment variable), so the rest of the code has no magic values.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # Optional .env file in the project root (never committed).

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
CURATED_DIR = PROJECT_ROOT / "data" / "curated"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

CHUNKS_PATH = ARTIFACTS_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "embeddings.npy"
INDEX_META_PATH = ARTIFACTS_DIR / "index_meta.json"

# --- Chunking ----------------------------------------------------------------
# Chunks are sized in the embedder's own tokens, not words: tables, numbers and
# OCR noise cost up to ~3 tokens per word ("0.85" -> "0" "." "85"), so a word
# limit let some chunks overflow the window and lose their tail silently.
EMBED_TOKENIZER = "BAAI/bge-small-en-v1.5"  # A fine-tuned bge-small keeps this tokenizer.
EMBED_MAX_TOKENS = 512        # bge-small's window, incl. the [CLS]/[SEP] tokens.
BREADCRUMB_TOKEN_RESERVE = 64  # The longest breadcrumb is ~60 tokens.
# Budget for a chunk's body. 400 matches what 300 words of prose already were
# (~370-420 tokens), so prose chunking barely changes; dense tables now split.
MAX_CHUNK_TOKENS = 400
MIN_TAIL_TOKENS = 50  # A smaller trailing piece is folded into the previous chunk.

# --- Models ------------------------------------------------------------------
BASE_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# Set EMBED_MODEL to the fine-tuned Hugging Face repo id once it exists.
EMBED_MODEL = os.getenv("EMBED_MODEL", BASE_EMBED_MODEL)
# BGE models retrieve better when *queries* (not passages) carry this prefix.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --- Retrieval ---------------------------------------------------------------
RRF_K = 60            # Standard constant for reciprocal rank fusion.
CANDIDATES_PER_RETRIEVER = 30
TOP_K = 6             # Chunks passed to the LLM.
MAX_ATTACHED = 4      # Extra chunks added by amendment expansion.
# Extra chunks added by sibling expansion (the other parts of a split curated
# table). Separate budget, so table parts never crowd out an amendment.
MAX_SIBLINGS_ATTACHED = 3
# Below this cosine similarity of the best dense hit, results are flagged as
# low-confidence. Provisional value; calibrated on the gold set's unanswerable questions.
LOW_CONFIDENCE_COSINE = 0.55

# --- Generation --------------------------------------------------------------
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")  # "ollama" | "none"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
LLM_TIMEOUT_SECONDS = 180
