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

EVAL_DIR = PROJECT_ROOT / "eval"
GOLD_PATH = EVAL_DIR / "gold.jsonl"                # Hand-verified evaluation questions.
GOLD_REVIEW_PATH = EVAL_DIR / "gold_review.md"     # Checking sheet rendered from GOLD_PATH.
EVAL_RESULTS_DIR = EVAL_DIR / "results"            # Reports + per-question JSON written by eval.run.

FINETUNE_DATA_DIR = PROJECT_ROOT / "finetune" / "data"
CANDIDATES_PATH = FINETUNE_DATA_DIR / "candidates.jsonl"  # Raw LLM questions per chunk (resumable).
TRAIN_PAIRS_PATH = FINETUNE_DATA_DIR / "train.jsonl"
VAL_PAIRS_PATH = FINETUNE_DATA_DIR / "val.jsonl"
PAIRS_REPORT_PATH = FINETUNE_DATA_DIR / "pairs_report.md"

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
# The fine-tuned bge-small (finetune/train.py), published on the Hugging Face Hub because it is
# over GitHub's 100 MB file limit. It shipped under the pre-committed rule (docs/PLAN.md §3).
# Used by `python -m src.index`; at query time the retriever loads the model named in index_meta.json.
FINETUNED_EMBED_MODEL = "shawnriju/bge-small-construction-rag"
EMBED_MODEL = os.getenv("EMBED_MODEL", FINETUNED_EMBED_MODEL)
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
# Identifier pinning: if the question names a number-like identifier ("Article 142")
# that occurs in at most this many chunks, those chunks are always included.
# Common ones ("0.2" in 15 chunks, "Section 34" in 93) don't trigger it.
RARE_IDENTIFIER_MAX_CHUNKS = 3
MAX_PINNED = 2
# Below this cosine similarity of the best dense hit, results are flagged as
# low-confidence. Calibrated on the gold set with the fine-tuned model (fine-tuning spread
# the cosine scale): flags 6/7 unanswerable and 1/26 answerable questions. Was 0.55 for the base model.
LOW_CONFIDENCE_COSINE = 0.51

# --- Generation --------------------------------------------------------------
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")  # "ollama" | "none"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
LLM_TIMEOUT_SECONDS = 180
# Greedy decoding + a fixed seed: the same question over the same sources always
# gets the same answer, so a demo or an eval run can be reproduced. (At 0.1, the
# Chennai question flipped to "not found" in 2 of 10 identical runs.)
LLM_TEMPERATURE = 0.0
LLM_SEED = 42
# Context window requested from Ollama. The largest prompt (6 hits + 3 table
# parts + 4 amendments, each <= ~450 tokens) is ~6k tokens, leaving room for the answer.
LLM_CONTEXT_TOKENS = 8192
# Ollama's health check should fail fast, so a missing server never stalls the CLI.
LLM_HEALTH_TIMEOUT_SECONDS = 3

# --- Fine-tuning data (finetune/make_pairs.py; docs/PLAN.md section 3) -------------
QUESTIONS_PER_CHUNK = 2   # Run time depends on the number of chunks (one LLM call each), not on this.
FINETUNE_SEED = 42
# A question sharing a run of this many words with its passage is too easy: it teaches string matching.
COPIED_PHRASE_WORDS = 6
# Two questions whose word sets overlap this much are near-duplicates; the later one is dropped.
NEAR_DUPLICATE_JACCARD = 0.8
# If the passage isn't in the BM25 or dense top N for its own question, the question is probably bad.
RETRIEVABLE_TOP_N = 50
# Leak guard: drop a training question this similar (base-embedder cosine) to any gold question.
GOLD_LEAK_COSINE = 0.85
# Hard negative = the best BM25 hit among the top N that isn't the passage, its section or its table.
HARD_NEGATIVE_POOL = 10
VALIDATION_FRACTION = 0.1  # Held back from training to watch for overfitting.

# --- Fine-tuning run (finetune/train.py, GTX 1650 with 4 GB) --------------------
FINETUNE_OUTPUT_DIR = PROJECT_ROOT / "finetune" / "output" / "bge-small-construction"  # gitignored
TRAIN_EPOCHS = 3              # The best epoch on the validation split (MAP@100) is kept (catches overfitting).
TRAIN_LEARNING_RATE = 2e-5
TRAIN_WARMUP_RATIO = 0.1
# Contrastive loss: every question is also contrasted with the other passages in its batch, so a
# bigger batch gives more negatives. GradCache (CachedMultipleNegativesRankingLoss) keeps that
# 32-question batch but runs it through the GPU 8 at a time, so 512-token passages fit in 4 GB.
TRAIN_BATCH_SIZE = 32
TRAIN_MINI_BATCH_SIZE = 8
