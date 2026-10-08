"""Fine-tune bge-small on the corpus question -> passage pairs (docs/PLAN.md section 3).

Run in the training venv, on the GPU:
    .venv-train\\Scripts\\activate
    python -m finetune.train

Input: finetune/data/train.jsonl and val.jsonl from `python -m finetune.make_pairs filter`.
Output: the model in finetune/output/bge-small-construction/ (gitignored; it goes to the HF Hub),
and finetune/data/train_summary.json (settings, counts, validation scores before/after; committed).

Loss: MultipleNegativesRankingLoss (each question is pulled towards its passage and pushed away
from its hard negative and from every other passage in the batch), in its cached (GradCache) form
so a 32-question batch fits in 4 GB. The batch sampler never puts the same text twice in a batch,
so a passage with two questions is never used as a "wrong" passage for its own other question.
After each epoch the validation questions are searched against all chunks; the best epoch (by the
evaluator's primary metric, MAP@100 in this library version) is kept.

Heavy training imports live inside `train()`, so the runtime venv (no `datasets`) can still import
and test the helpers here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from finetune.make_pairs import Pair
from src.config import (
    BASE_EMBED_MODEL,
    BGE_QUERY_PREFIX,
    CHUNKS_PATH,
    FINETUNE_DATA_DIR,
    FINETUNE_OUTPUT_DIR,
    FINETUNE_SEED,
    PROJECT_ROOT,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_LEARNING_RATE,
    TRAIN_MINI_BATCH_SIZE,
    TRAIN_PAIRS_PATH,
    TRAIN_WARMUP_RATIO,
    VAL_PAIRS_PATH,
)
from src.schema import Chunk, load_chunks

TRAIN_SUMMARY_PATH = FINETUNE_DATA_DIR / "train_summary.json"
CHECKPOINT_DIR = FINETUNE_OUTPUT_DIR.parent / "checkpoints"
LOGGING_STEPS = 5


def load_pairs(path: Path, chunks: dict[str, Chunk]) -> list[Pair]:
    """Read pairs and check every chunk id still exists (the chunks are frozen; this guards it)."""
    with path.open(encoding="utf-8") as f:
        pairs = [Pair(**json.loads(line)) for line in f if line.strip()]
    unknown = {cid for p in pairs for cid in (p.chunk_id, p.negative_id) if cid not in chunks}
    if unknown:
        raise ValueError(f"{path.name} references chunks that are not in the corpus: {sorted(unknown)[:5]}")
    return pairs


def training_columns(pairs: list[Pair], chunks: dict[str, Chunk]) -> dict[str, list[str]]:
    """(anchor, positive, negative) text columns, embedded exactly as at search time:
    the question with the BGE query prefix, the passages as `index_text` (breadcrumb + body)."""
    return {
        "anchor": [BGE_QUERY_PREFIX + p.query for p in pairs],
        "positive": [chunks[p.chunk_id].index_text for p in pairs],
        "negative": [chunks[p.negative_id].index_text for p in pairs],
    }


def retrieval_eval_inputs(pairs: list[Pair], chunks: dict[str, Chunk]) -> tuple[dict, dict, dict]:
    """Queries, corpus (all chunks) and relevant chunk per query, for the validation evaluator."""
    queries = {f"q{i}": BGE_QUERY_PREFIX + p.query for i, p in enumerate(pairs)}
    corpus = {cid: chunk.index_text for cid, chunk in chunks.items()}
    relevant = {f"q{i}": {p.chunk_id} for i, p in enumerate(pairs)}
    return queries, corpus, relevant


def train() -> None:
    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU visible. Run this in the training venv (.venv-train) on the GPU machine.")

    chunks = {c.chunk_id: c for c in load_chunks(CHUNKS_PATH)}
    train_pairs = load_pairs(TRAIN_PAIRS_PATH, chunks)
    val_pairs = load_pairs(VAL_PAIRS_PATH, chunks)
    print(f"Training on {len(train_pairs)} pairs, validating on {len(val_pairs)}, GPU: {torch.cuda.get_device_name(0)}",
          flush=True)

    model = SentenceTransformer(BASE_EMBED_MODEL, device="cuda")
    evaluator = InformationRetrievalEvaluator(*retrieval_eval_inputs(val_pairs, chunks), name="val",
                                              show_progress_bar=False)
    # Evaluate the base model first: it is the "before" score, and the evaluator only knows the
    # name of its primary metric (used to pick the best epoch) after it has run once.
    before = evaluator(model)
    if not evaluator.primary_metric:
        raise RuntimeError("The validation evaluator reported no primary metric; cannot select the best epoch.")
    print(f"Validation before training: {evaluator.primary_metric} = {before[evaluator.primary_metric]:.4f}", flush=True)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        learning_rate=TRAIN_LEARNING_RATE,
        warmup_ratio=TRAIN_WARMUP_RATIO,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{evaluator.primary_metric}",
        greater_is_better=True,
        logging_steps=LOGGING_STEPS,
        seed=FINETUNE_SEED,
        data_seed=FINETUNE_SEED,
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_dict(training_columns(train_pairs, chunks)),
        loss=CachedMultipleNegativesRankingLoss(model, mini_batch_size=TRAIN_MINI_BATCH_SIZE),
        evaluator=evaluator,
    )

    start = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - start
    after = evaluator(model)  # The best epoch has been reloaded (load_best_model_at_end).
    print(f"Validation after training:  {evaluator.primary_metric} = {after[evaluator.primary_metric]:.4f}")

    model.save(str(FINETUNE_OUTPUT_DIR))
    summary = {
        "base_model": BASE_EMBED_MODEL,
        "output": FINETUNE_OUTPUT_DIR.relative_to(PROJECT_ROOT).as_posix(),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "settings": {"epochs": TRAIN_EPOCHS, "learning_rate": TRAIN_LEARNING_RATE, "warmup_ratio": TRAIN_WARMUP_RATIO,
                     "batch_size": TRAIN_BATCH_SIZE, "mini_batch_size": TRAIN_MINI_BATCH_SIZE, "seed": FINETUNE_SEED,
                     "loss": "CachedMultipleNegativesRankingLoss", "batch_sampler": "no_duplicates"},
        "gpu": torch.cuda.get_device_name(0),
        "train_seconds": round(seconds, 1),
        # Relative path: this file is committed, so it must not carry the author's machine paths.
        "best_checkpoint": Path(trainer.state.best_model_checkpoint).relative_to(PROJECT_ROOT).as_posix(),
        "val_before": {k: round(v, 4) for k, v in before.items()},
        "val_after": {k: round(v, 4) for k, v in after.items()},
        "log": trainer.state.log_history,  # Loss every few steps + validation scores per epoch.
    }
    TRAIN_SUMMARY_PATH.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"Saved the model to {FINETUNE_OUTPUT_DIR} and the summary to {TRAIN_SUMMARY_PATH} ({seconds / 60:.1f} min)")


if __name__ == "__main__":
    train()
