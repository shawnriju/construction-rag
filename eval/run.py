"""Evaluate the RAG pipeline on the frozen gold set.

Usage:
    python -m eval.run retrieval  [--label base] [--model M]   # ranking + LLM context, ~1 min, no LLM
    python -m eval.run generation [--label base] [--model M]   # answer quality, ~15 min, needs Ollama
    python -m eval.run compare --base base --new finetuned     # per-question comparison + ship verdict

Without --model the committed index is used. With --model (an HF id or a local path, e.g. the
fine-tuned embedder) the chunks are embedded with that model in memory (~1 min on CPU), so two
embedders can be compared without a second index file (PLAN.md section 3, Delivery).

Reports and per-question JSON go to eval/results/ (see eval/retrieval.py and eval/generation.py).
"""

from __future__ import annotations

import argparse

from sentence_transformers import SentenceTransformer

from eval.generation import evaluate_generation
from eval.generation import write_results as write_generation_results
from eval.gold import load_gold
from eval.retrieval import evaluate_retrieval, format_comparison, load_results
from eval.retrieval import write_results as write_retrieval_results
from src.config import CHUNKS_PATH, EVAL_RESULTS_DIR
from src.index import embed_passages
from src.pipeline import RAGPipeline
from src.retrieve import Retriever
from src.schema import load_chunks


def _progress(message: str) -> None:
    print(message, flush=True)


def load_retriever(model_name: str | None) -> Retriever:
    """The committed index, or (with a model) the same chunks embedded with that model in memory."""
    if model_name is None:
        return Retriever.load()
    print(f"Embedding the chunks with {model_name} (in memory)...", flush=True)
    chunks = load_chunks(CHUNKS_PATH)
    model = SentenceTransformer(model_name, device="cpu")
    return Retriever(chunks, embed_passages(model, chunks), model)


def run_retrieval(label: str, model_name: str | None) -> None:
    results = evaluate_retrieval(load_retriever(model_name), load_gold(), progress=_progress)
    report, data = write_retrieval_results(results, label)
    print(f"Wrote {report} and {data.name}")


def run_generation(label: str, model_name: str | None) -> None:
    pipeline = RAGPipeline(retriever=load_retriever(model_name))
    if not pipeline.llm.available():
        # Scoring retrieval-only fallbacks as answers would be meaningless, so stop instead.
        raise SystemExit(f"LLM backend '{pipeline.llm.name}' is not reachable. Start Ollama and try again.")
    questions = load_gold()
    print(f"Asking {len(questions)} questions with {pipeline.llm.name} (about 15 minutes)...", flush=True)
    results = evaluate_generation(pipeline, questions, progress=_progress)
    report, data = write_generation_results(results, label, pipeline.llm.name)
    print(f"Wrote {report} and {data.name}")


def run_compare(base_label: str, new_label: str) -> None:
    report = EVAL_RESULTS_DIR / f"comparison_{base_label}_vs_{new_label}.md"
    text = format_comparison(load_results(base_label), load_results(new_label), base_label, new_label)
    report.write_text(text, encoding="utf-8")
    print(text[: text.index("## Ranking quality")].strip())
    print(f"Wrote {report}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline on the gold set.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("retrieval", "ranking quality, LLM context coverage, low-confidence cutoff (no LLM)"),
        ("generation", "answer quality: key values, abstention, citations (needs the LLM)"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--label", default="base", help="name for this run, used in the output file names")
        command.add_argument("--model", help="embedding model to evaluate instead of the committed index")
    compare = commands.add_parser("compare", help="compare two retrieval runs and apply the ship rule")
    compare.add_argument("--base", default="base", help="label of the reference run")
    compare.add_argument("--new", required=True, help="label of the run to judge (e.g. finetuned)")
    args = parser.parse_args()

    if args.command == "retrieval":
        run_retrieval(args.label, args.model)
    elif args.command == "generation":
        run_generation(args.label, args.model)
    else:
        run_compare(args.base, args.new)


if __name__ == "__main__":
    main()
