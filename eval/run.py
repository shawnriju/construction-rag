"""Evaluate the RAG pipeline on the frozen gold set.

Usage:
    python -m eval.run retrieval  [--label base]   # ranking + LLM context coverage, ~1 min, no LLM
    python -m eval.run generation [--label base]   # answer quality, ~15 min, needs Ollama running

Reports and per-question JSON go to eval/results/ (see eval/retrieval.py and eval/generation.py).
"""

from __future__ import annotations

import argparse

from eval.generation import evaluate_generation
from eval.generation import write_results as write_generation_results
from eval.gold import load_gold
from eval.retrieval import evaluate_retrieval
from eval.retrieval import write_results as write_retrieval_results
from src.pipeline import RAGPipeline
from src.retrieve import Retriever


def _progress(message: str) -> None:
    print(message, flush=True)


def run_retrieval(label: str) -> None:
    results = evaluate_retrieval(Retriever.load(), load_gold(), progress=_progress)
    report, data = write_retrieval_results(results, label)
    print(f"Wrote {report} and {data.name}")


def run_generation(label: str) -> None:
    pipeline = RAGPipeline()
    if not pipeline.llm.available():
        # Scoring retrieval-only fallbacks as answers would be meaningless, so stop instead.
        raise SystemExit(f"LLM backend '{pipeline.llm.name}' is not reachable. Start Ollama and try again.")
    questions = load_gold()
    print(f"Asking {len(questions)} questions with {pipeline.llm.name} (about 15 minutes)...", flush=True)
    results = evaluate_generation(pipeline, questions, progress=_progress)
    report, data = write_generation_results(results, label, pipeline.llm.name)
    print(f"Wrote {report} and {data.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline on the gold set.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("retrieval", "ranking quality, LLM context coverage, low-confidence cutoff (no LLM)"),
        ("generation", "answer quality: key values, abstention, citations (needs the LLM)"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--label", default="base", help="name for this run, used in the output file names")
    args = parser.parse_args()

    if args.command == "retrieval":
        run_retrieval(args.label)
    else:
        run_generation(args.label)


if __name__ == "__main__":
    main()
