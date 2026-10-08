"""Command-line interface: ask a question, get a cited answer.

Usage:
    python -m src.ask "What compensation is payable for delay under the CPWD contract?"
    python -m src.ask                      # interactive: ask several questions, models load once
    python -m src.ask "..." --debug        # also show BM25 / dense / fused ranks per source
    python -m src.ask "..." --no-llm       # retrieval only: ranked, cited passages
    python -m src.ask "..." --mode bm25    # single retriever (hybrid | bm25 | dense)
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from src.config import TOP_K
from src.generate import amendment_links
from src.pipeline import Answer, RAGPipeline

SNIPPET_CHARS = 300   # Passage preview length in retrieval-only mode.
WRAP_WIDTH = 100


def _snippet(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= SNIPPET_CHARS else flat[:SNIPPET_CHARS].rsplit(" ", 1)[0] + " ..."


def _indent(text: str, prefix: str) -> str:
    return textwrap.fill(text, WRAP_WIDTH, initial_indent=prefix, subsequent_indent=" " * len(prefix))


def format_answer(answer: Answer, debug: bool = False) -> str:
    """Render an Answer for the terminal (kept separate from printing so it can be tested)."""
    lines = [f"\nQuestion: {answer.question}", ""]
    header = "Answer" if answer.mode == "generated" else "Retrieval only"
    lines.append(f"{header} ({answer.seconds:.1f}s):")
    lines.extend(_indent(paragraph, "  ") for paragraph in answer.text.splitlines() if paragraph.strip())

    if answer.warnings:
        lines.append("\nWarnings:")
        lines.extend(_indent(warning, "  ! ") for warning in answer.warnings)

    links = amendment_links(answer.sources)
    cited = set(answer.citations.cited)
    lines.append("\nSources:" + ("  (* = cited in the answer)" if answer.mode == "generated" else ""))
    for i, hit in enumerate(answer.sources, start=1):
        chunk = hit.chunk
        mark = "*" if i in cited else " "
        tags = [tag for tag, on in (("AMENDMENT", chunk.is_amendment), ("curated", chunk.source == "curated")) if on]
        title = f" - {chunk.title}" if chunk.title else ""
        lines.append(_indent(f"{chunk.citation}{title}" + (f"  [{', '.join(tags)}]" if tags else ""), f" {mark}[S{i}] "))
        details = []
        if i in links:
            details.append("amended by " + ", ".join(f"[S{a}]" for a in links[i]))
        if hit.attached_reason:
            details.append(f"attached: {hit.attached_reason}")
        if debug:
            ranks = ", ".join(f"{name} #{rank}" for name, rank in hit.ranks.items()) or "not ranked (attached)"
            details.append(f"ranks: {ranks}; score {hit.score:.4f}")
        for detail in details:
            lines.append(_indent(detail, "        "))
        if answer.mode == "retrieval-only":
            lines.append(_indent(_snippet(chunk.text), "        > "))
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.ask",
        description="Ask a question about the CPWD contract, the Ssangyong judgment or IS 875 (Part 3).",
    )
    parser.add_argument("question", nargs="?", help="omit for interactive mode")
    parser.add_argument("--mode", choices=["hybrid", "bm25", "dense"], default="hybrid", help="retriever (default: hybrid)")
    parser.add_argument("--top-k", type=int, default=TOP_K, help=f"retrieved sources before expansion (default: {TOP_K})")
    parser.add_argument("--no-llm", action="store_true", help="skip generation; show ranked passages only")
    parser.add_argument("--debug", action="store_true", help="show per-retriever ranks and fused scores")
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    return args


def main(argv: list[str] | None = None) -> None:
    # Citations contain characters like "¶"; never crash on a console that can't show them.
    sys.stdout.reconfigure(errors="replace")
    args = _parse_args(argv)

    print("Loading the search index and models ...", file=sys.stderr)
    pipeline = RAGPipeline()

    def answer(question: str) -> None:
        result = pipeline.ask(question, top_k=args.top_k, mode=args.mode, use_llm=not args.no_llm)
        print(format_answer(result, debug=args.debug))

    if args.question:
        answer(args.question)
        return
    print("Ask a question (empty line or Ctrl+C to quit).", file=sys.stderr)
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        answer(question)


if __name__ == "__main__":
    main()
