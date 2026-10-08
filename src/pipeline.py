"""End-to-end question answering: retrieve -> generate -> validate citations.

Used by both the CLI (src/ask.py) and the Streamlit UI (src/ui.py).

The pipeline never fails because of the LLM: if the backend is missing,
unreachable, times out or errors, the answer degrades to "retrieval-only"
(the ranked, cited passages) with a warning saying why.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.cite import CitationReport, check_citations
from src.config import TOP_K
from src.generate import LLM, SYSTEM_PROMPT, LLMError, build_prompt, get_llm, source_blocks, tidy_answer
from src.retrieve import Hit, Retriever, SearchResult


RETRIEVAL_ONLY_TEXT = "Most relevant passages (no generated answer):"


@dataclass
class Answer:
    question: str
    text: str
    mode: str                      # "generated" | "retrieval-only"
    sources: list[Hit]             # Numbered sources: sources[0] is [S1].
    citations: CitationReport = field(default_factory=CitationReport)
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0


class RAGPipeline:
    def __init__(self, retriever: Retriever | None = None, llm: LLM | None = None):
        self.retriever = retriever or Retriever.load()
        self.llm = llm or get_llm()

    def ask(self, question: str, top_k: int = TOP_K, mode: str = "hybrid", use_llm: bool = True) -> Answer:
        start = time.perf_counter()
        result: SearchResult = self.retriever.search(question, top_k=top_k, mode=mode)
        warnings = []
        if result.low_confidence:
            warnings.append(
                f"Low retrieval confidence (best similarity {result.best_cosine:.2f}); "
                "the documents may not cover this question."
            )

        def retrieval_only(reason: str | None) -> Answer:
            if reason:
                warnings.append(f"{reason}; showing retrieved passages only.")
            return Answer(question, RETRIEVAL_ONLY_TEXT, "retrieval-only", result.hits, warnings=warnings,
                          seconds=time.perf_counter() - start)

        if not use_llm:
            return retrieval_only(None)
        if not self.llm.available():
            return retrieval_only(f"LLM backend '{self.llm.name}' is not reachable")
        try:
            text = tidy_answer(self.llm.complete(SYSTEM_PROMPT, build_prompt(question, result.hits)))
        except LLMError as error:
            return retrieval_only(f"LLM error ({error})")

        report = check_citations(text, len(result.hits), source_blocks(result.hits), question)
        if report.invalid:
            warnings.append(f"Answer cites non-existent sources: {', '.join(f'S{n}' for n in report.invalid)}.")
        if report.uncited_sentences:
            warnings.append(f"{len(report.uncited_sentences)} sentence(s) in the answer have no citation.")
        if report.ungrounded_numbers:
            numbers = ", ".join(dict.fromkeys(number for number, _ in report.ungrounded_numbers))
            warnings.append(f"Number(s) not found in the sources they cite: {numbers}. Check them against the sources.")
        return Answer(question, text, "generated", result.hits, report, warnings, time.perf_counter() - start)
