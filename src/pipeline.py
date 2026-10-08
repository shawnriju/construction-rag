"""End-to-end question answering: retrieve -> generate -> validate citations.

Used by both the CLI (src/ask.py) and the Streamlit UI (src/ui.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.cite import CitationReport, check_citations
from src.config import TOP_K
from src.generate import SYSTEM_PROMPT, LLM, build_prompt, get_llm
from src.retrieve import Hit, Retriever, SearchResult


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

        if not use_llm or not self.llm.available():
            if use_llm:
                warnings.append(f"LLM backend '{self.llm.name}' is not reachable; showing retrieved passages only.")
            text = "Most relevant passages (no generated answer):"
            return Answer(question, text, "retrieval-only", result.hits, warnings=warnings,
                          seconds=time.perf_counter() - start)

        text = self.llm.complete(SYSTEM_PROMPT, build_prompt(question, result.hits))
        report = check_citations(text, len(result.hits))
        if report.invalid:
            warnings.append(f"Answer cites non-existent sources: {', '.join(f'S{n}' for n in report.invalid)}.")
        if report.uncited_sentences:
            warnings.append(f"{len(report.uncited_sentences)} sentence(s) in the answer have no citation.")
        return Answer(question, text, "generated", result.hits, report, warnings, time.perf_counter() - start)
