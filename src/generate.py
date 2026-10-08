"""Prompt construction and the LLM adapter.

The LLM is swappable through config (LLM_BACKEND / OLLAMA_MODEL env vars).
Adding a new backend means writing one class with `available()` and
`complete()`; nothing else in the pipeline changes.
"""

from __future__ import annotations

from typing import Protocol

import requests

from src.cite import NOT_FOUND
from src.config import LLM_BACKEND, LLM_TIMEOUT_SECONDS, OLLAMA_MODEL, OLLAMA_URL
from src.retrieve import Hit

SYSTEM_PROMPT = f"""You are a careful assistant for construction engineers. You answer questions using ONLY the numbered sources you are given.

Rules:
1. End every sentence that states a fact with its source marker(s), for example [S1] or [S2][S4].
2. Use only information found in the sources. Never use outside knowledge.
3. If the sources do not contain the answer, reply exactly: "{NOT_FOUND}"
4. Some sources are AMENDMENTS. An amendment overrides the original text it amends: give the amended value or wording, say that it was amended, and cite both the original and the amendment.
5. Copy numbers, percentages, clause numbers and paragraph numbers exactly as written in the sources.
6. Be concise: two to six sentences, or a short list."""


def _source_notes(hits: list[Hit]) -> dict[int, str]:
    """Explicit 'this overrides that' notes, so the model does not have to infer them."""
    notes: dict[int, str] = {}
    for i, amendment in enumerate(hits, start=1):
        if not amendment.chunk.is_amendment:
            continue
        targets = set(amendment.chunk.amends)
        superseded = [
            j for j, other in enumerate(hits, start=1)
            if not other.chunk.is_amendment and targets & set(other.chunk.covers)
        ]
        refs = ", ".join(f"[S{j}]" for j in superseded)
        notes[i] = "AMENDMENT - overrides " + (refs if refs else "the original text") + " where they differ."
        for j in superseded:
            notes.setdefault(j, f"Partly superseded by amendment [S{i}].")
    return notes


def build_prompt(question: str, hits: list[Hit]) -> str:
    notes = _source_notes(hits)
    blocks = []
    for i, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        header = f"[S{i}] {chunk.citation}"
        if chunk.title:
            header += f" - {chunk.title}"
        if i in notes:
            header += f"\nNOTE: {notes[i]}"
        blocks.append(f"{header}\n{chunk.text}")
    sources = "\n\n".join(blocks)
    return f"Sources:\n\n{sources}\n\nQuestion: {question}\n\nAnswer (cite sources as [S#]):"


# --- LLM backends --------------------------------------------------------------


class LLM(Protocol):
    name: str

    def available(self) -> bool: ...

    def complete(self, system: str, prompt: str) -> str: ...


class OllamaLLM:
    """Local model served by Ollama (https://ollama.com)."""

    def __init__(self, model: str = OLLAMA_MODEL, url: str = OLLAMA_URL):
        self.model = model
        self.url = url.rstrip("/")
        self.name = f"ollama:{model}"

    def available(self) -> bool:
        try:
            tags = requests.get(f"{self.url}/api/tags", timeout=3).json()
        except requests.RequestException:
            return False
        return any(m["name"].startswith(self.model) for m in tags.get("models", []))

    def complete(self, system: str, prompt: str) -> str:
        response = requests.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                "stream": False,
                # Low temperature: we want faithful extraction, not creativity.
                "options": {"temperature": 0.1, "num_ctx": 8192},
            },
            timeout=LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()


class NoLLM:
    """Retrieval-only mode: used when no generator is configured or reachable."""

    name = "none"

    def available(self) -> bool:
        return False

    def complete(self, system: str, prompt: str) -> str:
        raise RuntimeError("No LLM backend configured.")


def get_llm(backend: str = LLM_BACKEND) -> LLM:
    if backend == "ollama":
        return OllamaLLM()
    if backend == "none":
        return NoLLM()
    raise ValueError(f"Unknown LLM_BACKEND '{backend}' (expected 'ollama' or 'none').")
