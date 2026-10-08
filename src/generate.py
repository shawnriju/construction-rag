"""Prompt construction and the LLM adapter.

The LLM is swappable through config (LLM_BACKEND / OLLAMA_MODEL env vars).
Adding a new backend means writing one class with `available()` and
`complete()`; nothing else in the pipeline changes.

Prompt design notes (measured on qwen2.5:3b, see docs/progress.md):
  * An amendment in a separate source block is easy for a small model to
    miss: it quoted the superseded Table 28 value (1.0) even with an explicit
    "overrides [S2]" note. The amendment text is therefore also placed INSIDE
    the original's block, directly above the text it changes.
  * The question is repeated after the sources, so it is the last thing the
    model reads before answering, however long the sources are.
  * The strong amendment rule had a side effect: the model said a formula was
    "amended" because an unrelated amendment was in its sources. Rule 4 now
    limits mentions to amendments that change the text the answer uses.
"""

from __future__ import annotations

import re
from typing import Protocol

import requests

from src.cite import NOT_FOUND
from src.config import (
    LLM_BACKEND,
    LLM_CONTEXT_TOKENS,
    LLM_HEALTH_TIMEOUT_SECONDS,
    LLM_SEED,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    OLLAMA_MODEL,
    OLLAMA_URL,
)
from src.retrieve import Hit

SYSTEM_PROMPT = f"""You answer questions about construction documents using ONLY the numbered sources provided.

Rules:
1. Put source markers right after each fact, inside the sentence, for example: "The limit is 25 days [S3]."
2. Use only information in the sources. Never use outside knowledge.
3. If the sources do not contain the answer, reply exactly: "{NOT_FOUND}"
4. AMENDMENTS: a source marked "AMENDED" has been changed by a later amendment, which is shown with it. The amendment replaces the original wording. When an amendment changes the value or text your answer uses, give the amended value, say what it was amended from, and cite both sources, for example: "The limit is 25 days (amended from 20 days) [S3][S4]." Do not mention amendments that change other provisions.
5. A source may use an older name for a place; a note inside that source gives today's name. Treat the two names as the same place.
6. Copy numbers, units, clause numbers and paragraph numbers exactly as written.
7. Write plain text only. Write formulas in plain text, for example: A = B x C^2. Do not use LaTeX or markdown math.
8. Be concise: two to six sentences, or a short list."""


# --- Prompt -----------------------------------------------------------------------


def amendment_links(hits: list[Hit]) -> dict[int, list[int]]:
    """Map each original source number to the numbers of the amendments (in `hits`) that change it.

    Source numbers are 1-based, matching the [S#] markers in the prompt.
    """
    links: dict[int, list[int]] = {}
    for a, amendment in enumerate(hits, start=1):
        if not amendment.chunk.is_amendment:
            continue
        targets = set(amendment.chunk.amends)
        for o, original in enumerate(hits, start=1):
            if not original.chunk.is_amendment and targets & set(original.chunk.covers):
                links.setdefault(o, []).append(a)
    return links


def _refs(numbers: list[int]) -> str:
    return ", ".join(f"[S{n}]" for n in numbers)


def source_blocks(hits: list[Hit]) -> list[str]:
    """The text of each numbered source exactly as the model sees it (blocks[0] is [S1]).

    Used for the prompt and by the citation check, so both see the same text.
    """
    links = amendment_links(hits)
    amended_by = {a: [o for o, amendments in links.items() if a in amendments] for a in range(1, len(hits) + 1)}

    blocks = []
    for i, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        header = f"[S{i}] {chunk.citation}"
        if chunk.title:
            header += f" - {chunk.title}"
        body = chunk.text
        if chunk.is_amendment:
            targets = amended_by[i]
            header += f"\nThis is an AMENDMENT to {_refs(targets)}." if targets else "\nThis is an AMENDMENT."
        if i in links:
            changes = "\n".join(f"AMENDED by [S{a}]: {hits[a - 1].chunk.text}" for a in links[i])
            body = f"{changes}\nOriginal text (superseded where the amendment differs):\n{body}"
        blocks.append(f"{header}\n{body}")
    return blocks


_LATEX_DELIMITER = re.compile(r"\\[\[\]()]")
_BLANK_LINES = re.compile(r"\n{3,}")


def tidy_answer(text: str) -> str:
    r"""Remove LaTeX math delimiters (\[ \] \( \)) that the model adds despite the plain-text rule.

    Cosmetic and deterministic: "\[ p_z = 0.6 v_z^2 \]" becomes "p_z = 0.6 v_z^2".
    """
    return _BLANK_LINES.sub("\n\n", _LATEX_DELIMITER.sub("", text)).strip()


def build_prompt(question: str, hits: list[Hit]) -> str:
    sources = "\n\n".join(source_blocks(hits))
    return f"Question: {question}\n\nSources:\n\n{sources}\n\nQuestion: {question}\n\nAnswer (cite sources as [S#]):"


# --- LLM backends --------------------------------------------------------------


class LLMError(RuntimeError):
    """The backend was reachable but failed to produce an answer (timeout, HTTP error, bad response)."""


class LLM(Protocol):
    name: str

    def available(self) -> bool: ...

    def complete(self, system: str, prompt: str) -> str:
        """Return the answer text, or raise LLMError."""
        ...


class OllamaLLM:
    """Local model served by Ollama (https://ollama.com)."""

    def __init__(self, model: str = OLLAMA_MODEL, url: str = OLLAMA_URL):
        self.model = model
        self.url = url.rstrip("/")
        self.name = f"ollama:{model}"

    def available(self) -> bool:
        try:
            response = requests.get(f"{self.url}/api/tags", timeout=LLM_HEALTH_TIMEOUT_SECONDS)
            response.raise_for_status()
            models = response.json().get("models", [])
        except (requests.RequestException, ValueError):
            return False
        return any(m.get("name", "").startswith(self.model) for m in models)

    def complete(self, system: str, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {
                        "temperature": LLM_TEMPERATURE,
                        "seed": LLM_SEED,
                        "num_ctx": LLM_CONTEXT_TOKENS,
                    },
                },
                timeout=LLM_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return response.json()["message"]["content"].strip()
        except requests.Timeout as error:
            raise LLMError(f"{self.name} did not answer within {LLM_TIMEOUT_SECONDS}s") from error
        except (requests.RequestException, ValueError, KeyError, TypeError) as error:
            raise LLMError(f"{self.name} failed: {error}") from error


class NoLLM:
    """Retrieval-only mode: used when no generator is configured or reachable."""

    name = "none"

    def available(self) -> bool:
        return False

    def complete(self, system: str, prompt: str) -> str:
        raise LLMError("No LLM backend configured.")


def get_llm(backend: str = LLM_BACKEND) -> LLM:
    if backend == "ollama":
        return OllamaLLM()
    if backend == "none":
        return NoLLM()
    raise ValueError(f"Unknown LLM_BACKEND '{backend}' (expected 'ollama' or 'none').")
