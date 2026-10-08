"""The gold evaluation set: hand-verified questions with the passages that answer them.

Each question names its gold chunks *and* says where they are in the PDF (section,
PDF page) with a short verbatim quote. The chunk id is what the scorer uses; the
section, page and quote make every record self-describing, so if a rebuild ever
renames or reshapes chunks, `validate` reports exactly which questions broke and
a human can repair them from the PDF.

File format (`eval/gold.jsonl`, one JSON object per line):

    {"id": "cpwd-01", "question": "...", "doc": "cpwd", "type": "numeric",
     "wording": "natural", "answer": "...", "must_include": ["1", "10"],
     "evidence": [{"chunk_id": "...", "section": "Clause 2", "pdf_pages": [16],
                   "quote": "shall not exceed 10 % (ten percent)"}],
     "note": "..."}

Usage:
    python -m eval.gold            # validate eval/gold.jsonl against artifacts/chunks.jsonl
    python -m eval.gold --review   # also (re)write the checking sheet eval/gold_review.md
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.cite import NOT_FOUND
from src.config import CHUNKS_PATH, GOLD_PATH, GOLD_REVIEW_PATH
from src.schema import Chunk, load_chunks

DOCS = ("cpwd", "ssangyong", "is875", "cross", "none")
TYPES = ("fact", "numeric", "table", "amendment", "cross_doc", "unanswerable")
# "natural": worded the way an engineer or lawyer would ask, not copied from the text.
# "document": uses the document's own wording. Reported separately, because the
# fine-tuned embedder is expected to help mostly on natural wording.
WORDINGS = ("natural", "document")


@dataclass(frozen=True)
class Evidence:
    """One passage needed to answer a question, located three ways (id, page, quote)."""

    chunk_id: str
    section: str
    pdf_pages: tuple[int, ...]
    quote: str  # Verbatim from the chunk text (whitespace and case are ignored when matching).


@dataclass(frozen=True)
class GoldQuestion:
    """One evaluation question.

    `evidence` lists every passage the full answer needs. Retrieval is scored two ways:
    "found" (at least one evidence chunk retrieved) and "fully supported" (all of them).
    An unanswerable question has no evidence, and its answer is the exact NOT_FOUND phrase.
    """

    id: str
    question: str
    doc: str
    type: str
    wording: str
    answer: str
    must_include: tuple[str, ...] = ()  # Key values a correct answer must state.
    evidence: tuple[Evidence, ...] = ()
    note: str = ""

    @property
    def gold_chunk_ids(self) -> tuple[str, ...]:
        return tuple(e.chunk_id for e in self.evidence)

    @property
    def answerable(self) -> bool:
        return self.type != "unanswerable"

    @classmethod
    def from_dict(cls, data: dict) -> "GoldQuestion":
        evidence = tuple(
            Evidence(e["chunk_id"], e["section"], tuple(e["pdf_pages"]), e["quote"])
            for e in data.get("evidence", [])
        )
        return cls(
            id=data["id"],
            question=data["question"],
            doc=data["doc"],
            type=data["type"],
            wording=data["wording"],
            answer=data["answer"],
            must_include=tuple(data.get("must_include", [])),
            evidence=evidence,
            note=data.get("note", ""),
        )


def load_gold(path: Path = GOLD_PATH) -> list[GoldQuestion]:
    """Read the gold set. Raises ValueError naming the line if a record is malformed."""
    questions = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                questions.append(GoldQuestion.from_dict(json.loads(line)))
            except (KeyError, TypeError, json.JSONDecodeError) as err:
                raise ValueError(f"{path.name} line {line_no}: malformed gold record ({err!r})") from err
    return questions


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def validate(questions: list[GoldQuestion], chunks: list[Chunk]) -> list[str]:
    """Return every problem found (empty list = valid).

    Checks the record itself (ids, allowed labels, answerable vs unanswerable) and
    that each evidence entry still matches the corpus: the chunk exists, its PDF
    pages agree, and the quote is in its text. The last two catch a chunk id that
    survived a rebuild but now holds different text.
    """
    by_id = {c.chunk_id: c for c in chunks}
    problems: list[str] = []
    seen: set[str] = set()

    for q in questions:
        where = f"{q.id}:"
        if q.id in seen:
            problems.append(f"{where} duplicate id")
        seen.add(q.id)
        if q.doc not in DOCS:
            problems.append(f"{where} doc {q.doc!r} not in {DOCS}")
        if q.type not in TYPES:
            problems.append(f"{where} type {q.type!r} not in {TYPES}")
        if q.wording not in WORDINGS:
            problems.append(f"{where} wording {q.wording!r} not in {WORDINGS}")
        if not q.question.strip() or not q.answer.strip():
            problems.append(f"{where} empty question or answer")

        if q.answerable:
            if not q.evidence:
                problems.append(f"{where} answerable question without evidence")
        elif q.evidence or q.answer != NOT_FOUND or q.doc != "none":
            problems.append(f"{where} unanswerable questions need no evidence, doc 'none' and answer {NOT_FOUND!r}")

        for e in q.evidence:
            chunk = by_id.get(e.chunk_id)
            if chunk is None:
                problems.append(f"{where} chunk {e.chunk_id!r} ({e.section}, PDF {list(e.pdf_pages)}) is not in the corpus")
                continue
            if not set(e.pdf_pages) <= set(chunk.pdf_pages):
                problems.append(f"{where} {e.chunk_id} is on PDF {chunk.pdf_pages}, gold says {list(e.pdf_pages)}")
            if _normalise(e.quote) not in _normalise(chunk.text):
                problems.append(f"{where} quote not found in {e.chunk_id}: {e.quote[:60]!r}")
    return problems


def _printed_pages(chunk: Chunk, pdf_pages: tuple[int, ...]) -> list[str]:
    """Printed page numbers for the given PDF pages of a chunk.

    The parsers fill `pdf_pages` and `printed_pages` in step (one printed page per PDF page);
    if a chunk ever breaks that, fall back to all of its printed pages rather than guess.
    """
    if len(chunk.pdf_pages) != len(chunk.printed_pages):
        return list(chunk.printed_pages)
    printed = dict(zip(chunk.pdf_pages, chunk.printed_pages))
    return [printed[p] for p in pdf_pages if p in printed]


def render_review_sheet(questions: list[GoldQuestion], chunks: list[Chunk]) -> str:
    """Markdown checking sheet: per question, the expected answer and where to look in the PDF."""
    by_id = {c.chunk_id: c for c in chunks}
    lines = [
        "# Gold set - checking sheet",
        "",
        "Generated by `python -m eval.gold --review` from `eval/gold.jsonl`; edit the JSONL, not this file.",
        "",
        "For each question: open the PDF page, find the quote, and check that the expected answer is right",
        "and complete. Mark `[x]` when it is correct, or write what is wrong underneath.",
        "Unanswerable questions: check that the documents really don't answer them.",
        "",
    ]
    for q in questions:
        lines += [
            f"## {q.id} - {q.type}, {q.wording} wording",
            "",
            f"- [ ] **Q:** {q.question}",
            f"- **Expected answer:** {q.answer}",
        ]
        if q.must_include:
            lines.append(f"- **Must state:** {', '.join(q.must_include)}")
        for e in q.evidence:
            chunk = by_id.get(e.chunk_id)
            doc = chunk.doc_title if chunk else "?"
            pages = ", ".join(str(p) for p in e.pdf_pages)
            printed_pages = _printed_pages(chunk, e.pdf_pages) if chunk else []
            printed = f" (printed p. {', '.join(printed_pages)})" if printed_pages else ""
            lines.append(f"- **Look at:** {doc}, {e.section}, PDF p. {pages}{printed}")
            lines.append(f"  > {e.quote}")
        if q.note:
            lines.append(f"- *Why this question:* {q.note}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the gold set (and optionally write the checking sheet).")
    parser.add_argument("--review", action="store_true", help=f"write {GOLD_REVIEW_PATH.name}")
    args = parser.parse_args()

    questions = load_gold()
    chunks = load_chunks(CHUNKS_PATH)
    problems = validate(questions, chunks)
    for problem in problems:
        print("PROBLEM", problem)

    counts: dict[str, int] = {}
    for q in questions:
        for key in (f"doc={q.doc}", f"type={q.type}", f"wording={q.wording}"):
            counts[key] = counts.get(key, 0) + 1
    print(f"{len(questions)} questions; " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))

    if args.review:
        GOLD_REVIEW_PATH.write_text(render_review_sheet(questions, chunks), encoding="utf-8")
        print(f"Wrote {GOLD_REVIEW_PATH}")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
