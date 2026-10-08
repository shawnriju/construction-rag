"""Build (question -> passage, hard negative) training data for the embedder fine-tune.

PLAN.md section 3. Two steps, so the slow LLM part can be resumed and checked first:

    python -m finetune.make_pairs generate [--limit 30]   # LLM questions per chunk (Ollama)
    python -m finetune.make_pairs filter                  # filters, hard negatives, train/val split

`generate` asks the local LLM for QUESTIONS_PER_CHUNK questions per chunk and appends them to
finetune/data/candidates.jsonl after every chunk, so an interrupted run resumes where it stopped
(and a --limit trial is reused by the full run). It prints one progress line per chunk with an ETA.

`filter` keeps a question only if it is not a near-duplicate, does not mention "the passage", does
not copy a long phrase from its passage (too easy: teaches string matching), is not too similar to a
gold question (leak guard), and its passage is in the BM25 or dense top N for it (otherwise the
question is probably bad).
Each kept question gets one hard negative: the best BM25 hit that is not the passage, its section,
its table or a linked amendment (so "Clause 10CC part 2" is never taught as wrong for a 10CC question).
Writes train.jsonl / val.jsonl (chunk ids, not texts: the chunks are frozen) and pairs_report.md.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from eval.gold import load_gold
from src.config import (
    BGE_QUERY_PREFIX,
    CANDIDATES_PATH,
    COPIED_PHRASE_WORDS,
    FINETUNE_SEED,
    GOLD_LEAK_COSINE,
    HARD_NEGATIVE_POOL,
    NEAR_DUPLICATE_JACCARD,
    PAIRS_REPORT_PATH,
    QUESTIONS_PER_CHUNK,
    RETRIEVABLE_TOP_N,
    TRAIN_PAIRS_PATH,
    VAL_PAIRS_PATH,
    VALIDATION_FRACTION,
)
from src.generate import LLM, LLMError, get_llm
from src.retrieve import Retriever
from src.schema import Chunk

SYSTEM_PROMPT = "You write realistic search questions for training a document retrieval system. Output only the questions."

QUESTION_PROMPT = """Passage from {doc_title}, {section}:
\"\"\"
{text}
\"\"\"

Write {n} different questions that this passage answers. Write them the way an engineer, contractor or
lawyer would really ask: in your own words, without copying phrases from the passage, and without
mentioning clause, paragraph, table or page numbers. Each question must make sense on its own.
Output only the questions, one per line, with no numbering."""

_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+\s*[.):]|q\d*\s*[.):])\s*", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+(?:[.'-][a-z0-9]+)*")
_PART_SUFFIX = re.compile(r"\s*\(part \d+/\d+\)$")
# Real users never say "the passage": such a question was written about the prompt, not the topic.
_MENTIONS_PROMPT = re.compile(r"\b(?:the|this|given) (?:passage|excerpt|text)\b", re.IGNORECASE)
MIN_QUESTION_WORDS = 4
REPORT_SAMPLE_SIZE = 20
REPORT_DROPPED_EXAMPLES = 3

# --- generate --------------------------------------------------------------------


def parse_questions(raw: str, limit: int = QUESTIONS_PER_CHUNK) -> list[str]:
    """Questions from the LLM reply: one per line, list markers and quotes stripped, duplicates removed."""
    questions: list[str] = []
    for line in raw.splitlines():
        question = _LIST_MARKER.sub("", line).strip().strip("\"'").strip()
        if question.endswith("?") and len(question.split()) >= MIN_QUESTION_WORDS and question not in questions:
            questions.append(question)
    return questions[:limit]


def load_candidates(path: Path = CANDIDATES_PATH) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def spread_sample(chunks: list[Chunk], limit: int) -> list[Chunk]:
    """`limit` chunks spread evenly over the corpus (all three documents), deterministic."""
    if limit >= len(chunks):
        return list(chunks)
    return [chunks[i * len(chunks) // limit] for i in range(limit)]


def _duration(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}m{seconds:02d}s"


def generate(chunks: list[Chunk], llm: LLM, path: Path = CANDIDATES_PATH, limit: int | None = None) -> None:
    """Ask the LLM for questions per chunk; append each result immediately (resumable)."""
    done = {c["chunk_id"] for c in load_candidates(path) if not c.get("error")}
    todo = [c for c in (spread_sample(chunks, limit) if limit else chunks) if c.chunk_id not in done]
    print(f"{len(done)} chunks already done, {len(todo)} to go with {llm.name}.", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with path.open("a", encoding="utf-8") as f:
        for n, chunk in enumerate(todo, start=1):
            prompt = QUESTION_PROMPT.format(doc_title=chunk.doc_title, section=chunk.section, text=chunk.text,
                                            n=QUESTIONS_PER_CHUNK)
            record = {"chunk_id": chunk.chunk_id, "questions": []}
            try:
                record["questions"] = parse_questions(llm.complete(SYSTEM_PROMPT, prompt))
            except LLMError as error:
                record["error"] = str(error)  # Retried on the next run.
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            elapsed = time.perf_counter() - start
            left = elapsed / n * (len(todo) - n)
            status = record.get("error") or f"{len(record['questions'])} questions"
            print(f"[{n}/{len(todo)}] {chunk.chunk_id}: {status} | {_duration(elapsed)} elapsed, ~{_duration(left)} left",
                  flush=True)


# --- filter --------------------------------------------------------------------------


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def copies_phrase(question: str, passage: str, n: int = COPIED_PHRASE_WORDS) -> bool:
    """True if the question repeats a run of n consecutive words from the passage."""
    q, p = words(question), words(passage)
    passage_ngrams = {tuple(p[i:i + n]) for i in range(len(p) - n + 1)}
    return any(tuple(q[i:i + n]) in passage_ngrams for i in range(len(q) - n + 1))


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def section_family(chunk: Chunk) -> str:
    """'Table 2 (part 3/4)' -> 'Table 2'; '¶ 33 (part 2/8)' -> '¶ 33'."""
    return _PART_SUFFIX.sub("", chunk.section)


def related(a: Chunk, b: Chunk) -> bool:
    """Same passage, section, table, or linked by an amendment: never a valid negative for each other."""
    if a.chunk_id == b.chunk_id:
        return True
    if a.doc_id == b.doc_id and section_family(a) == section_family(b):
        return True
    if a.parent and a.parent == b.parent:
        return True
    return bool(set(a.amends) & set(b.covers) or set(b.amends) & set(a.covers))


def hard_negative(positive: Chunk, bm25_ranked: list[int], chunks: list[Chunk]) -> Chunk | None:
    """The best-ranked BM25 hit (within HARD_NEGATIVE_POOL) that is unrelated to the positive."""
    for i in bm25_ranked[:HARD_NEGATIVE_POOL]:
        if not related(positive, chunks[i]):
            return chunks[i]
    return None


@dataclass
class Pair:
    query: str
    chunk_id: str
    negative_id: str


@dataclass
class Dropped:
    query: str
    chunk_id: str
    reason: str
    detail: str = ""


def filter_candidates(candidates: list[dict], retriever: Retriever, gold_questions: list[str]) -> tuple[list[Pair], list[Dropped]]:
    """Apply the filters in order (cheap ones first) and mine one hard negative per kept question."""
    by_id = {c.chunk_id: i for i, c in enumerate(retriever.chunks)}
    kept: list[tuple[str, str]] = []
    dropped: list[Dropped] = []
    seen: list[set[str]] = []
    for record in candidates:
        chunk = retriever.chunks[by_id[record["chunk_id"]]]
        for question in record["questions"]:
            tokens = set(words(question))
            if any(jaccard(tokens, other) >= NEAR_DUPLICATE_JACCARD for other in seen):
                dropped.append(Dropped(question, chunk.chunk_id, "near-duplicate"))
            elif _MENTIONS_PROMPT.search(question):
                dropped.append(Dropped(question, chunk.chunk_id, "refers to 'the passage'"))
            elif copies_phrase(question, chunk.text):
                dropped.append(Dropped(question, chunk.chunk_id, "copies a phrase from its passage"))
            else:
                kept.append((question, chunk.chunk_id))
            seen.append(tokens)

    if not kept:
        return [], dropped
    def encode(texts: list[str]) -> np.ndarray:
        # Same encoding as Retriever.dense_ranking (BGE query prefix, normalised), batched for speed.
        return retriever.model.encode([BGE_QUERY_PREFIX + t for t in texts], normalize_embeddings=True, batch_size=64)

    query_vectors = encode([q for q, _ in kept])
    gold_similarity = query_vectors @ encode(gold_questions).T if gold_questions else np.zeros((len(kept), 1))
    dense_scores = query_vectors @ retriever.embeddings.T

    pairs: list[Pair] = []
    for row, (question, chunk_id) in enumerate(kept):
        index = by_id[chunk_id]
        closest = int(np.argmax(gold_similarity[row]))
        if gold_questions and gold_similarity[row, closest] >= GOLD_LEAK_COSINE:
            detail = f"{gold_similarity[row, closest]:.2f} to gold: {gold_questions[closest]}"
            dropped.append(Dropped(question, chunk_id, "too similar to a gold question", detail))
            continue
        bm25 = [i for i, _ in retriever.bm25_ranking(question, k=RETRIEVABLE_TOP_N)]
        dense = {int(i) for i in np.argsort(-dense_scores[row])[:RETRIEVABLE_TOP_N]}
        if index not in bm25 and index not in dense:
            dropped.append(Dropped(question, chunk_id, f"passage not in BM25 or dense top {RETRIEVABLE_TOP_N}"))
            continue
        negative = hard_negative(retriever.chunks[index], bm25, retriever.chunks)
        if negative is None:
            dropped.append(Dropped(question, chunk_id, "no unrelated hard negative in the BM25 top 10"))
            continue
        pairs.append(Pair(question, chunk_id, negative.chunk_id))
    return pairs, dropped


def split(pairs: list[Pair], fraction: float = VALIDATION_FRACTION, seed: int = FINETUNE_SEED) -> tuple[list[Pair], list[Pair]]:
    """Deterministic shuffle, then the first `fraction` becomes the validation split."""
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    n_val = round(len(shuffled) * fraction)
    return shuffled[n_val:], shuffled[:n_val]


def write_pairs(pairs: list[Pair], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(p), ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")


def format_pairs_report(candidates: list[dict], pairs: list[Pair], dropped: list[Dropped],
                        train: list[Pair], val: list[Pair], chunks: dict[str, Chunk]) -> str:
    total = sum(len(c["questions"]) for c in candidates)
    errors = sum(1 for c in candidates if c.get("error"))
    reasons = Counter(d.reason for d in dropped)
    lines = [
        "# Fine-tuning pairs - filter report",
        "",
        f"Chunks with questions: {sum(1 for c in candidates if c['questions'])} "
        f"(LLM errors: {errors}). Candidate questions: {total}.",
        f"**Kept: {len(pairs)}/{total} ({100 * len(pairs) / total:.0f}%)** -> train {len(train)}, validation {len(val)}."
        if total else "No candidate questions.",
        "",
        "| Dropped because | count |",
        "|---|---|",
        *(f"| {reason} | {count} |" for reason, count in reasons.most_common()),
        "",
        f"## Sample of kept questions ({REPORT_SAMPLE_SIZE}, spread over the list)",
        "",
        "Check: would a real engineer or lawyer ask this? Is it answered by the passage? Is the negative really wrong?",
        "",
    ]
    step = max(1, len(pairs) // REPORT_SAMPLE_SIZE)
    for pair in pairs[::step][:REPORT_SAMPLE_SIZE]:
        positive, negative = chunks[pair.chunk_id], chunks[pair.negative_id]
        lines.append(f"- **{pair.query}**  \n  passage: {positive.citation}  \n  negative: {negative.citation}")
    lines += ["", "## Examples of dropped questions", ""]
    for reason in reasons:
        for d in [d for d in dropped if d.reason == reason][:REPORT_DROPPED_EXAMPLES]:
            detail = f" ({d.detail})" if d.detail else ""
            lines.append(f"- *{reason}*: {d.query} [{d.chunk_id}]{detail}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training pairs for the embedder fine-tune.")
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate", help="ask the LLM for questions per chunk (resumable)")
    gen.add_argument("--limit", type=int, help="only this many chunks, spread over the corpus (a trial run)")
    commands.add_parser("filter", help="filter questions, mine hard negatives, write train/val + report")
    args = parser.parse_args()

    retriever = Retriever.load()
    if args.command == "generate":
        llm = get_llm()
        if not llm.available():
            raise SystemExit(f"LLM backend '{llm.name}' is not reachable. Start Ollama and try again.")
        generate(retriever.chunks, llm, limit=args.limit)
        return

    candidates = load_candidates()
    pairs, dropped = filter_candidates(candidates, retriever, [q.question for q in load_gold()])
    train, val = split(pairs)
    write_pairs(train, TRAIN_PAIRS_PATH)
    write_pairs(val, VAL_PAIRS_PATH)
    chunks = {c.chunk_id: c for c in retriever.chunks}
    PAIRS_REPORT_PATH.write_text(format_pairs_report(candidates, pairs, dropped, train, val, chunks), encoding="utf-8")
    print(f"Kept {len(pairs)} questions: train {len(train)}, validation {len(val)}. Report: {PAIRS_REPORT_PATH}")


if __name__ == "__main__":
    main()
