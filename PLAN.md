# Construction RAG — Plan

> The design agreed before building. Where the build differed, this file has been updated to match.
> Live status, per-module notes and open issues are in `progress.md`.
> Revised 2026-10-08 after an external review: scope trimmed (strict split → future work, Streamlit built last), gold set ~30 with more
> unanswerable questions, small-n reporting rules, stronger training-query filtering, and a staged query generator (§3).

A small, cited RAG over three Indian construction documents of different types, plus a fine-tuned
retrieval embedder. Scope target: about one day of work. This is an interview demo, not a production system.

## 1. Corpus

| Doc | Type | Pages | Extraction reality | Chunk unit | Citation form |
|---|---|---|---|---|---|
| CPWD GCC 2020 | Contract | 112 | Text layer with noise: `1234567890…` ruler rows, legacy-font Hindi gibberish, left-margin side headings | Clause / sub-clause (e.g. 10CC, 25) | `CPWD GCC 2020, Clause 10CC, p.28 (PDF 30)` |
| Ssangyong v NHAI (SC, 2019) | Judgment | 90 | Clean text with numbered paragraphs | Groups of paragraphs up to ~400 embedder tokens; a long ¶ is split into "part i/n" | `Ssangyong v NHAI, ¶48, p.88` |
| IS 875 (Part 3):1987 | Standard | 67 | Scan with an OCR layer; tables are unusable; **Amendments 1–3 appended** | Clause (5.3.2.1 …) + 4 curated tables | `IS 875-3, Cl. 5.3.2.2 / Table 2, p.12` |

Every citation carries both the **PDF page** and the **printed page** (they differ in all three docs).

**Amendments.** Amd 2 changes Table 28 (row 2, col 2) from 1.0 to 1.8, rewrites 6.2.2.8/6.2.2.9, swaps
the Fig 13B/13C captions, and more. Amd 3 rewrites clause 5.5. These are chunked separately with
`is_amendment=True, amends=[...]`. At retrieval time, if an original chunk is selected, the
application **pulls in the amendments that amend it** and labels them as superseding. This happens in
code, not only in the prompt. The scope of the corpus is stated plainly as "IS 875-3:1987 with Amd 1–3".
I don't make claims about later revisions unless they are verified.

**Curated content** (`data/curated/is875.yaml`, `source=curated`): Table 1 (k1, with its Note as a separate
entry), Table 2 (k2), Table 28, and Appendix A (city basic wind speeds, plus a labelled curator note mapping old
city names to new ones, e.g. Madras→Chennai). Tables are transcribed **exactly as printed**: Table 28 keeps 1.0,
and the 1.8 correction exists only in the Amd 2 chunk, so the amendment handling is what produces the right answer.
All three amendments are curated item by item, because the Amd 3 page has no text layer.
Table 2 and Appendix A are too long for the embedder's 512-token window, so they are stored as parts
(Table 2 by height, Appendix A alphabetically), each repeating the caption and column headings. Parts are
searched one by one, but when any part is retrieved the code attaches the rest (**sibling expansion**,
"small-to-big"), so the LLM always gets the whole table. This matters for interpolation between rows that sit in different parts.

## 2. Architecture

```
PDFs ─ PyMuPDF ─ per-doc cleaner ─ structure-aware chunker ─ chunks.jsonl (+metadata)
                                                   │
                         ┌─────────────────────────┴─────────────┐
                      BM25 (rank_bm25)                  BGE-small embeddings (numpy)
                         └──────────────┬────────────────────────┘
                                   RRF fusion → top 6 → sibling expansion → amendment expansion
                                        │
                     prompt with numbered sources [S1..S6], "cite every claim, else say not found"
                                        │
                     LLM adapter: Ollama (qwen2.5:3b-instruct) │ optional Groq/Gemini key │ retrieval-only
                                        │
                     citation validator (unknown [S#] / uncited sentences flagged)
                                        │
                              CLI (src.ask --debug) / Streamlit UI
```

| Layer | Choice | Why |
|---|---|---|
| Extraction | PyMuPDF | Fast, keeps pages, no system deps |
| Chunk size | ≤400 tokens of body, measured with bge-small's own tokenizer; the build fails if any chunk + breadcrumb exceeds 512 | Numbers and OCR text cost up to ~3 tokens per word, so a word limit let tables overflow the window and lose their tail silently |
| Sparse | rank_bm25 | Exact identifiers: "Clause 10CC", "Table 28", "Section 34" |
| Dense | BAAI/bge-small-en-v1.5 → fine-tuned | 33M params, fast on CPU, cheap to fine-tune |
| Store | numpy + JSONL behind a `Retriever` interface | ~450 chunks needs no vector DB; Qdrant/pgvector is a one-file swap |
| Fusion | Reciprocal Rank Fusion (k=60) | Needs no score calibration |
| Reranker | **Optional, off by default** | Only if time allows; measured in the ablation |
| LLM | Ollama qwen2.5:3b-instruct, swapped via env var | Free and local; fits a 4 GB GPU or runs on CPU |
| Abstention | Best dense cosine below a threshold → low-confidence warning; LLM told to say "Not found in the provided documents."; threshold calibrated on unanswerable gold questions | "Not in the provided documents" |
| UI | CLI + Streamlit, **Streamlit built last** (after eval and fine-tuning) | The CLI answers "how to ask a question"; Streamlit is for the demo |

## 3. Fine-tuning element: domain-adapted retrieval embedder

**Decision:** fine-tune `bge-small-en-v1.5` (the model that turns text into vectors for meaning-based search)
on (question → passage) pairs from this corpus.

**In plain words (the interview version).** The base embedder learned "what text means" from general web data.
It has never seen how Indian contracts, judgments and wind-load standards phrase things ("Engineer-in-Charge",
"patent illegality", "k2 factor"). We show it a few thousand examples of *"this question is answered by this
passage, and not by that similar-looking one"*. Training pulls each question's vector towards its passage and
pushes it away from the wrong ones. Then we check, on questions it has never seen, whether the right passage now
ranks higher. Training is about 50 lines of standard `sentence-transformers` code and takes minutes on the GTX 1650.

**Why this and not something else:**
- *vs LoRA on the generator:* retrieval sets the ceiling for RAG quality (the LLM can't cite a passage it never
  received), and retrieval has an objective before/after metric (Recall@k, MRR). A generator LoRA is hard to
  evaluate and risks confident, ungrounded answers.
- *vs a classifier (e.g. "which document is this question about?"):* simpler, but with three documents BM25 already
  routes most questions correctly, so it would barely change the answers.

**Assumptions, and how each is checked (not just assumed):**

| # | Assumption | How we check it |
|---|---|---|
| A1 | Retrieval has room to improve (if the base hybrid already finds every gold passage, fine-tuning can't show a gain) | Baseline eval first: look at hybrid **and dense-only** R@1/R@5. If hybrid R@5 is near 100%, say so and focus the claim on dense-only and R@1 |
| A2 | A 3B local model can write useful training questions | Checkpoint before training: filter pass rate + a manual read of ~20 questions. Groq only if this fails (§3, staged generator) |
| A3 | Training questions don't leak the test | Gold set frozen before any training; drop training questions too similar to any gold question |
| A4 | "Wrong" passages used in training really are wrong | Hard negatives never come from the same section or table as the positive (avoids teaching "Clause 10CC part 2 is irrelevant to a Clause 10CC question") |
| A5 | A gain is real, not luck | Per-question wins/losses (n ≈ 30), fixed random seed, no "improves" claim from a 1–2 question gap |

**Data.** 3–4 questions per chunk from **qwen2.5:3b** (~1,300+ candidates from 448 chunks), filtered: near-duplicates;
questions that copy a long phrase from the chunk (too easy, teaches string matching); questions whose passage isn't in
the BM25+dense top 50 (probably a bad question); questions too similar to a gold question (leak guard). Groq is not set up
now; it is considered only if the A2 checkpoint fails or the fine-tune shows no gain. If both runs happen, both are reported.

**Hard negatives (kept simple):** for each question, 1 passage from the BM25 top 10 that is *not* from the positive's
section or table. These are the near-misses (10CA vs 10CC, neighbouring ¶s), which is exactly what the model must learn to separate.

**Training:** `MultipleNegativesRankingLoss` (each question is also contrasted with every other passage in the batch,
for free), lr ~2e-5, 1–3 epochs, fixed seed, 10% of the training questions held back as a validation split to catch
overfitting. Runs locally on CUDA (`requirements-train.txt`); the reviewer's install stays CPU-only.

**Protocol: transductive.** Train on questions over all chunks; test on the human gold questions (new wording, same
documents). This matches the task (a RAG bot over *these* documents) and is reported as **corpus adaptation, not
generalisation to new documents**. A strict hold-out split (sections never seen in training) is **future work**: with
~30 gold questions it would leave ~6 to measure on, too few to mean anything.

**Pre-committed success rule (fixed before training):** compare base vs fine-tuned on the frozen gold set, both dense-only
and hybrid. The fine-tuned model **ships only if** it has more per-question wins than losses on hybrid hit@5 (ties broken by
MRR) **and** breaks no amendment-sensitive question. Otherwise the base model ships, and the result is reported honestly.
The low-confidence threshold is recalibrated for whichever model ships, because fine-tuning changes the similarity scale.

**Delivery:** the fine-tuned model goes to a free Hugging Face Hub repo and downloads automatically. If that fails, the app uses the base model.

## 4. Evaluation

**Gold set** (`eval/gold.jsonl`, about 30 questions, drafted by Claude and **each verified by hand against the PDF page**):
about 7 per document, a few amendment-sensitive (Table 28 φ=0.2, Cl. 5.5), 2–3 cross-document (e.g. CPWD 10CA/10CC
escalation vs the Ssangyong price-adjustment dispute; **the first to cut**), and **6–8 unanswerable** (enough to set
the low-confidence threshold, which is otherwise only a rough heuristic). Each record: `question, gold_chunk_ids, answer_key, type, doc`.
Synthetic training queries are **never** used for evaluation. **The gold set is frozen before any fine-tuning run.**

**Retrieval ablation:** Recall@1/5/10 and MRR, broken down by doc and question type:

| Retriever | R@1 | R@5 | MRR |
|---|---|---|---|
| BM25 | | | |
| Dense (base) | | | |
| Hybrid (base) | | | |
| Dense (fine-tuned) | | | |
| Hybrid (fine-tuned) | | | |
| (+ reranker, if built) | | | |

**Generation:** answer correctness (exact match on numeric questions, manual check otherwise), citation correctness
(the cited chunk supports the claim), and abstention accuracy. Plus a manual review of about 15 answers.
**Reporting with a small n (≈30):** counts next to every percentage ("22/30"), and **per-question wins/losses** between
two systems ("fine-tune fixed 4, broke 1") rather than a claim of "improves" from a 2-point gap. Retrieval quality and
generation quality are reported separately; the 3B generator is the swappable weak link. Bootstrap CIs are optional.

## 5. Repo layout

```
README.md            setup + "ask a question" in ≤5 commands
PLAN.md              this file
DECISIONS.md         assumptions, rejected alternatives, failure modes, scale path
data/pdfs/           the three source PDFs
data/curated/        is875.yaml: hand-curated IS 875 tables + amendments
src/
  config.py          paths, model names, constants (env-var overrides)
  schema.py          Chunk dataclass + JSONL save/load
  ingest/            pdf.py, text.py (shared), cpwd.py, ssangyong.py, is875.py, curated.py, build.py
  index.py           embeddings → artifacts/ (BM25 is rebuilt at load, not persisted)
  retrieve.py        Retriever: bm25, dense, hybrid(RRF), amendment expansion
  generate.py        LLM adapter (ollama | none), prompt; a groq backend only if needed later (§3)
  cite.py            citation validator
  pipeline.py        retrieve → generate → validate (shared by CLI and UI)
  ask.py             CLI (--debug prints BM25/dense/RRF rankings)
  ui.py              Streamlit (built last)
finetune/            make_pairs.py, train.py
eval/                gold.jsonl, run.py, report.md, compare_chunking.py (before/after check for the chunking fix)
artifacts/           chunks.jsonl, embeddings.npy, index_meta.json  (prebuilt, committed)
requirements.txt          runtime, CPU torch
requirements-train.txt    training, CUDA torch
```

## 6. Reviewer experience

```
git clone <repo> && cd construction-rag
python -m venv .venv && .venv\Scripts\activate      # or source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:3b-instruct                     # optional: without it, retrieval-only mode
python -m src.ask "What compensation is payable for delay under the CPWD contract?"
streamlit run src/ui.py
```

The prebuilt index is committed, so no re-ingestion is needed. Without Ollama, the app still returns ranked, cited passages.
Before submitting, it is tested in a fresh venv on a clean clone.

## 7. Build order (vertical slice first)

As built, steps 1–3 were done together (all three parsers + curated content before retrieval); the rest is unchanged.

1. Scaffold, extraction, **CPWD cleaner + chunker** (the riskiest piece), sample chunks reviewed.
2. BM25 + CLI over CPWD only. This is the first end-to-end answer.
3. Ssangyong + IS 875 chunkers, curated tables, amendment metadata.
4. Dense + RRF + `--debug`; Ollama generation + citation validator + retrieval-only fallback.
5. Gold set drafted, verified by you, baselines run.
6. Synthetic pairs (qwen first; Groq if needed, see §3) → fine-tune → transductive protocol → ablation table.
7. Streamlit UI.
8. README, DECISIONS.md (incl. an "expected failure modes" section and why the Hindi documents were skipped), HF Hub upload, clean-machine test.
Future work (DECISIONS.md): strict hold-out split for the fine-tune.

**If time runs short, cut in this order:** reranker → cross-document gold questions → Streamlit polish (a basic UI stays).
**Never cut:** eval, amendment handling, clean-machine test.

## 8. Explicitly out of scope
OCR pipeline, vector DB server, agents, query rewriting/HyDE, generator LoRA, auth, async ingestion, Docker.
These are described in DECISIONS.md as the path to scale (100+ docs: Qdrant/pgvector, layout-model OCR for tables,
document versioning so amendments supersede cleanly, an ingestion queue).
