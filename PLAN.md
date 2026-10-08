# Construction RAG — Frozen Plan

A small, cited RAG over three Indian construction documents of different types, plus a fine-tuned
retrieval embedder. Scope target: about one day of work. This is an interview demo, not a production system.

## 1. Corpus

| Doc | Type | Pages | Extraction reality | Chunk unit | Citation form |
|---|---|---|---|---|---|
| CPWD GCC 2020 | Contract | 112 | Text layer with noise: `1234567890…` ruler rows, legacy-font Hindi gibberish, left-margin side headings | Clause / sub-clause (e.g. 10CC, 25) | `CPWD GCC 2020, Clause 10CC, p.28 (PDF 30)` |
| Ssangyong v NHAI (SC, 2019) | Judgment | 90 | Clean text with numbered paragraphs | Groups of paragraphs, ~300–400 tokens, never split mid-¶ | `Ssangyong v NHAI, ¶48, p.88` |
| IS 875 (Part 3):1987 | Standard | 67 | Scan with an OCR layer; tables are unusable; **Amendments 1–3 appended** | Clause (5.3.2.1 …) + 3 curated tables | `IS 875-3, Cl. 5.3.2.2 / Table 2, p.12` |

Every citation carries both the **PDF page** and the **printed page** (they differ in all three docs).

**Amendments.** Amd 2 changes Table 28 (row 2, col 2) from 1.0 to 1.8, rewrites 6.2.2.8/6.2.2.9, swaps
the Fig 13B/13C captions, and more. Amd 3 rewrites clause 5.5. These are chunked separately with
`is_amendment=True, amends=[...]`. At retrieval time, if an original chunk is selected, the
application **pulls in the amendments that amend it** and labels them as superseding. This happens in
code, not only in the prompt. The scope of the corpus is stated plainly as "IS 875-3:1987 with Amd 1–3".
I don't make claims about later revisions unless they are verified.

**Curated tables** (`data/curated_tables/*.md`, `source=curated`): Table 2 (k2), Table 28 (with the Amd 2
fix applied and noted), and Appendix A (city basic wind speeds).

## 2. Architecture

```
PDFs ─ PyMuPDF ─ per-doc cleaner ─ structure-aware chunker ─ chunks.jsonl (+metadata)
                                                   │
                         ┌─────────────────────────┴─────────────┐
                      BM25 (rank_bm25)                  BGE-small embeddings (numpy)
                         └──────────────┬────────────────────────┘
                                   RRF fusion → amendment expansion → top 6
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
| Sparse | rank_bm25 | Exact identifiers: "Clause 10CC", "Table 28", "Section 34" |
| Dense | BAAI/bge-small-en-v1.5 → fine-tuned | 33M params, fast on CPU, cheap to fine-tune |
| Store | numpy + JSONL behind a `Retriever` interface | ~1–2K chunks needs no vector DB; Qdrant/pgvector is a one-file swap |
| Fusion | Reciprocal Rank Fusion (k=60) | Needs no score calibration |
| Reranker | **Optional, off by default** | Only if time allows; measured in the ablation |
| LLM | Ollama qwen2.5:3b-instruct, swapped via env var | Free and local; fits a 4 GB GPU or runs on CPU |
| Abstention | RRF/dense score threshold, calibrated on unanswerable gold questions | "Not in the provided documents" |
| UI | Streamlit + CLI | Answer, source cards (doc, clause/¶, both pages, snippet) |

## 3. Fine-tuning element: domain-adapted retrieval embedder

**Decision:** fine-tune `bge-small-en-v1.5` on corpus-specific (query → passage) pairs with hard negatives.

**Why this instead of LoRA on the generator:** retrieval sets the ceiling for RAG quality, and an embedder
fine-tune has an objective before/after metric (Recall@k, MRR). It also trains in minutes on a GTX 1650.
A generator LoRA is hard to evaluate and risks confident, ungrounded answers.

- **Data:** about 150–300 synthetic queries (2 per chunk) from qwen2.5:3b. Filters: drop near-duplicates,
  drop queries that copy a long n-gram from the chunk, and drop queries whose positive isn't in BM25+dense top-50.
- **Hard negatives:** sibling/adjacent clauses (10CA vs 10CC, 5.3.1 vs 5.3.2, neighbouring ¶s) and BM25 near-misses.
- **Training:** sentence-transformers, `MultipleNegativesRankingLoss` with explicit hard negatives, lr ~1e-5 to 2e-5,
  1–3 epochs, small validation split. Run locally on CUDA (`requirements-train.txt`).
- **Protocol (both are reported):**
  1. *Transductive:* train on synthetic queries over all chunks and evaluate on human gold questions.
     This mirrors production, where you adapt to the corpus you serve. Only the query wording is unseen.
  2. *Strict:* hold out ~20% of sections entirely (no training queries) and report gold questions on those sections only.
     This measures generalisation. The sample is small, so the result is labelled as noisy.
- **Pre-committed rule:** ship whichever embedder wins on the held-out gold set. If the fine-tune doesn't help,
  ship the base model and report that honestly.
- **Delivery:** the fine-tuned model goes to a free Hugging Face Hub repo and downloads automatically. If that fails, the app uses the base model.

## 4. Evaluation

**Gold set** (`eval/gold.jsonl`, about 36 questions, drafted by Claude and **each verified by hand against the PDF page**):
10 per document, 3 cross-document (e.g. CPWD 10CA/10CC escalation vs the Ssangyong price-adjustment dispute),
and 3 unanswerable or amendment-sensitive. Each record: `question, gold_chunk_ids, answer_key, type, doc`.
Synthetic training queries are **never** used for evaluation.

**Retrieval ablation:** Recall@1/5/10 and MRR, broken down by doc and question type:

| Retriever | R@1 | R@5 | MRR |
|---|---|---|---|
| BM25 | | | |
| Dense (base) | | | |
| Hybrid (base) | | | |
| Hybrid (fine-tuned), transductive | | | |
| Hybrid (fine-tuned), strict split | | | |
| (+ reranker, if built) | | | |

**Generation:** answer correctness (exact match on numeric questions, manual check otherwise), citation correctness
(the cited chunk supports the claim), and abstention accuracy. Plus a manual review of about 15 answers.
The numbers are indicative only because n≈36. Bootstrap CIs are optional.

## 5. Repo layout

```
README.md            setup + "ask a question" in ≤5 commands
PLAN.md              this file
DECISIONS.md         assumptions, rejected alternatives, failure modes, scale path
data/pdfs/           the three source PDFs
data/curated_tables/ hand-curated IS 875 tables (markdown)
src/
  ingest/            extract.py, clean_cpwd.py, chunk_cpwd.py, chunk_ssangyong.py, chunk_is875.py
  index.py           build BM25 + embeddings → artifacts/
  retrieve.py        Retriever: bm25, dense, hybrid(RRF), amendment expansion
  generate.py        LLM adapter (ollama | groq | none), prompt
  cite.py            citation validator
  ask.py             CLI (--debug prints BM25/dense/RRF rankings)
  ui.py              Streamlit
finetune/            make_pairs.py, train.py
eval/                gold.jsonl, run.py, report.md
artifacts/           chunks.jsonl, bm25.pkl, embeddings_*.npy  (prebuilt, committed)
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

1. Scaffold, extraction, **CPWD cleaner + chunker** (the riskiest piece), sample chunks reviewed.
2. BM25 + CLI over CPWD only. This is the first end-to-end answer.
3. Ssangyong + IS 875 chunkers, curated tables, amendment metadata.
4. Dense + RRF + `--debug`; Ollama generation + citation validator + retrieval-only fallback.
5. Gold set drafted, verified by you, baselines run.
6. Synthetic pairs → fine-tune → both protocols → ablation table.
7. Streamlit UI, README, DECISIONS.md, HF Hub upload, clean-machine test.

**If time runs short, cut in this order:** reranker → UI polish → strict-split protocol.
**Never cut:** eval, amendment handling, clean-machine test.

## 8. Explicitly out of scope
OCR pipeline, vector DB server, agents, query rewriting/HyDE, generator LoRA, auth, async ingestion, Docker.
These are described in DECISIONS.md as the path to scale (100+ docs: Qdrant/pgvector, layout-model OCR for tables,
document versioning so amendments supersede cleanly, an ingestion queue).
