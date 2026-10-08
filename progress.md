# Construction RAG — Progress Log

> **Purpose:** a single hand-off document. Give this file to a new chat and it has all the context needed
> to continue. It is also the author's own record of what was built, how, and why.
> **Keep it updated** after every coding change: update "Current status" and "Next steps", and add any new decisions.

_Last updated: 2026-10-08 (Phases 0–2 done and pushed as the first commit; token-based chunking fix + sibling expansion done, awaiting user test + commit; Phase 3 code written but not yet run)_

---

## 1. The assignment (verbatim intent)

Build a small **RAG pipeline over 2–3 Indian construction documents of different types** that answers questions
**with citations back to the source**, plus **a fine-tuning element** (a fine-tuned/adapted model, classifier,
extractor, or similar). The reviewers care most about **reasoning, assumptions, decision-making, how the work is
tested and evaluated**. It is roughly **one day of work**. Free and open-source tools are fine. It must include a
**run file**: setup steps and how to ask a question. It is for a job application followed by an interview,
so it is not a production system.

**Hard requirement from the user:** the whole stack must be **free** and **easy for the invigilator to run** on their machine.

## 2. Working agreements with the user (follow these)

- **Work in phases.** After each phase, stop, summarise, and give exact test commands. Wait for the go-ahead.
- **Never `git commit`.** The user commits after testing. Suggest a commit message instead.
- **Slow installs** (venv, pip, model pulls): give the user the commands to run themselves; don't run them.
- **Don't search the user's personal folders.** Ask them to put files in a named project folder.
- Code style: readable, well-structured, typed, docstrings explaining *why*, no magic numbers (they live in `src/config.py`).
- Keep this `progress.md` updated.
- **Explain choices slowly, in plain language.** Don't present terse A/B/C options; say what each option means and recommend one. The user wants to understand the code fully once the base implementation is done (a walkthrough is planned).
- **Commit messages:** suggest a single, plain-English line (the user prefers one line over a multi-line body). Only describe what has been tested.

## 3. Environment

- Windows 11, PowerShell, Python 3.12.7, project at `C:\Dev\Projects\construction-rag`, venv in `.venv`.
- Hardware: 16 GB RAM, **GTX 1650 (4 GB VRAM)**. GPU is only for fine-tuning; the runtime is CPU-only.
- Ollama is installed and running with `qwen2.5:3b-instruct` pulled.
- Git: repo **https://github.com/shawnriju/construction-rag** (public, branch `main`). First commit `74f8585` pushed by the user.
- `.gitignore` excludes `.venv/`, caches, `.env`, `.streamlit/secrets.toml`, `artifacts/models/`, `finetune/output/`, editor/OS files and **`docs/prior-plans/`** (draft plans kept local only; `PLAN.md` is the reviewed version). **Committed on purpose:** the PDFs (~13.5 MB, under GitHub's 100 MB file limit) and `artifacts/chunks.jsonl`, `embeddings.npy`, `index_meta.json`, so the reviewer can ask questions without rebuilding.
- Runtime deps (`requirements.txt`, CPU torch): torch 2.4.1, sentence-transformers 3.2.1, transformers 4.45.2,
  pymupdf 1.24.11, rank-bm25 0.2.2, numpy 1.26.4, requests, streamlit 1.39.0, python-dotenv, pyyaml 6.0.3. All installed.
  `tokenizers` and `huggingface_hub` (used by `token_count`) come in with transformers, so nothing new needs installing.
- Running commands from a restricted network (e.g. Claude's sandbox): set `HF_HUB_OFFLINE=1`, or Hugging Face's online check can hang. Not needed on a normal network.
- `requirements-train.txt` (CUDA torch, datasets, accelerate) is **not installed yet**. It's needed in the fine-tuning phase.

## 4. Corpus (the 3 chosen documents, in `data/pdfs/`)

| doc_id | File | Type | Pages | Extraction reality |
|---|---|---|---|---|
| `cpwd` | `CPWD-GCC-2020_construction-contract_EN.pdf` | Contract (General Conditions of Contract 2020) | 112 | Text layer is good but noisy: "1234567890…" ruler rows, legacy-font Hindi gibberish, two-column layout with left **margin headings** (clause titles). Printed page = PDF page − 2. |
| `ssangyong` | `Ssangyong-v-NHAI_SupremeCourt-arbitration_EN.pdf` | Supreme Court judgment (2019, arbitration, s.34) | 90 | Clean. One block per line; numbered paragraphs ¶1–50; quotes indented; 6 section headings. Printed page = PDF page. |
| `is875` | `IS875-3-1987_wind-loads_EN.pdf` | Engineering standard (wind loads) | 67 | **Scanned + OCR.** Two columns, OCR errors ("vaiue", "TEBRAIN"), **tables unusable**. **Amendments 1–3 appended** (PDF 64–67). **Amd 3 (p.67) has NO text layer.** Printed page = PDF page − 4. |

**Why these three:** they are three different types (contract, judgment, standard) with three different extraction
problems, and they link to each other. CPWD price escalation (Cl. 10C/10CA/10CC) and arbitration (Cl. 25) relate to
Ssangyong (a price-adjustment formula dispute and s.34 arbitration review), which gives cross-document questions.

Prior planning notes are in `docs/prior-plans/` (Claude's initial plan, GPT's review, and a Gemini plan PDF). They are **local only (gitignored)**.

## 5. Frozen decisions (from the planning Q&A with the user)

| Topic | Decision | Why / trade-off |
|---|---|---|
| LLM for answers | **Ollama `qwen2.5:3b-instruct`** by default; **retrieval-only fallback** if Ollama isn't running; backend swappable via env var | Free and local. The demo never hard-fails. A free hosted key (Groq/Gemini) is an optional future add-on. |
| Delivery | **GitHub repo** (code + prebuilt chunks/index). **Fine-tuned model goes on Hugging Face Hub** and downloads automatically. | The fine-tuned bge-small (~130 MB) exceeds GitHub's 100 MB file limit. Git LFS was rejected (bandwidth quota). |
| Docker | **No** | A torch image is several GB, and Ollama inside Docker on Windows adds friction. Plain pip in a clean venv instead. |
| Interface | **CLI (`--debug` shows rankings) + Streamlit** | The CLI is for transparency and development; Streamlit is for the demo. |
| Retrieval | **Hybrid BM25 + dense (`BAAI/bge-small-en-v1.5`), fused with RRF (k=60)** | The corpus is full of exact identifiers (Clause 10CC, Table 28, s.34(2)(a)(iii)), which BM25 handles; paraphrased questions need dense retrieval. RRF needs no score calibration. |
| Chunk size | **≤400 tokens of body, counted with bge-small's own tokenizer** (was 300 *words*); build fails if any embedded text (breadcrumb + body) > 512 tokens; over-long curated tables are split into parts that repeat the header | Numbers/symbols/OCR cost up to ~3 tokens per word ("0.85" = 3 tokens), so the word limit let 11 chunks overflow the 512-token window and the embedder silently dropped their tails. Done before the gold set because it renames chunk ids. 400 ≈ the old prose size, so prose granularity stays about the same. |
| Vector store | numpy matrix + JSONL behind a `Retriever` class | Only about 450 chunks, so no vector DB is needed. Swapping to Qdrant/pgvector is a one-file change. |
| Reranker | **Optional, off.** Only if time allows (measured in the ablation). | Not needed for the story. |
| Fine-tuning element | **Fine-tune the bge-small embedder** on corpus question→passage pairs with **hard negatives** (sibling clauses, BM25 near-misses) | Retrieval sets the RAG ceiling. It has an objective before/after metric (Recall@k, MRR) and trains in minutes on the GTX 1650. A generator LoRA was rejected: hard to evaluate, and it risks confident ungrounded answers. |
| Training data | **LLM-synthetic queries** (qwen2.5:3b, ~2 per chunk, ~150–300 pairs), filtered (dedupe, no copied n-grams, positive must be retrievable in the top 50) | Fast. **Gold eval questions are never synthetic.** |
| Training location | Local GTX 1650 (`requirements-train.txt`) | The reviewer's install stays CPU-only. |
| Eval protocol | **Report both:** (1) *transductive*: train on synthetic queries over all chunks, test on human gold questions; (2) *strict*: hold out ~20% of sections entirely | Pre-empts "you memorised the chunks". The strict split is small and noisy, and is labelled as such. Rule: **ship whichever embedder wins; if fine-tuning doesn't help, ship the base model and say so.** |
| Gold set | ~36 questions (10 per doc, 3 cross-doc, 3 unanswerable/amendment-sensitive). **Claude drafts them with gold clause/¶/page; the user verifies each against the PDF.** | Defensible in the interview. |
| IS 875 tables | **Hand-curate 4:** Table 1 (k1 risk coefficients, + its Note as a separate entry), Table 2 (k2), Table 28 (single-frame force coefficients), Appendix A (city wind speeds) | OCR tables are unusable. Tables 1 and 2 together with Vb give the core formula Vz = Vb·k1·k2·k3. Every value was checked against the rendered page image. Hand-curation doesn't scale; the scale path is a layout/vision OCR model. |
| Curated tables are kept **exactly as printed** (Table 28 row 0.2 = **1.0**) | Confirmed with the user | Provenance: the citation must match what the page shows. Corrections live only in amendment chunks, so the amendment handling is what makes the answer right (1.8). This is tested explicitly in the eval (see §11). |
| Split tables → **sibling expansion** ("small-to-big") | A curated table split into parts (Table 2 ×4, Appendix A ×2) is *searched* part by part but *handed to the LLM whole*: when any part is retrieved, `retrieve.py` attaches the missing parts (own budget, `MAX_SIBLINGS_ATTACHED = 3`, so they never crowd out amendments). Curated tables only; OCR table parts are noisy and are left alone. | Dense search can't tell rows apart by number, and an interpolation question needs rows from two parts (e.g. 225 m needs the 200 m row in part 2 and the 250 m row in part 3). Measured: without it, 15/16 table probes get the needed rows into context; with it, 16/16. Cost: ~0.9 extra chunks per question on average (7.0 → 7.9). This is a standard RAG pattern (parent-document retrieval) and a good interview point. |
| Amendments | Curated **per item**, with `is_amendment=True` and `amends=[...]`. Retrieval attaches them **in code** (not only in the prompt). | A naive RAG would quote superseded text. This is the strongest point of the project's story. |
| Scope cuts if late | Cut the reranker, then UI polish, then the strict split. **Never cut** eval, amendment handling, or the clean-machine test. | |
| Out of scope | OCR pipeline, vector DB server, agents, HyDE/query rewriting, generator LoRA, auth, Docker | Described as the "scale path" in DECISIONS.md (to be written). |

Full plan: `PLAN.md`.

## 6. Architecture (as built)

```
data/pdfs/*.pdf ──► per-doc parser (layout-aware) ──┐
data/curated/is875.yaml ──► curated loader ─────────┴─► artifacts/chunks.jsonl (448 chunks)
                                                            │
                         python -m src.index ──► artifacts/embeddings.npy + index_meta.json
                                                            │
question ─► Retriever.search():  BM25 (rebuilt at load) + dense (bge-small, query prefix)
                                 ─► RRF fusion ─► top 6 ─► sibling expansion (+ up to 3) ─► amendment expansion (+ up to 4)
          ─► build_prompt (numbered [S1..Sn], amendment NOTES) ─► Ollama ─► citation check ─► Answer
                                   (Ollama down ─► retrieval-only Answer)
```

### Chunk model (`src/schema.py`)
`Chunk(chunk_id, doc_id, doc_title, section, text, pdf_pages, printed_pages, title, part, covers, is_amendment, amends, source)`
- `breadcrumb` = `doc_title > part > section - title`. `index_text` = breadcrumb + text. **Embedding and BM25 both use `index_text`**, so short chunks inherit context.
- `citation` looks like `CPWD GCC 2020, Clause 10CC, p. 28 (PDF 30)`. Both the printed and the PDF page are kept. Amendments have no printed page, so they cite as `PDF p. 65-66`.
- `covers` = provision ids inside the chunk (e.g. `["Cl. 6.2.2.8","Cl. 6.2.2.9"]`, `["Table 28"]`). It is matched against amendments' `amends`.
- `parent` = for one part of a split curated table, the whole table's name (`"Table 2"`); empty otherwise. Used by sibling expansion. It isn't part of `index_text`, so adding it didn't change the embeddings.
- Chunk size limit: **400 tokens of body** (`MAX_CHUNK_TOKENS`), counted with bge-small's tokenizer (`token_count` in `text.py`). Breadcrumbs take ≤ ~60 tokens (`BREADCRUMB_TOKEN_RESERVE = 64`), so breadcrumb + body + [CLS]/[SEP] stays ≤ 512 (`EMBED_MAX_TOKENS`); `build.py` enforces this. A tiny tail (< 50 tokens, `MIN_TAIL_TOKENS`) is folded into the previous window. Largest chunk after the fix: 504 tokens (a Table 2 part).

## 7. Implementation notes per module

| File | What it does / key techniques |
|---|---|
| `src/config.py` | All paths, model names and constants; env-var overrides (`EMBED_MODEL`, `LLM_BACKEND`, `OLLAMA_MODEL`, `OLLAMA_URL`); loads `.env`. |
| `src/schema.py` | `Chunk` dataclass + `save_chunks`/`load_chunks` (JSONL). |
| `src/ingest/pdf.py` | PyMuPDF → `TextBlock(pdf_page, x0, y0, x1, y1, text)`, sorted top-to-bottom. All parsers use **positions**, not just text. |
| `src/ingest/text.py` | Shared helpers: `normalize` (smart quotes, dot leaders), `join_lines` (de-hyphenation: keeps "Engineer-in-Charge", repairs "construc-tion"), `looks_like_legacy_hindi`, **`token_count`** (bge-small `tokenizer.json` via the `tokenizers` library, no model weights; tries the local HF cache first because the online check can hang on a restricted network; counts add up across whitespace joins), `pack` (greedy paragraph packing, sentence-level split of oversized paragraphs, word-level hard split as a last resort, tail folding), **`Unit`/`Window`/`group_units`** (merge small consecutive units in the same group up to 400 tokens; split oversized units into "part i/n"; fold a heading-only stub into the next long unit). |
| `src/ingest/cpwd.py` | Pages 4–79 and 96–111 only. Skipped: cover/OM/bilingual index (1–3), bilingual register forms (80–95), back cover. Detects **part** from the running header (y<80); strips the footer ("…Engineering Excellence"), ruler rows and Hindi gibberish. **Margin headings** (x0<100, x1<160) only in the two-column parts. All-caps short blocks become headings (a rule excludes formulas like `N = 0.85 M`). **Clauses part:** a new section when a block's first line is exactly `Clause <id>` (68 clauses found: 1 … 41, incl. 10CC, 19A–L, 29A/B); the first margin heading becomes the title, later ones go inline as `[heading]`. **Other parts:** packed windows labelled with item numbers (monotonic; numbered headings and `Article N` are authoritative; continuation shows `item 8 (cont.)`). → **201 chunks.** |
| `src/ingest/ssangyong.py` | Drops the signature stamp (x1 ≤ 99) and the footer. Groups lines by indentation, a **gap > 38pt**, and a **new group at every page break** (exact page citations). ¶ starts = a block whose first line is `N.` at x < 110 **and sequential numbering** (rejects numbers inside quotations). Headings = ≤15-word groups without closing punctuation immediately before a ¶ start. Uses `group_units` → labels like `¶ 23-24`, `¶ 33 (part 2/9)`; the case header is its own chunk. → **119 chunks**, all 50 ¶ and all 6 headings, all 90 pages covered. |
| `src/ingest/is875.py` | Pages 7–12 and 15–62. Skipped: map image p13, blank p14, back page 63, amendment pages 64–67 (curated instead). Reads the **left column then the right**. Clause numbers are found **anywhere in a block** (the OCR merges columns), but are only accepted if `_is_plausible_next` holds (child / next sibling / next ancestor, up to 1 skipped number). Top-level numbers must look like `1. SCOPE` (dot + CAPS); this stopped the foreword's "Part 2 Imposed loads" from being read as clause 2. **Band reading:** the page is cut into horizontal bands at every full-width `TABLE`/`APPENDIX` heading and each band is read left-then-right. Without this, two-column text above a mid-page table (e.g. the Class B/C definitions of 5.3.2.2 on PDF 15) was pulled below the table. OCR `l`/`I` are read as 1 in ids. `TABLE N` must increase by 1–3 (rejects the OCR "TABLE 28" that is really Table 20). Appendix clauses are keyed like `C-2.1`. Units merge only within the same top-level clause; tables never merge. Table 1, Table 2, Table 28 and Appendix A OCR are dropped in favour of the curated versions. `part` = top-level title (e.g. "5. Wind Speed And Pressure"). → **92 chunks.** |
| `data/curated/is875.yaml` | **Hand-transcribed** Table 1 (k1; table and its Note are separate entries to fit the 512-token embedder window; Note wording verbatim incl. the original's typos), Table 2 (**4 parts by height**: 10–30, 50–200, 250–350, 400–500 m; each repeats caption, column headings and Notes 1–2), Table 28 (**as printed: row 0.2 flat-sided = 1.0**), and Appendix A (**2 parts**: Agra–Jamshedpur, Jhansi–Visakhapatnam; each has a clearly labelled curator note mapping old→new names for *its own* cities, e.g. Madras→Chennai in part 2). Split entries use `header`/`parts`/`footer` instead of `text`; the rows were moved, not retyped (checked: every Table 2 row and Appendix A city/speed identical before and after). **All amendments:** Amd 1 (Dec 1997, 3 items), Amd 2 (Mar 2002, 20 items, incl. Table 28 1.0→1.8 and Cl. 6.2.2.8 0.8→−0.8 / 0.5→−0.5), Amd 3 (Mar 2006, 4 items, incl. the rewrite of Cl. 5.5 "Cyclonic Wind Velocity"). `covers` of the curated tables = only themselves. **⚠ The user must verify these transcriptions against the PDF pages.** |
| `src/ingest/curated.py` | YAML → chunks. A split table becomes one chunk per part: `section="Table 2 (part 2/4)"`, id `is875:curated:table-2-part-2-4`, title + part label, **`covers` = the whole table** so amendments still attach, and **`parent` = the table name** so sibling expansion can find the other parts. One chunk per amendment item: `section="Amendment No. 2, item 10"`, `amends=["Table 28"]`, `source="curated"`. → **36 chunks** (Table 1, Table 1 Note, Table 2 ×4, Table 28, Appendix A ×2 + 27 amendment items). |
| `src/ingest/build.py` | `python -m src.ingest.build`: runs all parsers, **validates** (unique ids, no empty chunks, page provenance present, **every chunk's embedded text ≤ 512 tokens**), writes `chunks.jsonl`, prints a summary table (incl. max tokens per doc) and **the chunk ids added/removed vs the previous `chunks.jsonl`** (gold questions and training pairs reference ids, so renames must be visible). |
| `src/index.py` | `python -m src.index [--model X]`: embeds `index_text` (normalized, CPU, batch 32) → `embeddings.npy`. `index_meta.json` stores the model name and a **fingerprint of the chunks**. BM25 is not persisted (rebuilt at load in ms; no pickle). |
| `src/retrieve.py` | `tokenize` keeps clause ids intact (`10cc`, `6.2.2.8`, `c-2.1`; "10 CC"→"10cc") and drops stopwords. `Retriever.load()` **refuses a stale index** (fingerprint mismatch). `search(query, top_k=6, mode="hybrid"\|"bm25"\|"dense", expand=True, siblings=True)` returns `SearchResult(hits, best_cosine)`; each `Hit` has per-retriever `ranks` (for `--debug`). **Sibling expansion** (runs first): after the first retrieved part of a split table, attach its missing parts in document order (`attached_reason` = "rest of Table 2: Table 2 (part 3/4) was retrieved"); max 3. **Amendment expansion:** after an original chunk, attach amendments targeting its `covers`; after a retrieved amendment, attach the original it changes. Max 4 extra. `low_confidence` if best cosine < 0.55 (provisional; to be calibrated on the gold set). |
| `src/cite.py` | `check_citations(answer, n_sources)` → `CitationReport(cited, invalid, uncited_sentences)`. Handles `[S1][S2]` and `[S1, S3]`. Exact phrase `NOT_FOUND = "Not found in the provided documents."` |
| `src/generate.py` | `SYSTEM_PROMPT` (cite every factual sentence; sources only; exact not-found phrase; **amendments override, cite both**; copy numbers exactly; concise). `build_prompt` numbers sources `[S#] citation - title` and adds explicit **NOTE lines** ("AMENDMENT - overrides [S1]" / "Partly superseded by amendment [S2]") so the 3B model needn't infer it. `OllamaLLM` (`/api/chat`, temperature 0.1, num_ctx 8192, `available()` health check), `NoLLM`, `get_llm()`. |
| `src/pipeline.py` | `RAGPipeline.ask(question, top_k, mode, use_llm)` → `Answer(text, mode "generated"/"retrieval-only", sources, citations, warnings, seconds)`. Falls back to retrieval-only with a warning if Ollama is unreachable. |

### Smoke-test results (retrieval, before the gold set)
Re-run after the token-chunking fix (448 chunks):
- "What does Clause 10CC provide for escalation?" → Clause 10CC chunks ranked 1–3 by both retrievers.
- "force coefficient single frame solidity 0.2 flat-sided" → **Table 28 #1, Amendment 2 item 10 (1.0→1.8) #2**.
- "basic wind speed in Chennai" / "Visakhapatnam" → Appendix A part 2 #1 (BM25 and dense); "Mumbai" → part 1 #1. The per-part curator notes work.
- "k2 factor at 300 m height terrain category 3 class B" → Table 2 part 3 (250–350 m) #1, all 4 parts in the top 4.
- "design wind speed multiplier at 450 m height" → Cl. 5.3.2 #1, then Table 2 parts 1 and 3; the correct part 4 isn't in the top 4 (it is in the top 6). Dense embeddings can't tell table rows apart by number, which is why sibling expansion exists (§5).
- Full before/after numbers: `python -m eval.compare_chunking` (see §10).

Before the fix:
- "open ended cylinder internal pressure h/D 0.3" → Amd 2 items 5–6 (sign fixes) + Cl. 6.2.2.8.
- "patent illegality … international commercial arbitration" → Ssangyong ¶30–31 #1.
- "compensation for delay by contractor" → CPWD Clause 2 (all three chunks) top 3.

## 8. Known limitations (to go in DECISIONS.md)
- IS 875: on a few pages (≈ PDF 41–42) the OCR interleaves the two columns, so some sentences are out of order and 6.3.2.4 is detected before 6.3.2.3.
- IS 875: the OCR lost some table headings (Tables 12, 14, 18, 19, 22, 26, 32). Their text is absorbed into the previous table's chunk. The remaining table chunks are OCR-noisy; only the curated Tables 1, 2 and 28 and Appendix A are reliable.
- CPWD: the first-aid list inside Model Rule 3 skews item labels for 2 chunks (pages are still correct). Formulas (10CA/10CC) extract imperfectly.
- Ssangyong: long paragraphs that quote other judgments (¶17, 33, 39–41, 44) become many "part i/n" chunks.
- Hand-curated content doesn't scale (scale path: layout/vision OCR plus document versioning).
- **City names (decided):** Appendix A uses 1987 names (Madras, Bombay, Calcutta…). A clearly labelled **curator note** inside the curated Appendix A chunk maps old→new names (Madras→Chennai, etc.), so "wind speed in Chennai" retrieves it. Trade-off: it's added text, not from the PDF (marked `source: curated`). The scalable alternative is a query-time alias list. A city that isn't in Appendix A at all can only be read off the Fig. 1 map, which is an image and isn't indexed, so the system should answer "not found". **To do:** add a Chennai question and an unlisted-city question to the gold set.
- **Corpus scope:** answers come from IS 875-3:**1987 with Amd 1–3** only. A newer edition exists, but it isn't in the corpus. Neither the system nor the docs should describe it or call the 1987 values "current" (to be stated in the README/UI banner).
- The `LOW_CONFIDENCE_COSINE=0.55` threshold is a guess until calibrated.
- **Token overflow (fixed 2026-10-08):** chunks were limited by *words* (300), but bge-small truncates at **512 tokens** without any error. Measured tokens per word: CPWD 1.23, Ssangyong 1.40, IS 875 1.47 (median), and up to 2.95 in OCR tables. **11 chunks** were over the limit: curated Table 2 (1001 tokens) and Appendix A (526), OCR Tables 4/5/6/8/17/23, Ssangyong ¶37 part 3, the CPWD Annexure. Their tails were invisible to dense retrieval (BM25 and the LLM saw the full text). Fix: token-based packing (≤400 body tokens), Table 2 and Appendix A split into parts with repeated headers, and a build check (≤512). Result: 454 → **448 chunks**, max 504 tokens. **Side effect:** the sizes of *all* windows changed, not only the 11. CPWD prose (~1.2 tokens/word) now fits more per chunk (215 → 201); Ssangyong (~1.4) slightly less (118 → 119). Of 454 old ids, 367 kept their id, but 136 of those have different content (mostly CPWD, whose ids are positional, e.g. `clause-3:4`). 87 ids removed, 81 added. This was acceptable because no gold set or training pairs existed yet.
- **Split tables and the LLM (fixed 2026-10-08 with sibling expansion, §5):** splitting Table 2 meant the LLM could get only some of its rows. Before the split, it got the whole table whenever Table 2 was retrieved.
- **Before/after measurement** (`eval/compare_chunking.py`, old = commit `74f8585`, deterministic, no LLM):
  1. Chunks over the 512-token window: **11 → 0**.
  2. Lost-tail findability (a passage from each old truncated tail used as a dense query; rank of the chunk that holds it): **better for 6 of 11** (e.g. Ssangyong ¶37 189 → 52, OCR Table 5 28 → 1, Table 17 26 → 3), **worse for 3** (OCR Table 4 4 → 18, Table 23 part 1 14 → 25, Table 8 3 → 4), same for 2. The queries are verbatim OCR text and n = 11, so this is a diagnostic, not a headline. Purely numeric table text ranks poorly either way (curated Table 2 tail 38 → 39): dense embeddings don't encode numbers well.
  3. Table lookups (16 questions; does the needed row reach the LLM's context?): **old 15/16, new without siblings 15/16, new with siblings 16/16.** Old missed "k2 at 100 m" (Table 2 not in the top 6). New without siblings missed the 225 m interpolation (needs parts 2 and 3). Average chunks sent to the LLM: 7.6 / 7.0 / 7.9.
  - **Honest verdict:** the guaranteed win is correctness (nothing is silently truncated, and the build enforces it). The measured retrieval gains on these small probes are modest and mostly positive. The real verdict comes from the gold-set eval (Phase 5), which should include the 100 m, 450 m and 225 m-interpolation questions.

## 9. Repo layout (current)
```
PLAN.md  progress.md  requirements.txt  requirements-train.txt  .gitignore
data/pdfs/ (3 PDFs)   data/curated/is875.yaml   docs/prior-plans/ (local only, gitignored)
src/config.py schema.py index.py retrieve.py cite.py generate.py pipeline.py
src/ingest/{pdf,text,cpwd,ssangyong,is875,curated,build}.py
artifacts/chunks.jsonl embeddings.npy index_meta.json   (generated)
finetune/ (empty package, planned)   eval/compare_chunking.py (before/after check for the chunking fix)
```

## 10. Current status

| Phase | State |
|---|---|
| 0. Planning, scaffold, env | ✅ |
| 1. Ingestion (448 chunks: cpwd 201, ssangyong 119, is875 128 = 92 OCR + 36 curated) | ✅ built and checked |
| 1a. Token-based chunking fix (≤512 tokens guaranteed; Table 2 / Appendix A split) + sibling expansion + `eval/compare_chunking.py` | 🟡 **done and measured by Claude; awaiting user test + commit** |
| 2. Retrieval (hybrid + amendment expansion) | ✅ built and smoke-tested |
| 3. Generation (prompt, Ollama, citation check, pipeline) | 🟡 **code written, never executed** |
| 4. CLI `src/ask.py` (+ `--debug`, `--no-llm`, `--mode`) and Streamlit `src/ui.py` | ⏳ |
| 5. Gold set (`eval/gold.jsonl`, user-verified) + `eval/run.py` (Recall@1/5/10, MRR, per doc/type; generation: citation correctness, numeric exact match, abstention) + baselines | ⏳ |
| 6. Fine-tuning (`finetune/make_pairs.py`, `finetune/train.py`), both protocols, ablation table, push to HF Hub | ⏳ |
| 7. README (≤5-command run), DECISIONS.md, clean-machine test | ⏳ |

The index is **up to date** with the token-chunking fix: Claude rebuilt `chunks.jsonl` (448) and `embeddings.npy` on 2026-10-08. They are **not committed yet**.

### How to test what exists (venv active)
```powershell
python -m src.ingest.build      # prints max tokens per doc + chunk ids added/removed vs the previous build
python -m src.index
python -c "from src.retrieve import Retriever; r=Retriever.load(); [print(h.chunk.citation, h.ranks, h.attached_reason) for h in r.search('force coefficient single frame solidity ratio 0.2 flat-sided').hits]"
python -m eval.compare_chunking   # before/after vs commit 74f8585 (use --old-ref <commit> to compare with another build)
```

## 11. Next steps (in order)
1. **Phase 3–4 (next):** run the generation path against Ollama for the first time; fix prompt/citation issues. Add the `src/ask.py` CLI (answer + numbered sources + warnings; `--debug` prints the BM25/dense/RRF ranks; `--no-llm`; `--mode`). **Stop for the user to test.**
1a. ~~Token-overflow fix~~ and ~~sibling expansion~~ done (§5, §8). The user tests and commits them.
2. Streamlit UI: question box, answer, source cards (doc, clause/¶, both pages, snippet, "attached because…", curated/amendment badges), and a corpus-scope banner ("IS 875-3:1987 with Amd 1–3").
3. **Amendment test plan (agreed):** (a) a deterministic check that each original provision in the results brings its amendment along; (b) amendment-sensitive gold questions, scored on the amended value: Table 28 φ=0.2 → **1.8**; Cl. 6.2.2.8 open cylinder → **−0.8 / −0.5**; Cl. 5.5 → **refer to IS 15498:2004** (not ×1.15); Cl. 7.1 → **"satisfies… or"**; Table 1 third class → **+ "permanent walls"** (Amd 3); (c) an ablation `expand=False` vs `expand=True`. If the 3B model ignores the amendment notes: strengthen the prompt first, then consider annotating the data, and document it honestly.
4. Draft the ~36 gold questions with gold `chunk_id`s and pages, for the user to verify; then build `eval/run.py`, the baselines (BM25 / dense / hybrid), and calibrate the low-confidence threshold.
5. Fine-tuning: synthetic pairs + hard negatives → train on the GTX 1650 → both protocols → ablation → HF Hub.
6. README, DECISIONS.md (decisions, trade-offs, limitations, scale path), clean-machine test.

**Last commit:** `74f8585` "Parse the 3 PDFs into cited sections, add manually checked IS 875 tables and amendments, and build hybrid keyword + semantic search" (Phases 0–2). The generation code (`generate.py`, `pipeline.py`, `cite.py`) is in that commit but hasn't been run yet.

### Starting a new chat
Give the new chat this file and `PLAN.md`. Pick up at **§11 step 1 (Phase 3–4)**. Remember the working agreements in §2: one phase at a time, the user runs installs and commits, and explanations should be plain and unhurried.
