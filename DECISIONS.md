# Decisions, assumptions and limits

This file explains **why** the project is built the way it is: what was chosen, what was rejected and why, what was assumed and how each assumption was checked, and where the system is expected to fail.

For setup and how to ask a question, see [README.md](README.md). For the original plan, see [PLAN.md](PLAN.md). The full evaluation reports are in [`eval/results/`](eval/results/).

---

## In one minute

- **What it is:** a question-answering system over three Indian construction documents (a contract, a court judgment and an engineering standard) that answers **only from those documents** and **cites the clause and page** for every claim.
- **The fine-tuning part:** the model that powers "meaning search" was fine-tuned on these documents. It finds the right passage more often (in the top 5 for 23 of 26 test questions, up from 20 with meaning search alone).
- **The most important design choice:** **official amendments are attached in code.** If the system finds a table that was later amended, the amendment comes with it, so the answer gives the corrected value, not the outdated printed one.
- **The honest headline:** search is good (the evidence reaches the answer-writing model for 24 of 26 questions). The weak link is the small, free, local answer-writing model (3 billion parameters), which gets **22 of 33** questions right. Most wrong answers happen when the evidence was already in front of it. A bigger model is the obvious next step, and switching takes one setting.
- **Everything is free and runs on a laptop.** No API keys, no paid services, no GPU needed to use it.

---

## A few terms used below

| Term | Plain meaning |
|---|---|
| **Chunk** | One passage of a document (a clause, a few paragraphs or one table), the unit the system searches. There are 448. |
| **Keyword search (BM25)** | Finds passages that contain the same words as the question. Very good for exact names like "Clause 10CC" or "Table 28". |
| **Meaning search (dense / embeddings)** | A small model turns every passage and the question into a list of numbers (an "embedding") so that passages with similar meaning are close together. Good for questions in everyday words. |
| **Hybrid search** | Run both, then merge the two ranked lists (with "Reciprocal Rank Fusion", which just combines positions in the two lists). |
| **LLM** | The language model that writes the answer from the passages (here: qwen2.5 3B, run locally with Ollama). |
| **Gold set** | 33 test questions with known correct answers, written and checked by hand against the PDFs. |
| **Hit@5** | "Was a correct passage in the top 5 search results?" Reported as a count, e.g. 23/26. |
| **MRR** | A 0–1 score for how high the first correct passage ranks (1.0 = always first). |

---

## 1. The documents

**Decision:** three documents of three different kinds:

| Document | Kind | Why it's interesting |
|---|---|---|
| CPWD General Conditions of Contract 2020 | Contract | Clause structure, page rulers and broken Hindi font text mixed into the English |
| Ssangyong v NHAI (Supreme Court, 2019) | Judgment | Long numbered paragraphs, many quotes from other judgments |
| IS 875 (Part 3):1987 with Amendments 1–3 | Standard | A **scan** with poor text recognition, unreadable tables, and amendments that change values |

**Why:** each one has a different reading problem, so the system has to handle real variety instead of one clean format. They also connect: the CPWD price-escalation clauses relate to the price-adjustment dispute in Ssangyong, which allows questions that need two documents.

**Scope stated plainly:** only the 1987 edition of IS 875 Part 3 with its three amendments. A newer edition exists but isn't in the documents, so the system doesn't describe it, and the README says so.

### Why the Hindi text was skipped

The CPWD contract is partly bilingual: its index and its register forms (PDF pages 80–95) are mostly Hindi, and some headings are in both languages. The Hindi is stored in an old "legacy" font (Kruti Dev), so when the text is pulled out of the PDF it comes out as **meaningless Latin letters** (for example "ds- yks- fu- fo-"), not real Hindi. Reading it properly would need a font-conversion step or OCR, plus a search model and an answer model that handle Hindi well. The English text already covers the same content, so:

- the Hindi-only pages are skipped,
- Hindi fragments inside English pages are detected and removed (`looks_like_legacy_hindi` in `src/ingest/text.py`),
- the system is English-only, and this is stated as a limit.

---

## 2. Reading the PDFs and cutting them into passages

**Decision:** one reader per document, which uses the **position** of text on the page (not just the text), and cuts by the document's own structure: clauses for the contract, numbered paragraphs for the judgment, clauses and tables for the standard.

**Why:** a citation like "Clause 10CC, p. 28" is only useful if a chunk actually is Clause 10CC. Cutting every N words would split clauses in the middle and mix two clauses into one chunk. Position matters because, for example, the contract prints clause titles in the left margin, and the standard is printed in two columns.

**Every citation shows two page numbers:** the printed page and the PDF page. They differ in all three documents (by 0, 2 and 4 pages), and a reviewer checking a citation needs the PDF page to find it.

**Size limit, measured in tokens, not words.** The meaning-search model only reads the first **512 tokens** of a passage and **silently ignores the rest**. Originally chunks were limited to 300 *words*, but numbers and scanned text cost up to 3 tokens per word, so 11 chunks were too long and their endings were invisible to meaning search. **Fix:** chunks are limited to 400 tokens (counted with the model's own tokenizer), and the build **fails** if any chunk is over 512. Long tables are split into parts, each repeating the table's title and column headings.

---

## 3. Hand-typed tables and amendments

**Decision:** the four most important IS 875 tables (Table 1, Table 2, Table 28 and Appendix A, the city wind speeds) and **all 27 amendment items** were typed in by hand (`data/curated/is875.yaml`) and checked against the page images.

**Why:** the scanned tables come out as jumbled numbers, and one amendment page has no text at all. Without hand-typing, the system couldn't answer the most basic wind-load questions.

**Rules kept strictly:**
- **Typed exactly as printed.** Table 28 shows **1.0** for one value, which Amendment 2 later changed to **1.8**. The table keeps 1.0. The correction lives only in the amendment, so the amendment handling (next section) is what produces the right answer. This makes the citations honest: what the system quotes is what the page shows.
- **Anything added by the curator is kept separate.** The 1987 table uses old city names (Madras, Bombay). Modern names come from a separate file (`data/curated/place_aliases.yaml`) and are added at build time in square brackets: "Madras [now Chennai] 50". This way "wind speed in Chennai" works, and it's clear which words are not from the PDF.
- Explanations added to amendment items are marked "[Curator's explanation: ...]" and kept apart from the printed wording.

**Trade-off:** hand-typing doesn't scale. With hundreds of documents, a layout-aware OCR model would replace it (see section 11).

---

## 4. Search: keywords + meaning, combined

**Decision:** run keyword search (BM25) and meaning search (the embedding model) for every question, merge the two lists, and send the **top 6** passages to the answer-writing model.

**Why both:** these documents are full of exact identifiers ("Clause 10CC", "Section 34", "Table 28") that keyword search handles well and meaning search handles poorly. But people also ask in their own words ("what happens if the contractor is late?"), which needs meaning search. Measured on the gold set, they fail on **different** questions: hybrid fixed 7 questions that keyword search missed and 7 that meaning search missed.

**Why this way of merging (Reciprocal Rank Fusion):** it only uses each passage's *position* in each list, so the two very different kinds of scores never need to be put on the same scale.

**Rejected or postponed:**
- **A vector database** (Qdrant, pgvector): 448 passages fit in a small in-memory table. The search code sits behind one `Retriever` class, so swapping in a database is a one-file change when needed.
- **A reranker** (a second model that re-sorts the results): optional in the plan, cut first when time ran short. The results are good enough without it.

### Three things added on top of the plain top 6

These are attached **in code**, so they don't depend on the model noticing anything.

1. **Amendments come with what they amend.** If a passage that an amendment changes is retrieved, the amendment is attached, and the amendment text is placed **inside** that passage's block in the prompt ("AMENDED by [S2]: ..."). An earlier version put the amendment in a separate block with a note saying it overrides the original, and the small model **still answered 1.0 instead of 1.8**. Placing it inline fixed that.
2. **Split tables are sent whole.** If one part of a split table is found, the other parts are attached. A question like "k2 at 225 m" needs rows from two different parts to interpolate.
3. **Rare identifiers are pinned.** If the question names something that appears in 3 or fewer passages (for example "Article 142" appears in exactly one), those passages are always included. Without this, "What relief did the Supreme Court grant under Article 142?" answered "Not found" because meaning search ranked the right paragraph 63rd.

**Measured effect** (does all the needed evidence reach the model?): plain top 6 = **17/26**; with all three additions = **22/26**. Amendments matter most (without them: 19/26).

---

## 5. Writing the answer

**Decision:** a free local model, **qwen2.5 3B, run with Ollama**. It is told to use only the given passages, cite every claim with [S1], [S2] ... markers, and otherwise reply exactly "Not found in the provided documents."

**Why:**
- **Free and private:** nothing leaves the computer, and the reviewer needs no API key.
- **Runs on an ordinary laptop** (about 20 seconds per answer on a CPU).
- **Swappable:** the model name is one setting (`OLLAMA_MODEL`), so a bigger model can be tried without code changes.

**If Ollama isn't installed or running, the app never crashes.** It shows the ranked, cited passages instead, with a note saying why.

**Answers are repeatable:** temperature 0 and a fixed seed. At a slightly higher temperature, the same question sometimes flipped between a right answer and "not found", which makes demos and evaluations unreliable.

**Rejected: a paid API (OpenAI, Anthropic, Groq).** It would give better answers, but it breaks the "free and easy to run" requirement, and the reviewer would need a key.

---

## 6. Checking the answer

**Decision:** after the answer is written, simple code checks it and shows **warnings** (it doesn't block the answer):

- a citation marker points to a source that doesn't exist,
- a sentence has no citation,
- **a number in the answer doesn't appear in the source it cites** (wrong numbers are the most dangerous mistake in engineering and contract questions),
- search confidence is low (the best meaning-search match is weak), so the documents may not cover the question.

**Why warnings and not blocking:** the checks are mechanical and sometimes over-cautious. A user who sees the warning and the sources can judge for themselves.

**Limit:** the checks can't tell whether a sentence's *meaning* is supported. A wrong claim that happens to use numbers from the cited passage passes (see section 9).

---

## 7. The fine-tuning element

**Decision:** fine-tune the meaning-search model (`BAAI/bge-small-en-v1.5`, 33 million parameters) on pairs of questions and passages from these documents.

**Why this and not something else:**
- **vs. fine-tuning the answer-writing model (LoRA):** search sets the ceiling, because the model can't cite a passage it never received. Search improvement can also be measured objectively (did the right passage rank higher?). Fine-tuning the answer writer is hard to measure, and it risks confident answers that aren't grounded in the documents.
- **vs. a classifier** (for example "which document is this question about?"): with only three documents, keyword search already routes most questions correctly, so it would barely change anything.

**How the training data was made:**
1. The local qwen model wrote **2 questions for each of the 448 passages** (896 questions).
2. Filters removed bad ones: near-duplicates, questions that copy a 6-word phrase from the passage (too easy, it would teach word matching), questions saying "the passage", questions whose passage can't be found at all (probably a bad question), and **questions too similar to any gold test question** (so the test isn't leaked into training). **683 kept.**
3. Each question also got one **"hard negative"**: a passage that looks similar but is wrong (from keyword search's top 10, never from the same clause or table). This teaches the model to separate near-misses like Clause 10CA vs 10CC.
4. A small trial of 30 passages was read by hand **before** the full run, to check the questions were realistic. They were (82% passed the filters), so no stronger question-writer was needed.

**Training:** standard `sentence-transformers` code, 3 rounds, about **6 minutes** on a laptop GPU (GTX 1650, 4 GB). 10% of the questions were held back to pick the best round.

**The success rule was fixed before training:** the fine-tuned model ships only if, on the hand-checked gold questions, it fixes more questions than it breaks (top 5 of hybrid search) **and** breaks no amendment question. Fixing the rule first prevents "trying until it wins".

**Result:** fixed 4, broke 2 (both still in the top 5, just one place lower), no amendment question broken → **it ships.**

| Search | Hit@5 before | Hit@5 after | MRR before | MRR after |
|---|---|---|---|---|
| Meaning search only | 20/26 | **23/26** | 0.590 | **0.645** |
| Hybrid (what the app uses) | 22/26 | **23/26** | 0.622 | **0.670** |

The gain is mostly on questions asked in everyday words (13/15, up from 12/15), which is where it was expected.

**How it's delivered:** the model is published on Hugging Face ([`shawnriju/bge-small-construction-rag`](https://huggingface.co/shawnriju/bge-small-construction-rag)) and downloads automatically on the first question. It's too big for GitHub (~130 MB, over GitHub's 100 MB file limit); Git LFS was rejected because of its bandwidth quota.

**No silent fallback to the original model.** The stored search index was built with the fine-tuned model, and embeddings only make sense with the model that made them. If the download fails, the app stops with a clear message instead of quietly giving worse results.

**The honest caveat: this is "corpus adaptation", not "generalisation".** The model was trained on questions about *these* documents and tested on different (hand-written) questions about the *same* documents. That matches the task, a question-answering bot over these specific documents, but it doesn't show the model would help on new documents. A strict test (hold back whole sections from training) was left as future work: with only ~30 test questions it would leave about 6 to measure on, too few to mean anything.

---

## 8. How it was tested

### The gold set: 33 hand-checked questions

- 7 contract, 7 judgment, 10 standard, 2 that need two documents, and **7 that the documents can't answer** (to check the system says "Not found" instead of inventing).
- 5 depend on an amendment (e.g. Table 28 must give 1.8, not 1.0).
- Mixed wording: 15 in everyday words, 11 in the document's own words, reported separately.
- Every expected answer was **checked by hand against the PDF page**, using a checking sheet with the page and an exact quote (`eval/gold_review.md`).
- **Frozen before any fine-tuning**, and never used for training.
- Each expected passage is stored three ways (passage id, page, exact quote), and an automated test fails if a rebuild ever breaks one.

### Search and answers are measured separately

- **Search:** did the right passage rank high (Hit@1/5/10, MRR), and did *all* needed evidence reach the model ("fully supported", e.g. Table 28 **and** its amendment)?
- **Answers:** does the answer state the key values (checked automatically, including minus signs), does it say "Not found" when it should, does it cite the right passage? The 6 questions without simple key values were read by hand.
- **Every wrong answer is labelled** as a *search* failure (the evidence never reached the model) or a *writing* failure (the evidence was there, the model got it wrong). This is what shows where the weak link is.

### Rules for small numbers

With about 30 questions, one question is about 3 percentage points. So:
- counts are shown next to every percentage ("23/26", not just "88%"),
- two systems are compared **question by question** ("fixed 4, broke 2"), not by a percentage gap,
- no "improves" claim is made from a 1–2 question difference.

### Automated tests

128 tests (`python -m pytest`, about 10 seconds, no internet or Ollama needed). They cover the citation checks, amendment and table attachment, the search tokenizer, the fallback when Ollama is down, the command-line output, the evaluation scoring and the training data helpers. They also check the committed data itself: the index matches the passages, no passage is over 512 tokens, every amendment's target exists, and the gold set still matches the passages.

### Results

| | Original search model | Fine-tuned search model |
|---|---|---|
| Answerable questions answered right | 15/26 | **16/26** |
| Unanswerable questions correctly "Not found" | **7/7** | 6/7 |
| **Total** | **22/33** | **22/33** |
| Answer cites a correct passage | 16/26 | **19/26** |

With the fine-tuned model, **8 of the 10 wrong answers had the right evidence in front of the model**; only 2 were search failures. The fine-tune improves ranking and citations, but it doesn't change which passages reach the model on this test set, so the final score stays the same. This is the expected result, reported as it is.

---

## 9. Expected failure modes

Where the system is known to go wrong, with a real example of each from testing:

| What goes wrong | Example | Why | Caught? |
|---|---|---|---|
| **Reading the wrong cell of a wide table** | "k2 at 20 m, category 2, class A" → 1.10 instead of 1.07 | The 3B model struggles to find one cell in a 12-column table, even when the whole table is in front of it | No. Search was correct; this is a writing failure |
| **Boundary conditions** | Open cylinder with h/D = 0.3 → −0.5 instead of −0.8 (the amendment says −0.8 when h/D is "not less than 0.3") | The small model misreads "not less than" | No |
| **Saying the opposite of the text** | Ssangyong: says "patent illegality *is* a ground" for international arbitration, though the retrieved paragraph says it isn't | Small-model reasoning error on legal negation | No. The most serious error found |
| **Mixing up two documents** | Cross-document question: gives Ssangyong's formula as CPWD's | Too many similar passages for a small model | No |
| **"Not found" when the answer is there** | 3 answerable gold questions; also "basic wind speed in Bangalore?" (a tester's question): the city list with "Bangalore [now Bengaluru] 33" is source S1, yet the answer is "Not found", while Calicut/Kozhikode works | The model is cautious, and picking one entry out of a long list of 38 cities is unreliable for a 3B model (like the wide tables). Answers are repeatable, so the same wording fails every time | Partly: the right source is shown, but no warning fires, because a "Not found" has no numbers to check |
| **Inventing an answer to an out-of-scope question** | "Seismic zone factor for Delhi?" → "1.0" (from an unrelated topography clause) | The number exists in the cited passage, so the number check can't catch it | **Yes: the low-confidence warning fires.** It isn't turned into a hard "Not found" because that would also block a correct answer found by pinning |
| **Different words than the document** | "What did the court finally decide?" vs the text's "set aside / uphold" | Vocabulary mismatch; the fine-tune moved the right paragraph from 38th to 24th, still outside the top 6 | No; it's the clearest case for further fine-tuning |
| **Formula unreadable in the scan** | Design wind pressure "pz = 0.6 Vz²" is garbled in the OCR | The model fills in the formula from its own knowledge | Partly: the number check flags the citation |
| **Citing too much or too little** | One citation group at the end of a paragraph | Small-model habit | Yes: "uncited sentence" warnings |

Most of these are writing failures by the small answer model. They're kept visible in the reports rather than patched one by one with prompt tweaks.

---

## 10. Other known limits

- **Scanned standard:** on a few pages the scan's text mixes the two columns, so some sentences are out of order. Only the four hand-typed tables are reliable; other IS 875 tables are noisy scanned text.
- **Lost table headings:** the scan lost the headings of some tables. For the five that amendments target, the code maps each one to the chunk that absorbed its text, so its amendment still attaches.
- **Contract formulas** (price escalation 10CA/10CC) don't extract cleanly from the PDF.
- **Long judgment paragraphs** that quote other cases are split into several "part 1/9, 2/9 ..." chunks.
- **The low-confidence cutoff (0.51)** was set from only 7 unanswerable questions: it flags 6 of 7 of them and 1 of 26 answerable ones. It's a hint, not a rule.
- **English only** (see section 1).
- **Cities not listed in Appendix A** can only be read off the wind-speed map, which is an image and isn't searched, so the right answer for them is "Not found".

---

## 11. Packaging: making it easy for a reviewer to run

| Decision | Why | Trade-off / what was rejected |
|---|---|---|
| **The prebuilt passages and search index are committed** to the repo (with the three PDFs, ~13.5 MB) | The reviewer can ask a question right after `pip install`, without re-reading the PDFs (which takes minutes) | A bigger repo. The index must be rebuilt whenever the passages change; a test checks they match |
| **The app refuses an out-of-date index.** The index stores a fingerprint of the passages and the name of the model that built it | If the passages change but the index isn't rebuilt, search results would be silently wrong. A clear error ("run `python -m src.index`") is better | None worth mentioning |
| **Keyword index rebuilt at start-up**, not saved to disk | It takes milliseconds, and saving it would need Python's `pickle` format, which can run code when loaded | Slightly slower start (not noticeable) |
| **CPU-only PyTorch for the reviewer** (`requirements.txt`) | A much smaller download, and it works on any laptop | Answers and search run a bit slower than on a GPU |
| **Training in a separate environment** (`.venv-train`, `requirements-train.txt`, CUDA PyTorch) | The tested runtime environment is never touched by the large GPU packages. Found along the way: a plain `torch==2.4.1` pin would have been "already satisfied" by the CPU build, so training would have silently run on the CPU. The pin is `2.4.1+cu121` | Two environments to maintain, but the reviewer only needs one |
| **Command line first, a simple web page (Streamlit) last** | The command line answers "how do I ask a question?" and is easiest to test. The web page is for the demo and was built after evaluation and fine-tuning, so it couldn't eat into them | The web page is basic: it shows the same answer, warnings and sources as the command line, with no extra polish. Testers found the sources could be easier to read; noted as a next step |
| **No Docker** | A PyTorch image is several GB, and Ollama inside Docker on Windows is awkward. A plain virtual environment is simpler | The reviewer installs Python packages themselves (the README walks through it on Windows, macOS and Linux) |
| **Settings in one file** (`src/config.py`), changeable with environment variables or a `.env` file | No hidden "magic numbers" in the code; the model names can be swapped without code changes | None worth mentioning |

---

## 12. How the plan changed along the way

The plan was reviewed twice before and during the build. What was taken on board, and what wasn't:

**External review of the plan (another AI model, before building).** Its main point: the plan was strong, but too big for "about one day".
- **Taken on:** about 30 gold questions with 6–8 unanswerable ones; counts and per-question wins/losses instead of "improves" claims; the guard against test questions leaking into training; harder filters on training questions; the strict hold-out test moved to future work; the web page built last; cross-document questions first to cut; this "expected failure modes" section; search and answers reported separately.
- **Staged instead of decided up front:** using a stronger (free-tier cloud) model to write the training questions. The rule was: try the local qwen model first, check a small trial by hand, and switch only if it failed. It passed, so the switch was never needed.
- **Rejected:** committing the model with Git LFS as a backup (bandwidth quota; the model is over GitHub's file limit).

**Project review before the evaluation phase.**
- **No fallback to the original search model** (see section 7): simpler and always correct.
- **Gold questions tagged by wording** (everyday vs the document's own words), decided before training so the tagging couldn't favour the fine-tune.
- **Two search scores** ("found" and "fully supported"), because an amendment question needs two passages.
- **Gold answers locked to their passages** (id, page and exact quote, checked by a test), and **chunking frozen** once the gold set was frozen.
- **2 training questions per passage instead of 3–4:** the run time depends on the number of passages (one model call each), and 2 gave enough data.

**Review of the gold set.** The gold questions were drafted by Claude (the AI assistant used to build this), then checked by hand against the PDFs and reviewed again by another AI model (Gemini). Both of its findings were verified before acting on them, and both were real:
- one contract answer was incomplete: a second clause (Clause 24, a 5-year obligation for structural defects) also applies, so that question now needs both clauses;
- several evidence pages pointed to a passage's whole page range instead of the exact page. 20 were corrected, more than the review had listed.

**Order of scope cuts, agreed early:** reranker first, then cross-document questions, then web-page polish. **Never cut:** the evaluation, the amendment handling, and a test on a clean machine. In the end the reranker was cut and the web page was kept basic (no polish).

---

## 13. Smaller decisions and trade-offs

Each of these came from a problem found while building or testing.

| Area | Decision | Why / what it fixed | Trade-off |
|---|---|---|---|
| Chunking | The token-based size fix was done **before** the gold set was written | The fix renamed many passage ids, and gold questions point to passage ids. Doing it later would have broken them | 87 old passage ids disappeared and 81 new ones appeared (fine at that point, nothing depended on them yet) |
| Keyword search | The tokenizer keeps clause ids whole ("10 CC" → `10cc`, `6.2.2.8`), but never glues short English words onto numbers | A bug glued "Section 34 **of**" into `34of`, so keyword search missed most "Section 34" passages. Fixed: one test question went from 38th to 5th | None |
| Split tables | Table parts are attached with **their own budget** (up to 3), separate from amendments (up to 4) | So table parts never push amendments out of the prompt | About 1 extra passage per question on average |
| City names | Modern names placed **inline** next to the old one ("Madras [now Chennai] 50"), not in a note at the end of the table | With a note at the end, the model answered "Chennai" correctly only 8 times out of 10; inline, 10 out of 10 | Added text that isn't in the PDF (marked as curator text, kept in a separate file). A search-time alias list would scale better |
| Prompt | The question is asked **before and after** the sources, and each rule has a one-line example | Small models lose track of the question after a long block of sources | A slightly longer prompt |
| Prompt | Mention an amendment **only if it changes the value used in the answer** | The model started saying "this formula is amended" about any amended passage, even unrelated ones | The answer no longer says "amended from 1.0"; the sources list still shows "amended by [S2]" |
| Answer clean-up | Formatting codes (LaTeX) are stripped **in code** | The model ignored the "plain text only" instruction | None |
| Garbled formula | The unreadable wind-pressure formula was **not** hand-typed | Keeping the hand-typed content small and focused; the number check flags the weak citation | The model fills in the formula from its own knowledge (correct, but not from the document) |
| Wide tables | Not fixed (row-by-row rendering left as a next step) | Good enough for a demo; the model says "Not found" or picks a nearby cell, and the evaluation shows how often | 2 table questions answered wrong |
| Low-confidence cutoff | Kept at the old value until after fine-tuning, then set once from the shipped model (0.55 → 0.51) | Fine-tuning changes the similarity scale, so the cutoff has to be recalibrated for the model that ships | Calibrated on only 7 unanswerable questions |
| Low-confidence cutoff | A **warning**, not a hard "Not found" | A hard rule would also block a correct answer that was found by identifier pinning | One invented answer gets through, with the warning shown |
| Fine-tune training | Large batches (32) processed 8 at a time ("cached" loss), and a batch never contains the same passage twice | Fits 512-token passages into a 4 GB laptop GPU; a passage with two questions is never treated as "wrong" for itself | A bit slower training (6 minutes in total) |
| Fine-tune training | Hard negatives never come from the same clause family, the same table, or an amendment link | Otherwise the model would learn "part 2 of Clause 10CC is unrelated to a Clause 10CC question", which is false | Fewer, but cleaner, negatives |
| Fine-tune training | The best of 3 rounds is kept, judged on the 10% held-back questions | Guards against over-training | None (all 3 rounds scored about the same) |

---

## 14. What I'd do next, and how it would scale

**With more time, in order of expected value:**
1. **A bigger answer-writing model** (for example a 7B model through Ollama, or an API). Most wrong answers are writing failures, so this is the biggest lever. It needs no code change, only `OLLAMA_MODEL`, and the same evaluation can be rerun.
2. **Write tables out row by row** for the model ("Height 20 m: Category 2, Class A = 1.07; ..."). This likely fixes the wide-table failures (untested).
3. **A reranker** and more fine-tuning data aimed at "outcome" questions (the vocabulary mismatch).
4. **A strict hold-out test** for the fine-tune: hold back whole sections from training, with a larger gold set so the numbers mean something.
5. **Clearer source display** (tester feedback): for example a short highlighted snippet of the cited sentence, and grouping the parts of one clause together.

**At 100+ documents:**
- a vector database (Qdrant or pgvector) behind the existing `Retriever` class,
- a layout-aware OCR or vision model for scanned tables instead of hand-typing,
- **document versioning**, so amendments and new editions replace old text in a managed way (today's amendment links are hand-curated),
- an ingestion queue that processes new documents in the background.

**Deliberately left out** (not needed for three documents and a demo): an OCR pipeline, a vector database server, agents, query rewriting, fine-tuning the answer model, user accounts, and Docker. Docker was rejected because a PyTorch image is several GB, and running Ollama inside Docker on Windows adds friction; a plain virtual environment is easier for a reviewer.
