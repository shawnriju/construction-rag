# Construction RAG

Ask questions about three Indian construction documents and get answers **with citations** (document, clause and page).

It also includes a **fine-tuned search model**: a small embedding model trained on these documents so it finds the right passage more often.

| Document | Type | What it covers |
|---|---|---|
| CPWD General Conditions of Contract 2020 | Contract | Delay compensation, escalation, defects liability, arbitration, ... |
| Ssangyong v NHAI (Supreme Court, 2019) | Court judgment | Setting aside an arbitral award, "patent illegality", Article 142 |
| IS 875 (Part 3):1987 with Amendments 1–3 | Engineering standard | Wind loads: wind speeds, k1/k2/k3 factors, force coefficients |

> **Note:** only the 1987 edition of IS 875 Part 3 (with its Amendments 1–3) is included. Newer editions are not in the documents, so the answers don't cover them.

Works on **Windows, macOS and Linux**. Every step below shows the command for each system. Where the command is the same everywhere, it says so.

---

## Before you start

You need:

1. **Python 3.10, 3.11 or 3.12.** (Tested on 3.12. Python 3.13 does **not** work yet, because the pinned PyTorch version has no 3.13 build.)
2. **Git**, to download the code.
3. **An internet connection the first time you ask a question.** The fine-tuned search model (~130 MB) downloads automatically from Hugging Face, once.
4. **About 2 GB of free disk space** (plus ~2 GB if you install the optional answer-writing model).
5. **Optional: [Ollama](https://ollama.com/download)**, a free program that runs the AI model which writes the answers on your own computer.
   **Without Ollama the project still works:** you get the most relevant passages with their citations, just no written answer.

No API keys, no accounts, no GPU needed.

### Check what you already have

Open a terminal:
- **Windows:** press the Start key, type **PowerShell**, and open it.
- **macOS:** open **Terminal** (Applications → Utilities → Terminal, or press Cmd+Space and type "Terminal").
- **Linux:** open your **Terminal** app (on Ubuntu: Ctrl+Alt+T).

Then check Python and Git:

**Windows (PowerShell):**
```
python --version
git --version
```

**macOS / Linux:**
```
python3 --version
git --version
```

If Python shows 3.10, 3.11 or 3.12 and Git shows any version, skip to [Setup](#setup-one-time-about-510-minutes).

### Install Python or Git if they're missing

**Windows:**
- Python: download **Python 3.12** from https://www.python.org/downloads/windows/ and run the installer. **Tick "Add python.exe to PATH"** on the first screen.
- Git: download it from https://git-scm.com/download/win and install with the default options.
- Then close PowerShell and open it again, so it finds the new programs.

**macOS:**
- Python: download **Python 3.12** from https://www.python.org/downloads/macos/ and run the installer (or, if you use Homebrew: `brew install python@3.12`).
- Git: run `xcode-select --install` and click **Install** (or, with Homebrew: `brew install git`).

**Linux (Ubuntu / Debian):**
```
sudo apt update
sudo apt install python3 python3-venv python3-pip git
```
On other Linux versions, install `python3`, its `venv` module and `git` with your package manager (for example `sudo dnf install python3 git` on Fedora).

---

## Setup (one time, about 5–10 minutes)

Run these steps one by one in the terminal.

### Step 1: Download the code

**Windows, macOS and Linux** (same commands):
```
git clone https://github.com/shawnriju/construction-rag.git
cd construction-rag
```

### Step 2: Create a virtual environment and turn it on

A virtual environment keeps this project's packages separate from everything else on your computer.

**Windows (PowerShell):**
```
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```
python3 -m venv .venv
source .venv/bin/activate
```

You'll see `(.venv)` at the start of your terminal line when it's on.
From now on, `python` works on every system (no need for `python3`) as long as the virtual environment is on.
(If you see an error here, see [Troubleshooting](#troubleshooting).)

### Step 3: Install the packages

**Windows, macOS and Linux** (same command):
```
pip install -r requirements.txt
```

This takes a few minutes. It installs a small CPU-only version of PyTorch, so no graphics card is needed.

### Step 4 (optional): Install the answer-writing model

Skip this step if you only want to see the search results.

**First, install Ollama:**

- **Windows:** download the installer from https://ollama.com/download and run it. Ollama then runs in the background (you'll see its icon near the clock).
- **macOS:** download the app from https://ollama.com/download, move it to Applications, and open it once. It then runs in the background (you'll see its icon in the menu bar).
- **Linux:** run
  ```
  curl -fsSL https://ollama.com/install.sh | sh
  ```
  This installs Ollama and starts it as a background service.

**Then download the model** (~2 GB, one time).

**Windows, macOS and Linux** (same command):
```
ollama pull qwen2.5:3b-instruct
```

That's it, you're set up.

---

## Ask a question

Make sure the virtual environment is on (you see `(.venv)`), then:

**Windows, macOS and Linux** (same command):
```
python -m src.ask "What is the basic wind speed in Chennai?"
```

The first run takes a little longer because it downloads the search model. After that, an answer takes about 20 seconds on a normal laptop CPU.

**Example output:**

```
Question: What is the force coefficient for a single frame with solidity ratio 0.2 and flat-sided members?

Answer (23.1s):
  The force coefficient for a single frame with solidity ratio 0.2 and flat-sided members is 1.8
  [S1][S2].

Sources:  (* = cited in the answer)
 *[S1] IS 875 (Part 3):1987, Amendment No. 2, item 10, PDF p. 65-66 - Amendment No. 2 (March 2002) -
       amends Table 28  [AMENDMENT, curated]
 *[S2] IS 875 (Part 3):1987, Table 28, p. 46 (PDF 50) - Force coefficients for single frames
       [curated]
        amended by [S1]
  [S3] IS 875 (Part 3):1987, Cl. 6.3.3.3-6.3.3.4, p. 46 (PDF 50) - Singleframes
  ...
```

How to read it:
- `[S1]`, `[S2]` in the answer point to the numbered sources below it. A `*` marks the sources the answer actually cites.
- Each source shows the **printed page** of the document and the **PDF page** (they differ), so you can open the PDF in `data/pdfs/` and check.
- `[AMENDMENT]` means the passage is an official amendment. Here, Table 28 prints **1.0**, but Amendment 2 changed it to **1.8**, and the answer correctly gives 1.8.
- `[curated]` means the content was typed in by hand from the PDF, because the scanned table couldn't be read reliably.
- Any warnings (for example "low retrieval confidence") appear between the answer and the sources. Take them seriously.

### More ways to ask

These commands are **the same on Windows, macOS and Linux**:

| What you want | Command |
|---|---|
| Ask several questions in a row (models load only once) | `python -m src.ask` then type questions; press Enter on an empty line (or Ctrl+C, on every system including macOS) to quit |
| Only the search results, no written answer | `python -m src.ask --no-llm "your question"` |
| See how each passage was ranked by keyword and meaning search | `python -m src.ask --debug "your question"` |
| Use only one kind of search | `python -m src.ask --mode bm25 "..."` (keywords) or `--mode dense` (meaning) |
| All options | `python -m src.ask --help` |

If Ollama isn't installed or running, you automatically get the search-results-only answer, with a note saying why.

### Coming back later

Each time you open a new terminal, go to the project folder and turn the virtual environment on again before asking questions:

**Windows (PowerShell):**
```
cd construction-rag
.venv\Scripts\activate
```

**macOS / Linux:**
```
cd construction-rag
source .venv/bin/activate
```

To turn it off when you're done, type `deactivate` (same on every system).

### Questions to try

- What is the force coefficient for a single frame with solidity ratio 0.2 and flat-sided members? *(answer: 1.8, from Amendment 2)*
- What is the basic wind speed in Chennai? *(50 m/s; the 1987 table calls it Madras)*
- What compensation is payable for delay under the CPWD contract?
- How long is the contractor liable for defects after the work is completed under the CPWD contract?
- What relief did the Supreme Court grant under Article 142 in Ssangyong?
- How should cyclonic wind velocity be accounted for in design? *(Amendment 3 points to IS 15498)*
- What is the penalty for late submission of a tender in the Delhi Metro contract? *(not in the documents, so it should say "Not found")*

---

## How it works (short version)

1. **Reading the PDFs.** Each document has its own reader, because each has different problems: the contract has page rulers and Hindi text from a broken font, the judgment has numbered paragraphs, and the standard is a scan with poor text recognition. They are split into 448 passages ("chunks"), each one clause or a few paragraphs long, with its page numbers.
2. **Hand-typed tables and amendments.** The scanned tables of IS 875 couldn't be read, so 4 key tables and all 27 amendment items were typed in by hand, exactly as printed (`data/curated/`).
3. **Searching.** Every question is searched two ways: **keyword search** (good for exact names like "Clause 10CC") and **meaning search** (good for questions in your own words). The two result lists are merged.
4. **Adding what's needed.** If a passage that an amendment changes is found, the amendment is attached automatically. If one part of a split table is found, the rest of the table is attached.
5. **Writing the answer.** A small local model (qwen2.5 3B via Ollama) writes the answer using only those passages, and must cite each claim or say "Not found in the provided documents."
6. **Checking the answer.** The code checks the citations, for example that every number in the answer really appears in the source it cites, and warns if not.

## The fine-tuning part

The meaning search uses a small model called `bge-small-en-v1.5`. It was trained on general web text and has never seen phrases like "Engineer-in-Charge" or "patent illegality". So it was **fine-tuned on these documents**:

- A local AI model wrote 896 practice questions about the passages. 683 survived quality filters (for example, questions that just copied the passage's wording were dropped).
- The model was trained to match each question to its passage and not to a similar-looking wrong one. This took about 6 minutes on a laptop GPU.
- It was then tested on **33 questions written and checked by hand**, which were never used in training.

The fine-tuned model is published at [huggingface.co/shawnriju/bge-small-construction-rag](https://huggingface.co/shawnriju/bge-small-construction-rag) and downloads automatically.

## Results

On the 26 hand-checked questions that have an answer (plus 7 that deliberately don't):

**Finding the right passage** (in the top 5 search results):

| Search model | Meaning search only | Keyword + meaning (what the app uses) |
|---|---|---|
| Original model | 20/26 | 22/26 |
| **Fine-tuned model** | **23/26** | **23/26** |

**Answering correctly** (whole system): **22/33**, both with the original and the fine-tuned model.

What that means, in plain words:
- The fine-tuning clearly helps search, most of all for questions asked in everyday words rather than the document's own words.
- It doesn't change the final score. The right passage already reached the answer-writing model in 24 of 26 questions either way. Most wrong answers come from that small 3B model misreading passages it did have (for example, picking the wrong cell in a wide table). A bigger model is the obvious next step.
- When a question isn't covered by the documents, the system almost always says "Not found" (6 of 7 times with the fine-tuned model) instead of making something up.

Full reports: [`eval/results/`](eval/results/). The reasoning behind each decision, and the known limits: [`DECISIONS.md`](DECISIONS.md). The original plan: [`PLAN.md`](PLAN.md).

---

## For developers

All commands in this section are **the same on Windows, macOS and Linux** (with the virtual environment on), except setting environment variables, which is shown for each system.

### Run the tests

```
pip install -r requirements-dev.txt
python -m pytest
```

128 tests, about 10 seconds. They need no internet and no Ollama.

### Re-run the evaluation

```
python -m eval.run retrieval --label finetuned     # search quality, about 1 minute
python -m eval.run generation --label finetuned    # answer quality, needs Ollama, about 11 minutes
python -m eval.gold --review                       # check the gold questions against the passages
```

Reports are written to `eval/results/`.

### Rebuild the passages and search index (not needed to use the app)

The ready-made index is already included. To rebuild it from the PDFs:

```
python -m src.ingest.build
python -m src.index
```

### Settings

| Variable | Default | What it does |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:3b-instruct` | Which Ollama model writes the answers (pull it first with `ollama pull`) |
| `OLLAMA_URL` | `http://localhost:11434` | Where Ollama is running |
| `LLM_BACKEND` | `ollama` | Set to `none` to always show search results only |

How to change one, for example to try a bigger answer-writing model:

**Windows (PowerShell)**, for the current terminal window:
```
$env:OLLAMA_MODEL = "qwen2.5:7b-instruct"
python -m src.ask "your question"
```

**macOS / Linux**, for the current terminal window:
```
export OLLAMA_MODEL=qwen2.5:7b-instruct
python -m src.ask "your question"
```

**Any system, permanently:** create a file named `.env` in the project folder containing the line:
```
OLLAMA_MODEL=qwen2.5:7b-instruct
```

The fine-tuning scripts (`finetune/`) need an NVIDIA GPU with CUDA and a separate environment (`requirements-train.txt`). You don't need them to use the app.

### Project layout

```
data/pdfs/        the three source PDFs
data/curated/     hand-typed IS 875 tables and amendments, plus today's city names
src/              the app: PDF reading (ingest/), search, answer writing, citation checks, CLI
artifacts/        the ready-made passages and search index
finetune/         practice-question generation and training scripts
eval/             the 33 hand-checked questions, scoring code, and results
tests/            automated tests
```

---

## Troubleshooting

**"running scripts is disabled on this system" (Windows, Step 2).**
Run this once, then try `.venv\Scripts\activate` again:
```
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**"ensurepip is not available" or "No module named venv" (Linux, Step 2).**
The venv module is a separate package on Ubuntu and Debian. Install it, delete the half-made folder, and repeat Step 2:
```
sudo apt install python3-venv
rm -rf .venv
```

**`python` or `python3` is not found.**
- Windows: reinstall Python and tick **"Add python.exe to PATH"**, then open a new PowerShell window. Or try `py` instead of `python`.
- macOS / Linux: use `python3` (not `python`) before the virtual environment is on. If it still isn't found, install Python as shown in [Before you start](#install-python-or-git-if-theyre-missing).

**`git` gives "xcrun: error: invalid active developer path" (macOS).**
Run `xcode-select --install`, click **Install**, then try again.

**`pip install` fails while installing torch.**
Your Python version is probably too new (3.13) or too old. It must be 3.10, 3.11 or 3.12. If you have several versions installed, make the virtual environment with a specific one. Delete the old `.venv` folder first, then:
- Windows: `py -3.12 -m venv .venv`
- macOS / Linux: `python3.12 -m venv .venv`

Then turn it on and run Step 3 again.

**"Could not load the embedding model ..." when you ask the first question.**
The search model couldn't download. Check your internet connection and try again. It only needs to download once.

**You see "LLM backend 'ollama' is not reachable; showing retrieved passages only."**
Ollama isn't running, or the model isn't downloaded.
- Windows / macOS: open the Ollama app (from the Start menu or Applications).
- Linux: start it with `ollama serve` in a second terminal (or `sudo systemctl start ollama`).
- Then, on any system, make sure the model is there with `ollama pull qwen2.5:3b-instruct`, and ask again.
