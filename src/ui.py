"""Streamlit web page: ask a question, see the cited answer and its sources.

Run from the project folder:
    streamlit run src/ui.py

A thin layer over RAGPipeline (the same code the CLI uses), so the answer,
warnings and sources are exactly what `python -m src.ask` would show.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# `streamlit run src/ui.py` puts src/ (not the project root) on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from src.config import TOP_K  # noqa: E402
from src.generate import amendment_links  # noqa: E402
from src.pipeline import Answer, RAGPipeline  # noqa: E402

EXAMPLES = [
    "What is the force coefficient for a single frame with solidity ratio 0.2 and flat-sided members?",
    "What is the basic wind speed in Chennai?",
    "What compensation is payable for delay under the CPWD contract?",
    "How long is the contractor liable for defects after the work is completed under the CPWD contract?",
    "What relief did the Supreme Court grant under Article 142 in Ssangyong?",
    "What is the penalty for late submission of a tender in the Delhi Metro contract?",
]
OPEN_WITHOUT_ANSWER = 3   # Retrieval-only: open the top passages, since they are the answer.
SCOPE_NOTE = (
    "Answers come only from three documents: **CPWD General Conditions of Contract 2020**, "
    "**Ssangyong v NHAI (Supreme Court, 2019)** and **IS 875 (Part 3):1987 with Amendments 1–3**. "
    "Newer editions of IS 875 are not included."
)

# Characters Streamlit's Markdown would treat as formatting ($ starts LaTeX, * and _ emphasis, ...).
_MARKDOWN_SPECIALS = re.compile(r"([\\`*_\[\]#<>|~$])")


def _plain(text: str) -> str:
    """Show document text as written: escape Markdown and keep its line breaks."""
    return _MARKDOWN_SPECIALS.sub(r"\\\1", text).replace("\n", "  \n")


@st.cache_resource(show_spinner="Loading the search index and models (the first run downloads ~130 MB) ...")
def load_pipeline() -> RAGPipeline:
    return RAGPipeline()


def _use_example(question: str) -> None:
    st.session_state.question = question
    st.session_state.ask_now = True


def render_answer(answer: Answer, debug: bool) -> None:
    st.subheader("Answer")
    if answer.mode == "generated":
        st.markdown(_plain(answer.text))
    else:
        st.info("No written answer. These are the most relevant passages, best match first.")
    for warning in answer.warnings:
        st.warning(warning)
    st.caption(f"{answer.seconds:.1f} seconds")

    st.subheader("Sources")
    st.caption(
        "[S1], [S2] ... in the answer point to these sources. ★ = cited in the answer. "
        "Pages: the printed page number, then the page in the PDF file (in data/pdfs/). "
        "Click a source to read the passage."
    )
    links = amendment_links(answer.sources)
    cited = set(answer.citations.cited)
    for i, hit in enumerate(answer.sources, start=1):
        chunk = hit.chunk
        star = "★ " if i in cited else ""
        badge = "  ·  AMENDMENT" if chunk.is_amendment else ""
        expanded = i in cited or (answer.mode == "retrieval-only" and i <= OPEN_WITHOUT_ANSWER)
        with st.expander(f"{star}[S{i}]  {_plain(chunk.citation)}{badge}", expanded=expanded):
            if chunk.title:
                st.markdown(f"**{_plain(chunk.title)}**")
            notes = []
            if chunk.is_amendment:
                notes.append("Official amendment: it replaces the original wording where the two differ.")
            if chunk.source == "curated":
                notes.append("Typed in by hand from the PDF, because the scanned page couldn't be read reliably.")
            if i in links:
                notes.append("Changed by a later amendment: " + ", ".join(f"[S{a}]" for a in links[i]) + ".")
            if hit.attached_reason:
                notes.append(f"Added automatically: {hit.attached_reason}.")
            if debug:
                ranks = ", ".join(f"{name} #{rank}" for name, rank in hit.ranks.items()) or "not ranked (attached)"
                notes.append(f"Ranks: {ranks}; fused score {hit.score:.4f}")
            for note in notes:
                st.caption(_plain(note))
            st.markdown(_plain(chunk.text))


def main() -> None:
    st.set_page_config(page_title="Construction RAG", page_icon="🏗️", layout="centered")
    st.title("Construction RAG")
    st.caption("Ask a question about Indian construction documents. Every answer cites its sources.")
    st.info(SCOPE_NOTE)

    pipeline = load_pipeline()

    with st.sidebar:
        st.header("Settings")
        if pipeline.llm.available():
            st.success(f"Answer model running: {pipeline.llm.name}")
        else:
            st.warning("Ollama isn't running, or its model isn't downloaded, so you'll see the most relevant "
                       "passages only (see the README).")
        use_llm = st.toggle("Write an answer", value=True, help="Off: show only the ranked passages.")
        mode = st.selectbox(
            "Search",
            ["hybrid", "bm25", "dense"],
            format_func={"hybrid": "Keywords + meaning (default)", "bm25": "Keywords only",
                         "dense": "Meaning only"}.get,
        )
        debug = st.checkbox("Show ranking details")
        st.header("Try a question")
        for example in EXAMPLES:
            st.button(example, on_click=_use_example, args=(example,), use_container_width=True)

    with st.form("ask"):
        question = st.text_input("Your question", key="question",
                                 placeholder="e.g. What is the basic wind speed in Chennai?")
        submitted = st.form_submit_button("Ask", type="primary")

    if (submitted or st.session_state.pop("ask_now", False)) and question.strip():
        spinner = "Searching and writing the answer (about 20 seconds) ..." if use_llm else "Searching ..."
        with st.spinner(spinner):
            st.session_state.answer = pipeline.ask(question.strip(), top_k=TOP_K, mode=mode, use_llm=use_llm)

    if "answer" in st.session_state:
        render_answer(st.session_state.answer, debug)


if __name__ == "__main__":
    main()
