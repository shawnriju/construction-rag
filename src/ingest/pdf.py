"""Thin wrapper around PyMuPDF that returns positioned text blocks.

Every document-specific parser works on `TextBlock`s rather than raw page text,
because layout (x/y position) is what tells us whether a block is a margin
heading, a running header, a footer or body text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass(frozen=True)
class TextBlock:
    """A block of text on a page, with its bounding box in PDF points."""

    pdf_page: int  # 1-based
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def lines(self) -> list[str]:
        return [line.strip() for line in self.text.splitlines() if line.strip()]


def read_blocks(pdf_path: Path, pages: range | None = None) -> dict[int, list[TextBlock]]:
    """Return {pdf_page: [blocks sorted top-to-bottom, then left-to-right]}.

    Args:
        pdf_path: PDF file to read.
        pages: 1-based page numbers to read (all pages if None).
    """
    result: dict[int, list[TextBlock]] = {}
    with fitz.open(pdf_path) as doc:
        page_numbers = pages if pages is not None else range(1, len(doc) + 1)
        for page_no in page_numbers:
            page = doc[page_no - 1]
            blocks = [
                TextBlock(page_no, x0, y0, x1, y1, text)
                for x0, y0, x1, y1, text, *_ in page.get_text("blocks")
                if text.strip()
            ]
            result[page_no] = sorted(blocks, key=lambda b: (round(b.y0), b.x0))
    return result
