"""Small Markdown helpers shared by the evaluation reports."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar("T")


def markdown_table(header: list[str], rows: Iterable[list[str]]) -> list[str]:
    """A Markdown table as a list of lines."""
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def group_by(items: Iterable[T], key: Callable[[T], str]) -> dict[str, list[T]]:
    """Group items by a key, sorted by key (e.g. results by doc, type or wording)."""
    groups: dict[str, list[T]] = {}
    for item in items:
        groups.setdefault(key(item), []).append(item)
    return dict(sorted(groups.items()))


def yes_no(flag: bool) -> str:
    """'yes', or a bold '**no**' so failures stand out in a per-question table."""
    return "yes" if flag else "**no**"
