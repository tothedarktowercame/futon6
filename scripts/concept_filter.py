#!/usr/bin/env python3
"""A caller-supplied filter of phrases that are not concepts, applied while the spine runs.

A concept vocabulary mined from raw prose picks up the phrasing mathematical writing shares
across every field ("proof of theorem", "main result", "sufficiently large"). Which phrases those
are is curated outside the spine; the spine only honours the list, where concepts are first
recorded, so a filtered phrase never enters the defined-index, the concordance's vocabulary or the
hit-list -- and costs nothing downstream.

Off unless `FUTON6_CONCEPT_FILTER` names a filter file, so a run without one is unchanged.

File format (one entry per line; candidates are compared in `warp_hitlist.canon` form):
    ## <category>     entries below belong to this category
    <phrase>          matches a concept exactly
    re: <regex>       matches a whole concept (fullmatch) -- for regular families
    # ...             comment
"""
from __future__ import annotations

import re
import sys as _sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[0]))
import futon6_config as config

from pathlib import Path


@dataclass(frozen=True)
class FilterEntry:
    category: str
    text: str                        # the line as written, for audits
    pattern: re.Pattern[str] | None  # None for an exact phrase


class ConceptFilter:
    def __init__(self, entries: list[FilterEntry]):
        self.entries = entries
        self._phrases = {e.text: e for e in entries if e.pattern is None}
        self._patterns = [e for e in entries if e.pattern is not None]

    def match(self, concept: str) -> FilterEntry | None:
        """The entry that filters this canonical concept, or None if it is kept."""
        hit = self._phrases.get(concept)
        if hit is not None:
            return hit
        return next((e for e in self._patterns if e.pattern.fullmatch(concept)), None)


def load(path: Path) -> ConceptFilter:
    entries: list[FilterEntry] = []
    category = "uncategorised"
    for number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or (line.startswith("#") and not line.startswith("##")):
            continue
        if line.startswith("##"):
            category = line.lstrip("#").strip()
        elif line.startswith("re:"):
            expression = line[3:].strip()
            try:
                entries.append(FilterEntry(category, line, re.compile(expression)))
            except re.error as error:
                raise ValueError(f"{path}:{number}: bad regex {expression!r}: {error}") from error
        else:
            entries.append(FilterEntry(category, line, None))
    return ConceptFilter(entries)


@lru_cache(maxsize=None)
def _load_cached(path: str) -> ConceptFilter:
    return load(Path(path))


def configured() -> ConceptFilter | None:
    """The run's filter (`FUTON6_CONCEPT_FILTER`), or None when the run has none."""
    path = config.concept_filter_path()
    return _load_cached(str(path)) if path is not None else None
