#!/usr/bin/env python3
r"""Sentence-sized units of an expository region, with offsets and ids.

S3 offers a proof's clauses as marked units and takes each node's mathematics from
the source by offset, so a node's quote is a span the reader can be shown. S4 had no
such thing: a scope cites first_line/last_line and its fill is a string the model
types. In mark7master-20260921 only 289 of 762 filled scopes (38%) say anything that
appears in the passage word for word, and a scope can only ever be shown a line at a
time, however small the thing it is about.

Prose has no S1 clause marks to offer -- bind/let and quant/universal are the
furniture of statements, not of exposition -- so the units are carved here: the
region's sentences, split outside math ($...$ and displays hold periods of their
own: "0<i<n." and "\dots"), each with its offsets in the paper and an id cut from
its own text, as the proof units are (candidate_spans.unit_id).
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

import re

import candidate_spans
import expository_region_extract as regions

MIN_CHARS = 12          # shorter than this is a fragment, not a sentence
MAX_UNITS = 60          # a listing longer than this is not read; regions this long are rare
# A line that is only commands and their arguments: \section{...}\label{...}, \eeg.
MARKUP_LINE = re.compile(r"\s*(?:\\[A-Za-z@]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})*\s*)+")
ABBREVIATIONS = {"e.g", "i.e", "cf", "resp", "etc", "vs", "Fig", "Thm", "Def", "Prop", "Lem",
                 "Ch", "Sec", "no", "Nr", "al", "Dr", "Prof", "Mr", "Ms", "St"}


def sentence_bounds(text: str) -> list[tuple[int, int]]:
    """Sentence ranges over `text`, never splitting inside math or after an abbreviation."""
    out, start, i, n = [], 0, 0, len(text)
    depth_math = False
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n:                      # \$ and friends are not delimiters
            i += 2
            continue
        if ch == "$":
            if text.startswith("$$", i):
                i += 2
            else:
                i += 1
            depth_math = not depth_math
            continue
        if not depth_math and ch in ".!?":
            word = re.search(r"([A-Za-z]+)$", text[start:i])
            if word and word.group(1) in ABBREVIATIONS:
                i += 1
                continue
            j = i + 1
            while j < n and text[j] in ")]}'\"":
                j += 1
            if j >= n or text[j].isspace():
                out.append((start, j))
                while j < n and text[j].isspace():
                    j += 1
                start, i = j, j
                continue
        if not depth_math and ch == "\n":
            # A paragraph break ends a sentence; so does a line that is only markup
            # (\section{...}\label{...}), which belongs to no sentence after it.
            line_start = text.rfind("\n", 0, i) + 1
            if text.startswith("\n\n", i) or MARKUP_LINE.fullmatch(text[line_start:i]):
                out.append((start, i))
                while i < n and text[i].isspace():
                    i += 1
                start = i
                continue
        i += 1
    if start < n:
        out.append((start, n))
    return [(a, b) for a, b in out if text[a:b].strip()]


def units_for(text: str, starts: list[int], lo_line: int, hi_line: int) -> list[dict]:
    """The region's sentences as citable units: id, line, offsets into `text`, text."""
    begin = starts[lo_line - 1] if 0 < lo_line <= len(starts) else 0
    end = starts[hi_line] if 0 < hi_line < len(starts) else len(text)
    window = text[begin:end]
    out = []
    for a, b in sentence_bounds(window):
        piece = window[a:b].strip()
        # A unit with no running prose (a lone \section{...}\label{...}) gives a scope
        # nothing to read, and is not something to cite.
        if len(piece) < MIN_CHARS or regions.prose_words(piece) < 3:
            continue
        start = begin + a + (len(window[a:b]) - len(window[a:b].lstrip()))
        line = max(1, sum(1 for s in starts if s <= start))
        out.append({"id": candidate_spans.unit_id(line, piece), "line": line,
                    "start": start, "end": start + len(piece), "text": piece})
    seen: dict[str, int] = {}
    for u in out:
        k = seen.get(u["id"], 0)
        seen[u["id"]] = k + 1
        if k:
            u["id"] = f"{u['id']}{chr(ord('a') + k - 1)}"
    return out[:MAX_UNITS]


def locate_in_units(fill: str, units: list[dict]) -> list[int] | None:
    """Where a fill sits inside the units it cites, as [start, end) in the paper."""
    words = [w for w in re.split(r"\s+", (fill or "").strip().rstrip(".")) if w]
    if not words:
        return None
    pattern = r"\s+".join(re.escape(w) for w in words)
    for u in units:
        m = re.search(pattern, u["text"], re.I)
        if m:
            return [u["start"] + m.start(), u["start"] + m.end()]
    return None
