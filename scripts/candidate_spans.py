#!/usr/bin/env python3
"""Attach S1's clause-sized marked units to a proof candidate, with offsets.

Line numbers were the wrong granularity for a node's mathematics. A LaTeX line
routinely carries a hypothesis AND the conclusion drawn from it: 0705.0102 line
622 states two hypotheses and a preenvelope claim in one sentence, so three
correctly-glossed nodes could only cite the same line and the edge between them
read as "this text implies this same text". 43 of 648 edges in the first clean
corpus did that.

S1 had already segmented that line -- bind/let for the category, assume/explicit
for the corigid object, quant/universal for the preenvelope -- each with
character offsets into the document. This puts those units in the candidate, so
the candidate stays self-contained and the model can select a clause instead of
a line.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SPAN_KINDS = ("bind/let", "assume/explicit", "quant/universal", "constrain/relation",
              "bind/typed", "definiendum", "definiens", "let-binder",
              "constrain/such-that", "env/proposition", "env/proof")
MIN_CHARS, MAX_CHARS = 4, 400


def line_starts(text: str) -> list[int]:
    return [0] + [i + 1 for i, ch in enumerate(text) if ch == "\n"]


def unit_id(line: int, text: str) -> str:
    """A unit's id names its line and its text, so it cannot be produced by counting.

    With ids s1..sn a model that never sees the unit list can still answer: node i
    cites s_i, and 87% of the nodes in mark7master-20260921 did exactly that. An id
    like L349-3f9a has to be read off the listing, and it carries the evidence (the
    line) for whoever checks the citation afterwards."""
    return f"L{line}-{hashlib.sha1(text.encode()).hexdigest()[:4]}"


def spans_for(candidate: dict, marks_doc: dict) -> list[dict]:
    """In-window marked units, source-ordered, deduplicated by (start, end)."""
    text = marks_doc.get("text", "")
    starts = line_starts(text)
    lo, hi = candidate.get("window-lines") or [1, 1]
    begin = starts[lo - 1] if 0 <= lo - 1 < len(starts) else 0
    end = starts[hi] if hi < len(starts) else len(text)

    seen, out = set(), []
    for mark in marks_doc.get("marks", ()):
        if mark.get("kind") not in SPAN_KINDS or "start" not in mark:
            continue
        a, b = mark["start"], mark["end"]
        if not (begin <= a and b <= end) or not (MIN_CHARS < b - a < MAX_CHARS):
            continue
        if (a, b) in seen:
            continue
        seen.add((a, b))
        line = max(1, sum(1 for st in starts if st <= a))
        out.append({"kind": mark["kind"], "start": a, "end": b, "text": text[a:b],
                    "line": line, "id": unit_id(line, text[a:b])})
    out.sort(key=lambda sp: (sp["start"], sp["end"]))
    # Two identical clauses on one line would share an id; the second gets a suffix.
    seen_ids: dict[str, int] = {}
    for sp in out:
        n = seen_ids.get(sp["id"], 0)
        seen_ids[sp["id"]] = n + 1
        if n:
            sp["id"] = f"{sp['id']}{chr(ord('a') + n - 1)}"
    return out


def attach(candidate_path: Path, marks_dir: Path) -> int:
    candidate = json.loads(candidate_path.read_text())
    marks = marks_dir / f"fable-{candidate['paper-id']}-dp-emacs.json"
    if not marks.is_file():
        return 0
    spans = spans_for(candidate, json.loads(marks.read_text()))
    candidate["spans"] = spans
    candidate_path.write_text(json.dumps(candidate))
    return len(spans)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: candidate_spans.py <candidates-dir> <marks-dir>")
        return 2
    cands, marks = Path(argv[0]), Path(argv[1])
    counts = [attach(p, marks) for p in sorted(cands.glob("*.candidate.json"))]
    got = [c for c in counts if c]
    print(f"attached spans to {len(got)}/{len(counts)} candidates; "
          f"{sum(counts)} spans, median {sorted(got)[len(got) // 2] if got else 0}")
    return 0 if len(got) == len(counts) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
