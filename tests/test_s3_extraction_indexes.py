"""S3's per-window lookups, built once per paper, answer exactly as scanning every mark did."""
from __future__ import annotations

from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import candidate_spans as cs  # noqa: E402
import mark3_extract_candidates as mx  # noqa: E402

KINDS = ["let-binder", "bind/let", "definiendum", "definiens", "symbol-grounded", "proof-move", "cite",
         "math", "symbol", "assume/explicit", "quant/universal", "env/proof", "claim"]


def random_marks(rng: random.Random, n_chars: int) -> list[dict]:
    marks = []
    for _ in range(rng.randint(0, 60)):
        a = rng.randint(0, n_chars)
        mark = {"start": a, "end": a + rng.randint(-2, 60), "kind": rng.choice(KINDS)}
        if rng.random() < 0.8:
            mark["tip"] = f"tip {rng.randint(0, 9)}"
        marks.append(mark)
    return marks


def binder_context_by_scan(marks, starts, before_line):
    out = []
    for m in marks:
        if m.get("kind") in {"let-binder", "bind/let", "definiendum", "definiens"} \
                and mx.mark_line(m, starts) < before_line and m.get("tip"):
            out.append(f"({m['kind']}) {m['tip']}")
    return out[-12:]


def enrichment_by_scan(marks, starts, lo_line, hi_line):
    out = []
    for m in marks:
        if m.get("kind") in mx.ENRICH_KINDS and m.get("tip"):
            ln = mx.mark_line(m, starts)
            if lo_line <= ln <= hi_line:
                out.append({"line": ln, "kind": m["kind"], "tip": m["tip"]})
    out.sort(key=lambda r: (r["line"], r["kind"]))
    return out[:mx.ENRICH_CAP]


def spans_by_scan(candidate, marks_doc):
    text = marks_doc.get("text", "")
    starts = cs.line_starts(text)
    lo, hi = candidate.get("window-lines") or [1, 1]
    begin = starts[lo - 1] if 0 <= lo - 1 < len(starts) else 0
    end = starts[hi] if hi < len(starts) else len(text)
    seen, out = set(), []
    for mark in marks_doc.get("marks", ()):
        if mark.get("kind") not in cs.SPAN_KINDS or "start" not in mark:
            continue
        a, b = mark["start"], mark["end"]
        if not (begin <= a and b <= end) or not (cs.MIN_CHARS < b - a < cs.MAX_CHARS):
            continue
        if (a, b) in seen:
            continue
        seen.add((a, b))
        line = max(1, sum(1 for st in starts if st <= a))
        out.append({"kind": mark["kind"], "start": a, "end": b, "text": text[a:b],
                    "line": line, "id": cs.unit_id(line, text[a:b])})
    out.sort(key=lambda sp: (sp["start"], sp["end"]))
    seen_ids = {}
    for sp in out:
        n = seen_ids.get(sp["id"], 0)
        seen_ids[sp["id"]] = n + 1
        if n:
            sp["id"] = f"{sp['id']}{chr(ord('a') + n - 1)}"
    return out


class ExtractionIndexTests(unittest.TestCase):
    def test_window_lookups_match_the_scans(self):
        rng = random.Random(7)
        for _ in range(400):
            text = "".join(rng.choice(["x", " ", "\n", "Let $y$ be", "\n\n"]) for _ in range(rng.randint(1, 80)))
            starts = mx.line_starts(text)
            marks = random_marks(rng, len(text))
            for _ in range(8):
                lo = rng.randint(0, len(starts) + 1)
                hi = lo + rng.randint(-1, 6)
                self.assertEqual(mx.binder_context(marks, starts, lo), binder_context_by_scan(marks, starts, lo))
                self.assertEqual(mx.window_enrichment(marks, starts, lo, hi),
                                 enrichment_by_scan(marks, starts, lo, hi))
                doc = {"text": text, "marks": marks}
                candidate = {"window-lines": [max(1, lo), max(1, hi)]}
                self.assertEqual(cs.spans_for(candidate, doc), spans_by_scan(candidate, doc))

    def test_window_bindings_and_terms_match_the_scans(self):
        rng = random.Random(8)
        for _ in range(300):
            def occurrences():
                rows = [{"line": rng.randint(1, 30), "binder": rng.choice([None, 0, 1]),
                         "rule": "in-section", "definition-site": rng.random() < 0.2}
                        for _ in range(rng.randint(0, 12))]
                if rng.random() < 0.7:
                    rows.sort(key=lambda r: r["line"])  # the pipeline's lists are in line order
                return rows
            strat = {"symbols": [{"name": f"s{i}", "occurrences": occurrences(),
                                  "binders": [{"type": "t", "line": 1}, {"type": "", "line": 2}]}
                                 for i in range(rng.randint(0, 6))],
                     "terms": [{"term": f"t{i}", "line": 3, "uses": occurrences()} for i in range(rng.randint(0, 6))]}
            for _ in range(6):
                lo = rng.randint(0, 31)
                hi = lo + rng.randint(-1, 10)
                bindings, terms = mx.window_bindings(strat, lo, hi), mx.window_terms(strat, lo, hi)
                mx._STRAT = None  # the scan below must not reuse the lines index
                scan = mx._StratLines.__new__(mx._StratLines)
                scan.strat = strat
                scan.symbols = [None] * len(strat["symbols"])
                scan.terms = [None] * len(strat["terms"])
                mx._STRAT = scan
                self.assertEqual(bindings, mx.window_bindings(strat, lo, hi))
                self.assertEqual(terms, mx.window_terms(strat, lo, hi))
                mx._STRAT = None


if __name__ == "__main__":
    unittest.main()
