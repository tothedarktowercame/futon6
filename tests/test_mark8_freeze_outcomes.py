from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts import mark8_freeze_outcomes as freeze


def candidate(directory: Path, pid: str, passage: str, lines=(1, 10), subdir=""):
    root = directory / subdir
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{pid}.candidate.json").write_text(json.dumps({
        "schema": "iatc-candidate/v5-proof", "paper-id": passage.split(":", 1)[0],
        "proof-id": pid, "passage-id": passage, "window-lines": list(lines)}))


def raw(pairs, *, quote_verdicts=None):
    comprehension = {"proofs": [{"pid": pid, "verdict": verdict} for pid, _, verdict in pairs]}
    quote = {"graphs": [{"passage": passage, "nodes": [
        {"verdict": value} for value in (quote_verdicts or ["agrees", "unclear", "uncheckable"])]}
        for _, passage, _ in pairs]}
    return comprehension, quote


class FreezeOutcomesTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.candidates = self.root / "candidates"
        self.candidates.mkdir()

    def write_sources(self, comprehension, quote):
        comp = self.root / "comprehension.json"
        quotes = self.root / "quote.json"
        comp.write_text(json.dumps(comprehension))
        quotes.write_text(json.dumps(quote))
        return comp, quotes

    def freeze(self, pairs, *, quote_verdicts=None):
        comprehension, quote = raw(pairs, quote_verdicts=quote_verdicts)
        comp, quotes = self.write_sources(comprehension, quote)
        return freeze.freeze(self.candidates, comp, quotes)

    def test_exact_mapping_preserves_math_slash_and_safe_ids(self):
        pairs = [("old_math_p0", "math/0301001:proof0:L1-10", "weak-extraction"),
                 ("safe_math_p0", "math__0301001:proof0:L1-10", "well-formed")]
        for pid, passage, _ in pairs:
            candidate(self.candidates, pid, passage)
        outcomes, quotes, audit = self.freeze(pairs)
        self.assertEqual([row["passage-id"] for row in outcomes["records"]],
                         ["math/0301001:proof0:L1-10", "math__0301001:proof0:L1-10"])
        self.assertEqual([row["weak-extraction"] for row in outcomes["records"]], [True, False])
        self.assertEqual(audit["source-counts"], {"candidates": 2, "comprehension": 2,
                                                  "quote-agreement": 2})
        self.assertEqual(len(quotes["records"]), 2)

    def test_duplicate_candidate_stem_and_passage_are_refused(self):
        passage = "p:proof0:L1-2"
        candidate(self.candidates, "p__p0", passage, subdir="a")
        candidate(self.candidates, "p__p0", passage, subdir="b")
        comp, quotes = self.write_sources(*raw([("p__p0", passage, "well-formed")]))
        with self.assertRaisesRegex(ValueError, "duplicate candidate stem"):
            freeze.freeze(self.candidates, comp, quotes)

        for path in self.candidates.rglob("*.json"):
            path.unlink()
        candidate(self.candidates, "p__p0", passage, subdir="a")
        candidate(self.candidates, "q__p0", passage, subdir="b")
        with self.assertRaisesRegex(ValueError, "duplicate candidate passage-id"):
            freeze.freeze(self.candidates, comp, quotes)

    def test_missing_mapping_is_refused(self):
        candidate(self.candidates, "p__p0", "p:proof0:L1-2")
        comp, quotes = self.write_sources(*raw([("other__p0", "other:proof0:L1-2", "well-formed")]))
        with self.assertRaisesRegex(ValueError, "mapping is not exact"):
            freeze.freeze(self.candidates, comp, quotes)

    def test_candidate_superset_is_allowed_and_unused_rows_are_accounted(self):
        used = ("p__p0", "p:proof0:L1-2", "well-formed")
        candidate(self.candidates, used[0], used[1])
        candidate(self.candidates, "extra__p0", "extra:proof0:L1-2")
        outcomes, _, audit = self.freeze([used])
        self.assertEqual(len(outcomes["records"]), 1)
        self.assertEqual(audit["omissions"]["unused-candidates"], ["extra__p0"])

    def test_quote_share_and_zero_checkable_omission(self):
        pairs = [("p__p0", "p:proof0:L1-10", "partial-comprehension")]
        candidate(self.candidates, pairs[0][0], pairs[0][1])
        _, quotes, _ = self.freeze(pairs, quote_verdicts=["agrees", "unclear", "uncheckable"])
        self.assertEqual(quotes["records"][0]["agrees-share"], .5)
        self.assertEqual((quotes["records"][0]["agrees"], quotes["records"][0]["checkable"]), (1, 2))

        _, quotes, audit = self.freeze(pairs, quote_verdicts=["uncheckable"])
        self.assertEqual(quotes["records"], [])
        self.assertEqual(audit["omissions"]["zero-checkable"], [pairs[0][1]])

    def test_baseline_is_candidate_only_bounded_and_declared(self):
        pairs = [("p__p0", "p:proof0:L1-500", "weak-proof")]
        candidate(self.candidates, pairs[0][0], pairs[0][1], lines=(1, 500))
        outcomes, _, _ = self.freeze(pairs)
        self.assertEqual(outcomes["baseline-name"], freeze.BASELINE_NAME)
        self.assertEqual(outcomes["baseline"]["uses-outcome-labels"], False)
        self.assertEqual(outcomes["records"][0]["baseline-proxy"], 1.0)

    def test_invalid_verdict_and_nonfinite_input_are_refused(self):
        pairs = [("p__p0", "p:proof0:L1-2", "invented")]
        candidate(self.candidates, pairs[0][0], pairs[0][1])
        with self.assertRaisesRegex(ValueError, "invalid comprehension verdict"):
            self.freeze(pairs)
        comprehension, quote = raw([("p__p0", pairs[0][1], "well-formed")])
        comprehension["proofs"][0]["noun"] = float("nan")
        comp, quotes = self.write_sources(comprehension, quote)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            freeze.freeze(self.candidates, comp, quotes)

    def test_output_bytes_are_deterministic_under_source_and_creation_order(self):
        pairs = [("b__p0", "b:proof0:L2-4", "well-formed"),
                 ("a__p0", "a:proof0:L1-2", "weak-extraction")]
        for pid, passage, _ in pairs:
            candidate(self.candidates, pid, passage)
        first = self.freeze(pairs)
        second = self.freeze(list(reversed(pairs)))
        for left, right in zip(first, second):
            self.assertEqual(freeze.encoded(left), freeze.encoded(right))


if __name__ == "__main__":
    unittest.main()
