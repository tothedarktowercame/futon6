from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import wooliness_index as wool


def fixture(term="alpha", *, prior=False, resolved=False, with_term=True, with_cite=True):
    text = "heading\nalpha cites B\nending\n"
    marks = {"paper-id": "A", "text": text, "marks": []}
    if with_term:
        start = text.index(term)
        marks["marks"].append({"kind": "concept", "start": start, "end": start + len(term),
                               "fields": [["term", term], ["source", "lexicon"]]})
    candidate = {"schema": "expo-candidate/v2", "paper-id": "A",
                 "passage-id": "A:r1:L2-2", "window-lines": [2, 2]}
    strategies = {"schema": "markup-strategies/v1", "terms": []}
    if prior:
        strategies["terms"].append({"term": term, "at": 0, "line": 1, "uses": []})
    records = []
    if with_cite:
        cite_start = text.index("B")
        records.append({"char-anchor": [cite_start, cite_start + 1],
                        "resolved-corpus-id": "B" if resolved else None})
    citations = {"A": {"schema": "futon6/h7-cite-resolution/v1", "paper-id": "A", "records": records},
                 "B": {"schema": "futon6/h7-cite-resolution/v1", "paper-id": "B", "records": []}}
    encyclopedia = {"schema": "concept-encyclopedia-v0", "entries": [{
        "concept": term, "gloss": {"paper": "B", "text": "definition"},
        "defined_in": {"n_papers": 1, "sample": ["B"]}}]}
    return candidate, marks, strategies, citations, encyclopedia


def one(*args, **kwargs):
    candidate, marks, strategies, citations, encyclopedia = fixture(*args, **kwargs)
    return wool.build([candidate], {"A": marks}, {"A": strategies}, citations,
                      encyclopedia)["records"][0]


class WoolinessTests(unittest.TestCase):
    def test_weights_schema_formula_and_bounds(self):
        row = one(resolved=False)
        self.assertEqual(wool.WEIGHTS, {"U": 0.45, "C": 0.25, "D": 0.30})
        self.assertAlmostEqual(row["W"], .45 * row["U"] + .25 * row["C"] + .30 * row["D"])
        for name in ("U", "C", "D", "W"):
            self.assertGreaterEqual(row[name], 0)
            self.assertLessEqual(row[name], 1)

    def test_prior_definition_lowers_u_d_and_w(self):
        # Remove the encyclopedia witness: the only change is a frozen in-paper definition.
        candidate, marks, strategies, citations, encyclopedia = fixture(prior=False, with_cite=False)
        encyclopedia["entries"] = []
        before = wool.build([candidate], {"A": marks}, {"A": strategies}, citations,
                            encyclopedia)["records"][0]
        strategies["terms"] = [{"term": "alpha", "at": 0, "line": 1, "uses": []}]
        after = wool.build([candidate], {"A": marks}, {"A": strategies}, citations,
                           encyclopedia)["records"][0]
        self.assertEqual((before["U"], before["D"]), (1.0, 1.0))
        self.assertEqual((after["U"], after["D"]), (0.0, 0.0))
        self.assertLess(after["W"], before["W"])

    def test_resolving_citation_lowers_c_d_and_w(self):
        before = one(resolved=False)
        after = one(resolved=True)
        self.assertEqual((before["C"], before["D"]), (1.0, 1.0))
        self.assertEqual((after["C"], after["D"]), (0.0, 1 / 3))
        self.assertLess(after["W"], before["W"])

    def test_components_are_monotone_in_the_frozen_formula(self):
        def composite(u, c, d):
            return min(1.0, max(0.0, .45 * u + .25 * c + .30 * d))
        base = composite(.2, .2, .2)
        self.assertGreater(composite(.3, .2, .2), base)
        self.assertGreater(composite(.2, .3, .2), base)
        self.assertGreater(composite(.2, .2, .3), base)

    def test_empty_denominators_are_zero(self):
        row = one(with_term=False, with_cite=False)
        self.assertEqual((row["U"], row["C"], row["D"], row["W"]), (0.0, 0.0, 0.0, 0.0))

    def test_output_bytes_are_deterministic_under_reordered_inputs(self):
        c1, marks, strategies, citations, encyclopedia = fixture(resolved=True)
        c2 = dict(c1, **{"passage-id": "A:r2:L2-2"})
        a = wool.build([c2, c1], {"A": marks}, {"A": strategies},
                       dict(reversed(list(citations.items()))), encyclopedia)
        b = wool.build([c1, c2], {"A": marks}, {"A": strategies}, citations, encyclopedia)
        self.assertEqual(json.dumps(a, indent=2, sort_keys=True).encode(),
                         json.dumps(b, indent=2, sort_keys=True).encode())

    def test_input_api_structurally_refuses_model_outputs(self):
        self.assertEqual(wool.PRECALL_INPUT_ROLES, {
            "candidate", "marks", "strategies", "citation-index", "concept-encyclopedia"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text("{}")
            for forbidden in ("graph", "model-output", "outcome", "clean"):
                with self.subTest(role=forbidden), self.assertRaisesRegex(ValueError, "not a frozen pre-call"):
                    wool.load_role(forbidden, path)
            with self.assertRaisesRegex(ValueError, "frozen candidate schema"):
                wool.load_role("candidate", path)


if __name__ == "__main__":
    unittest.main()
