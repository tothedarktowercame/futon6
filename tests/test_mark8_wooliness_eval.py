from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mark8_wooliness_eval as evaluation


def inputs(rows):
    wool = {"schema": "futon6/mark8-wooliness/v1", "records": []}
    outcomes = {"schema": evaluation.OUTCOME_SCHEMA, "baseline-name": "quote-risk/v1", "records": []}
    quotes = {"schema": evaluation.QUOTE_SCHEMA, "records": []}
    for identifier, score, label, baseline in rows:
        paper = identifier.split(":", 1)[0]
        wool["records"].append({"passage-id": identifier, "paper-id": paper,
                                "W": score, "U": score, "C": 0.0, "D": 0.0})
        outcomes["records"].append({"passage-id": identifier, "status": "accepted",
                                    "weak-extraction": label, "baseline-proxy": baseline})
        quotes["records"].append({"passage-id": identifier, "agrees-share": 1 - baseline})
    return wool, outcomes, quotes


class AucTests(unittest.TestCase):
    def test_perfect_inverted_and_tied_auc(self):
        self.assertEqual(evaluation.auc([1, 0], [True, False]), 1.0)
        self.assertEqual(evaluation.auc([0, 1], [True, False]), 0.0)
        self.assertEqual(evaluation.auc([.5, .5], [True, False]), .5)
        self.assertIsNone(evaluation.auc([1, .5], [True, True]))


class EvaluationTests(unittest.TestCase):
    def test_deterministic_bytes_under_reordered_inputs(self):
        rows = [("p:b", .8, True, .7), ("p:a", .2, False, .3)]
        first = inputs(rows)
        second = inputs(list(reversed(rows)))
        a = evaluation.evaluate(*first, top_k=2)
        b = evaluation.evaluate(*second, top_k=2)
        self.assertEqual(json.dumps(a, indent=2, sort_keys=True).encode(),
                         json.dumps(b, indent=2, sort_keys=True).encode())

    def test_missing_duplicate_and_refused_ids_are_accounted_not_joined(self):
        wool, outcomes, quotes = inputs([
            ("ok", .8, True, .4), ("duplicate", .7, True, .4),
            ("refused", .6, True, .4), ("missing-quote", .5, False, .4)])
        wool["records"].append(dict(wool["records"][1]))
        refused = next(row for row in outcomes["records"] if row["passage-id"] == "refused")
        refused["status"] = "refused"
        quotes["records"] = [row for row in quotes["records"] if row["passage-id"] != "missing-quote"]
        outcomes["records"].append({"passage-id": "outcome-only", "status": "accepted",
                                    "weak-extraction": False, "baseline-proxy": .1})

        report = evaluation.evaluate(wool, outcomes, quotes)

        self.assertEqual(report["join"]["joined"], 1)
        self.assertEqual(report["join"]["duplicates"]["wooliness"], ["duplicate"])
        self.assertEqual(report["join"]["refused-ids"], ["refused"])
        self.assertIn("missing-quote", report["join"]["unmatched"]["wooliness"])
        self.assertIn("outcome-only", report["join"]["unmatched"]["outcomes"])

    def test_old_style_arxiv_ids_are_exact_keys_and_remain_distinct(self):
        wool, outcomes, quotes = inputs([
            ("math/0301001:proof0", .8, True, .7),
            ("math__0301001:proof0", .2, False, .3)])
        report = evaluation.evaluate(wool, outcomes, quotes)
        self.assertEqual([row["passage-id"] for row in report["C4-points"]],
                         ["math/0301001:proof0", "math__0301001:proof0"])

    def test_insufficient_coverage_cannot_pass(self):
        wool, outcomes, quotes = inputs([("joined", .9, True, .1), ("missing", .1, False, .9)])
        outcomes["records"] = outcomes["records"][:1]
        quotes["records"] = quotes["records"][:1]
        report = evaluation.evaluate(wool, outcomes, quotes)
        self.assertEqual(report["gate"]["status"], "insufficient")
        self.assertEqual(report["join"]["coverage"], .5)

    def test_gate_requires_auc_threshold_and_baseline_comparison(self):
        rows = []
        for i in range(10):
            rows.append((f"p:weak-{i}", .9, True, .6))
            rows.append((f"p:strong-{i}", .1, False, .4))
        report = evaluation.evaluate(*inputs(rows))
        self.assertEqual(report["gate"]["status"], "eligible")
        self.assertEqual(report["auc"], {"W": 1.0, "baseline": 1.0,
                                         "baseline-name": "quote-risk/v1"})
        self.assertEqual(len(report["calibration"]), 10)
        top = report["C6-attention"]["rows"][0]
        self.assertEqual(set(top["components"]), {"W", "quote-disagreement", "baseline-proxy"})

    def test_module_imports_only_standard_library_and_reads_only_cli_files(self):
        tree = ast.parse((ROOT / "scripts" / "mark8_wooliness_eval.py").read_text())
        imports = {alias.name.split(".")[0] for node in ast.walk(tree)
                   if isinstance(node, ast.Import) for alias in node.names}
        imports |= {node.module.split(".")[0] for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.module}
        self.assertEqual(imports, {"__future__", "argparse", "json", "pathlib", "typing"})
        self.assertEqual(evaluation.INPUT_FILES, ("wooliness", "outcomes", "quote-agreement"))


if __name__ == "__main__":
    unittest.main()
