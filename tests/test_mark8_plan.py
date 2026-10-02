from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mark8_plan as plan


def candidate(item: str, *, paper: str = "p1", selected: bool = False,
              span=(1, 3), clauses=None, **extra):
    row = {"paper-id": paper, "passage-id": item, "window-lines": list(span),
           "baseline-selected": selected, **extra}
    if clauses is not None:
        row["clause-spans"] = clauses
    return row


class Mark8PlanTests(unittest.TestCase):
    def test_outputs_are_byte_deterministic_across_input_enumeration_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = candidate("s3-b", paper="p2"), candidate("s3-a")
            x, y = candidate("s4-b", paper="p2"), candidate("s4-a", selected=True)
            plan.write_outputs(root / "left", *plan.build_plan([a, b], [x, y], budget=3, policy="fixture/v1"))
            plan.write_outputs(root / "right", *plan.build_plan([b, a], [y, x], budget=3, policy="fixture/v1"))
            for name in ("precheck-queue.jsonl", "allocation.json", "allocation-analysis.json"):
                self.assertEqual((root / "left" / name).read_bytes(), (root / "right" / name).read_bytes())

    def test_every_input_is_accounted_once_and_sets_do_not_overlap(self):
        queue, allocation, analysis = plan.build_plan(
            [candidate("s3-a"), candidate("s3-b")],
            [candidate("s4-a", selected=True), candidate("s4-b")], budget=3, policy="fixture/v1")
        self.assertEqual(len(queue), analysis["input-items"])
        self.assertEqual(len(queue), len({(row["stage"], row["item-id"]) for row in queue}))
        selected = {(row["stage"], row["item-id"]) for row in allocation["selected"]}
        deferred = {(row["stage"], row["item-id"]) for row in allocation["deferred"]}
        self.assertTrue(selected.isdisjoint(deferred))
        self.assertEqual(selected | deferred, {(row["stage"], row["item-id"]) for row in queue})

    def test_duplicate_identity_is_refused_instead_of_double_spending(self):
        with self.assertRaisesRegex(ValueError, "duplicate candidate identities"):
            plan.build_plan([candidate("same"), candidate("same")], [], budget=1, policy="fixture/v1")

    def test_exact_bad_clause_span_is_precheck_refused_but_graph_cycle_is_not_examined(self):
        bad = candidate("bad", span=(10, 20), clauses=[[9, 12]])
        graph_claim = candidate("cycle", proposed_graph={"edges": [["a", "a"]]})
        queue, _, _ = plan.build_plan([bad, graph_claim], [], budget=2, policy="fixture/v1")
        by_id = {row["item-id"]: row for row in queue}
        self.assertEqual(by_id["bad"]["reason"], "invalid-clause-spans")
        self.assertEqual(by_id["bad"]["allocation"], "precheck-refused")
        self.assertEqual(by_id["cycle"]["precheck"], "eligible")
        self.assertEqual(by_id["cycle"]["allocation"], "selected")

    def test_budget_is_conserved_for_zero_under_and_oversubscribed_inputs(self):
        cases = ((0, 1, 1, 0, 0), (5, 1, 1, 2, 3), (2, 2, 2, 2, 0))
        for budget, s3_count, s4_count, selected, unspent in cases:
            with self.subTest(budget=budget, inputs=s3_count + s4_count):
                _, allocation, analysis = plan.build_plan(
                    [candidate(f"s3-{i}") for i in range(s3_count)],
                    [candidate(f"s4-{i}") for i in range(s4_count)], budget=budget, policy="fixture/v1")
                self.assertEqual(allocation["selected-calls"], selected)
                self.assertEqual(allocation["unspent-calls"], unspent)
                self.assertEqual(selected + unspent, budget)
                self.assertTrue(analysis["budget-conserved"])

    def test_one_refused_s3_slot_becomes_exactly_one_additional_s4_selection(self):
        s3 = [candidate("s3-good"), candidate("s3-bad", span=(3, 1))]
        s4 = [candidate("s4-baseline", selected=True), candidate("s4-fill-1"), candidate("s4-fill-2")]
        queue, allocation, analysis = plan.build_plan(s3, s4, budget=3, policy="fixture/v1")
        self.assertEqual(analysis["selected-by-stage"], {"S3": 1, "S4": 2})
        self.assertEqual(analysis["additional-s4-selections"], 1)
        self.assertEqual({row["item-id"] for row in allocation["selected"]},
                         {"s3-good", "s4-baseline", "s4-fill-1"})
        self.assertEqual(next(row for row in queue if row["item-id"] == "s3-bad")["allocation"],
                         "precheck-refused")

    def test_loader_accepts_object_and_list_and_sorts_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.json").write_text(json.dumps([candidate("b"), candidate("c")]))
            (root / "a.json").write_text(json.dumps(candidate("a")))
            self.assertEqual([plan.identity(row)[0] for row in
                              plan.load_candidates([root / "b.json", root / "a.json"])],
                             ["a", "b", "c"])
