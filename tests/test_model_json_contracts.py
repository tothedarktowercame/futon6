# ---------------------------------------------------------------------------
# KNOWN FAILING WITHOUT THE INPUTS BELOW (recorded 2026-09-23)
#
# Documented rather than repaired. Each cause is stated so a reader can tell a
# missing input or upstream drift from a defect in the code under test.
#
# 1 failure + 4 errors - babashka:
#   `bb` (babashka) is not on PATH. The gate chain shells out to it and dies
#   with FileNotFoundError before any assertion runs. babashka is a required
#   tool for these tests, not an optional one - see scripts/preflight.py, which
#   refuses without it. Put `bb` on PATH and these pass.
# ---------------------------------------------------------------------------
"""S4 and S7: schema-constrained JSON from the model, EDN written by code, no re-prompting."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import expository_json
import mark3_expository_loop as expo_loop
import stage_accounting as accounting

# expo-candidate/v2: the region's sentence units, which a scope cites and quotes from.
UNITS = [{"id": "L20-0a1b", "line": 20, "start": 100, "end": 135,
          "text": "In Section 2 we define\nthe notation."},
         {"id": "L22-7c3d", "line": 22, "start": 136, "end": 165,
          "text": "Section 3\nproves the theorem."}]
CANDIDATE = {"schema": "expo-candidate/v2", "paper-id": "1111.0001", "passage-id": "1111.0001:leaf-0001:L20-24",
             "region-id": "leaf-0001", "region-type": "leaf-section", "window-lines": [20, 24],
             "source-window": "In Section 2 we define\nthe notation.\nSection 3\nproves the theorem.\nx",
             "units": UNITS,
             "enrichment": [], "vocab-path": "holes/excursions/expository-superpod-vocab.edn"}


def scope(kind="rationale/telos/organization-roadmap", units=("L20-0a1b",),
          fill="Section 2 we define the notation", held=""):
    return {"kind": kind, "units": list(units), "fill": fill, "held_reason": held}


class ExpositoryContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kinds = expository_json.vocabulary()

    def test_vocabulary_keeps_nested_kinds_with_their_own_slots(self):
        self.assertEqual(len(self.kinds), 16)
        self.assertEqual(self.kinds["rationale/telos"], "telos")
        self.assertEqual(self.kinds["rationale/telos/organization-roadmap"], "roadmap")
        self.assertNotIn("perf/Agree", self.kinds)

    def test_problems_and_code_written_edn_pass_the_gate(self):
        doc = {"scopes": [scope(), scope(kind="heuristic-plausibility", units=("L22-7c3d",), fill="",
                                          held='no expectation is stated; "held" honestly')]}
        self.assertEqual(expository_json.problems(doc, 20, 24, self.kinds, UNITS), [])
        for bad, text in (({"scopes": [scope(fill="", held="")]}, "exactly one"),
                          ({"scopes": [scope(fill="x", held="y")]}, "exactly one"),
                          ({"scopes": [scope(units=())]}, "cites no unit"),
                          # the fill must be the passage's words, not the model's about them
                          ({"scopes": [scope(fill="the notation of the paper")]}, "is not in the unit"),
                          ({"scopes": [scope(kind="perf/Agree")]}, "not in the vocabulary"),
                          ({"scopes": []}, "nonempty")):
            self.assertTrue(any(text in p for p in expository_json.problems(bad, 20, 24, self.kinds, UNITS)), text)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "g.edn"
            path.write_text(expository_json.to_edn(doc, CANDIDATE, self.kinds, "m"))
            gate = subprocess.run(["bb", str(ROOT / "scripts/expository_argcheck.bb"), str(path)],
                                  capture_output=True, text=True)
            self.assertEqual(gate.returncode, 0, gate.stdout + gate.stderr)
            text = path.read_text()
        self.assertIn(':slot-fill {:roadmap "Section 2 we define the notation"}', text)
        self.assertIn(':units ["L20-0a1b"]', text)            # what the scope reads
        self.assertIn(':fill-span [103 135]', text)           # and the extent of its own words
        self.assertIn(':held {:reason "no expectation is stated; \\"held\\" honestly"}', text)

    def test_loop_accounts_each_outcome_and_retries_only_failures(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            cands = base / "cands"
            cands.mkdir()
            for i in range(3):
                c = dict(CANDIDATE, **{"passage-id": f"1111.0001:leaf-000{i}:L20-24", "region-id": f"leaf-000{i}"})
                (cands / f"1111.0001.leaf-000{i}.candidate.json").write_text(json.dumps(c))
            answers = {"1111.0001:leaf-0000:L20-24": json.dumps({"scopes": [scope()]}),
                       "1111.0001:leaf-0001:L20-24": json.dumps({"scopes": [scope(fill="", held="")]}),
                       "1111.0001:leaf-0002:L20-24": expo_loop.ModelCallError("output truncated at max_tokens=2048")}
            calls = []

            def call(_prompt, candidate, _kinds, _model):
                calls.append(candidate["passage-id"])
                answer = answers[candidate["passage-id"]]
                if isinstance(answer, Exception):
                    raise answer
                return answer

            def invoke(invocation):
                adir = base / "accounting" / invocation
                args = argparse.Namespace(candidates=str(cands), out=str(base / "out"), backend="openai", model="m")
                with patch.object(expo_loop, "call_openai", call), \
                        patch.dict(os.environ, {"RUN_ID": "r", "FUTON6_RUN_DIR": str(base),
                                                accounting.DIR_ENV: str(adir), accounting.INVOCATION_ENV: invocation}):
                    rc = expo_loop.run(args)
                return rc, {e["id"]: e for e in accounting.load(adir, "S4", "loop")["items"]}

            rc, items = invoke("S4-a001")
            self.assertEqual(rc, 0)  # refused regions are recorded, not a loop failure
            self.assertEqual([items[k]["status"] for k in sorted(items)], ["accepted", "rejected", "errored"])
            self.assertIn("exactly one", items["1111.0001:leaf-0001:L20-24"]["reason"])
            answers["1111.0001:leaf-0001:L20-24"] = json.dumps({"scopes": [scope()]})
            answers["1111.0001:leaf-0002:L20-24"] = json.dumps({"scopes": [scope(kind="generalisation")]})
            calls.clear()
            rc, items = invoke("S4-a002")
            self.assertEqual(rc, 0)
            self.assertNotIn("1111.0001:leaf-0000:L20-24", calls)

    def test_request_is_schema_constrained_at_temperature_zero(self):
        seen = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}).encode()

        def urlopen(request, timeout):
            seen.update(json.loads(request.data))
            return Response()

        with patch("urllib.request.urlopen", urlopen):
            expo_loop.call_openai("p", CANDIDATE, self.kinds, "m")
        self.assertEqual(seen["temperature"], 0)
        self.assertEqual(seen["response_format"]["json_schema"]["schema"],
                         expository_json.schema(20, 24, self.kinds, UNITS))
        # the model may only cite units this region actually has
        scope_props = seen["response_format"]["json_schema"]["schema"]["properties"]["scopes"]["items"]["properties"]
        self.assertEqual(scope_props["units"]["items"]["enum"], [u["id"] for u in UNITS])


class CleanTypingContract(unittest.TestCase):
    GRAPH = """{:paper/id "9999.0003" :nodes [{:id :n1 :kind :claim :text "A"} {:id :n2 :kind :claim :text "B"}
 {:id :n3 :kind :claim :text "C"}]
 :edges [{:id :e1 :kind :infer :premise [:n1] :conclusion :n2 :warrant {:kind :claim :text "w"}}
         {:id :e2 :kind :infer :premise [:n2] :conclusion :n3 :warrant {:kind :claim :text "w"}}]}"""

    def run_typing(self, answer):
        import clean_box_typing as typing
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            graphs = base / "graphs"
            graphs.mkdir()
            final = graphs / "9999.0003__p0.edn"
            final.write_text(self.GRAPH)
            accounting.record_acceptance(graphs, "9999.0003__p0", final, {"path": "fixture"})
            adir = base / "accounting"
            seen = {}

            def query(endpoint, model, prompt, sk, methods):
                seen["schema"] = typing.typing_schema(sk, methods)
                seen["prompt"] = prompt
                if isinstance(answer, Exception):
                    raise answer
                return answer(sk) if callable(answer) else answer

            argv = ["clean_box_typing.py", "--graphs", str(graphs), "--out", str(base / "clean")]
            with patch.object(sys, "argv", argv), patch.object(typing, "query_model", query), \
                    patch.object(typing, "wait_for_server", lambda _e: None), \
                    patch.dict(os.environ, {accounting.DIR_ENV: str(adir)}), self.assertRaises(SystemExit) as exit:
                typing.main()
            item = accounting.load(adir, "S7", "typing")["items"][0]
            return exit.exception.code, item, seen

    def test_schema_keys_are_box_ids_and_valid_typing_is_accepted(self):
        code, item, seen = self.run_typing(lambda sk: {b["id"]: "reduce-to-known-result" for b in sk["boxes"]})
        self.assertEqual((code, item["status"]), (0, "accepted"))
        self.assertEqual(sorted(seen["schema"]["required"]), sorted(seen["schema"]["properties"]))
        self.assertEqual(len(seen["schema"]["required"]), 2)
        self.assertNotIn("_macro", seen["prompt"])

    def test_contract_violation_rejects_without_reprompting_and_endpoint_failure_errors(self):
        import clean_box_typing as typing
        code, item, _ = self.run_typing({"e1": "not-a-method"})
        self.assertEqual((code, item["status"]), (1, "rejected"))
        self.assertIn("typing contract", item["reason"])
        code, item, _ = self.run_typing(typing.TypingCallError("output truncated at max_tokens=600"))
        self.assertEqual((code, item["status"]), (1, "errored"))


if __name__ == "__main__":
    unittest.main()
