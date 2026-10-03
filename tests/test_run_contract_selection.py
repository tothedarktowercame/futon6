"""FUTON6_RUN_CONTRACT selects the rules a run follows, and mark7-v3 runs the 2026-09-21 rules.

mark7-v4 (Joe, 2026-09-22) lists S3's units under ids cut from their text, and makes S4 cite
sentence units and quote them word for word. The code had hard-coded both, so selecting mark7-v3
still ran them. Each case runs in a fresh interpreter: the contract is read when a module loads.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]

PROBE = r'''
import json, sys
sys.path.insert(0, "scripts")
import candidate_spans, expository_json, mark3_expository_loop as s4, mark3_extract_candidates as s3x
import mark3_extract_expository_candidates as s4x, mark3_iatc_loop as s3
text = "Let $x$ be a real number with $x>0$.\nThen we have $x^2>0$ as required.\n"
marks = {"text": text, "marks": [{"kind": "bind/let", "start": 0, "end": 35}]}
spans = candidate_spans.spans_for({"window-lines": [1, 2]}, marks)
cand = {"window-lines": [10, 12], "source-window": "a", "spans": spans, "proof-lines": [10, 12],
        "enrichment": [], "bindings": [], "defined-terms": []}
region = {"window-lines": [1, 1], "source-window": "x", "region-type": "inflight", "enrichment": []}
doc = {"scopes": [{"kind": "k", "first_line": 1, "last_line": 1, "fill": "not in the text", "held_reason": ""}]}
print(json.dumps({
    "s3-schema": s3x.SCHEMA_PROOF, "span-ids": [sp.get("id") for sp in spans],
    "s3-lists-units": "Marked source units" in s3.build_prompt(cand, s3.NODES_TASK),
    "s4-schema": s4x.SCHEMA, "s4-loop-schema": s4.CANDIDATE_SCHEMA,
    "s4-prompt-units": '"units": the ids' in s4.SYSTEM,
    "s4-line-scope-problems": expository_json.problems(doc, 1, 1, {"k": ""}, ()),
}))
'''


def probe(contract: str | None) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "FUTON6_RUN_CONTRACT"}
    if contract:
        env["FUTON6_RUN_CONTRACT"] = contract
    done = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


class ContractSelection(unittest.TestCase):
    def test_the_default_runs_mark7_v4(self):
        got = probe(None)
        self.assertEqual((got["s3-schema"], got["s4-schema"], got["s4-loop-schema"]),
                         ("iatc-candidate/v5-proof", "expo-candidate/v2", "expo-candidate/v2"))
        self.assertTrue(all(i and i.startswith("L1-") for i in got["span-ids"]))
        self.assertTrue(got["s3-lists-units"] and got["s4-prompt-units"])

    def test_mark7_v3_runs_the_old_rules(self):
        got = probe("mark7-v3")
        self.assertEqual((got["s3-schema"], got["s4-schema"], got["s4-loop-schema"]),
                         ("iatc-candidate/v4-proof", "expo-candidate/v1", "expo-candidate/v1"))
        self.assertEqual(got["span-ids"], [None])          # numbered s1..sn at prompt time
        self.assertFalse(got["s3-lists-units"] or got["s4-prompt-units"])
        self.assertEqual(got["s4-line-scope-problems"], [])  # a line-range scope's fill is not checked


if __name__ == "__main__":
    unittest.main()
