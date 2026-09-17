"""S3: proofs come from S1; the model returns schema JSON; code writes the EDN graph."""
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
import iatc_json
import mark3_iatc_loop as loop
import stage_accounting as accounting


def node(kind="claim", text="x", lo=10, hi=10, citation=""):
    return {"kind": kind, "text": text, "citation": citation, "first_line": lo, "last_line": hi}


def step(premises, conclusion, relation="implies", kind="stated", warrant="by the lemma", lo=10, hi=11):
    return {"relation": relation, "premises": premises, "conclusion": conclusion,
            "warrant_kind": kind, "warrant": warrant, "first_line": lo, "last_line": hi}


CAND = {"paper-id": "1111.0001", "proof-id": "1111.0001__p0", "passage-id": "1111.0001:proof0:L10-14",
        "window-lines": [10, 14], "proof-lines": [12, 14], "schema": loop.CANDIDATE_SCHEMA,
        "proved": {"kind": "lemma", "lines": [10, 11], "text": "\\begin{lemma} A iff B \\end{lemma}"},
        "source-window": "a\nb\nc\nd\ne", "binder-context": [], "enrichment": []}


class Contract(unittest.TestCase):
    def test_schemas_bound_lines_vocabularies_and_node_references(self):
        line = iatc_json.nodes_schema(10, 14)["properties"]["nodes"]["items"]["properties"]["first_line"]
        self.assertEqual((line["minimum"], line["maximum"]), (10, 14))
        step = iatc_json.steps_schema(10, 14, 3)["properties"]["steps"]["items"]["properties"]
        self.assertIn("iff", step["relation"]["enum"])
        # the second call knows how many nodes exist, so a step cannot cite node 4 of 3
        self.assertEqual((step["conclusion"]["minimum"], step["conclusion"]["maximum"]), (1, 3))
        self.assertEqual(step["premises"]["items"]["maximum"], 3)

    def test_code_checks_what_the_schema_cannot(self):
        good = {"nodes": [node(text="A"), node(text="B"), node(text="C")],
                "steps": [step([1], 2), step([2], 3)]}
        self.assertEqual(iatc_json.problems(good, 10, 14), [])
        cycle = {"nodes": [node(text="A"), node(text="B")], "steps": [step([1], 2), step([2], 1)]}
        self.assertTrue(any("EARLIER" in p or "only concluded by step" in p for p in iatc_json.problems(cycle, 10, 14)))
        out_of_order = {"nodes": [node(), node(), node()], "steps": [step([2], 3), step([1], 2)]}
        self.assertTrue(iatc_json.problems(out_of_order, 10, 14))
        for bad, text in (({"nodes": [node(), node()], "steps": [step([5], 2)]}, "refers to node"),
                          ({"nodes": [node(), node()], "steps": [step([2], 2)]}, "both premise and conclusion"),
                          ({"nodes": [node(lo=12, hi=11), node()], "steps": [step([1], 2)]}, "ordered range"),
                          ({"nodes": [node(), node()], "steps": [step([1], 2, lo=9)]}, "inside 10-14"),
                          ("not json object", "not an object")):
            self.assertTrue(any(text in p for p in iatc_json.problems(bad, 10, 14)), (bad, text))
        iff = {"nodes": [node(text="A"), node(text="B")], "steps": [step([1], 2, relation="iff")]}
        self.assertEqual(iatc_json.problems(iff, 10, 14), [])
        # a construction step establishes the object it builds (34 such steps in the
        # 98-graph corpus), and a proof establishes the paper's own labelled statement
        construction = {"nodes": [node(text="hypotheses"), node(kind="object", text="the product P"),
                                  node(kind="ref", text="the theorem", citation="Theorem~\\ref{main}")],
                        "steps": [step([1], 2, relation="by-construction"), step([2], 3)]}
        self.assertEqual(iatc_json.problems(construction, 10, 14), [])

    def test_code_written_edn_passes_gates_and_reads_back(self):
        doc = {"nodes": [node(text='A with \\otimes and "quotes"', lo=12, hi=12),
                         node(kind="ref", text="Lemma 3 of [AR]", citation="[AR, 2.36]", lo=12, hi=12),
                         node(kind="ref", text="a known fact", lo=13, hi=13),
                         node(text="B holds", lo=14, hi=14)],
                "steps": [step([1, 2, 3], 4, kind="missing", warrant="accessibility of the embedding", lo=12, hi=14)]}
        self.assertEqual(iatc_json.problems(doc, 10, 14), [])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "1111.0001__p0.edn"
            path.write_text(iatc_json.to_edn(doc, CAND, "model-x"))
            check = subprocess.run(["bb", str(ROOT / "scripts/iatc_argcheck.bb"), str(path)], capture_output=True, text=True)
            self.assertEqual(check.returncode, 0, check.stdout + check.stderr)
            substance = subprocess.run([sys.executable, str(ROOT / "scripts/substance_gate.py"), str(path), "--kind", "iatc"],
                                       capture_output=True, text=True)
            self.assertEqual(substance.returncode, 0, substance.stdout + substance.stderr)
            import r2d_concept_coverage as r2d
            graph = r2d.load_edn(path)
        self.assertEqual(graph["nodes"][0]["text"], 'A with \\otimes and "quotes"')
        self.assertEqual(graph["edges"][0]["premise"], [":n1", ":n2", ":n3"])
        self.assertEqual(graph["source"], {"lines": [10, 14], "kind": ":proof"})
        wanted = {h.get("wanted") for h in graph["holes"]}
        self.assertIn(":accessibility-of-the-embedding", wanted)
        self.assertIn(":n3", {h.get("node") for h in graph["holes"]})


class ProofCandidates(unittest.TestCase):
    def test_candidates_are_outermost_s1_proofs_with_their_statement(self):
        import mark3_extract_candidates as extract
        text = "\n".join(["intro", "\\begin{lemma}A\\end{lemma}", "\\begin{proof}", "step",
                          "\\begin{proof}inner\\end{proof}", "\\end{proof}", "it is easy to see prose", "end"]) + "\n"
        starts = extract.line_starts(text)
        pos = lambda line: starts[line - 1]
        marks = [{"kind": "env/lemma", "start": pos(2), "end": pos(3) - 1},
                 {"kind": "env/proof", "start": pos(3), "end": pos(7) - 1},
                 {"kind": "env/proof", "start": pos(5), "end": pos(6) - 1},
                 {"kind": "proof-move", "start": pos(7), "end": pos(7) + 5, "tip": "easy"}]
        with tempfile.TemporaryDirectory() as d:
            Path(d, "fable-1111.0001-dp-emacs.json").write_text(json.dumps({"text": text, "marks": marks}))
            with patch.object(extract, "MARKS_DIR", Path(d)):
                cands = extract.extract_all("1111.0001")
        self.assertEqual(len(cands), 1)
        c = cands[0]
        self.assertEqual((c["proof-lines"], c["window-lines"], c["schema"]), ([3, 6], [2, 6], loop.CANDIDATE_SCHEMA))
        self.assertEqual(c["proved"]["kind"], "lemma")
        self.assertNotIn("prose", c["source-window"])


class Loop(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.cands = self.base / "candidates"
        self.cands.mkdir()
        for i in range(3):
            c = dict(CAND, **{"proof-id": f"1111.0001__p{i}", "passage-id": f"1111.0001:proof{i}:L10-14"})
            (self.cands / f"1111.0001__p{i}.candidate.json").write_text(json.dumps(c))
        self.out = self.base / "graphs"
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {"RUN_ID": "r", "FUTON6_RUN_DIR": str(self.base)}).start()
        patch.object(loop, "run_rung2", lambda _g, report, gate: (report.write_text("{:pass true}"), (True, "rung2-pass"))[1]).start()

    def responses(self, by_proof):
        """Serve each proof's phase answers in order; a bare value answers both."""
        def call(_prompt, cand, _model, _schema):
            answer = by_proof[cand["proof-id"]]
            if isinstance(answer, Exception):
                raise answer
            return answer.pop(0) if isinstance(answer, list) else answer
        return call

    def invoke(self, invocation, by_proof):
        adir = self.base / "accounting" / invocation
        args = argparse.Namespace(candidates=str(self.cands), out=str(self.out), backend="openai",
                                  model="m", rung2_gate=False, loss_log_interval=0)
        with patch.object(loop, "call_openai", self.responses(by_proof)), \
                patch.dict(os.environ, {accounting.DIR_ENV: str(adir), accounting.INVOCATION_ENV: invocation}):
            rc = loop.run(args)
        return rc, {e["id"]: e for e in accounting.load(adir, "S3", "loop")["items"]}

    def doc(self, text):
        """The two phase answers for one proof, in call order."""
        return [json.dumps({"nodes": [node(text=f"{text} hypothesis", lo=12, hi=12),
                                      node(text=f"{text} result", lo=14, hi=14)]}),
                json.dumps({"steps": [step([1], 2, warrant=f"{text} argument", lo=12, hi=14)]})]

    def test_rejected_errored_and_accepted_items_then_retry_only_the_failures(self):
        cycle = [json.dumps({"nodes": [node(text="A"), node(text="B")]}),
                 json.dumps({"steps": [step([1], 2), step([2], 1)]})]
        rc, items = self.invoke("S3-a001", {"1111.0001__p0": self.doc("first"), "1111.0001__p1": cycle,
                                            "1111.0001__p2": loop.ModelCallError(0, "output truncated at max_tokens=8192")})
        self.assertEqual(rc, 1)
        self.assertEqual({k: v["status"] for k, v in items.items()},
                         {"1111.0001__p0": "accepted", "1111.0001__p1": "rejected", "1111.0001__p2": "errored"})
        self.assertIn("contract:", items["1111.0001__p1"]["reason"])
        self.assertIn("truncated", items["1111.0001__p2"]["reason"])
        self.assertTrue((self.out / ".attempts/r/S3-a001/1111.0001__p1.nodes.json").is_file())
        self.assertTrue((self.out / ".attempts/r/S3-a001/1111.0001__p1.steps.json").is_file())
        self.assertFalse((self.out / "1111.0001__p1.edn").exists())

        calls = []
        answers = {"1111.0001__p1": self.doc("second"), "1111.0001__p2": self.doc("third")}
        def record(prompt, cand, model, schema):
            calls.append(cand["proof-id"])
            return answers[cand["proof-id"]].pop(0)
        adir = self.base / "accounting" / "S3-a002"
        args = argparse.Namespace(candidates=str(self.cands), out=str(self.out), backend="openai",
                                  model="m", rung2_gate=False, loss_log_interval=0)
        with patch.object(loop, "call_openai", record), \
                patch.dict(os.environ, {accounting.DIR_ENV: str(adir), accounting.INVOCATION_ENV: "S3-a002"}):
            loop.run(args)
        items = {e["id"]: e for e in accounting.load(adir, "S3", "loop")["items"]}
        self.assertEqual(sorted(set(calls)), ["1111.0001__p1", "1111.0001__p2"])  # accepted item not resampled
        self.assertEqual({v["status"] for v in items.values()}, {"accepted"})
        self.assertEqual(items["1111.0001__p0"]["attempts"][0]["carried-from"], "S3-a001")

    def test_final_without_provenance_is_errored_and_legacy_candidates_refused(self):
        self.out.mkdir()
        (self.out / "1111.0001__p0.edn").write_text("{}")
        _, items = self.invoke("S3-a001", {f"1111.0001__p{i}": self.doc(str(i)) for i in range(3)})
        self.assertEqual(items["1111.0001__p0"]["status"], "errored")
        (self.cands / "old.candidate.json").write_text(json.dumps({"schema": "iatc-candidate/v2-enriched"}))
        self.assertFalse(loop.require_candidates(sorted(self.cands.glob("*.candidate.json"))))

    def test_requests_schema_at_temperature_zero(self):
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
            loop.call_openai("prompt", CAND, "m", iatc_json.nodes_schema(10, 14))
        self.assertEqual(seen["temperature"], 0)
        self.assertEqual(seen["response_format"]["json_schema"]["schema"], iatc_json.nodes_schema(10, 14))


if __name__ == "__main__":
    unittest.main()
