"""Stage 3: per-item accounting, retry history, selection, and false-valid refusal."""
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
import run_manifest as manifest
import stage_accounting as accounting
import linode_stepper as stepper
import mark3_iatc_loop as iatc_loop
import mark3_extract_expository_candidates as expo_extract

GRAPH = """{:paper/id "9999.0001"
 :passage/id "9999.0001:p"
 :nodes [{:id :p :kind :claim :text "premise" :source {:lines [1 1]}}
         {:id :c :kind :claim :text "conclusion" :source {:lines [2 2]}}]
 :edges [{:id :e :kind :infer :premise [:p] :conclusion :c
          :warrant {:kind :missing-warrant :wanted :lemma}
          :source {:lines [1 2]}}]
 :holes [{:kind :missing-warrant :edge :e :wanted :lemma}]}"""


class AccountingRules(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)

    def test_records_are_unique_explained_and_checkpointed(self):
        ledger = accounting.Accounting("S6", "assemble", ["a", "b"], self.base)
        ledger.record("a", "accepted")
        self.assertEqual(json.loads((self.base / "S6.assemble.json").read_text())["counts"]["accepted"], 1)
        with self.assertRaisesRegex(ValueError, "twice"):
            ledger.record("a", "accepted")
        with self.assertRaisesRegex(ValueError, "reason"):
            ledger.record("b", "rejected")
        with self.assertRaisesRegex(ValueError, "already written"):
            accounting.Accounting("S6", "assemble", ["a"], self.base)

    def test_problems_name_unaccounted_extra_rejected_deferred_and_missing_artifacts(self):
        ledger = accounting.Accounting("S4", "select", ["a", "b", "c"], self.base)
        ledger.record("a", "accepted", artifacts=["missing.json"])
        ledger.record("b", "deferred", "cap 1")
        ledger.record("x", "rejected", "G7: wire graph has a cycle")
        found = "\n".join(accounting.problems(ledger.document(), ["a", "b", "c"], run_dir=self.base))
        for text in ("1 unaccounted", "outside the inputs", "1 rejected", "G7", "1 deferred", "missing/empty artifact"):
            self.assertIn(text, found)
        allowed = accounting.problems(ledger.document(), ["a", "b", "c"], run_dir=self.base, allow_deferred=True)
        self.assertFalse(any("deferred" in p for p in allowed))

    def test_acceptance_provenance_refuses_stale_and_edited_finals(self):
        final = self.base / "p.edn"
        final.write_text("{}")
        self.assertIsNone(accounting.carried_acceptance(self.base, "p", final)[0])
        accounting.record_acceptance(self.base, "p", final, {"attempt": 1, "path": "a"})
        self.assertEqual(accounting.carried_acceptance(self.base, "p", final)[0]["attempt"], 1)
        (self.base / "stale.edn").write_text("{}")
        accepted, refused = accounting.accepted_finals(self.base)
        self.assertEqual([item for item, _ in accepted], ["p"])
        self.assertEqual([name for name, _ in refused], ["stale.edn"])
        final.write_text("{:edited true}")
        self.assertIn("differs", accounting.carried_acceptance(self.base, "p", final)[1])


class StepperAttempts(unittest.TestCase):
    """A rejected item fails the stage but leaves an attempt row; a retry adds history."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.run_dir = self.base / "run"
        self.ids = self.base / "ids"
        self.ids.write_text("1111.0001\n2222.0002\n")
        self.addCleanup(patch.stopall)
        patch.object(manifest, "source_identity", return_value={"code": "fixture"}).start()
        patch.object(manifest, "substrate_identity", return_value={"substrate": "fixture"}).start()
        clean = {k: v for k, v in os.environ.items() if not k.startswith("FUTON6_") and k not in ("RUN_ID", "CORPUS")}
        patch.dict(os.environ, clean, clear=True).start()
        with manifest.lock(self.run_dir):
            self.doc = manifest.prepare(self.run_dir, "r", "c", self.ids)
        patch.dict(os.environ, manifest.environment(self.run_dir, self.doc)).start()

    def stage(self, statuses):
        # Write S6 accounting from a real subprocess, the way producers do.
        script = ("import sys; sys.path.insert(0, %r); import stage_accounting as a; "
                  "l = a.Accounting('S6', 'assemble', ['1111.0001', '2222.0002']); "
                  % str(ROOT / "scripts"))
        for item, status in statuses.items():
            script += f"l.record({item!r}, {status!r}, 'fixture reason' if {status!r} != 'accepted' else '', outputs=[{item!r}]); "
        command = f"{sys.executable} -c \"{script}\""
        with patch.dict(stepper.OPS, {"S6": {"cmd": command}}), patch.object(stepper, "DEPS", {"S6": []}):
            return stepper.run([{"id": "S6", "name": "fixture", "compute": "cpu", "halt": False, "go": []}],
                               "superpod", True, str(self.run_dir), "c", "r", [])

    def attempts(self):
        return [json.loads(l) for l in (self.run_dir / stepper.ATTEMPTS).read_text().splitlines()]

    def test_rejection_fails_with_evidence_then_retry_passes_with_history(self):
        self.assertEqual(self.stage({"1111.0001": "accepted", "2222.0002": "rejected"}), 3)
        self.assertFalse(stepper.ledger_has(str(self.run_dir), "S6", "c"))
        first = self.attempts()[0]
        self.assertEqual((first["invocation"], first["outcome"], first["command_rc"]), ("S6-a001", "rejected", 0))
        self.assertEqual(first["accounting"]["assemble"]["rejected"], 1)
        self.assertTrue(any("fixture reason" in p for p in first["problems"]))

        self.assertEqual(self.stage({"1111.0001": "accepted"}), 3)          # unaccounted paper
        self.assertIn("unaccounted", " ".join(self.attempts()[1]["problems"]))

        self.assertEqual(self.stage({"1111.0001": "accepted", "2222.0002": "accepted"}), 0)
        rows = self.attempts()
        self.assertEqual([r["invocation"] for r in rows], ["S6-a001", "S6-a002", "S6-a003"])
        self.assertEqual(stepper.ledger_entry(str(self.run_dir), "S6", "c")["invocation"], "S6-a003")
        # Evidence of the failed attempts is still on disk.
        self.assertTrue((self.run_dir / "accounting/S6/S6-a001/S6.assemble.json").is_file())

    def test_missing_accounting_cannot_pass_even_with_zero_exit(self):
        with patch.dict(stepper.OPS, {"S6": {"cmd": "true"}}), patch.object(stepper, "DEPS", {"S6": []}):
            rc = stepper.run([{"id": "S6", "name": "fixture", "compute": "cpu", "halt": False, "go": []}],
                             "superpod", True, str(self.run_dir), "c", "r", [])
        self.assertEqual(rc, 3)
        self.assertIn("missing accounting", self.attempts()[0]["problems"][0])

    def test_downstream_expected_items_come_from_ledgered_upstream_accounting(self):
        upstream = self.run_dir / "accounting/S3/S3-a001"
        ledger = accounting.Accounting("S3", "loop", ["p0", "p1"], upstream)
        ledger.record("p0", "accepted", outputs=["p0"])
        ledger.record("p1", "rejected", "gate", outputs=[])
        stepper.ledger_record(str(self.run_dir), "S3", "c", "r", "S3-a001")
        typing = accounting.Accounting("S7", "typing", ["p0"], self.run_dir / "accounting/S7/S7-a001")
        typing.record("p0", "accepted", outputs=["p0"])
        self.assertEqual(stepper.accounting_problems(str(self.run_dir), "S7", "S7-a001", "c")[0], [])


class ExpositorySelection(unittest.TestCase):
    def candidates(self, n):
        return [{"paper-id": "1111.0001", "region-id": f"r{i}", "passage-id": f"1111.0001:r{i}",
                 "window-lines": [10 * (n - i), 10 * (n - i) + 3]} for i in range(n)]

    def test_even_spacing_in_source_order_is_deterministic(self):
        selected, deferred = expo_extract.select_even(self.candidates(10), 3)
        lines = [c["window-lines"][0] for c in selected]
        self.assertEqual(lines, [10, 40, 70])                 # source order, spread across the paper
        self.assertEqual(len(deferred), 7)
        self.assertEqual(expo_extract.select_even(self.candidates(10), 0)[1], [])
        self.assertEqual(len(expo_extract.select_even(self.candidates(2), 5)[0]), 2)

    def test_manifest_pins_cap_and_algorithm(self):
        with tempfile.TemporaryDirectory() as d, \
                patch.object(manifest, "source_identity", return_value={}), \
                patch.object(manifest, "substrate_identity", return_value={}), \
                patch.dict(os.environ, {"FUTON6_EXPOSITORY_CAP_PER_PAPER": "30"}):
            ids = Path(d) / "ids"
            ids.write_text("1111.0001\n")
            with manifest.lock(Path(d) / "run"):
                doc = manifest.prepare(Path(d) / "run", "r", "c", ids)
            self.assertEqual(doc["selection"]["expository-cap"], 30)
            self.assertEqual(doc["selection"]["expository-selection"], manifest.EXPOSITORY_SELECTION)
            with patch.dict(os.environ, {"FUTON6_EXPOSITORY_CAP_PER_PAPER": "20"}), manifest.lock(Path(d) / "run"):
                with self.assertRaisesRegex(ValueError, "selection"):
                    manifest.prepare(Path(d) / "run", "r", "c", ids)
            with patch.dict(os.environ, {"FUTON6_EXPOSITORY_CAP_PER_PAPER": "-1"}), manifest.lock(Path(d) / "other"):
                with self.assertRaisesRegex(ValueError, "nonnegative"):
                    manifest.prepare(Path(d) / "other", "r", "c", ids)


class IatcLoopRetry(unittest.TestCase):
    """A valid candidate survives another candidate's rejection; a retry keeps it."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.cands = self.base / "candidates"
        self.cands.mkdir()
        for pid in ("9999.0001__p0", "9999.0001__p1"):
            (self.cands / f"{pid}.candidate.json").write_text(json.dumps({
                "schema": "iatc-candidate/v2-enriched", "paper-id": "9999.0001", "proof-id": pid,
                "window-lines": [1, 2], "source-window": "1 premise\n2 conclusion",
                "binder-context": [], "enrichment": []}))
        self.out = self.base / "graphs"
        self.addCleanup(patch.stopall)
        patch.object(iatc_loop, "load_seeds", lambda _n: "").start()
        patch.object(iatc_loop, "call_stub", lambda _p, _c, _a: GRAPH).start()
        patch.object(iatc_loop, "run_rung2", lambda _g, report, gate: (report.write_text("{}"), (True, "rung2-pass"))[1]).start()
        patch.object(iatc_loop, "candidate_check", lambda _e, _c: (True, "ok")).start()
        patch.object(iatc_loop, "MAX_ATTEMPTS", 2).start()
        patch.dict(os.environ, {"RUN_ID": "r", "FUTON6_RUN_DIR": str(self.base)}).start()

    def invoke(self, invocation, reject):
        gate = lambda path: (False, "argcheck: dangling") if reject in path.name else (True, "ok")
        adir = self.base / "accounting" / invocation
        args = argparse.Namespace(candidates=str(self.cands), out=str(self.out), backend="stub",
                                  model="stub", shots=0, rung2_gate=False)
        with patch.object(iatc_loop, "gate_one", gate), \
                patch.dict(os.environ, {accounting.DIR_ENV: str(adir), accounting.INVOCATION_ENV: invocation}):
            rc = iatc_loop.run(args)
        return rc, accounting.load(adir, "S3", "loop")

    def test_rejected_item_keeps_history_and_is_retried_without_resampling_accepted(self):
        rc, doc = self.invoke("S3-a001", reject="__p1")
        self.assertEqual(rc, 1)
        by = {e["id"]: e for e in doc["items"]}
        self.assertEqual(by["9999.0001__p0"]["status"], "accepted")
        self.assertEqual(by["9999.0001__p1"]["status"], "rejected")
        self.assertEqual(len(by["9999.0001__p1"]["attempts"]), 2)
        self.assertFalse((self.out / "9999.0001__p1.edn").exists())

        rc, doc = self.invoke("S3-a002", reject="nothing")
        by = {e["id"]: e for e in doc["items"]}
        self.assertEqual(by["9999.0001__p0"]["attempts"][0]["carried-from"], "S3-a001")
        self.assertEqual(by["9999.0001__p1"]["status"], "accepted")
        # Both invocations' attempt files remain.
        self.assertTrue((self.out / ".attempts/r/S3-a001/9999.0001__p1.attempt1.edn").is_file())
        self.assertTrue((self.out / ".attempts/r/S3-a002/9999.0001__p1.attempt0.edn").is_file())

    def test_final_without_provenance_is_errored_not_resumed(self):
        self.out.mkdir()
        (self.out / "9999.0001__p0.edn").write_text(GRAPH)
        rc, doc = self.invoke("S3-a001", reject="nothing")
        self.assertEqual(rc, 1)
        by = {e["id"]: e for e in doc["items"]}
        self.assertEqual(by["9999.0001__p0"]["status"], "errored")
        self.assertIn("provenance", by["9999.0001__p0"]["reason"])


class PaperGraphsAndCleans(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)

    def marks(self, paper, text, marks):
        directory = self.base / "marks"
        directory.mkdir(exist_ok=True)
        (directory / f"fable-{paper}-dp-emacs.json").write_text(json.dumps({"text": text, "marks": marks}))
        return directory

    def test_malformed_paper_does_not_stop_later_papers(self):
        text = "Proof. early\n\\begin{theorem}T\\end{theorem}\n\\begin{proof}P\\end{proof}\n"
        marks = self.marks("1111.0001", text, [{"kind": "env/proof", "start": 0, "end": 12}])
        self.marks("2222.0002", text, [{"kind": "env/theorem", "start": 13, "end": 40},
                                       {"kind": "env/proof", "start": 41, "end": 70}])
        ids = self.base / "ids"
        ids.write_text("1111.0001\n2222.0002\n")
        adir = self.base / "accounting"
        result = subprocess.run([sys.executable, str(ROOT / "scripts/paper_graph_assemble.py"), "--list", str(ids),
                                 "--marks-dir", str(marks), "--out", str(self.base / "B")],
                                capture_output=True, text=True, env={**os.environ, accounting.DIR_ENV: str(adir)})
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        doc = accounting.load(adir, "S6", "assemble")
        self.assertEqual({e["id"]: e["status"] for e in doc["items"]},
                         {"1111.0001": "rejected", "2222.0002": "accepted"})
        self.assertTrue((self.base / "B/2222.0002.B.json").is_file())
        self.assertTrue((self.base / "B/1111.0001.B.json").is_file())      # inspectable evidence

    def test_g7_rejection_fails_s7_instead_of_counting_as_clean(self):
        graphs = self.base / "graphs"
        graphs.mkdir()
        cyclic = """{:paper/id "9999.0002" :nodes [{:id :a :kind :claim :text "A"} {:id :b :kind :claim :text "B"}]
 :edges [{:id :e1 :kind :infer :premise [:a] :conclusion :b :warrant {:kind :claim :text "w"}}
         {:id :e2 :kind :infer :premise [:b] :conclusion :a :warrant {:kind :claim :text "w"}}]}"""
        (graphs / "9999.0002__p0.edn").write_text(cyclic)
        (graphs / "9999.0003__p0.edn").write_text(GRAPH)
        adir = self.base / "accounting"
        result = subprocess.run([sys.executable, str(ROOT / "scripts/clean_box_typing.py"), "--graphs", str(graphs),
                                 "--out", str(self.base / "clean"), "--stub"],
                                cwd=ROOT, capture_output=True, text=True,
                                env={**os.environ, accounting.DIR_ENV: str(adir)})
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        by = {e["id"]: e for e in accounting.load(adir, "S7", "typing")["items"]}
        self.assertEqual(by["9999.0002__p0"]["status"], "rejected")
        self.assertIn("G7", by["9999.0002__p0"]["reason"])
        self.assertEqual(by["9999.0003__p0"]["status"], "accepted")


if __name__ == "__main__":
    unittest.main()


class ReplayAcceptance(unittest.TestCase):
    """Replay passes only when accounting shows every item accepted and graphs parse clean."""

    PROOF = """{:paper/id "1111.0001" :source {:lines [1 3] :kind :proof}
 :nodes [{:kind :claim :id :p :text "premise" :source {:lines [1 1]}}
         {:id :c :kind :claim :text "conclusion" :source {:lines [2 2]}}]
 :edges [{:id :e :kind :infer :premise [:p] :conclusion :c :source {:lines [1 2]}
          :warrant {:kind :claim :text "w"}}]}"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.run_dir = self.base / "run"
        ids = self.base / "ids"
        ids.write_text("1111.0001\n")
        self.addCleanup(patch.stopall)
        patch.object(manifest, "source_identity", return_value={}).start()
        patch.object(manifest, "substrate_identity", return_value={}).start()
        clean = {k: v for k, v in os.environ.items() if not k.startswith("FUTON6_") and k not in ("RUN_ID", "CORPUS")}
        patch.dict(os.environ, clean, clear=True).start()
        with manifest.lock(self.run_dir):
            self.doc = manifest.prepare(self.run_dir, "r", "c", ids)
        art = lambda key: self.run_dir / self.doc["artifacts"][key]
        for key, name in (("marks", "fable-1111.0001-dp-emacs.json"), ("loss", "dashboard.json"),
                          ("candidates", "1111.0001__p0.candidate.json")):
            art(key).mkdir(parents=True)
            (art(key) / name).write_text('{"fixture": true}')
        art("graphs").mkdir(parents=True)
        self.graph = art("graphs") / "1111.0001__p0.edn"
        self.graph.write_text(self.PROOF)
        (self.run_dir / "metrics.jsonl").write_text(json.dumps({"run_id": "r", "corpus_id": "c", "stage": "S1"}) + "\n")
        stepper.ledger_record(str(self.run_dir), "S1", "c", "r", "S1-a001")

    def s3(self, loop_status, reason=""):
        adir = self.run_dir / "accounting/S3/S3-a001"
        extract = accounting.Accounting("S3", "extract", ["1111.0001"], adir)
        extract.record("1111.0001", "accepted", paper="1111.0001", outputs=["1111.0001__p0"],
                       artifacts=["artifacts/candidates/1111.0001__p0.candidate.json"])
        loop = accounting.Accounting("S3", "loop", ["1111.0001__p0"], adir)
        loop.record("1111.0001__p0", loop_status, reason, paper="1111.0001", outputs=["1111.0001__p0"],
                    artifacts=["artifacts/graphs/1111.0001__p0.edn"] if loop_status == "accepted" else [])
        stepper.ledger_record(str(self.run_dir), "S3", "c", "r", "S3-a001")

    def replay(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/replay_e2e.py"), "--run-dir", str(self.run_dir),
                                 "--through", "S3"], capture_output=True, text=True)
        return result.returncode, result.stdout + result.stderr

    def test_fully_accepted_prefix_passes_with_key_order_independent_parsing(self):
        self.s3("accepted")
        rc, out = self.replay()
        self.assertEqual(rc, 0, out)
        self.assertIn("0/2 unresolved", out)

    def test_rejected_item_prevents_a_false_fully_valid_replay(self):
        # Even if a ledger row were forged over it, the accounting still shows the rejection.
        self.s3("rejected", "G7: wire graph has a cycle")
        rc, out = self.replay()
        self.assertNotEqual(rc, 0, out)
        self.assertIn("[FAIL] A1-item-accounting", out)
        self.assertIn("G7", out)

    def test_inline_premise_and_out_of_passage_anchor_fail_with_zero_tolerance(self):
        self.graph.write_text(self.PROOF.replace("[:p]", '[{:kind :claim :text "inline"}]')
                                        .replace("{:lines [2 2]}", "{:lines [9 9]}"))
        self.s3("accepted")
        rc, out = self.replay()
        self.assertNotEqual(rc, 0, out)
        self.assertIn("[FAIL] S2-refs-resolve", out)
        self.assertIn("[FAIL] S3-anchors-in-passage", out)


class AnatomyDetection(unittest.TestCase):
    """S1 causes of the malformed 0708.1921 / 0708.2185 paper objects."""

    def setUp(self):
        import dp_paper_view
        self.dpv = dp_paper_view

    def test_sentence_ending_proof_is_not_a_proof_heading(self):
        text = ("\\begin{document}\nWe will provide\nthe mi\\-ssing proof. Our argument is short.\n"
                "and this completes\nthe proof.\n\\frp\n"
                "\\noindent{\\bf Proof.} A real proof that runs long enough to count.\n\\qed\n"
                "\\textit{Proof.} Another proof, also long enough to be a region.\n\\qed\n")
        starts = [text[m.start():m.end()].strip() for m in self.dpv._TEXT_PROOF_START_RE.finditer(text)]
        self.assertEqual(len(starts), 2, starts)
        self.assertEqual(len(self.dpv.detect_text_proofs(text)), 2)

    def test_tac_let_aliases_open_statements_and_proofs(self):
        text = ("\\newtheorem{axiom}{Axiom}\n\\let\\thm\\theorem\n\\let\\lem\\lemma\n\\let\\eth\\endtheorem\n"
                "\\let\\prf\\proof\n\\let\\frp\\endproof\n\\begin{document}\n"
                "\\thm\\label{a} Statement A. \\eth\n\\prf Proof of A. \\frp\n"
                "\\lem Statement B. \\eth\n\\prf Proof of B. \\frp\n")
        kinds = [(m["kind"], text[m["start"]:m["start"] + 4]) for m in
                 sorted(self.dpv.detect_macro_environments(text, {}), key=lambda m: m["start"])]
        self.assertEqual(kinds, [("env/theorem", "\\thm"), ("env/proof", "\\prf"),
                                 ("env/lemma", "\\lem"), ("env/proof", "\\prf")])

    def test_newcommand_aliases_and_learned_theorem_titles(self):
        text = ("\\newtheorem{thrm}{Theorem}\n\\newtheorem{protodefinition}{Definition}\n"
                "\\newenvironment{prf}{\\noindent\\textbf{Proof: }}{$\\Box$}\n"
                "\\newcommand{\\thm}{\\begin{thrm}}\n\\newcommand{\\ethm}{\\end{thrm}}\n"
                "\\newcommand{\\pf}{\\begin{prf}}\n\\newcommand{\\epf}{\\end{prf}}\n\\begin{document}\n"
                "\\thm T \\ethm\n\\pf P \\epf\n\\begin{protodefinition} D \\end{protodefinition}\n")
        learned = self.dpv.learn_environment_names(text)
        self.assertEqual(learned, {"thrm": "theorem", "protodefinition": "definition", "prf": "proof"})
        self.assertEqual(sorted(m["kind"] for m in self.dpv.detect_macro_environments(text, learned)),
                         ["env/proof", "env/theorem"])
        body = text.index("\\begin{protodefinition}")
        self.assertEqual([m["kind"] for m in self.dpv.detect_tex_environments(text[body:], body, learned)],
                         ["env/definition"])


class InferenceGraphGate(unittest.TestCase):
    """S3 rejects graphs S7 cannot type, while the model can still repair them."""

    BASE = """{:paper/id "9999.0004" :passage/id "9999.0004:p"
 :nodes [{:id :a :kind :claim :text "A" :source {:lines [1 1]}}
         {:id :b :kind :claim :text "B" :source {:lines [2 2]}}]
 :edges [%s]}"""

    def gate(self, edges):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "9999.0004__p0.edn"
            path.write_text(self.BASE % edges)
            result = subprocess.run(["bb", str(ROOT / "scripts/iatc_argcheck.bb"), str(path)],
                                    capture_output=True, text=True)
        return result.returncode, result.stdout + result.stderr

    def edge(self, eid, premise, conclusion, relation=":implies"):
        return (f"{{:id {eid} :kind :infer :relation {relation} :premise [{premise}] :conclusion {conclusion} "
                f":warrant {{:kind :claim :text \"w\"}} :source {{:lines [1 2]}}}}")

    def test_equivalence_as_two_implications_is_rejected_single_iff_edge_passes(self):
        rc, out = self.gate(self.edge(":e1", ":a", ":b") + " " + self.edge(":e2", ":b", ":a"))
        self.assertEqual(rc, 1, out)
        self.assertIn("[inference-cycle]", out)
        self.assertIn(":relation :iff", out)
        rc, out = self.gate(self.edge(":e1", ":a", ":b", ":iff"))
        self.assertEqual(rc, 0, out)

    def test_infer_edge_without_conclusion_is_rejected(self):
        rc, out = self.gate("{:id :e1 :kind :infer :relation :because :premise [:a] "
                            ":warrant {:kind :claim :text \"w\"} :source {:lines [1 2]}}")
        self.assertEqual(rc, 1, out)
        self.assertIn("[infer-shape]", out)
