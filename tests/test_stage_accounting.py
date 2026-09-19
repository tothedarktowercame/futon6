"""Stage 3: per-item accounting, retry history, selection, and false-valid refusal."""
from __future__ import annotations

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
        allowed = accounting.problems(ledger.document(), ["c", "b", "a"], run_dir=self.base, allow_deferred=True)
        self.assertFalse(any("declared inputs" in p for p in allowed))       # order does not matter
        self.assertFalse(any("deferred" in p for p in allowed))

    def test_publish_writes_provenance_before_the_final(self):
        final = self.base / "q.edn"
        with patch("os.replace", side_effect=OSError("killed before the final was written")):
            with self.assertRaises(OSError):
                accounting.publish_accepted(self.base, "q", final, b"{:ok true}", {"path": "a"})
        self.assertFalse(final.exists())                    # no final without provenance
        accounting.publish_accepted(self.base, "q", final, b"{:ok true}", {"path": "a"})
        self.assertIsNotNone(accounting.carried_acceptance(self.base, "q", final)[0])

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


class StageFixture:
    """One fixture S6 stage, run through the real stepper, for the cases below."""

    papers = ("1111.0001", "2222.0002")

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.run_dir = self.base / "run"
        self.ids = self.base / "ids"
        self.ids.write_text("".join(f"{paper}\n" for paper in self.papers))
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
                  "l = a.Accounting('S6', 'assemble', %r); "
                  % (str(ROOT / "scripts"), list(self.papers)))
        for item, status in statuses.items():
            script += f"l.record({item!r}, {status!r}, 'fixture reason' if {status!r} != 'accepted' else '', outputs=[{item!r}]); "
        command = f"{sys.executable} -c \"{script}\""
        with patch.dict(stepper.OPS, {"S6": {"cmd": command}}), patch.object(stepper, "DEPS", {"S6": []}):
            return stepper.run([{"id": "S6", "name": "fixture", "compute": "cpu", "halt": False, "go": []}],
                               "superpod", True, str(self.run_dir), "c", "r", [])

    def attempts(self):
        return [json.loads(l) for l in (self.run_dir / stepper.ATTEMPTS).read_text().splitlines()]


class RefusalsDoNotStopTheRun(StageFixture, unittest.TestCase):
    """A mining run continues past items the contract refused, and says so.

    Halting a whole run on per-item findings cost a 124-paper window over three
    proofs the S3 contract called circular; the refusals were the run's output,
    not its failure. What still stops a stage is accounting that cannot describe
    the corpus, or a yield collapse that says the contract itself is wrong.
    """
    papers = ("1111.0001", "2222.0002", "3333.0003", "4444.0004")

    def test_refusals_within_the_floor_pass_and_are_recorded_everywhere(self):
        rc = self.stage({"1111.0001": "accepted", "2222.0002": "accepted",
                         "3333.0003": "accepted", "4444.0004": "rejected"})
        self.assertEqual(rc, 0)                                  # 3/4 = the 0.75 floor
        row = self.attempts()[0]
        self.assertEqual(row["outcome"], "pass-with-refusals")
        self.assertEqual(row["problems"], [])
        self.assertTrue(any("fixture reason" in r for r in row["refused"]))
        entry = stepper.ledger_entry(str(self.run_dir), "S6", "c")
        self.assertEqual(entry["counts"]["assemble"]["rejected"], 1)
        self.assertTrue(entry["refused"])                        # the ledger says the corpus shrank
        self.assertEqual(accounting.stage_problems(self.run_dir, "S6", "S6-a001", "c")[0], [])

    def test_a_yield_collapse_still_stops_and_names_the_floor(self):
        rc = self.stage({"1111.0001": "accepted", "2222.0002": "accepted",
                         "3333.0003": "rejected", "4444.0004": "rejected"})
        self.assertEqual(rc, 3)
        self.assertFalse(stepper.ledger_has(str(self.run_dir), "S6", "c"))
        self.assertIn("below this run's floor", " ".join(self.attempts()[0]["problems"]))

    def test_an_unaccounted_item_stops_however_good_the_yield(self):
        # Everything recorded was accepted, so the yield is 100% - but a paper the
        # producer never mentioned is the mark6 loss, not a finding about the data.
        rc = self.stage({"1111.0001": "accepted", "2222.0002": "accepted", "3333.0003": "accepted"})
        self.assertEqual(rc, 3)
        self.assertIn("unaccounted", " ".join(self.attempts()[0]["problems"]))

    def test_the_floor_is_the_one_the_manifest_pinned(self):
        with patch.dict(os.environ, {"FUTON6_ITEM_FLOOR": "1.0"}):
            floor_run = self.base / "strict"
            with manifest.lock(floor_run):
                doc = manifest.prepare(floor_run, "r2", "c2", self.ids)
        self.assertEqual(doc["acceptance"]["item-floor"], 1.0)
        self.assertEqual(manifest.item_floor(doc), 1.0)
        # A manifest written before the floor existed is judged by the default,
        # so a run halted by a few refusals resumes instead of restarting.
        self.assertEqual(manifest.item_floor({"run-id": "old"}), manifest.DEFAULT_ITEM_FLOOR)


class StepperAttempts(StageFixture, unittest.TestCase):
    """Attempt history, resume, and refusal of a stage that cannot show its work."""

    def test_collapsed_stage_fails_with_evidence_then_retry_passes_with_history(self):
        self.assertEqual(self.stage({"1111.0001": "accepted", "2222.0002": "rejected"}), 3)
        self.assertFalse(stepper.ledger_has(str(self.run_dir), "S6", "c"))
        first = self.attempts()[0]
        self.assertEqual((first["invocation"], first["outcome"], first["command_rc"]), ("S6-a001", "rejected", 0))
        self.assertEqual(first["accounting"]["assemble"]["rejected"], 1)
        # The refusal is evidence about a paper; the reason to stop is the yield.
        self.assertTrue(any("fixture reason" in r for r in first["refused"]))
        self.assertTrue(any("below this run's floor" in p for p in first["problems"]))

        self.assertEqual(self.stage({"1111.0001": "accepted"}), 3)          # unaccounted paper
        self.assertIn("unaccounted", " ".join(self.attempts()[1]["problems"]))

        self.assertEqual(self.stage({"1111.0001": "accepted", "2222.0002": "accepted"}), 0)
        rows = self.attempts()
        self.assertEqual([r["invocation"] for r in rows], ["S6-a001", "S6-a002", "S6-a003"])
        self.assertEqual(stepper.ledger_entry(str(self.run_dir), "S6", "c")["invocation"], "S6-a003")
        # Evidence of the failed attempts is still on disk.
        self.assertTrue((self.run_dir / "accounting/S6/S6-a001/S6.assemble.json").is_file())

    def test_killed_invocation_is_recorded_as_interrupted_and_numbering_continues(self):
        killed = accounting.Accounting("S6", "assemble", ["1111.0001", "2222.0002"],
                                       self.run_dir / "accounting/S6/S6-a001")
        killed.record("1111.0001", "accepted", outputs=["1111.0001"])       # checkpoint, then SIGKILL
        self.assertEqual(stepper.next_invocation(str(self.run_dir), "S6", "r", "c"), "S6-a002")
        row = self.attempts()[0]
        self.assertEqual((row["invocation"], row["outcome"]), ("S6-a001", "interrupted"))
        self.assertEqual(row["accounting"]["assemble"]["unaccounted"], 1)
        self.assertEqual(self.stage({"1111.0001": "accepted", "2222.0002": "accepted"}), 0)
        self.assertEqual([r["invocation"] for r in self.attempts()], ["S6-a001", "S6-a002"])

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

    def test_g7_rejection_is_recorded_per_item_not_counted_as_clean(self):
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
        # The typing ran and its output is consistent, so it exits 0; the cyclic proof
        # is a rejected item, and whether S7 passes is the runner's call against the
        # run's floor. What must never happen is the proof going missing quietly.
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
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

    def test_inline_map_premise_is_rejected(self):
        rc, out = self.gate("{:id :e1 :kind :infer :relation :because :premise [{:kind :claim :text \"x\"}] "
                            ":conclusion :b :warrant {:kind :claim :text \"w\"} :source {:lines [1 2]}}")
        self.assertEqual(rc, 1, out)
        self.assertIn("inline maps", out)


class WarrantVocabularyArtifact(unittest.TestCase):
    def test_no_holes_still_writes_an_explicit_empty_vocabulary(self):
        with tempfile.TemporaryDirectory() as d:
            graphs = Path(d) / "graphs"
            graphs.mkdir()
            (graphs / "1111.0001__p0.edn").write_text(
                '{:paper/id "1111.0001" :nodes [{:id :n1 :kind :claim :text "A"}] '
                ':edges [{:id :e1 :kind :infer :premise [:n1] :conclusion :n1 '
                ':warrant {:kind :claim :text "stated in the proof"}}] :holes []}')
            out = Path(d) / "hole-vocabulary.json"
            run = subprocess.run([sys.executable, str(ROOT / "scripts/warrant_normalize.py"),
                                  "--graphs", str(graphs), "--out", str(out)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertTrue(out.is_file(), "S9 passed without writing its artifact; replay fails two stages later")
            self.assertEqual(json.loads(out.read_text())["wanted"], 0)
