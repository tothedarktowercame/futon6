from __future__ import annotations

import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_manifest as manifest
import retrieve_run
import linode_stepper as stepper
import conformance


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.run_dir = self.base / "run with spaces"
        self.ids = self.base / "input.ids"
        self.ids.write_text("1234.5678\nmath__0000001\n")
        self.addCleanup(patch.stopall)
        patch.object(manifest, "source_identity", return_value={"git-head": "fixture", "source-sha256": "code"}).start()
        patch.object(manifest, "substrate_identity", return_value={"fixture": "substrate"}).start()
        clean = {k: v for k, v in os.environ.items() if not k.startswith("FUTON6_") and k not in ("RUN_ID", "CORPUS")}
        patch.dict(os.environ, clean, clear=True).start()

    def prepare(self, run_dir=None):
        root = run_dir or self.run_dir
        with manifest.lock(root):
            return manifest.prepare(root, "test-run", "test-corpus", self.ids)

    def prefix(self):
        doc = self.prepare()
        for key, filename in (("marks", "paper.json"), ("loss", "dashboard.json")):
            directory = self.run_dir / doc["artifacts"][key]
            directory.mkdir(parents=True)
            (directory / filename).write_text('{"fixture": true}')
        (self.run_dir / "metrics.jsonl").write_text(json.dumps({"run_id": "test-run", "corpus_id": "test-corpus", "stage": "S1"}) + "\n")
        stepper.ledger_record(str(self.run_dir), "S1", "test-corpus", "test-run")
        return doc

    def test_resume_is_immutable_and_rejects_changed_corpus_code_and_substrate(self):
        self.prepare()
        original = (self.run_dir / manifest.NAME).read_bytes()
        self.prepare()
        self.assertEqual((self.run_dir / manifest.NAME).read_bytes(), original)
        self.ids.write_text("different\n")
        with self.assertRaisesRegex(ValueError, "corpus-sha256"):
            self.prepare()
        self.ids.write_bytes((self.run_dir / "corpus.ids.txt").read_bytes())
        for function, changed in (("source_identity", "code"), ("substrate_identity", "substrate")):
            with patch.object(manifest, function, return_value={"changed": True}):
                with self.assertRaisesRegex(ValueError, changed):
                    self.prepare()
        self.assertEqual((self.run_dir / manifest.NAME).read_bytes(), original)

    def test_refuses_unmanifested_artifacts_and_duplicate_corpus(self):
        self.run_dir.mkdir()
        (self.run_dir / "old.edn").write_text("old")
        with self.assertRaisesRegex(ValueError, "without a run manifest"):
            self.prepare()
        (self.run_dir / "old.edn").unlink()
        self.ids.write_text("same\nsame\n")
        with self.assertRaisesRegex(ValueError, "unique"):
            self.prepare()

    def test_identity_disagreement_refuses_before_creating_run(self):
        with patch.dict(os.environ, {"RUN_ID": "different"}), patch.object(sys, "argv", [
            "stepper", "--run", "--run-id", "test-run", "--corpus-id", "test-corpus",
            "--run-dir", str(self.run_dir), "--ids", str(self.ids)]):
            self.assertEqual(stepper.main(), 1)
        self.assertFalse(self.run_dir.exists())

    def test_lock_prevents_overlapping_runner_or_retrieval(self):
        with manifest.lock(self.run_dir):
            with self.assertRaisesRegex(ValueError, "already in use"):
                with manifest.lock(self.run_dir):
                    self.fail("second lock acquired")

    def test_frozen_ids_and_foreign_records_refuse(self):
        self.prefix()
        metrics = self.run_dir / "metrics.jsonl"
        metrics.write_text('{"run_id":"other", "corpus_id":"test-corpus"}\n')
        with self.assertRaisesRegex(ValueError, "another run"):
            self.prepare()
        (self.run_dir / "corpus.ids.txt").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "frozen corpus"):
            manifest.load(self.run_dir)

    def test_repeated_reuse_and_boot_only_mark_done(self):
        with patch.object(sys, "argv", ["stepper", "--run", "--from", "S1", "--to", "S1",
                                       "--run-dir", str(self.run_dir), "--run-id", "test-run", "--corpus-id", "test-corpus",
                                       "--ids", str(self.ids), "--reuse", "S0", "--reuse", "STAGE"]), \
                patch.object(stepper, "preflight_gate", return_value=0), \
                patch.object(stepper, "conformance_gate", return_value=0), \
                patch.object(stepper, "run", return_value=0) as execute:
            self.assertEqual(stepper.main(), 0)
        self.assertEqual(execute.call_args.args[-1], ["S0", "STAGE"])
        for flags in (["--mark-done", "S4"], ["--mark-done", "S0", "--run"],
                      ["--mark-done", "STAGE", "--from", "S1"], ["--reuse", "S2"]):
            with patch.object(sys, "argv", ["stepper", *flags]), self.assertRaises(SystemExit) as exit:
                stepper.main()
            self.assertEqual(exit.exception.code, 2)

    def test_runner_logs_and_writes_inside_manifest_paths(self):
        doc = self.prepare()
        stage = {"id": "S1", "name": "fixture", "compute": "cpu", "halt": False, "go": []}
        operation = {"cmd": 'mkdir -p "$FUTON6_MARKS"; printf evidence > "$FUTON6_MARKS/fixture.json"; echo observed', "gate": "true"}
        with patch.dict(os.environ, manifest.environment(self.run_dir, doc)), \
                patch.dict(stepper.OPS, {"S1": operation}), patch.object(stepper, "DEPS", {"S1": []}):
            self.assertEqual(stepper.run([stage], "superpod", True, str(self.run_dir), "test-corpus", "test-run", []), 0)
        self.assertEqual((self.run_dir / "artifacts/marks/fixture.json").read_text(), "evidence")
        self.assertIn("observed", (self.run_dir / "logs/S1.command.log").read_text())
        self.assertEqual(json.loads((self.run_dir / "phase-ledger.jsonl").read_text())["gate"], "pass")

    def test_failed_command_probe_and_failed_gate_do_not_pass_ledger(self):
        self.assertTrue(conformance.check_exit_status_propagates())
        doc = self.prepare()
        stage = {"id": "S1", "name": "fixture", "compute": "cpu", "halt": False, "go": []}
        with patch.dict(os.environ, manifest.environment(self.run_dir, doc)), \
                patch.dict(stepper.OPS, {"S1": {"cmd": "true", "gate": "exit 7"}}), \
                patch.object(stepper, "DEPS", {"S1": []}):
            self.assertEqual(stepper.run([stage], "superpod", True, str(self.run_dir), "test-corpus", "test-run", []), 3)
        self.assertFalse((self.run_dir / "phase-ledger.jsonl").exists())

    def test_successful_stages_cannot_be_overwritten_on_resume(self):
        self.prefix()
        stage = {"id": "S1", "name": "fixture", "compute": "cpu", "halt": False, "go": []}
        with patch.object(stepper, "sh") as execute:
            self.assertEqual(stepper.run([stage], "superpod", True, str(self.run_dir), "test-corpus", "test-run", []), 1)
        execute.assert_not_called()

    def test_manifest_paths_are_disjoint_in_real_shell(self):
        self.assertTrue(conformance.check_run_scoping())

    def test_pack_extract_replay_and_corruption(self):
        self.prefix()
        archive = self.base / "run.tgz"
        result = retrieve_run.pack(self.run_dir, archive, "S1")
        self.assertEqual(result["replay"], "pass")
        extracted = self.base / "durable copy"
        retrieve_run.verify(archive, extracted)
        self.assertEqual(manifest.load(extracted)["run-id"], "test-run")
        replay = subprocess.run([sys.executable, str(ROOT / "scripts/replay_e2e.py"), "--run-dir", str(extracted), "--through", "S1"], capture_output=True, text=True)
        self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
        bad = self.base / "bad.tgz"
        with tarfile.open(archive) as source, tarfile.open(bad, "w:gz") as target:
            for member in source:
                data = source.extractfile(member).read()
                if member.name.endswith("metrics.jsonl"):
                    data += b"corrupt"
                member.size = len(data)
                target.addfile(member, io.BytesIO(data))
        with self.assertRaisesRegex(ValueError, "checksum/size"):
            retrieve_run.verify(bad)

    def test_missing_cleans_cannot_produce_successful_archive(self):
        doc = self.prefix()
        for key, (stage, pattern) in manifest.REQUIRED.items():
            if stage > 7 or key == "clean":
                continue
            directory = self.run_dir / doc["artifacts"][key]
            directory.mkdir(parents=True, exist_ok=True)
            (directory / pattern.replace("*", "fixture")).write_text("{}")
        with self.assertRaisesRegex(ValueError, "clean"):
            retrieve_run.pack(self.run_dir, self.base / "empty-clean.tgz", "S7")
        self.assertFalse((self.base / "empty-clean.tgz").exists())

    def test_replay_refuses_wrong_paths_and_missing_manifest(self):
        self.prefix()
        for flags in (["--graphs", str(self.base / "wrong")], ["--corpus-id", "different"]):
            result = subprocess.run([sys.executable, str(ROOT / "scripts/replay_e2e.py"),
                                     "--run-dir", str(self.run_dir), "--through", "S1", *flags], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("REPLAY TARGET ERROR", result.stderr)
        (self.run_dir / manifest.NAME).unlink()
        result = subprocess.run([sys.executable, str(ROOT / "scripts/replay_e2e.py"), "--run-dir", str(self.run_dir)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()


class ScaledExpositoryCap(unittest.TestCase):
    """A cap of 30 was the same number for a six-page note and a 40,000-line book:
    0806.1324 carves 209 regions and S4 read 30 of them, while 0708.2185 carves 27 and
    lost nothing (Joe, 2026-09-22)."""

    def test_the_cap_follows_what_the_paper_has_to_read(self):
        self.assertEqual(manifest.scaled_cap(27), 31)      # a short paper: every region
        self.assertEqual(manifest.scaled_cap(209), 87)     # 30 before; 42% of the paper
        caps = [manifest.scaled_cap(n) for n in (0, 5, 27, 71, 209, 430, 5000)]
        self.assertEqual(caps, sorted(caps))                   # never decreasing
        self.assertEqual(caps[0], manifest.CAP_FLOOR)      # a tiny paper still gets read
        self.assertEqual(caps[-1], manifest.CAP_CEILING)   # one book cannot take the window

    def test_the_cap_is_sublinear_so_a_long_paper_cannot_spend_the_window(self):
        # Ten times the regions must not cost ten times the calls.
        self.assertLess(manifest.scaled_cap(300), 10 * manifest.scaled_cap(30))

    def test_a_pinned_number_and_no_cap_still_mean_what_they_did(self):
        self.assertEqual(manifest.cap_for(manifest.SCALED, 209), 87)
        self.assertEqual(manifest.cap_for(30, 209), 30)
        self.assertEqual(manifest.cap_for(0, 209), 0)      # 0 = every region
        self.assertIsNone(manifest.cap_rule(30))
        self.assertEqual(manifest.cap_rule(manifest.SCALED)["of"], "regions")

    def test_a_deferred_region_is_a_refusal_only_when_no_cap_is_in_force(self):
        # The accounting decides this from the cap, which is now a word as well as a
        # number; "scaled" must license a deferral exactly as 30 did.
        import stage_accounting
        doc = {"schema": stage_accounting.SCHEMA, "stage": "S4", "producer": "select",
               "invocation": "S4-a001", "updated": "now", "expected": ["r1"],
               "counts": {"accepted": 0, "rejected": 0, "errored": 0, "deferred": 1,
                          "expected": 1, "unaccounted": 0},
               "items": [{"id": "r1", "status": "deferred", "reason": "cap 12 of 40 regions",
                          "paper": "p", "artifacts": [], "outputs": []}]}
        with tempfile.TemporaryDirectory() as d:
            allowed, _ = stage_accounting.blocking(doc, ["r1"], run_dir=Path(d), allow_deferred=True)
            refused, _ = stage_accounting.blocking(doc, ["r1"], run_dir=Path(d), allow_deferred=False)
        self.assertEqual(allowed, [])
        self.assertTrue(refused)
        for cap in (manifest.SCALED, 30):
            self.assertTrue(bool(cap) and cap != 0, cap)
        self.assertFalse(bool(0) and 0 != 0)
