"""Configuration must select the same resources in checks and actual consumers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import futon6_config as config


class ConfigurationTests(unittest.TestCase):
    def test_override_precedence_and_checkout_independence(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "renamed checkout"
            root.mkdir()
            with patch.object(config, "ROOT", root), patch.dict(os.environ, {
                "FUTON_CODE_ROOT": str(Path(d) / "siblings"),
                "FUTON3_ROOT": "../patterns elsewhere",
                "FUTON6_EPRINTS": "sources",
                "FUTON6_BACKGROUND_CORPUS_INDEX": "authority.json",
            }, clear=True):
                self.assertEqual(config.sibling("futon3"), Path(d) / "patterns elsewhere")
                self.assertEqual(config.sibling("futon3c"), Path(d) / "siblings/futon3c")
                self.assertEqual(config.eprints(), root / "sources")
                self.assertEqual(config.authority(), root / "authority.json")
                self.assertEqual(config.ROOT, root)

    def test_explicit_missing_eprints_never_fall_back(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            populated = root / "storage/futon6/data/arxiv-math-ct-eprints"
            populated.mkdir(parents=True)
            (populated / "paper.tex").write_text("paper")
            with patch.object(config, "ROOT", root / "checkout"), patch.dict(os.environ, {
                "FUTON_CODE_ROOT": d,
            }, clear=True):
                self.assertEqual(config.eprints(), populated)
                os.environ["FUTON6_EPRINTS"] = "absent"
                self.assertEqual(config.eprints(), root / "checkout/absent")

    def test_interpreter_quotes_flags_and_venv_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            interpreter = Path(d) / "python with spaces"
            interpreter.symlink_to(sys.executable)
            argv = [str(interpreter), "-u", "-X", "utf8"]
            with patch.dict(os.environ, {"FUTON6_PYTHON_CMD": shlex.join(argv)}):
                self.assertEqual(config.python_argv(), argv)
                self.assertEqual(shlex.split(config.python_command()), argv)
                self.assertEqual(json.loads(config.child_environment()["FUTON6_PYTHON_ARGV_JSON"]), argv)
            for command in ("", "/no/such/python -u"):
                with patch.dict(os.environ, {"FUTON6_PYTHON_CMD": command}):
                    with self.assertRaises(ValueError):
                        config.python_argv()

    def test_babashka_uses_selected_interpreter_and_flags(self):
        if not shutil.which("bb"):
            self.skipTest("bb is required for the cross-language subprocess check")
        with tempfile.TemporaryDirectory() as d:
            wrapper = Path(d) / "python recorder"
            log = Path(d) / "arguments.json"
            wrapper.write_text("#!" + sys.executable + "\nimport json, os, sys\n"
                               "open(os.environ['ARGV_LOG'], 'w').write(json.dumps(sys.argv[1:]))\n"
                               "print('{:check :concept-coverage :pass true}')\n")
            wrapper.chmod(0o755)
            with patch.dict(os.environ, {"FUTON6_PYTHON_CMD": shlex.join([str(wrapper), "-u", "-X", "utf8"])}):
                env = config.child_environment()
            env["ARGV_LOG"] = str(log)
            result = subprocess.run(["bb", "-e", '(load-file "scripts/iatc_semcheck.bb") '
                                     '(println (concept-check-file "some graph.edn"))'],
                                    cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(log.read_text()),
                             ["-u", "-X", "utf8", "scripts/r2d_concept_coverage.py", "--edn", "some graph.edn"])

    def test_endpoint_identity_and_public_record(self):
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "http://user:secret@host:9000/v1/",
                                     "MODEL": "served-model", "OPENAI_API_KEY": "secret-key"}):
            effective = config.effective()
            self.assertEqual(effective["endpoint"], "http://host:9000/v1")
            self.assertEqual(effective["model"], "served-model")
            self.assertNotIn("secret", json.dumps(effective))
            self.assertEqual(config.child_environment()["OPENAI_API_KEY"], "secret-key")

    def test_s3_wrapper_uses_checkout_interpreter_and_remote_endpoint(self):
        with tempfile.TemporaryDirectory() as d:
            checkout = Path(d) / "renamed checkout"
            scripts = checkout / "scripts"
            scripts.mkdir(parents=True)
            for name in ("linode-4gpu-run.sh", "futon6_config.py"):
                shutil.copyfile(ROOT / "scripts" / name, scripts / name)
            candidates = checkout / "candidates"
            candidates.mkdir()
            (candidates / "paper.candidate.json").write_text(json.dumps({"schema": "iatc-candidate/v3-proof"}))
            # Only the model boundary is stubbed. Run the real shell wrapper,
            # its candidate validation, interpreter selection and environment.
            (scripts / "mark3_iatc_loop.py").write_text(
                "import json, os, sys\nfrom pathlib import Path\n"
                "Path('observed.json').write_text(json.dumps({"
                "'cwd': str(Path.cwd()), 'argv': sys.argv, 'python': sys.executable, "
                "'flags': sys._xoptions, 'endpoint': os.environ['OPENAI_BASE_URL'], "
                "'key': os.environ['OPENAI_API_KEY']}))\n")
            binaries = Path(d) / "bin"
            binaries.mkdir()
            curl = binaries / "curl"
            curl.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >> "$CURL_LOG"\nexit 0\n')
            curl.chmod(0o755)
            interpreter = binaries / "python with spaces"
            interpreter.symlink_to(sys.executable)
            env = dict(os.environ)
            for key in ("REPO", "FUTON6_PYTHON_ARGV_JSON"):
                env.pop(key, None)
            env.update(FUTON6_PYTHON_CMD=shlex.join([str(interpreter), "-u", "-X", "utf8"]),
                       FUTON6_PYTHON=sys.executable, CANDIDATES=str(candidates),
                       OPENAI_BASE_URL="http://configured-host:9123/v1", OPENAI_API_KEY="test-key",
                       MODEL="configured-model", RUN_EVAL="0", CURL_LOG=str(Path(d) / "curl.log"),
                       PATH=str(binaries) + os.pathsep + env["PATH"])
            result = subprocess.run(["bash", str(scripts / "linode-4gpu-run.sh")], cwd=d,
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            observed = json.loads((checkout / "observed.json").read_text())
            self.assertEqual(observed["cwd"], str(checkout))
            self.assertEqual(observed["python"], str(interpreter))
            self.assertEqual(observed["flags"].get("utf8"), True)
            self.assertEqual(observed["endpoint"], env["OPENAI_BASE_URL"])
            self.assertEqual(observed["key"], "test-key")
            self.assertEqual(observed["argv"][-1], "configured-model")
            self.assertIn(env["OPENAI_BASE_URL"] + "/models", (Path(d) / "curl.log").read_text())

    def test_relocated_readers_and_preflight_use_same_paths(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            checkout = base / "renamed checkout"
            shutil.copytree(ROOT / "scripts", checkout / "scripts",
                            ignore=shutil.ignore_patterns("__pycache__", "fold_embed"))
            (checkout / "holes").mkdir()
            for contract in ("superpod-dag-contract.md", "linode-stepper-contract.md"):
                shutil.copyfile(ROOT / "holes" / contract, checkout / "holes" / contract)
            eprints = base / "external sources"
            eprints.mkdir()
            (eprints / "test.tex").write_text("paper")
            siblings = base / "siblings"
            patterns = base / "configured patterns"
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith(("FUTON", "NLAB_", "NNEXUS_", "PLANETMATH_", "MATHLIB4_"))}
            env.update(FUTON6_EPRINTS=str(eprints), FUTON_CODE_ROOT=str(siblings),
                       FUTON3_ROOT=str(patterns), FUTON6_ANATOMY=str(base / "anatomy"))
            probe = '''import json, sys
sys.path.insert(0, 'scripts')
import futon6_config as config, preflight, anatomy_v0_sweep, warp_bib, warp_run
import dp_paper_view, cas_select, check_invariants, log_loss
from pathlib import Path
assert config.ROOT == Path.cwd()
assert config.eprints() == anatomy_v0_sweep.DEFAULT_EPRINTS == warp_bib.DEFAULT_EPRINTS
assert config.eprints() == dp_paper_view.EPRINTS == warp_run.EPRINTS
assert preflight.resolve_eprints()[0] == str(config.eprints())
assert cas_select.FUTON3 == config.sibling('futon3')
assert check_invariants.ROOT == log_loss.ROOT == config.ROOT
assert warp_run.ANATOMY == config.anatomy()
print(json.dumps(config.effective()))
'''
            result = subprocess.run([sys.executable, "-c", probe], cwd=checkout,
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            effective = json.loads(result.stdout)
            self.assertEqual(effective["eprints"], str(eprints))
            self.assertEqual(effective["siblings"]["futon3"], str(patterns))
            self.assertFalse((siblings / "futon6").exists())
            plan = subprocess.run([sys.executable, "scripts/linode_stepper.py", "--plan"],
                                  cwd=checkout, env=env, capture_output=True, text=True)
            self.assertEqual(plan.returncode, 0, plan.stderr)
            self.assertFalse((checkout / ".venv").exists())
            self.assertIn(str(checkout), plan.stdout)

    def test_run_records_configuration_and_threads_it_to_preflight(self):
        import linode_stepper as stepper
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "source.ids"
            source.write_text("1234.5678\n")
            run_dir = Path(d) / "run"
            with patch.dict(os.environ, {"RUN_ID": "config-test", "CORPUS": "config-corpus"}), \
                    patch.object(sys, "argv", ["stepper", "--run", "--run-dir", str(run_dir),
                                               "--ids", str(source), "--run-id", "config-test", "--corpus-id", "config-corpus"]), \
                    patch.object(stepper, "load_stages", return_value=[]), \
                    patch.object(stepper.manifest, "substrate_identity", return_value={"fixture": "hash"}), \
                    patch.object(stepper.subprocess, "run", return_value=SimpleNamespace(
                        returncode=1, stdout="test refusal", stderr="")) as launch:
                self.assertEqual(stepper.main(), 1)
            record = json.loads((run_dir / "host-config.jsonl").read_text())
            self.assertEqual(record["run-id"], "config-test")
            self.assertEqual(record["configuration"]["checkout"], str(ROOT))
            sent = launch.call_args.kwargs["env"]
            self.assertEqual(json.loads(sent["FUTON6_PYTHON_ARGV_JSON"]), record["configuration"]["python-argv"])
            self.assertEqual(sent["FUTON6_EPRINTS"], record["configuration"]["eprints"])
            self.assertNotIn("OPENAI_API_KEY", record["configuration"])


if __name__ == "__main__":
    unittest.main()
