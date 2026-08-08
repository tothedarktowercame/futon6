"""The stepper's process exit status must reflect the stage outcome.

Regression test for the release blocker found in the 2026-08-07 review: every
failure path in `run()` printed and returned None, and `main()` exited 0
regardless. An outer scheduler therefore recorded a successful process while the
stepper reported that it had stopped — on an unattended cluster window, the
difference between "the run finished" and "the run stopped four hours in and
nobody noticed".

Exit codes: 0 completed or deliberately halted · 1 refused/blocked ·
2 stage command failed · 3 stage gate failed.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "bin", "python")
if not os.path.exists(PY):
    PY = sys.executable
STEPPER = os.path.join(ROOT, "scripts", "linode_stepper.py")


def _run(args, env=None):
    e = dict(os.environ)
    e.setdefault("RUN_ID", "exitstatus")
    e.setdefault("CORPUS", "exit-status-test")
    if env:
        e.update(env)
    return subprocess.run([PY, STEPPER] + args, cwd=ROOT, capture_output=True,
                          text=True, env=e, timeout=180)


def _stepper():
    import importlib.util
    spec = importlib.util.spec_from_file_location("linode_stepper", STEPPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_preflight_gate_refuses_before_any_stage():
    """No dependency, no run. The gate is mandatory and has no override."""
    with tempfile.TemporaryDirectory() as d:
        p = _run(["--run", "--profile", "superpod", "--from", "S1", "--to", "S1",
                  "--run-dir", d, "--corpus-id", "x", "--run-id", "x", "--no-halt"],
                 env={"FUTON6_EPRINTS": "/nonexistent"})
    out = p.stdout + p.stderr
    assert "REFUSING TO START" in out, out[-400:]
    assert p.returncode != 0, "a refused run must not report success"


def test_blocked_stage_exits_nonzero():
    """A stage whose upstream has no ledger entry must fail the process.

    Exercised in-process: `main()` now runs the mandatory preflight first, which
    on a dev box refuses before this path is ever reached. The gate belongs to
    main(), the blocking contract belongs to run(), and testing the second
    through the first would only ever re-test the gate.
    """
    mod = _stepper()
    with tempfile.TemporaryDirectory() as d:
        stages = [{"id": "S5", "name": "comprehension", "compute": "cpu",
                   "deps": ["S2", "S3"], "cmd": "true", "halt": False, "go": []}]
        rc = mod.run(stages, "superpod", True, d, "no-such-corpus", "exitstatus", [])
    assert rc != 0, "a blocked stage must not report success"


def test_plan_exits_zero():
    """Planning is not running; it must stay a success."""
    p = _run(["--plan", "--profile", "superpod"])
    assert p.returncode == 0, p.stdout + p.stderr


def test_stage_command_failure_exits_nonzero(tmp_path=None):
    """A failing stage command must fail the process AND write no ledger entry."""
    import json
    with tempfile.TemporaryDirectory() as d:
        # S1 with an id manifest that does not exist -> emit_marks fails.
        p = _run(["--run", "--profile", "superpod", "--from", "S1", "--to", "S1",
                  "--reuse", "S0", "STAGE", "--ids", "holes/__does_not_exist__.txt",
                  "--run-dir", d, "--corpus-id", "exit-status-test",
                  "--run-id", "exitstatus", "--no-halt"])
        assert p.returncode != 0, "a failed stage command must not report success"
        ledger = os.path.join(d, "phase-ledger.jsonl")
        rows = []
        if os.path.exists(ledger):
            rows = [json.loads(l) for l in open(ledger)]
        assert not any(r.get("stage") == "S1" and r.get("gate") == "pass" for r in rows), \
            "a failed stage must not leave a passing ledger record"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"  FAIL {name}: {e}")
    print(f"\n{'all pass' if not fails else str(fails) + ' failing'}")
    sys.exit(fails)
