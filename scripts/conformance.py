#!/usr/bin/env python3
"""POST-PREFLIGHT CONFORMANCE: does this host behave the way the pipeline assumes?

Third gate in the sequence, and the one that answers a different question:

    preflight.py    is everything the run needs PRESENT?
    conformance.py  does this host BEHAVE as the pipeline assumes?      <- here
    replay_e2e.py   did the run produce sound artifacts?

Presence is not behaviour. A host can have every dependency installed, every
path resolvable and a model served, and still be a host on which this pipeline
produces confident nonsense — because the *serving stack* differs from the one
the code was written against. That is not hypothetical: every LLM stage here now
relies on the endpoint honouring `response_format: {"type": "json_schema"}`. On
llama.cpp it binds. On a stack that accepts the field and ignores it, nothing
errors; the model answers with its own key names, every lookup misses, and each
stage falls back to a deterministic template. The run completes, the gates pass,
and the output is a stub wearing the model's voice (see H28/H29).

So this exists to give the window's operator an early, cheap abort. It runs in
a couple of minutes against the real endpoint and the real stage machinery, and
every check is behavioural: something is made to happen and the result is
compared with what the contract says must happen. Three are NEGATIVE checks —
they assert that a thing correctly FAILS — because a gate that cannot fail is
the failure mode this project keeps rediscovering.

    python scripts/conformance.py --endpoint http://host:8000/v1 --model <name>
    python scripts/conformance.py ... --json report.json     # for the run record

Exit code is the number of failed checks, so `&&` and schedulers do the right
thing. A non-zero exit means STOP AND LOOK, not "proceed with caveats": the
whole point is that these failures are invisible downstream.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "bin", "python")
if not os.path.exists(PY):
    PY = sys.executable

R: list[tuple[str, bool, str, str]] = []


def rec(name, ok, detail, remedy=""):
    R.append((name, ok, detail, remedy))
    return ok


def _post(endpoint, payload, timeout=300):
    req = urllib.request.Request(
        f"{endpoint.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'x')}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ------------------------------------------------------------- LLM behaviour

def check_schema_binds(endpoint, model):
    """THE decisive check: does a JSON schema actually constrain generation?

    Asked under `enum: ["purple", "octagonal"]`, a conforming stack answers that
    a ripe banana is purple. An absurd answer is the *passing* result: it proves
    the grammar overrode the model's own knowledge. A sensible answer ("yellow")
    means the field was accepted and ignored, and every LLM stage downstream is
    silently running unconstrained.
    """
    schema = {"type": "object",
              "properties": {"colour": {"type": "string", "enum": ["purple", "octagonal"]}},
              "required": ["colour"], "additionalProperties": False}
    try:
        o = _post(endpoint, {"model": model, "temperature": 0, "max_tokens": 32,
                             "messages": [{"role": "user",
                                           "content": "What colour is a ripe banana? Answer as JSON."}],
                             "response_format": {"type": "json_schema", "json_schema": {
                                 "name": "c", "strict": True, "schema": schema}}})
    except Exception as e:  # noqa: BLE001
        return rec("llm:schema-binds", False, f"request failed ({type(e).__name__}: {e})",
                   "does this stack support response_format json_schema at all?")
    txt = (o.get("choices") or [{}])[0].get("message", {}).get("content", "")
    bound = "yellow" not in txt.lower() and ("purple" in txt.lower() or "octagonal" in txt.lower())
    return rec("llm:schema-binds", bound,
               "grammar overrides the model (answered outside its own knowledge)" if bound
               else f"schema NOT enforced — replied {txt[:60]!r}",
               "every LLM stage assumes enforced schemas; unenforced, they degrade to "
               "template output that looks like a result (H28). Use a stack with grammar "
               "support, or do not run the LLM stages here")


def check_schema_maxlength(endpoint, model):
    """`maxLength` must bound the *string*, not just the token budget.

    A stack that ignores it truncates mid-word at max_tokens instead, which
    yields unparseable JSON and silent template fallback (H33).
    """
    schema = {"type": "object",
              "properties": {"s": {"type": "string", "maxLength": 40}},
              "required": ["s"]}
    try:
        o = _post(endpoint, {"model": model, "temperature": 0, "max_tokens": 256,
                             "messages": [{"role": "user",
                                           "content": "Describe category theory at length, as JSON {\"s\": ...}"}],
                             "response_format": {"type": "json_schema", "json_schema": {
                                 "name": "m", "strict": True, "schema": schema}}})
    except Exception as e:  # noqa: BLE001
        return rec("llm:maxlength-binds", False, f"request failed ({type(e).__name__}: {e})", "")
    ch = (o.get("choices") or [{}])[0]
    txt = ch.get("message", {}).get("content", "")
    try:
        val = json.loads(txt).get("s", "")
        ok = len(val) <= 40 and ch.get("finish_reason") == "stop"
        detail = f"bounded at {len(val)} chars, finish={ch.get('finish_reason')}"
    except ValueError:
        ok, detail = False, f"reply did not parse ({txt[:50]!r})"
    return rec("llm:maxlength-binds", ok, detail,
               "without maxLength the model runs to max_tokens and truncates mid-string, "
               "which reads downstream as an unparseable reply (H33)")


def check_throughput(endpoint, model, tokens=128):
    """Measure decode rate, so window arithmetic is measured rather than assumed."""
    try:
        t = time.time()
        o = _post(endpoint, {"model": model, "temperature": 0, "max_tokens": tokens,
                             "messages": [{"role": "user", "content": "Count slowly from 1 to 60."}]})
        dt = time.time() - t
    except Exception as e:  # noqa: BLE001
        return rec("llm:throughput", False, f"request failed ({type(e).__name__}: {e})", "")
    n = (o.get("usage") or {}).get("completion_tokens") or 0
    rate = n / dt if dt else 0
    # 818 Tier-1 calls x ~120 completion tokens is the pilot corpus's cascade cost.
    est_h = (818 * 120 / rate / 3600) if rate else float("inf")
    return rec("llm:throughput", rate > 0,
               f"{rate:.1f} tok/s → the 818-call cascade would take ~{est_h:.1f} h at this rate",
               "" if rate > 20 else
               "slow enough that the LLM stages dominate the window; check batching/parallelism")


# --------------------------------------------------- stage-machinery behaviour

def _stepper():
    """Import the stepper, or return None if its deps are absent.

    conformance is otherwise stdlib-only and is documented as runnable
    standalone, but the stepper imports edn_format. On a host that has not run
    the setup yet -- exactly the host this gate exists to inspect -- that raised
    ModuleNotFoundError and conformance died with a traceback instead of a
    verdict. Demonstrated on linode-chicago, 2026-08-13: a clean checkout, no
    venv, and the operator gets a stack trace where the contract promises a
    named failure and a remedy.
    """
    import importlib.util
    try:
        spec = importlib.util.spec_from_file_location(
            "linode_stepper", os.path.join(ROOT, "scripts", "linode_stepper.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as exc:
        return exc


def check_gate_refuses_unavailable(exc):
    """Report an unimportable stepper as a NAMED failure, never a crash."""
    missing = getattr(exc, "name", None)
    return rec("gate:refuses-bad-input", False,
               f"cannot load the stepper to read its gate: {type(exc).__name__}"
               + (f" (missing module '{missing}')" if missing else ""),
               f"pip install {missing}" if missing
               else "run scripts/linode-postsetup-deps.sh, then re-run")


def check_gate_refuses():
    """NEGATIVE: S3's gate, handed a degenerate graph, must FAIL.

    Runs the gate the STEPPER declares, read from `OPS["S3"]["gate"]`, rather
    than a copy of it. The first version of this check ran `iatc_argcheck` alone
    and reported "the gate cannot fail" — argcheck is rung-0 (wiring
    well-formedness) and an empty graph is trivially well-wired; it is
    `substance_gate` at rung-1 that refuses degeneracy, and the chain does refuse.
    A conformance check that duplicates the pipeline's definition will drift from
    it and raise false alarms about it, so this one derives from it.

    The check matters because three shipped checks in this project could not
    fail (H22 unexecuted, H25 un-failable floor, H28 fabricated content), and a
    gate that certifies everything is indistinguishable from one that works
    until the corpus is wrong.
    """
    mod = _stepper()
    if isinstance(mod, Exception):
        return check_gate_refuses_unavailable(mod)
    gate = (mod.OPS.get("S3") or {}).get("gate")
    if not gate:
        return rec("gate:refuses-bad-input", False, "S3 declares no gate", "")
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "broken__p0.edn"), "w") as fh:
            fh.write("{:nodes [] :edges [] :holes []}\n")
        cmd = gate.format(PY=mod.PY, IDS=mod.IDS).replace(mod.GRAPHS, d)
        p = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True,
                           text=True, timeout=600)
    refused = p.returncode != 0
    return rec("gate:refuses-bad-input", refused,
               "S3's gate chain rejects a degenerate graph" if refused
               else "S3's gate chain ACCEPTED a graph with no nodes and no edges",
               "a gate that cannot fail certifies nothing; do not run with it")


def check_exit_status_propagates():
    """NEGATIVE: a failed stage must exit non-zero.

    An outer scheduler that sees 0 records a successful window while the run
    stopped hours earlier (the original release blocker).
    """
    with tempfile.TemporaryDirectory() as d:
        p = subprocess.run(
            [PY, os.path.join(ROOT, "scripts", "linode_stepper.py"), "--run",
             "--profile", "superpod", "--from", "S1", "--to", "S1", "--no-halt",
             "--ids", "holes/__does_not_exist__.txt", "--run-dir", d,
             "--corpus-id", "conformance", "--run-id", "conformance"],
            cwd=ROOT, capture_output=True, text=True, timeout=600)
    ok = p.returncode != 0
    return rec("stage:exit-status-propagates", ok,
               f"a failing stage exits {p.returncode}" if ok
               else "a FAILING stage exited 0 — a scheduler would record success",
               "without this an unattended window cannot distinguish finished from stopped")


def check_run_scoping():
    """Artifact paths must carry the run id rather than a shared directory.

    Shared artifact directories are how one corpus's graphs land in another
    corpus's counts (H35). The paths are interpolated by the SHELL each stage
    runs in, not by Python, so the test is that `$RUN_ID` appears in them — an
    earlier version of this check compared two `--plan` outputs under different
    run ids and called them identical, which they are and must be: Python never
    substitutes the variable.
    """
    mod = _stepper()
    consts = {"CAND": mod.CAND, "GRAPHS": mod.GRAPHS, "CLEAN": mod.CLEAN,
              "STEPS": mod.STEPS, "RUNG3": mod.RUNG3, "DEMO": mod.DEMO,
              "RUN": mod.RUN}
    if hasattr(mod, "EXPO"):
        consts["EXPO"] = mod.EXPO
    unscoped = sorted(k for k, v in consts.items() if "$RUN_ID" not in v)
    ok = not unscoped
    return rec("run:artifact-paths-scoped", ok,
               f"all {len(consts)} artifact roots carry $RUN_ID" if ok
               else f"shared across runs: {', '.join(unscoped)}",
               "shared artifact directories put one corpus's outputs in another's counts (H35)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=os.environ.get("OPENAI_BASE_URL"))
    ap.add_argument("--model", default=os.environ.get("MODEL", "mark4-70b"))
    ap.add_argument("--json", help="write the report here, for the run record")
    ap.add_argument("--skip-llm", action="store_true",
                    help="stage-machinery checks only (no endpoint required)")
    a = ap.parse_args()

    if not a.skip_llm:
        if not a.endpoint:
            rec("llm:endpoint", False, "no --endpoint given", "pass --endpoint or set OPENAI_BASE_URL")
        else:
            check_schema_binds(a.endpoint, a.model)
            check_schema_maxlength(a.endpoint, a.model)
            check_throughput(a.endpoint, a.model)
    check_gate_refuses()
    check_exit_status_propagates()
    check_run_scoping()

    w = max(len(n) for n, _, _, _ in R)
    bad = [x for x in R if not x[1]]
    print("conformance — does this host behave as the pipeline assumes?\n")
    for name, ok, detail, remedy in R:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {name:<{w}}  {detail}")
        if not ok and remedy:
            print(f"         -> {remedy}")
    print(f"\n{len(R) - len(bad)}/{len(R)} checks pass")
    if bad:
        print("\n  ABORT THE WINDOW. These failures are silent downstream: the run will\n"
              "  complete, the gates will pass, and the artifacts will be wrong in ways\n"
              "  no later check can see. Fix or drop the affected stages before starting.")
    else:
        print("\n  CONFORMS — the host behaves as the pipeline assumes. Safe to start.")

    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"checks": [{"name": n, "ok": o, "detail": d} for n, o, d, _ in R],
                       "passed": len(R) - len(bad), "total": len(R)}, fh, indent=2)
        print(f"  report: {a.json}")
    return len(bad)


if __name__ == "__main__":
    sys.exit(main())
