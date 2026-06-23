#!/usr/bin/env python3
"""Linode pipeline stepper — supervised, gated, resumable runner.

Reads the stage contract (holes/linode-stepper-contract.md, embedded EDN) and
drives the stages in order: precondition (inputs present) -> command -> postcondition
GATE -> per-stage report -> HALT for inspection at the contract's halt points. The
executor IS the gate (mark3 lesson); host-only GPU/LLM stages are flagged and the
run stops there locally (run them on the Linode host, then resume with --from).

  --plan                 print the executable plan (the contract made runnable)
  --run [--from S6 --to S8] [--no-halt]    execute (default S1..S9)
  --on-host              permit host-only (GPU) stages to actually run

This is the supervised sibling of clean_pipeline.sh: gated, halting, resumable.
"""
import argparse
import re
import subprocess
import sys
import os
import edn_format as edn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACT = os.path.join(ROOT, "holes", "linode-stepper-contract.md")
PY = ".venv/bin/python"


def kw(x):
    s = str(x)
    return s[1:] if s.startswith(":") else s


def load_stages():
    txt = open(CONTRACT).read()
    block = re.search(r"```edn\n(.*?)\n```", txt, re.S).group(1)
    c = {kw(k): v for k, v in dict(edn.loads(block)).items()}
    out = []
    for s in c["stages"]:
        sd = {kw(k): v for k, v in dict(s).items()}
        out.append({"id": kw(sd["id"]), "name": str(sd["name"]),
                    "compute": kw(sd["compute"]), "halt": bool(sd.get("halt")),
                    "go": [kw(g) for g in sd.get("go-no-go", [])]})
    return out


# dispatch: stage-id -> what to actually run. local stages have a cmd (+ optional
# gate); host-only stages carry the command to run on the Linode GPU host.
# host commands follow futon0/README-linode.md (StackScript 2142757 box).
# dispatch re-synced to the CORRECTED DAG (linode-stepper-contract.md, 2026-06-23):
# S1 anatomy · S2 concepts · S3 IATC(all-proofs) ∥ S4 expository · S5 comprehension ·
# S6 paper-graph(B) · S7 CLean-embed · S8 export · S9 APM/mining. Emitting stages take
# --run-dir $RUN so the run produces the slope report (INSTANTIATE-GPU). After S8:
#   {PY} scripts/metric_harness.py --from-records $RUN   (emitted completeness/quality)
#   {PY} scripts/metric_harness.py                       (accretion slopes, leave-one-out)
IDS = "holes/math-ct-200.ids.txt"
RUN = "data/runs/$RUN_ID"           # set per run; stages emit MetricRecords here
CAND = "data/iatc-candidates-run"
GRAPHS = "data/iatc-argument-graphs/run"
CLEAN = "holes/clean-run"
DEMO = "data/showcases/clean-run-demo"
OPS = {
    "S0": {"local": False, "note": "README-linode: StackScript 2142757; linode-postsetup-deps.sh; "
           "hf pre-pull 70B (retry loop); linode-4gpu-setup.sh (serve vLLM, flashinfer off)"},
    "S1": {"local": False, "note": f"{{PY}} scripts/emit_marks.py --list {IDS} --run-dir {RUN} "
           "(whole-paper anatomy from pre-fetched eprints; emits any-markup-coverage)",
           "gate": "check_invariants wf=0 across the batch"},
    "S2": {"local": False, "note": "warp substrate build (concordance) -> concept-index (corpus-fresh); "
           "G-coverage INLINE via coverage_inline.py on raw S1 concepts",
           "gate": "G-coverage: raw coverage rises with corpus-fraction"},
    "S3": {"local": False, "note": f"{{PY}} scripts/mark3_extract_candidates.py --papers <ids> --all-proofs "
           f"--out {CAND}  ->  CANDIDATES={CAND} OUT={GRAPHS} scripts/linode-4gpu-run.sh "
           "(mark3_iatc_loop 70B; one graph per proof)",
           "gate": f"bb scripts/iatc_argcheck.bb {GRAPHS} && {{PY}} scripts/substance_gate.py {GRAPHS} (finals-only)"},
    "S4": {"local": False, "note": "{PY} scripts/mark3_extract_expository_candidates.py --papers <ids> "
           "--out data/expository-candidates-run  ->  {PY} scripts/mark3_expository_loop.py "
           "--candidates data/expository-candidates-run --out data/expository-scope-graphs/run "
           f"--backend openai --run-dir {RUN} (ALL regions; cap/sample at scale)",
           "gate": "expository_argcheck (self-gated in loop)"},
    "S5": {"local": True, "inputs": [GRAPHS],
           "cmd": "{PY} scripts/clean_comprehension.py  # rung-ladder + R2d + strategy_recognizer",
           "gate": "G-comprehension: verdict separates weak-extraction from weak-proof"},
    "S6": {"local": True, "inputs": [GRAPHS],
           "cmd": f"{{PY}} scripts/paper_graph_assemble.py --paper <pid> --iatc {GRAPHS} --run-dir {RUN}",
           "gate": "B wellformed: every proof attaches to a statement; orphans flagged"},
    "S7": {"local": False, "note": f"{{PY}} scripts/clean_box_typing.py --graphs {GRAPHS} --out {CLEAN} "
           f"--endpoint http://localhost:$PORT/v1/chat/completions --model mark4-70b --run-dir {RUN}  ->  "
           f"{{PY}} scripts/clean_structure_embed.py --clean-dir {CLEAN} --out {DEMO}",
           "gate": f"bb scripts/clean_vocab_gate.bb {CLEAN} && {{PY}} scripts/clean_entropy_gate.py "
           f"--embed {DEMO}/clean-embed.json"},
    "S8": {"local": True, "inputs": [CLEAN],
           "cmd": f"{{PY}} scripts/clean_graph_export.py --clean-dir {CLEAN} --out {DEMO}/ingest "
           f"--embed-json {DEMO}/clean-embed.json"},
    "S9": {"local": True, "inputs": [GRAPHS],
           "cmd": "{PY} scripts/mark4_apm_structure_coverage.py ; {PY} scripts/clean_hole_harvest.py  # optional CPU tails"},
}


def sh(cmd):
    return subprocess.run(cmd, shell=True, cwd=ROOT).returncode


def order(stages, frm, to):
    ids = [s["id"] for s in stages]
    i0 = ids.index(frm) if frm else 0
    i1 = ids.index(to) + 1 if to else len(ids)
    return stages[i0:i1]


def plan(stages):
    print("LINODE STEPPER — executable plan\n")
    for s in stages:
        op = OPS.get(s["id"], {})
        loc = "local" if op.get("local") else "HOST-ONLY"
        print(f"{s['id']} {s['name']:22s} [{s['compute']:8s} {loc:9s}] "
              f"{'⏸HALT' if s['halt'] else '     '}  go/no-go: {','.join(s['go']) or '—'}")
        if op.get("cmd"):
            print(f"     cmd : {op['cmd'].format(PY=PY)}")
        if op.get("gate"):
            print(f"     gate: {op['gate'].format(PY=PY)}")
        if op.get("note"):
            print(f"     note: {op['note']}")


def run(stages, no_halt, on_host):
    for s in stages:
        op = OPS.get(s["id"], {})
        print(f"\n=== {s['id']} {s['name']} [{s['compute']}] ===")
        if not op.get("local") and not on_host:
            print(f"⏸ HOST-ONLY — run on the Linode GPU host, then resume with "
                  f"--from {s['id']}")
            print(f"   {op.get('note','')}")
            return
        # precondition: inputs present
        missing = [p for p in op.get("inputs", []) if not os.path.exists(os.path.join(ROOT, p))]
        if missing:
            print(f"✗ precondition FAILED — missing input(s): {missing}")
            return
        # command
        if op.get("cmd"):
            print(f"$ {op['cmd'].format(PY=PY)}")
            if sh(op["cmd"].format(PY=PY)) != 0:
                print(f"✗ {s['id']} command FAILED — stopping")
                return
        # postcondition gate
        if op.get("gate"):
            print(f"[gate] {op['gate'].format(PY=PY)}")
            if sh(op["gate"].format(PY=PY)) != 0:
                print(f"✗ {s['id']} GATE FAILED ({','.join(s['go'])}) — stopping for fix")
                return
        print(f"✓ {s['id']} done")
        if s["halt"] and not no_halt:
            print(f"⏸ HALT — inspect {s['id']} output; resume with "
                  f"--from {stages[stages.index(s)+1]['id'] if stages.index(s)+1 < len(stages) else '(done)'}")
            return
    print("\n✓ run complete")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--from", dest="frm", default=None)
    ap.add_argument("--to", default=None)
    ap.add_argument("--no-halt", action="store_true")
    ap.add_argument("--on-host", action="store_true")
    args = ap.parse_args()
    stages = load_stages()
    if args.plan or not args.run:
        plan(stages)
    if args.run:
        run(order(stages, args.frm, args.to), args.no_halt, args.on_host)


if __name__ == "__main__":
    main()
