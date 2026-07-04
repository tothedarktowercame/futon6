#!/usr/bin/env python3
"""Pipeline stepper — supervised, gated, resumable runner (Linode box or superpod).

Reads the stage contract (holes/linode-stepper-contract.md, embedded EDN) and
drives the stages in order: precondition (inputs present) -> command -> postcondition
GATE -> per-stage report -> HALT for inspection at the contract's halt points. The
executor IS the gate (mark3 lesson).

SINGLE-HOST (corrected 2026-06-23): there is NO dev/box split. After the STAGE step
(rsync of eprints + the ~68MB substrate + futon3 patterns onto the host), EVERY stage
S1..S9 runs on the one host — box or superpod. The earlier "S2/S5 are dev-local" was a
data-staging gap mistaken for a distributed topology, not a real requirement. Only S0
(provision) and STAGE (rsync from dev) are from-dev bootstrap steps.

  --plan [--profile linode|superpod]   print the executable plan (the contract made runnable)
  --run [--from S6 --to S8] [--no-halt] [--run-dir .. --corpus-id ..]   execute (default S1..S9)

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
PY = ".venv/bin/python -u"   # -u: unbuffered → stage output streams live (no buffered black box)


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


# dispatch: stage-id -> the command to run ON THE HOST (box or superpod). After STAGE,
# the ENTIRE pipeline is single-host — no dev/box split (that was a staging gap, not a
# topology). S0 (provision) and STAGE (rsync from dev) are the only from-dev bootstrap
# steps; both are "boot" (note-only, halt). CORRECTED DAG (2026-06-23):
# S1 anatomy · S2 concepts · S3 IATC(all-proofs) ∥ S4 expository · S5 comprehension ·
# S6 paper-graph(B) · S7 CLean-embed · S8 export · S9 APM/mining. Emitting stages take
# --run-dir $RUN so the run produces the slope report. After S8:
#   {PY} scripts/metric_harness.py --from-records $RUN   (emitted completeness/quality)
#   {PY} scripts/metric_harness.py                       (accretion slopes, leave-one-out)
IDS = "holes/math-ct-200.ids.txt"
RUN = "data/runs/$RUN_ID"           # set per run; stages emit MetricRecords here
CAND = "data/iatc-candidates-run"
GRAPHS = "data/iatc-argument-graphs/run"
CLEAN = "holes/clean-run"
DEMO = "data/showcases/clean-run-demo"
OPS = {
    "S0": {"boot": True, "note": "<profile.s0> — provision the host + serve the model"},
    "STAGE": {"boot": True, "note": "<profile.stage> — rsync eprints + the ~68MB substrate + futon3 "
              "patterns onto the host. DEREFERENCE symlinks (rsync -L / tar -h): dev uses a storage/ "
              "overlay, so a naive copy ships dangling links and S2/S5 then can't read the substrate."},
    "S1": {"cmd": "{PY} scripts/emit_marks.py --list {IDS} --run-dir " + RUN + " --run-id $RUN_ID --corpus-id $CORPUS",
           "gate": "{PY} scripts/check_invariants.py --corpus",
           "crit": "wf=0 across the batch — read data/loss/dashboard.json at the halt"},
    "S2": {"cmd": "{PY} scripts/coverage_inline.py  # concept-substrate corpus-fresh; G-coverage inline",
           "crit": "G-coverage: raw coverage rises with corpus-fraction"},
    "S3": {"cmd": f"{{PY}} scripts/mark3_extract_candidates.py --list {{IDS}} --all-proofs --out {CAND} && "
           f"CANDIDATES={CAND} OUT={GRAPHS} bash scripts/linode-4gpu-run.sh",
           "gate": f"bb scripts/iatc_argcheck.bb {GRAPHS} && {{PY}} scripts/substance_gate.py {GRAPHS}",
           "note": "substance gate reads finals only; the run wrapper reuses the enriched "
                   "candidates S3 just extracted (no silent 10-paper re-extract)"},
    "S4": {"cmd": "{PY} scripts/mark3_extract_expository_candidates.py --list {IDS} "
           "--out data/expository-candidates-run && {PY} scripts/mark3_expository_loop.py "
           "--candidates data/expository-candidates-run --out data/expository-scope-graphs/run "
           f"--backend openai --run-dir {RUN}",
           "crit": "expository_argcheck (self-gated in loop)",
           "note": "ALL regions by default — cap/sample per paper at archive scale "
                   "(mark7 playbook: ~30 regions/paper so S4 doesn't dominate the window)"},
    "S5": {"cmd": f"{{PY}} scripts/clean_comprehension.py --graphs {GRAPHS} --candidates {CAND} --run-dir {RUN} "
           "# rung-ladder + R2d + strategy_recognizer (reads the staged substrate + futon3 patterns)",
           "crit": "G-comprehension: verdict separates weak-extraction from weak-proof"},
    "S6": {"cmd": "while read -r pid; do [ -n \"$pid\" ] || continue; "
           f"{{PY}} scripts/paper_graph_assemble.py --paper $pid --iatc {GRAPHS} --run-dir {RUN} "
           "|| exit 1; done < {IDS}",
           "crit": "B wellformed: every proof attaches to a statement; orphans flagged"},
    "S7": {"cmd": f"{{PY}} scripts/clean_box_typing.py --graphs {GRAPHS} --out {CLEAN} "
           f"--endpoint http://localhost:$PORT/v1/chat/completions --model mark4-70b --run-dir {RUN} && "
           f"{{PY}} scripts/clean_structure_embed.py --clean-dir {CLEAN} --out {DEMO}",
           "gate": f"bb scripts/clean_vocab_gate.bb {CLEAN} && {{PY}} scripts/clean_entropy_gate.py "
           f"--embed {DEMO}/clean-embed.json"},
    "S8": {"cmd": f"{{PY}} scripts/clean_graph_export.py --clean-dir {CLEAN} --out {DEMO}/ingest "
           f"--embed-json {DEMO}/clean-embed.json"},
    "S9": {"cmd": "{PY} scripts/mark4_apm_structure_coverage.py ; {PY} scripts/clean_hole_harvest.py  # optional CPU tails"},
    # --- LEARNING LAYER (the 'improve as we run' instrumentation; CPU post-stages) ---
    "S10": {"cmd": f"{{PY}} scripts/iatc_lexicon_harvest.py --graphs {GRAPHS} --run-dir {RUN} && "
            f"{{PY}} scripts/iatc_move_reground.py && {{PY}} scripts/expository_reground.py",
            "crit": "move-lexicon harvested (relations+warrants+expository moves); reground lift >= 0"},
    "S11": {"cmd": f"{{PY}} scripts/sfc_struct_canon.py --formulae {RUN}/def-formulae.txt ; "
            f"{{PY}} scripts/clean_paper_signature.py --embed {DEMO}/clean-embed.json",
            "crit": "structural canonical shapes + whole-paper signatures produced"},
    "S12": {"cmd": f"{{PY}} scripts/accretion_curves.py --graphs {GRAPHS} --candidates {CAND} --run-dir {RUN}",
            "crit": "ACCRETION SWEEP: every tier metric checkpointed at log-spaced n -> rising curves"},
    "RETRIEVE": {"boot": True, "halt": True, "note": "<profile.retrieve> — pull ALL run outputs to dev BEFORE teardown"},
}


def sh(cmd):
    return subprocess.run(cmd, shell=True, cwd=ROOT).returncode


# ---- scale profiles (same stage commands; S0 + scale differ — the generalization test) ----
_STAGE_MANIFEST = ("eprints (the sample's *.tar.gz) + ~68MB substrate "
                   "(data/warp/{concept-index,def-snippets,defined-index,concept-usage}.json, "
                   "data/concept-encyclopedia-ct.json) + futon3 patterns "
                   "(futon3/resources/sigils/patterns-index.tsv, futon3/library)")
# the RUN OUTPUTS to pull back to dev BEFORE teardown (mark6 lost the CLeans + paper-graphs
# B by pulling only the embed JSON — never delete the box until all of these are on dev).
_RETRIEVE_MANIFEST = ("data/iatc-argument-graphs/$RUN_ID (IATC graphs), holes/clean-$RUN_ID "
                      "(CLeans EDN), data/iatc-paper-graphs/$RUN_ID (object B), "
                      "data/showcases/clean-$RUN_ID-demo (embed+ingest), "
                      "data/expository-scope-graphs/$RUN_ID, data/runs/$RUN_ID (metrics+ledger)")
_RETRIEVE_CMD = ("rsync -avz root@$BOX:'futon6/{data/iatc-argument-graphs,holes/clean,"
                 "data/iatc-paper-graphs,data/showcases,data/expository-scope-graphs,data/runs}/*$RUN_ID*' "
                 "<dev>/  # verify counts, THEN teardown")
PROFILES = {
    "linode": {
        "banner": "LINODE — small / single StackScript box (the reduced-scale end-to-end)",
        "s0": "README-linode: StackScript 2142757; linode-postsetup-deps.sh; hf pre-pull 70B; "
              "linode-4gpu-setup.sh (vLLM 70B, TP=4)",
        "stage": f"rsync -L (DEREFERENCE symlinks) {_STAGE_MANIFEST} to the box, then run S1..S9 there",
        "retrieve": f"PULL run outputs to dev before teardown: {_RETRIEVE_MANIFEST}. {_RETRIEVE_CMD}",
        "scale": "sample: holes/math-ct-200.ids.txt OR a 15-paper citation neighborhood "
                 "(math-ct-neighborhood) + matched random",
    },
    "superpod": {
        "banner": "SUPERPOD — whole math.XX domain / 8-GPU cluster, overnight (LLaMA-only)",
        "s0": "cluster alloc (SLURM/queue); serve LLaMA across 8 GPUs (TP=8); "
              "linode-postsetup-deps.sh; hf pre-pull; corpus-id = domain@date",
        "stage": f"rsync -L (DEREFERENCE symlinks) {_STAGE_MANIFEST} to the cluster scratch ONCE, "
                 "then run S1..S9 there — compute/disk are never the constraint",
        "retrieve": f"PULL run outputs to dev/durable store before releasing the alloc: {_RETRIEVE_MANIFEST}",
        "scale": "ENTIRE math.XX domain (build_ct_manifest over the domain); LLM stages S3/S4/S7 "
                 "at batch concurrency; S2 MUST be corpus-fresh (no --reuse)",
    },
}


def load_deps():
    """:depends-on per stage, from the superpod DAG contract (single source of truth)."""
    txt = open(os.path.join(ROOT, "holes", "superpod-dag-contract.md")).read()
    for b in re.findall(r"```edn\n(.*?)\n```", txt, re.S):
        if ":dag" in b and ":pipeline" in b:
            c = {kw(k): v for k, v in dict(edn.loads(b)).items()}
            out = {}
            for m in c["dag"]:
                sd = {kw(k): v for k, v in dict(m).items()}
                out[kw(sd["id"])] = [kw(d) for d in (sd.get("depends-on") or [])]
            return out
    return {}


# ---- phase-completeness ledger (the superpod contract's teeth) ----
def _ledger(run_dir):
    return os.path.join(run_dir, "phase-ledger.jsonl")


def ledger_record(run_dir, stage, corpus_id, run_id):
    import json
    os.makedirs(run_dir, exist_ok=True)
    open(_ledger(run_dir), "a").write(
        json.dumps({"stage": stage, "corpus_id": corpus_id, "run_id": run_id, "gate": "pass"}) + "\n")


def ledger_has(run_dir, stage, corpus_id):
    import json
    p = _ledger(run_dir)
    if not run_dir or not os.path.exists(p):
        return False
    for line in open(p):
        r = json.loads(line)
        if r.get("stage") == stage and r.get("corpus_id") == corpus_id:
            return True
    return False


def completeness_block(stage, deps, run_dir, corpus_id, reuse):
    """Return a refusal message if any upstream dep lacks a passing ledger entry for THIS
    corpus (the DAG-completeness discipline). S2 is corpus-fresh — never satisfiable by --reuse."""
    if not run_dir:
        return None
    for d in deps:
        if ledger_has(run_dir, d, corpus_id):
            continue
        if d in reuse and d != "S2":
            continue
        extra = " (S2 must be corpus-fresh — NOT --reuse-able)" if d == "S2" else f" (run it, or --reuse {d})"
        return (f"✗ {stage} BLOCKED — upstream {d} has no passing ledger entry for "
                f"corpus '{corpus_id}'{extra}")
    return None


def order(stages, frm, to):
    ids = [s["id"] for s in stages]
    i0 = ids.index(frm) if frm else 0
    i1 = ids.index(to) + 1 if to else len(ids)
    return stages[i0:i1]


DEPS = load_deps()


def _boot_note(profile, sid):
    return {"STAGE": PROFILES[profile]["stage"], "RETRIEVE": PROFILES[profile]["retrieve"]}.get(
        sid, PROFILES[profile]["s0"])


def plan(stages, profile):
    pr = PROFILES[profile]
    print(f"PIPELINE STEPPER — {pr['banner']}\n  scale: {pr['scale']}")
    print("  single-host: after STAGE, S1..S9 all run on the host (no dev/box split)\n")
    for s in stages:
        op = OPS.get(s["id"], {})
        deps = ",".join(DEPS.get(s["id"], [])) or "—"
        tag = "BOOT(dev)" if op.get("boot") else "host"
        print(f"{s['id']} {s['name']:22s} [{s['compute']:8s} {tag:9s}] "
              f"{'⏸HALT' if s['halt'] else '     '}  deps: {deps}  go/no-go: {','.join(s['go']) or '—'}")
        if op.get("boot"):
            print(f"     note: {_boot_note(profile, s['id'])}")
            continue
        if op.get("cmd"):
            print(f"     cmd : {op['cmd'].format(PY=PY, IDS=IDS)}")
        if op.get("gate"):
            print(f"     gate: {op['gate'].format(PY=PY, IDS=IDS)}")
        if op.get("crit"):
            print(f"     crit: {op['crit']}  (criterion — judged at the halt, not executed)")
        if op.get("note"):
            print(f"     note: {op['note']}")


def run(stages, profile, no_halt, run_dir, corpus_id, run_id, reuse):
    print(f"=== {PROFILES[profile]['banner']} | corpus={corpus_id} run={run_id} ===")
    for s in stages:
        op = OPS.get(s["id"], {})
        print(f"\n=== {s['id']} {s['name']} [{s['compute']}] ===")
        # DAG-completeness: every upstream dep must have a passing ledger entry for this corpus
        block = completeness_block(s["id"], DEPS.get(s["id"], []), run_dir, corpus_id, reuse)
        if block:
            print(block)
            return
        if op.get("boot"):   # S0 provision / STAGE rsync — done from dev, then resume on the host
            nxt = stages[stages.index(s) + 1]["id"] if stages.index(s) + 1 < len(stages) else "(done)"
            print(f"⏸ BOOT step — do this from dev, then run the stepper ON THE HOST with --from {nxt}:")
            print(f"   {_boot_note(profile, s['id'])}")
            return
        missing = [p for p in op.get("inputs", []) if not os.path.exists(os.path.join(ROOT, p))]
        if missing:
            print(f"✗ precondition FAILED — missing input(s): {missing}")
            return
        if op.get("cmd"):
            print(f"$ {op['cmd'].format(PY=PY, IDS=IDS)}")
            if sh(op["cmd"].format(PY=PY, IDS=IDS)) != 0:
                print(f"✗ {s['id']} command FAILED — stopping")
                return
        if op.get("gate"):
            print(f"[gate] {op['gate'].format(PY=PY, IDS=IDS)}")
            if sh(op["gate"].format(PY=PY, IDS=IDS)) != 0:
                print(f"✗ {s['id']} GATE FAILED ({','.join(s['go'])}) — stopping for fix")
                return
        if op.get("crit"):  # human criterion, judged at the halt — never a shell command
            print(f"[crit] {op['crit']}")
        if run_dir:
            ledger_record(run_dir, s["id"], corpus_id, run_id)
        print(f"✓ {s['id']} done" + (f" (ledger: {corpus_id})" if run_dir else ""))
        if s["halt"] and not no_halt:
            nxt = stages[stages.index(s) + 1]["id"] if stages.index(s) + 1 < len(stages) else "(done)"
            print(f"⏸ HALT — inspect {s['id']} output; resume with --from {nxt}")
            return
    print("\n✓ run complete")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--profile", choices=["linode", "superpod"], default="linode")
    ap.add_argument("--from", dest="frm", default=None)
    ap.add_argument("--to", default=None)
    ap.add_argument("--no-halt", action="store_true")
    ap.add_argument("--ids", help="override the run's id-list — threads through every per-paper "
                    "stage (S1/S3/S4/S6); e.g. a shard slice for data-parallel")
    ap.add_argument("--run-dir", help="phase-ledger + emit dir (data/runs/<run-id>)")
    ap.add_argument("--corpus-id", default="adhoc")
    ap.add_argument("--run-id", default="adhoc")
    ap.add_argument("--reuse", nargs="*", default=[], help="upstream stages to accept from --reuse (never S2)")
    ap.add_argument("--mark-done", nargs="*", default=[], help="record boot steps (S0/STAGE) as ledger-passed")
    args = ap.parse_args()
    if args.ids:
        global IDS
        IDS = args.ids
    stages = load_stages()
    # inject the STAGE bootstrap step right after S0 (rsync substrate -> host; not in the contract EDN)
    ids = [s["id"] for s in stages]
    if "STAGE" not in ids and "S0" in ids:
        i = ids.index("S0") + 1
        stages.insert(i, {"id": "STAGE", "name": "stage substrate", "compute": "io",
                          "halt": True, "go": []})
    present = {s["id"] for s in stages}
    for sid, nm in [("S10", "lexicon+reground"), ("S11", "structural+whole-paper"),
                    ("S12", "accretion-sweep")]:   # the learning layer (CPU post-stages)
        if sid not in present:
            stages.append({"id": sid, "name": nm, "compute": "cpu", "halt": False, "go": []})
    if "RETRIEVE" not in [s["id"] for s in stages]:   # pull outputs before teardown (mark6 lesson)
        stages.append({"id": "RETRIEVE", "name": "pull run outputs", "compute": "io",
                       "halt": True, "go": []})
    if args.mark_done:
        for sid in args.mark_done:
            ledger_record(args.run_dir, sid, args.corpus_id, args.run_id)
            print(f"ledger: {sid} marked done for corpus {args.corpus_id}")
        return
    if args.plan or not args.run:
        plan(stages, args.profile)
    if args.run:
        run(order(stages, args.frm, args.to), args.profile, args.no_halt,
            args.run_dir, args.corpus_id, args.run_id, args.reuse)


if __name__ == "__main__":
    main()
