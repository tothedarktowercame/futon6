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
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import futon6_config as config
import run_manifest as manifest
import stage_accounting as accounting
try:
    import edn_format as edn
    _EDN_IMPORT_ERROR = None
except ImportError as _exc:          # inspectable without it; see _MissingDeps
    edn = None
    _EDN_IMPORT_ERROR = _exc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACT = os.path.join(ROOT, "holes", "linode-stepper-contract.md")
PY = config.python_command()  # configured interpreter, safely quoted for stage shells


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
IDS = "holes/mark7-16.ids.txt"
# Shell variables are installed only from the validated run manifest. All
# artifacts live inside --run-dir, so copying that directory preserves paths.
RUN = '"$FUTON6_RUN_DIR"'
CAND = '"$FUTON6_CANDIDATES"'
GRAPHS = '"$FUTON6_GRAPHS"'
EXPO = '"$FUTON6_EXPO"'
CLEAN = '"$FUTON6_CLEAN"'
STEPS = '"$FUTON6_STEPS"'
RUNG3 = '"$FUTON6_RUNG3"'
PAPERG = '"$FUTON6_PAPER_GRAPHS"'
DEMO = '"$FUTON6_DEMO"'
MARKS = '"$FUTON6_MARKS"'
EXPO_CAND = '"$FUTON6_EXPO_CANDIDATES"'
OPS = {
    "S0": {"boot": True, "note": "<profile.s0> — provision the host + serve the model"},
    "STAGE": {"boot": True, "note": "<profile.stage> — rsync eprints + the ~68MB substrate + futon3 "
              "patterns onto the host. DEREFERENCE symlinks (rsync -L / tar -h): dev uses a storage/ "
              "overlay, so a naive copy ships dangling links and S2/S5 then can't read the substrate."},
    "S1": {"cmd": "{PY} scripts/emit_marks.py --list {IDS} --run-dir " + RUN + " --run-id $RUN_ID --corpus-id $CORPUS --out " + MARKS,
           "gate": f"{{PY}} scripts/check_invariants.py --corpus --golden-dir {MARKS} --loss-dir \"$FUTON6_LOSS\"",
           "crit": "wf=0 across the batch — read artifacts/loss/dashboard.json under --run-dir at the halt"},
    "S2": {"cmd": "{PY} scripts/warp_substrate_check.py --ids {IDS} && "
           "{PY} scripts/coverage_inline.py --concepts data/warp/concept-usage.json --field paper_concepts",
           "note": "substrate-corpus match is now a measured gate (E-superpod-hardening H1 tier 1); "
                   "committed concept-usage is df>=10-filtered so the coverage curve reads flat — "
                   "the raw-stream instrument needs S1 to dump per-paper raw concepts (tier 2)",
           "crit": "G-coverage: raw coverage rises with corpus-fraction"},
    "S3": {"cmd": f"{{PY}} scripts/mark3_extract_candidates.py --list {{IDS}} --all-proofs --out {CAND} && "
           f"CANDIDATES={CAND} OUT={GRAPHS} bash scripts/linode-4gpu-run.sh && "
           # MEASUREMENTS, not gates. Both were absent from every stage, so a run
           # produced no evidence about either and both had to be reconstructed
           # afterwards -- the retry rate could not be (H37), and the anchor rate
           # was reconstructed wrongly (H38). `|| true` is deliberate and narrow:
           # anchor-faithfulness currently exits non-zero BY DESIGN while the
           # frame mismatch is open, and a known-red measurement must not
           # masquerade as a stage failure. Its output is kept, not discarded.
           f"(bb scripts/iatc_anchor_faithfulness.bb {GRAPHS} "
           f"> {RUN}/anchor-faithfulness.txt 2>&1 || true) && "
           f"tail -3 {RUN}/anchor-faithfulness.txt",
           "gate": f"bb scripts/iatc_argcheck.bb {GRAPHS} && {{PY}} scripts/substance_gate.py {GRAPHS}",
           "note": "substance gate reads finals only; the run wrapper reuses the enriched "
                   "candidates S3 just extracted (no silent 10-paper re-extract). "
                   "Emits two measurements the pipeline previously never took: the "
                   "in-loop retry rate (retry-rate-$RUN_ID.json, the honesty bound "
                   "on first-pass quality) and anchor-faithfulness, which reports "
                   "frame mismatch separately from drift (H38)"},
    "S4": {"cmd": "{PY} scripts/mark3_extract_expository_candidates.py --list {IDS} "
           f"--out {EXPO_CAND} && {{PY}} scripts/mark3_expository_loop.py "
           f"--candidates {EXPO_CAND} --out {EXPO} "
           "--backend openai --model ${{MODEL:-meta-llama/Llama-3.1-8B-Instruct}} "
           f"--run-dir {RUN} --run-id $RUN_ID --corpus-id $CORPUS",
           "crit": "expository_argcheck (self-gated in loop)",
           "note": "all regions unless FUTON6_EXPOSITORY_CAP_PER_PAPER pins a cap in the manifest; "
                   "then even spacing in source order, with unselected regions accounted as deferred"},
    # S5 now BUILDS its own rung-3 half. Both producers are deterministic (no model):
    # cas_segment turns gated graphs into proof steps, rung3_technique turns those into
    # technique gap maps, and only then does comprehension have a strategy axis to score.
    # They were absent from the DAG entirely, so `weak-proof` was unreachable and every
    # proof scored no-structure — the criterion below could not be met by construction
    # (E-superpod-hardening H13). Closing it needed no GPU, only the wiring.
    "S5": {"cmd": f"{{PY}} scripts/cas_segment.py {GRAPHS}/*.edn --out-dir {STEPS} && "
           f"{{PY}} scripts/rung3_technique.py --steps-dir {STEPS} --out-dir {RUNG3} && "
           f"{{PY}} scripts/clean_comprehension.py --graphs {GRAPHS} --candidates {CAND} "
           f"--steps {STEPS} --rung3 {RUNG3} --run-dir {RUN} "
           "--run-id $RUN_ID --corpus-id $CORPUS",
           "crit": "G-comprehension: verdict separates weak-extraction from weak-proof"},
    # Every paper is assembled and accounted even when one is malformed; the stage
    # still fails on any rejected or errored paper object.
    "S6": {"cmd": f"{{PY}} scripts/paper_graph_assemble.py --list {{IDS}} --iatc {GRAPHS} --expo {EXPO} "
           f"--run-dir {RUN} --run-id $RUN_ID --corpus-id $CORPUS --out {PAPERG} --marks-dir {MARKS}",
           "gate": f"test -d {PAPERG} && "
                   f"test $(ls {PAPERG}/*.B.json 2>/dev/null | wc -l) -gt 0",
           "crit": "B wellformed: every proof attaches to a statement; orphans flagged"},
    "S7": {"cmd": f"{{PY}} scripts/clean_box_typing.py --graphs {GRAPHS} --out {CLEAN} "
           '--endpoint "${{OPENAI_BASE_URL%/}}/chat/completions" --model "$MODEL" '
           f"--run-dir {RUN} --run-id $RUN_ID --corpus-id $CORPUS && "
           f"{{PY}} scripts/clean_structure_embed.py --clean-dir {CLEAN} --out {DEMO}",
           "gate": f"bb scripts/clean_vocab_gate.bb {CLEAN} && {{PY}} scripts/clean_entropy_gate.py "
           f"--embed {DEMO}/clean-embed.json"},
    # S8's render tail is part of the stage, not an option. It reads THIS run's
    # graphs (--graph-dir) and writes under $RUN, so the pages describe the
    # corpus that produced them; the layer modules were pinned to loop-run-70b
    # and to Joe's absolute checkout path until 2026-08-07, which made
    # `render_run --all` report "16/16 rendered" over pages whose pipeline
    # overlays were empty. Span counts now equal the artifact counts (98 IATC,
    # 280 expository), which is the check worth keeping.
    "S8": {"cmd": f"{{PY}} scripts/clean_graph_export.py --clean-dir {CLEAN} --out {DEMO}/ingest "
           f"--embed-json {DEMO}/clean-embed.json"
           f" && FUTON6_IATC_DIR={GRAPHS} FUTON6_EXPO_DIR={EXPO} "
           f"{{PY}} scripts/render_run.py --all --graph-dir {GRAPHS} --out-dir {RUN}/render"},
    # S9 is run-scoped: both tails read THIS run's graphs and write under $RUN.
    # They previously defaulted to the global graph tree (recursive, every run
    # ever made) and to a shared demo path outside any run directory, so their
    # products were neither about this corpus nor collected by RETRIEVE.
    # DEPRECATED and removed from this pipeline: mark4_apm_structure_coverage.
    # It matches APM proof scopes against literature scopes produced by the
    # nlab-wiring detector — inputs this corpus does not generate and never has.
    # It was chained here with `;`, so it failed on every run since it was
    # written without anyone seeing it. It belongs to the APM programme; invoke
    # it there against APM's own inputs. S9's product is the pass-3 hole map and
    # the normalized hole vocabulary, both run-scoped.
    "S9": {"cmd": f"{{PY}} scripts/clean_hole_harvest.py --graphs {GRAPHS} "
           f"--out {RUN}/pass3-holes.json && "
           f"{{PY}} scripts/warrant_normalize.py --graphs {GRAPHS} "
           f"--out {RUN}/hole-vocabulary.json",
           "crit": "pass-3 hole map + normalized (type, concept) hole vocabulary, both run-scoped"},
    # --- LEARNING LAYER (the 'improve as we run' instrumentation; CPU post-stages) ---
    "S10": {"cmd": f"{{PY}} scripts/iatc_lexicon_harvest.py --graphs {GRAPHS} --run-dir {RUN} "
            "--run-id $RUN_ID --corpus-id $CORPUS && "
            f"{{PY}} scripts/iatc_move_reground.py --graphs {GRAPHS} --candidates {CAND} && "
            f"{{PY}} scripts/expository_reground.py --scopes {EXPO} "
            "--measure-ids {IDS}",
            "crit": "move-lexicon harvested (relations+warrants+expository moves); reground lift >= 0"},
    # `;` here swallowed a FileNotFoundError on every run, so S11 reported PASS while
    # half of it had never executed (H22). sfc_struct_canon now records an explicit
    # refusal artifact instead of dying, so `&&` is safe and a real failure surfaces.
    "S11": {"cmd": f"{{PY}} scripts/def_formulae_extract.py --ids {{IDS}} --out {RUN}/def-formulae.txt && "
            f"{{PY}} scripts/sfc_struct_canon.py --formulae {RUN}/def-formulae.txt "
            f"--run-dir {RUN} --run-id $RUN_ID --corpus-id $CORPUS && "
            f"{{PY}} scripts/clean_paper_signature.py --embed {DEMO}/clean-embed.json "
            f"--run-dir {RUN} --run-id $RUN_ID --corpus-id $CORPUS",
            "crit": "structural canonical shapes + whole-paper signatures produced"},
    "S12": {"cmd": f"{{PY}} scripts/accretion_curves.py --graphs {GRAPHS} --candidates {CAND} "
            f"--run-dir {RUN} --run-id $RUN_ID --corpus-id $CORPUS",
            "crit": "ACCRETION SWEEP: every tier metric checkpointed at log-spaced n -> rising curves"},
    "RETRIEVE": {"boot": True, "halt": True, "note": "<profile.retrieve> — pull ALL run outputs to dev BEFORE teardown"},
}


def sh(cmd, log_path=None):
    if log_path is None:
        return subprocess.run(cmd, shell=True, cwd=ROOT, env=config.child_environment()).returncode
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as log:
        with subprocess.Popen(cmd, shell=True, cwd=ROOT, env=config.child_environment(),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()


# ---- scale profiles (same stage commands; S0 + scale differ — the generalization test) ----
_STAGE_MANIFEST = ("eprints (the sample's *.tar.gz) + ~68MB substrate "
                   "(data/warp/{concept-index,def-snippets,defined-index,concept-usage}.json, "
                   "data/concept-encyclopedia-ct.json, data/background-corpus-index.json) + futon3 patterns "
                   "(futon3/resources/sigils/patterns-index.tsv, futon3/library)")
# the RUN OUTPUTS to pull back to dev BEFORE teardown (mark6 lost the CLeans + paper-graphs
# B by pulling only the embed JSON — never delete the box until all of these are on dev).
_RETRIEVE_MANIFEST = "the manifest, frozen corpus, ledger, metrics, logs and artifacts under --run-dir"
_RETRIEVE_CMD = ("python scripts/retrieve_run.py pack --run-dir <run-dir> --output <archive.tgz>; "
                 "copy the archive to durable storage, then verify it there before teardown")
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


def ledger_record(run_dir, stage, corpus_id, run_id, invocation=None):
    os.makedirs(run_dir, exist_ok=True)
    with open(_ledger(run_dir), "a") as handle:
        handle.write(json.dumps({"stage": stage, "corpus_id": corpus_id, "run_id": run_id,
                                 "gate": "pass", "invocation": invocation}) + "\n")


def _rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ledger_entry(run_dir, stage, corpus_id):
    if not run_dir:
        return None
    for r in _rows(_ledger(run_dir)):
        if r.get("stage") == stage and r.get("corpus_id") == corpus_id and r.get("gate") == "pass":
            return r
    return None


def ledger_has(run_dir, stage, corpus_id):
    return ledger_entry(run_dir, stage, corpus_id) is not None


# ---- per-invocation attempt history and item accounting (Stage 3) ----
# Every execution of a stage appends one row here, whatever its outcome, so a
# failed or rejected attempt leaves counts and reasons rather than only log text.
ATTEMPTS = "stage-attempts.jsonl"
# stage -> [(producer, inputs)]. Inputs name where the expected item ids come
# from: the frozen corpus, or the accepted outputs of an earlier producer (in this
# invocation for the same stage; in the ledgered invocation for another stage).
ACCOUNTING = {
    "S3": [("extract", "corpus"), ("loop", "S3.extract")],
    "S4": [("extract", "corpus"), ("select", "S4.extract"), ("loop", "S4.select")],
    "S6": [("assemble", "corpus")],
    "S7": [("typing", "S3.loop")],
}


def next_invocation(run_dir, stage):
    n = sum(1 for r in _rows(os.path.join(run_dir, ATTEMPTS)) if r.get("stage") == stage)
    return f"{stage}-a{n + 1:03d}"


def accounting_dir(run_dir, stage, invocation):
    return os.path.join(run_dir, "accounting", stage, invocation)


def accounting_problems(run_dir, stage, invocation, corpus_id):
    """Check a stage's item accounting; return (problems, per-producer counts)."""
    doc = manifest.load(Path(run_dir))
    loaded, counts, found = {}, {}, []
    for producer, source in ACCOUNTING.get(stage, []):
        try:
            if source == "corpus":
                expected = doc["papers"]
            else:
                src_stage, src_producer = source.split(".")
                if src_stage == stage:
                    upstream = loaded[source]
                else:
                    entry = ledger_entry(run_dir, src_stage, corpus_id)
                    if not entry or not entry.get("invocation"):
                        raise ValueError(f"{source}: no ledgered accounting for upstream stage")
                    upstream = accounting.load(accounting_dir(run_dir, src_stage, entry["invocation"]),
                                               src_stage, src_producer)
                expected = accounting.accepted_outputs(upstream)
            current = accounting.load(accounting_dir(run_dir, stage, invocation), stage, producer)
        except (KeyError, ValueError, OSError) as exc:
            found.append(f"{stage}.{producer}: {exc}")
            break
        loaded[f"{stage}.{producer}"] = current
        counts[producer] = current["counts"]
        allow_deferred = (stage, producer) == ("S4", "select") and doc["selection"]["expository-cap"] > 0
        found += accounting.problems(current, expected, run_dir=Path(run_dir), allow_deferred=allow_deferred)
    return found, counts


def completeness_block(stage, deps, run_dir, corpus_id, reuse):
    """Return a refusal message if any upstream dep lacks a passing ledger entry for THIS
    corpus (the DAG-completeness discipline). S2 is corpus-fresh — never satisfiable by --reuse."""
    if not run_dir:
        return None
    for d in deps:
        if ledger_has(run_dir, d, corpus_id):
            continue
        if d in reuse and d in ("S0", "STAGE"):
            continue
        extra = " (S2 must be corpus-fresh — NOT --reuse-able)" if d == "S2" else (f" (run it, or --reuse {d})" if d in ("S0", "STAGE") else " (run it in this manifest)")
        return (f"✗ {stage} BLOCKED — upstream {d} has no passing ledger entry for "
                f"corpus '{corpus_id}'{extra}")
    return None


def order(stages, frm, to):
    ids = [s["id"] for s in stages]
    i0 = ids.index(frm) if frm else 0
    i1 = ids.index(to) + 1 if to else len(ids)
    if i0 >= i1:
        raise ValueError("--from must precede or equal --to")
    return stages[i0:i1]


class _MissingDeps(dict):
    """Stand-in for DEPS when edn_format is absent.

    The stepper cannot RUN without edn_format -- it parses the DAG contract from
    EDN. But conformance imports this module only to read OPS and the path
    constants, and DEPS is built at import time, so a missing edn_format took
    down the whole import and left two conformance checks unevaluatable on any
    host that had not installed it. Documenting that as a limitation was the
    wrong call: the gate exists to inspect unprepared hosts, so it must work on
    one.

    Inspection now succeeds; the first attempt to USE dependencies raises with
    the real cause, so nothing runs on silently-empty dependency data.
    """

    def _die(self, *_a, **_k):
        raise RuntimeError(
            "stage dependencies unavailable: edn_format is not installed, so the "
            f"DAG contract could not be parsed ({_EDN_IMPORT_ERROR}). "
            "Run: pip install edn_format") from _EDN_IMPORT_ERROR

    __getitem__ = _die
    get = _die
    items = _die
    keys = _die
    values = _die


DEPS = load_deps() if edn is not None else _MissingDeps()


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


def preflight_gate(ids: str) -> int:
    """Refuse to start unless every declared dependency is present. 0 = go.

    preflight.py existed and printed "DO NOT START", and nothing consulted it —
    so it was advice, and advice was not what it was for. LaTeXML was audited,
    fixed, recorded READY, and then absent from the run host for a month while
    S11's :structure lift was a silent no-op; an eprint store was mis-reported as
    missing because only an env var was checked. Both classes are exactly what a
    preflight catches, and neither was caught, because running it was optional.

    There is no override flag. A run that cannot satisfy its dependencies is not
    a run with a caveat, it is a run whose numbers mean nothing, and the cost of
    finding that out at stage 7 of 12 on a booked window is the whole window.
    """
    import subprocess
    print("preflight (mandatory) ...")
    # PY is a command string with a flag ("... python -u"), not a bare path.
    p = subprocess.run([*shlex.split(PY), os.path.join(ROOT, "scripts", "preflight.py"), "--ids", ids],
                       cwd=ROOT, text=True, capture_output=True, env=config.child_environment())
    print(p.stdout.rstrip() or p.stderr.rstrip())
    if p.returncode != 0:
        print(f"\nREFUSING TO START: {p.returncode} preflight check(s) failed.\n"
              "  Each failure above names its remedy. Fix them, or run\n"
              "  scripts/preflight.py --fix for the automatable ones.")
        return 1
    return 0


def conformance_gate(ids: str) -> int:
    """Refuse to start unless the host BEHAVES as the pipeline assumes. 0 = go.

    Runs after preflight because it asks the next question. Preflight settles
    whether the dependencies are present; this settles whether they act the way
    the code was written against — which is a different thing on a host whose
    model-serving stack is not the one we developed on. An endpoint that accepts
    `response_format: json_schema` and ignores it produces no error at any point:
    the stages complete, the gates pass, and the artifacts are templates.

    Cheap on purpose (~2 minutes) so that aborting a booked window costs minutes
    rather than the window.
    """
    import subprocess
    print("conformance (mandatory) ...")
    cmd = [*shlex.split(PY), os.path.join(ROOT, "scripts", "conformance.py")]
    if os.environ.get("OPENAI_BASE_URL"):
        cmd += ["--endpoint", os.environ["OPENAI_BASE_URL"]]
    if os.environ.get("MODEL"):
        cmd += ["--model", os.environ["MODEL"]]
    p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, env=config.child_environment())
    print(p.stdout.rstrip() or p.stderr.rstrip())
    if p.returncode != 0:
        print(f"\nREFUSING TO START: {p.returncode} conformance check(s) failed.")
        return 1
    return 0


def attempt(s, run_dir, corpus_id, run_id):
    """One execution of a computational stage: command, gate, item accounting.

    The attempt row is written whatever happens, so rejected and errored items stay
    inspectable. Only a zero command status, a zero gate status and accounting with
    every expected item accepted produce a passing ledger row.
    """
    op = OPS.get(s["id"], {})
    sid = s["id"]
    invocation = next_invocation(run_dir, sid) if run_dir else None
    row = {"stage": sid, "run_id": run_id, "corpus_id": corpus_id, "invocation": invocation,
           "started": datetime.now(timezone.utc).isoformat(),
           "command_rc": None, "gate_rc": None, "accounting": None, "problems": []}
    if run_dir:
        adir = accounting_dir(run_dir, sid, invocation)
        os.makedirs(adir)          # a fresh directory per invocation; never merge histories
        os.environ[accounting.DIR_ENV] = adir
        os.environ[accounting.INVOCATION_ENV] = invocation
    outcome = 0
    try:
        if op.get("cmd"):
            print(f"$ {op['cmd'].format(PY=PY, IDS=IDS)}")
            row["command_rc"] = sh(op["cmd"].format(PY=PY, IDS=IDS), os.path.join(run_dir, "logs", sid + ".command.log") if run_dir else None)
            if row["command_rc"] != 0:
                print(f"✗ {sid} command FAILED — stopping")
                outcome = 2
        if not outcome and op.get("gate"):
            print(f"[gate] {op['gate'].format(PY=PY, IDS=IDS)}")
            row["gate_rc"] = sh(op["gate"].format(PY=PY, IDS=IDS), os.path.join(run_dir, "logs", sid + ".gate.log") if run_dir else None)
            if row["gate_rc"] != 0:
                print(f"✗ {sid} GATE FAILED ({','.join(s['go'])}) — stopping for fix")
                outcome = 3
        if run_dir and sid in ACCOUNTING:
            # Checked even after a failed command: the counts are the evidence of
            # what was attempted, and the reasons say why it did not pass.
            row["problems"], row["accounting"] = accounting_problems(run_dir, sid, invocation, corpus_id)
            for problem in row["problems"]:
                print(f"  [accounting] {problem}")
            if row["problems"] and not outcome:
                print(f"✗ {sid} ACCOUNTING FAILED — not every item was accepted")
                outcome = 3
    finally:
        os.environ.pop(accounting.DIR_ENV, None)
        os.environ.pop(accounting.INVOCATION_ENV, None)
        row["outcome"] = "pass" if not outcome else ("command-failed" if outcome == 2 else "rejected")
        row["finished"] = datetime.now(timezone.utc).isoformat()
        if run_dir:
            with open(os.path.join(run_dir, ATTEMPTS), "a") as handle:
                handle.write(json.dumps(row) + "\n")
    if outcome:
        return outcome
    if op.get("crit"):  # human criterion, judged at the halt — never a shell command
        print(f"[crit] {op['crit']}")
    if run_dir:
        ledger_record(run_dir, sid, corpus_id, run_id, invocation)
    print(f"✓ {sid} done" + (f" (ledger: {corpus_id}, {invocation})" if run_dir else ""))
    return 0


def run(stages, profile, no_halt, run_dir, corpus_id, run_id, reuse):
    """Execute stages; RETURN AN EXIT CODE rather than merely printing.

    0 = ran to completion (or halted deliberately); 1 = refused/blocked;
    2 = a stage command failed; 3 = a stage gate or its item accounting failed.

    Previously every failure path printed and returned None, and main() exited
    0 regardless — so an outer scheduler recorded success while the stepper had
    stopped. For an unattended cluster window that is the difference between
    "the run finished" and "the run stopped four hours in and nobody knew".
    """
    print(f"=== {PROFILES[profile]['banner']} | corpus={corpus_id} run={run_id} ===")
    # A successful stage is immutable within this run. Re-executing it could
    # invalidate already-passing downstream evidence, even if this attempt fails.
    completed = [s["id"] for s in stages if not OPS.get(s["id"], {}).get("boot")
                 and ledger_has(run_dir, s["id"], corpus_id)]
    if completed:
        print(f"REFUSING TO RERUN completed stages: {', '.join(completed)}; "
              "resume at the next stage or start a new run directory")
        return 1
    for s in stages:
        op = OPS.get(s["id"], {})
        print(f"\n=== {s['id']} {s['name']} [{s['compute']}] ===")
        # DAG-completeness: every upstream dep must have a passing ledger entry for this corpus
        block = completeness_block(s["id"], DEPS.get(s["id"], []), run_dir, corpus_id, reuse)
        if block:
            print(block)
            return 1
        if op.get("boot"):   # S0 provision / STAGE rsync — done from dev, then resume on the host
            nxt = stages[stages.index(s) + 1]["id"] if stages.index(s) + 1 < len(stages) else "(done)"
            print(f"⏸ BOOT step — do this from dev, then run the stepper ON THE HOST with --from {nxt}:")
            print(f"   {_boot_note(profile, s['id'])}")
            # E-superpod-hardening H2: boot steps write no ledger entry, so the ledger
            # will BLOCK the next stage unless the resume names them in --reuse. Say so.
            boots = [x["id"] for x in stages[:stages.index(s) + 1] if OPS[x["id"]].get("boot")]
            reuse_hint = " ".join(sorted(set(boots + (reuse or []))))
            print(f"   then resume with: --from {nxt} --reuse {reuse_hint}"
                  f"   (boot steps never ledger-record; without --reuse the next stage is BLOCKED)")
            return 0
        missing = [p for p in op.get("inputs", []) if not os.path.exists(os.path.join(ROOT, p))]
        if missing:
            print(f"✗ precondition FAILED — missing input(s): {missing}")
            return 1
        rc = attempt(s, run_dir, corpus_id, run_id)
        if rc:
            return rc
        if s["halt"] and not no_halt:
            nxt = stages[stages.index(s) + 1]["id"] if stages.index(s) + 1 < len(stages) else "(done)"
            print(f"⏸ HALT — inspect {s['id']} output; resume with --from {nxt}")
            return 0
    print("\n✓ run complete")
    return 0


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
    ap.add_argument("--corpus-id")
    ap.add_argument("--run-id")
    ap.add_argument("--reuse", nargs="+", action="extend", default=[], choices=["S0", "STAGE"], help="completed boot steps; repeated options accumulate; computational stages require ledger evidence")
    ap.add_argument("--mark-done", nargs="+", choices=["S0", "STAGE"], default=[], help="terminal boot bookkeeping; cannot combine with --run, --plan, --from, --to or --reuse")
    args = ap.parse_args()
    if args.mark_done and (args.run or args.plan or args.frm or args.to or args.reuse or args.no_halt):
        ap.error("--mark-done is terminal boot bookkeeping; invoke execution separately")
    global IDS
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
    if not args.run and not args.mark_done:
        if args.ids:
            IDS = shlex.quote(args.ids)
        print("host configuration: " + json.dumps(config.effective(), sort_keys=True))
        plan(stages, args.profile)
        return 0
    try:
        run_id = manifest.identity(args.run_id, "RUN_ID", "--run-id")
        corpus_id = manifest.identity(args.corpus_id, "CORPUS", "--corpus-id")
        run_dir = Path(args.run_dir or os.path.join("data", "runs", run_id))
        run_dir = (Path(ROOT) / run_dir).resolve()
        if args.ids:
            source_ids = (Path(ROOT) / args.ids).resolve()
        elif (run_dir / manifest.NAME).exists():
            source_ids = run_dir / "corpus.ids.txt"
        else:
            source_ids = Path(ROOT) / IDS
        os.environ.update(config.child_environment())
        with manifest.lock(run_dir):
            doc = manifest.prepare(run_dir, run_id, corpus_id, source_ids)
            os.environ.update(manifest.environment(run_dir, doc))
            IDS = shlex.quote(str(run_dir / doc["ids"]))
            effective = doc["host-configuration"]
            print("host configuration: " + json.dumps(effective, sort_keys=True))
            with (run_dir / "host-config.jsonl").open("a") as handle:
                handle.write(json.dumps({"recorded-at": datetime.now(timezone.utc).isoformat(),
                                         "run-id": run_id, "configuration": effective}) + "\n")
            if args.mark_done:
                for sid in args.mark_done:
                    ledger_record(str(run_dir), sid, corpus_id, run_id)
                    print(f"ledger: {sid} marked done for corpus {corpus_id}")
                return 0
            rc = preflight_gate(str(run_dir / doc["ids"])) or conformance_gate(str(run_dir / doc["ids"]))
            if rc:
                return rc
            return run(order(stages, args.frm, args.to), args.profile, args.no_halt,
                       str(run_dir), corpus_id, run_id, sorted(set(args.reuse)))
    except (OSError, ValueError) as exc:
        print(f"REFUSING TO START: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
