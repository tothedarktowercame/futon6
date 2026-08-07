#!/usr/bin/env python3
"""Fast end-to-end replay over PRE-COMPUTED artifacts (no model, no re-parse).

Why this exists
---------------
The slow e2e (mine 12 papers, ~2 days on CPU) is what establishes that the
stages *compute* correctly. But of the 21 hazards found on 2026-08-06/07,
almost none were compute failures — they were **accounting** failures: a stage
naming its output per paper while its consumer looked per proof (H13), scripts
reading fixture corpora instead of the run (H19), the learning layer printing
findings it never persisted (H15/H16), graphs passing one reader and failing the
next (H12), an id family collapsing to its archive name (H14/H19b).

Every one of those is checkable in seconds against artifacts that already exist.
So this harness turns the hazard ledger into an executable regression suite: one
check per hazard class, each asserting a *conservation*, *identity*, *shape*, or
*persistence* invariant that the slow run is supposed to establish.

  python scripts/replay_e2e.py --run-dir data/runs/mark7z \\
      --ids holes/mark7z-e2e.ids.txt --corpus-id math-ct-e2e-12

Exit code is the number of failing checks (0 = clean), so it drops into CI or a
pre-flight gate before booking a cluster window.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

RESULTS: list[tuple[str, bool, str, str, str]] = []   # (id, ok, headline, hazard, needs)

STAGE_ORDER = [f"S{i}" for i in range(1, 13)]


def _reached(needs: str, through: str) -> bool:
    """Is a check meaningful yet, given the run has completed `through`?"""
    try:
        return STAGE_ORDER.index(needs) <= STAGE_ORDER.index(through)
    except ValueError:
        return True


def check(cid: str, hazard: str, needs: str = "S1"):
    """`needs` = earliest stage after which this invariant must hold.

    Checks that hold at a PREFIX are the ones worth running mid-window: they
    answer "is this run producing garbage?" after 12 papers rather than after
    twenty hours. Completeness/persistence checks need the whole pipeline and
    are skipped until then.
    """
    def deco(fn):
        def run(*a, through="S12", **k):
            if not _reached(needs, through):
                return True                            # not yet applicable
            try:
                ok, msg = fn(*a, **k)
            except Exception as e:                     # a check must never crash the suite
                ok, msg = False, f"check raised {type(e).__name__}: {e}"
            RESULTS.append((cid, ok, msg, hazard, needs))
            return ok is not False
        return run
    return deco


def _graphs(d):
    return sorted(g for g in glob.glob(os.path.join(d, "*.edn")) if "rung2" not in g)


def _stem(p):
    b = os.path.basename(p)
    return b[:-4] if b.endswith(".edn") else b


# --------------------------------------------------------------------------
# CONSERVATION — every artifact of stage N has a counterpart in stage N+1,
# or an explicit logged reason. Catches silent collapse (H13) and silent drop.
# --------------------------------------------------------------------------

@check("C1-steps-per-proof", "H13", needs="S3")
def c1(graphs_dir, steps_dir):
    gs = {_stem(g) for g in _graphs(graphs_dir)}
    if not gs:
        return False, "no proof graphs found"
    st = {os.path.basename(p)[:-len(".steps.json")]
          for p in glob.glob(os.path.join(steps_dir, "*.steps.json"))
          if "rung2" not in p}
    missing = gs - st
    return (not missing,
            f"{len(gs - missing)}/{len(gs)} proofs have a steps file"
            + (f"; missing e.g. {sorted(missing)[:3]}" if missing else ""))


@check("C2-clean-accounting", "S7 accounting", needs="S7")
def c2(graphs_dir, clean_dir, logs):
    gs = {_stem(g) for g in _graphs(graphs_dir)}
    cl = {os.path.basename(p)[:-len(".clean.edn")]
          for p in glob.glob(os.path.join(clean_dir, "*.clean.edn"))}
    unaccounted = set()
    logtext = ""
    for lg in logs:
        if os.path.exists(lg):
            logtext += open(lg, errors="replace").read()
    for g in gs - cl:
        if g not in logtext:            # neither typed nor mentioned as rejected/failed
            unaccounted.add(g)
    return (not unaccounted,
            f"{len(cl)} typed, {len(gs - cl)} untyped of {len(gs)}; "
            f"{len(unaccounted)} unaccounted"
            + (f" e.g. {sorted(unaccounted)[:3]}" if unaccounted else ""))


# --------------------------------------------------------------------------
# IDENTITY — every artifact belongs to THIS corpus. Catches fixture-reading
# (H19) and stale-corpus measurement.
# --------------------------------------------------------------------------

@check("I1-artifacts-in-manifest", "H19", needs="S3")
def i1(graphs_dir, ids_file):
    from paper_ids import proof_pid_from_graph_name
    want = {l.strip() for l in open(ids_file) if l.strip()}
    seen = {proof_pid_from_graph_name(g) for g in _graphs(graphs_dir)}
    stray = seen - want
    return (not stray,
            f"{len(seen)} distinct papers, all in manifest"
            if not stray else f"{len(stray)} papers NOT in manifest: {sorted(stray)[:4]}")


@check("I2-metrics-tagged", "H15", needs="S1")
def i2(run_dir, corpus_id):
    p = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(p):
        return False, "metrics.jsonl absent"
    tags, adhoc = set(), 0
    for ln in open(p):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        c = r.get("corpus_id")
        tags.add(c)
        if c == "adhoc":
            adhoc += 1
    if corpus_id not in tags:
        return False, f"NO records for {corpus_id}; tags present: {sorted(t for t in tags if t)[:3]}"
    if adhoc:
        # Untagged records mean some stage is not threading its ids (the H15
        # secondary defect) — worth seeing, but it corrupts provenance, not
        # artifacts, so it must never be the reason a cluster window is abandoned.
        return "warn", f"{adhoc} records tagged 'adhoc' (a stage is not threading --run-id/--corpus-id)"
    return True, f"all records tagged; corpora present: {sorted(t for t in tags if t)[:3]}"


@check("I3-id-families", "H14/H19b", needs="S3")
def i3(graphs_dir):
    from paper_ids import proof_pid_from_graph_name
    pids = {proof_pid_from_graph_name(g) for g in _graphs(graphs_dir)}
    old = {p for p in pids if "__" in p}
    new = {p for p in pids if "__" not in p}
    bare = {p for p in pids if p in ("math", "cond-mat", "alg-geom", "")}
    return (not bare and old and new,
            f"{len(old)} old-style + {len(new)} new-style ids parsed"
            + ("; COLLAPSED ids present: " + str(sorted(bare)) if bare else ""))


# --------------------------------------------------------------------------
# SHAPE — artifacts are readable and internally consistent by the CONSUMING
# reader, not merely by the producing one. Catches H12/H18 and dangling refs.
# --------------------------------------------------------------------------

@check("S1-python-readable", "H12/H18", needs="S3")
def s1(graphs_dir):
    import r2d_concept_coverage as r2d
    from pathlib import Path
    bad = []
    gs = _graphs(graphs_dir)
    for g in gs:
        try:
            r2d.load_edn(Path(g))
        except Exception as e:
            bad.append((os.path.basename(g), type(e).__name__))
    return (not bad,
            f"{len(gs) - len(bad)}/{len(gs)} gated graphs load through the Python reader"
            + (f"; e.g. {bad[:2]}" if bad else ""))


@check("S2-refs-resolve", "R6a/R6b", needs="S3")
def s2(graphs_dir):
    dangling = 0
    total = 0
    for g in _graphs(graphs_dir):
        t = open(g, errors="replace").read()
        ids = set(re.findall(r"\{:id :([a-zA-Z0-9-]+), :kind :(?:object|claim|ref)", t))
        for m in re.finditer(r":(?:premise|conclusion) :([a-zA-Z0-9-]+)", t):
            total += 1
            if m.group(1) not in ids:
                dangling += 1
    pct = 100.0 * dangling / total if total else 0
    return (pct < 5.0, f"{dangling}/{total} dangling premise/conclusion refs ({pct:.1f}%)")


@check("S3-anchors-in-passage", "H21", needs="S3")
def s3(graphs_dir):
    out = 0
    total = 0
    for g in _graphs(graphs_dir):
        t = open(g, errors="replace").read()
        pm = re.search(r":source \{:lines \[(\d+) (\d+)\], :kind :proof\}", t)
        if not pm:
            continue
        lo, hi = int(pm.group(1)), int(pm.group(2))
        for m in re.finditer(r":source \{:lines \[(\d+) (\d+)\]\}", t):
            a, b = int(m.group(1)), int(m.group(2))
            total += 1
            if a < lo or b > hi:
                out += 1
    pct = 100.0 * out / total if total else 0
    return (pct < 5.0, f"{out}/{total} node anchors outside their own passage ({pct:.1f}%)")


# --------------------------------------------------------------------------
# PERSISTENCE — the run directory contains what RETRIEVE promises. Catches the
# whole H15/H16 class: findings that exist only in a terminal.
# --------------------------------------------------------------------------

@check("P1-retrieve-manifest", "H15/H16", needs="S12")
def p1(run_dir):
    promised = {
        "phase-ledger.jsonl": "stage ledger",
        "metrics.jsonl": "MetricRecords",
        "inference-lexicon.json": "harvested move lexicon (H15)",
        "accretion-curve.json": "accretion curve, machine-readable (H16)",
    }
    missing = [f"{k} ({v})" for k, v in promised.items()
               if not os.path.exists(os.path.join(run_dir, k))]
    return (not missing,
            f"{len(promised) - len(missing)}/{len(promised)} promised artifacts present"
            + ("; MISSING " + "; ".join(missing) if missing else ""))


@check("P2-ledger-complete", "mark5 lesson", needs="S12")
def p2(run_dir, corpus_id):
    p = os.path.join(run_dir, "phase-ledger.jsonl")
    if not os.path.exists(p):
        return False, "phase-ledger.jsonl absent"
    seen = set()
    for ln in open(p):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("corpus_id") == corpus_id:
            seen.add(r.get("stage"))
    want = {f"S{i}" for i in range(1, 13)}
    missing = sorted(want - seen, key=lambda s: int(s[1:]))
    return (not missing, f"{len(want & seen)}/12 stages ledgered for {corpus_id}"
            + (f"; missing {missing}" if missing else ""))


@check("P4-skips-recorded", "stage_skip", needs="S9")
def p4(run_dir):
    """Every optional step must have left either a product or a written refusal.

    A step that printed "skipping" and moved on is indistinguishable, after the
    fact, from a step that was never reached. This asserts the run directory can
    account for its own omissions."""
    d = os.path.join(run_dir, "skipped")
    if not os.path.isdir(d):
        return "warn", "no skip records — either nothing skipped, or skips were not recorded"
    recs = glob.glob(os.path.join(d, "*.json"))
    named = []
    for r in recs:
        try:
            named.append(json.load(open(r)).get("step", os.path.basename(r)))
        except Exception:
            named.append(os.path.basename(r))
    return True, f"{len(recs)} optional step(s) recorded as skipped: {', '.join(named)}"


@check("P3-curve-is-rising", "S12 criterion", needs="S12")
def p3(run_dir):
    p = os.path.join(run_dir, "accretion-curve.json")
    if not os.path.exists(p):
        return False, "accretion-curve.json absent"
    d = json.load(open(p))
    pts = d.get("points", [])
    return (bool(d.get("rising")) and len(pts) >= 3,
            f"{len(pts)} checkpoints, rise {d.get('rise')}, rising={d.get('rising')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="data/runs/mark7z")
    ap.add_argument("--graphs", default="data/iatc-argument-graphs/run")
    ap.add_argument("--steps", default="data/cas-select-steps/run")
    ap.add_argument("--clean", default="holes/clean-run")
    ap.add_argument("--ids", default="holes/mark7z-e2e.ids.txt")
    ap.add_argument("--corpus-id", default="math-ct-e2e-12")
    ap.add_argument("--logs", nargs="*", default=["mark7z-s7.log", "mark7z-s7-retry.log",
                                                  "mark7z-s7-sweep.log"])
    ap.add_argument("--through", default="S12", choices=[f"S{i}" for i in range(1, 13)],
                    help="how far the run has got. Checks needing later stages are "
                         "skipped, so this doubles as a MID-RUN ABORT GATE: mine a "
                         "prefix, run with --through S3, and a non-zero exit means "
                         "the run is producing garbage and the window should be "
                         "reclaimed rather than spent.")
    a = ap.parse_args()

    def R(p):
        return p if os.path.isabs(p) else os.path.join(ROOT, p)

    T = a.through
    c1(R(a.graphs), R(a.steps), through=T)
    c2(R(a.graphs), R(a.clean), [R(x) for x in a.logs], through=T)
    i1(R(a.graphs), R(a.ids), through=T)
    i2(R(a.run_dir), a.corpus_id, through=T)
    i3(R(a.graphs), through=T)
    s1(R(a.graphs), through=T)
    s2(R(a.graphs), through=T)
    s3(R(a.graphs), through=T)
    p1(R(a.run_dir), through=T)
    p2(R(a.run_dir), a.corpus_id, through=T)
    p3(R(a.run_dir), through=T)
    p4(R(a.run_dir), through=T)

    if not RESULTS:
        print("no checks applicable at --through " + a.through)
        return 0
    width = max(len(c) for c, _, _, _, _ in RESULTS)
    fails = 0
    print(f"replay-e2e over pre-computed artifacts  "
          f"(corpus {a.corpus_id}, run complete through {a.through})\n")
    warns = 0
    for cid, ok, msg, hz, needs in RESULTS:
        if ok is False:
            fails += 1
            tag = "FAIL"
        elif ok == "warn":
            warns += 1
            tag = "WARN"
        else:
            tag = "PASS"
        print(f"  [{tag}] {cid:<{width}}  {msg}   ({hz})")
    skipped = 12 - len(RESULTS)
    print(f"\n{len(RESULTS) - fails - warns}/{len(RESULTS)} pass, {warns} warn, {fails} fail"
          + (f"  ({skipped} not yet applicable)" if skipped else ""))
    if fails:
        print("\n  *** ABORT RECOMMENDED ***  The run is producing artifacts that fail\n"
              "  invariants the rest of the pipeline depends on. Nothing downstream will\n"
              "  repair them, so the remaining window is better spent regenerating after\n"
              "  a fix than continuing. Failing checks name their hazard class above.")
    else:
        print("\n  CONTINUE — every artifact invariant checkable at this point holds."
              + ("  (Warnings above affect provenance, not artifacts.)" if warns else ""))
    return fails


if __name__ == "__main__":
    sys.exit(main())
