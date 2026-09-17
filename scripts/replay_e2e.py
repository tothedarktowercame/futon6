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
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import run_manifest as manifest
import stage_accounting as accounting

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

@check("C1-steps-per-proof", "H13", needs="S5")
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
def c2(run_dir, corpus_id, clean_dir):
    # Every S3-accepted graph must be a typed, gated CLean. A graph merely
    # mentioned in a log (rejected, failed) used to count as accounted for, so a
    # G7 rejection could sit inside a replay PASS.
    s3 = accounting.ledgered_invocation(run_dir, "S3", corpus_id)
    s7 = accounting.ledgered_invocation(run_dir, "S7", corpus_id)
    if not (s3 and s7):
        return False, "S3/S7 have no ledgered accounting invocation"
    graphs = set(accounting.accepted_outputs(accounting.load(accounting.directory(run_dir, "S3", s3), "S3", "loop")))
    typing = accounting.load(accounting.directory(run_dir, "S7", s7), "S7", "typing")
    typed = {e["id"] for e in typing["items"] if e["status"] == "accepted"}
    files = {os.path.basename(p)[:-len(".clean.edn")] for p in glob.glob(os.path.join(clean_dir, "*.clean.edn"))}
    not_typed = sorted(graphs - typed)
    mismatch = sorted(typed ^ files)
    return (not not_typed and not mismatch,
            f"{len(typed)}/{len(graphs)} accepted graphs typed"
            + (f"; not typed e.g. {not_typed[:3]}" if not_typed else "")
            + (f"; CLean files disagree with accounting e.g. {mismatch[:3]}" if mismatch else ""))


@check("A1-item-accounting", "Stage 3 accounting", needs="S3")
def a1(run_dir, corpus_id, through):
    # Re-verify each ledgered item-level stage from its own accounting, on this
    # (possibly retrieved) copy: all expected items accepted, artifacts present.
    checked, found = [], []
    for stage in accounting.STAGES:
        if not _reached(stage, through):
            continue
        invocation = accounting.ledgered_invocation(run_dir, stage, corpus_id)
        if not invocation:
            found.append(f"{stage}: no ledgered invocation")
            continue
        problems, _ = accounting.stage_problems(Path(run_dir), stage, invocation, corpus_id)
        found += problems
        checked.append(f"{stage}@{invocation}")
    return (not found, f"{len(checked)} stage(s) fully accepted: {', '.join(checked)}"
            if not found else f"{len(found)} problem(s): " + " | ".join(found[:3]))


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
    tags, untagged, malformed = set(), 0, 0
    for ln in open(p):
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except Exception:
            malformed += 1
            continue
        tags.add(r.get("corpus_id"))
        if "adhoc" in (r.get("corpus_id"), r.get("run_id")) or r.get("stage") not in STAGE_ORDER:
            untagged += 1
    if corpus_id not in tags:
        return False, f"NO records for {corpus_id}; tags present: {sorted(t for t in tags if t)[:3]}"
    # Provenance is part of acceptance: an `adhoc` or stageless record means some
    # producer did not thread identities, so the metrics cannot be attributed.
    return (not untagged and not malformed,
            f"all records tagged for {corpus_id}" if not (untagged or malformed)
            else f"{untagged} adhoc/stageless and {malformed} malformed record(s)")


@check("I3-id-families", "H14/H19b", needs="S3")
def i3(graphs_dir, run_dir, corpus_id):
    # Parsed paper ids must be exactly the papers that S3 accounting says produced
    # accepted graphs. Requiring both old- and new-style ids was a property of one
    # historical corpus; a single-family corpus is valid, a collapsed id is not.
    from paper_ids import proof_pid_from_graph_name
    invocation = accounting.ledgered_invocation(run_dir, "S3", corpus_id)
    if not invocation:
        return False, "S3 has no ledgered accounting invocation"
    doc = accounting.load(accounting.directory(run_dir, "S3", invocation), "S3", "loop")
    want = {e["paper"] for e in doc["items"] if e["status"] == "accepted"}
    seen = {proof_pid_from_graph_name(g) for g in _graphs(graphs_dir)}
    bare = {p for p in seen if p in ("math", "cond-mat", "alg-geom", "", None)}
    old = {p for p in seen if p and "__" in p}
    return (not bare and seen == want,
            f"{len(old)} old-style + {len(seen - old)} new-style paper ids parse to the accounted papers"
            if not bare and seen == want else
            f"parsed {sorted(map(str, seen ^ want))[:4]} disagree with accounting"
            + ("; COLLAPSED ids present: " + str(sorted(map(str, bare))) if bare else ""))


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


def _parsed(graphs_dir):
    import r2d_concept_coverage as r2d
    for g in _graphs(graphs_dir):
        yield os.path.basename(g), r2d.load_edn(Path(g))


def _refs(value):
    return value if isinstance(value, list) else ([] if value is None else [value])


@check("S2-refs-resolve", "R6a/R6b", needs="S3")
def s2(graphs_dir):
    # Parsed, not regex-matched: the regex required `{:id :x, :kind :claim` in that
    # key order and ASCII ids, so it reported 19/683 dangling on the 98-graph run
    # where the parsed graphs have none missing. Acceptance tolerance is zero.
    total, bad = 0, []
    for name, g in _parsed(graphs_dir):
        ids = {n.get("id") for n in g.get("nodes", [])}
        for e in g.get("edges", []):
            for field in ("premise", "conclusion"):
                for ref in _refs(e.get(field)):
                    total += 1
                    if isinstance(ref, dict):
                        bad.append(f"{name} {e.get('id')} {field} is an inline map, not a node id")
                    elif ref not in ids:
                        bad.append(f"{name} {e.get('id')} {field} {ref} is not a node")
    return (not bad, f"{len(bad)}/{total} unresolved premise/conclusion refs"
            + (f"; e.g. {bad[:2]}" if bad else ""))


@check("S3-anchors-in-passage", "H21", needs="S3")
def s3(graphs_dir):
    # Every node and edge anchor lies inside its graph's own :kind :proof passage.
    total, bad, unanchored = 0, [], []
    for name, g in _parsed(graphs_dir):
        source = g.get("source") or {}
        if source.get("kind") != ":proof" or len(source.get("lines") or []) != 2:
            unanchored.append(name)
            continue
        lo, hi = source["lines"]
        for part in (*g.get("nodes", []), *g.get("edges", [])):
            lines = (part.get("source") or {}).get("lines")
            if not lines:
                continue
            total += 1
            if len(lines) != 2 or lines[0] < lo or lines[1] > hi:
                bad.append(f"{name} {part.get('id')} {lines} outside [{lo} {hi}]")
    return (not bad and not unanchored,
            f"{len(bad)}/{total} anchors outside their passage; {len(unanchored)} graph(s) without a proof passage"
            + (f"; e.g. {(bad or unanchored)[:2]}" if bad or unanchored else ""))


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
        if r.get("corpus_id") == corpus_id and r.get("gate") == "pass":
            seen.add(r.get("stage"))
    want = {f"S{i}" for i in range(1, 13)}
    missing = sorted(want - seen, key=lambda s: int(s[1:]))
    return (not missing, f"{len(want & seen)}/12 stages ledgered for {corpus_id}"
            + (f"; missing {missing}" if missing else ""))


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
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--graphs")
    ap.add_argument("--steps")
    ap.add_argument("--clean")
    ap.add_argument("--ids")
    ap.add_argument("--corpus-id")
    ap.add_argument("--logs", nargs="*")
    ap.add_argument("--through", default="S12", choices=[f"S{i}" for i in range(1, 13)],
                    help="how far the run has got. Checks needing later stages are "
                         "skipped, so this doubles as a MID-RUN ABORT GATE: mine a "
                         "prefix, run with --through S3, and a non-zero exit means "
                         "the run is producing garbage and the window should be "
                         "reclaimed rather than spent.")
    a = ap.parse_args()
    RESULTS.clear()

    try:
        run_dir = (Path(ROOT) / a.run_dir).resolve()
        doc = manifest.load(run_dir)
        manifest.validate_records(run_dir, doc)
        manifest.require_artifacts(run_dir, doc, a.through)
        if a.corpus_id and a.corpus_id != doc["corpus-id"]:
            raise ValueError("--corpus-id disagrees with run manifest")
        a.corpus_id = doc["corpus-id"]
        for key in ("graphs", "steps", "clean"):
            expected = manifest.contained(run_dir, doc["artifacts"][key])
            given = getattr(a, key)
            if given and (Path(ROOT) / given).resolve() != expected.resolve():
                raise ValueError(f"--{key} disagrees with run manifest")
            setattr(a, key, str(expected))
        frozen = run_dir / doc["ids"]
        if a.ids and manifest.digest((Path(ROOT) / a.ids).resolve()) != doc["corpus-sha256"]:
            raise ValueError("--ids content disagrees with frozen corpus")
        a.ids = str(frozen)
        expected_logs = [str(manifest.contained(run_dir, p)) for p in doc["logs"]]
        if a.logs is not None and [str((Path(ROOT) / p).resolve()) for p in a.logs] != expected_logs:
            raise ValueError("--logs disagrees with run manifest")
        a.logs = expected_logs
        a.run_dir = str(run_dir)
    except (OSError, ValueError, KeyError) as exc:
        print(f"REPLAY TARGET ERROR: {exc}", file=sys.stderr)
        return 2

    def R(p):
        return p if os.path.isabs(p) else os.path.join(ROOT, p)

    T = a.through
    c1(R(a.graphs), R(a.steps), through=T)
    a1(R(a.run_dir), a.corpus_id, T, through=T)
    c2(R(a.run_dir), a.corpus_id, R(a.clean), through=T)
    i1(R(a.graphs), R(a.ids), through=T)
    i2(R(a.run_dir), a.corpus_id, through=T)
    i3(R(a.graphs), R(a.run_dir), a.corpus_id, through=T)
    s1(R(a.graphs), through=T)
    s2(R(a.graphs), through=T)
    s3(R(a.graphs), through=T)
    p1(R(a.run_dir), through=T)
    p2(R(a.run_dir), a.corpus_id, through=T)
    p3(R(a.run_dir), through=T)

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
    elif warns:
        print("\n  ACCEPTANCE INCOMPLETE — resolve the warnings before accepting this run.")
    else:
        print("\n  CONTINUE — every artifact invariant checkable at this point holds.")
    return fails + warns


if __name__ == "__main__":
    sys.exit(main())
