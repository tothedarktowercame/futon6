#!/usr/bin/env python3
"""Recorded pilot go/no-go for the corpus-relative comprehension floor.

E-comprehension-foundation, the Linode go/no-go question: does the composed
comprehension score (a) RISE as the corpus grows, and (b) SEPARATE
weak-extraction from weak-proof?

Method (pure CPU, deterministic, no network):
  (a) Corpus growth: score the same run's argument graphs with the grounding
      substrate restricted to successive slices of the run's own paper list
      (2 papers, 5 papers, all), then the full substrate.  Comprehension of a
      fixed proof set must not fall as the corpus grows.
  (b) Separation: from the recorded verdict distributions — weak-extraction
      may appear at any comprehension, weak-proof must appear ONLY at
      comp >= hi.  A synthetic probe (the verdict() gate applied to a
      comprehended proof with genuine ungrounded moves vs an incomprehended
      one with the same gap count) records both classes side by side so the
      separation is visible even when the real pilot corpus contains no
      comprehended proof yet.

Writes one JSON record; exit code 0 on go, 3 on no-go.

Usage:
  futon6/.venv/bin/python scripts/comprehension_pilot.py \
      [--graphs data/iatc-argument-graphs/loop-run-70b] \
      [--out holes/excursions/E-comprehension-foundation.pilot.json]
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import clean_comprehension as cc  # noqa: E402

SLICES = [2, 5, "all", "full-substrate"]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def run_floor(graphs, papers_file, out, full=False):
    """One clean_comprehension run; returns its parsed report."""
    cmd = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/clean_comprehension.py"),
           "--graphs", graphs, "--out", str(out)]
    if papers_file is not None:
        cmd += ["--substrate-papers", str(papers_file)]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"clean_comprehension failed: {res.stderr[-400:]}")
    return json.load(open(out))


def verdict_dist(report):
    dist = {}
    for r in report["proofs"]:
        dist[r["verdict"]] = dist.get(r["verdict"], 0) + 1
    return dist


def separation_probe(lo, hi):
    """The gate applied to the SAME gap under two comprehension levels: the
    disambiguation a bare hole-count can never make."""
    v_low, c_low = cc.verdict(0.2, 0.3, 5, 0, lo, hi)     # not understood, 5 gaps
    v_high, c_high = cc.verdict(0.95, 0.9, 5, 0, lo, hi)  # understood, same 5 gaps
    return {"same_gaps": 5,
            "low_comprehension": {"noun": 0.2, "strategy": 0.3, "comp": c_low,
                                  "verdict": v_low,
                                  "path": cc.next_path(c_low, hi)},
            "high_comprehension": {"noun": 0.95, "strategy": 0.9, "comp": c_high,
                                   "verdict": v_high,
                                   "path": cc.next_path(c_high, hi)}}


def go_no_go(sizes, probe, hi):
    """Pure decision from the recorded evidence (unit-tested)."""
    comps = [s["mean_comprehension"] for s in sizes]
    comps = [c for c in comps if c is not None]
    rises = all(b >= a - 1e-9 for a, b in zip(comps, comps[1:])) and comps[-1] > comps[0]
    low_separated = probe["low_comprehension"]["verdict"] == "weak-extraction" and \
        probe["low_comprehension"]["path"] == "study-more/ask-for-help"
    high_separated = probe["high_comprehension"]["verdict"] == "weak-proof" and \
        probe["high_comprehension"]["comp"] is not None and \
        probe["high_comprehension"]["comp"] >= hi
    no_weak_proof_below_hi = all(
        r["verdict"] != "weak-proof" or (r["comprehension"] or 0) >= hi
        for s in sizes for r in s["report"]["proofs"])
    return {"rises_with_corpus": rises,
            "separates_weak_extraction_from_weak_proof": low_separated and high_separated,
            "no_weak_proof_below_hi": no_weak_proof_below_hi,
            "go": rises and low_separated and high_separated and no_weak_proof_below_hi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="data/iatc-argument-graphs/loop-run-70b")
    ap.add_argument("--lo", type=float, default=0.5)
    ap.add_argument("--hi", type=float, default=0.8)
    ap.add_argument("--out", default="holes/excursions/E-comprehension-foundation.pilot.json")
    args = ap.parse_args()

    papers = sorted(p.stem for p in (ROOT / args.graphs).glob("*.edn")
                    if not p.name.endswith(".rung2.edn"))
    sizes = []
    with tempfile.TemporaryDirectory() as td:
        for slice_id in SLICES:
            papers_file = None
            if slice_id == "all":
                papers_file = Path(td) / "papers-all.txt"
                papers_file.write_text("\n".join(papers))
            elif isinstance(slice_id, int):
                papers_file = Path(td) / f"papers-{slice_id}.txt"
                papers_file.write_text("\n".join(papers[:slice_id]))
            out = Path(td) / f"floor-{slice_id}.json"
            report = run_floor(args.graphs, papers_file, out)
            sizes.append({"corpus": str(slice_id), "n_papers_scoped":
                          (len(papers) if papers_file is None else
                           len(papers_file.read_text().split())),
                          "mean_comprehension": mean([r["comprehension"] for r in report["proofs"]]),
                          "verdict_distribution": verdict_dist(report),
                          "per_proof": {r["pid"]: r["comprehension"] for r in report["proofs"]},
                          "report": report})
    probe = separation_probe(args.lo, args.hi)
    decision = go_no_go(sizes, probe, args.hi)
    record = {"schema": "comprehension-pilot-v1",
              "question": "does the corpus-relative comprehension score rise with "
                          "corpus growth and separate weak-extraction from weak-proof?",
              "graphs": args.graphs, "lo": args.lo, "hi": args.hi,
              "sizes": [{k: v for k, v in s.items() if k != "report"} for s in sizes],
              "separation_probe": probe, "decision": decision,
              "go_no_go": "go" if decision["go"] else "no-go"}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(record, open(out, "w"), indent=2, sort_keys=True)
    print(f"wrote {out}")
    print("mean comprehension by corpus size:",
          [(s["corpus"], s["mean_comprehension"]) for s in sizes])
    print("separation probe:", probe["low_comprehension"]["verdict"], "vs",
          probe["high_comprehension"]["verdict"])
    print("go/no-go:", record["go_no_go"], decision)
    sys.exit(0 if decision["go"] else 3)


if __name__ == "__main__":
    main()
