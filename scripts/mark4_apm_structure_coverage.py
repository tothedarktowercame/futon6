#!/usr/bin/env python3
"""mark4 apm-structure-match — final pipeline stage (CPU).

Coverage of APM proof scopes by literature (eprint) scopes: for each prelim proof,
what fraction of its typed scope-hyperedges is matched by some eprint scope. Three
match flavors, coarsest -> tightest, so the saturation artifact is visible:
  - type-only            : same hx/type exists in the pool        (loose ceiling)
  - type + any symbol    : + >=1 shared symbol token              (single letters saturate)
  - type + multichar sym : + >=1 shared >=3-char symbol token     (first discriminative cut)

This is the EASY-DEFAULT matcher (Joe: "later we'll need something much better" =
symbol-class / role-typed overlap, or pgvector embedding per Rob's neo4j+pgvector
pattern). Inputs are produced by nlab-wiring.detect_scopes over the proof .tex and
the eprint LaTeX.

Usage:
  python scripts/mark4_apm_structure_coverage.py \
    --proof-scopes  /home/joe/code/storage/apm/apm-proof-scopes.json \
    --eprint-scopes /home/joe/code/storage/apm/eprint-scopes.json \
    --out /home/joe/code/storage/apm/mark4-apm-coverage.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

GATE_MULTICHAR_MEAN = 0.20
GATE_MULTICHAR_MEDIAN = 0.10
GATE_TAIL_GE80 = 10

MNUMBER = re.compile(r"\\mNumber\{[^}]*\}")
CTRL = re.compile(r"\\[a-zA-Z]+")
TOK = re.compile(r"[a-zA-Z]\w*")


def symbols(scope: dict) -> set[str]:
    out: set[str] = set()
    for end in scope.get("hx/ends", []):
        for key in ("latex", "text"):
            v = end.get(key)
            if not v:
                continue
            v = CTRL.sub(" ", MNUMBER.sub(" ", v))
            out |= set(TOK.findall(v.lower()))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proof-scopes", type=Path, required=True)
    ap.add_argument("--eprint-scopes", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    proof = json.loads(a.proof_scopes.read_text())
    epr = json.loads(a.eprint_scopes.read_text())

    etypes: set[str] = set()
    etype_syms: dict[str, set[str]] = defaultdict(set)
    etype_multi: dict[str, set[str]] = defaultdict(set)
    for scopes in epr.values():
        for s in scopes:
            t = s.get("hx/type")
            etypes.add(t)
            ss = symbols(s)
            etype_syms[t] |= ss
            etype_multi[t] |= {x for x in ss if len(x) >= 3}

    def frac(scopes, pred):
        return sum(1 for s in scopes if pred(s)) / len(scopes)

    rows = {}
    to, ta, tm = [], [], []
    for pid, scopes in proof.items():
        if not scopes:
            continue
        t_only = frac(scopes, lambda s: s.get("hx/type") in etypes)
        t_any = frac(scopes, lambda s: symbols(s) & etype_syms.get(s.get("hx/type"), set()))
        t_multi = frac(scopes, lambda s: {x for x in symbols(s) if len(x) >= 3}
                       & etype_multi.get(s.get("hx/type"), set()))
        rows[pid] = {"n_scopes": len(scopes), "type_only": t_only,
                     "type_any_symbol": t_any, "type_multichar": t_multi}
        to.append(t_only); ta.append(t_any); tm.append(t_multi)

    buckets = [0, .2, .4, .6, .8, 1.01]
    hist = [0] * (len(buckets) - 1)
    for v in tm:
        for i in range(len(buckets) - 1):
            if buckets[i] <= v < buckets[i + 1]:
                hist[i] += 1
                break

    multi_mean = statistics.mean(tm)
    multi_median = statistics.median(tm)
    tail_ge80 = sum(1 for v in tm if v >= 0.8)
    summary = {
        "proofs_scored": len(to),
        "proofs_zero_scope": sum(1 for v in proof.values() if not v),
        "eprints": len(epr),
        "type_only": {"mean": statistics.mean(to), "median": statistics.median(to)},
        "type_any_symbol": {"mean": statistics.mean(ta), "median": statistics.median(ta)},
        "type_multichar": {"mean": multi_mean, "median": multi_median},
        "type_multichar_hist_0_20_40_60_80_100": hist,
        "confirmatory_tail_ge80pct": tail_ge80,
        "chosen_metric": "type_multichar",
        "gate": {
            "metric": "type_multichar",
            "min_mean": GATE_MULTICHAR_MEAN,
            "min_median": GATE_MULTICHAR_MEDIAN,
            "min_confirmatory_tail_ge80pct": GATE_TAIL_GE80,
            "rationale": (
                "type_only and type_any_symbol saturate; multichar symbols are "
                "the first discriminative cut before role-typed or embedding matchers"
            ),
        },
        "gate_pass": (
            multi_mean >= GATE_MULTICHAR_MEAN
            and multi_median >= GATE_MULTICHAR_MEDIAN
            and tail_ge80 >= GATE_TAIL_GE80
        ),
    }
    print(json.dumps(summary, indent=2))
    if a.out:
        a.out.write_text(json.dumps({"summary": summary, "per_proof": rows}, indent=2))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
