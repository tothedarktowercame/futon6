#!/usr/bin/env python3
"""Substance gate for mark3 Codex-pool output — the anti-shell-gaming check.

The structural checkers (concept_argcheck.bb, iatc_argcheck.bb) verify
well-formedness and reference resolution, but NOT whether real work happened.
Twice the pool emitted shells that passed by construction:
  * H1: 200 concept entries with self-referential `:given [{:var :X}]` and
    "carries the structure described by the corpus gloss" filler.
  * H2: 181 IATC graphs all with the identical 2-node / 1-edge / 2-hole shape.

This gate complements (does not replace) the structural checkers. Run BOTH.
It operates on a directory (a batch) because the strongest signals are
cross-item: a deterministic shell-emitter produces a near-uniform structural
distribution and reuses a handful of canned warrants, whereas genuine
reconstruction varies per passage.

Usage:
    python scripts/substance_gate.py <dir-or-file> [...] [--kind auto|concept|iatc]
    python scripts/substance_gate.py --self-check

Exit codes: 0 all pass · 1 substance failure · 2 bad input.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

# --- thresholds (tunable) ---
MIN_SET_FOR_DIST = 3          # below this, skip cross-item distribution checks
DOMINANT_SHAPE_FRAC = 0.80    # >= this share on one (nodes,edges,holes) triple => template collapse
DOMINANT_WARRANT_FRAC = 0.60  # >= this share of holes on one :wanted bucket => canned warrants

FILLER_PATTERNS = [
    (re.compile(r":var\s+:X\b"), "self-referential placeholder :var :X"),
    (re.compile(r"carries the structure described by"), "carrier boilerplate :data"),
    (re.compile(r':signature\s+"X\s*:\s'), "self-referential signature \"X : <name>\""),
]


def detect_kind(text: str) -> str:
    if ":nodes" in text and ":edges" in text:
        return "iatc"
    if ":concept/id" in text:
        return "concept"
    return "unknown"


def iatc_features(text: str) -> dict:
    nodes_seg = text[text.find(":nodes"): text.find(":edges")] if ":edges" in text else ""
    holes_at = text.rfind(":holes")
    edges_seg = text[text.find(":edges"): holes_at] if holes_at > 0 else text[text.find(":edges"):]
    holes_seg = text[holes_at:] if holes_at > 0 else ""
    n_nodes = nodes_seg.count(":id ")
    n_edges = edges_seg.count(":kind :infer")
    n_holes = holes_seg.count("{:kind")
    wanted = re.findall(r":wanted\s+(:[\w./-]+)", text)
    return {"nodes": n_nodes, "edges": n_edges, "holes": n_holes, "wanted": wanted}


def check_concept_item(path: Path, text: str) -> list[str]:
    fails = []
    for pat, why in FILLER_PATTERNS:
        if pat.search(text):
            fails.append(f"filler: {why}")
    holes_empty = re.search(r":holes\s*\[\s*\]", text) is not None
    has_axiom = ":statement" in text
    has_data = re.search(r":data\s*\[\s*\{", text) is not None
    if holes_empty and not (has_axiom or has_data):
        fails.append("vacuous: :holes [] but no :axioms :statement and no :data")
    return fails


def check_iatc_item(path: Path, text: str, feats: dict) -> list[str]:
    fails = []
    for pat, why in FILLER_PATTERNS:
        if pat.search(text):
            fails.append(f"filler: {why}")
    if feats["nodes"] < 2:
        fails.append(f"thin: {feats['nodes']} node(s) — no argument structure")
    if feats["edges"] < 1:
        fails.append("thin: 0 inference edges")
    return fails


def gate_dir(files: list[Path], kind: str) -> tuple[bool, list[str]]:
    """Return (ok, report-lines) for one homogeneous batch."""
    lines = []
    ok = True
    parsed = []
    for f in sorted(files):
        text = f.read_text()
        k = kind if kind != "auto" else detect_kind(text)
        parsed.append((f, text, k))

    # per-item
    iatc_feats = []
    for f, text, k in parsed:
        if k == "concept":
            item_fails = check_concept_item(f, text)
        elif k == "iatc":
            feats = iatc_features(text)
            iatc_feats.append(feats)
            item_fails = check_iatc_item(f, text, feats)
        else:
            item_fails = [f"unknown artifact kind (no :nodes/:edges or :concept/id)"]
        if item_fails:
            ok = False
            lines.append(f"FAIL {f}")
            lines += [f"  [item] {m}" for m in item_fails]
        else:
            lines.append(f"PASS {f}")

    # cross-item (IATC distribution + warrant reuse)
    if iatc_feats and len(iatc_feats) >= MIN_SET_FOR_DIST:
        shapes = Counter((x["nodes"], x["edges"], x["holes"]) for x in iatc_feats)
        top_shape, top_n = shapes.most_common(1)[0]
        frac = top_n / len(iatc_feats)
        if frac >= DOMINANT_SHAPE_FRAC:
            ok = False
            lines.append(
                f"FAIL [set] structural template collapse: {top_n}/{len(iatc_feats)} "
                f"({frac:.0%}) graphs share shape (nodes,edges,holes)={top_shape} "
                f">= {DOMINANT_SHAPE_FRAC:.0%} — looks emitted, not reconstructed"
            )
        all_wanted = [w for x in iatc_feats for w in x["wanted"]]
        if len(all_wanted) >= MIN_SET_FOR_DIST:
            wc = Counter(all_wanted)
            tw, tn = wc.most_common(1)[0]
            wfrac = tn / len(all_wanted)
            if wfrac >= DOMINANT_WARRANT_FRAC:
                ok = False
                lines.append(
                    f"FAIL [set] canned warrants: {tn}/{len(all_wanted)} ({wfrac:.0%}) "
                    f"missing-warrants reuse {tw} >= {DOMINANT_WARRANT_FRAC:.0%} — "
                    f"warrants must name the SPECIFIC elided justification"
                )
    return ok, lines


def collect(paths: list[str]) -> list[Path]:
    out = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            out += sorted(pp.glob("*.edn"))
        elif pp.is_file():
            out.append(pp)
    return out


def run(paths: list[str], kind: str) -> int:
    files = collect(paths)
    if not files:
        print("substance_gate: no .edn files found", file=sys.stderr)
        return 2
    ok, lines = gate_dir(files, kind)
    print("\n".join(lines))
    n_fail = sum(1 for l in lines if l.startswith("FAIL"))
    print(f"\nsubstance-gate: {len(files)} file(s), {n_fail} failure line(s) — "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


FIX = Path(__file__).resolve().parent.parent / "holes" / "substance-gate" / "fixtures"


def self_check() -> int:
    cases = [
        ("good-iatc", "iatc", True),
        ("good-concept", "concept", True),
        ("bad-iatc", "iatc", False),
        ("bad-concept", "concept", False),
    ]
    all_ok = True
    for sub, kind, expect_pass in cases:
        d = FIX / sub
        files = collect([str(d)])
        if not files:
            print(f"SELF-CHECK MISSING fixtures: {d}")
            all_ok = False
            continue
        ok, _ = gate_dir(files, kind)
        good = ok == expect_pass
        all_ok = all_ok and good
        print(f"  {sub:14} expect={'PASS' if expect_pass else 'FAIL'} "
              f"got={'PASS' if ok else 'FAIL'}  {'OK' if good else '*** WRONG ***'}")
    print("SELF-CHECK", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--kind", choices=["auto", "concept", "iatc"], default="auto")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        return self_check()
    if not a.paths:
        ap.error("give a dir/file or --self-check")
    return run(a.paths, a.kind)


if __name__ == "__main__":
    sys.exit(main())
