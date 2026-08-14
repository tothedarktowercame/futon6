"""RAW-CTL: enriched-vs-raw control arm, finals-only, paper-matched.

The 2026-06-18 run exists; only its analytic was wrong. That report globbed a
directory containing both proof graphs and `.rung2.edn` reports, so it counted
20 artifacts for 10 papers and scored substance 10/20 -- the 10 "failures" were
the reports. It also compared arms with different paper sets (10 vs 8).

This recomputes on finals only, over the papers present in BOTH arms.
"""
import json, os, re, subprocess, sys
sys.path.insert(0, "scripts")
from run_artifacts import proof_graphs

ROOT = os.getcwd()
RAW = "data/exp-20260618/loop-run-70b-raw"
ENR = "data/exp-20260618/loop-run-70b"

def papers(d):
    return {os.path.basename(p)[:-4] for p in proof_graphs(d)}

def warrants(path):
    """(resolved, total) warrant edges in one graph."""
    t = open(path, errors="replace").read()
    kinds = re.findall(r":warrant \{:kind :([a-z-]+)", t)
    total = len(kinds)
    resolved = sum(1 for k in kinds if k != "missing-warrant")
    return resolved, total

def gate(files):
    p = subprocess.run([".venv/bin/python", "scripts/substance_gate.py", *files],
                       capture_output=True, text=True, cwd=ROOT)
    m = re.search(r"(\d+) file\(s\), (\d+) failure line\(s\)", p.stdout + p.stderr)
    return (int(m.group(1)), int(m.group(2))) if m else (len(files), -1)

both = sorted(papers(RAW) & papers(ENR))
print(f"papers in both arms: {len(both)}  ({', '.join(both)})\n")
print(f"{'arm':10s} {'graphs':>6s} {'nodes':>6s} {'edges':>6s} {'warrants':>9s} "
      f"{'resolved':>9s} {'grounding':>10s} {'substance':>11s}")
rows = {}
for label, d in (("enriched", ENR), ("raw", RAW)):
    files = [os.path.join(d, f"{p}.edn") for p in both]
    res = tot = nodes = edges = 0
    for f in files:
        r, t = warrants(f)
        res += r; tot += t
        txt = open(f, errors="replace").read()
        nodes += len(re.findall(r":kind :(?:object|claim|ref)", txt))
        edges += len(re.findall(r":kind :infer", txt))
    n, fails = gate(files)
    g = 100.0 * res / tot if tot else 0.0
    rows[label] = (g, res, tot, nodes, edges)
    print(f"{label:10s} {len(files):6d} {nodes:6d} {edges:6d} {tot:9d} {res:9d} "
          f"{g:9.1f}% {n - fails:6d}/{n:<4d}")

ge, gr = rows["enriched"][0], rows["raw"][0]
print(f"\nwarrant grounding: enriched {ge:.1f}% vs raw {gr:.1f}%  "
      f"(delta {ge - gr:+.1f} pts, {ge/gr if gr else float('inf'):.2f}x)")
print(f"graph size: enriched {rows['enriched'][3]} nodes / {rows['enriched'][4]} edges; "
      f"raw {rows['raw'][3]} nodes / {rows['raw'][4]} edges")
