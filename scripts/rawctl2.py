"""RAW-CTL re-analysis using the eval harness's OWN metric definition."""
import os, sys, subprocess, re
from pathlib import Path
sys.path.insert(0, "scripts")
from run_artifacts import proof_graphs
from mark3_eval_harness import warrant_resolution_counts

RAW = "data/exp-20260618/loop-run-70b-raw"
ENR = "data/exp-20260618/loop-run-70b"
papers = lambda d: {os.path.basename(p)[:-4] for p in proof_graphs(d)}
both = sorted(papers(RAW) & papers(ENR))

print(f"paper-matched on {len(both)} papers present in both arms\n")
print(f"{'arm':10s} {'files':>5s} {'resolved':>9s} {'edges':>6s} {'grounding':>10s}")
out = {}
for label, d in (("enriched", ENR), ("raw", RAW)):
    files = [Path(os.path.join(d, f"{p}.edn")) for p in both]
    res, tot = warrant_resolution_counts(files)
    g = 100.0 * res / tot if tot else 0.0
    out[label] = (res, tot, g)
    print(f"{label:10s} {len(files):5d} {res:9d} {tot:6d} {g:9.1f}%")

(re_, te, ge), (rr, tr, gr) = out["enriched"], out["raw"]
print(f"\nharness metric — enriched {ge:.1f}% ({re_}/{te}) vs raw {gr:.1f}% ({rr}/{tr})")
print(f"delta {ge-gr:+.1f} pts on {te+tr} total inference edges across both arms")

# what the ORIGINAL report claimed, for the record
print("\noriginal 2026-06-18 report: raw 12.50% (4/32), enriched 21.4%")
print("  - counted 20 'artifacts' for 10 papers (graphs + .rung2.edn reports)")
print("  - compared a 10-paper raw arm against an 8-paper enriched arm")
