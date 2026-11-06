#!/usr/bin/env python3
"""Compute per-paper tension-scalar T_total + Laplacian summary over a batch.

Reads `<batch>/output/paper-hypergraphs.json` and emits TSV with one row
per paper:

    paper_id  n_claims  n_unpaired  T_total  top_support_id  top_support_count

Where:
  - `n_claims`   = number of `type=="claim"` vertices in the paper's
                   hypergraph (theorems / propositions / lemmas).
  - `n_unpaired` = number of claim vertices with no incident
                   `type=="derivation"` edge in the role `target`.
  - `T_total`    = n_unpaired / n_claims (the v0 paper-level tension).
                   Empty-claim papers report `T_total = -1` (sentinel).
  - `top_support_id` / `top_support_count`: the non-claim vertex co-occurring
                   most often with unpaired claims (a Δ-Laplacian proxy for
                   "load-bearing concept the punchline depends on").

Mission: M-superpod-mark3 (Track B — geometry on existing data).
Excursion: futon3/holes/excursions/E-Ttotal.md.
Pilot validation: E-math-prototype-pilot.md §"Tension scalar demo".

Cheap and pure-Python. Designed to run on the 5k-paper mfuton batches in
under a few seconds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


def compute_one_paper(paper: dict) -> dict:
    """Compute T_total and Laplacian summary for one paper's hypergraph."""
    nodes = paper.get("nodes", [])
    edges = paper.get("edges", [])

    claim_ids: set[str] = set()
    for n in nodes:
        if n.get("type") == "claim":
            claim_ids.add(n.get("id", ""))
    n_claims = len(claim_ids)

    # An unpaired claim has no derivation edge with role `target`.
    grounded_claims: set[str] = set()
    for e in edges:
        if e.get("type") != "derivation":
            continue
        roles = e.get("roles") or {}
        for vid, role in roles.items():
            if role == "target" and vid in claim_ids:
                grounded_claims.add(vid)

    unpaired = claim_ids - grounded_claims
    n_unpaired = len(unpaired)
    T_total = (n_unpaired / n_claims) if n_claims else -1.0

    # Δ-Laplacian proxy: for each non-claim vertex v adjacent (via any edge)
    # to an unpaired claim u, increment counter[v]. The argmax is the
    # vertex most "exposed" to the paper's open tensions — a candidate
    # load-bearing concept.
    incidence: Counter[str] = Counter()
    for e in edges:
        ends = e.get("ends") or list((e.get("roles") or {}).keys())
        if not ends:
            continue
        unpaired_in_edge = [v for v in ends if v in unpaired]
        if not unpaired_in_edge:
            continue
        non_claim_in_edge = [v for v in ends if v not in claim_ids]
        for v in non_claim_in_edge:
            incidence[v] += len(unpaired_in_edge)

    if incidence:
        top_id, top_count = incidence.most_common(1)[0]
    else:
        top_id, top_count = "", 0

    return {
        "paper_id": paper.get("paper_id", ""),
        "n_claims": n_claims,
        "n_unpaired": n_unpaired,
        "T_total": T_total,
        "top_support_id": top_id,
        "top_support_count": top_count,
        # Geometry-artifact fields (mark3, mission M-superpod-mark3 R-3
        # geometry artifact): the explicit unpaired-claim ids and the full
        # Δ-Laplacian incidence map, sorted for deterministic output.
        "unpaired_claim_ids": sorted(unpaired),
        "laplacian_summary": {
            "incidence": {k: incidence[k] for k in sorted(incidence)},
            "top_support_id": top_id,
            "top_support_count": top_count,
        },
    }


GEOMETRY_SCHEMA_VERSION = "mark3-geometry-v1"


def compute_batch_geometry(rows: list[dict]) -> dict:
    """Batch-level geometry: tension and Laplacian summary over all papers.

    Batch tension is total_unpaired / total_claims across the whole batch
    (empty-claim papers contribute no claims), with the per-paper mean over
    papers that have claims reported alongside.  The batch Laplacian summary
    aggregates the per-paper incidence maps: a concept ranked high here is
    adjacent to unpaired claims across many papers.
    """
    with_claims = [r for r in rows if r["n_claims"] > 0]
    total_claims = sum(r["n_claims"] for r in rows)
    total_unpaired = sum(r["n_unpaired"] for r in rows)
    agg: Counter[str] = Counter()
    for r in rows:
        for vid, count in r["laplacian_summary"]["incidence"].items():
            agg[vid] += count
    if agg:
        top_id, top_count = agg.most_common(1)[0]
    else:
        top_id, top_count = "", 0
    return {
        "n_papers": len(rows),
        "n_papers_with_claims": len(with_claims),
        "total_claims": total_claims,
        "total_unpaired": total_unpaired,
        "batch_T_total": (total_unpaired / total_claims) if total_claims else -1.0,
        "mean_paper_T_total": (
            sum(r["T_total"] for r in with_claims) / len(with_claims)
        ) if with_claims else -1.0,
        "laplacian_summary": {
            "incidence": {k: agg[k] for k in sorted(agg)},
            "top_support_id": top_id,
            "top_support_count": top_count,
        },
    }


def build_geometry_document(papers: list[dict], rows: list[dict]) -> dict:
    """Assemble the deterministic geometry.json artifact for a batch.

    No timestamps: running the generator twice over the same input yields
    byte-identical output (fixed-point requirement for generated artifacts).
    """
    return {
        "schema_version": GEOMETRY_SCHEMA_VERSION,
        "n_papers": len(rows),
        "batch": compute_batch_geometry(rows),
        "papers": [
            {
                "paper_id": r["paper_id"],
                "n_claims": r["n_claims"],
                "n_unpaired": r["n_unpaired"],
                "T_total": r["T_total"],
                "unpaired_claim_ids": r["unpaired_claim_ids"],
                "laplacian_summary": r["laplacian_summary"],
            }
            for r in rows
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("batch_root", type=Path,
                    help="Path to the batch directory (containing output/paper-hypergraphs.json)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output TSV path (default: <batch_root>/output/paper-T.tsv)")
    ap.add_argument("--json-out", type=Path, default=None,
                    help="Output geometry.json path (default: "
                         "<batch_root>/output/geometry.json; use --no-json to skip)")
    ap.add_argument("--no-json", action="store_true",
                    help="Do not emit the geometry.json artifact (TSV only)")
    args = ap.parse_args()

    pdir = args.batch_root
    if (pdir / "output" / "paper-hypergraphs.json").exists():
        pdir = pdir / "output"
    elif not (pdir / "paper-hypergraphs.json").exists():
        print(f"[error] paper-hypergraphs.json not found under {args.batch_root}",
              file=sys.stderr)
        return 2
    src = pdir / "paper-hypergraphs.json"
    out = args.out or (pdir / "paper-T.tsv")

    print(f"[compute-paper-T] reading {src}", file=sys.stderr)
    with src.open() as f:
        papers = json.load(f)
    print(f"[compute-paper-T] {len(papers)} papers", file=sys.stderr)

    rows = [compute_one_paper(p) for p in papers]

    with out.open("w") as f:
        f.write("paper_id\tn_claims\tn_unpaired\tT_total\ttop_support_id\ttop_support_count\n")
        for r in rows:
            f.write(f"{r['paper_id']}\t{r['n_claims']}\t{r['n_unpaired']}"
                    f"\t{r['T_total']:.4f}\t{r['top_support_id']}"
                    f"\t{r['top_support_count']}\n")
    print(f"[compute-paper-T] wrote {out}", file=sys.stderr)

    if not args.no_json:
        json_out = args.json_out or (pdir / "geometry.json")
        geometry = build_geometry_document(papers, rows)
        tmp_json = json_out.with_suffix(json_out.suffix + ".tmp")
        with tmp_json.open("w") as f:
            json.dump(geometry, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_json, json_out)
        print(f"[compute-paper-T] wrote {json_out}", file=sys.stderr)

    # Distribution summary to stderr.
    valid = [r["T_total"] for r in rows if r["T_total"] >= 0]
    if valid:
        valid.sort()
        n = len(valid)
        mean = sum(valid) / n

        def pct(p):
            idx = max(0, min(n - 1, int(round(p * (n - 1)))))
            return valid[idx]

        print(f"[compute-paper-T] T_total distribution over {n} papers"
              f" (excluding {len(rows) - n} empty-claim papers):", file=sys.stderr)
        print(f"  mean      = {mean:.3f}", file=sys.stderr)
        print(f"  p10       = {pct(0.10):.3f}", file=sys.stderr)
        print(f"  p25       = {pct(0.25):.3f}", file=sys.stderr)
        print(f"  median    = {pct(0.50):.3f}", file=sys.stderr)
        print(f"  p75       = {pct(0.75):.3f}", file=sys.stderr)
        print(f"  p90       = {pct(0.90):.3f}", file=sys.stderr)
        n_zero = sum(1 for v in valid if v == 0)
        n_one = sum(1 for v in valid if v >= 0.999)
        print(f"  zero      = {n_zero}  ({100*n_zero/n:.1f}%)", file=sys.stderr)
        print(f"  one       = {n_one}  ({100*n_one/n:.1f}%)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
