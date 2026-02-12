#!/usr/bin/env python3
"""
Direction D probe for GPL-H: near-rank-1 reformulation diagnostics.

This script measures how close Y_t(v) is to rank-1 along Case-2b barrier
trajectories, with emphasis on the rank-gap metric

    gap_t(v) := tr(Y_t(v)) / ||Y_t(v)||.

A value near 1 indicates rank-1 dominance.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np


@dataclass
class StepRow:
    graph: str
    family: str
    n: int
    eps: float
    t: int
    r_t: int
    active: int
    min_score: float
    mean_score: float
    mean_drift: float
    mean_gap: float
    p90_gap: float
    max_gap: float
    mean_rank1_ratio: float


def load_base_module(repo_root: Path):
    base_path = repo_root / "scripts" / "verify-p6-gpl-h.py"
    spec = importlib.util.spec_from_file_location("p6base", base_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def family_name(graph_name: str) -> str:
    if graph_name.startswith("ER_"):
        return "ER"
    if graph_name.startswith("RandReg_"):
        return "RandReg"
    if graph_name.startswith("DisjCliq_"):
        return "DisjCliq"
    if graph_name.startswith("Dumbbell_"):
        return "Dumbbell"
    if graph_name.startswith("Barbell_"):
        return "Barbell"
    if graph_name.startswith("K_"):
        return "K"
    if graph_name.startswith("C_"):
        return "C"
    return graph_name.split("_", 1)[0]


def build_suite(base, nmax: int, rng):
    suite = []
    for n in range(8, nmax + 1, 4):
        suite.append((f"K_{n}", n, base.complete_graph(n)))
        suite.append((f"C_{n}", n, base.cycle_graph(n)))
        if n >= 8:
            k = n // 2
            bn, be = base.barbell_graph(k)
            suite.append((f"Barbell_{k}", bn, be))
        if n >= 12:
            dn, de = base.dumbbell_graph(n // 3)
            suite.append((f"Dumbbell_{n//3}", dn, de))
            cn, ce = base.disjoint_cliques(n // 3, 3)
            suite.append((f"DisjCliq_{n//3}x3", cn, ce))
        for p_er in [0.3, 0.5]:
            er_edges = base.erdos_renyi(n, p_er, rng)
            if len(er_edges) > n:
                suite.append((f"ER_{n}_p{p_er}", n, er_edges))
        for d_rr in [4, 6]:
            if d_rr < n:
                rr_edges = base.random_regular(n, d_rr, rng)
                if len(rr_edges) > n:
                    suite.append((f"RandReg_{n}_d{d_rr}", n, rr_edges))
    return suite


def analyze_instance(inst, c_step: float):
    n = inst.n
    eps = inst.epsilon
    I0 = inst.I0
    m0 = len(I0)

    edge_idx = {}
    for idx, (u, v) in enumerate(inst.edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx

    T = max(1, min(int(c_step * eps * n), m0 - 1))

    S_t = []
    S_set = set()
    M_t = np.zeros((n, n))

    rows = []

    for t in range(T):
        R_t = [v for v in I0 if v not in S_set]
        r_t = len(R_t)
        if r_t <= 0:
            break

        headroom = eps * np.eye(n) - M_t
        if np.min(np.linalg.eigvalsh(headroom)) < 1e-12:
            break

        B_t = np.linalg.inv(headroom)
        Bsqrt = np.linalg.cholesky(B_t + 1e-14 * np.eye(n))

        scores = []
        active_scores = []
        active_drifts = []
        gaps = []
        rank1_ratios = []

        best_v = None
        best_score = float("inf")

        for v in R_t:
            C_v = np.zeros((n, n))
            for u in S_t:
                key = (min(u, v), max(u, v))
                if key in edge_idx:
                    C_v += inst.X_edges[edge_idx[key]]

            Y_v = Bsqrt @ C_v @ Bsqrt.T
            evals = np.linalg.eigvalsh(Y_v)
            score_v = float(max(evals[-1], 0.0))
            scores.append(score_v)

            if score_v > 1e-10:
                drift_v = float(np.sum(evals[evals > 0]))
                gap_v = drift_v / score_v
                frob_sq = float(np.sum(evals * evals))
                rank1_ratio = (score_v * score_v / frob_sq) if frob_sq > 1e-14 else 1.0

                active_scores.append(score_v)
                active_drifts.append(drift_v)
                gaps.append(gap_v)
                rank1_ratios.append(rank1_ratio)

            if score_v < best_score:
                best_score = score_v
                best_v = v

        if gaps:
            rows.append(
                StepRow(
                    graph=inst.graph_name,
                    family=family_name(inst.graph_name),
                    n=inst.n,
                    eps=inst.epsilon,
                    t=t,
                    r_t=r_t,
                    active=len(gaps),
                    min_score=float(min(scores)) if scores else 0.0,
                    mean_score=float(np.mean(active_scores)),
                    mean_drift=float(np.mean(active_drifts)),
                    mean_gap=float(np.mean(gaps)),
                    p90_gap=float(np.percentile(gaps, 90)),
                    max_gap=float(np.max(gaps)),
                    mean_rank1_ratio=float(np.mean(rank1_ratios)),
                )
            )

        if best_v is None:
            break

        S_t.append(best_v)
        S_set.add(best_v)

        for u in S_t[:-1]:
            key = (min(best_v, u), max(best_v, u))
            if key in edge_idx:
                M_t += inst.X_edges[edge_idx[key]]

    return rows


def q(arr: np.ndarray, p: float) -> float:
    return float(np.quantile(arr, p)) if arr.size else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Direction D probe for GPL-H")
    ap.add_argument("--nmax", type=int, default=48)
    ap.add_argument("--eps", nargs="+", type=float,
                    default=[0.1, 0.12, 0.15, 0.2, 0.25, 0.3])
    ap.add_argument("--c-step", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260212)
    ap.add_argument("--json-out", type=str, default="")
    args = ap.parse_args()

    # Base helper uses np.random.shuffle internally; seed global RNG for reproducibility.
    np.random.seed(args.seed)

    repo_root = Path(__file__).resolve().parents[1]
    base = load_base_module(repo_root)

    rng = np.random.default_rng(args.seed)
    suite = build_suite(base, args.nmax, rng)

    rows: list[StepRow] = []
    case2b = 0

    for graph_name, n, edges in suite:
        for eps in args.eps:
            inst = base.find_case2b_instance(n, edges, eps, graph_name=graph_name)
            if inst is None:
                continue
            case2b += 1
            rows.extend(analyze_instance(inst, c_step=args.c_step))

    if not rows:
        print("No rows collected.")
        return

    gaps_step_mean = np.array([r.mean_gap for r in rows], dtype=float)
    gaps_step_max = np.array([r.max_gap for r in rows], dtype=float)
    rank1 = np.array([r.mean_rank1_ratio for r in rows], dtype=float)
    min_scores = np.array([r.min_score for r in rows], dtype=float)
    mean_scores = np.array([r.mean_score for r in rows], dtype=float)
    mean_drifts = np.array([r.mean_drift for r in rows], dtype=float)

    print("=" * 86)
    print("Direction D: Near-rank-1 diagnostics for GPL-H trajectories")
    print("=" * 86)
    print(f"Case-2b instances: {case2b}")
    print(f"Step rows:         {len(rows)}")
    print()

    print("Global step statistics:")
    print(f"  mean_gap:   mean={np.mean(gaps_step_mean):.4f} p90={q(gaps_step_mean,0.9):.4f} "
          f"p95={q(gaps_step_mean,0.95):.4f} max={np.max(gaps_step_mean):.4f}")
    print(f"  max_gap:    mean={np.mean(gaps_step_max):.4f} p90={q(gaps_step_max,0.9):.4f} "
          f"p95={q(gaps_step_max,0.95):.4f} max={np.max(gaps_step_max):.4f}")
    print(f"  rank1-ratio mean={np.mean(rank1):.4f} p10={q(rank1,0.1):.4f} "
          f"p05={q(rank1,0.05):.4f} min={np.min(rank1):.4f}")
    print(f"  min_score:  max={np.max(min_scores):.6f}")
    print(f"  mean_score: max={np.max(mean_scores):.6f}")
    print(f"  mean_drift: mean={np.mean(mean_drifts):.6f} p95={q(mean_drifts,0.95):.6f} max={np.max(mean_drifts):.6f}")
    print()

    for th in [1.02, 1.05, 1.10, 1.20, 1.40]:
        frac = float(np.mean(gaps_step_max <= th))
        print(f"max_gap <= {th:.2f}: {frac:.4f}")

    print("\nFamily breakdown:")
    families = sorted(set(r.family for r in rows))
    for fam in families:
        rr = [r for r in rows if r.family == fam]
        mg = np.array([r.mean_gap for r in rr], dtype=float)
        xg = np.array([r.max_gap for r in rr], dtype=float)
        rk = np.array([r.mean_rank1_ratio for r in rr], dtype=float)
        print(
            f"  {fam:<10} rows={len(rr):>4} "
            f"mean_gap_p95={q(mg,0.95):.3f} max_gap_max={np.max(xg):.3f} "
            f"rank1_mean={np.mean(rk):.3f}"
        )

    print("\nTop hard rows by max_gap:")
    hard = sorted(rows, key=lambda r: r.max_gap, reverse=True)[:12]
    for r in hard:
        print(
            f"  {r.graph:<16} n={r.n:>2} eps={r.eps:>4.2f} t={r.t:>2} "
            f"active={r.active:>2} mean_gap={r.mean_gap:.3f} "
            f"max_gap={r.max_gap:.3f} mean_score={r.mean_score:.3f} mean_drift={r.mean_drift:.3f}"
        )

    if args.json_out:
        out = {
            "case2b_instances": case2b,
            "rows": [asdict(r) for r in rows],
            "summary": {
                "mean_gap_mean": float(np.mean(gaps_step_mean)),
                "mean_gap_p95": q(gaps_step_mean, 0.95),
                "mean_gap_max": float(np.max(gaps_step_mean)),
                "max_gap_mean": float(np.mean(gaps_step_max)),
                "max_gap_p95": q(gaps_step_max, 0.95),
                "max_gap_max": float(np.max(gaps_step_max)),
                "rank1_ratio_mean": float(np.mean(rank1)),
                "rank1_ratio_p05": q(rank1, 0.05),
                "rank1_ratio_min": float(np.min(rank1)),
                "min_score_max": float(np.max(min_scores)),
                "mean_score_max": float(np.max(mean_scores)),
                "mean_drift_mean": float(np.mean(mean_drifts)),
                "mean_drift_p95": q(mean_drifts, 0.95),
                "mean_drift_max": float(np.max(mean_drifts)),
            },
        }
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"\nWrote JSON: {out_path}")


if __name__ == "__main__":
    main()
