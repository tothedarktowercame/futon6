#!/usr/bin/env python3
"""Cycle 1 verifier for the remaining P6 gap: ||M_t|| <= c*epsilon.

This script does two things:
1) Formal algebra checks for the correction decomposition:
     d_bar = uniform_d_bar + correction
   and
     correction <= (||M_t|| / (eps*(eps-||M_t||))) * tr(F_t)/r_t
     d_bar <= uniform_d_bar * eps/(eps-||M_t||)
2) Stress-tests candidate constants c over a graph/epsilon suite.

The script reuses diagnose() from verify-p6-dbar-mechanism.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_mechanism_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "verify-p6-dbar-mechanism.py"
    spec = importlib.util.spec_from_file_location("p6_mechanism", script_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def complete_graph(n: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def cycle_graph(n: int) -> List[Tuple[int, int]]:
    return [(i, (i + 1) % n) for i in range(n)]


def path_graph(n: int) -> List[Tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]


def barbell_graph(k: int) -> Tuple[int, List[Tuple[int, int]]]:
    n = 2 * k
    edges = []
    for i in range(k):
        for j in range(i + 1, k):
            edges.append((i, j))
    for i in range(k, n):
        for j in range(i + 1, n):
            edges.append((i, j))
    edges.append((k - 1, k))
    return n, edges


def erdos_renyi(n: int, p: float, rng: np.random.Generator) -> List[Tuple[int, int]]:
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((i, j))
    return edges


def build_graph_suite(seed: int) -> List[Tuple[str, int, List[Tuple[int, int]]]]:
    rng = np.random.default_rng(seed)
    suite: List[Tuple[str, int, List[Tuple[int, int]]]] = []

    # Deterministic families
    for n in [20, 40, 60, 80]:
        suite.append((f"K_{n}", n, complete_graph(n)))
    for n in [40, 80]:
        suite.append((f"C_{n}", n, cycle_graph(n)))
        suite.append((f"P_{n}", n, path_graph(n)))
    for k in [20, 30, 40]:
        n, edges = barbell_graph(k)
        suite.append((f"Barbell_{k}", n, edges))

    # Random families
    for n in [40, 60, 80]:
        for p in [0.2, 0.4, 0.6]:
            for rep in range(2):
                edges = erdos_renyi(n, p, rng)
                # Keep only connected-ish dense enough samples to avoid trivial cases
                if len(edges) >= n:
                    suite.append((f"ER_{n}_p{p}_r{rep}", n, edges))

    return suite


def progress_bucket(progress: float) -> str:
    if progress < 0.33:
        return "early"
    if progress < 0.67:
        return "mid"
    return "late"


def run_cycle1(seed: int, eps_list: List[float]) -> Dict:
    mech = load_mechanism_module()
    graphs = build_graph_suite(seed)

    steps = []
    algebra_failures = {
        "decomp_exact": 0,
        "corr_bound": 0,
        "dbar_bound": 0,
    }
    tol = 1e-8

    for gname, n, edges in graphs:
        for eps in eps_list:
            diag = mech.diagnose(n, edges, eps, gname)
            if not diag:
                continue

            for row in diag:
                if row["t"] == 0:
                    continue

                r_t = float(row["r_t"])
                dbar = float(row["dbar"])
                tr_f = float(row["tr_F"])
                norm_m = float(row["norm_M"])
                m0 = float(row["m0"])

                uniform = tr_f / (eps * r_t) if r_t > 0 else 0.0
                exact_corr = dbar - uniform

                # Spectral correction formula:
                # correction = (1/r_t) sum_i [lambda_i/(eps*(eps-lambda_i))] * (u_i^T F_t u_i)
                corr_spectral = 0.0
                for c in row["contributions"]:
                    lam = max(0.0, float(c["lam_M"]))
                    f_w = max(0.0, float(c["f_weight"]))
                    if eps - lam <= 1e-12:
                        continue
                    corr_spectral += (lam / (eps * (eps - lam))) * f_w
                corr_spectral /= r_t

                if abs(exact_corr - corr_spectral) > 1e-6:
                    algebra_failures["decomp_exact"] += 1

                corr_bound = np.inf
                dbar_bound = np.inf
                if eps - norm_m > 1e-12:
                    corr_bound = (norm_m / (eps * (eps - norm_m))) * (tr_f / r_t)
                    dbar_bound = uniform * eps / (eps - norm_m)

                if exact_corr - corr_bound > 1e-6:
                    algebra_failures["corr_bound"] += 1
                if dbar - dbar_bound > 1e-6:
                    algebra_failures["dbar_bound"] += 1

                horizon = max(1.0, eps * m0 / 3.0)
                progress = min(1.0, float(row["t"]) / horizon)

                steps.append(
                    {
                        "graph": gname,
                        "n": n,
                        "eps": eps,
                        "t": int(row["t"]),
                        "r_t": int(row["r_t"]),
                        "m0": int(row["m0"]),
                        "progress": progress,
                        "bucket": progress_bucket(progress),
                        "dbar": dbar,
                        "uniform_dbar": uniform,
                        "exact_corr": exact_corr,
                        "corr_spectral": corr_spectral,
                        "corr_bound": corr_bound,
                        "dbar_bound": dbar_bound,
                        "norm_m": norm_m,
                        "ratio_m_eps": norm_m / eps if eps > 0 else np.inf,
                        "gap_frac": (eps - norm_m) / eps if eps > 0 else np.inf,
                    }
                )

    if not steps:
        raise RuntimeError("No nontrivial steps found. Adjust graph suite or eps list.")

    ratio_all = np.array([s["ratio_m_eps"] for s in steps], dtype=float)
    dbar_all = np.array([s["dbar"] for s in steps], dtype=float)
    unif_all = np.array([s["uniform_dbar"] for s in steps], dtype=float)
    gap_all = np.array([s["gap_frac"] for s in steps], dtype=float)

    # Candidate c tests
    c_candidates = [0.20, 0.25, 0.30, 1.0 / 3.0, 0.35, 0.40, 0.50]

    def summarize_pass(mask):
        if np.sum(mask) == 0:
            return {}
        sub = ratio_all[mask]
        out = {}
        for c in c_candidates:
            out[f"{c:.6f}"] = bool(np.all(sub <= c + 1e-10))
        out["max_ratio"] = float(np.max(sub))
        out["q99_ratio"] = float(np.quantile(sub, 0.99))
        out["count"] = int(np.sum(mask))
        return out

    idx_early = np.array([s["bucket"] == "early" for s in steps], dtype=bool)
    idx_mid = np.array([s["bucket"] == "mid" for s in steps], dtype=bool)
    idx_late = np.array([s["bucket"] == "late" for s in steps], dtype=bool)
    idx_pre80 = np.array([s["progress"] <= 0.80 for s in steps], dtype=bool)
    idx_pre90 = np.array([s["progress"] <= 0.90 for s in steps], dtype=bool)

    summary = {
        "seed": seed,
        "eps_list": eps_list,
        "num_graphs": len(graphs),
        "num_steps": len(steps),
        "algebra_failures": algebra_failures,
        "global": {
            "max_ratio_m_eps": float(np.max(ratio_all)),
            "q99_ratio_m_eps": float(np.quantile(ratio_all, 0.99)),
            "max_dbar": float(np.max(dbar_all)),
            "max_uniform_dbar": float(np.max(unif_all)),
            "min_gap_frac": float(np.min(gap_all)),
            "worst_ratio_case": max(steps, key=lambda s: s["ratio_m_eps"]),
            "worst_dbar_case": max(steps, key=lambda s: s["dbar"]),
        },
        "c_pass": {
            "all_steps": summarize_pass(np.ones(len(steps), dtype=bool)),
            "pre80": summarize_pass(idx_pre80),
            "pre90": summarize_pass(idx_pre90),
            "early": summarize_pass(idx_early),
            "mid": summarize_pass(idx_mid),
            "late": summarize_pass(idx_late),
        },
    }

    return {"summary": summary, "steps": steps}


def print_human_summary(results: Dict):
    s = results["summary"]
    g = s["global"]
    print("=" * 88)
    print("P6 CYCLE-1 CODEx CHECK: correction algebra + c-constant stress test")
    print("=" * 88)
    print(
        f"graphs={s['num_graphs']}  eps={s['eps_list']}  steps={s['num_steps']}  seed={s['seed']}"
    )
    print(
        "algebra failures:",
        s["algebra_failures"],
    )
    print(
        f"max ||M||/eps={g['max_ratio_m_eps']:.6f}  q99={g['q99_ratio_m_eps']:.6f}  "
        f"min gap frac={(100*g['min_gap_frac']):.2f}%"
    )
    print(
        f"max dbar={g['max_dbar']:.6f}  max uniform dbar={g['max_uniform_dbar']:.6f}"
    )
    wc = g["worst_ratio_case"]
    print(
        f"worst ratio case: {wc['graph']} eps={wc['eps']} t={wc['t']} "
        f"ratio={wc['ratio_m_eps']:.6f} dbar={wc['dbar']:.6f} progress={wc['progress']:.3f}"
    )
    print("")
    print("c-pass table (all steps):")
    all_steps = s["c_pass"]["all_steps"]
    for k, v in all_steps.items():
        if k in {"max_ratio", "q99_ratio", "count"}:
            continue
        print(f"  c={k}: {'PASS' if v else 'FAIL'}")
    print(
        f"  max_ratio={all_steps['max_ratio']:.6f}  q99={all_steps['q99_ratio']:.6f}  n={all_steps['count']}"
    )
    print("")
    print("c-pass table (pre80 progress):")
    pre = s["c_pass"]["pre80"]
    for k, v in pre.items():
        if k in {"max_ratio", "q99_ratio", "count"}:
            continue
        print(f"  c={k}: {'PASS' if v else 'FAIL'}")
    print(
        f"  max_ratio={pre['max_ratio']:.6f}  q99={pre['q99_ratio']:.6f}  n={pre['count']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="RNG seed for ER graph generation",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("data/first-proof/problem6-codex-cycle1-results.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    eps_list = [0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    results = run_cycle1(args.seed, eps_list)
    print_human_summary(results)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(results["summary"], f, indent=2)
    print(f"\nWrote summary JSON: {args.out_json}")


if __name__ == "__main__":
    main()
