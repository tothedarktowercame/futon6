#!/usr/bin/env python3
"""Codex Cycle 2 verifier for Problem 6.

Tasks covered:
1) Verify the K_n eigenstructure derivation (algebraic + numerical checks).
2) Stress-test K_n extremality at larger n using a scalable low-rank greedy
   evaluator in the early window (where the known ER overshoot appears).
3) Investigate the ER_60_p0.5 outlier via I0 leverage non-uniformity.
4) Probe a Schur/majorization path; provide a partial result for M_t = 0.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ----------------------------
# Graph helpers
# ----------------------------

def edge_key(u: int, v: int) -> Tuple[int, int]:
    if u < v:
        return (u, v)
    return (v, u)


def complete_graph(n: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def barbell_graph(k: int) -> Tuple[int, List[Tuple[int, int]]]:
    n = 2 * k
    edges: List[Tuple[int, int]] = []
    for i in range(k):
        for j in range(i + 1, k):
            edges.append((i, j))
    for i in range(k, n):
        for j in range(i + 1, n):
            edges.append((i, j))
    edges.append((k - 1, k))
    return n, edges


def erdos_renyi(n: int, p: float, rng: np.random.Generator) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((i, j))
    return edges


def graph_laplacian(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    L = np.zeros((n, n), dtype=float)
    for u, v in edges:
        L[u, u] += 1.0
        L[v, v] += 1.0
        L[u, v] -= 1.0
        L[v, u] -= 1.0
    return L


def laplacian_pseudoinverse(L: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(L)
    mask = eigvals > tol
    if not np.any(mask):
        return np.zeros_like(L)
    U = eigvecs[:, mask]
    inv = 1.0 / eigvals[mask]
    return (U * inv) @ U.T


# ----------------------------
# K_n derivation checks (Task 1)
# ----------------------------

def laplacian_complete_subset(n: int, t: int) -> np.ndarray:
    """Ambient n x n Laplacian of K_t on vertices {0,...,t-1}."""
    L = np.zeros((n, n), dtype=float)
    if t <= 1:
        return L
    block = -np.ones((t, t), dtype=float)
    np.fill_diagonal(block, float(t - 1))
    L[:t, :t] = block
    return L


def laplacian_complete_bipartite(n: int, t: int) -> np.ndarray:
    """Ambient n x n Laplacian of K_{t,n-t} on partition S={0..t-1}, R={t..n-1}."""
    L = np.zeros((n, n), dtype=float)
    r = n - t
    if t <= 0 or r <= 0:
        return L
    L[:t, :t] += r * np.eye(t)
    L[t:, t:] += t * np.eye(r)
    L[:t, t:] -= 1.0
    L[t:, :t] -= 1.0
    return L


def verify_kn_derivation() -> Dict:
    numeric_cases = []

    n_vals = [20, 40, 80, 120]
    eps_vals = [0.2, 0.3, 0.5]

    max_err_eig_nonzero = 0.0
    max_err_eig_zero = 0.0
    max_err_proj_nonzero = 0.0
    max_err_proj_rest = 0.0
    max_err_trbf = 0.0
    max_err_dbar = 0.0
    bad_mult_cases = []

    tol = 1e-9

    for n in n_vals:
        for eps in eps_vals:
            t_max = min(int(eps * n / 3), n - 2)
            for t in range(2, max(3, t_max + 1)):
                if eps * n - t <= 1e-10:
                    continue

                M = (1.0 / n) * laplacian_complete_subset(n, t)
                F = (1.0 / n) * laplacian_complete_bipartite(n, t)

                evals, evecs = np.linalg.eigh(M)
                nz_idx = np.where(evals > 1e-10)[0]
                z_idx = np.where(evals <= 1e-10)[0]

                expected_nonzero_mult = t - 1
                expected_zero_mult = n - t + 1
                if len(nz_idx) != expected_nonzero_mult or len(z_idx) != expected_zero_mult:
                    bad_mult_cases.append(
                        {
                            "n": n,
                            "eps": eps,
                            "t": t,
                            "nz_mult": int(len(nz_idx)),
                            "z_mult": int(len(z_idx)),
                            "expected_nz": int(expected_nonzero_mult),
                            "expected_z": int(expected_zero_mult),
                        }
                    )

                if len(nz_idx) > 0:
                    err_nz = float(np.max(np.abs(evals[nz_idx] - (t / n))))
                    max_err_eig_nonzero = max(max_err_eig_nonzero, err_nz)
                if len(z_idx) > 0:
                    err_z = float(np.max(np.abs(evals[z_idx])))
                    max_err_eig_zero = max(max_err_eig_zero, err_z)

                P_nz = evecs[:, nz_idx] @ evecs[:, nz_idx].T if len(nz_idx) > 0 else np.zeros((n, n))
                P_rest = np.eye(n) - P_nz

                proj_nz_num = float(np.trace(P_nz @ F))
                proj_rest_num = float(np.trace(P_rest @ F))

                proj_nz_formula = (t - 1) * (n - t) / n
                proj_rest_formula = (t + 1) * (n - t) / n

                max_err_proj_nonzero = max(max_err_proj_nonzero, abs(proj_nz_num - proj_nz_formula))
                max_err_proj_rest = max(max_err_proj_rest, abs(proj_rest_num - proj_rest_formula))

                H = eps * np.eye(n) - M
                B = np.linalg.inv(H)
                trbf_num = float(np.trace(B @ F))
                trbf_formula = (
                    (t - 1) * (n - t) / (n * (eps - t / n))
                    + (t + 1) * (n - t) / (n * eps)
                )
                max_err_trbf = max(max_err_trbf, abs(trbf_num - trbf_formula))

                dbar_num = trbf_num / (n - t)
                dbar_formula = (t - 1) / (n * eps - t) + (t + 1) / (n * eps)
                max_err_dbar = max(max_err_dbar, abs(dbar_num - dbar_formula))

                if n == 80 and abs(eps - 0.5) < 1e-12 and t == 12:
                    numeric_cases.append(
                        {
                            "n": n,
                            "eps": eps,
                            "t": t,
                            "dbar_numeric": dbar_num,
                            "dbar_formula": dbar_formula,
                            "trbf_numeric": trbf_num,
                            "trbf_formula": trbf_formula,
                        }
                    )

    # Exact algebraic simplification checks using rationals.
    exact_algebra_ok = True
    exact_algebra_counterexample = None
    for n in [20, 40, 60]:
        for eps_frac in [Fraction(1, 5), Fraction(3, 10), Fraction(1, 2)]:
            max_t = int((eps_frac * n) / 3)
            for t in range(2, max(3, max_t + 1)):
                if n * eps_frac - t == 0:
                    continue
                lhs_trbf = (
                    Fraction((t - 1) * (n - t), n) / (eps_frac - Fraction(t, n))
                    + Fraction((t + 1) * (n - t), n) / eps_frac
                )
                lhs_dbar = lhs_trbf / Fraction(n - t, 1)
                rhs_dbar = Fraction(t - 1, n * eps_frac - t) + Fraction(t + 1, n * eps_frac)
                if lhs_dbar != rhs_dbar:
                    exact_algebra_ok = False
                    exact_algebra_counterexample = {
                        "n": n,
                        "eps": str(eps_frac),
                        "t": t,
                        "lhs": str(lhs_dbar),
                        "rhs": str(rhs_dbar),
                    }
                    break
            if not exact_algebra_ok:
                break
        if not exact_algebra_ok:
            break

    return {
        "numeric_grid": {
            "n": n_vals,
            "eps": eps_vals,
        },
        "max_errors": {
            "eig_nonzero_vs_t_over_n": max_err_eig_nonzero,
            "eig_zero": max_err_eig_zero,
            "proj_nonzero_trace": max_err_proj_nonzero,
            "proj_rest_trace": max_err_proj_rest,
            "tr_BF": max_err_trbf,
            "dbar": max_err_dbar,
        },
        "multiplicity_mismatches": bad_mult_cases,
        "exact_algebra_ok": exact_algebra_ok,
        "exact_algebra_counterexample": exact_algebra_counterexample,
        "spot_cases": numeric_cases,
    }


# ----------------------------
# Low-rank greedy evaluator
# ----------------------------

@dataclass
class GraphPrep:
    name: str
    n: int
    edges: List[Tuple[int, int]]
    adj: List[set]
    lplus: np.ndarray
    tau_list: np.ndarray
    edge_tau: Dict[Tuple[int, int], float]


def preprocess_graph(name: str, n: int, edges: List[Tuple[int, int]]) -> GraphPrep:
    L = graph_laplacian(n, edges)
    lplus = laplacian_pseudoinverse(L)

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    u_arr = np.array([u for u, _ in edges], dtype=int)
    v_arr = np.array([v for _, v in edges], dtype=int)
    tau_list = lplus[u_arr, u_arr] + lplus[v_arr, v_arr] - 2.0 * lplus[u_arr, v_arr]

    edge_tau: Dict[Tuple[int, int], float] = {}
    for idx, (u, v) in enumerate(edges):
        edge_tau[(u, v)] = float(tau_list[idx])

    return GraphPrep(
        name=name,
        n=n,
        edges=edges,
        adj=adj,
        lplus=lplus,
        tau_list=tau_list,
        edge_tau=edge_tau,
    )


def find_i0(prep: GraphPrep, eps: float) -> List[int]:
    heavy_adj = [set() for _ in range(prep.n)]
    for idx, (u, v) in enumerate(prep.edges):
        if prep.tau_list[idx] > eps:
            heavy_adj[u].add(v)
            heavy_adj[v].add(u)

    order = sorted(range(prep.n), key=lambda x: len(heavy_adj[x]))
    chosen: List[int] = []
    chosen_set = set()
    for v in order:
        if all(u not in chosen_set for u in heavy_adj[v]):
            chosen.append(v)
            chosen_set.add(v)
    return sorted(chosen)


def edge_dot(prep: GraphPrep, e1: Tuple[int, int], e2: Tuple[int, int]) -> float:
    a, b = e1
    c, d = e2
    Lp = prep.lplus
    return float(Lp[a, c] - Lp[a, d] - Lp[b, c] + Lp[b, d])


def i0_leverage_stats(prep: GraphPrep, i0: List[int]) -> Dict:
    i0_set = set(i0)
    ell = []
    for v in i0:
        s = 0.0
        for u in prep.adj[v]:
            if u in i0_set:
                s += prep.edge_tau[edge_key(u, v)]
        ell.append(s)

    arr = np.array(ell, dtype=float)
    mean = float(np.mean(arr)) if len(arr) else 0.0
    var = float(np.var(arr)) if len(arr) else 0.0
    std = float(np.std(arr)) if len(arr) else 0.0
    cv = std / mean if mean > 1e-14 else 0.0
    return {
        "m0": int(len(i0)),
        "ell_mean": mean,
        "ell_var": var,
        "ell_std": std,
        "ell_cv": float(cv),
        "ell_min": float(np.min(arr)) if len(arr) else 0.0,
        "ell_max": float(np.max(arr)) if len(arr) else 0.0,
    }


def run_lowrank_greedy(
    prep: GraphPrep,
    eps: float,
    t_cap: int | None,
) -> Dict:
    i0 = find_i0(prep, eps)
    m0 = len(i0)
    if m0 < 2:
        return {
            "graph": prep.name,
            "n": prep.n,
            "eps": eps,
            "m0": m0,
            "steps": [],
            "max_ratio": float("nan"),
            "i0_stats": i0_leverage_stats(prep, i0),
        }

    i0_set = set(i0)
    i0_adj: Dict[int, set] = {}
    for v in i0:
        i0_adj[v] = {u for u in prep.adj[v] if u in i0_set}

    horizon = max(1, min(int(eps * m0 / 3), m0 - 1))
    T = min(horizon, t_cap) if t_cap is not None else horizon

    S: List[int] = []
    S_set = set()
    internal_edges: List[Tuple[int, int]] = []

    rows = []

    for t in range(T):
        R = [v for v in i0 if v not in S_set]
        if not R:
            break

        k = len(internal_edges)
        if k == 0:
            G = np.zeros((0, 0), dtype=float)
            Kinv = np.zeros((0, 0), dtype=float)
            norm_m = 0.0
            barrier_ok = True
        else:
            G = np.zeros((k, k), dtype=float)
            for i in range(k):
                G[i, i] = edge_dot(prep, internal_edges[i], internal_edges[i])
                for j in range(i + 1, k):
                    v_ij = edge_dot(prep, internal_edges[i], internal_edges[j])
                    G[i, j] = v_ij
                    G[j, i] = v_ij

            evals_g = np.linalg.eigvalsh(G)
            norm_m = float(evals_g[-1]) if len(evals_g) else 0.0

            K = np.eye(k) - (1.0 / eps) * G
            min_k = float(np.min(np.linalg.eigvalsh(K)))
            barrier_ok = min_k > 1e-10
            if not barrier_ok:
                break
            Kinv = np.linalg.inv(K)

        scores: Dict[int, float] = {}
        traces: Dict[int, float] = {}

        for v in R:
            ev = [edge_key(u, v) for u in S if u in i0_adj[v]]
            s = len(ev)
            if s == 0:
                scores[v] = 0.0
                traces[v] = 0.0
                continue

            if k > 0:
                Q = np.zeros((s, k), dtype=float)
                for i in range(s):
                    for j in range(k):
                        Q[i, j] = edge_dot(prep, internal_edges[j], ev[i])
            else:
                Q = np.zeros((s, 0), dtype=float)

            H = np.zeros((s, s), dtype=float)
            for i in range(s):
                for j in range(i, s):
                    d0 = edge_dot(prep, ev[i], ev[j]) / eps
                    corr = 0.0
                    if k > 0:
                        corr = float(Q[i] @ Kinv @ Q[j]) / (eps * eps)
                    val = d0 + corr
                    H[i, j] = val
                    H[j, i] = val

            evals_h = np.linalg.eigvalsh(H)
            score = max(0.0, float(evals_h[-1]))
            trv = max(0.0, float(np.sum(evals_h)))
            scores[v] = score
            traces[v] = trv

        dbar = float(np.mean([traces[v] for v in R])) if R else 0.0

        if t > 0 and (m0 * eps - t) > 1e-12:
            dbar_kn = (t - 1) / (m0 * eps - t) + (t + 1) / (m0 * eps)
            ratio = dbar / dbar_kn
        else:
            dbar_kn = 0.0
            ratio = float("nan")

        rows.append(
            {
                "t": int(t),
                "r_t": int(len(R)),
                "norm_M": norm_m,
                "gap_frac": (eps - norm_m) / eps if eps > 0 else float("nan"),
                "dbar": dbar,
                "dbar_kn": dbar_kn,
                "ratio": ratio,
            }
        )

        best_v = min(R, key=lambda x: (scores[x], x))
        S.append(best_v)
        S_set.add(best_v)

        for u in S[:-1]:
            if u in i0_adj[best_v]:
                internal_edges.append(edge_key(u, best_v))

    ratios = [r["ratio"] for r in rows if np.isfinite(r["ratio"])]
    max_ratio = float(max(ratios)) if ratios else float("nan")

    return {
        "graph": prep.name,
        "n": prep.n,
        "eps": eps,
        "m0": m0,
        "horizon": horizon,
        "t_cap": t_cap,
        "steps": rows,
        "max_ratio": max_ratio,
        "i0_stats": i0_leverage_stats(prep, i0),
    }


# ----------------------------
# Task 2: large-n stress
# ----------------------------

def run_large_n_stress(seed: int, t_cap: int) -> Dict:
    rng = np.random.default_rng(seed)

    per_run = []
    per_n_summary = {}

    for n in [200, 500, 1000]:
        _, barbell_edges = barbell_graph(n // 2)
        er_edges_r0 = erdos_renyi(n, 0.5, rng)
        er_edges_r1 = erdos_renyi(n, 0.5, rng)

        graph_defs = [
            (f"ER_{n}_p0.5_r0", er_edges_r0),
            (f"ER_{n}_p0.5_r1", er_edges_r1),
            (f"Barbell_{n//2}", barbell_edges),
        ]

        preps = []
        for gname, gedges in graph_defs:
            preps.append(preprocess_graph(gname, n, gedges))

        max_ratio_this_n = 1.0
        max_ratio_this_n_ge2 = 1.0
        worst_case_this_n = {
            "graph": f"K_{n}",
            "eps": None,
            "t": None,
            "ratio": 1.0,
            "note": "K_n is exactly ratio 1 by formula",
        }
        worst_case_this_n_ge2 = {
            "graph": f"K_{n}",
            "eps": None,
            "t": None,
            "ratio": 1.0,
            "note": "K_n is exactly ratio 1 by formula",
        }

        # K_n analytic baseline (exact ratio = 1).
        for eps in [0.2, 0.3, 0.5]:
            per_run.append(
                {
                    "graph": f"K_{n}",
                    "n": n,
                    "eps": eps,
                    "mode": "analytic",
                    "max_ratio": 1.0,
                    "t_cap": t_cap,
                    "steps_analyzed": "all (formula)",
                }
            )

        # Non-K_n families: scalable low-rank greedy in early window.
        for prep in preps:
            for eps in [0.2, 0.3, 0.5]:
                run = run_lowrank_greedy(prep, eps, t_cap=t_cap)
                finite_rows = [r for r in run["steps"] if np.isfinite(r["ratio"])]
                finite_rows_ge2 = [r for r in finite_rows if r["t"] >= 2]
                if finite_rows:
                    worst_row = max(finite_rows, key=lambda x: x["ratio"])
                    t_worst = int(worst_row["t"])
                    r_worst = float(worst_row["ratio"])
                else:
                    t_worst = None
                    r_worst = float("nan")
                if finite_rows_ge2:
                    worst_row_ge2 = max(finite_rows_ge2, key=lambda x: x["ratio"])
                    t_worst_ge2 = int(worst_row_ge2["t"])
                    r_worst_ge2 = float(worst_row_ge2["ratio"])
                else:
                    t_worst_ge2 = None
                    r_worst_ge2 = float("nan")

                per_run.append(
                    {
                        "graph": run["graph"],
                        "n": run["n"],
                        "eps": run["eps"],
                        "mode": "lowrank_greedy",
                        "m0": run["m0"],
                        "horizon": run.get("horizon"),
                        "t_cap": run.get("t_cap"),
                        "max_ratio": run["max_ratio"],
                        "worst_t": t_worst,
                        "max_ratio_t_ge_2": r_worst_ge2,
                        "worst_t_ge_2": t_worst_ge2,
                        "i0_ell_cv": run["i0_stats"]["ell_cv"],
                    }
                )

                if np.isfinite(run["max_ratio"]) and run["max_ratio"] > max_ratio_this_n:
                    max_ratio_this_n = run["max_ratio"]
                    worst_case_this_n = {
                        "graph": run["graph"],
                        "eps": eps,
                        "t": t_worst,
                        "ratio": float(run["max_ratio"]),
                    }
                if np.isfinite(r_worst_ge2) and r_worst_ge2 > max_ratio_this_n_ge2:
                    max_ratio_this_n_ge2 = r_worst_ge2
                    worst_case_this_n_ge2 = {
                        "graph": run["graph"],
                        "eps": eps,
                        "t": t_worst_ge2,
                        "ratio": float(r_worst_ge2),
                    }

        per_n_summary[str(n)] = {
            "max_ratio": float(max_ratio_this_n),
            "overshoot_pct": float(100.0 * (max_ratio_this_n - 1.0)),
            "worst_case": worst_case_this_n,
            "max_ratio_t_ge_2": float(max_ratio_this_n_ge2),
            "overshoot_t_ge_2_pct": float(100.0 * (max_ratio_this_n_ge2 - 1.0)),
            "worst_case_t_ge_2": worst_case_this_n_ge2,
        }

    overs = [per_n_summary[str(n)]["overshoot_pct"] for n in [200, 500, 1000]]
    shrink = bool(overs[0] >= overs[1] >= overs[2])

    return {
        "t_cap": t_cap,
        "runs": per_run,
        "per_n_summary": per_n_summary,
        "overshoot_nonincreasing_with_n": shrink,
    }


# ----------------------------
# Low-rank vs dense cross-check
# ----------------------------

def load_dense_cycle2_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "scripts" / "verify-p6-cycle2-logdet-dbar.py"
    spec = importlib.util.spec_from_file_location("p6_cycle2_dense", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def crosscheck_lowrank_vs_dense(t_cap: int) -> Dict:
    mod = load_dense_cycle2_module()

    checks = []

    # Case 1: K_40
    edges_k40 = complete_graph(40)
    prep_k40 = preprocess_graph("K_40", 40, edges_k40)
    low_k40 = run_lowrank_greedy(prep_k40, 0.3, t_cap=t_cap)
    dense_k40 = mod.run_cycle2(40, edges_k40, 0.3, "K_40")
    checks.append(("K_40", 0.3, low_k40, dense_k40))

    # Case 2: ER_60_p0.5 outlier instance
    n60, er60 = build_cycle2_er_outlier_graph()
    prep_er60 = preprocess_graph("ER_60_p0.5_seed42", n60, er60)
    low_er60 = run_lowrank_greedy(prep_er60, 0.5, t_cap=t_cap)
    dense_er60 = mod.run_cycle2(n60, er60, 0.5, "ER_60_p0.5_seed42")
    checks.append(("ER_60_p0.5_seed42", 0.5, low_er60, dense_er60))

    out = []
    max_abs_dbar_err = 0.0
    max_abs_ratio_err = 0.0

    for name, eps, low, dense in checks:
        dense_map = {int(r["t"]): r for r in dense if int(r["t"]) > 0}
        low_map = {int(r["t"]): r for r in low["steps"] if int(r["t"]) > 0}
        common_t = sorted(set(dense_map.keys()) & set(low_map.keys()))
        if t_cap is not None:
            common_t = [t for t in common_t if t <= t_cap]

        dbar_errs = []
        ratio_errs = []
        for t in common_t:
            d_dense = float(dense_map[t]["dbar"])
            d_low = float(low_map[t]["dbar"])
            dbar_errs.append(abs(d_dense - d_low))

            r_dense = float(dense_map[t]["dbar_over_kn_exact"])
            r_low = float(low_map[t]["ratio"])
            ratio_errs.append(abs(r_dense - r_low))

        max_d = float(max(dbar_errs)) if dbar_errs else 0.0
        max_r = float(max(ratio_errs)) if ratio_errs else 0.0
        max_abs_dbar_err = max(max_abs_dbar_err, max_d)
        max_abs_ratio_err = max(max_abs_ratio_err, max_r)

        out.append(
            {
                "graph": name,
                "eps": eps,
                "compared_steps": len(common_t),
                "max_abs_dbar_error": max_d,
                "max_abs_ratio_error": max_r,
            }
        )

    return {
        "t_cap": t_cap,
        "cases": out,
        "max_abs_dbar_error": max_abs_dbar_err,
        "max_abs_ratio_error": max_abs_ratio_err,
    }


# ----------------------------
# Task 3: ER outlier analysis
# ----------------------------

def build_cycle2_er_outlier_graph() -> Tuple[int, List[Tuple[int, int]]]:
    """Reproduce ER_60_p0.5 from verify-p6-cycle2-logdet-dbar.py RNG order."""
    rng = np.random.default_rng(42)

    # First ER draw in that script: ER_40_p0.3
    _ = erdos_renyi(40, 0.3, rng)
    # Second draw: ER_60_p0.5 (target outlier instance)
    er60 = erdos_renyi(60, 0.5, rng)
    return 60, er60


def run_outlier_analysis() -> Dict:
    n, er60_edges = build_cycle2_er_outlier_graph()
    prep_er = preprocess_graph("ER_60_p0.5_seed42", n, er60_edges)
    prep_k = preprocess_graph("K_60", 60, complete_graph(60))

    eps = 0.5
    er_run = run_lowrank_greedy(prep_er, eps, t_cap=20)
    k_run = run_lowrank_greedy(prep_k, eps, t_cap=20)

    er_t6 = next((r for r in er_run["steps"] if r["t"] == 6), None)
    k_t6 = next((r for r in k_run["steps"] if r["t"] == 6), None)

    # Correlation study across the original Cycle-2 family set.
    rng = np.random.default_rng(42)
    suite = []
    for k in [20, 40, 60, 80, 100]:
        suite.append((f"K_{k}", k, complete_graph(k)))
    for k in [20, 30, 40]:
        n_b, e_b = barbell_graph(k)
        suite.append((f"Barbell_{k}", n_b, e_b))
    for nn, p in [(40, 0.3), (60, 0.5), (80, 0.3), (80, 0.5)]:
        suite.append((f"ER_{nn}_p{p}", nn, erdos_renyi(nn, p, rng)))

    epsilons = [0.15, 0.2, 0.3, 0.5]
    pairs = []

    for gname, gn, gedges in suite:
        prep = preprocess_graph(gname, gn, gedges)
        for e in epsilons:
            run = run_lowrank_greedy(prep, e, t_cap=20)
            mr = run["max_ratio"]
            if np.isfinite(mr):
                pairs.append(
                    {
                        "graph": gname,
                        "n": gn,
                        "eps": e,
                        "max_ratio": float(mr),
                        "ell_cv": float(run["i0_stats"]["ell_cv"]),
                        "ell_var": float(run["i0_stats"]["ell_var"]),
                    }
                )

    if pairs:
        max_ratio = np.array([p["max_ratio"] for p in pairs], dtype=float)
        excess = np.maximum(max_ratio - 1.0, 0.0)
        ell_cv = np.array([p["ell_cv"] for p in pairs], dtype=float)
        ell_var = np.array([p["ell_var"] for p in pairs], dtype=float)

        corr_cv = float(np.corrcoef(excess, ell_cv)[0, 1]) if np.std(ell_cv) > 1e-14 else 0.0
        corr_var = float(np.corrcoef(excess, ell_var)[0, 1]) if np.std(ell_var) > 1e-14 else 0.0
    else:
        corr_cv = 0.0
        corr_var = 0.0

    return {
        "er_outlier_step_t6": er_t6,
        "k60_step_t6": k_t6,
        "er_i0_stats": er_run["i0_stats"],
        "k60_i0_stats": k_run["i0_stats"],
        "er_max_ratio": er_run["max_ratio"],
        "k60_max_ratio": k_run["max_ratio"],
        "suite_pairs": pairs,
        "corr_excess_with_ell_cv": corr_cv,
        "corr_excess_with_ell_var": corr_var,
    }


# ----------------------------
# Task 4: Schur / majorization probe
# ----------------------------

def run_schur_probe() -> Dict:
    eps = 0.5

    # Partial result: when M_t = 0, B_t = (1/eps)I, so dbar only depends on tr(F_t), r_t.
    m0_diffs = []
    rng = np.random.default_rng(123)
    for _ in range(200):
        r_t = int(rng.integers(5, 50))
        tr_f = float(rng.uniform(0.1, 20.0))
        # two arbitrary "weight profiles" with same sum
        w1 = rng.random(10)
        w1 = tr_f * w1 / np.sum(w1)
        w2 = rng.random(10)
        w2 = tr_f * w2 / np.sum(w2)

        d1 = np.sum(w1) / (eps * r_t)
        d2 = np.sum(w2) / (eps * r_t)
        m0_diffs.append(abs(d1 - d2))

    # Nonzero-lambda sanity check: with only sum(w_i) fixed, objective is linear in w,
    # so extreme concentration beats uniform if coefficients differ.
    lam = np.array([0.26, 0.18, 0.11, 0.05, 0.0])
    coeff = 1.0 / (eps - lam)
    total_mass = 1.0

    w_uniform = np.full(len(lam), total_mass / len(lam))
    w_top = np.zeros(len(lam)); w_top[0] = total_mass
    w_bottom = np.zeros(len(lam)); w_bottom[-1] = total_mass

    obj_uniform = float(np.dot(coeff, w_uniform))
    obj_top = float(np.dot(coeff, w_top))
    obj_bottom = float(np.dot(coeff, w_bottom))

    return {
        "m0_case_max_abs_diff": float(max(m0_diffs) if m0_diffs else 0.0),
        "m0_case_mean_abs_diff": float(np.mean(m0_diffs) if m0_diffs else 0.0),
        "nonzero_lambda_demo": {
            "lambda": lam.tolist(),
            "coeff": coeff.tolist(),
            "objective_uniform": obj_uniform,
            "objective_extreme_top": obj_top,
            "objective_extreme_bottom": obj_bottom,
        },
    }


# ----------------------------
# Orchestration
# ----------------------------

def build_report_json(seed: int, t_cap_large_n: int) -> Dict:
    task1 = verify_kn_derivation()
    xchk = crosscheck_lowrank_vs_dense(t_cap=t_cap_large_n)
    task2 = run_large_n_stress(seed=seed, t_cap=t_cap_large_n)
    task3 = run_outlier_analysis()
    task4 = run_schur_probe()

    return {
        "meta": {
            "date": "2026-02-13",
            "agent": "Codex",
            "seed": seed,
            "large_n_t_cap": t_cap_large_n,
        },
        "task1_kn_derivation": task1,
        "crosscheck_lowrank_vs_dense": xchk,
        "task2_large_n_extremality": task2,
        "task3_er_outlier": task3,
        "task4_schur_probe": task4,
    }


def print_summary(results: Dict):
    t1 = results["task1_kn_derivation"]
    xc = results["crosscheck_lowrank_vs_dense"]
    t2 = results["task2_large_n_extremality"]
    t3 = results["task3_er_outlier"]
    t4 = results["task4_schur_probe"]

    print("=" * 88)
    print("P6 CYCLE 2 CODEX TASKS")
    print("=" * 88)

    print("\nTask 1: K_n derivation checks")
    print("  exact algebra:", "PASS" if t1["exact_algebra_ok"] else "FAIL")
    print("  max errors:")
    for k, v in t1["max_errors"].items():
        print(f"    {k}: {v:.3e}")
    print(f"  multiplicity mismatches: {len(t1['multiplicity_mismatches'])}")
    print(
        f"  low-rank vs dense cross-check: "
        f"max |dbar error|={xc['max_abs_dbar_error']:.3e}, "
        f"max |ratio error|={xc['max_abs_ratio_error']:.3e}"
    )

    print("\nTask 2: Large-n stress (early window)")
    print(f"  t_cap = {t2['t_cap']}")
    for n in [200, 500, 1000]:
        s = t2["per_n_summary"][str(n)]
        print(
            f"  n={n}: max ratio={s['max_ratio']:.6f} "
            f"(overshoot {s['overshoot_pct']:.3f}%), worst={s['worst_case']}")
        print(
            f"        t>=2 max ratio={s['max_ratio_t_ge_2']:.6f} "
            f"(overshoot {s['overshoot_t_ge_2_pct']:.3f}%), "
            f"worst={s['worst_case_t_ge_2']}"
        )
    print("  overshoot nonincreasing with n:", t2["overshoot_nonincreasing_with_n"])

    print("\nTask 3: ER outlier + leverage variance")
    er_t6 = t3["er_outlier_step_t6"]
    if er_t6 is not None:
        print(
            f"  ER outlier (t=6): ratio={er_t6['ratio']:.6f}, dbar={er_t6['dbar']:.6f}, "
            f"dbar_kn={er_t6['dbar_kn']:.6f}")
    print(
        f"  I0 leverage CV: ER={t3['er_i0_stats']['ell_cv']:.6f}, "
        f"K_60={t3['k60_i0_stats']['ell_cv']:.6f}")
    print(
        f"  corr(excess, ell_cv)={t3['corr_excess_with_ell_cv']:.4f}, "
        f"corr(excess, ell_var)={t3['corr_excess_with_ell_var']:.4f}")

    print("\nTask 4: Schur probe")
    print(f"  M_t=0 max |difference| under same tr(F): {t4['m0_case_max_abs_diff']:.3e}")
    demo = t4["nonzero_lambda_demo"]
    print(
        "  nonzero-lambda objective: "
        f"top={demo['objective_extreme_top']:.6f}, "
        f"uniform={demo['objective_uniform']:.6f}, "
        f"bottom={demo['objective_extreme_bottom']:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t-cap-large-n", type=int, default=8)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("data/first-proof/problem6-codex-cycle2-results.json"),
    )
    args = parser.parse_args()

    results = build_report_json(seed=args.seed, t_cap_large_n=args.t_cap_large_n)
    print_summary(results)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote JSON: {args.out_json}")


if __name__ == "__main__":
    main()
