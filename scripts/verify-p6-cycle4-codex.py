#!/usr/bin/env python3
"""Problem 6 Cycle 4 Codex verifier.

Implements numerical tasks from the Cycle 4 handoff:
- Task 4: interlacing-family probe on Y_t(v) characteristic polynomials.
- Task 5: per-vertex / per-edge alignment alpha probes and correlations.

Also emits summary diagnostics useful for Tasks 2-3 writeup context.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def to_jsonable(obj):
    """Convert NumPy scalars to native Python for JSON serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def edge_key(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


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


def star_graph(n: int) -> List[Tuple[int, int]]:
    return [(0, i) for i in range(1, n)]


def grid_graph(rows: int, cols: int) -> Tuple[int, List[Tuple[int, int]]]:
    n = rows * cols
    edges: List[Tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            if c + 1 < cols:
                edges.append((u, u + 1))
            if r + 1 < rows:
                edges.append((u, u + cols))
    return n, edges


def erdos_renyi(n: int, p: float, rng: np.random.Generator) -> List[Tuple[int, int]]:
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((i, j))
    return edges


def is_connected(n: int, edges: List[Tuple[int, int]]) -> bool:
    if n == 0:
        return True
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    seen = [False] * n
    stack = [0]
    seen[0] = True
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                stack.append(v)
    return all(seen)


def connected_er(n: int, p: float, seed: int) -> Tuple[List[Tuple[int, int]], int]:
    rng = np.random.default_rng(seed)
    for rep in range(200):
        edges = erdos_renyi(n, p, rng)
        if is_connected(n, edges):
            return edges, rep
    raise RuntimeError("failed to sample connected ER graph")


def graph_laplacian(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    L = np.zeros((n, n), dtype=float)
    for u, v in edges:
        L[u, u] += 1.0
        L[v, v] += 1.0
        L[u, v] -= 1.0
        L[v, u] -= 1.0
    return L


def pseudo_inv_and_half(L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(L)
    n = L.shape[0]
    Lplus = np.zeros((n, n), dtype=float)
    Lph = np.zeros((n, n), dtype=float)
    for i, lam in enumerate(eigvals):
        if lam > 1e-10:
            ui = eigvecs[:, i]
            Lplus += (1.0 / lam) * np.outer(ui, ui)
            Lph += (1.0 / np.sqrt(lam)) * np.outer(ui, ui)
    return Lplus, Lph


def compute_edge_data(
    n: int,
    edges: List[Tuple[int, int]],
    Lph: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    z_edges: List[np.ndarray] = []
    X_edges: List[np.ndarray] = []
    taus: List[float] = []
    for u, v in edges:
        b = np.zeros(n, dtype=float)
        b[u] = 1.0
        b[v] = -1.0
        z = Lph @ b
        z_edges.append(z)
        X_edges.append(np.outer(z, z))
        taus.append(float(np.dot(z, z)))
    return z_edges, X_edges, np.array(taus, dtype=float)


def find_i0(n: int, edges: List[Tuple[int, int]], taus: np.ndarray, eps: float) -> List[int]:
    heavy_adj = [set() for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        if taus[idx] > eps:
            heavy_adj[u].add(v)
            heavy_adj[v].add(u)

    i_set = set()
    for v in sorted(range(n), key=lambda vv: len(heavy_adj[vv])):
        if all(u not in i_set for u in heavy_adj[v]):
            i_set.add(v)
    return sorted(i_set)


def is_real_rooted(coeff: np.ndarray, imag_tol: float = 1e-7) -> Tuple[bool, np.ndarray, float]:
    roots = np.roots(coeff)
    max_imag = float(np.max(np.abs(np.imag(roots)))) if len(roots) else 0.0
    if max_imag <= imag_tol:
        r = np.sort(np.real(roots))
        return True, r, max_imag
    return False, np.array([]), max_imag


def interlace(a: np.ndarray, b: np.ndarray, tol: float = 1e-8) -> bool:
    if len(a) != len(b) or len(a) == 0:
        return False
    a = np.sort(a)
    b = np.sort(b)

    def pattern(x: np.ndarray, y: np.ndarray) -> bool:
        for i in range(len(x) - 1):
            if x[i] - tol > y[i]:
                return False
            if y[i] - tol > x[i + 1]:
                return False
        return x[-1] - tol <= y[-1]

    return pattern(a, b) or pattern(b, a)


def corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-14 or sy < 1e-14:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


@dataclass
class GraphPrep:
    name: str
    n: int
    edges: List[Tuple[int, int]]
    Lplus: np.ndarray
    Lph: np.ndarray
    z_edges: List[np.ndarray]
    X_edges: List[np.ndarray]
    taus: np.ndarray


def prepare_graph(name: str, n: int, edges: List[Tuple[int, int]]) -> GraphPrep:
    L = graph_laplacian(n, edges)
    Lplus, Lph = pseudo_inv_and_half(L)
    z_edges, X_edges, taus = compute_edge_data(n, edges, Lph)
    return GraphPrep(name, n, edges, Lplus, Lph, z_edges, X_edges, taus)


def run_single(
    gp: GraphPrep,
    eps: float,
    rng: np.random.Generator,
    num_partition_trials: int,
) -> Dict:
    n = gp.n
    i0 = find_i0(n, gp.edges, gp.taus, eps)
    i0_set = set(i0)
    m0 = len(i0)
    horizon = max(1, min(int(eps * m0 / 3), m0 - 1)) if m0 >= 2 else 0

    edge_idx: Dict[Tuple[int, int], int] = {}
    for idx, (u, v) in enumerate(gp.edges):
        if u in i0_set and v in i0_set:
            edge_idx[edge_key(u, v)] = idx

    z_cols = gp.Lph

    S: List[int] = []
    S_set = set()
    M = np.zeros((n, n), dtype=float)

    steps = []
    interlace_partition_trials = []
    alpha_vertices = []
    alpha_edges = []

    for _ in range(horizon + 1):
        R = [v for v in i0 if v not in S_set]
        if not R:
            break
        t = len(S)

        H = eps * np.eye(n) - M
        min_headroom = float(np.min(np.linalg.eigvalsh(H)))
        if min_headroom < 1e-10:
            break

        B = np.linalg.inv(H)
        Bsqrt = np.linalg.cholesky(B + 1e-14 * np.eye(n))

        eigvals_M, eigvecs_M = np.linalg.eigh(M)
        pos = eigvals_M > 1e-10
        if np.any(pos):
            U = eigvecs_M[:, pos]
            P_M = U @ U.T
        else:
            P_M = np.zeros((n, n), dtype=float)

        F = np.zeros((n, n), dtype=float)
        Ys = {}
        traces = {}
        scores = {}
        p_coeffs = {}

        for v in R:
            C_v = np.zeros((n, n), dtype=float)
            for u in S:
                idx = edge_idx.get(edge_key(u, v))
                if idx is not None:
                    C_v += gp.X_edges[idx]

            Y = Bsqrt @ C_v @ Bsqrt.T
            Y = 0.5 * (Y + Y.T)
            Ys[v] = Y
            ev = np.linalg.eigvalsh(Y)
            traces[v] = float(np.sum(ev[ev > 1e-12]))
            scores[v] = float(np.max(ev)) if len(ev) else 0.0
            p_coeffs[v] = np.real_if_close(np.poly(ev), tol=1e5).astype(float)
            F += C_v

        dbar = float(np.mean([traces[v] for v in R])) if R else 0.0

        # Task 4: average characteristic polynomial and interlacing probes.
        coeff_mat = np.stack([p_coeffs[v] for v in R], axis=0)
        q_coeff = np.mean(coeff_mat, axis=0)
        q_real, q_roots, q_im = is_real_rooted(q_coeff)
        q_max_root = float(np.max(q_roots)) if q_real and len(q_roots) else float("nan")
        q_over_dbar = q_max_root / dbar if dbar > 1e-14 and np.isfinite(q_max_root) else float("nan")

        partition_stats = {
            "real_rooted_all": 0,
            "interlace_ab": 0,
            "interlace_aq": 0,
            "interlace_bq": 0,
            "trials": 0,
            "first_fail": None,
        }

        if len(R) >= 2:
            R_arr = np.array(R, dtype=int)
            for trial in range(num_partition_trials):
                mask = rng.random(len(R_arr)) < 0.5
                if mask.all() or (~mask).all():
                    mask[0] = True
                    mask[-1] = False

                A = R_arr[mask]
                C = R_arr[~mask]
                A_coeff = np.mean(np.stack([p_coeffs[int(v)] for v in A], axis=0), axis=0)
                C_coeff = np.mean(np.stack([p_coeffs[int(v)] for v in C], axis=0), axis=0)

                a_real, a_roots, a_im = is_real_rooted(A_coeff)
                c_real, c_roots, c_im = is_real_rooted(C_coeff)

                real_all = a_real and c_real and q_real
                inter_ab = inter_aq = inter_cq = False
                if real_all:
                    inter_ab = interlace(a_roots, c_roots)
                    inter_aq = interlace(a_roots, q_roots)
                    inter_cq = interlace(c_roots, q_roots)

                partition_stats["trials"] += 1
                partition_stats["real_rooted_all"] += int(real_all)
                partition_stats["interlace_ab"] += int(inter_ab)
                partition_stats["interlace_aq"] += int(inter_aq)
                partition_stats["interlace_bq"] += int(inter_cq)

                if partition_stats["first_fail"] is None and not (real_all and inter_ab):
                    partition_stats["first_fail"] = {
                        "trial": trial,
                        "a_real": a_real,
                        "b_real": c_real,
                        "q_real": q_real,
                        "a_max_imag": a_im,
                        "b_max_imag": c_im,
                        "q_max_imag": q_im,
                        "interlace_ab": inter_ab,
                    }

        interlace_partition_trials.append(
            {
                "graph": gp.name,
                "eps": eps,
                "t": t,
                "stats": partition_stats,
            }
        )

        # Task 5: per-vertex alpha_v and resistance proxies.
        if len(S) > 0:
            S_arr = np.array(S, dtype=int)
            L_diag_S = np.diag(gp.Lplus)[S_arr]
        else:
            S_arr = np.array([], dtype=int)
            L_diag_S = np.array([], dtype=float)

        for v in R:
            z_v = z_cols[:, v]
            denom = float(np.dot(z_v, z_v))
            if denom > 1e-14:
                pz = P_M @ z_v
                alpha_v = float(np.dot(pz, pz) / denom)
            else:
                alpha_v = 0.0

            if len(S) > 0:
                rv = gp.Lplus[v, v] + L_diag_S - 2.0 * gp.Lplus[v, S_arr]
                eff_min = float(np.min(rv))
                eff_mean = float(np.mean(rv))
            else:
                eff_min = float("nan")
                eff_mean = float("nan")

            alpha_vertices.append(
                {
                    "graph": gp.name,
                    "eps": eps,
                    "t": t,
                    "v": int(v),
                    "alpha_v": alpha_v,
                    "z_norm_sq": denom,
                    "eff_res_min_to_S": eff_min,
                    "eff_res_mean_to_S": eff_mean,
                }
            )

        # Per-edge alpha on cross-edges.
        for u in S:
            for v in R:
                idx = edge_idx.get(edge_key(u, v))
                if idx is None:
                    continue
                z_e = gp.z_edges[idx]
                tau_e = float(gp.taus[idx])
                if tau_e <= 1e-14:
                    alpha_uv = 0.0
                else:
                    pz = P_M @ z_e
                    alpha_uv = float(np.dot(pz, pz) / tau_e)

                # lightweight effective-resistance proxy endpoints
                rv = float(gp.Lplus[u, u] + gp.Lplus[v, v] - 2.0 * gp.Lplus[u, v])
                alpha_edges.append(
                    {
                        "graph": gp.name,
                        "eps": eps,
                        "t": t,
                        "u": int(u),
                        "v": int(v),
                        "alpha_uv": alpha_uv,
                        "tau_uv": tau_e,
                        "R_eff_uv": rv,
                    }
                )

        steps.append(
            {
                "t": t,
                "r_t": len(R),
                "dbar": dbar,
                "Q_real_rooted": q_real,
                "Q_max_imag_root": q_im,
                "Q_max_root": q_max_root,
                "Q_max_root_over_dbar": q_over_dbar,
                "x": float(np.linalg.norm(M, ord=2) / eps) if eps > 0 else float("nan"),
                "rank_M": int(np.sum(pos)),
            }
        )

        if len(S) >= horizon:
            break

        best_v = min(R, key=lambda vv: (scores[vv], vv))
        S.append(best_v)
        S_set.add(best_v)

        for u in S[:-1]:
            idx = edge_idx.get(edge_key(u, best_v))
            if idx is not None:
                M += gp.X_edges[idx]

    return {
        "graph": gp.name,
        "n": gp.n,
        "eps": eps,
        "m0": m0,
        "horizon": horizon,
        "steps": steps,
        "interlace_partition_trials": interlace_partition_trials,
        "alpha_vertices": alpha_vertices,
        "alpha_edges": alpha_edges,
    }


def build_results(seed: int, num_partition_trials: int) -> Dict:
    er_edges, er_rep = connected_er(60, 0.5, seed=42)
    n_bar, e_bar = barbell_graph(40)
    n_grid, e_grid = grid_graph(8, 5)

    suite = [
        ("K_40", 40, complete_graph(40)),
        ("K_80", 80, complete_graph(80)),
        (f"ER_60_p0.5_seed42_rep{er_rep}", 60, er_edges),
        ("Barbell_40", n_bar, e_bar),
        ("Star_40", 40, star_graph(40)),
        ("Grid_8x5", n_grid, e_grid),
    ]
    eps_list = [0.2, 0.3, 0.5]

    rng = np.random.default_rng(seed)
    preps = [prepare_graph(name, n, edges) for (name, n, edges) in suite]

    runs = []
    for gp in preps:
        for eps in eps_list:
            runs.append(run_single(gp, eps, rng, num_partition_trials))

    # Aggregate summaries.
    q_ratios = []
    q_ratio_worst = None
    q_non_real = []
    interlace_failures = []

    all_alpha_v = []
    all_alpha_v_eff_min = []
    all_alpha_v_eff_mean = []
    all_alpha_edge = []
    all_alpha_edge_tau = []
    alpha_v_max = (-np.inf, None)
    alpha_e_max = (-np.inf, None)
    alpha_v_viols = []

    x_horizon = []

    for rr in runs:
        if rr["steps"]:
            last = rr["steps"][-1]
            x_horizon.append(
                {
                    "graph": rr["graph"],
                    "eps": rr["eps"],
                    "t": last["t"],
                    "x": last["x"],
                }
            )

        for s in rr["steps"]:
            if np.isfinite(s["Q_max_root_over_dbar"]):
                q_ratios.append(s["Q_max_root_over_dbar"])
                if q_ratio_worst is None or s["Q_max_root_over_dbar"] > q_ratio_worst["ratio"]:
                    q_ratio_worst = {
                        "graph": rr["graph"],
                        "eps": rr["eps"],
                        "t": s["t"],
                        "ratio": s["Q_max_root_over_dbar"],
                        "Q_max_root": s["Q_max_root"],
                        "dbar": s["dbar"],
                    }
            if not s["Q_real_rooted"]:
                q_non_real.append(
                    {
                        "graph": rr["graph"],
                        "eps": rr["eps"],
                        "t": s["t"],
                        "max_imag": s["Q_max_imag_root"],
                    }
                )

        for trial_row in rr["interlace_partition_trials"]:
            st = trial_row["stats"]
            if st["trials"] > 0 and st["interlace_ab"] < st["trials"]:
                interlace_failures.append(
                    {
                        "graph": rr["graph"],
                        "eps": rr["eps"],
                        "t": trial_row["t"],
                        "interlace_pass": st["interlace_ab"],
                        "trials": st["trials"],
                        "first_fail": st["first_fail"],
                    }
                )

        for av in rr["alpha_vertices"]:
            all_alpha_v.append(av["alpha_v"])
            if np.isfinite(av["eff_res_min_to_S"]):
                all_alpha_v_eff_min.append((av["alpha_v"], av["eff_res_min_to_S"]))
            if np.isfinite(av["eff_res_mean_to_S"]):
                all_alpha_v_eff_mean.append((av["alpha_v"], av["eff_res_mean_to_S"]))

            if av["alpha_v"] > alpha_v_max[0]:
                alpha_v_max = (
                    av["alpha_v"],
                    {
                        "graph": av["graph"],
                        "eps": av["eps"],
                        "t": av["t"],
                        "v": av["v"],
                        "alpha_v": av["alpha_v"],
                    },
                )
            if av["alpha_v"] > 0.5 + 1e-8:
                alpha_v_viols.append(
                    {
                        "graph": av["graph"],
                        "eps": av["eps"],
                        "t": av["t"],
                        "v": av["v"],
                        "alpha_v": av["alpha_v"],
                    }
                )

        for ae in rr["alpha_edges"]:
            all_alpha_edge.append(ae["alpha_uv"])
            all_alpha_edge_tau.append((ae["alpha_uv"], ae["tau_uv"]))
            if ae["alpha_uv"] > alpha_e_max[0]:
                alpha_e_max = (
                    ae["alpha_uv"],
                    {
                        "graph": ae["graph"],
                        "eps": ae["eps"],
                        "t": ae["t"],
                        "u": ae["u"],
                        "v": ae["v"],
                        "alpha_uv": ae["alpha_uv"],
                    },
                )

    if all_alpha_v_eff_min:
        x1 = np.array([a for a, _ in all_alpha_v_eff_min], dtype=float)
        y1 = np.array([b for _, b in all_alpha_v_eff_min], dtype=float)
        c1 = corr(x1, y1)
    else:
        c1 = float("nan")

    if all_alpha_v_eff_mean:
        x2 = np.array([a for a, _ in all_alpha_v_eff_mean], dtype=float)
        y2 = np.array([b for _, b in all_alpha_v_eff_mean], dtype=float)
        c2 = corr(x2, y2)
    else:
        c2 = float("nan")

    if all_alpha_edge_tau:
        x3 = np.array([a for a, _ in all_alpha_edge_tau], dtype=float)
        y3 = np.array([b for _, b in all_alpha_edge_tau], dtype=float)
        c3 = corr(x3, y3)
    else:
        c3 = float("nan")

    interlace_total_trials = 0
    interlace_total_pass = 0
    q_total_steps = 0
    q_root_le_dbar = 0

    for rr in runs:
        for trial_row in rr["interlace_partition_trials"]:
            st = trial_row["stats"]
            interlace_total_trials += st["trials"]
            interlace_total_pass += st["interlace_ab"]
        for s in rr["steps"]:
            if np.isfinite(s["Q_max_root_over_dbar"]):
                q_total_steps += 1
                if s["Q_max_root_over_dbar"] <= 1.0 + 1e-8:
                    q_root_le_dbar += 1

    summary = {
        "task4_interlacing": {
            "total_steps_with_ratio": q_total_steps,
            "Q_real_rooted_failures": len(q_non_real),
            "max_Q_root_over_dbar": float(max(q_ratios)) if q_ratios else float("nan"),
            "worst_Q_ratio_case": q_ratio_worst,
            "count_Q_root_le_dbar": q_root_le_dbar,
            "interlace_trials_total": interlace_total_trials,
            "interlace_trials_pass": interlace_total_pass,
            "interlace_pass_rate": (
                interlace_total_pass / interlace_total_trials if interlace_total_trials else float("nan")
            ),
            "interlace_failures": interlace_failures,
            "Q_real_rooted_failure_examples": q_non_real,
        },
        "task5_alpha": {
            "num_vertex_rows": len(all_alpha_v),
            "num_edge_rows": len(all_alpha_edge),
            "max_alpha_v": float(alpha_v_max[0]),
            "max_alpha_v_case": alpha_v_max[1],
            "max_alpha_uv": float(alpha_e_max[0]),
            "max_alpha_uv_case": alpha_e_max[1],
            "alpha_v_lt_half_all": len(alpha_v_viols) == 0,
            "alpha_v_gt_half_violations": alpha_v_viols,
            "corr_alpha_v_eff_res_min": c1,
            "corr_alpha_v_eff_res_mean": c2,
            "corr_alpha_uv_tau": c3,
            "horizon_x": x_horizon,
        },
    }

    return {
        "meta": {
            "date": "2026-02-13",
            "agent": "Codex",
            "seed": seed,
            "num_partition_trials": num_partition_trials,
            "suite": [name for name, _, _ in suite],
            "eps_list": eps_list,
            "num_runs": len(runs),
        },
        "summary": summary,
        "runs": runs,
    }


def print_summary(out: Dict):
    s4 = out["summary"]["task4_interlacing"]
    s5 = out["summary"]["task5_alpha"]

    print("=" * 92)
    print("P6 CYCLE 4 CODEX: INTERLACING + ALPHA PROBES")
    print("=" * 92)

    print("\nTask 4 (interlacing probe)")
    print(f"  steps with Q-root ratio: {s4['total_steps_with_ratio']}")
    print(f"  Q real-rooted failures: {s4['Q_real_rooted_failures']}")
    print(f"  max largest_root(Q)/dbar: {s4['max_Q_root_over_dbar']:.6f}")
    print(
        f"  partition interlace pass rate: {s4['interlace_trials_pass']}/"
        f"{s4['interlace_trials_total']} = {s4['interlace_pass_rate']:.4f}"
    )

    print("\nTask 5 (alpha probe)")
    print(f"  vertex rows: {s5['num_vertex_rows']}  edge rows: {s5['num_edge_rows']}")
    print(f"  max alpha_v: {s5['max_alpha_v']:.6f}")
    print(f"  max alpha_uv: {s5['max_alpha_uv']:.6f}")
    print(f"  alpha_v < 1/2 all: {s5['alpha_v_lt_half_all']}")
    print(f"  corr(alpha_v, eff_res_min_to_S): {s5['corr_alpha_v_eff_res_min']:.4f}")
    print(f"  corr(alpha_v, eff_res_mean_to_S): {s5['corr_alpha_v_eff_res_mean']:.4f}")
    print(f"  corr(alpha_uv, tau_uv): {s5['corr_alpha_uv_tau']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--partition-trials", type=int, default=10)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("data/first-proof/problem6-codex-cycle4-results.json"),
    )
    args = parser.parse_args()

    out = build_results(seed=args.seed, num_partition_trials=args.partition_trials)
    print_summary(out)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=to_jsonable)

    print(f"\nWrote JSON: {args.out_json}")


if __name__ == "__main__":
    main()
