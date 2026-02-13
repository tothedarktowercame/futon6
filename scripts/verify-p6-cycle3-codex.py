#!/usr/bin/env python3
"""Cycle 3 verifier for the Neumann alignment closure attempt in Problem 6.

Checks:
1) Candidate identity for tr(F_t) against t and tau=tr(M_t).
2) rho_k <= 1/2 for k=1,2,3,4.
3) Neumann amplification bound:
     dbar <= dbar_M0 * (2-x)/(2*(1-x)), x=||M||/eps.
4) x trajectory, including K_n exact x_t = t/(n*eps).
5) Intermediate inequality:
     tr(M^k F) <= mu_max^k * (t - tau), k=1,2,3.

Also performs sanity checks:
- tr(F_t) from matrix trace vs cross-edge leverage sum.
- F_t <= Pi - M_t in Loewner order.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


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
    raise RuntimeError(f"Failed to sample connected ER_{n}_p{p} after many retries")


def graph_laplacian(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    L = np.zeros((n, n), dtype=float)
    for u, v in edges:
        L[u, u] += 1.0
        L[v, v] += 1.0
        L[u, v] -= 1.0
        L[v, u] -= 1.0
    return L


def pseudo_sqrt_inv(L: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(L)
    out = np.zeros_like(L)
    for i, lam in enumerate(eigvals):
        if lam > 1e-10:
            out += (1.0 / np.sqrt(lam)) * np.outer(eigvecs[:, i], eigvecs[:, i])
    return out


def compute_edge_matrices(
    n: int,
    edges: List[Tuple[int, int]],
    Lph: np.ndarray,
) -> Tuple[List[np.ndarray], List[float]]:
    x_edges: List[np.ndarray] = []
    taus: List[float] = []
    for u, v in edges:
        b = np.zeros(n, dtype=float)
        b[u] = 1.0
        b[v] = -1.0
        z = Lph @ b
        xe = np.outer(z, z)
        x_edges.append(xe)
        taus.append(float(np.dot(z, z)))
    return x_edges, taus


def find_i0(n: int, edges: List[Tuple[int, int]], taus: List[float], eps: float) -> List[int]:
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


@dataclass
class RunConfig:
    name: str
    n: int
    edges: List[Tuple[int, int]]


@dataclass
class StepRow:
    t: int
    r_t: int
    tr_F: float
    cross_tau_sum: float
    tau: float
    selected_ell_sum: float
    identity_diff_t_tau: float
    identity_diff_sel_ell: float
    loewner_max_eig: float
    norm_M: float
    x: float
    kn_x_formula: float | None
    kn_x_error: float | None
    dbar: float
    dbar_m0: float
    neumann_bound: float
    neumann_ratio: float
    rho: Dict[str, float]
    lhs_k: Dict[str, float]
    rhs_k_t_tau: Dict[str, float]
    gap_k_t_tau: Dict[str, float]
    rhs_k_trF_half: Dict[str, float]
    gap_k_trF_half: Dict[str, float]


@dataclass
class RunResult:
    graph: str
    n: int
    eps: float
    m0: int
    horizon_T: int
    barrier_broken: bool
    break_t: int | None
    rows: List[StepRow]


def build_f_and_cross(
    n: int,
    S: List[int],
    R: List[int],
    edge_idx: Dict[Tuple[int, int], int],
    x_edges: List[np.ndarray],
    taus: List[float],
) -> Tuple[np.ndarray, float]:
    F = np.zeros((n, n), dtype=float)
    cross_tau_sum = 0.0
    for u in S:
        for v in R:
            idx = edge_idx.get(edge_key(u, v))
            if idx is None:
                continue
            F += x_edges[idx]
            cross_tau_sum += taus[idx]
    return F, cross_tau_sum


def build_c_v(
    n: int,
    v: int,
    S: List[int],
    edge_idx: Dict[Tuple[int, int], int],
    x_edges: List[np.ndarray],
) -> np.ndarray:
    C = np.zeros((n, n), dtype=float)
    for u in S:
        idx = edge_idx.get(edge_key(u, v))
        if idx is not None:
            C += x_edges[idx]
    return C


def run_greedy(cfg: RunConfig, eps: float, tol: float = 1e-10) -> RunResult:
    n = cfg.n
    edges = cfg.edges

    L = graph_laplacian(n, edges)
    Lph = pseudo_sqrt_inv(L)
    x_edges, taus = compute_edge_matrices(n, edges, Lph)

    # For connected graphs Pi = I - (1/n)J.
    Pi = np.eye(n) - np.ones((n, n), dtype=float) / n

    i0 = find_i0(n, edges, taus, eps)
    i0_set = set(i0)
    m0 = len(i0)
    horizon_T = max(1, min(int(eps * m0 / 3), m0 - 1)) if m0 >= 2 else 0

    edge_idx: Dict[Tuple[int, int], int] = {}
    for idx, (u, v) in enumerate(edges):
        if u in i0_set and v in i0_set:
            edge_idx[edge_key(u, v)] = idx

    ell = {}
    for v in i0:
        ev = 0.0
        for u in i0:
            if u == v:
                continue
            idx = edge_idx.get(edge_key(u, v))
            if idx is not None:
                ev += taus[idx]
        ell[v] = ev

    S: List[int] = []
    S_set = set()
    M = np.zeros((n, n), dtype=float)

    rows: List[StepRow] = []
    barrier_broken = False
    break_t: int | None = None

    for t in range(horizon_T + 1):
        R = [v for v in i0 if v not in S_set]
        r_t = len(R)
        if r_t == 0:
            break

        F, cross_tau_sum = build_f_and_cross(n, S, R, edge_idx, x_edges, taus)

        tau = float(np.trace(M))
        tr_F = float(np.trace(F))
        selected_ell_sum = float(sum(ell[v] for v in S))

        identity_diff_t_tau = tr_F - 2.0 * (len(S) - tau)
        identity_diff_sel_ell = tr_F - (selected_ell_sum - 2.0 * tau)

        # Loewner sanity: F <= Pi - M  <=>  lambda_max(F + M - Pi) <= 0.
        loewner_mat = (F + M - Pi)
        loewner_mat = 0.5 * (loewner_mat + loewner_mat.T)
        loewner_max_eig = float(np.max(np.linalg.eigvalsh(loewner_mat)))

        norm_M = float(np.linalg.norm(M, ord=2))
        x = norm_M / eps if eps > 0 else float("nan")

        kn_x_formula = None
        kn_x_error = None
        if cfg.name.startswith("K_") and m0 == n:
            if len(S) <= 1:
                kn_x_formula = 0.0
            else:
                kn_x_formula = len(S) / (n * eps)
            kn_x_error = abs(x - kn_x_formula)

        H = eps * np.eye(n) - M
        eigH = np.linalg.eigvalsh(H)
        if float(np.min(eigH)) < tol:
            barrier_broken = True
            break_t = t
            break

        B = np.linalg.inv(H)
        dbar = float(np.trace(B @ F) / r_t)
        dbar_m0 = float(tr_F / (eps * r_t)) if eps > 0 else float("nan")

        if x < 1.0 - 1e-12:
            neumann_factor = (2.0 - x) / (2.0 * (1.0 - x))
            neumann_bound = dbar_m0 * neumann_factor
            neumann_ratio = dbar / neumann_bound if neumann_bound > 1e-14 else float("nan")
        else:
            neumann_bound = float("inf")
            neumann_ratio = float("nan")

        rho: Dict[str, float] = {}
        lhs_k: Dict[str, float] = {}
        rhs_k_t_tau: Dict[str, float] = {}
        gap_k_t_tau: Dict[str, float] = {}
        rhs_k_trF_half: Dict[str, float] = {}
        gap_k_trF_half: Dict[str, float] = {}

        for k in [1, 2, 3, 4]:
            Mk = np.linalg.matrix_power(M, k)
            lhs = float(np.trace(Mk @ F))
            lhs_k[str(k)] = lhs

            denom = (norm_M ** k) * tr_F
            if tr_F > 1e-14 and norm_M > 1e-14:
                rho_k = lhs / denom
            else:
                rho_k = 0.0
            rho[str(k)] = rho_k

            if k <= 3:
                rhs1 = (norm_M ** k) * (len(S) - tau)
                rhs2 = (norm_M ** k) * (tr_F / 2.0)
                rhs_k_t_tau[str(k)] = rhs1
                rhs_k_trF_half[str(k)] = rhs2
                gap_k_t_tau[str(k)] = lhs - rhs1
                gap_k_trF_half[str(k)] = lhs - rhs2

        rows.append(
            StepRow(
                t=len(S),
                r_t=r_t,
                tr_F=tr_F,
                cross_tau_sum=cross_tau_sum,
                tau=tau,
                selected_ell_sum=selected_ell_sum,
                identity_diff_t_tau=identity_diff_t_tau,
                identity_diff_sel_ell=identity_diff_sel_ell,
                loewner_max_eig=loewner_max_eig,
                norm_M=norm_M,
                x=x,
                kn_x_formula=kn_x_formula,
                kn_x_error=kn_x_error,
                dbar=dbar,
                dbar_m0=dbar_m0,
                neumann_bound=neumann_bound,
                neumann_ratio=neumann_ratio,
                rho=rho,
                lhs_k=lhs_k,
                rhs_k_t_tau=rhs_k_t_tau,
                gap_k_t_tau=gap_k_t_tau,
                rhs_k_trF_half=rhs_k_trF_half,
                gap_k_trF_half=gap_k_trF_half,
            )
        )

        if len(S) >= horizon_T:
            break

        # Greedy choice for next vertex.
        Bsqrt = np.linalg.cholesky(B + 1e-14 * np.eye(n))
        scores = {}
        for v in R:
            C_v = build_c_v(n, v, S, edge_idx, x_edges)
            Y = Bsqrt @ C_v @ Bsqrt.T
            Y = 0.5 * (Y + Y.T)
            scores[v] = float(np.max(np.linalg.eigvalsh(Y)))

        best_v = min(R, key=lambda v: (scores[v], v))
        S.append(best_v)
        S_set.add(best_v)

        for u in S[:-1]:
            idx = edge_idx.get(edge_key(u, best_v))
            if idx is not None:
                M += x_edges[idx]

    return RunResult(
        graph=cfg.name,
        n=n,
        eps=eps,
        m0=m0,
        horizon_T=horizon_T,
        barrier_broken=barrier_broken,
        break_t=break_t,
        rows=rows,
    )


def row_to_json(r: StepRow) -> Dict:
    return {
        "t": r.t,
        "r_t": r.r_t,
        "tr_F": r.tr_F,
        "cross_tau_sum": r.cross_tau_sum,
        "tau": r.tau,
        "selected_ell_sum": r.selected_ell_sum,
        "identity_diff_t_tau": r.identity_diff_t_tau,
        "identity_diff_sel_ell": r.identity_diff_sel_ell,
        "loewner_max_eig": r.loewner_max_eig,
        "norm_M": r.norm_M,
        "x": r.x,
        "kn_x_formula": r.kn_x_formula,
        "kn_x_error": r.kn_x_error,
        "dbar": r.dbar,
        "dbar_m0": r.dbar_m0,
        "neumann_bound": r.neumann_bound,
        "neumann_ratio": r.neumann_ratio,
        "rho": r.rho,
        "lhs_k": r.lhs_k,
        "rhs_k_t_tau": r.rhs_k_t_tau,
        "gap_k_t_tau": r.gap_k_t_tau,
        "rhs_k_trF_half": r.rhs_k_trF_half,
        "gap_k_trF_half": r.gap_k_trF_half,
    }


def summarize(results: List[RunResult], tol: float = 1e-8) -> Dict:
    identity_max_t_tau = 0.0
    identity_argmax_t_tau = None
    identity_max_sel_ell = 0.0
    identity_argmax_sel_ell = None
    trF_cross_max = 0.0
    trF_cross_argmax = None

    loewner_max = -np.inf
    loewner_argmax = None

    rho_max = {str(k): -np.inf for k in [1, 2, 3, 4]}
    rho_argmax = {str(k): None for k in [1, 2, 3, 4]}
    rho_viol = {str(k): [] for k in [1, 2, 3, 4]}

    neumann_max_ratio = -np.inf
    neumann_argmax = None
    neumann_viol = []

    kn_x_max_err = 0.0
    kn_x_argmax = None

    horizon_rows = []
    horizon_x_viol = []

    k_gap_max_t_tau = {str(k): -np.inf for k in [1, 2, 3]}
    k_gap_argmax_t_tau = {str(k): None for k in [1, 2, 3]}
    k_viol_t_tau = {str(k): [] for k in [1, 2, 3]}

    k_gap_max_trF_half = {str(k): -np.inf for k in [1, 2, 3]}
    k_gap_argmax_trF_half = {str(k): None for k in [1, 2, 3]}
    k_viol_trF_half = {str(k): [] for k in [1, 2, 3]}

    barrier_breaks = []

    for rr in results:
        if rr.barrier_broken:
            barrier_breaks.append(
                {
                    "graph": rr.graph,
                    "eps": rr.eps,
                    "break_t": rr.break_t,
                }
            )

        if rr.rows:
            horizon_row = rr.rows[-1]
            horizon_rows.append(
                {
                    "graph": rr.graph,
                    "eps": rr.eps,
                    "n": rr.n,
                    "m0": rr.m0,
                    "horizon_T": rr.horizon_T,
                    "t_recorded": horizon_row.t,
                    "x_horizon": horizon_row.x,
                }
            )
            if horizon_row.x > (1.0 / 3.0 + tol):
                horizon_x_viol.append(
                    {
                        "graph": rr.graph,
                        "eps": rr.eps,
                        "t": horizon_row.t,
                        "x": horizon_row.x,
                    }
                )

        for row in rr.rows:
            d1 = abs(row.identity_diff_t_tau)
            if d1 > identity_max_t_tau:
                identity_max_t_tau = d1
                identity_argmax_t_tau = {
                    "graph": rr.graph,
                    "eps": rr.eps,
                    "t": row.t,
                    "tr_F": row.tr_F,
                    "tau": row.tau,
                    "lhs": row.tr_F,
                    "rhs": 2.0 * (row.t - row.tau),
                    "abs_diff": d1,
                }

            d2 = abs(row.identity_diff_sel_ell)
            if d2 > identity_max_sel_ell:
                identity_max_sel_ell = d2
                identity_argmax_sel_ell = {
                    "graph": rr.graph,
                    "eps": rr.eps,
                    "t": row.t,
                    "lhs": row.tr_F,
                    "rhs": row.selected_ell_sum - 2.0 * row.tau,
                    "abs_diff": d2,
                }

            d3 = abs(row.tr_F - row.cross_tau_sum)
            if d3 > trF_cross_max:
                trF_cross_max = d3
                trF_cross_argmax = {
                    "graph": rr.graph,
                    "eps": rr.eps,
                    "t": row.t,
                    "abs_diff": d3,
                }

            if row.loewner_max_eig > loewner_max:
                loewner_max = row.loewner_max_eig
                loewner_argmax = {
                    "graph": rr.graph,
                    "eps": rr.eps,
                    "t": row.t,
                    "max_eig_F_plus_M_minus_Pi": row.loewner_max_eig,
                }

            for k in [1, 2, 3, 4]:
                ks = str(k)
                rv = row.rho[ks]
                if rv > rho_max[ks]:
                    rho_max[ks] = rv
                    rho_argmax[ks] = {
                        "graph": rr.graph,
                        "eps": rr.eps,
                        "t": row.t,
                        "rho": rv,
                    }
                if rv > 0.5 + tol:
                    rho_viol[ks].append(
                        {
                            "graph": rr.graph,
                            "eps": rr.eps,
                            "t": row.t,
                            "rho": rv,
                        }
                    )

            if np.isfinite(row.neumann_ratio):
                if row.neumann_ratio > neumann_max_ratio:
                    neumann_max_ratio = row.neumann_ratio
                    neumann_argmax = {
                        "graph": rr.graph,
                        "eps": rr.eps,
                        "t": row.t,
                        "ratio": row.neumann_ratio,
                        "dbar": row.dbar,
                        "bound": row.neumann_bound,
                    }
                if row.neumann_ratio > 1.0 + tol:
                    neumann_viol.append(
                        {
                            "graph": rr.graph,
                            "eps": rr.eps,
                            "t": row.t,
                            "ratio": row.neumann_ratio,
                            "dbar": row.dbar,
                            "bound": row.neumann_bound,
                        }
                    )

            if row.kn_x_error is not None and row.kn_x_error > kn_x_max_err:
                kn_x_max_err = row.kn_x_error
                kn_x_argmax = {
                    "graph": rr.graph,
                    "eps": rr.eps,
                    "t": row.t,
                    "x": row.x,
                    "formula": row.kn_x_formula,
                    "abs_err": row.kn_x_error,
                }

            for k in [1, 2, 3]:
                ks = str(k)
                g1 = row.gap_k_t_tau[ks]
                if g1 > k_gap_max_t_tau[ks]:
                    k_gap_max_t_tau[ks] = g1
                    k_gap_argmax_t_tau[ks] = {
                        "graph": rr.graph,
                        "eps": rr.eps,
                        "t": row.t,
                        "lhs": row.lhs_k[ks],
                        "rhs": row.rhs_k_t_tau[ks],
                        "gap": g1,
                    }
                if g1 > tol:
                    k_viol_t_tau[ks].append(
                        {
                            "graph": rr.graph,
                            "eps": rr.eps,
                            "t": row.t,
                            "gap": g1,
                            "lhs": row.lhs_k[ks],
                            "rhs": row.rhs_k_t_tau[ks],
                        }
                    )

                g2 = row.gap_k_trF_half[ks]
                if g2 > k_gap_max_trF_half[ks]:
                    k_gap_max_trF_half[ks] = g2
                    k_gap_argmax_trF_half[ks] = {
                        "graph": rr.graph,
                        "eps": rr.eps,
                        "t": row.t,
                        "lhs": row.lhs_k[ks],
                        "rhs": row.rhs_k_trF_half[ks],
                        "gap": g2,
                    }
                if g2 > tol:
                    k_viol_trF_half[ks].append(
                        {
                            "graph": rr.graph,
                            "eps": rr.eps,
                            "t": row.t,
                            "gap": g2,
                            "lhs": row.lhs_k[ks],
                            "rhs": row.rhs_k_trF_half[ks],
                        }
                    )

    answer_holds = (
        all(len(rho_viol[str(k)]) == 0 for k in [1, 2, 3, 4])
        and len(neumann_viol) == 0
    )

    return {
        "task1_identity": {
            "max_abs_trF_minus_2_t_minus_tau": identity_max_t_tau,
            "argmax_trF_minus_2_t_minus_tau": identity_argmax_t_tau,
            "max_abs_trF_minus_selected_ell_minus_2tau": identity_max_sel_ell,
            "argmax_trF_minus_selected_ell_minus_2tau": identity_argmax_sel_ell,
            "max_abs_trace_vs_cross_tau_sum": trF_cross_max,
            "argmax_trace_vs_cross_tau_sum": trF_cross_argmax,
            "inferred_identity": "tr(F_t) = sum_{u in S_t} ell_u - 2*tr(M_t)",
        },
        "sanity_loewner": {
            "max_eig_F_plus_M_minus_Pi": loewner_max,
            "argmax": loewner_argmax,
        },
        "task2_rho_bounds": {
            "rho_max": rho_max,
            "rho_argmax": rho_argmax,
            "violations_gt_half": rho_viol,
        },
        "task3_neumann_bound": {
            "max_ratio_dbar_over_bound": neumann_max_ratio,
            "argmax": neumann_argmax,
            "violations_gt_1": neumann_viol,
        },
        "task4_x_trajectory": {
            "kn_max_abs_error": kn_x_max_err,
            "kn_argmax": kn_x_argmax,
            "horizon_rows": horizon_rows,
            "horizon_x_gt_one_third": horizon_x_viol,
        },
        "task5_intermediate_inequality": {
            "max_gap_t_minus_tau": k_gap_max_t_tau,
            "argmax_t_minus_tau": k_gap_argmax_t_tau,
            "violations_t_minus_tau": k_viol_t_tau,
            "max_gap_trF_half": k_gap_max_trF_half,
            "argmax_trF_half": k_gap_argmax_trF_half,
            "violations_trF_half": k_viol_trF_half,
        },
        "barrier_breaks": barrier_breaks,
        "key_question": {
            "holds_numerically_all_tested": answer_holds,
            "condition": "rho_k <= 1/2 for k=1..4 and Neumann ratio <= 1",
        },
    }


def print_summary(summary: Dict):
    t1 = summary["task1_identity"]
    t2 = summary["task2_rho_bounds"]
    t3 = summary["task3_neumann_bound"]
    t4 = summary["task4_x_trajectory"]
    t5 = summary["task5_intermediate_inequality"]

    print("=" * 92)
    print("P6 CYCLE 3 CODEX: NEUMANN ALIGNMENT VERIFICATION")
    print("=" * 92)

    print("\nTask 1: tr(F_t) identity")
    print(f"  max |tr(F_t) - 2(t-tau)| = {t1['max_abs_trF_minus_2_t_minus_tau']:.6e}")
    print(
        "  max |tr(F_t) - (sum_{u in S} ell_u - 2 tau)| = "
        f"{t1['max_abs_trF_minus_selected_ell_minus_2tau']:.6e}"
    )
    print(f"  max |trace(F_t) - cross_tau_sum| = {t1['max_abs_trace_vs_cross_tau_sum']:.6e}")
    print(f"  inferred identity: {t1['inferred_identity']}")

    print("\nSanity: F_t <= Pi - M_t")
    print(
        "  max eig(F_t + M_t - Pi) = "
        f"{summary['sanity_loewner']['max_eig_F_plus_M_minus_Pi']:.6e}"
    )

    print("\nTask 2: rho_k <= 1/2")
    for k in [1, 2, 3, 4]:
        ks = str(k)
        max_r = t2["rho_max"][ks]
        viol = len(t2["violations_gt_half"][ks])
        print(f"  k={k}: max rho_k = {max_r:.6f}, violations={viol}")

    print("\nTask 3: Neumann bound")
    print(f"  max dbar/bound = {t3['max_ratio_dbar_over_bound']:.6f}")
    print(f"  violations > 1: {len(t3['violations_gt_1'])}")

    print("\nTask 4: x trajectory")
    print(f"  K_n max |x - t/(n eps)| = {t4['kn_max_abs_error']:.6e}")
    print(f"  horizon x_t > 1/3 count = {len(t4['horizon_x_gt_one_third'])}")

    print("\nTask 5: tr(M^k F) <= mu^k*(t-tau)")
    for k in [1, 2, 3]:
        ks = str(k)
        print(
            f"  k={k}: max(lhs-rhs_t_tau)={t5['max_gap_t_minus_tau'][ks]:.6e}, "
            f"viol={len(t5['violations_t_minus_tau'][ks])}"
        )
    print("  (alternative rhs using tr(F_t)/2)")
    for k in [1, 2, 3]:
        ks = str(k)
        print(
            f"  k={k}: max(lhs-rhs_trF_half)={t5['max_gap_trF_half'][ks]:.6e}, "
            f"viol={len(t5['violations_trF_half'][ks])}"
        )

    print("\nKey question")
    print(f"  holds on all tested instances: {summary['key_question']['holds_numerically_all_tested']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("data/first-proof/problem6-codex-cycle3-results.json"),
    )
    args = parser.parse_args()

    # Test suite required by task prompt.
    er_edges, er_rep = connected_er(60, 0.5, seed=42)
    n_b, e_b = barbell_graph(40)
    n_g, e_g = grid_graph(8, 5)

    suite = [
        RunConfig("K_40", 40, complete_graph(40)),
        RunConfig("K_80", 80, complete_graph(80)),
        RunConfig(f"ER_60_p0.5_seed42_rep{er_rep}", 60, er_edges),
        RunConfig("Barbell_40", n_b, e_b),
        RunConfig("Star_40", 40, star_graph(40)),
        RunConfig("Grid_8x5", n_g, e_g),
    ]
    eps_list = [0.2, 0.3, 0.5]

    runs: List[RunResult] = []
    for cfg in suite:
        for eps in eps_list:
            runs.append(run_greedy(cfg, eps))

    summary = summarize(runs)
    print_summary(summary)

    out = {
        "meta": {
            "date": "2026-02-13",
            "agent": "Codex",
            "suite": [cfg.name for cfg in suite],
            "eps_list": eps_list,
            "num_runs": len(runs),
        },
        "summary": summary,
        "runs": [
            {
                "graph": rr.graph,
                "n": rr.n,
                "eps": rr.eps,
                "m0": rr.m0,
                "horizon_T": rr.horizon_T,
                "barrier_broken": rr.barrier_broken,
                "break_t": rr.break_t,
                "rows": [row_to_json(r) for r in rr.rows],
            }
            for rr in runs
        ],
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nWrote JSON: {args.out_json}")


if __name__ == "__main__":
    main()
