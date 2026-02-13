#!/usr/bin/env python3
"""Problem 6 Cycle 6: bridge verification for conditional dbar0 < 1 proof.

Implements handoff tasks:
1) Numerical comparison of three trajectories:
   - barrier greedy (min tr(Y) among barrier-feasible candidates)
   - leverage-degree prefix
   - modified greedy (process ell-order, add if barrier-feasible)
2) Order-comparison diagnostics for bridge option A.
3) Modified construction test for bridge option B.
4) Direct-induction delta probe for bridge option C.
5) Strict-light threshold (tau >= eps heavy) exhaustive small-n scan.
"""

from __future__ import annotations

import argparse
import json
import math
import runpy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


# Reuse tested graph/laplacian helpers from Cycle 5.
MOD = runpy.run_path(str(Path(__file__).with_name("verify-p6-cycle5-codex.py")))


def to_jsonable(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


@dataclass
class CaseData:
    name: str
    n: int
    edges: List[Tuple[int, int]]
    Lplus: np.ndarray
    Lph: np.ndarray
    zmat: np.ndarray
    taus: np.ndarray


def edge_key(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def find_i0_threshold(
    n: int,
    edges: Sequence[Tuple[int, int]],
    taus: np.ndarray,
    eps: float,
    heavy_ge: bool,
) -> List[int]:
    tol = 1e-12
    heavy_adj = [set() for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        # Tolerance avoids floating-point knife-edge misclassification at tau == eps.
        heavy = (taus[idx] >= eps - tol) if heavy_ge else (taus[idx] > eps + tol)
        if heavy:
            heavy_adj[u].add(v)
            heavy_adj[v].add(u)
    i_set = set()
    for v in sorted(range(n), key=lambda vv: len(heavy_adj[vv])):
        if all(u not in i_set for u in heavy_adj[v]):
            i_set.add(v)
    return sorted(i_set)


def build_cycle6_suite(seed: int) -> List[Tuple[str, int, List[Tuple[int, int]]]]:
    rng = np.random.default_rng(seed)
    out: List[Tuple[str, int, List[Tuple[int, int]]]] = []

    # C5 core families.
    for n in [40, 80, 100]:
        out.append((f"K_{n}", n, MOD["complete_graph"](n)))

    for d, s in [(3, 3103), (10, 3110), (50, 3150), (6, 6006)]:
        edges = MOD["randomize_regular_graph"](100, d, s)
        out.append((f"Reg_100_d{d}", 100, edges))

    for p, s in [(0.1, 9101), (0.3, 9303), (0.5, 9505)]:
        edges, rep = MOD["connected_er"](100, p, seed=s)
        out.append((f"ER_100_p{p}_rep{rep}", 100, edges))

    out.append(("Tree_prufer_100", 100, MOD["prufer_random_tree"](100, seed=777)))
    out.append(("Path_100", 100, MOD["path_graph"](100)))
    out.append(("Star_100", 100, MOD["star_graph"](100)))

    n, edges = MOD["complete_bipartite"](50, 50)
    out.append(("K_50_50", n, edges))
    n, edges = MOD["random_bipartite_connected"](50, 50, p=0.2, seed=4242)
    out.append(("BipRand_50_50_p0.2", n, edges))

    # C5b-style randomized families.
    ns = [60, 80, 120]
    for i in range(4):
        n = int(ns[i % len(ns)])
        p = float(0.15 + 0.6 * rng.random())
        edges, rep = MOD["connected_er"](n, p, seed=int(rng.integers(1, 1_000_000)))
        out.append((f"C5b_ER_n{n}_p{p:.3f}_rep{rep}_i{i}", n, edges))

    d_choices = [4, 6, 8, 10, 12, 16, 20]
    for i in range(4):
        n = int(ns[i % len(ns)])
        d = int(d_choices[i % len(d_choices)])
        if d >= n:
            d = n - 1
        if d % 2 == 1 and n % 2 == 1:
            d += 1
        if d >= n:
            d = n - 2
        edges = MOD["randomize_regular_graph"](n, d, seed=int(rng.integers(1, 1_000_000)), num_switches=4000)
        out.append((f"C5b_Reg_n{n}_d{d}_i{i}", n, edges))

    for i in range(3):
        n = int(ns[i % len(ns)])
        nl = n // 2
        nr = n - nl
        p = float(0.12 + 0.45 * rng.random())
        n2, edges = MOD["random_bipartite_connected"](nl, nr, p=p, seed=int(rng.integers(1, 1_000_000)))
        out.append((f"C5b_Bip_n{n2}_p{p:.3f}_i{i}", n2, edges))

    for i in range(3):
        n = int(ns[i % len(ns)])
        edges = MOD["prufer_random_tree"](n, seed=int(rng.integers(1, 1_000_000)))
        out.append((f"C5b_Tree_n{n}_i{i}", n, edges))

    return out


def prepare_case(name: str, n: int, edges: List[Tuple[int, int]]) -> CaseData:
    if not MOD["is_connected"](n, edges):
        raise RuntimeError(f"disconnected graph: {name}")
    L = MOD["graph_laplacian"](n, edges)
    Lplus, Lph = MOD["pseudo_inv_and_half"](L)
    zmat, taus = MOD["compute_edge_z"](n, edges, Lph)
    return CaseData(name=name, n=n, edges=edges, Lplus=Lplus, Lph=Lph, zmat=zmat, taus=taus)


def build_i0_graph_data(case: CaseData, eps: float, heavy_ge: bool) -> Dict:
    i0 = find_i0_threshold(case.n, case.edges, case.taus, eps, heavy_ge=heavy_ge)
    i0_set = set(i0)

    # Map restricted internal-to-I0 edges.
    edge_rows = []
    neighbors: Dict[int, List[Tuple[int, int, float]]] = {v: [] for v in i0}
    for idx, (u, v) in enumerate(case.edges):
        if u in i0_set and v in i0_set:
            tau = float(case.taus[idx])
            edge_rows.append((u, v, idx, tau))
            neighbors[u].append((v, idx, tau))
            neighbors[v].append((u, idx, tau))

    ell = {v: float(sum(tau for (_, _, tau) in neighbors[v])) for v in i0}
    ell_order = sorted(i0, key=lambda v: (ell[v], v))

    edge_set = {(u, v): (idx, tau) for (u, v, idx, tau) in edge_rows}
    edge_set.update({(v, u): (idx, tau) for (u, v, idx, tau) in edge_rows})

    return {
        "i0": i0,
        "i0_set": i0_set,
        "neighbors": neighbors,
        "edge_rows": edge_rows,
        "edge_set": edge_set,
        "ell": ell,
        "ell_order": ell_order,
    }


def partition_metrics(
    S: Sequence[int],
    i0: Sequence[int],
    edge_rows: Sequence[Tuple[int, int, int, float]],
    ell: Dict[int, float],
    eps: float,
) -> Dict:
    S_set = set(S)
    t = len(S)
    m = len(i0)
    r = m - t

    tau_internal = 0.0
    trF = 0.0
    for u, v, _, tau in edge_rows:
        in_u = u in S_set
        in_v = v in S_set
        if in_u and in_v:
            tau_internal += tau
        elif in_u != in_v:
            trF += tau

    dbar0 = trF / (r * eps) if (r > 0 and eps > 0) else 0.0
    sum_ell = float(sum(ell[v] for v in S))
    partial_avg_bound = (2.0 * t * (m - 1) / m) if m > 0 else 0.0

    return {
        "t": t,
        "r_t": r,
        "sum_ell_selected": sum_ell,
        "partial_averages_bound": partial_avg_bound,
        "sum_ell_minus_bound": sum_ell - partial_avg_bound,
        "tau_internal": tau_internal,
        "trF": trF,
        "dbar0": dbar0,
    }


def candidate_barrier_stats(
    case: CaseData,
    S_set: set[int],
    R: Sequence[int],
    neighbors: Dict[int, List[Tuple[int, int, float]]],
    M: np.ndarray,
    eps: float,
) -> Tuple[Dict[int, Dict[str, float]], float]:
    n = case.n
    H = eps * np.eye(n) - M
    evals_H, U_H = np.linalg.eigh(H)
    min_head = float(np.min(evals_H))
    if min_head <= 1e-11:
        return {}, min_head

    # Use the symmetric inverse square root H^{-1/2}, not a Cholesky factor of H^{-1}.
    Hinvhalf = np.zeros((n, n), dtype=float)
    for i, lam in enumerate(evals_H):
        if lam > 1e-12:
            ui = U_H[:, i]
            Hinvhalf += (1.0 / np.sqrt(lam)) * np.outer(ui, ui)

    out: Dict[int, Dict[str, float]] = {}
    for v in R:
        idxs = [idx for (u, idx, _) in neighbors[v] if u in S_set]
        if not idxs:
            out[v] = {
                "traceY": 0.0,
                "normY": 0.0,
                "cross_to_S": 0.0,
            }
            continue

        Zv = case.zmat[idxs]
        Pv = Hinvhalf @ Zv.T
        gram = Pv.T @ Pv
        evals = np.linalg.eigvalsh(gram)
        trY = float(np.sum(evals[evals > 1e-12]))
        normY = float(np.max(evals)) if len(evals) else 0.0
        cross_to_S = float(sum(tau for (u, _, tau) in neighbors[v] if u in S_set))
        out[v] = {
            "traceY": trY,
            "normY": normY,
            "cross_to_S": cross_to_S,
        }

    return out, min_head


def add_vertex_update_M(
    case: CaseData,
    S_set_after: set[int],
    v_added: int,
    neighbors: Dict[int, List[Tuple[int, int, float]]],
    M: np.ndarray,
) -> np.ndarray:
    idxs = [idx for (u, idx, _) in neighbors[v_added] if u in S_set_after and u != v_added]
    if idxs:
        Znew = case.zmat[idxs]
        M = M + Znew.T @ Znew
    return M


def run_barrier_greedy(
    case: CaseData,
    eps: float,
    i0_data: Dict,
    horizon: int,
) -> Dict:
    i0 = i0_data["i0"]
    ell = i0_data["ell"]
    neighbors = i0_data["neighbors"]
    edge_rows = i0_data["edge_rows"]

    S: List[int] = []
    S_set: set[int] = set()
    M = np.zeros((case.n, case.n), dtype=float)

    steps = []
    induction_rows = []

    # Baseline t=0.
    base = partition_metrics(S, i0, edge_rows, ell, eps)
    base.update(
        {
            "selected_v": None,
            "selected_ell": None,
            "feasible_candidates": None,
            "min_traceY": None,
            "min_normY": None,
            "M_norm": 0.0,
            "barrier_headroom_min_eig": eps,
        }
    )
    steps.append(base)

    completed = True
    fail_reason = None

    while len(S) < horizon:
        R = [v for v in i0 if v not in S_set]
        if not R:
            completed = False
            fail_reason = "no_remaining_vertices"
            break

        cand, min_head = candidate_barrier_stats(case, S_set, R, neighbors, M, eps)
        if min_head <= 1e-11:
            completed = False
            fail_reason = "barrier_headroom_nonpositive"
            break

        feasible = [v for v in R if cand[v]["normY"] < 1.0 - 1e-12]
        if not feasible:
            completed = False
            fail_reason = "no_barrier_feasible_vertex"
            break

        best_v = min(feasible, key=lambda v: (cand[v]["traceY"], v))

        # Direct-induction probe on selected vertex.
        t = len(S)
        r_t = len(R)
        pre = partition_metrics(S, i0, edge_rows, ell, eps)
        trF_t = pre["trF"]
        ell_v_to_S = float(sum(tau for (u, _, tau) in neighbors[best_v] if u in S_set))
        ell_v_to_Rnext = float(sum(tau for (u, _, tau) in neighbors[best_v] if (u not in S_set and u != best_v)))
        delta = ell_v_to_Rnext - ell_v_to_S
        rhs = (r_t - 1) * eps - trF_t
        lhs_minus_rhs = delta - rhs

        S.append(best_v)
        S_set.add(best_v)
        M = add_vertex_update_M(case, S_set, best_v, neighbors, M)

        post = partition_metrics(S, i0, edge_rows, ell, eps)
        M_eigs = np.linalg.eigvalsh(M)
        M_norm = float(np.max(M_eigs)) if len(M_eigs) else 0.0

        row = dict(post)
        row.update(
            {
                "selected_v": int(best_v),
                "selected_ell": float(ell[best_v]),
                "feasible_candidates": int(len(feasible)),
                "min_traceY": float(min(cand[v]["traceY"] for v in feasible)),
                "min_normY": float(min(cand[v]["normY"] for v in feasible)),
                "M_norm": M_norm,
                "barrier_headroom_min_eig": float(min_head),
            }
        )
        steps.append(row)

        induction_rows.append(
            {
                "t": t,
                "r_t": r_t,
                "selected_v": int(best_v),
                "ell_v_to_S_t": ell_v_to_S,
                "ell_v_to_R_next": ell_v_to_Rnext,
                "delta": delta,
                "rhs_budget": rhs,
                "delta_minus_rhs": lhs_minus_rhs,
                "condition_delta_lt_rhs": bool(delta < rhs),
                "trF_t": trF_t,
                "trF_t1": post["trF"],
                "recurrence_abs_err": abs(post["trF"] - (trF_t + delta)),
            }
        )

    return {
        "completed": completed,
        "selected_count": len(S),
        "horizon": horizon,
        "fail_reason": fail_reason,
        "steps": steps,
        "direct_induction": induction_rows,
    }


def run_leverage_prefix(
    case: CaseData,
    eps: float,
    i0_data: Dict,
    horizon: int,
) -> Dict:
    i0 = i0_data["i0"]
    ell = i0_data["ell"]
    edge_rows = i0_data["edge_rows"]
    order = i0_data["ell_order"]

    steps = []
    for t in range(horizon + 1):
        S = order[:t]
        row = partition_metrics(S, i0, edge_rows, ell, eps)
        row.update(
            {
                "selected_v": int(order[t - 1]) if t > 0 else None,
                "selected_ell": float(ell[order[t - 1]]) if t > 0 else None,
            }
        )
        steps.append(row)

    return {
        "completed": True,
        "selected_count": horizon,
        "horizon": horizon,
        "steps": steps,
    }


def run_modified_greedy(
    case: CaseData,
    eps: float,
    i0_data: Dict,
    horizon: int,
) -> Dict:
    i0 = i0_data["i0"]
    ell = i0_data["ell"]
    neighbors = i0_data["neighbors"]
    edge_rows = i0_data["edge_rows"]
    order = i0_data["ell_order"]

    S: List[int] = []
    S_set: set[int] = set()
    M = np.zeros((case.n, case.n), dtype=float)

    steps = []
    skips = []

    base = partition_metrics(S, i0, edge_rows, ell, eps)
    base.update(
        {
            "selected_v": None,
            "selected_ell": None,
            "processed_rank": None,
            "M_norm": 0.0,
            "barrier_headroom_min_eig": eps,
            "candidate_normY": None,
            "candidate_traceY": None,
        }
    )
    steps.append(base)

    processed = 0
    completed = True
    fail_reason = None

    for v in order:
        if len(S) >= horizon:
            break

        processed += 1

        R = [u for u in i0 if u not in S_set]
        cand, min_head = candidate_barrier_stats(case, S_set, R, neighbors, M, eps)
        if min_head <= 1e-11:
            completed = False
            fail_reason = "barrier_headroom_nonpositive"
            break

        stats = cand[v]
        feasible = stats["normY"] < 1.0 - 1e-12
        if feasible:
            S.append(v)
            S_set.add(v)
            M = add_vertex_update_M(case, S_set, v, neighbors, M)

            row = partition_metrics(S, i0, edge_rows, ell, eps)
            M_norm = float(np.max(np.linalg.eigvalsh(M))) if len(S) > 1 else 0.0
            row.update(
                {
                    "selected_v": int(v),
                    "selected_ell": float(ell[v]),
                    "processed_rank": processed,
                    "M_norm": M_norm,
                    "barrier_headroom_min_eig": float(min_head),
                    "candidate_normY": float(stats["normY"]),
                    "candidate_traceY": float(stats["traceY"]),
                }
            )
            steps.append(row)
        else:
            skips.append(
                {
                    "candidate_v": int(v),
                    "candidate_ell": float(ell[v]),
                    "processed_rank": processed,
                    "candidate_normY": float(stats["normY"]),
                    "candidate_traceY": float(stats["traceY"]),
                }
            )

    if len(S) < horizon:
        completed = False
        fail_reason = fail_reason or "ran_out_of_candidates_before_horizon"

    return {
        "completed": completed,
        "selected_count": len(S),
        "horizon": horizon,
        "fail_reason": fail_reason,
        "steps": steps,
        "skips": skips,
        "processed_total": processed,
    }


def run_three_way(case: CaseData, eps: float, heavy_ge: bool) -> Dict:
    i0_data = build_i0_graph_data(case, eps, heavy_ge=heavy_ge)
    m = len(i0_data["i0"])
    horizon = max(1, min(int(eps * m / 3), m - 1)) if m >= 2 else 0

    if m <= 1 or horizon == 0:
        # Trivial/vacuous case.
        base = {
            "completed": True,
            "selected_count": 0,
            "horizon": horizon,
            "steps": [partition_metrics([], i0_data["i0"], i0_data["edge_rows"], i0_data["ell"], eps)],
        }
        base["steps"][0].update({"selected_v": None, "selected_ell": None})

        return {
            "graph": case.name,
            "n": case.n,
            "eps": eps,
            "strict_threshold": heavy_ge,
            "m0": m,
            "horizon": horizon,
            "avg_ell": float(2.0 * (m - 1) / m) if m > 0 else 0.0,
            "i0": i0_data["i0"],
            "ell": {str(k): float(v) for k, v in i0_data["ell"].items()},
            "ell_order": [int(v) for v in i0_data["ell_order"]],
            "trajectories": {
                "barrier_greedy": {**base, "direct_induction": []},
                "leverage_prefix": dict(base),
                "modified_greedy": {**base, "skips": [], "processed_total": 0},
            },
        }

    barrier = run_barrier_greedy(case, eps, i0_data, horizon)
    prefix = run_leverage_prefix(case, eps, i0_data, horizon)
    modified = run_modified_greedy(case, eps, i0_data, horizon)

    return {
        "graph": case.name,
        "n": case.n,
        "eps": eps,
        "strict_threshold": heavy_ge,
        "m0": m,
        "horizon": horizon,
        "avg_ell": float(2.0 * (m - 1) / m),
        "i0": i0_data["i0"],
        "ell": {str(k): float(v) for k, v in i0_data["ell"].items()},
        "ell_order": [int(v) for v in i0_data["ell_order"]],
        "trajectories": {
            "barrier_greedy": barrier,
            "leverage_prefix": prefix,
            "modified_greedy": modified,
        },
    }


def build_small_exhaustive_suite(seed: int) -> List[Tuple[str, int, List[Tuple[int, int]]]]:
    rng = np.random.default_rng(seed)
    out: List[Tuple[str, int, List[Tuple[int, int]]]] = []

    for n in [10, 12, 14, 16, 18, 20]:
        out.append((f"K_{n}", n, MOD["complete_graph"](n)))

    for i, n in enumerate([12, 14, 16, 18, 20, 20, 18, 16]):
        p = float(0.2 + 0.55 * rng.random())
        edges, rep = MOD["connected_er"](n, p, seed=int(rng.integers(1, 1_000_000)))
        out.append((f"ER_{n}_i{i}_rep{rep}", n, edges))

    for i, (n, d) in enumerate([(12, 3), (14, 4), (16, 4), (18, 6), (20, 6), (20, 8)]):
        edges = MOD["randomize_regular_graph"](n, d, seed=int(rng.integers(1, 1_000_000)), num_switches=2500)
        out.append((f"Reg_{n}_d{d}_i{i}", n, edges))

    for i, n in enumerate([12, 14, 16, 18]):
        edges = MOD["prufer_random_tree"](n, seed=int(rng.integers(1, 1_000_000)))
        out.append((f"Tree_{n}_i{i}", n, edges))

    return out


def exhaustive_subset_scan(
    case: CaseData,
    eps: float,
    heavy_ge: bool,
) -> Dict:
    i0_data = build_i0_graph_data(case, eps, heavy_ge=heavy_ge)
    i0 = i0_data["i0"]
    m = len(i0)
    horizon = max(1, min(int(eps * m / 3), m - 1)) if m >= 2 else 0

    if m <= 1 or horizon == 0:
        return {
            "graph": case.name,
            "n": case.n,
            "eps": eps,
            "m0": m,
            "horizon": horizon,
            "max_dbar0": 0.0,
            "arg": None,
        }

    edge_rows = i0_data["edge_rows"]
    ell = i0_data["ell"]

    import itertools

    best_val = 0.0
    best_arg = None
    for t in range(1, horizon + 1):
        for S in itertools.combinations(i0, t):
            pm = partition_metrics(S, i0, edge_rows, ell, eps)
            dbar0 = pm["dbar0"]
            if dbar0 > best_val:
                best_val = dbar0
                best_arg = {
                    "t": t,
                    "r_t": int(pm["r_t"]),
                    "trF": float(pm["trF"]),
                }

    return {
        "graph": case.name,
        "n": case.n,
        "eps": eps,
        "m0": m,
        "horizon": horizon,
        "max_dbar0": float(best_val),
        "arg": best_arg,
    }


def summarize_three_way(rows: Sequence[Dict]) -> Dict:
    total_runs = len(rows)

    def iter_steps(traj_name: str):
        for rr in rows:
            for s in rr["trajectories"][traj_name]["steps"]:
                yield rr, s

    # Task 1/2: barrier sums vs partial-averages bound.
    barrier_viol = []
    barrier_above_avg_step = []
    worst_barrier_gap = None
    for rr in rows:
        m0 = rr["m0"]
        avg_ell = rr["avg_ell"]
        b = rr["trajectories"]["barrier_greedy"]
        for s in b["steps"]:
            t = int(s["t"])
            gap = float(s["sum_ell_minus_bound"])
            if worst_barrier_gap is None or gap > worst_barrier_gap["gap"]:
                worst_barrier_gap = {
                    "graph": rr["graph"],
                    "eps": rr["eps"],
                    "t": t,
                    "gap": gap,
                    "sum_ell": float(s["sum_ell_selected"]),
                    "bound": float(s["partial_averages_bound"]),
                }
            if gap > 1e-10:
                barrier_viol.append(
                    {
                        "graph": rr["graph"],
                        "eps": rr["eps"],
                        "t": t,
                        "sum_ell": float(s["sum_ell_selected"]),
                        "bound": float(s["partial_averages_bound"]),
                        "gap": gap,
                    }
                )

        # Approach 2 check: each selected ell <= average?
        for s in b["steps"]:
            if s.get("selected_ell") is None:
                continue
            if float(s["selected_ell"]) > avg_ell + 1e-10:
                barrier_above_avg_step.append(
                    {
                        "graph": rr["graph"],
                        "eps": rr["eps"],
                        "t": int(s["t"]),
                        "selected_ell": float(s["selected_ell"]),
                        "avg_ell": float(avg_ell),
                    }
                )

    # Task 3 summary.
    mod_fail = []
    mod_max_dbar0 = -np.inf
    mod_max_M_over_eps = -np.inf
    mod_worst_dbar0 = None
    mod_worst_M = None
    for rr in rows:
        eps = rr["eps"]
        mg = rr["trajectories"]["modified_greedy"]
        if not mg["completed"]:
            mod_fail.append(
                {
                    "graph": rr["graph"],
                    "eps": eps,
                    "selected_count": mg["selected_count"],
                    "horizon": mg["horizon"],
                    "fail_reason": mg.get("fail_reason"),
                }
            )

        for s in mg["steps"]:
            d0 = float(s["dbar0"])
            if d0 > mod_max_dbar0:
                mod_max_dbar0 = d0
                mod_worst_dbar0 = {"graph": rr["graph"], "eps": eps, "t": int(s["t"]), "dbar0": d0}

            M_norm = float(s.get("M_norm", 0.0))
            ratio = M_norm / eps if eps > 0 else 0.0
            if ratio > mod_max_M_over_eps:
                mod_max_M_over_eps = ratio
                mod_worst_M = {
                    "graph": rr["graph"],
                    "eps": eps,
                    "t": int(s["t"]),
                    "M_norm": M_norm,
                    "ratio": ratio,
                }

    # Task 4 summary.
    induction_rows = []
    for rr in rows:
        for x in rr["trajectories"]["barrier_greedy"].get("direct_induction", []):
            induction_rows.append({"graph": rr["graph"], "eps": rr["eps"], **x})

    cond_fail = [r for r in induction_rows if not r["condition_delta_lt_rhs"]]
    worst_delta_gap = max(induction_rows, key=lambda r: r["delta_minus_rhs"]) if induction_rows else None
    max_recur_err = max((r["recurrence_abs_err"] for r in induction_rows), default=0.0)

    # Three-way dbar0 comparisons at matching t.
    comp_rows = []
    for rr in rows:
        b = rr["trajectories"]["barrier_greedy"]["steps"]
        p = rr["trajectories"]["leverage_prefix"]["steps"]
        m = rr["trajectories"]["modified_greedy"]["steps"]
        # Map by t.
        bmap = {int(s["t"]): s for s in b}
        pmap = {int(s["t"]): s for s in p}
        mmap = {int(s["t"]): s for s in m}
        ts = sorted(set(bmap) & set(pmap) & set(mmap))
        for t in ts:
            comp_rows.append(
                {
                    "graph": rr["graph"],
                    "eps": rr["eps"],
                    "t": t,
                    "dbar0_barrier": float(bmap[t]["dbar0"]),
                    "dbar0_prefix": float(pmap[t]["dbar0"]),
                    "dbar0_modified": float(mmap[t]["dbar0"]),
                }
            )

    return {
        "num_runs": total_runs,
        "task1": {
            "barrier_sum_ell_le_partial_bound_all": len(barrier_viol) == 0,
            "barrier_sum_ell_violations": barrier_viol[:30],
            "barrier_sum_ell_violation_count": len(barrier_viol),
            "worst_barrier_sum_ell_gap": worst_barrier_gap,
            "modified_reaches_horizon_all": len(mod_fail) == 0,
            "modified_failures": mod_fail[:30],
            "modified_failure_count": len(mod_fail),
            "dbar0_comparison_rows": comp_rows[:600],
            "dbar0_comparison_count": len(comp_rows),
        },
        "task2": {
            "approach2_selected_ell_le_avg_all": len(barrier_above_avg_step) == 0,
            "approach2_counterexamples": barrier_above_avg_step[:30],
            "approach2_counterexample_count": len(barrier_above_avg_step),
            "approach3_empirical_domination_note": "Checked via sum_ell vs partial-average bound and per-step selected ell vs average; full stochastic-domination not established by numerics alone.",
        },
        "task3": {
            "modified_reaches_horizon_all": len(mod_fail) == 0,
            "modified_failure_count": len(mod_fail),
            "max_dbar0_modified": float(mod_max_dbar0) if mod_max_dbar0 > -np.inf else float("nan"),
            "worst_dbar0_modified": mod_worst_dbar0,
            "max_Mnorm_over_eps_modified": float(mod_max_M_over_eps) if mod_max_M_over_eps > -np.inf else float("nan"),
            "worst_Mnorm_over_eps_modified": mod_worst_M,
        },
        "task4": {
            "num_induction_rows": len(induction_rows),
            "condition_delta_lt_rhs_all": len(cond_fail) == 0,
            "condition_delta_lt_rhs_failure_count": len(cond_fail),
            "condition_delta_lt_rhs_failures": cond_fail[:30],
            "worst_delta_minus_rhs": worst_delta_gap,
            "max_recurrence_abs_err": float(max_recur_err),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260213)
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("data/first-proof/problem6-codex-cycle6-results.json"),
    )
    args = ap.parse_args()

    eps_list = [0.1, 0.2, 0.3, 0.5]

    suite_raw = build_cycle6_suite(args.seed)
    suite = [prepare_case(name, n, edges) for (name, n, edges) in suite_raw]

    rows = []
    for case in suite:
        for eps in eps_list:
            rr = run_three_way(case, eps, heavy_ge=False)
            rows.append(rr)

    summary = summarize_three_way(rows)

    # Task 5 strict-light exhaustive small-n scan.
    small_raw = build_small_exhaustive_suite(args.seed + 17)
    small_cases = [prepare_case(name, n, edges) for (name, n, edges) in small_raw]

    strict_rows = []
    for case in small_cases:
        for eps in [0.2, 0.3, 0.5]:
            strict_rows.append(exhaustive_subset_scan(case, eps, heavy_ge=True))

    strict_count_ge_1 = sum(1 for r in strict_rows if r["max_dbar0"] >= 1.0 - 1e-12)
    strict_max = max(strict_rows, key=lambda r: r["max_dbar0"]) if strict_rows else None
    k10_case = next((r for r in strict_rows if r["graph"] == "K_10" and abs(r["eps"] - 0.2) < 1e-12), None)

    task5 = {
        "strict_threshold_definition": "heavy iff tau_e >= eps; light iff tau_e < eps",
        "num_rows": len(strict_rows),
        "count_max_dbar0_ge_1": int(strict_count_ge_1),
        "max_row": strict_max,
        "k10_eps_0_2": k10_case,
        "rows": strict_rows,
    }

    out = {
        "meta": {
            "date": "2026-02-13",
            "agent": "Codex",
            "seed": args.seed,
            "eps_list": eps_list,
            "suite_size": len(suite),
            "strict_exhaustive_suite_size": len(small_cases),
            "base_script": "scripts/verify-p6-cycle5-codex.py",
        },
        "summary": {
            **summary,
            "task5": task5,
        },
        "runs": rows,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=to_jsonable)

    print("=" * 90)
    print("P6 CYCLE 6 CODEX")
    print("=" * 90)
    print(f"runs={len(rows)} suite={len(suite)}")
    print(
        "task1: barrier sum_ell<=bound all:",
        out["summary"]["task1"]["barrier_sum_ell_le_partial_bound_all"],
        " violations=",
        out["summary"]["task1"]["barrier_sum_ell_violation_count"],
    )
    print(
        "task1/3: modified reaches horizon all:",
        out["summary"]["task1"]["modified_reaches_horizon_all"],
        " failures=",
        out["summary"]["task1"]["modified_failure_count"],
    )
    print(
        "task3: max dbar0 modified=",
        out["summary"]["task3"]["max_dbar0_modified"],
        " max(Mnorm/eps)=",
        out["summary"]["task3"]["max_Mnorm_over_eps_modified"],
    )
    print(
        "task4: direct induction delta<rhs all:",
        out["summary"]["task4"]["condition_delta_lt_rhs_all"],
        " failures=",
        out["summary"]["task4"]["condition_delta_lt_rhs_failure_count"],
        " max recurrence err=",
        out["summary"]["task4"]["max_recurrence_abs_err"],
    )
    print(
        "task5: strict exhaustive count(max_dbar0>=1)=",
        out["summary"]["task5"]["count_max_dbar0_ge_1"],
    )
    print("wrote", args.out_json)


if __name__ == "__main__":
    main()
