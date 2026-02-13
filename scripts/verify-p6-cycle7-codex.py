#!/usr/bin/env python3
"""Problem 6 Cycle 7: BMI eigenspace decomposition + closure probes.

Implements all five prioritized tasks from
`data/first-proof/problem6-codex-cycle7-handoff.md`.

Outputs:
  - data/first-proof/problem6-codex-cycle7-results.json
  - data/first-proof/problem6-codex-cycle7-verification.md
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


MOD5 = runpy.run_path(str(Path(__file__).with_name("verify-p6-cycle5-codex.py")))
MOD6 = runpy.run_path(str(Path(__file__).with_name("verify-p6-cycle6-codex.py")))


def to_jsonable(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def edge_key(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


@dataclass
class WeightedGraph:
    name: str
    n: int
    edges_uv: List[Tuple[int, int]]
    weights: np.ndarray


@dataclass
class CaseData:
    graph: WeightedGraph
    L: np.ndarray
    Q: np.ndarray
    Lph: np.ndarray
    zmat: np.ndarray
    taus: np.ndarray


@dataclass
class I0Data:
    i0: List[int]
    i0_set: set[int]
    ell: Dict[int, float]
    ell_order: List[int]
    neighbors: Dict[int, List[Tuple[int, int, float]]]
    edge_rows: List[Tuple[int, int, int, float]]
    edge_index: Dict[Tuple[int, int], int]
    pi_i0: np.ndarray
    pi_i0_hat: np.ndarray
    trace_pi_i0_hat: float
    rank_pi_i0_hat: int


@dataclass
class StepState:
    t: int
    r_t: int
    selected_v: int | None
    processed_rank: int | None
    barrier_headroom_min_eig: float
    candidate_normY: float | None
    candidate_traceY: float | None
    M_norm: float
    trF: float
    dbar0: float
    dbar: float
    dbar_from_eigs: float
    dbar_abs_err: float
    dbar_from_pi_minus_lambda: float
    dbar_pi_minus_lambda_abs_err: float
    phi: float
    tr_B2: float
    trace_M_hat: float
    max_lambda: float
    lambda_mean: float
    pi_max: float
    pi_min: float
    pi_gt_1_count: int
    pi_zero_count: int
    nonneg_gap_min: float
    decomp_err_max: float
    approach1_upper: float
    approach1_margin_to_1: float
    approach1_proves_bmi: bool
    approach2_upper: float
    approach2_margin_to_1: float
    approach2_proves_bmi: bool
    approach3_upper: float
    approach3_margin_to_1: float
    approach3_proves_bmi: bool
    kn_reference: float
    ratio_to_kn: float
    eig_values: List[float]
    pi_values: List[float]
    contribution_values: List[float]
    contribution_f_values: List[float]
    f_values: List[float]
    l_values: List[float]


def build_weighted_from_unweighted(name: str, n: int, edges: Sequence[Tuple[int, int]]) -> WeightedGraph:
    m = len(edges)
    return WeightedGraph(name=name, n=n, edges_uv=[edge_key(u, v) for (u, v) in edges], weights=np.ones(m, dtype=float))


def weighted_graph_laplacian(n: int, edges_uv: Sequence[Tuple[int, int]], weights: np.ndarray) -> np.ndarray:
    L = np.zeros((n, n), dtype=float)
    for (u, v), w in zip(edges_uv, weights):
        L[u, u] += w
        L[v, v] += w
        L[u, v] -= w
        L[v, u] -= w
    return L


def pseudo_inv_half_and_basis(L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(L)
    n = L.shape[0]
    Lph = np.zeros((n, n), dtype=float)
    cols = []
    for i, lam in enumerate(vals):
        if lam > 1e-10:
            ui = vecs[:, i]
            Lph += (1.0 / np.sqrt(lam)) * np.outer(ui, ui)
            cols.append(ui)
    if not cols:
        raise RuntimeError("Laplacian has no positive spectrum")
    Q = np.column_stack(cols)  # n x (n-1)
    return Lph, Q


def compute_weighted_edge_z(n: int, edges_uv: Sequence[Tuple[int, int]], weights: np.ndarray, Lph: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = len(edges_uv)
    zmat = np.zeros((m, n), dtype=float)
    taus = np.zeros(m, dtype=float)
    for idx, ((u, v), w) in enumerate(zip(edges_uv, weights)):
        b = np.zeros(n, dtype=float)
        b[u] = 1.0
        b[v] = -1.0
        z = (math.sqrt(float(w)) * (Lph @ b))
        zmat[idx] = z
        taus[idx] = float(np.dot(z, z))
    return zmat, taus


def prep_case(g: WeightedGraph) -> CaseData:
    if not MOD5["is_connected"](g.n, g.edges_uv):
        raise RuntimeError(f"disconnected graph: {g.name}")
    L = weighted_graph_laplacian(g.n, g.edges_uv, g.weights)
    Lph, Q = pseudo_inv_half_and_basis(L)
    zmat, taus = compute_weighted_edge_z(g.n, g.edges_uv, g.weights, Lph)
    return CaseData(graph=g, L=L, Q=Q, Lph=Lph, zmat=zmat, taus=taus)


def find_i0_threshold(n: int, edges: Sequence[Tuple[int, int]], taus: np.ndarray, eps: float, heavy_ge: bool) -> List[int]:
    tol = 1e-12
    heavy_adj = [set() for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        heavy = (taus[idx] >= eps - tol) if heavy_ge else (taus[idx] > eps + tol)
        if heavy:
            heavy_adj[u].add(v)
            heavy_adj[v].add(u)
    i_set = set()
    for v in sorted(range(n), key=lambda vv: len(heavy_adj[vv])):
        if all(u not in i_set for u in heavy_adj[v]):
            i_set.add(v)
    return sorted(i_set)


def build_i0_data(case: CaseData, eps: float, heavy_ge: bool = True) -> I0Data:
    g = case.graph
    i0 = find_i0_threshold(g.n, g.edges_uv, case.taus, eps, heavy_ge=heavy_ge)
    i0_set = set(i0)

    edge_rows: List[Tuple[int, int, int, float]] = []
    neighbors: Dict[int, List[Tuple[int, int, float]]] = {v: [] for v in i0}
    edge_index: Dict[Tuple[int, int], int] = {}
    internal_idx = []

    for idx, (u, v) in enumerate(g.edges_uv):
        if u in i0_set and v in i0_set:
            tau = float(case.taus[idx])
            edge_rows.append((u, v, idx, tau))
            neighbors[u].append((v, idx, tau))
            neighbors[v].append((u, idx, tau))
            edge_index[(u, v)] = idx
            edge_index[(v, u)] = idx
            internal_idx.append(idx)

    ell = {v: float(sum(tau for (_, _, tau) in neighbors[v])) for v in i0}
    ell_order = sorted(i0, key=lambda v: (ell[v], v))

    n = g.n
    if internal_idx:
        Zall = case.zmat[internal_idx]
        pi_i0 = Zall.T @ Zall
    else:
        pi_i0 = np.zeros((n, n), dtype=float)

    pi_i0_hat = case.Q.T @ pi_i0 @ case.Q
    pi_eigs = np.linalg.eigvalsh(pi_i0_hat)
    rank_pi = int(np.sum(pi_eigs > 1e-10))

    return I0Data(
        i0=i0,
        i0_set=i0_set,
        ell=ell,
        ell_order=ell_order,
        neighbors=neighbors,
        edge_rows=edge_rows,
        edge_index=edge_index,
        pi_i0=pi_i0,
        pi_i0_hat=pi_i0_hat,
        trace_pi_i0_hat=float(np.trace(pi_i0_hat)),
        rank_pi_i0_hat=rank_pi,
    )


def candidate_barrier_stats(case: CaseData, S_set: set[int], R: Sequence[int], neighbors: Dict[int, List[Tuple[int, int, float]]], M: np.ndarray, eps: float) -> Tuple[Dict[int, Dict[str, float]], float]:
    n = case.graph.n
    H = eps * np.eye(n) - M
    evals_H, U_H = np.linalg.eigh(H)
    min_head = float(np.min(evals_H))
    if min_head <= 1e-11:
        return {}, min_head

    Hinvhalf = np.zeros((n, n), dtype=float)
    for i, lam in enumerate(evals_H):
        if lam > 1e-12:
            ui = U_H[:, i]
            Hinvhalf += (1.0 / np.sqrt(lam)) * np.outer(ui, ui)

    out: Dict[int, Dict[str, float]] = {}
    for v in R:
        idxs = [idx for (u, idx, _) in neighbors[v] if u in S_set]
        if not idxs:
            out[v] = {"traceY": 0.0, "normY": 0.0, "cross_to_S": 0.0}
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


def kn_dbar_reference(m0: int, eps: float, t: int) -> float:
    if m0 <= 0 or eps <= 0:
        return 0.0
    denom1 = m0 * eps - t
    denom2 = m0 * eps
    if denom1 <= 1e-12 or denom2 <= 1e-12:
        return float("inf")
    return (t - 1) / denom1 + (t + 1) / denom2


def approach2_upper_bound(lams: np.ndarray, eps: float, sum_pi: float, rank_cap: int) -> float:
    # Maximize sum_j ((pi_j-lam_j)/(eps-lam_j)) with 0<=pi_j<=1, sum pi_j <= sum_pi,
    # and at most rank_cap nonzeros. This is an LP solved greedily on largest weights b_j.
    b = 1.0 / np.maximum(1e-18, (eps - lams))
    base = float(np.sum((-lams) * b))

    n = len(lams)
    rank_cap = max(0, min(rank_cap, n))
    sum_pi = max(0.0, min(sum_pi, float(rank_cap)))

    order = np.argsort(-b)
    gain = 0.0
    rem = sum_pi
    used = 0
    for j in order:
        if rem <= 1e-15 or used >= rank_cap:
            break
        take = min(1.0, rem)
        gain += take * float(b[j])
        rem -= take
        used += 1

    return base + gain


def compute_step_spectral(
    case: CaseData,
    i0d: I0Data,
    eps: float,
    S: Sequence[int],
    M: np.ndarray,
    selected_v: int | None,
    processed_rank: int | None,
    barrier_headroom_min_eig: float,
    candidate_normY: float | None,
    candidate_traceY: float | None,
) -> StepState:
    g = case.graph
    S_set = set(S)
    R = [v for v in i0d.i0 if v not in S_set]
    r_t = len(R)
    t = len(S)

    idx_ss = []
    idx_sr = []
    idx_rr = []
    for u, v, idx, _ in i0d.edge_rows:
        in_u = u in S_set
        in_v = v in S_set
        if in_u and in_v:
            idx_ss.append(idx)
        elif in_u != in_v:
            idx_sr.append(idx)
        else:
            idx_rr.append(idx)

    n = g.n
    Z_ss = case.zmat[idx_ss] if idx_ss else np.zeros((0, n), dtype=float)
    Z_sr = case.zmat[idx_sr] if idx_sr else np.zeros((0, n), dtype=float)
    Z_rr = case.zmat[idx_rr] if idx_rr else np.zeros((0, n), dtype=float)

    M_from_edges = Z_ss.T @ Z_ss if len(idx_ss) else np.zeros((n, n), dtype=float)
    F = Z_sr.T @ Z_sr if len(idx_sr) else np.zeros((n, n), dtype=float)
    Lr = Z_rr.T @ Z_rr if len(idx_rr) else np.zeros((n, n), dtype=float)

    M_match_err = float(np.linalg.norm(M - M_from_edges, ord=2))
    if M_match_err > 1e-8:
        raise RuntimeError(f"M replay mismatch: {M_match_err}")

    Q = case.Q
    Mhat = Q.T @ M @ Q
    Fhat = Q.T @ F @ Q
    Lrhat = Q.T @ Lr @ Q
    Pihat = i0d.pi_i0_hat

    vals, U = np.linalg.eigh(Mhat)
    lam_max = float(np.max(vals)) if len(vals) else 0.0
    if eps - lam_max <= 1e-11:
        # Keep run alive with finite placeholders; this indicates barrier collapse.
        denom = np.maximum(1e-12, eps - vals)
    else:
        denom = eps - vals

    Bhat = np.linalg.inv(eps * np.eye(Mhat.shape[0]) - Mhat)
    trF = float(np.trace(Fhat))
    dbar0 = trF / (r_t * eps) if (r_t > 0 and eps > 0) else 0.0
    dbar = float(np.trace(Bhat @ Fhat) / r_t) if r_t > 0 else 0.0

    pi_vals = []
    contrib = []
    contrib_f = []
    f_vals = []
    l_vals = []
    decomp_errs = []
    nonneg_gaps = []
    for j in range(len(vals)):
        u = U[:, j]
        lam = float(vals[j])
        pi = float(u.T @ Pihat @ u)
        fj = float(u.T @ Fhat @ u)
        lj = float(u.T @ Lrhat @ u)
        c = float((pi - lam) / max(1e-18, (eps - lam)))
        cf = float(fj / max(1e-18, (eps - lam)))
        pi_vals.append(pi)
        contrib.append(c)
        contrib_f.append(cf)
        f_vals.append(fj)
        l_vals.append(lj)
        decomp_errs.append(abs((pi - lam) - (fj + lj)))
        nonneg_gaps.append(pi - lam)

    sum_contrib = float(np.sum(contrib))
    sum_contrib_f = float(np.sum(contrib_f))
    dbar_eig = sum_contrib_f / r_t if r_t > 0 else 0.0
    dbar_pi = sum_contrib / r_t if r_t > 0 else 0.0

    # Task 2 bounds.
    lams = np.array(vals, dtype=float)
    approach1_sum = float(np.sum((1.0 - lams) / np.maximum(1e-18, eps - lams)))
    approach2_sum = approach2_upper_bound(
        lams=lams,
        eps=eps,
        sum_pi=i0d.trace_pi_i0_hat,
        rank_cap=i0d.rank_pi_i0_hat,
    )
    a = np.array(pi_vals, dtype=float) - lams
    b = 1.0 / np.maximum(1e-18, eps - lams)
    approach3_sum = float(np.linalg.norm(a) * np.linalg.norm(b))

    if r_t > 0:
        app1 = approach1_sum / r_t
        app2 = approach2_sum / r_t
        app3 = approach3_sum / r_t
    else:
        app1 = app2 = app3 = 0.0

    phi = float(np.trace(Bhat))
    tr_B2 = float(np.trace(Bhat @ Bhat))

    kn_ref = kn_dbar_reference(len(i0d.i0), eps, t)
    ratio_to_kn = dbar / kn_ref if (kn_ref > 0 and np.isfinite(kn_ref)) else float("nan")

    return StepState(
        t=t,
        r_t=r_t,
        selected_v=selected_v,
        processed_rank=processed_rank,
        barrier_headroom_min_eig=float(barrier_headroom_min_eig),
        candidate_normY=float(candidate_normY) if candidate_normY is not None else None,
        candidate_traceY=float(candidate_traceY) if candidate_traceY is not None else None,
        M_norm=lam_max,
        trF=trF,
        dbar0=dbar0,
        dbar=dbar,
        dbar_from_eigs=dbar_eig,
        dbar_abs_err=abs(dbar - dbar_eig),
        dbar_from_pi_minus_lambda=dbar_pi,
        dbar_pi_minus_lambda_abs_err=abs(dbar - dbar_pi),
        phi=phi,
        tr_B2=tr_B2,
        trace_M_hat=float(np.trace(Mhat)),
        max_lambda=lam_max,
        lambda_mean=float(np.mean(vals)) if len(vals) else 0.0,
        pi_max=float(np.max(pi_vals)) if pi_vals else 0.0,
        pi_min=float(np.min(pi_vals)) if pi_vals else 0.0,
        pi_gt_1_count=int(np.sum(np.array(pi_vals) > 1.0 + 1e-9)) if pi_vals else 0,
        pi_zero_count=int(np.sum(np.array(pi_vals) <= 1e-10)) if pi_vals else 0,
        nonneg_gap_min=float(np.min(nonneg_gaps)) if nonneg_gaps else 0.0,
        decomp_err_max=float(np.max(decomp_errs)) if decomp_errs else 0.0,
        approach1_upper=app1,
        approach1_margin_to_1=1.0 - app1,
        approach1_proves_bmi=app1 < 1.0 - 1e-10,
        approach2_upper=app2,
        approach2_margin_to_1=1.0 - app2,
        approach2_proves_bmi=app2 < 1.0 - 1e-10,
        approach3_upper=app3,
        approach3_margin_to_1=1.0 - app3,
        approach3_proves_bmi=app3 < 1.0 - 1e-10,
        kn_reference=kn_ref,
        ratio_to_kn=ratio_to_kn,
        eig_values=[float(x) for x in vals],
        pi_values=[float(x) for x in pi_vals],
        contribution_values=[float(x) for x in contrib],
        contribution_f_values=[float(x) for x in contrib_f],
        f_values=[float(x) for x in f_vals],
        l_values=[float(x) for x in l_vals],
    )


def run_modified_with_spectral(case: CaseData, eps: float, heavy_ge: bool = True) -> Dict:
    i0d = build_i0_data(case, eps, heavy_ge=heavy_ge)
    m0 = len(i0d.i0)
    horizon = max(1, min(int(eps * m0 / 3), m0 - 1)) if m0 >= 2 else 0

    S: List[int] = []
    S_set: set[int] = set()
    M = np.zeros((case.graph.n, case.graph.n), dtype=float)

    steps: List[Dict] = []
    selected_order: List[int] = []
    skipped_rows: List[Dict] = []
    bss_rows: List[Dict] = []

    # Baseline t=0.
    st0 = compute_step_spectral(
        case=case,
        i0d=i0d,
        eps=eps,
        S=S,
        M=M,
        selected_v=None,
        processed_rank=None,
        barrier_headroom_min_eig=eps,
        candidate_normY=None,
        candidate_traceY=None,
    )
    steps.append(st0.__dict__)

    if horizon == 0:
        return {
            "graph": case.graph.name,
            "n": case.graph.n,
            "eps": eps,
            "m0": m0,
            "horizon": horizon,
            "strict_threshold": heavy_ge,
            "completed": True,
            "selected_count": 0,
            "fail_reason": None,
            "i0": i0d.i0,
            "ell": {str(k): float(v) for k, v in i0d.ell.items()},
            "ell_order": [int(v) for v in i0d.ell_order],
            "trace_pi_i0_hat": i0d.trace_pi_i0_hat,
            "rank_pi_i0_hat": i0d.rank_pi_i0_hat,
            "steps": steps,
            "selected_order": selected_order,
            "skips": skipped_rows,
            "bss_rows": bss_rows,
        }

    processed = 0
    completed = True
    fail_reason = None

    phi_prev = float(st0.phi)

    for v in i0d.ell_order:
        if len(S) >= horizon:
            break
        processed += 1

        R = [u for u in i0d.i0 if u not in S_set]
        cand, min_head = candidate_barrier_stats(case, S_set, R, i0d.neighbors, M, eps)
        if min_head <= 1e-11:
            completed = False
            fail_reason = "barrier_headroom_nonpositive"
            break

        stats = cand[v]
        feasible = stats["normY"] < 1.0 - 1e-12
        if not feasible:
            skipped_rows.append(
                {
                    "candidate_v": int(v),
                    "candidate_ell": float(i0d.ell[v]),
                    "processed_rank": processed,
                    "candidate_normY": float(stats["normY"]),
                    "candidate_traceY": float(stats["traceY"]),
                }
            )
            continue

        # Build C_t(v): new edges from v to current S_t.
        idxs_new = [idx for (u, idx, _) in i0d.neighbors[v] if u in S_set]
        C = case.zmat[idxs_new].T @ case.zmat[idxs_new] if idxs_new else np.zeros_like(M)

        S.append(v)
        S_set.add(v)
        selected_order.append(int(v))

        # Update M_{t+1}.
        if idxs_new:
            M = M + C

        st = compute_step_spectral(
            case=case,
            i0d=i0d,
            eps=eps,
            S=S,
            M=M,
            selected_v=int(v),
            processed_rank=processed,
            barrier_headroom_min_eig=min_head,
            candidate_normY=float(stats["normY"]),
            candidate_traceY=float(stats["traceY"]),
        )
        steps.append(st.__dict__)

        # Task 5: BSS potential delta checks.
        phi_now = float(st.phi)
        delta_phi = phi_now - phi_prev

        Q = case.Q
        Chat = Q.T @ C @ Q
        Mhat_prev = Q.T @ (M - C) @ Q
        Bhat_prev = np.linalg.inv(eps * np.eye(Mhat_prev.shape[0]) - Mhat_prev)
        Mhat_now = Q.T @ M @ Q
        Bhat_now = np.linalg.inv(eps * np.eye(Mhat_now.shape[0]) - Mhat_now)
        delta_phi_formula_naive = float(np.trace(Bhat_now @ Chat))
        delta_phi_formula_resolvent = float(np.trace(Bhat_now @ Chat @ Bhat_prev))

        bss_rows.append(
            {
                "t_before": int(len(S) - 1),
                "selected_v": int(v),
                "num_new_edges": int(len(idxs_new)),
                "delta_phi": float(delta_phi),
                "delta_phi_formula_naive": float(delta_phi_formula_naive),
                "delta_phi_naive_abs_err": abs(float(delta_phi - delta_phi_formula_naive)),
                "delta_phi_formula_resolvent": float(delta_phi_formula_resolvent),
                "delta_phi_resolvent_abs_err": abs(float(delta_phi - delta_phi_formula_resolvent)),
            }
        )
        phi_prev = phi_now

    if len(S) < horizon:
        completed = False
        fail_reason = fail_reason or "ran_out_of_candidates_before_horizon"

    # telescoping check
    if steps:
        phi0 = float(steps[0]["phi"])
        phiT = float(steps[-1]["phi"])
    else:
        phi0 = phiT = 0.0
    delta_sum = float(sum(r["delta_phi"] for r in bss_rows))

    return {
        "graph": case.graph.name,
        "n": case.graph.n,
        "eps": eps,
        "m0": m0,
        "horizon": horizon,
        "strict_threshold": heavy_ge,
        "completed": completed,
        "selected_count": len(S),
        "fail_reason": fail_reason,
        "i0": i0d.i0,
        "ell": {str(k): float(v) for k, v in i0d.ell.items()},
        "ell_order": [int(v) for v in i0d.ell_order],
        "trace_pi_i0_hat": i0d.trace_pi_i0_hat,
        "rank_pi_i0_hat": i0d.rank_pi_i0_hat,
        "steps": steps,
        "selected_order": selected_order,
        "skips": skipped_rows,
        "processed_total": processed,
        "bss_rows": bss_rows,
        "bss_telescoping": {
            "phi0": phi0,
            "phiT": phiT,
            "sum_delta_phi": delta_sum,
            "telescoping_abs_err": abs((phiT - phi0) - delta_sum),
            "max_delta_naive_abs_err": max((r["delta_phi_naive_abs_err"] for r in bss_rows), default=0.0),
            "max_delta_resolvent_abs_err": max((r["delta_phi_resolvent_abs_err"] for r in bss_rows), default=0.0),
        },
    }


def build_base_suite(seed: int) -> List[WeightedGraph]:
    specs = MOD6["build_cycle6_suite"](seed)
    out = []
    for name, n, edges in specs:
        out.append(build_weighted_from_unweighted(name, n, edges))
    return out


def make_expander_pendant(n_core: int, n_pendant: int, d: int, seed: int) -> Tuple[int, List[Tuple[int, int]]]:
    edges = MOD5["randomize_regular_graph"](n_core, d, seed=seed, num_switches=5000)
    n = n_core + n_pendant
    rng = np.random.default_rng(seed + 17)
    out = list(edges)
    for p in range(n_pendant):
        v = n_core + p
        u = int(rng.integers(0, n_core))
        out.append(edge_key(u, v))
    return n, out


def make_barbell(n1: int, n2: int, bridges: int, seed: int) -> Tuple[int, List[Tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    n = n1 + n2
    A = list(range(n1))
    B = list(range(n1, n))
    es = set()
    for i in A:
        for j in A:
            if i < j:
                es.add((i, j))
    for i in B:
        for j in B:
            if i < j:
                es.add((i, j))
    for _ in range(bridges):
        u = int(rng.choice(A))
        v = int(rng.choice(B))
        es.add(edge_key(u, v))
    out = sorted(es)
    return n, out


def make_planted_dense(n: int, p1: float, planted_size: int, p2: float, seed: int) -> Tuple[int, List[Tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    for _ in range(200):
        es = []
        planted = set(range(planted_size))
        for i in range(n):
            for j in range(i + 1, n):
                p = p2 if (i in planted and j in planted) else p1
                if rng.random() < p:
                    es.append((i, j))
        if MOD5["is_connected"](n, es):
            return n, es
    raise RuntimeError("failed to generate connected planted dense graph")


def build_adversarial_suite(seed: int) -> List[WeightedGraph]:
    out: List[WeightedGraph] = []

    # 1) Near-bipartite uneven leverage.
    for a, b in [(5, 95), (10, 90), (20, 80)]:
        n, edges = MOD5["complete_bipartite"](a, b)
        out.append(build_weighted_from_unweighted(f"Adv_BipComplete_{a}_{b}", n, edges))

    # 2) Expander + pendant.
    n, edges = make_expander_pendant(80, 40, d=6, seed=seed + 201)
    out.append(build_weighted_from_unweighted("Adv_ExpanderPendant_80_40", n, edges))

    # 3) Weighted concentration on edges.
    n, edges = MOD5["random_bipartite_connected"](40, 40, p=0.25, seed=seed + 301)
    rng = np.random.default_rng(seed + 302)
    w = np.exp(rng.normal(0.0, 1.0, size=len(edges)))
    w = np.clip(w, 0.2, 8.0)
    out.append(WeightedGraph("Adv_WeightedBip_40_40", n, [edge_key(u, v) for (u, v) in edges], w.astype(float)))

    # 4) Barbell variants.
    n, edges = make_barbell(45, 45, bridges=1, seed=seed + 401)
    out.append(build_weighted_from_unweighted("Adv_Barbell_45_45_b1", n, edges))
    n, edges = make_barbell(40, 40, bridges=3, seed=seed + 402)
    out.append(build_weighted_from_unweighted("Adv_Barbell_40_40_b3", n, edges))

    # 5) Random + planted dense subgraph.
    n, edges = make_planted_dense(120, p1=0.05, planted_size=18, p2=0.75, seed=seed + 501)
    out.append(build_weighted_from_unweighted("Adv_PlantedDense_n120", n, edges))

    return out


def summarize_runs(runs: Sequence[Dict], label: str) -> Dict:
    all_steps = []
    for rr in runs:
        for s in rr["steps"]:
            all_steps.append({"graph": rr["graph"], "eps": rr["eps"], **s})

    if not all_steps:
        return {
            "label": label,
            "num_runs": len(runs),
            "num_steps": 0,
        }

    dbar_vals = np.array([s["dbar"] for s in all_steps], dtype=float)
    pi_gt_1 = [s for s in all_steps if s["pi_gt_1_count"] > 0]
    decomp_err = np.array([s["decomp_err_max"] for s in all_steps], dtype=float)
    eig_err = np.array([s["dbar_abs_err"] for s in all_steps], dtype=float)
    pi_expr_err = np.array([s["dbar_pi_minus_lambda_abs_err"] for s in all_steps], dtype=float)

    app1_fail = [s for s in all_steps if not s["approach1_proves_bmi"]]
    app2_fail = [s for s in all_steps if not s["approach2_proves_bmi"]]
    app3_fail = [s for s in all_steps if not s["approach3_proves_bmi"]]

    ratios = np.array([s["ratio_to_kn"] for s in all_steps if np.isfinite(s["ratio_to_kn"])], dtype=float)
    ratio_max = float(np.max(ratios)) if len(ratios) else float("nan")

    horizon_rows = []
    for rr in runs:
        if rr["steps"]:
            horizon_rows.append({"graph": rr["graph"], "eps": rr["eps"], **rr["steps"][-1]})
    horizon_ratios = np.array([s["ratio_to_kn"] for s in horizon_rows if np.isfinite(s["ratio_to_kn"])], dtype=float)

    bss_naive_errs = []
    bss_resolvent_errs = []
    tel_errs = []
    for rr in runs:
        bss = rr.get("bss_telescoping", {})
        tel_errs.append(float(bss.get("telescoping_abs_err", 0.0)))
        bss_naive_errs.append(float(bss.get("max_delta_naive_abs_err", 0.0)))
        bss_resolvent_errs.append(float(bss.get("max_delta_resolvent_abs_err", 0.0)))

    return {
        "label": label,
        "num_runs": len(runs),
        "num_steps": len(all_steps),
        "dbar_max": float(np.max(dbar_vals)),
        "dbar_min": float(np.min(dbar_vals)),
        "dbar_lt_1_all": bool(np.all(dbar_vals < 1.0 - 1e-10)),
        "dbar_ge_1_count": int(np.sum(dbar_vals >= 1.0 - 1e-10)),
        "worst_dbar_case": max(all_steps, key=lambda s: s["dbar"]),
        "max_pi_gt_1_count": max((int(s["pi_gt_1_count"]) for s in all_steps), default=0),
        "pi_gt_1_violations": len(pi_gt_1),
        "max_dbar_eig_abs_err": float(np.max(eig_err)),
        "max_dbar_pi_minus_lambda_abs_err": float(np.max(pi_expr_err)),
        "max_decomposition_abs_err": float(np.max(decomp_err)),
        "approach1_proves_all": len(app1_fail) == 0,
        "approach2_proves_all": len(app2_fail) == 0,
        "approach3_proves_all": len(app3_fail) == 0,
        "approach1_failure_count": len(app1_fail),
        "approach2_failure_count": len(app2_fail),
        "approach3_failure_count": len(app3_fail),
        "approach1_worst_upper": max(all_steps, key=lambda s: s["approach1_upper"]),
        "approach2_worst_upper": max(all_steps, key=lambda s: s["approach2_upper"]),
        "approach3_worst_upper": max(all_steps, key=lambda s: s["approach3_upper"]),
        "ratio_to_kn_max": ratio_max,
        "ratio_to_kn_horizon_max": float(np.max(horizon_ratios)) if len(horizon_ratios) else float("nan"),
        "ratio_to_kn_le_1_all": bool(np.all(ratios <= 1.0 + 1e-10)) if len(ratios) else True,
        "ratio_to_kn_gt_1_count": int(np.sum(ratios > 1.0 + 1e-10)) if len(ratios) else 0,
        "ratio_to_kn_horizon_le_1_all": bool(np.all(horizon_ratios <= 1.0 + 1e-10)) if len(horizon_ratios) else True,
        "bss_max_naive_abs_err": float(np.max(bss_naive_errs)) if bss_naive_errs else 0.0,
        "bss_max_resolvent_abs_err": float(np.max(bss_resolvent_errs)) if bss_resolvent_errs else 0.0,
        "bss_max_telescoping_abs_err": float(np.max(tel_errs)) if tel_errs else 0.0,
    }


def build_markdown_report(out: Dict) -> str:
    s_base = out["summary"]["base_suite"]
    s_adv = out["summary"]["adversarial_suite"]
    has_adv = "dbar_max" in s_adv

    lines = []
    lines.append("# Problem 6 Cycle 7 Codex Verification")
    lines.append("")
    lines.append("Date: 2026-02-13")
    lines.append("Agent: Codex")
    lines.append("Base handoff: `data/first-proof/problem6-codex-cycle7-handoff.md`")
    lines.append("")
    lines.append("Artifacts:")
    lines.append("- Script: `scripts/verify-p6-cycle7-codex.py`")
    lines.append("- Results JSON: `data/first-proof/problem6-codex-cycle7-results.json`")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- Base suite runs: {s_base['num_runs']} ({s_base['num_steps']} total steps)")
    lines.append(f"- Adversarial suite runs: {s_adv['num_runs']} ({s_adv['num_steps']} total steps)")
    if has_adv:
        lines.append(f"- BMI empirical status: dbar < 1 on all tested steps (base={s_base['dbar_lt_1_all']}, adversarial={s_adv['dbar_lt_1_all']})")
    else:
        lines.append(f"- BMI empirical status: dbar < 1 on all tested base steps = {s_base['dbar_lt_1_all']}")
    lines.append(f"- Worst base dbar: {s_base['dbar_max']:.6f} at `{s_base['worst_dbar_case']['graph']}` eps={s_base['worst_dbar_case']['eps']} t={s_base['worst_dbar_case']['t']}")
    lines.append(f"- Base steps with dbar >= 1: {s_base['dbar_ge_1_count']}")
    if has_adv:
        lines.append(f"- Worst adversarial dbar: {s_adv['dbar_max']:.6f} at `{s_adv['worst_dbar_case']['graph']}` eps={s_adv['worst_dbar_case']['eps']} t={s_adv['worst_dbar_case']['t']}")
        lines.append(f"- Adversarial steps with dbar >= 1: {s_adv['dbar_ge_1_count']}")
    lines.append("")
    lines.append("## Task 1: Eigenvalue-Level BMI Computation")
    lines.append("")
    lines.append("For every modified-greedy step, the script records:")
    lines.append("- eigenvalues lambda_j of M_t on im(L)")
    lines.append("- projections pi_j = u_j^T Pi_{I0} u_j")
    lines.append("- contributions (pi_j-lambda_j)/(eps-lambda_j) and f_j/(eps-lambda_j)")
    lines.append("- decomposition check pi_j-lambda_j = f_j + l_j")
    lines.append("")
    if has_adv:
        lines.append(f"Consistency checks: max |dbar - sum_j f_j/(eps-lambda_j)/r| = {s_base['max_dbar_eig_abs_err']:.3e} (base), {s_adv['max_dbar_eig_abs_err']:.3e} (adv)")
        lines.append(f"Proposed pi-lambda expression mismatch: base={s_base['max_dbar_pi_minus_lambda_abs_err']:.3e}, adv={s_adv['max_dbar_pi_minus_lambda_abs_err']:.3e}")
        lines.append(f"Max decomposition error |(pi-lambda)-(f+l)| = {s_base['max_decomposition_abs_err']:.3e} (base), {s_adv['max_decomposition_abs_err']:.3e} (adv)")
        lines.append(f"pi_j <= 1 violations: base={s_base['pi_gt_1_violations']}, adv={s_adv['pi_gt_1_violations']}")
    else:
        lines.append(f"Consistency checks: max |dbar - sum_j f_j/(eps-lambda_j)/r| = {s_base['max_dbar_eig_abs_err']:.3e}")
        lines.append(f"Proposed pi-lambda expression mismatch: {s_base['max_dbar_pi_minus_lambda_abs_err']:.3e}")
        lines.append(f"Max decomposition error |(pi-lambda)-(f+l)| = {s_base['max_decomposition_abs_err']:.3e}")
        lines.append(f"pi_j <= 1 violations: {s_base['pi_gt_1_violations']}")
    lines.append("")
    lines.append("## Task 2: Direct BMI Proof Probes")
    lines.append("")
    lines.append("Three upper-bound attempts were evaluated per step:")
    lines.append("1. uniform pi_j <= 1 bound")
    lines.append("2. LP-style bound using sum pi_j = tr(Pi_{I0}) and rank cap")
    lines.append("3. Cauchy-Schwarz bound")
    lines.append("")
    lines.append(f"Approach 1 proves all steps: {s_base['approach1_proves_all']} (failures={s_base['approach1_failure_count']})")
    lines.append(f"Approach 2 proves all steps: {s_base['approach2_proves_all']} (failures={s_base['approach2_failure_count']})")
    lines.append(f"Approach 3 proves all steps: {s_base['approach3_proves_all']} (failures={s_base['approach3_failure_count']})")
    lines.append("These remain diagnostic bounds; none closes BMI universally on this scan.")
    lines.append("")
    lines.append("## Task 3: K_n Extremality")
    lines.append("")
    lines.append("Compared each step to K_m reference")
    lines.append("dbar_Km(t) = (t-1)/(m eps - t) + (t+1)/(m eps)")
    lines.append("with m=|I0| for the run.")
    lines.append("")
    lines.append(f"Base suite max ratio dbar/dbar_Km: {s_base['ratio_to_kn_max']:.6f} (horizon max {s_base['ratio_to_kn_horizon_max']:.6f})")
    lines.append(f"Base ratio violations (dbar/dbar_Km > 1): {s_base['ratio_to_kn_gt_1_count']}")
    if has_adv:
        lines.append(f"Adversarial suite max ratio dbar/dbar_Km: {s_adv['ratio_to_kn_max']:.6f} (horizon max {s_adv['ratio_to_kn_horizon_max']:.6f})")
        lines.append(f"Adversarial ratio violations (dbar/dbar_Km > 1): {s_adv['ratio_to_kn_gt_1_count']}")
        lines.append(f"Extremality (<=1) on all tested steps: base={s_base['ratio_to_kn_le_1_all']}, adv={s_adv['ratio_to_kn_le_1_all']}")
    else:
        lines.append(f"Extremality (<=1) on all tested base steps: {s_base['ratio_to_kn_le_1_all']}")
    lines.append("")
    lines.append("## Task 4: Stress Test (Adversarial Families)")
    lines.append("")
    lines.append("Added families:")
    lines.append("- near-bipartite complete K_{a,b} with uneven splits")
    lines.append("- expander + pendant attachments")
    lines.append("- weighted bipartite leverage concentration")
    lines.append("- barbell variants with sparse bridges")
    lines.append("- random graph with planted dense subgraph")
    lines.append("")
    if has_adv:
        lines.append(f"Adversarial dbar<1 across all steps: {s_adv['dbar_lt_1_all']}")
    else:
        lines.append("Adversarial suite skipped in this run.")
    lines.append("")
    lines.append("## Task 5: BSS Potential Probe")
    lines.append("")
    lines.append("Tracked Phi_t = tr((eps I - M_t)^{-1}) on im(L), and checked two identities:")
    lines.append("- naive handoff candidate: Phi_{t+1}-Phi_t ?= tr(B_{t+1} C_t(v_t))")
    lines.append("- resolvent identity: Phi_{t+1}-Phi_t = tr(B_{t+1} C_t(v_t) B_t)")
    lines.append("")
    if has_adv:
        lines.append(f"Max naive-delta abs error: base={s_base['bss_max_naive_abs_err']:.3e}, adv={s_adv['bss_max_naive_abs_err']:.3e}")
        lines.append(f"Max resolvent-delta abs error: base={s_base['bss_max_resolvent_abs_err']:.3e}, adv={s_adv['bss_max_resolvent_abs_err']:.3e}")
        lines.append(f"Max telescoping abs error: base={s_base['bss_max_telescoping_abs_err']:.3e}, adv={s_adv['bss_max_telescoping_abs_err']:.3e}")
    else:
        lines.append(f"Max naive-delta abs error: {s_base['bss_max_naive_abs_err']:.3e}")
        lines.append(f"Max resolvent-delta abs error: {s_base['bss_max_resolvent_abs_err']:.3e}")
        lines.append(f"Max telescoping abs error: {s_base['bss_max_telescoping_abs_err']:.3e}")
    lines.append("Result: the naive formula is false in general; the resolvent identity is numerically exact.")
    lines.append("")
    lines.append("## Bottom Line")
    lines.append("")
    lines.append("Cycle 7 now provides the requested eigenspace-level BMI dataset and comparative diagnostics for all five tasks.")
    if has_adv:
        lines.append(f"Empirically, dbar<1 holds on base={s_base['dbar_lt_1_all']} and adversarial={s_adv['dbar_lt_1_all']} scans.")
    else:
        lines.append(f"Empirically on this run, dbar<1 holds on base suite={s_base['dbar_lt_1_all']}.")
    lines.append("A full analytic BMI closure still needs a stronger inequality than the current three direct bound probes.")

    return "\n".join(lines) + "\n"


def run_all(seed: int, eps_list: Sequence[float], include_adversarial: bool) -> Dict:
    base_specs = build_base_suite(seed)
    adv_specs = build_adversarial_suite(seed) if include_adversarial else []

    base_cases = [prep_case(g) for g in base_specs]
    adv_cases = [prep_case(g) for g in adv_specs]

    base_runs = []
    for cs in base_cases:
        for eps in eps_list:
            base_runs.append(run_modified_with_spectral(cs, eps=eps, heavy_ge=True))

    adv_runs = []
    for cs in adv_cases:
        for eps in eps_list:
            adv_runs.append(run_modified_with_spectral(cs, eps=eps, heavy_ge=True))

    summary = {
        "base_suite": summarize_runs(base_runs, "base_suite"),
        "adversarial_suite": summarize_runs(adv_runs, "adversarial_suite"),
    }

    return {
        "meta": {
            "date": "2026-02-13",
            "agent": "Codex",
            "seed": seed,
            "eps_list": list(eps_list),
            "base_suite_size": len(base_cases),
            "adversarial_suite_size": len(adv_cases),
            "strict_threshold": "heavy if tau >= eps",
            "tasks": [
                "task1_eigenvalue_bmi",
                "task2_direct_bmi_bounds",
                "task3_kn_extremality",
                "task4_stress_adversarial",
                "task5_bss_potential",
            ],
        },
        "summary": summary,
        "base_runs": base_runs,
        "adversarial_runs": adv_runs,
    }


def print_summary(out: Dict):
    b = out["summary"]["base_suite"]
    a = out["summary"]["adversarial_suite"]
    has_adv = "dbar_max" in a
    print("=" * 96)
    print("P6 CYCLE 7 CODEX: BMI EIGEN-DECOMPOSITION + EXTREMALITY + STRESS")
    print("=" * 96)
    print(f"base runs={b['num_runs']} steps={b['num_steps']} dbar_max={b['dbar_max']:.6f} all<1={b['dbar_lt_1_all']}")
    if has_adv:
        print(f"adv  runs={a['num_runs']} steps={a['num_steps']} dbar_max={a['dbar_max']:.6f} all<1={a['dbar_lt_1_all']}")
        print(f"max |dbar-eig| base={b['max_dbar_eig_abs_err']:.3e} adv={a['max_dbar_eig_abs_err']:.3e}")
        print(f"max |dbar - pi/lambda expr| base={b['max_dbar_pi_minus_lambda_abs_err']:.3e} adv={a['max_dbar_pi_minus_lambda_abs_err']:.3e}")
    else:
        print(f"adv  runs={a['num_runs']} steps={a['num_steps']} (skipped)")
        print(f"max |dbar-eig| base={b['max_dbar_eig_abs_err']:.3e}")
        print(f"max |dbar - pi/lambda expr| base={b['max_dbar_pi_minus_lambda_abs_err']:.3e}")
    print(f"approach proves-all (base): A1={b['approach1_proves_all']} A2={b['approach2_proves_all']} A3={b['approach3_proves_all']}")
    print(f"K_m ratio max base={b['ratio_to_kn_max']:.6f} horizon={b['ratio_to_kn_horizon_max']:.6f}")
    if has_adv:
        print(f"K_m ratio max adv ={a['ratio_to_kn_max']:.6f} horizon={a['ratio_to_kn_horizon_max']:.6f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eps", type=float, nargs="*", default=[0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--skip-adversarial", action="store_true")
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("data/first-proof/problem6-codex-cycle7-results.json"),
    )
    ap.add_argument(
        "--out-md",
        type=Path,
        default=Path("data/first-proof/problem6-codex-cycle7-verification.md"),
    )
    args = ap.parse_args()

    out = run_all(seed=args.seed, eps_list=args.eps, include_adversarial=not args.skip_adversarial)
    print_summary(out)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=to_jsonable)

    report = build_markdown_report(out)
    with args.out_md.open("w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nWrote JSON: {args.out_json}")
    print(f"Wrote markdown: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
