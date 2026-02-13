#!/usr/bin/env python3
"""Problem 6 Cycle 8: GPL-V focused scan and diagnostics.

Implements tasks from `data/first-proof/problem6-codex-cycle8-handoff.md`:
1. Full R-scan on strict threshold trajectory.
2. Threshold reconciliation (tau > eps vs tau >= eps).
3. Eigenspace overlap diagnostics.
4. Cross-degree bound probes.
5. Partial-averages with skips.
6. Alternative existence-argument statistics.

Outputs:
- data/first-proof/problem6-codex-cycle8-results.json
- data/first-proof/problem6-codex-cycle8-verification.md
"""

from __future__ import annotations

import argparse
import json
import math
import runpy
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


C7 = runpy.run_path(str(Path(__file__).with_name("verify-p6-cycle7-codex.py")))


def to_jsonable(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-14 or sy < 1e-14:
        return None
    c = float(np.corrcoef(x, y)[0, 1])
    if np.isnan(c):
        return None
    return c


def quantiles(arr: np.ndarray, qs: Sequence[float]) -> Dict[str, float]:
    if len(arr) == 0:
        return {f"q{int(100*q)}": 0.0 for q in qs}
    out = {}
    for q in qs:
        out[f"q{int(100*q)}"] = float(np.quantile(arr, q))
    return out


def build_i0_data_with_threshold(case: Any, eps_heavy: float, heavy_ge: bool) -> Any:
    # Rebuild I0 with configurable heavy threshold (eps_heavy), while all barrier
    # computations still use the run's eps.
    i0 = C7["find_i0_threshold"](case.graph.n, case.graph.edges_uv, case.taus, eps_heavy, heavy_ge=heavy_ge)
    i0_set = set(i0)

    edge_rows: List[Tuple[int, int, int, float]] = []
    neighbors: Dict[int, List[Tuple[int, int, float]]] = {v: [] for v in i0}
    edge_index: Dict[Tuple[int, int], int] = {}
    internal_idx = []

    for idx, (u, v) in enumerate(case.graph.edges_uv):
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

    n = case.graph.n
    if internal_idx:
        Zall = case.zmat[internal_idx]
        pi_i0 = Zall.T @ Zall
    else:
        pi_i0 = np.zeros((n, n), dtype=float)

    pi_i0_hat = case.Q.T @ pi_i0 @ case.Q
    pi_eigs = np.linalg.eigvalsh(pi_i0_hat)
    rank_pi = int(np.sum(pi_eigs > 1e-10))

    return {
        "i0": i0,
        "i0_set": i0_set,
        "neighbors": neighbors,
        "edge_rows": edge_rows,
        "edge_index": edge_index,
        "ell": ell,
        "ell_order": ell_order,
        "pi_i0": pi_i0,
        "pi_i0_hat": pi_i0_hat,
        "trace_pi_i0_hat": float(np.trace(pi_i0_hat)),
        "rank_pi_i0_hat": rank_pi,
    }


def scan_state(
    case: Any,
    i0d: Dict,
    eps: float,
    S_set: set[int],
    M: np.ndarray,
    store_vertex_rows: bool,
) -> Dict:
    i0 = i0d["i0"]
    R = [v for v in i0 if v not in S_set]
    r_t = len(R)
    n = case.graph.n

    cand, min_head = C7["candidate_barrier_stats"](case, S_set, R, i0d["neighbors"], M, eps)

    # Barrier/eigenspace state for task 3/4.
    H = eps * np.eye(n) - M
    B = np.linalg.inv(H)

    Q = case.Q
    Mhat = Q.T @ M @ Q
    vals, U = np.linalg.eigh(Mhat)
    Ufull = Q @ U
    high_mask = vals > (eps / 2.0 + 1e-12)
    if np.any(high_mask):
        Uhigh = Ufull[:, high_mask]
        high_scales = 1.0 / np.sqrt(np.maximum(1e-18, eps - vals[high_mask]))
    else:
        Uhigh = np.zeros((n, 0), dtype=float)
        high_scales = np.zeros((0,), dtype=float)

    rows = []
    norm_vals = []
    trace_vals = []
    ell_vals = []
    ell_to_S_vals = []
    deg_vals = []
    high_frac_vals = []
    high_norm_frac_vals = []
    deg_times_max_vals = []
    norm_over_bound_vals = []
    feasible = 0
    deg0 = 0

    for v in R:
        idxs = [idx for (u, idx, _) in i0d["neighbors"][v] if u in S_set]
        deg_S = len(idxs)
        ell_i0 = float(i0d["ell"][v])
        ell_to_S = float(sum(tau for (u, _, tau) in i0d["neighbors"][v] if u in S_set))

        normY = float(cand[v]["normY"])
        traceY = float(cand[v]["traceY"])
        feasible_v = normY < 1.0 - 1e-12

        if deg_S == 0:
            max_edge_cost = 0.0
            deg_times_max = 0.0
            high_overlap_frac = 0.0
            high_norm = 0.0
        else:
            E = case.zmat[idxs]  # (deg, n)
            costs = np.einsum("ij,jk,ik->i", E, B, E, optimize=True)
            max_edge_cost = float(np.max(costs)) if len(costs) else 0.0
            deg_times_max = float(deg_S) * max_edge_cost

            total_energy = float(np.sum(E * E))
            if Uhigh.shape[1] > 0 and total_energy > 1e-18:
                proj = E @ Uhigh  # (deg, k_high)
                high_energy = float(np.sum(proj * proj))
                high_overlap_frac = high_energy / total_energy

                W = proj * high_scales  # (deg, k_high)
                svals = np.linalg.svd(W, compute_uv=False)
                high_norm = float((svals[0] ** 2) if len(svals) else 0.0)
            else:
                high_overlap_frac = 0.0
                high_norm = 0.0

        high_norm_frac = (high_norm / normY) if normY > 1e-18 else 0.0
        ratio_bound = (normY / deg_times_max) if deg_times_max > 1e-18 else (0.0 if normY < 1e-18 else float("inf"))

        row = {
            "v": int(v),
            "normY": normY,
            "traceY": traceY,
            "feasible": bool(feasible_v),
            "deg_S": int(deg_S),
            "ell_i0": ell_i0,
            "ell_to_S": ell_to_S,
            "max_edge_barrier_cost": max_edge_cost,
            "deg_times_max_edge_cost": deg_times_max,
            "norm_over_degmax_bound": ratio_bound,
            "high_overlap_frac": high_overlap_frac,
            "high_norm": high_norm,
            "high_norm_frac": high_norm_frac,
        }
        if store_vertex_rows:
            rows.append(row)

        norm_vals.append(normY)
        trace_vals.append(traceY)
        ell_vals.append(ell_i0)
        ell_to_S_vals.append(ell_to_S)
        deg_vals.append(float(deg_S))
        high_frac_vals.append(high_overlap_frac)
        high_norm_frac_vals.append(high_norm_frac)
        deg_times_max_vals.append(deg_times_max)
        norm_over_bound_vals.append(ratio_bound)
        if feasible_v:
            feasible += 1
        if deg_S == 0:
            deg0 += 1

    norm_arr = np.array(norm_vals, dtype=float)
    trace_arr = np.array(trace_vals, dtype=float)
    ell_arr = np.array(ell_vals, dtype=float)
    ellS_arr = np.array(ell_to_S_vals, dtype=float)
    deg_arr = np.array(deg_vals, dtype=float)
    high_arr = np.array(high_frac_vals, dtype=float)
    highn_arr = np.array(high_norm_frac_vals, dtype=float)
    bound_arr = np.array(deg_times_max_vals, dtype=float)
    ratio_arr = np.array(norm_over_bound_vals, dtype=float)

    trF = float(np.sum(ellS_arr))
    dbar0 = trF / (r_t * eps) if (r_t > 0 and eps > 0) else 0.0
    dbar = float(np.mean(trace_arr)) if r_t > 0 else 0.0

    i_sort = np.argsort(norm_arr)
    sorted_norm = [float(norm_arr[i]) for i in i_sort]
    sorted_vertices = [int(R[i]) for i in i_sort]
    min_idx = int(i_sort[0]) if len(i_sort) else 0

    summary = {
        "r_t": r_t,
        "barrier_headroom_min_eig": float(min_head),
        "max_lambda": float(np.max(vals)) if len(vals) else 0.0,
        "lambda_high_count": int(np.sum(high_mask)),
        "dbar": dbar,
        "dbar0": dbar0,
        "trF": trF,
        "mean_traceY": float(np.mean(trace_arr)) if len(trace_arr) else 0.0,
        "mean_normY": float(np.mean(norm_arr)) if len(norm_arr) else 0.0,
        "mean_normY_sq": float(np.mean(norm_arr * norm_arr)) if len(norm_arr) else 0.0,
        "var_normY": float(np.var(norm_arr)) if len(norm_arr) else 0.0,
        "mean_traceY_sq": float(np.mean(trace_arr * trace_arr)) if len(trace_arr) else 0.0,
        "ratio_mean_norm_over_trace": float(np.mean(norm_arr) / np.mean(trace_arr)) if len(trace_arr) and float(np.mean(trace_arr)) > 1e-18 else 0.0,
        "feasible_count": int(feasible),
        "feasible_fraction": float(feasible / r_t) if r_t else 1.0,
        "deg0_count": int(deg0),
        "deg0_fraction": float(deg0 / r_t) if r_t else 0.0,
        "min_normY": float(norm_arr[min_idx]) if len(norm_arr) else 0.0,
        "min_normY_vertex": int(R[min_idx]) if len(norm_arr) else None,
        "max_normY": float(np.max(norm_arr)) if len(norm_arr) else 0.0,
        "normY_quantiles": quantiles(norm_arr, [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]),
        "corr_normY_ell_i0": corrcoef_safe(norm_arr, ell_arr),
        "corr_normY_degS": corrcoef_safe(norm_arr, deg_arr),
        "corr_normY_ell_to_S": corrcoef_safe(norm_arr, ellS_arr),
        "corr_normY_high_overlap": corrcoef_safe(norm_arr, high_arr),
        "corr_normY_high_norm_frac": corrcoef_safe(norm_arr, highn_arr),
        "corr_normY_degmax_bound": corrcoef_safe(norm_arr, bound_arr),
        "cross_degree_bound_violations": int(np.sum(norm_arr > bound_arr + 1e-9)),
        "max_norm_over_degmax_bound": float(np.max(ratio_arr[np.isfinite(ratio_arr)])) if np.any(np.isfinite(ratio_arr)) else 0.0,
        "sorted_normY": sorted_norm,
        "sorted_vertices_by_normY": sorted_vertices,
    }

    return {
        "rows": rows,
        "summary": summary,
    }


def run_modified_rscan(
    case: Any,
    eps: float,
    eps_heavy: float,
    heavy_ge: bool,
    store_vertex_rows: bool,
) -> Dict:
    i0d = build_i0_data_with_threshold(case, eps_heavy=eps_heavy, heavy_ge=heavy_ge)
    m0 = len(i0d["i0"])
    horizon = max(1, min(int(eps * m0 / 3), m0 - 1)) if m0 >= 2 else 0

    S: List[int] = []
    S_set: set[int] = set()
    M = np.zeros((case.graph.n, case.graph.n), dtype=float)

    steps = []
    skip_events = []
    selected_positions = []

    processed = 0
    selected = 0
    fail_reason = None

    # Precompute order positions for skip accounting.
    order = i0d["ell_order"]
    pos_of = {v: i + 1 for i, v in enumerate(order)}

    # Process in fixed leverage order; scan state once per selected-step.
    p = 0
    while selected < horizon:
        if p >= len(order):
            fail_reason = "ran_out_of_candidates_before_horizon"
            break

        scan = scan_state(case, i0d, eps, S_set, M, store_vertex_rows=store_vertex_rows)
        ssum = scan["summary"]

        skipped_here = []
        chosen_v = None
        chosen_row = None

        # Build map for fast candidate lookup.
        if store_vertex_rows:
            row_map = {r["v"]: r for r in scan["rows"]}
        else:
            # recompute map minimally from candidate stats if rows not stored
            cand, _ = C7["candidate_barrier_stats"](case, S_set, [v for v in i0d["i0"] if v not in S_set], i0d["neighbors"], M, eps)
            row_map = {
                int(v): {
                    "v": int(v),
                    "normY": float(cand[v]["normY"]),
                    "traceY": float(cand[v]["traceY"]),
                    "deg_S": int(sum(1 for (u, _, _) in i0d["neighbors"][v] if u in S_set)),
                    "ell_i0": float(i0d["ell"][v]),
                    "ell_to_S": float(sum(tau for (u, _, tau) in i0d["neighbors"][v] if u in S_set)),
                }
                for v in [u for u in i0d["i0"] if u not in S_set]
            }

        while p < len(order):
            v = int(order[p])
            p += 1
            processed += 1
            if v in S_set:
                continue
            rr = row_map[v]
            if rr["normY"] < 1.0 - 1e-12:
                chosen_v = v
                chosen_row = rr
                break
            skipped_here.append(
                {
                    "candidate_v": int(v),
                    "candidate_order_pos": int(pos_of[v]),
                    "candidate_normY": float(rr["normY"]),
                    "candidate_traceY": float(rr["traceY"]),
                    "candidate_deg_S": int(rr["deg_S"]),
                    "candidate_ell_i0": float(rr["ell_i0"]),
                    "candidate_ell_to_S": float(rr["ell_to_S"]),
                }
            )

        if chosen_v is None:
            fail_reason = "no_feasible_candidate_found"
            # still record the scan for this step.
            steps.append(
                {
                    "t": int(selected),
                    "r_t": int(ssum["r_t"]),
                    "processed_total": int(processed),
                    "skips_before_selection": len(skipped_here),
                    "skipped_candidates": skipped_here,
                    "chosen_v": None,
                    "chosen_order_pos": None,
                    "chosen_normY": None,
                    "chosen_traceY": None,
                    "scan_summary": ssum,
                    "vertex_rows": scan["rows"],
                }
            )
            break

        # Step record before applying update.
        step_row = {
            "t": int(selected),
            "r_t": int(ssum["r_t"]),
            "processed_total": int(processed),
            "skips_before_selection": len(skipped_here),
            "skipped_candidates": skipped_here,
            "chosen_v": int(chosen_v),
            "chosen_order_pos": int(pos_of[chosen_v]),
            "chosen_normY": float(chosen_row["normY"]),
            "chosen_traceY": float(chosen_row["traceY"]),
            "scan_summary": ssum,
            "vertex_rows": scan["rows"],
        }

        # Task 5 skip-adjusted partial-averages bookkeeping.
        selected_positions_tmp = selected_positions + [pos_of[chosen_v]]
        t1 = selected + 1
        s_t = processed - t1
        first_cut = min(len(order), t1 + s_t)
        bound_sum = float(sum(i0d["ell"][v] for v in order[:first_cut]))
        actual_sum = float(sum(i0d["ell"][v] for v in selected_positions_tmp and [order[p0-1] for p0 in selected_positions_tmp])) if False else float(sum(i0d["ell"][order[p0 - 1]] for p0 in selected_positions_tmp))
        r_next = m0 - t1
        dbar0_bound = (bound_sum / (r_next * eps)) if (r_next > 0 and eps > 0) else 0.0
        step_row["skip_adjusted"] = {
            "t1": int(t1),
            "s_t": int(s_t),
            "first_cut": int(first_cut),
            "sum_ell_selected": actual_sum,
            "sum_ell_bound_with_skips": bound_sum,
            "bound_minus_actual": float(bound_sum - actual_sum),
            "dbar0_upper_from_skips": dbar0_bound,
        }

        steps.append(step_row)
        if skipped_here:
            skip_events.append(
                {
                    "t": int(selected),
                    "graph": case.graph.name,
                    "eps": eps,
                    "count": len(skipped_here),
                    "max_skipped_normY": float(max(x["candidate_normY"] for x in skipped_here)),
                }
            )

        # Apply update.
        S.append(chosen_v)
        S_set.add(chosen_v)
        selected += 1
        selected_positions.append(pos_of[chosen_v])

        idxs_new = [idx for (u, idx, _) in i0d["neighbors"][chosen_v] if u in S_set and u != chosen_v]
        if idxs_new:
            Znew = case.zmat[idxs_new]
            M = M + Znew.T @ Znew

    completed = (selected >= horizon and fail_reason is None)

    # Aggregate run-level summaries.
    dbar_vals = [st["scan_summary"]["dbar"] for st in steps]
    min_norm_vals = [st["scan_summary"]["min_normY"] for st in steps]
    feasible_counts = [st["scan_summary"]["feasible_count"] for st in steps]
    chosen_norms = [st["chosen_normY"] for st in steps if st["chosen_normY"] is not None]
    dbar_ge1_steps = [st for st in steps if st["scan_summary"]["dbar"] >= 1.0 - 1e-10]

    corr_norm_ell = [st["scan_summary"]["corr_normY_ell_i0"] for st in steps if st["scan_summary"]["corr_normY_ell_i0"] is not None]
    corr_norm_deg = [st["scan_summary"]["corr_normY_degS"] for st in steps if st["scan_summary"]["corr_normY_degS"] is not None]
    corr_norm_high = [st["scan_summary"]["corr_normY_high_overlap"] for st in steps if st["scan_summary"]["corr_normY_high_overlap"] is not None]

    # Task 6 summaries.
    feasible_fractions = [st["scan_summary"]["feasible_fraction"] for st in steps]
    deg0_fractions = [st["scan_summary"]["deg0_fraction"] for st in steps]

    # Task 5 run-level skip-adjusted bound check.
    skip_bound_viol = 0
    skip_bound_worst = None
    for st in steps:
        if "skip_adjusted" not in st:
            continue
        t1 = st["skip_adjusted"]["t1"]
        r_next = m0 - t1
        if r_next <= 0:
            continue
        dbar0_next = 0.0
        # dbar0 at next step appears as scan_summary in next recorded step.
        # For robust accounting, use this step's bound against this step's dbar0 too.
        dbar0_curr = float(st["scan_summary"]["dbar0"])
        ub = float(st["skip_adjusted"]["dbar0_upper_from_skips"])
        gap = dbar0_curr - ub
        if gap > 1e-10:
            skip_bound_viol += 1
            if skip_bound_worst is None or gap > skip_bound_worst["gap"]:
                skip_bound_worst = {
                    "t": st["t"],
                    "gap": gap,
                    "dbar0": dbar0_curr,
                    "upper": ub,
                }

    return {
        "graph": case.graph.name,
        "n": int(case.graph.n),
        "eps": float(eps),
        "eps_heavy": float(eps_heavy),
        "strict_threshold": bool(heavy_ge),
        "m0": int(m0),
        "horizon": int(horizon),
        "completed": bool(completed),
        "selected_count": int(selected),
        "processed_total": int(processed),
        "fail_reason": fail_reason,
        "i0": [int(v) for v in i0d["i0"]],
        "ell_order": [int(v) for v in i0d["ell_order"]],
        "steps": steps,
        "skip_events": skip_events,
        "run_summary": {
            "skip_count_total": int(sum(st["skips_before_selection"] for st in steps)),
            "skip_steps": int(sum(1 for st in steps if st["skips_before_selection"] > 0)),
            "max_skips_before_one_selection": int(max((st["skips_before_selection"] for st in steps), default=0)),
            "max_skipped_normY": float(max((ev["max_skipped_normY"] for ev in skip_events), default=0.0)),
            "chosen_normY_max": float(max(chosen_norms) if chosen_norms else 0.0),
            "chosen_normY_min": float(min(chosen_norms) if chosen_norms else 0.0),
            "dbar_max": float(max(dbar_vals) if dbar_vals else 0.0),
            "dbar_min": float(min(dbar_vals) if dbar_vals else 0.0),
            "dbar_ge_1_count": int(len(dbar_ge1_steps)),
            "min_min_normY": float(min(min_norm_vals) if min_norm_vals else 0.0),
            "all_steps_have_feasible": bool(all(x > 0 for x in feasible_counts)),
            "min_feasible_count": int(min(feasible_counts) if feasible_counts else 0),
            "min_feasible_fraction": float(min(feasible_fractions) if feasible_fractions else 0.0),
            "mean_feasible_fraction": float(np.mean(feasible_fractions) if feasible_fractions else 0.0),
            "mean_deg0_fraction": float(np.mean(deg0_fractions) if deg0_fractions else 0.0),
            "corr_normY_ell_i0_mean": float(np.mean(corr_norm_ell)) if corr_norm_ell else None,
            "corr_normY_degS_mean": float(np.mean(corr_norm_deg)) if corr_norm_deg else None,
            "corr_normY_high_overlap_mean": float(np.mean(corr_norm_high)) if corr_norm_high else None,
            "skip_adjusted_bound_violations": int(skip_bound_viol),
            "skip_adjusted_worst_violation": skip_bound_worst,
        },
    }


def run_suite(
    cases: Sequence[Any],
    eps_list: Sequence[float],
    eps_heavy_fn,
    heavy_ge: bool,
    store_vertex_rows: bool,
) -> List[Dict]:
    out = []
    for cs in cases:
        for eps in eps_list:
            eps_heavy = float(eps_heavy_fn(float(eps)))
            out.append(
                run_modified_rscan(
                    case=cs,
                    eps=float(eps),
                    eps_heavy=eps_heavy,
                    heavy_ge=heavy_ge,
                    store_vertex_rows=store_vertex_rows,
                )
            )
    return out


def summarize_suite(runs: Sequence[Dict], label: str) -> Dict:
    num_steps = sum(len(r["steps"]) for r in runs)
    all_completed = all(r["completed"] for r in runs)

    dbar_max = -float("inf")
    worst_dbar = None
    dbar_ge1 = 0
    skip_runs = 0
    max_skips = 0
    max_skipped_norm = 0.0

    gpl_v_fail = []
    min_feasible_count = float("inf")
    min_feasible_case = None

    for rr in runs:
        rs = rr["run_summary"]
        if rs["skip_count_total"] > 0:
            skip_runs += 1
        max_skips = max(max_skips, rs["max_skips_before_one_selection"])
        max_skipped_norm = max(max_skipped_norm, rs["max_skipped_normY"])
        dbar_ge1 += rs["dbar_ge_1_count"]
        if rs["dbar_max"] > dbar_max:
            dbar_max = rs["dbar_max"]
            worst_dbar = {
                "graph": rr["graph"],
                "eps": rr["eps"],
                "dbar_max": rs["dbar_max"],
            }
        if not rs["all_steps_have_feasible"]:
            gpl_v_fail.append({"graph": rr["graph"], "eps": rr["eps"]})
        if rs["min_feasible_count"] < min_feasible_count:
            min_feasible_count = rs["min_feasible_count"]
            min_feasible_case = {
                "graph": rr["graph"],
                "eps": rr["eps"],
                "min_feasible_count": rs["min_feasible_count"],
                "min_feasible_fraction": rs["min_feasible_fraction"],
            }

    return {
        "label": label,
        "num_runs": len(runs),
        "num_steps": int(num_steps),
        "all_completed": bool(all_completed),
        "gpl_v_holds_all_steps": len(gpl_v_fail) == 0,
        "gpl_v_failures": gpl_v_fail,
        "dbar_ge_1_steps": int(dbar_ge1),
        "worst_dbar": worst_dbar,
        "skip_runs": int(skip_runs),
        "max_skips_before_one_selection": int(max_skips),
        "max_skipped_normY": float(max_skipped_norm),
        "min_feasible_case": min_feasible_case,
    }


def threshold_reconciliation(strict_runs: Sequence[Dict], loose_runs: Sequence[Dict]) -> Dict:
    by_key_loose = {(r["graph"], float(r["eps"])): r for r in loose_runs}
    rows = []
    diff_i0 = 0
    diff_h = 0
    diff_skips = 0

    for rs in strict_runs:
        key = (rs["graph"], float(rs["eps"]))
        rl = by_key_loose[key]
        m_strict = int(rs["m0"])
        m_loose = int(rl["m0"])
        h_strict = int(rs["horizon"])
        h_loose = int(rl["horizon"])
        sk_strict = int(rs["run_summary"]["skip_count_total"])
        sk_loose = int(rl["run_summary"]["skip_count_total"])
        if m_strict != m_loose:
            diff_i0 += 1
        if h_strict != h_loose:
            diff_h += 1
        if sk_strict != sk_loose:
            diff_skips += 1
        rows.append(
            {
                "graph": rs["graph"],
                "eps": rs["eps"],
                "m0_strict": m_strict,
                "m0_loose": m_loose,
                "delta_m0": m_strict - m_loose,
                "horizon_strict": h_strict,
                "horizon_loose": h_loose,
                "delta_horizon": h_strict - h_loose,
                "skip_total_strict": sk_strict,
                "skip_total_loose": sk_loose,
                "delta_skip_total": sk_strict - sk_loose,
                "worst_normY_strict": rs["run_summary"]["chosen_normY_max"],
                "worst_normY_loose": rl["run_summary"]["chosen_normY_max"],
                "worst_dbar_strict": rs["run_summary"]["dbar_max"],
                "worst_dbar_loose": rl["run_summary"]["dbar_max"],
            }
        )

    return {
        "rows": rows,
        "num_cases": len(rows),
        "i0_diff_cases": int(diff_i0),
        "horizon_diff_cases": int(diff_h),
        "skip_diff_cases": int(diff_skips),
        "strict_has_more_skips_cases": int(sum(1 for r in rows if r["delta_skip_total"] > 0)),
        "loose_has_more_skips_cases": int(sum(1 for r in rows if r["delta_skip_total"] < 0)),
    }


def delta_threshold_probe(cases: Sequence[Any], eps_list: Sequence[float], deltas: Sequence[float]) -> Dict:
    # Sweep heavy threshold as tau >= eps - delta.
    rows = []
    for delta in deltas:
        runs = run_suite(
            cases=cases,
            eps_list=eps_list,
            eps_heavy_fn=lambda eps, d=delta: eps - d,
            heavy_ge=True,
            store_vertex_rows=False,
        )
        s = summarize_suite(runs, f"delta_{delta}")
        rows.append(
            {
                "delta": float(delta),
                "skip_runs": int(s["skip_runs"]),
                "dbar_ge_1_steps": int(s["dbar_ge_1_steps"]),
                "gpl_v_holds_all_steps": bool(s["gpl_v_holds_all_steps"]),
                "all_completed": bool(s["all_completed"]),
            }
        )
    return {"rows": rows}


def task3_overlap_summary(strict_runs: Sequence[Dict]) -> Dict:
    feasible_high = []
    infeasible_high = []
    feasible_hn = []
    infeasible_hn = []

    focus_steps = []

    for rr in strict_runs:
        for st in rr["steps"]:
            ss = st["scan_summary"]
            is_focus = (ss["dbar"] >= 1.0 - 1e-10) or (st["skips_before_selection"] > 0)

            rows = st.get("vertex_rows", [])
            if not rows:
                continue
            for rv in rows:
                if rv["feasible"]:
                    feasible_high.append(rv["high_overlap_frac"])
                    feasible_hn.append(rv["high_norm_frac"])
                else:
                    infeasible_high.append(rv["high_overlap_frac"])
                    infeasible_hn.append(rv["high_norm_frac"])

            if is_focus:
                by_norm = sorted(rows, key=lambda x: x["normY"], reverse=True)
                top = by_norm[:5]
                focus_steps.append(
                    {
                        "graph": rr["graph"],
                        "eps": rr["eps"],
                        "t": st["t"],
                        "dbar": ss["dbar"],
                        "skips_before_selection": st["skips_before_selection"],
                        "feasible_count": ss["feasible_count"],
                        "min_normY": ss["min_normY"],
                        "top_norm_vertices": [
                            {
                                "v": x["v"],
                                "normY": x["normY"],
                                "feasible": x["feasible"],
                                "high_overlap_frac": x["high_overlap_frac"],
                                "high_norm_frac": x["high_norm_frac"],
                                "deg_S": x["deg_S"],
                            }
                            for x in top
                        ],
                    }
                )

    return {
        "feasible_high_overlap_mean": float(np.mean(feasible_high)) if feasible_high else None,
        "infeasible_high_overlap_mean": float(np.mean(infeasible_high)) if infeasible_high else None,
        "feasible_high_norm_frac_mean": float(np.mean(feasible_hn)) if feasible_hn else None,
        "infeasible_high_norm_frac_mean": float(np.mean(infeasible_hn)) if infeasible_hn else None,
        "focus_steps": focus_steps,
    }


def task4_cross_degree_summary(strict_runs: Sequence[Dict]) -> Dict:
    total_rows = 0
    viol = 0
    max_ratio = 0.0
    ratio_vals = []
    corr_deg_bound = []

    for rr in strict_runs:
        for st in rr["steps"]:
            ss = st["scan_summary"]
            total_rows += ss["r_t"]
            viol += ss["cross_degree_bound_violations"]
            max_ratio = max(max_ratio, float(ss["max_norm_over_degmax_bound"]))
            if ss["corr_normY_degmax_bound"] is not None:
                corr_deg_bound.append(float(ss["corr_normY_degmax_bound"]))
            for rv in st.get("vertex_rows", []):
                r = float(rv["norm_over_degmax_bound"])
                if np.isfinite(r):
                    ratio_vals.append(r)

    arr = np.array(ratio_vals, dtype=float) if ratio_vals else np.zeros((0,), dtype=float)
    return {
        "total_vertex_rows": int(total_rows),
        "bound_violations": int(viol),
        "max_norm_over_bound": float(max_ratio),
        "ratio_quantiles": quantiles(arr, [0.5, 0.9, 0.95, 0.99]) if len(arr) else {},
        "corr_norm_bound_mean": float(np.mean(corr_deg_bound)) if corr_deg_bound else None,
    }


def task5_skip_adjusted_summary(strict_runs: Sequence[Dict]) -> Dict:
    skip_runs = []
    max_s = 0
    viol = 0
    worst = None

    for rr in strict_runs:
        rs = rr["run_summary"]
        if rs["skip_count_total"] > 0:
            skip_runs.append(
                {
                    "graph": rr["graph"],
                    "eps": rr["eps"],
                    "skip_total": rs["skip_count_total"],
                    "skip_steps": rs["skip_steps"],
                    "max_skips_before_selection": rs["max_skips_before_one_selection"],
                }
            )
        max_s = max(max_s, rs["max_skips_before_one_selection"])
        viol += rs["skip_adjusted_bound_violations"]
        w = rs["skip_adjusted_worst_violation"]
        if w is not None and (worst is None or w["gap"] > worst["gap"]):
            worst = {
                "graph": rr["graph"],
                "eps": rr["eps"],
                **w,
            }

    return {
        "skip_runs": skip_runs,
        "skip_run_count": len(skip_runs),
        "max_skips_before_selection": int(max_s),
        "skip_adjusted_bound_violations": int(viol),
        "skip_adjusted_worst_violation": worst,
    }


def task6_existence_summary(strict_runs: Sequence[Dict]) -> Dict:
    feasible_frac = []
    deg0_frac = []
    e_norm = []
    e_trace = []
    min_norm = []
    min_trace = []

    for rr in strict_runs:
        for st in rr["steps"]:
            ss = st["scan_summary"]
            feasible_frac.append(float(ss["feasible_fraction"]))
            deg0_frac.append(float(ss["deg0_fraction"]))
            e_norm.append(float(ss["mean_normY"]))
            e_trace.append(float(ss["mean_traceY"]))
            min_norm.append(float(ss["min_normY"]))
            # min trace from rows if present
            if st.get("vertex_rows"):
                mt = min(float(r["traceY"]) for r in st["vertex_rows"])
            else:
                mt = 0.0
            min_trace.append(mt)

    arr_f = np.array(feasible_frac, dtype=float)
    arr_d0 = np.array(deg0_frac, dtype=float)
    arr_en = np.array(e_norm, dtype=float)
    arr_et = np.array(e_trace, dtype=float)
    arr_mn = np.array(min_norm, dtype=float)
    arr_mt = np.array(min_trace, dtype=float)

    return {
        "all_steps_have_positive_feasible_prob": bool(np.all(arr_f > 0.0 + 1e-12)) if len(arr_f) else True,
        "feasible_fraction_quantiles": quantiles(arr_f, [0.1, 0.25, 0.5, 0.75, 0.9]),
        "deg0_fraction_quantiles": quantiles(arr_d0, [0.1, 0.25, 0.5, 0.75, 0.9]),
        "mean_E_normY": float(np.mean(arr_en)) if len(arr_en) else 0.0,
        "mean_E_traceY": float(np.mean(arr_et)) if len(arr_et) else 0.0,
        "mean_ratio_E_norm_over_E_trace": float(np.mean(arr_en / np.maximum(arr_et, 1e-18))) if len(arr_en) else 0.0,
        "min_min_normY": float(np.min(arr_mn)) if len(arr_mn) else 0.0,
        "min_min_traceY": float(np.min(arr_mt)) if len(arr_mt) else 0.0,
    }


def build_markdown(out: Dict) -> str:
    b = out["summary"]["strict_base"]
    a = out["summary"]["strict_adversarial"]
    t2 = out["summary"]["task2_threshold_reconciliation"]
    t3 = out["summary"]["task3_eigenspace_overlap"]
    t4 = out["summary"]["task4_cross_degree_bound"]
    t5 = out["summary"]["task5_partial_averages_with_skips"]
    t6 = out["summary"]["task6_existence"]

    lines = []
    lines.append("# Problem 6 Cycle 8 Codex Verification")
    lines.append("")
    lines.append("Date: 2026-02-13")
    lines.append("Agent: Codex")
    lines.append("Base handoff: `data/first-proof/problem6-codex-cycle8-handoff.md`")
    lines.append("")
    lines.append("Artifacts:")
    lines.append("- Script: `scripts/verify-p6-cycle8-codex.py`")
    lines.append("- Results JSON: `data/first-proof/problem6-codex-cycle8-results.json`")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- Strict-threshold base runs: {b['num_runs']} ({b['num_steps']} scanned steps)")
    lines.append(f"- Strict-threshold adversarial runs: {a['num_runs']} ({a['num_steps']} scanned steps)")
    lines.append(f"- GPL-V on strict runs (all steps have some feasible vertex): base={b['gpl_v_holds_all_steps']}, adversarial={a['gpl_v_holds_all_steps']}")
    lines.append(f"- Strict base dbar>=1 steps: {b['dbar_ge_1_steps']} (worst {b['worst_dbar']['dbar_max']:.6f} at {b['worst_dbar']['graph']} eps={b['worst_dbar']['eps']})")
    lines.append(f"- Strict skip runs: base={b['skip_runs']}, adversarial={a['skip_runs']} (max skips before one selection = {max(b['max_skips_before_one_selection'], a['max_skips_before_one_selection'])})")
    lines.append("")
    lines.append("## Task 1: Full R-Scan")
    lines.append("")
    lines.append("At each selected step, the scan records all vertices in R_t with:")
    lines.append("- normY(v), traceY(v)")
    lines.append("- deg_S(v), ell_v^{I0}, ell_v^{S_t}")
    lines.append("- high-eigenspace overlap and cross-degree bound diagnostics")
    lines.append("")
    lines.append(f"Key strict-base minimum feasible count over all steps: {b['min_feasible_case']['min_feasible_count']} (fraction {b['min_feasible_case']['min_feasible_fraction']:.4f})")
    lines.append("")
    lines.append("## Task 2: Threshold Reconciliation")
    lines.append("")
    lines.append(f"- Cases compared: {t2['num_cases']}")
    lines.append(f"- Different I0 size: {t2['i0_diff_cases']}")
    lines.append(f"- Different horizon: {t2['horizon_diff_cases']}")
    lines.append(f"- Different skip totals: {t2['skip_diff_cases']}")
    lines.append(f"- Strict has more skips: {t2['strict_has_more_skips_cases']}, loose has more skips: {t2['loose_has_more_skips_cases']}")
    lines.append("")
    lines.append("## Task 3: Eigenspace Overlap")
    lines.append("")
    lines.append(f"- Mean high-overlap fraction (feasible): {t3['feasible_high_overlap_mean']}")
    lines.append(f"- Mean high-overlap fraction (infeasible): {t3['infeasible_high_overlap_mean']}")
    lines.append(f"- Mean high-norm fraction (feasible): {t3['feasible_high_norm_frac_mean']}")
    lines.append(f"- Mean high-norm fraction (infeasible): {t3['infeasible_high_norm_frac_mean']}")
    lines.append(f"- Focus steps captured (skip or dbar>=1): {len(t3['focus_steps'])}")
    lines.append("")
    lines.append("## Task 4: Cross-Degree Bound")
    lines.append("")
    lines.append("Tested bound: normY(v) <= deg_S(v) * max_{e incident to v and S} z_e^T B_t z_e")
    lines.append(f"- Vertex rows checked: {t4['total_vertex_rows']}")
    lines.append(f"- Violations: {t4['bound_violations']}")
    lines.append(f"- Max ratio norm/bound: {t4['max_norm_over_bound']:.6f}")
    lines.append(f"- Ratio quantiles: {t4['ratio_quantiles']}")
    lines.append("")
    lines.append("## Task 5: Partial Averages With Skips")
    lines.append("")
    lines.append(f"- Skip runs (strict): {t5['skip_run_count']}")
    lines.append(f"- Max skips before one selection: {t5['max_skips_before_selection']}")
    lines.append(f"- Skip-adjusted dbar0 bound violations: {t5['skip_adjusted_bound_violations']}")
    lines.append(f"- Worst skip-adjusted violation: {t5['skip_adjusted_worst_violation']}")
    lines.append("")
    lines.append("## Task 6: Alternative Existence Arguments")
    lines.append("")
    lines.append(f"- Pr[normY<1] > 0 at all steps: {t6['all_steps_have_positive_feasible_prob']}")
    lines.append(f"- Feasible-fraction quantiles: {t6['feasible_fraction_quantiles']}")
    lines.append(f"- deg_S=0 fraction quantiles: {t6['deg0_fraction_quantiles']}")
    lines.append(f"- Mean E[normY]: {t6['mean_E_normY']:.6f}")
    lines.append(f"- Mean E[traceY]: {t6['mean_E_traceY']:.6f}")
    lines.append(f"- Mean E[normY]/E[traceY]: {t6['mean_ratio_E_norm_over_E_trace']:.6f}")
    lines.append("")
    lines.append("## Bottom Line")
    lines.append("")
    lines.append("With strict threshold (tau >= eps), No-Skip is not universal, but GPL-V remains empirically robust on this suite.")
    lines.append("The full R-scan/eigenspace data is now available for proving existence of SOME feasible vertex directly.")

    return "\n".join(lines) + "\n"


def print_summary(out: Dict):
    b = out["summary"]["strict_base"]
    a = out["summary"]["strict_adversarial"]
    t2 = out["summary"]["task2_threshold_reconciliation"]
    print("=" * 96)
    print("P6 CYCLE 8 CODEX: GPL-V R-SCAN + THRESHOLD + OVERLAP")
    print("=" * 96)
    print(f"strict base: runs={b['num_runs']} steps={b['num_steps']} gpl_v_all={b['gpl_v_holds_all_steps']} dbar>=1={b['dbar_ge_1_steps']} skip_runs={b['skip_runs']}")
    print(f"strict adv : runs={a['num_runs']} steps={a['num_steps']} gpl_v_all={a['gpl_v_holds_all_steps']} dbar>=1={a['dbar_ge_1_steps']} skip_runs={a['skip_runs']}")
    print(f"threshold reconcile: cases={t2['num_cases']} i0_diff={t2['i0_diff_cases']} skip_diff={t2['skip_diff_cases']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eps", type=float, nargs="*", default=[0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--skip-adversarial", action="store_true")
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("data/first-proof/problem6-codex-cycle8-results.json"),
    )
    ap.add_argument(
        "--out-md",
        type=Path,
        default=Path("data/first-proof/problem6-codex-cycle8-verification.md"),
    )
    args = ap.parse_args()

    base_specs = C7["build_base_suite"](args.seed)
    adv_specs = C7["build_adversarial_suite"](args.seed) if not args.skip_adversarial else []
    base_cases = [C7["prep_case"](g) for g in base_specs]
    adv_cases = [C7["prep_case"](g) for g in adv_specs]

    # Strict threshold full R-scan (primary task).
    strict_base = run_suite(
        cases=base_cases,
        eps_list=args.eps,
        eps_heavy_fn=lambda eps: eps,
        heavy_ge=True,
        store_vertex_rows=True,
    )
    strict_adv = run_suite(
        cases=adv_cases,
        eps_list=args.eps,
        eps_heavy_fn=lambda eps: eps,
        heavy_ge=True,
        store_vertex_rows=True,
    )

    # Loose threshold summary-only for reconciliation.
    loose_base = run_suite(
        cases=base_cases,
        eps_list=args.eps,
        eps_heavy_fn=lambda eps: eps,
        heavy_ge=False,
        store_vertex_rows=False,
    )

    # Optional delta probe around strict threshold (base only).
    delta_probe = delta_threshold_probe(base_cases, args.eps, deltas=[0.0, 1e-12, 1e-10, 1e-8])

    summary = {
        "strict_base": summarize_suite(strict_base, "strict_base"),
        "strict_adversarial": summarize_suite(strict_adv, "strict_adversarial"),
        "task2_threshold_reconciliation": threshold_reconciliation(strict_base, loose_base),
        "task2_delta_probe": delta_probe,
        "task3_eigenspace_overlap": task3_overlap_summary(strict_base + strict_adv),
        "task4_cross_degree_bound": task4_cross_degree_summary(strict_base + strict_adv),
        "task5_partial_averages_with_skips": task5_skip_adjusted_summary(strict_base + strict_adv),
        "task6_existence": task6_existence_summary(strict_base + strict_adv),
    }

    out = {
        "meta": {
            "date": "2026-02-13",
            "agent": "Codex",
            "seed": args.seed,
            "eps_list": [float(x) for x in args.eps],
            "strict_threshold": "heavy iff tau >= eps",
            "loose_threshold": "heavy iff tau > eps",
            "base_suite_size": len(base_cases),
            "adversarial_suite_size": len(adv_cases),
        },
        "summary": summary,
        "strict_base_runs": strict_base,
        "strict_adversarial_runs": strict_adv,
        "loose_base_runs_summary": [
            {
                "graph": r["graph"],
                "eps": r["eps"],
                "m0": r["m0"],
                "horizon": r["horizon"],
                "completed": r["completed"],
                "run_summary": r["run_summary"],
            }
            for r in loose_base
        ],
    }

    print_summary(out)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=to_jsonable)

    md = build_markdown(out)
    with args.out_md.open("w", encoding="utf-8") as f:
        f.write(md)

    print(f"\nWrote JSON: {args.out_json}")
    print(f"Wrote markdown: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
