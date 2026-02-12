#!/usr/bin/env python3
"""
Direction B probe for GPL-H: Hyperbolic barrier / self-concordance analysis.

The multivariate polynomial P(x) = det(eps*I - M_t - sum_v x_v C_t(v)) is
hyperbolic with respect to x=0. This script explores whether the hyperbolicity
cone structure (convexity, self-concordance, Frobenius averaging) provides
tighter bounds than trace averaging alone.

Key diagnostics per barrier step:
  - score(v)  = ||Y_t(v)||         (operator norm — what GPL-H needs)
  - drift(v)  = tr(Y_t(v))         (trace — what L1 controls)
  - frob(v)   = ||Y_t(v)||_F       (Frobenius — intermediate quantity)
  - sc_local(v) = frob(v)          (self-concordance local norm)

New partial result tested:
  If avg_v ||Y_t(v)||_F^2 < 1, then some v has ||Y_t(v)|| < 1.
  This is strictly between trace averaging and score control.

This is a computational diagnostic, not a proof.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import numpy as np
from numpy.linalg import eigh, norm, inv, cholesky


# --- Graph generators (aligned with verify-p6-gpl-h.py) ---

def complete_graph(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def cycle_graph(n):
    return [(i, (i + 1) % n) for i in range(n)]

def barbell_graph(k):
    edges = []
    for i in range(k):
        for j in range(i + 1, k):
            edges.append((i, j))
    for i in range(k, 2 * k):
        for j in range(i + 1, 2 * k):
            edges.append((i, j))
    edges.append((k - 1, k))
    return 2 * k, edges

def star_graph(n):
    return [(0, i) for i in range(1, n)]

def erdos_renyi(n, p, rng):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((i, j))
    return edges

def path_graph(n):
    return [(i, i + 1) for i in range(n - 1)]


# --- Core linear algebra ---

def graph_laplacian(n, edges):
    L = np.zeros((n, n))
    for u, v in edges:
        L[u, u] += 1; L[v, v] += 1
        L[u, v] -= 1; L[v, u] -= 1
    return L

def pseudoinverse_sqrt(L):
    eigvals, eigvecs = eigh(L)
    tol = 1e-10 * max(abs(eigvals))
    result = np.zeros_like(L)
    for i in range(len(eigvals)):
        if eigvals[i] > tol:
            result += (1.0 / np.sqrt(eigvals[i])) * np.outer(eigvecs[:, i], eigvecs[:, i])
    return result

def compute_edge_matrices(Lps, edges):
    n = Lps.shape[0]
    X_list, tau_list = [], []
    for u, v in edges:
        b = np.zeros(n); b[u] = 1.0; b[v] = -1.0
        Lb = Lps @ b
        X_e = np.outer(Lb, Lb)
        tau_e = Lb @ Lb
        X_list.append(X_e)
        tau_list.append(tau_e)
    return X_list, tau_list


# --- Case-2b instance finder ---

def find_case2b(n, edges, eps):
    """Find independent set I in G_H, check if Case 2b (alpha_I > eps)."""
    L = graph_laplacian(n, edges)
    Lps = pseudoinverse_sqrt(L)
    X_list, tau_list = compute_edge_matrices(Lps, edges)

    # Heavy subgraph
    adj_heavy = {v: set() for v in range(n)}
    for i, (u, v) in enumerate(edges):
        if tau_list[i] > eps:
            adj_heavy[u].add(v)
            adj_heavy[v].add(u)

    # Greedy independent set in G_H
    remaining = set(range(n))
    I = []
    for v in sorted(remaining, key=lambda x: len(adj_heavy.get(x, set()))):
        if v in remaining:
            I.append(v)
            remaining -= {v}
            remaining -= adj_heavy[v]
    I_set = set(I)

    # Internal edges
    internal_idx = [i for i, (u, v) in enumerate(edges) if u in I_set and v in I_set]
    if not internal_idx:
        return None

    # alpha_I
    M_I = sum(X_list[i] for i in internal_idx)
    alpha_I = norm(M_I, ord=2)
    if alpha_I <= eps:
        return None

    # Core regularization
    lev_deg = {v: 0.0 for v in I}
    for i in internal_idx:
        u, v = edges[i]
        lev_deg[u] += tau_list[i]
        lev_deg[v] += tau_list[i]
    T_I = sum(tau_list[i] for i in internal_idx)
    D = 4 * T_I / len(I) if len(I) > 0 else 0
    I0 = [v for v in I if lev_deg.get(v, 0) <= D]
    if len(I0) < 4:
        return None

    return {
        "n": n, "edges": edges, "X_list": X_list, "tau_list": tau_list,
        "I0": I0, "alpha_I": alpha_I, "D": D, "Lps": Lps,
    }


# --- Direction B diagnostics ---

@dataclass
class StepDiag:
    t: int
    r_t: int
    min_score: float
    avg_score: float
    min_drift: float
    avg_drift: float
    min_frob: float
    avg_frob_sq: float      # avg ||Y_t(v)||_F^2 — the Frobenius averaging bound
    hessian_diag_avg: float  # same as avg_frob_sq (self-concordance diagonal)
    hessian_offdiag_norm: float  # ||off-diagonal Hessian|| — interaction strength
    frob_avg_lt_1: bool      # key test: avg_frob_sq < 1 ?
    det_ratio_min: float     # min_v det(I - Y_t(v)) — positive iff score < 1
    trace_bound: float       # (tD/r_t)*tr(B_t) — the L1 trace bound
    rank_gap_avg: float      # avg drift/score ratio (measures eigenvalue spreading)


def run_direction_b(instance, eps, c_step=0.5):
    """Run barrier greedy on a Case-2b instance, collecting Direction B diagnostics."""
    n = instance["n"]
    edges = instance["edges"]
    X_list = instance["X_list"]
    tau_list = instance["tau_list"]
    I0 = instance["I0"]
    D = instance["D"]

    I0_set = set(I0)
    m0 = len(I0)
    T = max(1, int(c_step * eps * n))
    T = min(T, m0 - 2)

    M_t = np.zeros((n, n))
    S_t = set()
    remaining = list(I0)
    results = []

    for t in range(T):
        # Barrier
        gap = eps * np.eye(n) - M_t
        eigvals_gap = eigh(gap)[0]
        if min(eigvals_gap) < 1e-12:
            break
        B_t = inv(gap)
        try:
            B_t_sqrt = cholesky(B_t + 1e-14 * np.eye(n))
        except np.linalg.LinAlgError:
            break

        R_t = [v for v in remaining if v not in S_t]
        r_t = len(R_t)
        if r_t < 2:
            break

        # Compute Y_t(v) for each v in R_t
        scores, drifts, frobs, det_ratios = [], [], [], []
        frob_sqs = []
        Y_list = []

        for v in R_t:
            C_v = np.zeros((n, n))
            for i, (u, w) in enumerate(edges):
                if (u in S_t and w == v) or (w in S_t and u == v):
                    if u in I0_set and w in I0_set:
                        C_v += X_list[i]
            Y_v = B_t_sqrt @ C_v @ B_t_sqrt.T
            Y_list.append(Y_v)

            eigvals_Y = eigh(Y_v)[0]
            score_v = max(eigvals_Y)
            drift_v = sum(eigvals_Y)
            frob_v = np.sqrt(sum(ev ** 2 for ev in eigvals_Y))
            det_ratio = np.prod([1 - ev for ev in eigvals_Y])

            scores.append(max(score_v, 0))
            drifts.append(max(drift_v, 0))
            frobs.append(frob_v)
            frob_sqs.append(frob_v ** 2)
            det_ratios.append(det_ratio)

        # Hessian off-diagonal: sample a few pairs
        offdiag_norms = []
        n_pairs = min(20, r_t * (r_t - 1) // 2)
        if r_t >= 2 and n_pairs > 0:
            rng = np.random.default_rng(42 + t)
            pairs = rng.choice(r_t, size=(n_pairs, 2), replace=True)
            for pi, pj in pairs:
                if pi == pj:
                    continue
                # Hessian off-diagonal: tr(B_t C_i B_t C_j)
                cross = np.trace(Y_list[pi] @ Y_list[pj])
                offdiag_norms.append(abs(cross))

        tr_Bt = np.trace(B_t)
        trace_bound = (t * D / r_t) * tr_Bt if r_t > 0 else float('inf')

        # Rank gap: drift / score ratio (measures spreading)
        rank_gaps = []
        for s, d in zip(scores, drifts):
            if s > 1e-12:
                rank_gaps.append(d / s)

        active_scores = [s for s in scores if s > 1e-10]
        active_frob_sqs = [f for f, s in zip(frob_sqs, scores) if s > 1e-10]

        diag = StepDiag(
            t=t,
            r_t=r_t,
            min_score=min(scores) if scores else float('inf'),
            avg_score=np.mean(scores) if scores else 0,
            min_drift=min(drifts) if drifts else float('inf'),
            avg_drift=np.mean(drifts) if drifts else 0,
            min_frob=min(frobs) if frobs else float('inf'),
            avg_frob_sq=np.mean(active_frob_sqs) if active_frob_sqs else 0,
            hessian_diag_avg=np.mean(active_frob_sqs) if active_frob_sqs else 0,
            hessian_offdiag_norm=np.mean(offdiag_norms) if offdiag_norms else 0,
            frob_avg_lt_1=(np.mean(active_frob_sqs) < 1.0) if active_frob_sqs else True,
            det_ratio_min=min(det_ratios) if det_ratios else 0,
            trace_bound=trace_bound,
            rank_gap_avg=np.mean(rank_gaps) if rank_gaps else 1.0,
        )
        results.append(diag)

        # Greedy: pick vertex with min score
        best_v_idx = int(np.argmin(scores))
        best_v = R_t[best_v_idx]
        if scores[best_v_idx] >= 1.0 - 1e-12:
            break

        # Update
        C_best = np.zeros((n, n))
        for i, (u, w) in enumerate(edges):
            if (u in S_t and w == best_v) or (w in S_t and u == best_v):
                if u in I0_set and w in I0_set:
                    C_best += X_list[i]
        M_t = M_t + C_best
        S_t.add(best_v)

    return results


# --- Main ---

def main():
    ap = argparse.ArgumentParser(description="Direction B probe for GPL-H")
    ap.add_argument("--nmax", type=int, default=24)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.15, 0.2, 0.25, 0.3])
    ap.add_argument("--c-step", type=float, default=0.5)
    args = ap.parse_args()

    rng = np.random.default_rng(2026)

    # Graph families
    families = []
    for n in range(8, args.nmax + 1, 2):
        families.append((f"K_{n}", n, complete_graph(n)))
        families.append((f"C_{n}", n, cycle_graph(n)))
        if n >= 10:
            families.append((f"Path_{n}", n, path_graph(n)))
            families.append((f"Star_{n}", n, star_graph(n)))
        if n >= 10 and n % 2 == 0:
            k = n // 2
            bn, be = barbell_graph(k)
            families.append((f"Barbell_{k}", bn, be))
        families.append((f"ER_{n}_0.4", n, erdos_renyi(n, 0.4, rng)))
        families.append((f"ER_{n}_0.6", n, erdos_renyi(n, 0.6, rng)))

    print("=" * 90)
    print("Direction B: Hyperbolic Barrier / Self-Concordance Diagnostics")
    print("=" * 90)

    total_instances = 0
    total_steps = 0
    worst_min_score = 0.0
    worst_avg_frob_sq = 0.0
    frob_bound_holds = 0
    frob_bound_fails = 0
    rank_gap_values = []

    header = (f"{'Graph':<18} {'eps':>5} {'steps':>5} {'worst_min_sc':>12} "
              f"{'worst_avg_F2':>12} {'F2<1?':>5} {'avg_rank_gap':>12} "
              f"{'trace_bnd':>10}")
    print(f"\n{header}")
    print("-" * 90)

    for name, n, edges in families:
        if not edges:
            continue
        for eps in args.eps:
            inst = find_case2b(n, edges, eps)
            if inst is None:
                continue

            diags = run_direction_b(inst, eps, c_step=args.c_step)
            if not diags:
                continue

            total_instances += 1
            total_steps += len(diags)

            w_min_sc = max(d.min_score for d in diags)
            w_avg_f2 = max(d.avg_frob_sq for d in diags)
            f2_ok = all(d.frob_avg_lt_1 for d in diags)
            avg_rg = np.mean([d.rank_gap_avg for d in diags])
            worst_tb = max(d.trace_bound for d in diags)

            worst_min_score = max(worst_min_score, w_min_sc)
            worst_avg_frob_sq = max(worst_avg_frob_sq, w_avg_f2)
            rank_gap_values.append(avg_rg)

            if f2_ok:
                frob_bound_holds += 1
            else:
                frob_bound_fails += 1

            f2_str = "YES" if f2_ok else "NO"
            print(f"{name:<18} {eps:>5.2f} {len(diags):>5} {w_min_sc:>12.6f} "
                  f"{w_avg_f2:>12.6f} {f2_str:>5} {avg_rg:>12.3f} "
                  f"{worst_tb:>10.3f}")

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"Case-2b instances: {total_instances}")
    print(f"Steps analyzed:    {total_steps}")
    print(f"Worst min score:   {worst_min_score:.6f}  (GPL-H needs < 1)")
    print(f"Worst avg ||Y||_F^2: {worst_avg_frob_sq:.6f}  (Frobenius bound needs < 1)")
    print(f"Frobenius bound holds: {frob_bound_holds}/{frob_bound_holds + frob_bound_fails}")
    if rank_gap_values:
        print(f"Avg rank gap (drift/score): {np.mean(rank_gap_values):.3f} "
              f"(1 = rank-1, higher = more spreading)")
    print()

    if worst_avg_frob_sq < 1.0:
        print(">>> Frobenius averaging bound < 1 in ALL tested steps!")
        print("    This is strictly better than trace averaging.")
        print("    If provable, it gives: some v has ||Y_t(v)||_F < 1 => ||Y_t(v)|| < 1.")
    else:
        print(f">>> Frobenius averaging bound EXCEEDS 1 in some steps "
              f"(worst: {worst_avg_frob_sq:.6f}).")
        print("    The self-concordance / Frobenius approach is not universal.")

    print(f"\nNOT tested: GPL-H itself (open). See scripts/verify-p6-gpl-h.py.")


if __name__ == "__main__":
    main()
