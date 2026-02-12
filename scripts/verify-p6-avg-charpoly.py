#!/usr/bin/env python3
"""Check whether the average characteristic polynomial approach closes GPL-H.

For each nontrivial step, compute:
  p_avg(x) = (1/|A_t|) Σ_{v∈A_t} det(xI - Y_t(v))

By the MSS largest-root lemma: if the largest root of p_avg is < 1,
then some v has ||Y_t(v)|| < 1.

This script checks:
  1. The largest root of p_avg at each nontrivial step.
  2. Whether largest_root < 1 universally at c_step = 1/3.
"""

from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("p6base", repo_root / "scripts" / "verify-p6-gpl-h.py")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


def build_suite(nmax, rng):
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
    return suite


def charpoly_coeffs(Y, d):
    """Compute coefficients of det(xI - Y) as a polynomial in x.

    Returns array c where p(x) = Σ_k c[k] x^k, degree d.
    Uses eigenvalues: det(xI - Y) = Π_i (x - λ_i).
    """
    evals = np.linalg.eigvalsh(Y)
    # p(x) = Π_i (x - λ_i)
    # Use numpy polynomial: start with [1] and multiply by (x - λ_i)
    p = np.array([1.0])
    for lam in evals:
        p = np.convolve(p, [1.0, -lam])
    return p


def largest_real_root(coeffs):
    """Find the largest real root of polynomial with given coefficients.

    coeffs[0]*x^d + coeffs[1]*x^{d-1} + ... + coeffs[d]
    """
    if len(coeffs) <= 1:
        return float('-inf')
    roots = np.roots(coeffs)
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    if len(real_roots) == 0:
        return float('-inf')
    return float(np.max(real_roots))


def analyze_step_charpoly(inst, c_step):
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
        if not R_t:
            break
        headroom = eps * np.eye(n) - M_t
        if np.min(np.linalg.eigvalsh(headroom)) < 1e-12:
            break

        B_t = np.linalg.inv(headroom)
        Bsqrt = np.linalg.cholesky(B_t + 1e-14 * np.eye(n))

        avg_poly = None
        n_active = 0
        best_v, best_score = None, float("inf")
        max_score = 0.0
        min_score_active = float("inf")

        for v in R_t:
            C_v = np.zeros((n, n))
            for u in S_t:
                key = (min(u, v), max(u, v))
                if key in edge_idx:
                    C_v += inst.X_edges[edge_idx[key]]
            Y_v = Bsqrt @ C_v @ Bsqrt.T
            evals = np.linalg.eigvalsh(Y_v)
            s_v = float(max(evals[-1], 0.0))

            if s_v > 1e-10:
                n_active += 1
                coeffs = charpoly_coeffs(Y_v, n)
                if avg_poly is None:
                    avg_poly = coeffs.copy()
                else:
                    # Pad to same length if needed
                    maxlen = max(len(avg_poly), len(coeffs))
                    ap = np.zeros(maxlen)
                    cp = np.zeros(maxlen)
                    ap[maxlen - len(avg_poly):] = avg_poly
                    cp[maxlen - len(coeffs):] = coeffs
                    avg_poly = ap + cp
                max_score = max(max_score, s_v)
                min_score_active = min(min_score_active, s_v)

            if s_v < best_score:
                best_score = s_v
                best_v = v

        if n_active > 0 and avg_poly is not None and best_score > 1e-12:
            avg_poly /= n_active
            lr = largest_real_root(avg_poly)
            rows.append({
                "graph": inst.graph_name, "n": n, "eps": eps, "t": t,
                "active": n_active,
                "min_score": best_score,
                "max_score": max_score,
                "largest_root_avg": lr,
                "root_lt_1": lr < 1.0,
            })

        if best_v is None:
            break
        S_t.append(best_v)
        S_set.add(best_v)
        for u in S_t[:-1]:
            key = (min(best_v, u), max(best_v, u))
            if key in edge_idx:
                M_t += inst.X_edges[edge_idx[key]]

    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=64)
    ap.add_argument("--c-step", type=float, default=1/3)
    ap.add_argument("--seed", type=int, default=20260212)
    args = ap.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    suite = build_suite(args.nmax, rng)
    eps_list = [0.1, 0.12, 0.15, 0.2, 0.25, 0.3]

    all_rows = []
    case2b = 0

    for graph_name, n, edges in suite:
        for eps in eps_list:
            inst = base.find_case2b_instance(n, edges, eps, graph_name=graph_name)
            if inst is None:
                continue
            case2b += 1
            all_rows.extend(analyze_step_charpoly(inst, args.c_step))

    nontrivial = [r for r in all_rows if r["min_score"] > 1e-12]
    violations = [r for r in nontrivial if not r["root_lt_1"]]

    print("=" * 80)
    print(f"AVERAGE CHARPOLY CHECK  (c_step = {args.c_step:.4f}, nmax = {args.nmax})")
    print("=" * 80)
    print(f"Case-2b instances: {case2b}")
    print(f"Nontrivial rows:   {len(nontrivial)}")
    print(f"Violations (largest_root >= 1): {len(violations)}")

    if nontrivial:
        lrs = [r["largest_root_avg"] for r in nontrivial]
        print(f"\nLargest root of avg charpoly:")
        print(f"  max  = {max(lrs):.6f}")
        print(f"  mean = {np.mean(lrs):.6f}")
        print(f"  p95  = {np.quantile(lrs, 0.95):.6f}")

    worst = sorted(nontrivial, key=lambda r: r["largest_root_avg"], reverse=True)[:10]
    if worst:
        print("\nTop 10 worst rows:")
        for r in worst:
            print(f"  {r['graph']:<20} n={r['n']:>3} eps={r['eps']:.2f} t={r['t']:>2} "
                  f"max_s={r['max_score']:.6f} root={r['largest_root_avg']:.6f} "
                  f"<1? {r['root_lt_1']}")


if __name__ == "__main__":
    main()
