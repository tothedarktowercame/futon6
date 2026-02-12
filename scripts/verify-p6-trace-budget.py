#!/usr/bin/env python3
"""Check the trace budget argument for GPL-H.

Key identity: dbar = tr(B_t Q_t) / |A_t|
where Q_t = Σ_{e∈E_c} X_e (crossing edges sum).

When M_t = 0: B_t = (1/ε)I, so dbar = tr(Q_t)/(ε·|A_t|).

We compute:
  - tr(Q_t): total crossing leverage
  - |A_t|: number of active vertices
  - ||Q_t||: operator norm of crossing sum
  - The "anisotropy ratio" tr(B_t Q_t) / (||B_t|| · tr(Q_t))
    (= 1 if Q is aligned with B's top eigenspace, < 1 if spread out)

The anisotropy ratio is the key: if it's bounded away from 1,
then the scalar bound dbar ≤ tr(Q)/(|A_t|·(ε-||M||)) is conservative,
and the true dbar is smaller.
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


def analyze_trace_budget(inst, c_step):
    n = inst.n
    eps = inst.epsilon
    I0 = inst.I0
    I0_set = set(I0)
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
        eig_h = np.linalg.eigvalsh(headroom)
        if np.min(eig_h) < 1e-12:
            break

        B_t = np.linalg.inv(headroom)

        # Crossing edges sum Q_t
        Q_t = np.zeros((n, n))
        for u in S_t:
            for v in R_t:
                key = (min(u, v), max(u, v))
                if key in edge_idx:
                    Q_t += inst.X_edges[edge_idx[key]]

        tr_Q = np.trace(Q_t)
        norm_Q = np.linalg.norm(Q_t, ord=2)
        tr_BQ = np.trace(B_t @ Q_t)
        norm_B = np.linalg.norm(B_t, ord=2)
        norm_M = np.linalg.norm(M_t, ord=2)

        # Count active vertices
        Bsqrt = np.linalg.cholesky(B_t + 1e-14 * np.eye(n))
        n_active = 0
        best_v, best_score = None, float("inf")
        for v in R_t:
            C_v = np.zeros((n, n))
            for u in S_t:
                key = (min(u, v), max(u, v))
                if key in edge_idx:
                    C_v += inst.X_edges[edge_idx[key]]
            Y_v = Bsqrt @ C_v @ Bsqrt.T
            s_v = float(np.linalg.norm(Y_v, ord=2))
            if s_v > 1e-10:
                n_active += 1
            if s_v < best_score:
                best_score = s_v
                best_v = v

        if n_active > 0 and best_score > 1e-12:
            dbar_actual = tr_BQ / n_active
            # Scalar bound: dbar ≤ tr_Q / (n_active * (eps - norm_M))
            dbar_scalar = tr_Q / (n_active * max(eps - norm_M, 1e-15))
            # "Fresh" bound: dbar when B_t = (1/eps)I
            dbar_fresh = tr_Q / (n_active * eps)
            # Anisotropy: how much of Q's trace lands in B's amplified space
            aniso = tr_BQ / (norm_B * tr_Q) if tr_Q > 1e-15 else 0.0

            rows.append({
                "graph": inst.graph_name, "n": n, "eps": eps, "t": t,
                "active": n_active, "r_t": len(R_t),
                "tr_Q": tr_Q, "norm_Q": norm_Q,
                "norm_M": norm_M, "norm_B": norm_B,
                "dbar_actual": dbar_actual,
                "dbar_scalar": dbar_scalar,
                "dbar_fresh": dbar_fresh,
                "anisotropy": aniso,
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
    for graph_name, n, edges in suite:
        for eps in eps_list:
            inst = base.find_case2b_instance(n, edges, eps, graph_name=graph_name)
            if inst is None:
                continue
            all_rows.extend(analyze_trace_budget(inst, args.c_step))

    nontrivial = [r for r in all_rows if r["dbar_actual"] > 1e-12]
    print(f"Nontrivial rows: {len(nontrivial)}")

    if nontrivial:
        dba = [r["dbar_actual"] for r in nontrivial]
        dbs = [r["dbar_scalar"] for r in nontrivial]
        dbf = [r["dbar_fresh"] for r in nontrivial]
        ani = [r["anisotropy"] for r in nontrivial]

        print(f"\ndbar_actual:  worst = {max(dba):.6f}  mean = {np.mean(dba):.6f}")
        print(f"dbar_scalar:  worst = {max(dbs):.6f}  mean = {np.mean(dbs):.6f}")
        print(f"dbar_fresh:   worst = {max(dbf):.6f}  mean = {np.mean(dbf):.6f}")
        print(f"anisotropy:   worst = {max(ani):.6f}  mean = {np.mean(ani):.6f}")

        print(f"\ndbar_fresh >= 1: {sum(1 for x in dbf if x >= 1)}/{len(dbf)}")
        print(f"dbar_actual >= 1: {sum(1 for x in dba if x >= 1)}/{len(dba)}")

        # Check: does dbar_fresh < 1 close GPL-H?
        # dbar_fresh = tr(Q)/(eps*|A_t|) — this is what dbar would be if M_t = 0.
        print(f"\nKey question: does dbar_fresh < 1 hold universally?")
        if max(dbf) < 1.0:
            print(f"  YES! worst dbar_fresh = {max(dbf):.6f} < 1")
            print(f"  This means: even ignoring barrier amplification,")
            print(f"  the raw trace budget tr(Q)/(eps*|A_t|) < 1.")
        else:
            print(f"  NO. worst dbar_fresh = {max(dbf):.6f} >= 1")

        # Show instances where norm_M > 0 (nontrivial barrier)
        has_barrier = [r for r in nontrivial if r["norm_M"] > 1e-10]
        print(f"\nRows with ||M_t|| > 0: {len(has_barrier)}/{len(nontrivial)}")
        if has_barrier:
            worst_bar = sorted(has_barrier, key=lambda r: r["dbar_actual"], reverse=True)[:5]
            print("Worst dbar_actual with ||M_t|| > 0:")
            for r in worst_bar:
                print(f"  {r['graph']:<20} n={r['n']:>3} eps={r['eps']:.2f} t={r['t']:>2} "
                      f"dbar={r['dbar_actual']:.4f} fresh={r['dbar_fresh']:.4f} "
                      f"scalar={r['dbar_scalar']:.4f} ||M||={r['norm_M']:.4f} "
                      f"aniso={r['anisotropy']:.4f}")


if __name__ == "__main__":
    main()
