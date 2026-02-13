#!/usr/bin/env python3
"""Problem 6 amplification diagnostics.

Evaluates candidate bounds for the amplification ratio

  amp_t := dbar_t / dbar_t^{M=0}

where
  dbar_t     = tr((eps I - M_t)^{-1} A_t)
  dbar_t^{M=0} = tr(A_t)/eps
  x_t        = ||M_t||/eps.

Candidate inequalities tested:
  1) amp_t <= 1/(1-x_t)               (crude spectral bound)
  2) amp_t <= 1 + x_t                 (empirical candidate)
  3) amp_t <= 1 + x_t/(2(1-x_t))      (K_n-matching candidate)

Also reports first Neumann coefficient
  rho1_t := tr(M_t A_t) / (||M_t|| tr(A_t)).
"""

import argparse
import random
import sys
import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p6base", REPO_ROOT / "scripts" / "verify-p6-gpl-h.py"
)
P6 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = P6
SPEC.loader.exec_module(P6)


def greedy_rows(n, edges, eps, rng_np, rng_py):
    """Run min-||Y|| greedy and collect amplification diagnostics by step."""
    L = P6.graph_laplacian(n, edges)
    Lph = P6.pseudo_sqrt_inv(L)
    X_edges, taus = P6.compute_edge_matrices(n, edges, Lph)

    heavy_adj = [set() for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        if taus[idx] > eps:
            heavy_adj[u].add(v)
            heavy_adj[v].add(u)

    order = list(range(n))
    rng_py.shuffle(order)
    I0, I0_set = [], set()
    for v in order:
        if all(u not in I0_set for u in heavy_adj[v]):
            I0.append(v)
            I0_set.add(v)

    if len(I0) < 4:
        return []

    internal_idx = [
        idx for idx, (u, v) in enumerate(edges) if u in I0_set and v in I0_set
    ]
    if not internal_idx:
        return []

    M_I0 = np.zeros((n, n))
    for idx in internal_idx:
        M_I0 += X_edges[idx]

    # Focus on Case-2b-like nontrivial instances.
    if np.linalg.norm(M_I0, ord=2) <= eps:
        return []

    edge_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx

    T = max(2, min(int(eps * n / 3), len(I0) - 1))

    S_t, S_set = [], set()
    M_t = np.zeros((n, n))
    rows = []

    for t in range(T):
        R_t = [v for v in I0 if v not in S_set]
        if not R_t:
            break

        H_t = eps * np.eye(n) - M_t
        if np.min(np.linalg.eigvalsh(H_t)) < 1e-12:
            break

        Hinv = np.linalg.inv(H_t)
        Hsqrt_inv = np.linalg.cholesky(Hinv + 1e-14 * np.eye(n))

        norms = []
        C_list = []
        trY_list = []

        for v in R_t:
            C_v = np.zeros((n, n))
            for u in S_t:
                key = (min(u, v), max(u, v))
                if key in edge_idx:
                    C_v += X_edges[edge_idx[key]]

            Y_v = Hsqrt_inv @ C_v @ Hsqrt_inv.T
            Y_v = (Y_v + Y_v.T) / 2

            C_list.append(C_v)
            trY_list.append(float(np.trace(Y_v)))
            norms.append(float(np.linalg.norm(Y_v, ord=2)))

        A_t = np.zeros((n, n))
        for C_v in C_list:
            A_t += C_v
        A_t /= len(C_list)

        dbar = float(np.mean(trY_list))
        trA = float(np.trace(A_t))
        dbar_m0 = trA / eps if trA > 1e-16 else 0.0

        M_norm = float(np.linalg.norm(M_t, ord=2))
        x = M_norm / eps

        amp = dbar / dbar_m0 if dbar_m0 > 1e-16 else 1.0

        b_crude = 1.0 / (1.0 - x) if x < 1 else float("inf")
        b_lin = 1.0 + x
        b_half = 1.0 + x / (2.0 * (1.0 - x)) if x < 1 else float("inf")

        rho1 = None
        if trA > 1e-16 and M_norm > 1e-16:
            rho1 = float(np.trace(M_t @ A_t) / (M_norm * trA))

        rows.append(
            {
                "n": n,
                "eps": eps,
                "t": t,
                "dbar": dbar,
                "dbar_m0": dbar_m0,
                "amp": amp,
                "x": x,
                "b_crude": b_crude,
                "b_lin": b_lin,
                "b_half": b_half,
                "rho1": rho1,
                "r_t": len(R_t),
            }
        )

        # Continue with min-||Y|| step.
        best_idx = int(np.argmin(norms))
        best_v = R_t[best_idx]
        S_t.append(best_v)
        S_set.add(best_v)

        for u in S_t[:-1]:
            key = (min(u, best_v), max(u, best_v))
            if key in edge_idx:
                M_t += X_edges[edge_idx[key]]

    return rows


def run_suite(seed, nmax):
    rng_np = np.random.default_rng(seed)
    rng_py = random.Random(seed)

    graphs = []
    for n in range(16, nmax + 1, 8):
        graphs.append((f"K_{n}", n, P6.complete_graph(n)))
        graphs.append((f"C_{n}", n, P6.cycle_graph(n)))

        if n >= 16:
            k = n // 2
            bn, be = P6.barbell_graph(k)
            graphs.append((f"Barbell_{k}", bn, be))

        if n >= 24:
            cn, ce = P6.disjoint_cliques(n // 3, 3)
            graphs.append((f"DisjCliq_{n//3}x3", cn, ce))

        for p in [0.2, 0.35, 0.5, 0.7]:
            er = P6.erdos_renyi(n, p, rng_np)
            if len(er) > n:
                graphs.append((f"ER_{n}_p{p}", n, er))

        for d in [4, 6, 8]:
            if d < n:
                rr = P6.random_regular(n, d, rng_np)
                if len(rr) > n:
                    graphs.append((f"RandReg_{n}_d{d}", n, rr))

    rows = []
    for gname, n, edges in graphs:
        for eps in [0.12, 0.15, 0.2, 0.25, 0.3, 0.4]:
            out = greedy_rows(n, edges, eps, rng_np, rng_py)
            for r in out:
                r["graph"] = gname
            rows.extend(out)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--nmax", type=int, default=80)
    args = ap.parse_args()

    rows = run_suite(args.seed, args.nmax)

    if not rows:
        print("No nontrivial Case-2b-like rows produced.")
        return

    amps = np.array([r["amp"] for r in rows])
    xs = np.array([r["x"] for r in rows])

    viol_crude = [r for r in rows if r["amp"] > r["b_crude"] + 1e-9]
    viol_lin = [r for r in rows if r["amp"] > r["b_lin"] + 1e-9]
    viol_half = [
        r
        for r in rows
        if r["x"] < 1 and r["amp"] > r["b_half"] + 1e-9
    ]

    rho1_vals = np.array([r["rho1"] for r in rows if r["rho1"] is not None])

    worst_amp = max(rows, key=lambda r: r["amp"])
    worst_x = max(rows, key=lambda r: r["x"])

    print("AMPLIFICATION CANDIDATE DIAGNOSTICS")
    print("=" * 78)
    print(f"rows: {len(rows)}")
    print(f"max amp: {worst_amp['amp']:.6f} at {worst_amp['graph']} eps={worst_amp['eps']} t={worst_amp['t']}")
    print(f"max x=||M||/eps: {worst_x['x']:.6f} at {worst_x['graph']} eps={worst_x['eps']} t={worst_x['t']}")
    print()
    print(f"violations amp <= 1/(1-x): {len(viol_crude)}")
    print(f"violations amp <= 1+x: {len(viol_lin)}")
    print(f"violations amp <= 1 + x/(2(1-x)): {len(viol_half)}")

    ratio_lin = amps / (1 + xs)
    print()
    print(f"amp/(1+x): max={np.max(ratio_lin):.6f}, p99={np.percentile(ratio_lin,99):.6f}, mean={np.mean(ratio_lin):.6f}")

    if len(rho1_vals) > 0:
        print()
        print("rho1 := tr(MA)/(||M|| tr(A)) stats")
        print(f"min={np.min(rho1_vals):.6f}, median={np.median(rho1_vals):.6f}, p99={np.percentile(rho1_vals,99):.6f}, max={np.max(rho1_vals):.6f}")
        print(f"count(rho1 > 0.5)={np.sum(rho1_vals > 0.5 + 1e-9)}")

    print("\nTop 12 amplification rows:")
    top = sorted(rows, key=lambda r: r["amp"], reverse=True)[:12]
    for r in top:
        print(
            f"  {r['graph']:<18s} eps={r['eps']:.2f} t={r['t']:>2d} "
            f"amp={r['amp']:.3f} x={r['x']:.3f} d0={r['dbar_m0']:.3f} d={r['dbar']:.3f} "
            f"lin={r['b_lin']:.3f} half={r['b_half']:.3f} crude={r['b_crude']:.3f}"
        )


if __name__ == "__main__":
    main()
