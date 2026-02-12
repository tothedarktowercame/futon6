#!/usr/bin/env python3
"""Track M_t growth during barrier greedy to verify boot-strap hypothesis.

Key question: how does ||M_t|| grow with t under the leverage-aware greedy?
If ||M_t|| stays small relative to ε, then the amplified dbar bound
C_lev·t/((ε-||M_t||)·r_t) stays < 1 throughout.

We track:
- ||M_t|| at each step
- The "effective barrier gap" ε - ||M_t||
- The amplified dbar: C_lev·t/((ε-||M_t||)·r_t)
- The log-determinant potential Φ(t) = log det(εI - M_t)
"""

import numpy as np
import sys
import importlib.util
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("p6base", repo_root / "scripts" / "verify-p6-gpl-h.py")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


def complete_bipartite(t, r):
    n = t + r
    edges = [(i, j) for i in range(t) for j in range(t, n)]
    return n, edges


def track_mt_growth(n, edges, eps, C_lev=2.0, c_step=1/3):
    """Run barrier greedy and track M_t growth at each step."""
    L = base.graph_laplacian(n, edges)
    Lphalf = base.pseudo_sqrt_inv(L)
    X_edges, taus = base.compute_edge_matrices(n, edges, Lphalf)

    # Build I0 with leverage constraint
    heavy_adj = [set() for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        if taus[idx] > eps:
            heavy_adj[u].add(v)
            heavy_adj[v].add(u)

    I_set = set()
    vertices = list(range(n))
    np.random.shuffle(vertices)
    for v in vertices:
        if all(u not in I_set for u in heavy_adj[v]):
            I_set.add(v)

    ell_I = {}
    for v in I_set:
        ell_v = sum(taus[idx] for idx, (u, w) in enumerate(edges)
                    if (u == v and w in I_set) or (w == v and u in I_set))
        ell_I[v] = ell_v

    lev_bound = C_lev / eps
    I0 = sorted(v for v in I_set if ell_I[v] <= lev_bound)

    if len(I0) < 3:
        return None

    I0_set = set(I0)
    internal = [(u, v) for u, v in edges if u in I0_set and v in I0_set]
    if not internal:
        return None

    M_I = sum(X_edges[idx] for idx, (u, v) in enumerate(edges)
              if u in I0_set and v in I0_set)
    if np.linalg.norm(M_I, ord=2) <= eps:
        return None  # Case 2a

    edge_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx

    I0_adj = {v: set() for v in I0}
    for u, v in edges:
        if u in I0_set and v in I0_set:
            I0_adj[u].add(v)
            I0_adj[v].add(u)

    m0 = len(I0)
    T = max(1, min(int(c_step * eps * n), m0 - 1))

    S_t = []
    S_set = set()
    M_t = np.zeros((n, n))
    steps = []

    for t in range(T):
        R_t = [v for v in I0 if v not in S_set]
        r_t = len(R_t)
        if r_t == 0:
            break

        mt_norm = float(np.linalg.norm(M_t, ord=2))
        gap = eps - mt_norm

        if gap < 1e-12:
            break

        # Log-det potential
        eigvals = np.linalg.eigvalsh(eps * np.eye(n) - M_t)
        phi = float(np.sum(np.log(np.maximum(eigvals, 1e-300))))

        headroom = eps * np.eye(n) - M_t
        B_t = np.linalg.inv(headroom)
        Bsqrt = np.linalg.cholesky(B_t + 1e-14 * np.eye(n))

        # Compute scores (among eligible low-leverage vertices)
        eligible = [v for v in R_t if ell_I.get(v, 999) <= lev_bound]
        scores = {}
        for v in R_t:
            C_v = np.zeros((n, n))
            for u in S_t:
                key = (min(u, v), max(u, v))
                if key in edge_idx:
                    C_v += X_edges[edge_idx[key]]
            Y_v = Bsqrt @ C_v @ Bsqrt.T
            scores[v] = float(np.linalg.norm(Y_v, ord=2))

        min_score_all = min(scores.values())

        # dbar computation
        dbar_vals = []
        for v in R_t:
            C_v = np.zeros((n, n))
            for u in S_t:
                key = (min(u, v), max(u, v))
                if key in edge_idx:
                    C_v += X_edges[edge_idx[key]]
            Y_v = Bsqrt @ C_v @ Bsqrt.T
            dbar_vals.append(float(np.trace(Y_v)))
        dbar = np.mean(dbar_vals) if dbar_vals else 0

        # Leverage sum over S_t
        lev_sum_st = sum(ell_I.get(u, 0) for u in S_t)

        # Amplified bound: C_lev * t / ((eps - ||M_t||) * r_t)
        amplified_dbar = C_lev * t / (gap * r_t) if gap > 0 and r_t > 0 else float('inf')

        # Zero-score count
        zero_count = sum(1 for v in R_t if scores[v] < 1e-14)
        is_phase2 = (zero_count == 0 and t > 0)

        steps.append({
            "t": t,
            "r_t": r_t,
            "phase": 2 if is_phase2 else 1,
            "mt_norm": mt_norm,
            "gap": gap,
            "phi": phi,
            "min_score": min_score_all,
            "dbar": dbar,
            "lev_sum_st": lev_sum_st,
            "amplified_dbar": amplified_dbar,
        })

        # Select best eligible vertex
        elig_scores = {v: scores[v] for v in eligible}
        if elig_scores:
            best_v = min(elig_scores, key=elig_scores.get)
        elif R_t:
            best_v = min(scores, key=scores.get)
        else:
            break

        S_t.append(best_v)
        S_set.add(best_v)
        for u in S_t[:-1]:
            key = (min(best_v, u), max(best_v, u))
            if key in edge_idx:
                M_t += X_edges[edge_idx[key]]

    return {"steps": steps, "m0": m0, "T": T, "I0_size": len(I0)}


def main():
    np.random.seed(42)

    print("=" * 70)
    print("M_t GROWTH TRACKING DURING LEVERAGE-AWARE BARRIER GREEDY")
    print("=" * 70)

    C_lev = 2.0
    rng = np.random.default_rng(42)

    # Build test suite including K_{t,r} and standard graphs
    test_graphs = []

    # Standard graphs
    for n in range(8, 65, 4):
        test_graphs.append((f"K_{n}", n, base.complete_graph(n)))
        test_graphs.append((f"C_{n}", n, base.cycle_graph(n)))
        if n >= 8:
            k = n // 2
            bn, be = base.barbell_graph(k)
            test_graphs.append((f"Barbell_{k}", bn, be))
        if n >= 12:
            dn, de = base.dumbbell_graph(n // 3)
            test_graphs.append((f"Dumbbell_{n//3}", dn, de))
            cn, ce = base.disjoint_cliques(n // 3, 3)
            test_graphs.append((f"DisjCliq_{n//3}x3", cn, ce))
        for p_er in [0.3, 0.5]:
            er_edges = base.erdos_renyi(n, p_er, rng)
            if len(er_edges) > n:
                test_graphs.append((f"ER_{n}_p{p_er}", n, er_edges))

    # K_{t,r}
    for t_val, r_val in [(2, 20), (3, 30), (5, 30), (5, 50)]:
        n, edges = complete_bipartite(t_val, r_val)
        test_graphs.append((f"K_{{{t_val},{r_val}}}", n, edges))

    eps_list = [0.1, 0.12, 0.15, 0.2, 0.25, 0.3]

    all_steps = []
    total_instances = 0
    max_mt_norm = 0.0
    max_amplified_dbar = 0.0
    worst_graph = ""

    for graph_name, n, edges in test_graphs:
        for eps in eps_list:
            res = track_mt_growth(n, edges, eps, C_lev=C_lev)
            if res is None:
                continue
            if not res["steps"]:
                continue

            total_instances += 1
            for s in res["steps"]:
                s["graph"] = graph_name
                s["n"] = n
                s["eps"] = eps

            all_steps.extend(res["steps"])

            # Track worst M_t norm
            max_mt_this = max(s["mt_norm"] for s in res["steps"])
            if max_mt_this > max_mt_norm:
                max_mt_norm = max_mt_this
                worst_graph = f"{graph_name} eps={eps}"

            max_amp = max(s["amplified_dbar"] for s in res["steps"]
                         if s["amplified_dbar"] < float('inf'))
            if max_amp > max_amplified_dbar:
                max_amplified_dbar = max_amp

    print(f"\nTotal Case-2b instances: {total_instances}")
    print(f"Total steps: {len(all_steps)}")

    # Phase 2 analysis
    p2_steps = [s for s in all_steps if s["phase"] == 2]
    print(f"Phase 2 steps: {len(p2_steps)}")

    if p2_steps:
        mt_norms = [s["mt_norm"] for s in p2_steps]
        gaps = [s["gap"] for s in p2_steps]
        dbars = [s["dbar"] for s in p2_steps]
        amp_dbars = [s["amplified_dbar"] for s in p2_steps
                     if s["amplified_dbar"] < float('inf')]
        min_scores = [s["min_score"] for s in p2_steps]

        print(f"\n--- Phase 2 ||M_t|| ---")
        print(f"  min  = {min(mt_norms):.6f}")
        print(f"  mean = {np.mean(mt_norms):.6f}")
        print(f"  max  = {max(mt_norms):.6f}")
        print(f"  max / eps: {max(s['mt_norm']/s['eps'] for s in p2_steps):.4f}")

        print(f"\n--- Phase 2 barrier gap (ε - ||M_t||) ---")
        print(f"  min  = {min(gaps):.6f}")
        print(f"  mean = {np.mean(gaps):.6f}")
        print(f"  max  = {max(gaps):.6f}")
        print(f"  min gap / eps: {min(s['gap']/s['eps'] for s in p2_steps):.4f}")

        print(f"\n--- Phase 2 actual dbar ---")
        print(f"  min  = {min(dbars):.6f}")
        print(f"  mean = {np.mean(dbars):.6f}")
        print(f"  max  = {max(dbars):.6f}")

        if amp_dbars:
            print(f"\n--- Phase 2 amplified dbar bound ---")
            print(f"  min  = {min(amp_dbars):.6f}")
            print(f"  mean = {np.mean(amp_dbars):.6f}")
            print(f"  max  = {max(amp_dbars):.6f}")

        print(f"\n--- Phase 2 min score ---")
        print(f"  max  = {max(min_scores):.6f}")
        violations = sum(1 for s in min_scores if s >= 1.0 - 1e-10)
        print(f"  violations: {violations}/{len(p2_steps)}")

    # Log-det potential analysis
    print(f"\n--- Log-det potential Φ(t) = log det(εI - M_t) ---")
    # Show trajectory for a few instances
    seen = set()
    shown = 0
    for s in all_steps:
        key = (s["graph"], s["n"], s["eps"])
        if key in seen:
            continue
        if s["phase"] == 2 and shown < 8:
            seen.add(key)
            # Get full trajectory for this instance
            traj = [s2 for s2 in all_steps if (s2["graph"], s2["n"], s2["eps"]) == key]
            print(f"\n  {s['graph']} n={s['n']} eps={s['eps']:.2f}:")
            for st in traj:
                phase_mark = "P2" if st["phase"] == 2 else "P1"
                print(f"    t={st['t']:>2} {phase_mark} ||M||={st['mt_norm']:.4f} "
                      f"gap={st['gap']:.4f} Φ={st['phi']:.2f} "
                      f"dbar={st['dbar']:.4f} score={st['min_score']:.4f}"
                      f"{' !!!' if st['min_score'] >= 1.0 else ''}")
            shown += 1

    # Summary statistics: ||M_t||/ε vs t/T
    print(f"\n--- ||M_t||/ε as fraction of greedy horizon ---")
    for frac in [0.25, 0.5, 0.75, 1.0]:
        relevant = [s for s in all_steps
                    if s["t"] > 0 and s["t"] / max(1, int(s["eps"] * s["n"] / 3)) >= frac - 0.05
                    and s["t"] / max(1, int(s["eps"] * s["n"] / 3)) < frac + 0.05]
        if relevant:
            ratios = [s["mt_norm"] / s["eps"] for s in relevant]
            print(f"  t/T ≈ {frac:.2f}: ||M||/ε mean={np.mean(ratios):.4f} "
                  f"max={max(ratios):.4f} (n={len(relevant)} steps)")

    print(f"\n--- Overall worst cases ---")
    print(f"  max ||M_t||: {max_mt_norm:.6f} ({worst_graph})")
    print(f"  max amplified dbar: {max_amplified_dbar:.6f}")
    print(f"  Any GPL-H' violations: {'YES' if any(s['min_score'] >= 1.0 - 1e-10 for s in all_steps) else 'NO'}")


if __name__ == "__main__":
    main()
