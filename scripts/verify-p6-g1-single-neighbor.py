#!/usr/bin/env python3
"""G1 new angle: single-neighbor vertices at Phase 2 entry.

Key insight: at Phase 2 entry, M_t = 0 and S_t is independent. Any vertex v
with exactly 1 S_t-neighbor in I_0 has Y_t(v) = (1/ε) X_{u,v} (rank-1), so
||Y_t(v)|| = τ_{u,v}/ε < 1 by strict H1.

So GPL-H holds at Phase 2 entry IF any single-neighbor vertex exists.

By double counting: if E_cross < 2*r_t, then such a vertex exists.
At t=1: E_cross = deg(u_1) ≤ m_0-1 < 2(m_0-1). ALWAYS holds!

Additional insight: after adding a Phase 2 vertex, its non-neighbors in R_t
become zero-score (Phase 1 resumes). We track Phase 2 "episode" lengths.
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


def analyze_phase2_episodes(inst, c_step):
    """Track Phase 1/Phase 2 transitions and single-neighbor structure."""
    n = inst.n
    eps = inst.epsilon
    I0 = inst.I0
    I0_set = set(I0)
    m0 = len(I0)

    # I0-subgraph adjacency
    I0_adj = {v: set() for v in I0}
    edge_idx = {}
    for idx, (u, v) in enumerate(inst.edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx
        if u in I0_set and v in I0_set:
            I0_adj[u].add(v)
            I0_adj[v].add(u)

    T = max(1, min(int(c_step * eps * n), m0 - 1))
    S_t = []
    S_set = set()
    M_t = np.zeros((n, n))

    step_data = []
    current_phase = 1
    episode_start = None
    episodes = []

    for t in range(T):
        R_t = [v for v in I0 if v not in S_set]
        r_t = len(R_t)
        if r_t == 0:
            break

        headroom = eps * np.eye(n) - M_t
        if np.min(np.linalg.eigvalsh(headroom)) < 1e-12:
            break

        B_t = np.linalg.inv(headroom)
        Bsqrt = np.linalg.cholesky(B_t + 1e-14 * np.eye(n))

        # Count S_t-neighbors for each v in R_t
        zero_score_count = 0
        single_neighbor_count = 0
        multi_neighbor_count = 0
        best_v, best_score = None, float("inf")

        for v in R_t:
            st_neighbors = sum(1 for u in S_t if u in I0_adj[v])
            if st_neighbors == 0:
                zero_score_count += 1
            elif st_neighbors == 1:
                single_neighbor_count += 1
            else:
                multi_neighbor_count += 1

            # Compute actual score
            C_v = np.zeros((n, n))
            for u in S_t:
                key = (min(u, v), max(u, v))
                if key in edge_idx:
                    C_v += inst.X_edges[edge_idx[key]]
            s_v = float(np.linalg.norm(Bsqrt @ C_v @ Bsqrt.T, ord=2))
            if s_v < best_score:
                best_score = s_v
                best_v = v

        # S_t independence check
        s_independent = True
        for i, u1 in enumerate(S_t):
            for u2 in S_t[i+1:]:
                if u2 in I0_adj.get(u1, set()):
                    s_independent = False
                    break
            if not s_independent:
                break

        # Phase determination
        is_phase2 = (zero_score_count == 0 and t > 0)

        # M_t norm
        mt_norm = float(np.linalg.norm(M_t, ord=2))

        # Track episode transitions
        if is_phase2 and current_phase == 1:
            episode_start = t
        if not is_phase2 and current_phase == 2 and episode_start is not None:
            episodes.append({"start": episode_start, "end": t, "length": t - episode_start})
        current_phase = 2 if is_phase2 else 1

        step_data.append({
            "t": t,
            "r_t": r_t,
            "phase": 2 if is_phase2 else 1,
            "zero_score": zero_score_count,
            "single_neighbor": single_neighbor_count,
            "multi_neighbor": multi_neighbor_count,
            "s_independent": s_independent,
            "best_score": best_score,
            "mt_norm": mt_norm,
        })

        if best_v is None:
            break
        S_t.append(best_v)
        S_set.add(best_v)
        for u in S_t[:-1]:
            key = (min(best_v, u), max(best_v, u))
            if key in edge_idx:
                M_t += inst.X_edges[edge_idx[key]]

    # Close final episode
    if current_phase == 2 and episode_start is not None:
        episodes.append({"start": episode_start, "end": T, "length": T - episode_start})

    return step_data, episodes


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=96)
    ap.add_argument("--c-step", type=float, default=1/3)
    ap.add_argument("--seed", type=int, default=20260212)
    args = ap.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    suite = build_suite(args.nmax, rng)
    eps_list = [0.1, 0.12, 0.15, 0.2, 0.25, 0.3]

    all_steps = []
    all_episodes = []
    total_instances = 0

    for graph_name, n, edges in suite:
        for eps in eps_list:
            inst = base.find_case2b_instance(n, edges, eps, graph_name=graph_name)
            if inst is None:
                continue
            total_instances += 1
            steps, episodes = analyze_phase2_episodes(inst, args.c_step)
            for s in steps:
                s["graph"] = graph_name
                s["n"] = n
                s["eps"] = eps
                s["m0"] = len(inst.I0)
            all_steps.extend(steps)
            for ep in episodes:
                ep["graph"] = graph_name
                ep["n"] = n
                ep["eps"] = eps
            all_episodes.extend(episodes)

    print(f"Total Case-2b instances: {total_instances}")
    print(f"Total steps: {len(all_steps)}")

    # Phase 2 steps analysis
    p2_steps = [s for s in all_steps if s["phase"] == 2]
    print(f"\nPhase 2 steps: {len(p2_steps)}")

    if p2_steps:
        # At Phase 2 steps, how many have single-neighbor vertices?
        has_single = sum(1 for s in p2_steps if s["single_neighbor"] > 0)
        print(f"Phase 2 steps with single-neighbor vertices: {has_single}/{len(p2_steps)} "
              f"({100*has_single/len(p2_steps):.1f}%)")

        # Single-neighbor fraction
        single_fracs = [s["single_neighbor"] / s["r_t"] for s in p2_steps if s["r_t"] > 0]
        print(f"Single-neighbor fraction in Phase 2:")
        print(f"  min  = {min(single_fracs):.4f}")
        print(f"  mean = {np.mean(single_fracs):.4f}")
        print(f"  max  = {max(single_fracs):.4f}")

        # S_t independence at Phase 2
        indep_p2 = sum(1 for s in p2_steps if s["s_independent"])
        print(f"\nS_t independent at Phase 2 steps: {indep_p2}/{len(p2_steps)} "
              f"({100*indep_p2/len(p2_steps):.1f}%)")

        # M_t norm at Phase 2
        mt_norms = [s["mt_norm"] for s in p2_steps]
        print(f"\n||M_t|| at Phase 2 steps:")
        print(f"  min  = {min(mt_norms):.6f}")
        print(f"  mean = {np.mean(mt_norms):.6f}")
        print(f"  max  = {max(mt_norms):.6f}")
        print(f"  M_t = 0 (< 1e-10): {sum(1 for m in mt_norms if m < 1e-10)}/{len(mt_norms)}")

        # Phase 2 steps where single-neighbor FAILS
        no_single = [s for s in p2_steps if s["single_neighbor"] == 0]
        if no_single:
            print(f"\nPhase 2 steps with NO single-neighbor vertex: {len(no_single)}")
            for s in no_single[:20]:
                print(f"  {s['graph']:<20} n={s['n']:>3} eps={s['eps']:.2f} t={s['t']:>2} "
                      f"r_t={s['r_t']:>3} zero={s['zero_score']} "
                      f"single={s['single_neighbor']} multi={s['multi_neighbor']} "
                      f"indep={s['s_independent']} ||M||={s['mt_norm']:.4f} "
                      f"score={s['best_score']:.4f}")
        else:
            print(f"\nALL Phase 2 steps have single-neighbor vertices!")

    # Phase 2 episode analysis
    print(f"\n--- Phase 2 Episodes ---")
    print(f"Total episodes: {len(all_episodes)}")
    if all_episodes:
        lengths = [ep["length"] for ep in all_episodes]
        print(f"Episode lengths:")
        print(f"  min  = {min(lengths)}")
        print(f"  mean = {np.mean(lengths):.2f}")
        print(f"  max  = {max(lengths)}")

        # Length distribution
        from collections import Counter
        len_dist = Counter(lengths)
        print(f"  distribution: {dict(sorted(len_dist.items()))}")

        # Long episodes
        long = [ep for ep in all_episodes if ep["length"] > 3]
        if long:
            print(f"\nLong episodes (>3 steps):")
            for ep in sorted(long, key=lambda e: e["length"], reverse=True)[:15]:
                print(f"  {ep['graph']:<20} n={ep['n']:>3} eps={ep['eps']:.2f} "
                      f"start={ep['start']:>2} length={ep['length']:>3}")

    # Phase 2 entry steps (first step of each episode)
    entry_steps = [s for s in all_steps if s["phase"] == 2 and
                   (s["t"] == 0 or any(s2["t"] == s["t"] - 1 and s2["phase"] == 1
                                        for s2 in all_steps if s2["graph"] == s["graph"] and s2["n"] == s["n"] and s2["eps"] == s["eps"]))]
    # Simpler: first Phase 2 step per instance
    seen = set()
    entry_steps = []
    for s in all_steps:
        key = (s["graph"], s["n"], s["eps"])
        if s["phase"] == 2 and key not in seen:
            seen.add(key)
            entry_steps.append(s)

    if entry_steps:
        print(f"\nPhase 2 ENTRY steps: {len(entry_steps)}")
        entry_single = sum(1 for s in entry_steps if s["single_neighbor"] > 0)
        entry_indep = sum(1 for s in entry_steps if s["s_independent"])
        entry_mt0 = sum(1 for s in entry_steps if s["mt_norm"] < 1e-10)
        print(f"  with single-neighbor vertex: {entry_single}/{len(entry_steps)}")
        print(f"  S_t independent: {entry_indep}/{len(entry_steps)}")
        print(f"  M_t = 0: {entry_mt0}/{len(entry_steps)}")
        print(f"\n  => At Phase 2 entry: S_t indep + M_t=0 + single-neighbor → GPL-H by H1")


if __name__ == "__main__":
    main()
