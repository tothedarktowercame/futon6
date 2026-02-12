#!/usr/bin/env python3
"""
Direction A probe for GPL-H (Strongly Rayleigh on edge indicators).

Goal: test whether SR-like sampling on cross edges (between S_t and R_t)
can transfer to useful control of deterministic grouped star loads Y_t(v).

Distributions tested:
1) Product Bernoulli p=0.5  (strongly Rayleigh)
2) Product Bernoulli p=1/deg_R(v) per edge at R-side endpoint (SR)
3) Uniform spanning forest on cross graph components via Wilson (SR for trees)

This is a diagnostic script, not a proof.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
import numpy as np


# ---------- Graph generators ----------

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


def erdos_renyi(n, p, rng):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((i, j))
    return edges


def random_regular(n, d, rng):
    edges_set = set()
    for _ in range(d * n * 5):
        u, v = rng.integers(0, n, size=2)
        if u != v:
            edges_set.add((min(u, v), max(u, v)))
    adj = [0] * n
    kept = []
    for u, v in sorted(edges_set):
        if adj[u] < d and adj[v] < d:
            kept.append((u, v))
            adj[u] += 1
            adj[v] += 1
    return kept


def dumbbell_graph(k):
    edges = []
    for i in range(k):
        for j in range(i + 1, k):
            edges.append((i, j))
    bridge_start = k
    edges.append((k - 1, bridge_start))
    edges.append((bridge_start, bridge_start + 1))
    for i in range(bridge_start + 1, bridge_start + 1 + k):
        for j in range(i + 1, bridge_start + 1 + k):
            edges.append((i, j))
    return bridge_start + 1 + k, edges


def disjoint_cliques(k, num_cliques):
    n = k * num_cliques
    edges = []
    for c in range(num_cliques):
        base = c * k
        for i in range(k):
            for j in range(i + 1, k):
                edges.append((base + i, base + j))
    for c in range(num_cliques - 1):
        edges.append((c * k + k - 1, (c + 1) * k))
    return n, edges


# ---------- Spectral primitives ----------

def graph_laplacian(n, edges, weights=None):
    L = np.zeros((n, n))
    if weights is None:
        weights = [1.0] * len(edges)
    for idx, (u, v) in enumerate(edges):
        w = weights[idx]
        L[u, u] += w
        L[v, v] += w
        L[u, v] -= w
        L[v, u] -= w
    return L


def pseudo_sqrt_inv(L):
    eigvals, eigvecs = np.linalg.eigh(L)
    d = len(eigvals)
    Lphalf = np.zeros((d, d))
    for i in range(d):
        if eigvals[i] > 1e-10:
            Lphalf += (1.0 / np.sqrt(eigvals[i])) * np.outer(eigvecs[:, i], eigvecs[:, i])
    return Lphalf


def compute_edge_matrices(n, edges, Lphalf, weights=None):
    if weights is None:
        weights = [1.0] * len(edges)
    X_edges = []
    taus = []
    for idx, (u, v) in enumerate(edges):
        w = weights[idx]
        b = np.zeros(n)
        b[u] = 1.0
        b[v] = -1.0
        z = Lphalf @ b
        Xe = w * np.outer(z, z)
        tau = w * np.dot(z, z)
        X_edges.append(Xe)
        taus.append(tau)
    return X_edges, taus


@dataclass
class Case2bInstance:
    n: int
    edges: list
    epsilon: float
    I0: list
    X_edges: list
    taus: list
    alpha_I: float
    graph_name: str = ""


def find_case2b_instance(n, edges, epsilon, graph_name=""):
    L = graph_laplacian(n, edges)
    Lphalf = pseudo_sqrt_inv(L)
    X_edges, taus = compute_edge_matrices(n, edges, Lphalf)

    heavy_edges_idx = [i for i, t in enumerate(taus) if t > epsilon]
    adj_heavy = [set() for _ in range(n)]
    for idx in heavy_edges_idx:
        u, v = edges[idx]
        adj_heavy[u].add(v)
        adj_heavy[v].add(u)

    I_set = set()
    vertices = list(range(n))
    np.random.shuffle(vertices)
    for v in vertices:
        if all(u not in I_set for u in adj_heavy[v]):
            I_set.add(v)

    if len(I_set) < epsilon * n / 3 * 0.8:
        return None

    I = sorted(I_set)

    edge_in_I = []
    for idx, (u, v) in enumerate(edges):
        if u in I_set and v in I_set:
            edge_in_I.append(idx)

    if not edge_in_I:
        return None

    M_I = sum(X_edges[idx] for idx in edge_in_I)
    alpha_I = np.linalg.norm(M_I, ord=2)
    if alpha_I <= epsilon:
        return None

    ell = {}
    for v in I:
        ell_v = 0.0
        for idx, (u, w) in enumerate(edges):
            if (u == v and w in I_set) or (w == v and u in I_set):
                ell_v += taus[idx]
        ell[v] = ell_v

    T_I = sum(taus[idx] for idx in edge_in_I)
    D_bound = 4 * T_I / len(I) if len(I) > 0 else 0
    I0 = [v for v in I if ell[v] <= max(D_bound * 2, 12 / epsilon)]
    if len(I0) < len(I) / 3:
        I0 = I

    return Case2bInstance(
        n=n,
        edges=edges,
        epsilon=epsilon,
        I0=I0,
        X_edges=X_edges,
        taus=taus,
        alpha_I=alpha_I,
        graph_name=graph_name,
    )


# ---------- SR-like sampling on cross edges ----------

def connected_components(n_nodes, adj):
    seen = [False] * n_nodes
    comps = []
    for s in range(n_nodes):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v, _ in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps


def wilson_ust_component(comp_nodes, adj, rng):
    """Uniform spanning tree edge indices on a connected component (Wilson)."""
    comp_set = set(comp_nodes)
    root = comp_nodes[0]
    in_tree = {root}
    selected = set()

    for start in comp_nodes:
        if start in in_tree:
            continue

        nodes = [start]
        edges = []
        pos = {start: 0}

        while nodes[-1] not in in_tree:
            u = nodes[-1]
            nbrs = [(v, eidx) for (v, eidx) in adj[u] if v in comp_set]
            if not nbrs:
                break
            k = rng.integers(0, len(nbrs))
            v, eidx = nbrs[k]

            if v in pos:
                j = pos[v]
                for w in nodes[j + 1:]:
                    pos.pop(w, None)
                nodes = nodes[: j + 1]
                edges = edges[:j]
            else:
                edges.append(eidx)
                nodes.append(v)
                pos[v] = len(nodes) - 1

        for i, eidx in enumerate(edges):
            selected.add(eidx)
            in_tree.add(nodes[i])
            in_tree.add(nodes[i + 1])

    return selected


def sample_spanning_forest(n_nodes, edge_list, rng):
    """
    edge_list: list of (a,b,global_edge_idx)
    returns set of selected global edge idx.
    """
    adj = [[] for _ in range(n_nodes)]
    for a, b, eidx in edge_list:
        adj[a].append((b, eidx))
        adj[b].append((a, eidx))

    selected = set()
    comps = connected_components(n_nodes, adj)
    for comp in comps:
        if len(comp) <= 1:
            continue
        selected |= wilson_ust_component(comp, adj, rng)
    return selected


@dataclass
class StepTransferStats:
    measure: str
    graph: str
    n: int
    eps: float
    t: int
    active_vertices: int
    full_min_active: float
    full_max_active: float
    mean_min_sample_active: float
    q10_min_sample_active: float
    q90_min_sample_active: float
    ratio_fullmin_to_meanmin: float
    max_ratio_full_to_mean_vertex: float
    median_ratio_full_to_mean_vertex: float


def analyze_direction_a_on_instance(inst: Case2bInstance, samples=200, c_step=0.5, seed=0):
    rng = np.random.default_rng(seed)

    n = inst.n
    eps = inst.epsilon
    I0 = inst.I0
    d = n

    edge_idx = {}
    for idx, (u, v) in enumerate(inst.edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx

    m0 = len(I0)
    T = max(1, min(int(c_step * eps * n), m0 - 1))

    S_t = []
    S_set = set()
    M_t = np.zeros((d, d))

    out = []

    for t in range(T):
        R_t = [v for v in I0 if v not in S_set]
        if not R_t:
            break

        headroom = eps * np.eye(d) - M_t
        if np.min(np.linalg.eigvalsh(headroom)) < 1e-12:
            break

        B_t = np.linalg.inv(headroom)
        Bsqrt = np.linalg.cholesky(B_t + 1e-14 * np.eye(d))

        # Build cross-edge atoms A_e^{(t)} and star mapping by R-side vertex
        cross_edges = []  # tuples: (uS, vR, global_idx, A)
        by_v = {v: [] for v in R_t}

        for v in R_t:
            for u in S_t:
                key = (min(u, v), max(u, v))
                if key in edge_idx:
                    gidx = edge_idx[key]
                    A = Bsqrt @ inst.X_edges[gidx] @ Bsqrt.T
                    cross_edges.append((u, v, gidx, A))
                    by_v[v].append(len(cross_edges) - 1)

        # Full deterministic scores
        full_scores = {}
        for v in R_t:
            C = np.zeros((d, d))
            for ci in by_v[v]:
                C += cross_edges[ci][3]
            full_scores[v] = float(np.linalg.norm(C, ord=2))

        active_vs = [v for v in R_t if full_scores[v] > 1e-10]

        # If no active vertices yet, skip transfer diagnostics but continue greedy
        if len(active_vs) >= 2 and len(cross_edges) > 0:
            # local node map for cross graph (S_t union active R_t)
            S_nodes = sorted(set(u for u in S_t))
            R_nodes = sorted(active_vs)
            local_nodes = S_nodes + R_nodes
            node_id = {x: i for i, x in enumerate(local_nodes)}

            # map local-edge index -> (R endpoint, A)
            active_cross = []
            for u, v, gidx, A in cross_edges:
                if v in active_vs:
                    active_cross.append((u, v, gidx, A))

            # For measure construction
            degR = {v: 0 for v in active_vs}
            for _, v, _, _ in active_cross:
                degR[v] += 1

            measure_data = {
                "bern_p50": [],
                "bern_deg": [],
                "ust_forest": [],
            }

            # Per-vertex sampled score accumulators
            acc = {
                name: {v: [] for v in active_vs}
                for name in measure_data
            }

            for _ in range(samples):
                # 1) Bernoulli p=0.5
                sel1 = set()
                for i, _e in enumerate(active_cross):
                    if rng.random() < 0.5:
                        sel1.add(i)

                # 2) Bernoulli p=1/deg_R(v)
                sel2 = set()
                for i, (_u, v, _g, _A) in enumerate(active_cross):
                    p = 1.0 / max(1, degR[v])
                    if rng.random() < p:
                        sel2.add(i)

                # 3) UST forest on cross graph (components)
                edge_list_local = []
                for i, (u, v, _g, _A) in enumerate(active_cross):
                    edge_list_local.append((node_id[u], node_id[v], i))
                sel3 = sample_spanning_forest(len(local_nodes), edge_list_local, rng)

                chosen = {
                    "bern_p50": sel1,
                    "bern_deg": sel2,
                    "ust_forest": sel3,
                }

                for name, sel in chosen.items():
                    min_score = float("inf")
                    for v in active_vs:
                        C = np.zeros((d, d))
                        for i, (_u, vv, _g, A) in enumerate(active_cross):
                            if vv == v and i in sel:
                                C += A
                        s = float(np.linalg.norm(C, ord=2))
                        acc[name][v].append(s)
                        if s < min_score:
                            min_score = s
                    if min_score == float("inf"):
                        min_score = 0.0
                    measure_data[name].append(min_score)

            # Summarize each measure
            full_min_active = min(full_scores[v] for v in active_vs)
            full_max_active = max(full_scores[v] for v in active_vs)

            for name in ["bern_p50", "bern_deg", "ust_forest"]:
                mins = np.array(measure_data[name], dtype=float)
                mean_min = float(np.mean(mins))

                mean_by_v = {
                    v: float(np.mean(acc[name][v]))
                    for v in active_vs
                }

                ratios = [
                    full_scores[v] / max(1e-9, mean_by_v[v])
                    for v in active_vs
                ]

                out.append(
                    StepTransferStats(
                        measure=name,
                        graph=inst.graph_name,
                        n=inst.n,
                        eps=inst.epsilon,
                        t=t,
                        active_vertices=len(active_vs),
                        full_min_active=full_min_active,
                        full_max_active=full_max_active,
                        mean_min_sample_active=mean_min,
                        q10_min_sample_active=float(np.percentile(mins, 10)),
                        q90_min_sample_active=float(np.percentile(mins, 90)),
                        ratio_fullmin_to_meanmin=full_min_active / max(1e-9, mean_min),
                        max_ratio_full_to_mean_vertex=float(np.max(ratios)),
                        median_ratio_full_to_mean_vertex=float(np.median(ratios)),
                    )
                )

        # Greedy update (same as baseline): pick v with minimum full score
        best_v = min(R_t, key=lambda v: full_scores[v]) if R_t else None
        if best_v is None:
            break

        S_t.append(best_v)
        S_set.add(best_v)
        for u in S_t[:-1]:
            key = (min(best_v, u), max(best_v, u))
            if key in edge_idx:
                M_t += inst.X_edges[edge_idx[key]]

    return out


def build_suite(nmax, rng):
    suite = []
    for n in range(8, nmax + 1, 4):
        suite.append((f"K_{n}", n, complete_graph(n)))
        suite.append((f"C_{n}", n, cycle_graph(n)))
        if n >= 8:
            k = n // 2
            bn, be = barbell_graph(k)
            suite.append((f"Barbell_{k}", bn, be))
        if n >= 12:
            dn, de = dumbbell_graph(n // 3)
            suite.append((f"Dumbbell_{n//3}", dn, de))
            cn, ce = disjoint_cliques(n // 3, 3)
            suite.append((f"DisjCliq_{n//3}x3", cn, ce))
        for p_er in [0.3, 0.5]:
            e = erdos_renyi(n, p_er, rng)
            if len(e) > n:
                suite.append((f"ER_{n}_p{p_er}", n, e))
        for d_rr in [4, 6]:
            if d_rr < n:
                e = random_regular(n, d_rr, rng)
                if len(e) > n:
                    suite.append((f"RandReg_{n}_d{d_rr}", n, e))
    return suite


def summarize(stats):
    if not stats:
        print("No Direction A transfer stats collected.")
        return

    print("\nGlobal transfer diagnostics by measure:")
    for measure in ["bern_p50", "bern_deg", "ust_forest"]:
        rows = [s for s in stats if s.measure == measure]
        if not rows:
            continue

        r1 = np.array([s.ratio_fullmin_to_meanmin for s in rows])
        r2 = np.array([s.max_ratio_full_to_mean_vertex for s in rows])
        r3 = np.array([s.median_ratio_full_to_mean_vertex for s in rows])
        mmin = np.array([s.mean_min_sample_active for s in rows])
        nz = mmin > 1e-6

        print(f"\n[{measure}]")
        print(f"  rows: {len(rows)}")
        print(f"  rows with E[min sampled] > 1e-6: {int(np.sum(nz))}/ {len(rows)}")
        print(f"  ratio full_min / E[min sampled]: median={np.median(r1):.4f}, "
              f"90%={np.percentile(r1,90):.4f}, max={np.max(r1):.4f}")
        if np.any(nz):
            print(f"    (nondegenerate rows only) median={np.median(r1[nz]):.4f}, "
                  f"90%={np.percentile(r1[nz],90):.4f}, max={np.max(r1[nz]):.4f}")
        print(f"  max_v full_score_v / E[sampled_score_v]: median={np.median(r2):.4f}, "
              f"90%={np.percentile(r2,90):.4f}, max={np.max(r2):.4f}")
        print(f"  median_v full_score_v / E[sampled_score_v]: median={np.median(r3):.4f}, "
              f"90%={np.percentile(r3,90):.4f}, max={np.max(r3):.4f}")

        worst = sorted(rows, key=lambda s: s.max_ratio_full_to_mean_vertex, reverse=True)[:10]
        print("  top-10 worst step ratios:")
        for w in worst:
            print(
                f"    {w.graph:<18} n={w.n:>2} eps={w.eps:>4.2f} t={w.t:>2} "
                f"active={w.active_vertices:>2} "
                f"full_min={w.full_min_active:.4f} E[min]={w.mean_min_sample_active:.4f} "
                f"max_ratio={w.max_ratio_full_to_mean_vertex:.4f}"
            )


def main():
    ap = argparse.ArgumentParser(description="Direction A probe for GPL-H")
    ap.add_argument("--nmax", type=int, default=20)
    ap.add_argument("--eps", nargs="+", type=float,
                    default=[0.12, 0.15, 0.2, 0.25, 0.3])
    ap.add_argument("--c-step", type=float, default=0.5)
    ap.add_argument("--samples", type=int, default=160)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-instances", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    suite = build_suite(args.nmax, rng)

    all_stats = []
    n_case2b = 0

    for graph_name, n, edges in suite:
        for eps in args.eps:
            inst = find_case2b_instance(n, edges, eps, graph_name=graph_name)
            if inst is None:
                continue

            n_case2b += 1
            seed = int(rng.integers(0, 2**31 - 1))
            stats = analyze_direction_a_on_instance(
                inst,
                samples=args.samples,
                c_step=args.c_step,
                seed=seed,
            )
            all_stats.extend(stats)

            if args.limit_instances > 0 and n_case2b >= args.limit_instances:
                break
        if args.limit_instances > 0 and n_case2b >= args.limit_instances:
            break

    print("=" * 78)
    print("DIRECTION A PROBE: SR edge sampling -> vertex-star transfer")
    print("=" * 78)
    print(f"Case-2b instances analyzed: {n_case2b}")
    print(f"Step-level transfer rows:   {len(all_stats)}")

    summarize(all_stats)


if __name__ == "__main__":
    main()
