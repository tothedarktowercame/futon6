#!/usr/bin/env python3
"""M-differentiable-math pilot: does negative Ollivier-Ricci curvature mark
meaningful BOTTLENECK concepts on the real math tag graph?

Substrate (verified 2026-05-31): storage/math-processed-gpu/relations.json is a
flat array of {from, to, type} edges; the dominant type is "tagged-with" linking
se-math-<id> questions to se-tag-<name> tags. We build the TAG CO-OCCURRENCE
graph (two tags adjacent iff they co-tag >= MIN_COOCC questions, weighted by
co-occurrence count) and compute Ollivier-Ricci curvature on each edge.

Ollivier-Ricci:  kappa(x,y) = 1 - W1(mu_x, mu_y) / d(x,y)
  - mu_x = neighbour mass of x (here: normalized co-occurrence weights, with a
    lazy self-mass alpha), mu_y likewise.
  - d = shortest-path / hop distance on the tag graph (d(x,y)=1 for an edge).
  - W1 = exact earth-mover distance between mu_x and mu_y over the pairwise
    ground distances, solved as a small LP (scipy.optimize.linprog).
NEGATIVE kappa on an edge => the two tags' neighbourhoods are far apart
(little overlap) => the edge BRIDGES otherwise-separate clusters => a bottleneck
/ connector concept. POSITIVE kappa => the edge sits inside a dense community.

This is the curvature half of C-substrate-completion's ground-metric contract,
tested on data that exists. No build required.

Usage: python3 scripts/ricci_bottleneck_pilot.py [MIN_COOCC] [MAX_TAGS]
"""
import sys, time, pickle, json
from collections import defaultdict, Counter
import numpy as np
from scipy.optimize import linprog

Q2TAGS = "/tmp/q2tags.pkl"   # {question_id: [tag,...]} produced by the parse step
MIN_COOCC = int(sys.argv[1]) if len(sys.argv) > 1 else 30
MAX_TAGS  = int(sys.argv[2]) if len(sys.argv) > 2 else 400
ALPHA = 0.5  # lazy self-mass (standard OR idle-walk parameter)


def build_cooccurrence(q2tags):
    pair = Counter()
    deg = Counter()
    for tags in q2tags.values():
        u = sorted(set(tags))
        for t in u:
            deg[t] += 1
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                pair[(u[i], u[j])] += 1
    return pair, deg


def w1(mu_x_keys, mu_x_w, mu_y_keys, mu_y_w, dist):
    """Exact W1 between two discrete distributions via LP."""
    m, n = len(mu_x_keys), len(mu_y_keys)
    c = np.zeros(m * n)
    for i, a in enumerate(mu_x_keys):
        for j, b in enumerate(mu_y_keys):
            c[i * n + j] = dist(a, b)
    # marginals: rows sum to mu_x, cols sum to mu_y
    A_eq = np.zeros((m + n, m * n))
    b_eq = np.concatenate([mu_x_w, mu_y_w])
    for i in range(m):
        A_eq[i, i * n:(i + 1) * n] = 1.0
    for j in range(n):
        A_eq[m + j, j::n] = 1.0
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    return res.fun if res.success else None


def main():
    t0 = time.time()
    q2tags = pickle.load(open(Q2TAGS, "rb"))
    pair, deg = build_cooccurrence(q2tags)
    # keep the MAX_TAGS most frequent tags (tractable LP count)
    top = {t for t, _ in deg.most_common(MAX_TAGS)}
    adj = defaultdict(dict)
    for (a, b), w in pair.items():
        if w >= MIN_COOCC and a in top and b in top:
            adj[a][b] = w
            adj[b][a] = w
    nodes = sorted(adj)
    edges = {(a, b) for a in adj for b in adj[a] if a < b}
    print(f"tag graph: {len(nodes)} nodes, {len(edges)} edges "
          f"(MIN_COOCC={MIN_COOCC}, MAX_TAGS={MAX_TAGS}) [{time.time()-t0:.1f}s]")

    # neighbour distribution: lazy self-mass ALPHA, rest spread by co-occ weight
    def mu(x):
        nb = adj[x]
        tot = sum(nb.values())
        keys = [x] + list(nb)
        w = [ALPHA] + [(1 - ALPHA) * v / tot for v in nb.values()]
        return keys, np.array(w)

    # ground distance: 1 if adjacent or equal, else 2 (BFS-capped; cheap proxy)
    def d(a, b):
        if a == b:
            return 0.0
        if b in adj[a]:
            return 1.0
        return 2.0

    results = []
    for (a, b) in edges:
        kx, wx = mu(a)
        ky, wy = mu(b)
        emd = w1(kx, wx, ky, wy, d)
        if emd is None:
            continue
        kappa = 1.0 - emd / d(a, b)
        results.append((kappa, a, b, adj[a][b]))
    results.sort()

    short = lambda t: t.replace("se-tag-", "")
    print(f"\ncomputed kappa on {len(results)} edges [{time.time()-t0:.1f}s]")
    print("\n=== MOST NEGATIVE kappa (candidate BOTTLENECK / bridge concepts) ===")
    for k, a, b, w in results[:15]:
        print(f"  {k:+.3f}  {short(a)} -- {short(b)}  (co-occ {w})")
    print("\n=== MOST POSITIVE kappa (dense intra-community edges) ===")
    for k, a, b, w in results[-10:]:
        print(f"  {k:+.3f}  {short(a)} -- {short(b)}  (co-occ {w})")

    out = "/home/joe/code/futon6/resources/differentiable-math/ricci-tag-curvature.json"
    with open(out, "w") as f:
        json.dump({"min_coocc": MIN_COOCC, "max_tags": MAX_TAGS, "alpha": ALPHA,
                   "n_nodes": len(nodes), "n_edges": len(results),
                   "edges": [{"kappa": k, "a": short(a), "b": short(b), "cooccur": w}
                             for k, a, b, w in results]}, f)
    print(f"\nwrote {out}  [{time.time()-t0:.1f}s total]")


if __name__ == "__main__":
    main()
