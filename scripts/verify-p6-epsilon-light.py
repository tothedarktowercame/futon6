#!/usr/bin/env python3
"""
Verify Problem 6: epsilon-light subsets by exact computation on small graphs.

For each graph and epsilon, find the maximum size of an epsilon-light subset S,
i.e., the largest |S| such that L_S <= epsilon * L (PSD condition).

Tests:
1. K_n: should give |S| = floor(epsilon * n)
2. C_n (cycle): should give |S| >= c * epsilon * n for some c > 0
3. Star S_n: should give |S| = n-1 (exclude center) or |S| ~ epsilon * n (include center)
4. Path P_n: check against bound
5. "Worst case" graphs: barbell, etc.
"""
import numpy as np
from itertools import combinations

def graph_laplacian(n, edges):
    """Compute the graph Laplacian from edge list."""
    L = np.zeros((n, n))
    for u, v in edges:
        L[u, u] += 1
        L[v, v] += 1
        L[u, v] -= 1
        L[v, u] -= 1
    return L

def induced_laplacian(n, edges, S):
    """Compute the Laplacian of the induced subgraph on S."""
    S_set = set(S)
    induced_edges = [(u, v) for u, v in edges if u in S_set and v in S_set]
    return graph_laplacian(n, induced_edges)

def is_epsilon_light(L, L_S, epsilon, tol=1e-10):
    """Check if epsilon*L - L_S is PSD."""
    M = epsilon * L - L_S
    eigenvalues = np.linalg.eigvalsh(M)
    return np.all(eigenvalues >= -tol)

def max_epsilon_light_subset(n, edges, epsilon):
    """
    Find the maximum size of an epsilon-light subset (brute force).
    Only feasible for small n.
    """
    L = graph_laplacian(n, edges)
    best_size = 0
    best_subset = []

    # Check subsets from largest to smallest
    for size in range(n, 0, -1):
        if size <= best_size:
            break
        found = False
        for S in combinations(range(n), size):
            L_S = induced_laplacian(n, edges, S)
            if is_epsilon_light(L, L_S, epsilon):
                best_size = size
                best_subset = list(S)
                found = True
                break
        if found:
            break

    return best_size, best_subset

def complete_graph(n):
    return [(i, j) for i in range(n) for j in range(i+1, n)]

def cycle_graph(n):
    return [(i, (i+1) % n) for i in range(n)]

def path_graph(n):
    return [(i, i+1) for i in range(n-1)]

def star_graph(n):
    """Star with center 0 and n-1 leaves."""
    return [(0, i) for i in range(1, n)]

def barbell_graph(k):
    """Two complete K_k joined by a single edge. Total vertices = 2k."""
    edges = []
    # First clique: vertices 0..k-1
    for i in range(k):
        for j in range(i+1, k):
            edges.append((i, j))
    # Second clique: vertices k..2k-1
    for i in range(k, 2*k):
        for j in range(i+1, 2*k):
            edges.append((i, j))
    # Bridge edge
    edges.append((k-1, k))
    return 2*k, edges

def main():
    print("=" * 70)
    print("PROBLEM 6: Epsilon-Light Subset Verification")
    print("=" * 70)

    epsilons = [0.1, 0.25, 0.5, 0.75]

    # --- Test 1: Complete graph K_n ---
    print("\n--- K_n (expected: |S| = floor(eps * n)) ---")
    for n in [6, 8, 10]:
        edges = complete_graph(n)
        print(f"\nK_{n}:")
        for eps in epsilons:
            size, S = max_epsilon_light_subset(n, edges, eps)
            expected = int(eps * n)
            match = "✓" if size == expected else "✗"
            c_eff = size / (eps * n) if eps * n > 0 else float('inf')
            print(f"  eps={eps:.2f}: |S|={size}, expected={expected}, c_eff={c_eff:.3f} {match}")

    # --- Test 2: Cycle graph C_n ---
    print("\n--- C_n (testing c >= ?) ---")
    for n in [8, 10, 12]:
        edges = cycle_graph(n)
        print(f"\nC_{n}:")
        for eps in epsilons:
            size, S = max_epsilon_light_subset(n, edges, eps)
            c_eff = size / (eps * n) if eps * n > 0 else float('inf')
            print(f"  eps={eps:.2f}: |S|={size}, c_eff={c_eff:.3f}")

    # --- Test 3: Path graph P_n ---
    print("\n--- P_n (testing c >= ?) ---")
    for n in [8, 10, 12]:
        edges = path_graph(n)
        print(f"\nP_{n}:")
        for eps in epsilons:
            size, S = max_epsilon_light_subset(n, edges, eps)
            c_eff = size / (eps * n) if eps * n > 0 else float('inf')
            print(f"  eps={eps:.2f}: |S|={size}, c_eff={c_eff:.3f}")

    # --- Test 4: Star graph S_n ---
    print("\n--- Star S_n ---")
    for n in [8, 10, 12]:
        edges = star_graph(n)
        print(f"\nStar_{n}:")
        for eps in epsilons:
            size, S = max_epsilon_light_subset(n, edges, eps)
            c_eff = size / (eps * n) if eps * n > 0 else float('inf')
            includes_center = 0 in S if S else False
            print(f"  eps={eps:.2f}: |S|={size}, c_eff={c_eff:.3f}, center={'in' if includes_center else 'out'}")

    # --- Test 5: Barbell graph ---
    print("\n--- Barbell (two K_k joined by bridge) ---")
    for k in [4, 5]:
        n, edges = barbell_graph(k)
        print(f"\nBarbell_{k} (n={n}):")
        for eps in epsilons:
            size, S = max_epsilon_light_subset(n, edges, eps)
            c_eff = size / (eps * n) if eps * n > 0 else float('inf')
            print(f"  eps={eps:.2f}: |S|={size}, c_eff={c_eff:.3f}")

    # --- Summary: find minimum c across all graphs ---
    print("\n" + "=" * 70)
    print("MINIMUM c_eff = |S| / (eps * n) across all tests:")
    print("=" * 70)

    all_results = []
    test_cases = [
        ("K_8", 8, complete_graph(8)),
        ("K_10", 10, complete_graph(10)),
        ("C_8", 8, cycle_graph(8)),
        ("C_10", 10, cycle_graph(10)),
        ("C_12", 12, cycle_graph(12)),
        ("P_8", 8, path_graph(8)),
        ("P_10", 10, path_graph(10)),
        ("Star_10", 10, star_graph(10)),
    ]
    bk, bk_edges = barbell_graph(4)
    test_cases.append(("Barbell_4", bk, bk_edges))
    bk2, bk2_edges = barbell_graph(5)
    test_cases.append(("Barbell_5", bk2, bk2_edges))

    min_c = float('inf')
    worst_case = ""

    for name, n, edges in test_cases:
        for eps in epsilons:
            size, _ = max_epsilon_light_subset(n, edges, eps)
            c_eff = size / (eps * n)
            if c_eff < min_c:
                min_c = c_eff
                worst_case = f"{name}, eps={eps}"

    print(f"Minimum c_eff = {min_c:.4f} at {worst_case}")
    print(f"Solution claims c >= 1/2 = 0.5000")
    print(f"{'✓ Consistent' if min_c >= 0.49 else '✗ VIOLATION — c < 1/2!'}")

if __name__ == "__main__":
    main()
