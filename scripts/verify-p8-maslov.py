#!/usr/bin/env python3
"""
Verify Problem 8: Maslov index computation for Lagrangian vertex singularities.

Tests the core claim: 4-face vertices have Maslov index 0 (smoothable),
while 3-face vertices generically have Maslov index ±1 (obstructed).

The Maslov index of a cyclic sequence of Lagrangian planes L_1, ..., L_k
is computed via the Kashiwara-Wall signature of a quadratic form on the
direct sum L_1 ⊕ ... ⊕ L_k.
"""
import numpy as np
from itertools import combinations

def is_lagrangian(L, omega):
    """Check if a 2D subspace (given as 4x2 matrix) is Lagrangian."""
    # omega restricted to L should be zero
    return np.allclose(L.T @ omega @ L, 0, atol=1e-10)

def maslov_triple(L1, L2, L3, omega):
    """
    Compute the Maslov index of a triple of Lagrangian planes.

    The Kashiwara index tau(L1, L2, L3) is the signature of the quadratic form
    Q on L1 ⊕ L2 ⊕ L3 defined by:
        Q(v1, v2, v3) = omega(v1, v2) + omega(v2, v3) + omega(v3, v1)

    For 2D Lagrangian planes in R^4, this is a form on a 6D space.
    """
    n = L1.shape[0]  # = 4
    k = L1.shape[1]  # = 2

    # Build the 6x6 matrix of the quadratic form
    # Blocks: Q_{ij} for (i,j) in {1,2,3}
    # Q_{12} = omega restricted to L1 x L2, etc.
    # Q(v1,v2,v3) = v1^T omega v2 + v2^T omega v3 + v3^T omega v1
    # Symmetrized: Q_sym = (Q + Q^T)/2

    Q12 = L1.T @ omega @ L2  # 2x2
    Q23 = L2.T @ omega @ L3  # 2x2
    Q31 = L3.T @ omega @ L1  # 2x2

    # The full form on (v1, v2, v3) in R^2 x R^2 x R^2:
    # Q = [[0, Q12, -Q31^T], [-Q12^T, 0, Q23], [Q31, -Q23^T, 0]]
    # Wait, let me be more careful.
    # Q(v1,v2,v3) = v1^T (L1^T omega L2) v2 + v2^T (L2^T omega L3) v3 + v3^T (L3^T omega L1) v1
    # In block form as a bilinear form:
    Z = np.zeros((k, k))
    Q_full = np.block([
        [Z,   Q12,     Q31.T],  # omega(v1,v2) from first term, omega(v1,v3) from third term (transposed)
        [Q12.T, Z,     Q23],    # wait, this isn't right
        [Q31, Q23.T,   Z]
    ])
    # Actually: Q(v,w) as bilinear form where v=(v1,v2,v3), w=(w1,w2,w3)
    # The form is: omega(v1,w2) + omega(v2,w3) + omega(v3,w1)
    # So Q_{12} block = L1^T omega L2 (maps v1,w2)
    # Q_{23} block = L2^T omega L3 (maps v2,w3)
    # Q_{31} block = L3^T omega L1 (maps v3,w1)
    # All other blocks are 0.

    Q_bilinear = np.block([
        [Z,          Q12,        Z],
        [Z,          Z,          Q23],
        [Q31,        Z,          Z]
    ])

    # Symmetrize to get the quadratic form matrix
    Q_sym = (Q_bilinear + Q_bilinear.T) / 2

    # Signature = (number of positive eigenvalues) - (number of negative eigenvalues)
    eigenvalues = np.linalg.eigvalsh(Q_sym)
    n_pos = np.sum(eigenvalues > 1e-10)
    n_neg = np.sum(eigenvalues < -1e-10)

    return int(n_pos - n_neg)

def maslov_loop(planes, omega):
    """
    Compute the Maslov index of a cyclic loop of Lagrangian planes.

    For a loop L_1 -> L_2 -> ... -> L_k -> L_1, the Maslov index is the
    sum of consecutive triple indices:
        mu = sum_{i} tau(L_i, L_{i+1}, L_{i+2})

    Actually, for a cyclic sequence, the total Maslov index decomposes as
    a sum of triangle contributions. For our purposes, we use the
    Wall-Kashiwara formula directly on the full cyclic form.
    """
    k = len(planes)
    if k == 3:
        return maslov_triple(planes[0], planes[1], planes[2], omega)

    # For k=4: decompose into two triples sharing an edge
    # mu(L1,L2,L3,L4) = tau(L1,L2,L3) + tau(L1,L3,L4)
    # (triangulation of the quadrilateral)
    if k == 4:
        t1 = maslov_triple(planes[0], planes[1], planes[2], omega)
        t2 = maslov_triple(planes[0], planes[2], planes[3], omega)
        return t1 + t2

    # General case: fan triangulation from vertex 0
    total = 0
    for i in range(1, k-1):
        total += maslov_triple(planes[0], planes[i], planes[i+1], omega)
    return total


def random_lagrangian(rng):
    """Generate a random Lagrangian plane in R^4 = C^2."""
    # A Lagrangian plane is the image of a 4x2 matrix L with L^T omega L = 0.
    # Equivalently: L = [[A], [B]] where A, B are 2x2 and A^T B is symmetric.
    # Simple construction: L = [[I], [S]] where S is symmetric.
    S = rng.standard_normal((2, 2))
    S = (S + S.T) / 2  # symmetrize
    L = np.vstack([np.eye(2), S])
    # Normalize columns
    L = L / np.linalg.norm(L, axis=0, keepdims=True)
    return L

def random_lagrangian_general(rng):
    """Generate a random Lagrangian plane using U(2) action."""
    # Start with the standard Lagrangian R^2 ⊂ C^2
    # Apply a random unitary matrix U ∈ U(2) to get U · R^2
    # In real coordinates: if U = A + iB, then U · R^2 = col(A, B) for the real part
    from scipy.stats import ortho_group
    # Use a random 2x2 real matrix and form a "unitary-like" rotation
    theta = rng.uniform(0, 2 * np.pi)
    phi = rng.uniform(0, np.pi)
    psi = rng.uniform(0, 2 * np.pi)
    # Parametrize U(2): U = diag(e^{i alpha}, 1) * SU(2) * diag(e^{i beta}, 1)
    # For SU(2): [[cos theta, -sin theta e^{i phi}], [sin theta e^{-i phi}, cos theta]]
    c, s = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)

    U_real = np.array([
        [c, -s * cp],
        [s * cp, c]
    ])
    U_imag = np.array([
        [0, -s * sp],
        [-s * sp, 0]
    ])

    # The Lagrangian plane is the image of [[U_real], [U_imag]]
    L = np.vstack([U_real, U_imag])
    return L


def main():
    # Standard symplectic form on R^4: omega = [[0, I], [-I, 0]]
    omega = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-1, 0, 0, 0],
        [0, -1, 0, 0]
    ], dtype=float)

    rng = np.random.default_rng(42)

    print("=" * 60)
    print("PROBLEM 8: Maslov Index Verification")
    print("=" * 60)

    # --- Test 1: Known Lagrangian planes ---
    print("\n--- Test 1: Verify Lagrangian property ---")

    # Standard Lagrangian: R^2 x {0} in R^4
    L_std = np.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=float)
    print(f"L_std = R^2 x 0: Lagrangian? {is_lagrangian(L_std, omega)}")

    # Another Lagrangian: {0} x R^2
    L_imag = np.array([[0, 0], [0, 0], [1, 0], [0, 1]], dtype=float)
    print(f"L_imag = 0 x R^2: Lagrangian? {is_lagrangian(L_imag, omega)}")

    # Diagonal Lagrangian: graph of identity
    L_diag = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float) / np.sqrt(2)
    print(f"L_diag = graph(I): Lagrangian? {is_lagrangian(L_diag, omega)}")
    # This should NOT be Lagrangian: omega(e1+e3, e2+e4) = omega(e1,e4) + omega(e3,e2) = 1 + (-1)?
    # Let me recalculate: omega(e1,e3)=1, omega(e2,e4)=1, omega(e1,e4)=0, omega(e2,e3)=0
    # For v=(1,0,1,0)/sqrt2 and w=(0,1,0,1)/sqrt2: v^T omega w = (0+0+(-1)+0+1+0+0+0)/2...
    # Let me just compute it numerically.

    # Graph of symmetric matrix S: L = [[I], [S]] is Lagrangian iff S = S^T
    S1 = np.array([[1, 0], [0, -1]], dtype=float)  # symmetric
    L_graph = np.vstack([np.eye(2), S1])
    L_graph = L_graph / np.linalg.norm(L_graph, axis=0, keepdims=True)
    print(f"L = graph(diag(1,-1)): Lagrangian? {is_lagrangian(L_graph, omega)}")

    # --- Test 2: Maslov index of 3 transverse Lagrangian planes ---
    print("\n--- Test 2: Maslov index of TRIPLES (3-face vertex) ---")

    # Three standard Lagrangian planes that are pairwise transverse
    L1 = np.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=float)  # y1=y2=0

    S2 = np.array([[1, 0], [0, 1]], dtype=float)  # graph of I
    L2 = np.vstack([np.eye(2), S2])
    L2 = L2 / np.linalg.norm(L2, axis=0, keepdims=True)

    S3 = np.array([[0, 1], [1, 0]], dtype=float)  # graph of [[0,1],[1,0]]
    L3 = np.vstack([np.eye(2), S3])
    L3 = L3 / np.linalg.norm(L3, axis=0, keepdims=True)

    print(f"L1 Lagrangian? {is_lagrangian(L1, omega)}")
    print(f"L2 Lagrangian? {is_lagrangian(L2, omega)}")
    print(f"L3 Lagrangian? {is_lagrangian(L3, omega)}")

    mu3 = maslov_triple(L1, L2, L3, omega)
    print(f"Maslov index tau(L1, L2, L3) = {mu3}")

    # Try multiple random triples
    print("\nRandom triples (100 trials):")
    maslov_counts = {}
    for _ in range(100):
        planes = [random_lagrangian(rng) for _ in range(3)]
        # Check all are Lagrangian
        if not all(is_lagrangian(p, omega) for p in planes):
            continue
        mu = maslov_triple(planes[0], planes[1], planes[2], omega)
        maslov_counts[mu] = maslov_counts.get(mu, 0) + 1

    print(f"Distribution of Maslov index for random triples: {dict(sorted(maslov_counts.items()))}")

    # --- Test 3: Maslov index of 4-face configurations ---
    print("\n--- Test 3: Maslov index of QUADRUPLES (4-face vertex) ---")

    # The key claim: 4 Lagrangian planes arranged as two transverse pairs
    # have total Maslov index 0.

    # Pair 1 (Sheet A): L1 and L3 are "opposite" faces
    # Pair 2 (Sheet B): L2 and L4 are "opposite" faces
    # The two sheets cross transversally

    L1 = np.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=float)  # y=0

    L3 = np.vstack([np.eye(2), 2*np.eye(2)])  # graph of 2I (same "family" as L1)
    L3 = L3 / np.linalg.norm(L3, axis=0, keepdims=True)

    # L2 and L4 should be transverse to both L1 and L3
    S2 = np.array([[0, 1], [1, 0]], dtype=float)
    L2 = np.vstack([np.eye(2), S2])
    L2 = L2 / np.linalg.norm(L2, axis=0, keepdims=True)

    S4 = np.array([[0, -1], [-1, 0]], dtype=float)
    L4 = np.vstack([np.eye(2), S4])
    L4 = L4 / np.linalg.norm(L4, axis=0, keepdims=True)

    print(f"L1 Lagrangian? {is_lagrangian(L1, omega)}")
    print(f"L2 Lagrangian? {is_lagrangian(L2, omega)}")
    print(f"L3 Lagrangian? {is_lagrangian(L3, omega)}")
    print(f"L4 Lagrangian? {is_lagrangian(L4, omega)}")

    mu4 = maslov_loop([L1, L2, L3, L4], omega)
    print(f"Maslov index of loop (L1,L2,L3,L4) = {mu4}")

    # Also check the two triangulations separately
    t1 = maslov_triple(L1, L2, L3, omega)
    t2 = maslov_triple(L1, L3, L4, omega)
    print(f"  Triangle (L1,L2,L3): tau = {t1}")
    print(f"  Triangle (L1,L3,L4): tau = {t2}")
    print(f"  Sum = {t1 + t2}")

    # Random quadruples with paired structure
    print("\nRandom paired quadruples (100 trials):")
    maslov_counts_4 = {}
    for _ in range(100):
        # Generate two "sheets" as pairs of Lagrangian planes
        S_a1 = rng.standard_normal((2,2)); S_a1 = (S_a1 + S_a1.T)/2
        S_a2 = S_a1 + 0.5 * np.eye(2)  # nearby symmetric matrix (same "sheet")
        S_b1 = rng.standard_normal((2,2)); S_b1 = (S_b1 + S_b1.T)/2
        S_b2 = S_b1 + 0.5 * np.eye(2)  # nearby (same "sheet")

        planes = []
        for S in [S_a1, S_b1, S_a2, S_b2]:  # interleaved: A1, B1, A2, B2
            L = np.vstack([np.eye(2), S])
            L = L / np.linalg.norm(L, axis=0, keepdims=True)
            planes.append(L)

        if not all(is_lagrangian(p, omega) for p in planes):
            continue
        mu = maslov_loop(planes, omega)
        maslov_counts_4[mu] = maslov_counts_4.get(mu, 0) + 1

    print(f"Distribution of Maslov index for paired quadruples: {dict(sorted(maslov_counts_4.items()))}")

    # Random UNPAIRED quadruples (4 independent Lagrangian planes)
    print("\nRandom UNPAIRED quadruples (100 trials):")
    maslov_counts_4u = {}
    for _ in range(100):
        planes = [random_lagrangian(rng) for _ in range(4)]
        if not all(is_lagrangian(p, omega) for p in planes):
            continue
        mu = maslov_loop(planes, omega)
        maslov_counts_4u[mu] = maslov_counts_4u.get(mu, 0) + 1

    print(f"Distribution of Maslov index for unpaired quadruples: {dict(sorted(maslov_counts_4u.items()))}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
If the 4-face claim is correct:
- Random TRIPLES should have nonzero Maslov index (generically ±2)
- PAIRED quadruples (A1,B1,A2,B2 with A,B from same sheet) should have
  Maslov index 0 (the two triple contributions cancel)
- UNPAIRED quadruples should have variable Maslov index

The 4-face condition forces the paired structure, which forces
Maslov index 0, which enables Lagrangian surgery.
""")


if __name__ == "__main__":
    main()
