#!/usr/bin/env python3
"""
Problem 8 Maslov verification v2: enforce the ACTUAL geometric constraint.

At a 4-valent vertex of a polyhedral Lagrangian that is a topological
submanifold, adjacent faces SHARE AN EDGE. This means:
    L_i = span(e_{i-1,i}, e_{i,i+1})
with the Lagrangian condition omega(e_{i-1,i}, e_{i,i+1}) = 0.

The key discovery: this forces a symplectic direct sum decomposition
R^4 = V_1 + V_2 where V_j = span(opposite edges), making the Maslov
index exactly 0. The v1 script missed this because it generated
random quadruples without enforcing edge-sharing.
"""
import numpy as np

def maslov_triple(L1, L2, L3, omega):
    """Kashiwara-Wall index of a Lagrangian triple."""
    k = L1.shape[1]
    Z = np.zeros((k, k))
    Q12 = L1.T @ omega @ L2
    Q23 = L2.T @ omega @ L3
    Q31 = L3.T @ omega @ L1
    Q_bilinear = np.block([
        [Z,    Q12,  Z],
        [Z,    Z,    Q23],
        [Q31,  Z,    Z]
    ])
    Q_sym = (Q_bilinear + Q_bilinear.T) / 2
    eigenvalues = np.linalg.eigvalsh(Q_sym)
    n_pos = np.sum(eigenvalues > 1e-10)
    n_neg = np.sum(eigenvalues < -1e-10)
    return int(n_pos - n_neg)

def maslov_quadruple(planes, omega):
    """Maslov index of a 4-cycle via fan triangulation."""
    t1 = maslov_triple(planes[0], planes[1], planes[2], omega)
    t2 = maslov_triple(planes[0], planes[2], planes[3], omega)
    return t1 + t2

omega = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [-1, 0, 0, 0],
    [0, -1, 0, 0]
], dtype=float)

rng = np.random.default_rng(42)

print("=" * 70)
print("PROBLEM 8 v2: Maslov index with edge-sharing constraint")
print("=" * 70)

# ===================================================================
# PART 1: Algebraic structure
# ===================================================================
print("\n--- Part 1: The symplectic direct sum decomposition ---")
print()
print("At a 4-valent vertex, 4 edge vectors e1,e2,e3,e4 ∈ R^4 satisfy:")
print("  omega(e4, e1) = omega(e1, e2) = omega(e2, e3) = omega(e3, e4) = 0")
print("  (adjacent edges are symplectically orthogonal)")
print()
print("In basis (e1, e3, e2, e4), omega becomes block diagonal:")
print("  omega = [[0, a], [-a, 0]] ⊕ [[0, b], [-b, 0]]")
print("  V1 = span(e1, e3), V2 = span(e2, e4)")
print()

# ===================================================================
# PART 2: Generate VALID 4-valent vertex configurations
# ===================================================================
print("--- Part 2: Random valid 4-valent vertices (1000 trials) ---\n")

maslov_counts = {}
n_valid = 0
n_degenerate = 0

for trial in range(1000):
    # Generate 4 random edge vectors in R^4
    edges = rng.standard_normal((4, 4))  # e1, e2, e3, e4 as rows

    # We need: omega(e_{i}, e_{i+1 mod 4}) = 0 for all i
    # Strategy: generate e1 freely, then project each subsequent edge
    # to be symplectically orthogonal to the previous one, while
    # maintaining the Lagrangian conditions.

    # Better: directly construct from the symplectic decomposition.
    # Choose V1 = span(e1, e3) and V2 = span(e2, e4) to be symplectically
    # complementary. Then construct edges as:
    #   e1 ∈ V1, e2 ∈ V2, e3 ∈ V1, e4 ∈ V2

    # Generate a random symplectomorphism A (4x4, A^T omega A = omega)
    # Then V1 = A(span(e_1, e_3 standard)) and V2 = A(span(e_2, e_4 standard))

    # Simple approach: pick random vectors with the right orthogonality
    e1 = rng.standard_normal(4)

    # e2 must satisfy omega(e1, e2) = 0, i.e., e2 ⊥_omega e1
    # The symplectic complement of e1 is 3-dimensional
    # omega(e1, e2) = e1^T omega e2 = (omega^T e1)^T e2 = -(omega e1)^T e2
    omega_e1 = omega @ e1  # normal to the constraint hyperplane
    e2 = rng.standard_normal(4)
    e2 = e2 - (omega_e1 @ e2) / (omega_e1 @ omega_e1) * omega_e1  # project

    # e3 must satisfy omega(e2, e3) = 0
    omega_e2 = omega @ e2
    e3 = rng.standard_normal(4)
    e3 = e3 - (omega_e2 @ e3) / (omega_e2 @ omega_e2) * omega_e2

    # e4 must satisfy omega(e3, e4) = 0 AND omega(e4, e1) = 0
    omega_e3 = omega @ e3
    omega_e1_for_e4 = omega @ e1  # omega(e4, e1) = e4^T omega e1 = -(omega e1)^T ... hmm
    # omega(e4, e1) = e4^T omega e1
    # We need: e4 · (omega e3) = 0 AND e4 · (omega.T e1) = 0
    # i.e., e4 · (omega e3) = 0 AND e4 · (-omega e1) = 0
    n1 = omega @ e3  # omega(e3, e4) = e3^T omega e4 = n1^T e4... wait
    # omega(e3, e4) = e3^T omega e4. For this to be 0: (omega^T e3)^T e4 = 0
    # omega^T = -omega, so: (-omega e3)^T e4 = 0, i.e., (omega e3) · e4 = 0
    n1 = omega @ e3  # e4 must be perpendicular to this

    # omega(e4, e1) = e4^T omega e1. For this to be 0: (omega e1) · e4 = 0... wait
    # omega(e4, e1) = e4^T omega e1 = (omega^T e4)^T e1... let me just use the bilinear form
    # omega(e4, e1) = sum_{ij} omega_{ij} e4_i e1_j = e4^T (omega @ e1) ... no
    # omega is a matrix: omega(u,v) = u^T omega v
    # So omega(e4, e1) = e4^T omega e1
    # For this to be zero: e4 · (omega e1) = 0? No: e4^T (omega e1) = (omega e1)^T e4 if omega e1 is treated as a column...
    # Actually e4^T omega e1 is a scalar. e4^T times (omega times e1) = e4 dot (omega @ e1).
    # So the constraint is e4 · (omega @ e1) = 0.
    n2 = omega @ e1  # e4 must be perpendicular to this too

    # e4 must be perpendicular to both n1 and n2 (in Euclidean sense)
    # This gives a 2D subspace (generically)
    # Project a random vector onto the orthogonal complement of {n1, n2}
    N = np.column_stack([n1, n2])
    # Check rank
    if np.linalg.matrix_rank(N, tol=1e-8) < 2:
        n_degenerate += 1
        continue

    e4 = rng.standard_normal(4)
    # Project out components along n1 and n2
    Q, _ = np.linalg.qr(N)
    e4 = e4 - Q @ (Q.T @ e4)

    if np.linalg.norm(e4) < 1e-8:
        n_degenerate += 1
        continue

    # Verify all constraints
    tol = 1e-8
    c1 = abs(e1 @ omega @ e2) < tol  # omega(e1, e2) = 0
    c2 = abs(e2 @ omega @ e3) < tol  # omega(e2, e3) = 0
    c3 = abs(e3 @ omega @ e4) < tol  # omega(e3, e4) = 0
    c4 = abs(e4 @ omega @ e1) < tol  # omega(e4, e1) = 0

    if not (c1 and c2 and c3 and c4):
        n_degenerate += 1
        continue

    # Check that {e1,e2,e3,e4} span R^4 (non-degenerate vertex)
    E = np.column_stack([e1, e2, e3, e4])
    if abs(np.linalg.det(E)) < 1e-6:
        n_degenerate += 1
        continue

    # Also verify symplectic non-degeneracy: a = omega(e1,e3) ≠ 0, b = omega(e2,e4) ≠ 0
    a = e1 @ omega @ e3
    b = e2 @ omega @ e4
    if abs(a) < 1e-6 or abs(b) < 1e-6:
        n_degenerate += 1
        continue

    n_valid += 1

    # Build the 4 Lagrangian planes
    L1 = np.column_stack([e4, e1])
    L2 = np.column_stack([e1, e2])
    L3 = np.column_stack([e2, e3])
    L4 = np.column_stack([e3, e4])

    # Verify Lagrangian
    for i, L in enumerate([L1, L2, L3, L4], 1):
        val = L.T @ omega @ L
        assert np.allclose(val, 0, atol=1e-8), f"L{i} not Lagrangian!"

    # Compute Maslov index
    mu = maslov_quadruple([L1, L2, L3, L4], omega)
    maslov_counts[mu] = maslov_counts.get(mu, 0) + 1

print(f"Valid configurations: {n_valid}")
print(f"Degenerate (skipped): {n_degenerate}")
print(f"Maslov index distribution: {dict(sorted(maslov_counts.items()))}")

if maslov_counts:
    all_zero = all(k == 0 for k in maslov_counts.keys())
    print(f"\nAll Maslov indices zero? {'YES ✓' if all_zero else 'NO ✗'}")

# ===================================================================
# PART 3: Verify the symplectic decomposition structure
# ===================================================================
print("\n--- Part 3: Verify V1 ⊕ V2 symplectic decomposition ---\n")

# Take one valid configuration and check the block structure
e1 = rng.standard_normal(4)
omega_e1 = omega @ e1
e2 = rng.standard_normal(4)
e2 = e2 - (omega_e1 @ e2) / (omega_e1 @ omega_e1) * omega_e1
omega_e2 = omega @ e2
e3 = rng.standard_normal(4)
e3 = e3 - (omega_e2 @ e3) / (omega_e2 @ omega_e2) * omega_e2
n1 = omega @ e3
n2 = omega @ e1
N = np.column_stack([n1, n2])
e4 = rng.standard_normal(4)
Q, _ = np.linalg.qr(N)
e4 = e4 - Q @ (Q.T @ e4)

a_val = e1 @ omega @ e3
b_val = e2 @ omega @ e4

print(f"omega(e1, e2) = {e1 @ omega @ e2:.6f}  (should be 0)")
print(f"omega(e2, e3) = {e2 @ omega @ e3:.6f}  (should be 0)")
print(f"omega(e3, e4) = {e3 @ omega @ e4:.6f}  (should be 0)")
print(f"omega(e4, e1) = {e4 @ omega @ e1:.6f}  (should be 0)")
print(f"omega(e1, e3) = {a_val:.6f}  (= a, should be ≠ 0)")
print(f"omega(e2, e4) = {b_val:.6f}  (= b, should be ≠ 0)")
print(f"omega(e1, e4) = {e1 @ omega @ e4:.6f}  (should be 0)")
print(f"omega(e2, e3) = {e2 @ omega @ e3:.6f}  (should be 0, redundant)")
print()
print("Omega matrix in edge basis (e1, e2, e3, e4):")
edges_matrix = np.column_stack([e1, e2, e3, e4])
omega_in_basis = edges_matrix.T @ omega @ edges_matrix
np.set_printoptions(precision=4, suppress=True)
print(omega_in_basis)
print()
print("In reordered basis (e1, e3, e2, e4):")
reorder = [0, 2, 1, 3]
omega_reordered = omega_in_basis[np.ix_(reorder, reorder)]
print(omega_reordered)
print()
print("This should be block diagonal: [[0,a],[-a,0]] ⊕ [[0,b],[-b,0]]")

# ===================================================================
# PART 4: Compare with v1's random (non-edge-sharing) quadruples
# ===================================================================
print("\n--- Part 4: Comparison with v1 (no edge-sharing) ---\n")

def random_lagrangian(rng):
    S = rng.standard_normal((2, 2))
    S = (S + S.T) / 2
    L = np.vstack([np.eye(2), S])
    L = L / np.linalg.norm(L, axis=0, keepdims=True)
    return L

maslov_no_edge = {}
for _ in range(1000):
    planes = [random_lagrangian(rng) for _ in range(4)]
    mu = maslov_quadruple(planes, omega)
    maslov_no_edge[mu] = maslov_no_edge.get(mu, 0) + 1

print(f"Random quadruples WITHOUT edge-sharing (1000 trials):")
print(f"  {dict(sorted(maslov_no_edge.items()))}")
print()

maslov_with_edge = maslov_counts if maslov_counts else {0: 0}
print(f"Valid quadruples WITH edge-sharing ({n_valid} trials):")
print(f"  {dict(sorted(maslov_with_edge.items()))}")

# ===================================================================
# PART 5: Why 3-face vertices are different
# ===================================================================
print("\n--- Part 5: Why 3 faces CAN have nonzero Maslov ---\n")
print("With 3 edges e1, e2, e3 in R^4:")
print("  omega(e1, e2) = omega(e2, e3) = omega(e3, e1) = 0")
print("  => omega vanishes on ALL pairs! => e1, e2, e3 span a 3D")
print("  ISOTROPIC subspace (omega = 0 on it).")
print()
print("But R^4 with a symplectic form has max isotropic dimension 2!")
print("So either:")
print("  (a) e1, e2, e3 span only 2D (degenerate vertex), or")
print("  (b) The constraint is actually omega(e_i, e_{i+1}) = 0 for")
print("      adjacent pairs only: omega(e1,e2) = omega(e2,e3) = omega(e3,e1) = 0")
print("      This IS all pairs for 3 vertices, so {e1,e2,e3} is isotropic.")
print()
print("Since max isotropic dim = 2, three edges can span at most 2D,")
print("so the vertex is degenerate (not a topological submanifold point).")
print()
print("OR: for 3 faces, only omega(e1,e2) = omega(e2,e3) = 0 and")
print("omega(e3,e1) ≠ 0. But omega(e3,e1) = 0 IS required (face L_3 =")
print("span(e_{23}, e_{31}) must be Lagrangian). So 3-face Lagrangian")
print("vertices in R^4 are necessarily degenerate!")
print()
print("WAIT — this means the 3-face case might be impossible as a")
print("topological submanifold. The obstruction isn't Maslov index;")
print("it's that 3 Lagrangian faces can't meet at a non-degenerate")
print("vertex in R^4 at all!")

# Verify: try to construct a non-degenerate 3-face Lagrangian vertex
print("\n--- Attempting to construct 3-face Lagrangian vertex ---")
n_attempts = 1000
n_3face_valid = 0
for _ in range(n_attempts):
    e1 = rng.standard_normal(4)
    omega_e1 = omega @ e1
    e2 = rng.standard_normal(4)
    e2 = e2 - (omega_e1 @ e2) / (omega_e1 @ omega_e1) * omega_e1
    # For 3-face: need omega(e2, e3) = 0 AND omega(e3, e1) = 0
    omega_e2 = omega @ e2
    omega_e1_col = omega @ e1  # omega(e3, e1) = e3^T omega e1 = e3 · (omega e1)
    # e3 must be perp to omega_e2 AND omega_e1
    N = np.column_stack([omega_e2, omega_e1_col])
    if np.linalg.matrix_rank(N, tol=1e-8) < 2:
        continue
    e3 = rng.standard_normal(4)
    Q, _ = np.linalg.qr(N)
    e3 = e3 - Q @ (Q.T @ e3)
    if np.linalg.norm(e3) < 1e-8:
        continue
    # Check: do e1, e2, e3 span a 3D subspace?
    E3 = np.column_stack([e1, e2, e3])
    rank = np.linalg.matrix_rank(E3, tol=1e-6)
    if rank == 3:
        # Check all omega conditions
        c1 = abs(e1 @ omega @ e2) < 1e-8
        c2 = abs(e2 @ omega @ e3) < 1e-8
        c3 = abs(e3 @ omega @ e1) < 1e-8
        if c1 and c2 and c3:
            n_3face_valid += 1
            # This would be a 3D isotropic subspace!
            print(f"  Found 3D isotropic subspace! (rank={rank})")
            print(f"    omega(e1,e2) = {e1 @ omega @ e2:.8f}")
            print(f"    omega(e2,e3) = {e2 @ omega @ e3:.8f}")
            print(f"    omega(e3,e1) = {e3 @ omega @ e1:.8f}")
            break

if n_3face_valid == 0:
    print(f"  No valid 3-face vertex found in {n_attempts} attempts.")
    print(f"  (3 edges with pairwise omega=0 can span at most 2D in R^4)")
else:
    print(f"  Found {n_3face_valid} valid 3-face vertices")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
1. Edge-sharing constraint forces omega to vanish on adjacent edge pairs
2. This creates a symplectic direct sum R^4 = V1 ⊕ V2 (opposite edges)
3. Each Lagrangian face decomposes as (line in V1) ⊕ (line in V2)
4. Maslov index = sum of winding numbers in V1 and V2 = 0 + 0 = 0
5. Therefore: 4-face polyhedral Lagrangian with local flatness has
   Maslov index EXACTLY 0 at every vertex
6. The 3-face case is even more constrained: 3 edges with pairwise
   omega = 0 are isotropic, so span at most 2D — a 3-face vertex
   cannot be non-degenerate in R^4!
""")
