# Problem 8: Lagrangian Smoothing of Polyhedral Surfaces with 4-Valent Vertices

## Problem Statement

A **polyhedral Lagrangian surface** K in R^4 is a finite polyhedral complex all
of whose faces are Lagrangian 2-planes, which is a topological submanifold of
R^4. A **Lagrangian smoothing** of K is a Hamiltonian isotopy K_t of smooth
Lagrangian submanifolds for t in (0,1], extending to a topological isotopy on
[0,1], with K_0 = K.

**Question:** If K has exactly 4 faces meeting at every vertex, does K
necessarily have a Lagrangian smoothing?

## Answer

**Yes.** A polyhedral Lagrangian surface with 4 faces per vertex always admits
a Lagrangian smoothing.

**Confidence: Medium-high.** The v2 argument (symplectic direct sum
decomposition at each vertex) is verified numerically: 998/998 random valid
4-valent configurations give Maslov index exactly 0. The decomposition proof
is algebraic (not just heuristic), and the surgery step invokes classical
results (Polterovich 1991).

## Solution

### 1. Setup: Lagrangian planes in R^4

Identify R^4 = C^2 with symplectic form omega = dx_1 ∧ dy_1 + dx_2 ∧ dy_2.
A Lagrangian plane is a 2-dimensional subspace L where omega|_L = 0. The space
of Lagrangian planes (the Lagrangian Grassmannian) is:

    Lambda(2) = U(2)/O(2)

This is a manifold of dimension 3 with pi_1(Lambda(2)) = Z (the Maslov class).

Each face of K lies in a Lagrangian plane. At each edge, two Lagrangian faces
meet at a dihedral angle. At each vertex, 4 Lagrangian faces meet.

### 2. Local structure at a 4-valent vertex

At a vertex v, the 4 faces L_1, L_2, L_3, L_4 (in cyclic order around v) are
Lagrangian half-planes meeting along edges:

    e_{12} = L_1 ∩ L_2,  e_{23} = L_2 ∩ L_3,  e_{34} = L_3 ∩ L_4,  e_{41} = L_4 ∩ L_1

These 4 edges are rays from v. Each face is spanned by its two boundary edges:

    L_i = span(e_{i-1,i}, e_{i,i+1})

The Lagrangian condition omega|_{L_i} = 0 requires:

    omega(e_{i-1,i}, e_{i,i+1}) = 0  for each i (mod 4)

### 3. The symplectic direct sum decomposition (key new argument)

**Theorem (v2).** The 4 edge vectors e_1 = e_{12}, e_2 = e_{23}, e_3 = e_{34},
e_4 = e_{41} satisfy omega(e_i, e_{i+1}) = 0 for all i (mod 4). This forces
a symplectic direct sum decomposition:

    R^4 = V_1 ⊕ V_2

where V_1 = span(e_1, e_3) and V_2 = span(e_2, e_4) are symplectic 2-planes
with omega|_{V_1 x V_2} = 0.

**Proof.** Write the omega matrix in the basis (e_1, e_2, e_3, e_4). The
conditions omega(e_i, e_{i+1}) = 0 kill 4 of the 6 independent entries,
leaving only:

    a = omega(e_1, e_3) ≠ 0,    b = omega(e_2, e_4) ≠ 0

(Nonzero because omega is non-degenerate and {e_1, e_2, e_3, e_4} is a
basis — guaranteed by the topological submanifold condition at v.)

In the reordered basis (e_1, e_3, e_2, e_4), the omega matrix is:

    [[0, a, 0, 0], [-a, 0, 0, 0], [0, 0, 0, b], [0, 0, -b, 0]]

This is block diagonal: (V_1, a) ⊕ (V_2, b), a symplectic direct sum. ∎

**Consequence:** Each Lagrangian face decomposes as a direct sum of lines:

    L_1 = span(e_4) ⊕ span(e_1)  ⊂  V_2 ⊕ V_1
    L_2 = span(e_1) ⊕ span(e_2)  ⊂  V_1 ⊕ V_2
    L_3 = span(e_2) ⊕ span(e_3)  ⊂  V_2 ⊕ V_1
    L_4 = span(e_3) ⊕ span(e_4)  ⊂  V_1 ⊕ V_2

Each L_i takes one line from V_1 and one from V_2, which is automatically
Lagrangian in V_1 ⊕ V_2 (omega restricted to a line is zero).

### 4. Maslov index vanishes exactly

The Maslov index of the vertex loop L_1 → L_2 → L_3 → L_4 → L_1 in
Lambda(2) decomposes via the direct sum:

    mu = mu_1 + mu_2

where mu_j is the winding number of the component loop in V_j.

**In V_1 = span(e_1, e_3):** The loop traces
    span(e_1) → span(e_1) → span(e_3) → span(e_3) → span(e_1)

This is a back-and-forth path (e_1 → e_1 → e_3 → e_3 → e_1), not a
winding loop. The winding number is 0.

**In V_2 = span(e_2, e_4):** Similarly:
    span(e_4) → span(e_2) → span(e_2) → span(e_4) → span(e_4)

Winding number 0.

**Total Maslov index: mu = 0 + 0 = 0.**

**Numerical verification:** 998/998 random valid 4-valent configurations
(with edge-sharing enforced) give Maslov index exactly 0. In contrast,
random quadruples without edge-sharing give nonzero Maslov index ~45% of
the time (see scripts/verify-p8-maslov-v2.py).

### 5. Why 4 is special: the 3-face obstruction

For a 3-face vertex, the 3 edge vectors e_1, e_2, e_3 must satisfy:

    omega(e_1, e_2) = omega(e_2, e_3) = omega(e_3, e_1) = 0

This means omega vanishes on ALL pairs — the span is an isotropic subspace.
But in (R^4, omega), the maximum isotropic dimension is 2 (= half the
dimension). Three independent isotropic vectors cannot exist.

**Therefore: a non-degenerate 3-face Lagrangian vertex is impossible in R^4.**

The 4-face condition is precisely the right valence: it gives enough edges to
span R^4 while the cyclic omega-orthogonality creates the symplectic direct
sum that forces Maslov index 0.

For 5 or more faces: the omega-orthogonality conditions are over-determined
and generically have no solution with the edges spanning R^4. The 4-face case
is the generic sweet spot.

### 6. Lagrangian surgery at transverse double points

The symplectic direct sum decomposition shows that each 4-valent vertex is a
**transverse Lagrangian crossing** of two smooth sheets:

    Sheet A = L_1 ∪ L_3  (uses edges e_4, e_1, e_2, e_3 → contributes V_1-component)
    Sheet B = L_2 ∪ L_4  (uses edges e_1, e_2, e_3, e_4 → contributes V_2-component)

Each sheet is a piecewise-Lagrangian surface with a crease (bend along a line
through v), and the two sheets cross transversally at v.

**Classical result (Polterovich 1991, Lalonde-Sikorav 1991):** A transverse
Lagrangian crossing with Maslov index 0 can be resolved by Lagrangian surgery,
replacing a neighborhood of v with a smooth Lagrangian annulus. The surgery is:

1. Choose Darboux coordinates adapted to the V_1 ⊕ V_2 decomposition
2. Replace the crossing with a smooth Lagrangian "neck" connecting the sheets
3. The neck is parameterized by a generating function y = grad S(x)
4. The resulting smooth Lagrangian is Hamiltonian isotopic to the original
   (the isotopy parameter t controls the neck width, t → 0 gives the crossing)

### 7. Global smoothing

**Step 7a: Resolve vertices.** At each 4-valent vertex, apply Lagrangian
surgery using the V_1 ⊕ V_2 decomposition from Section 3. This resolves the
vertex singularity into a smooth Lagrangian.

**Step 7b: Smooth edges.** After vertex resolution, the remaining singularities
are edges (creases between adjacent faces). Each edge is a curve along which
two Lagrangian planes meet at a dihedral angle. These creases are resolved by
generating function interpolation:

If the two faces near an edge are y = grad S_1(x) and y = grad S_2(x) in
Darboux coordinates, the smoothing interpolates: y = grad S_t(x) where S_t
smoothly transitions from S_1 to S_2.

**Step 7c: Hamiltonian isotopy.** All vertex resolutions and edge smoothings
are Hamiltonian isotopies (each generated by a compactly supported Hamiltonian).
Their composition gives the global smoothing K_t.

### 8. Topological constraints

For K to be a compact topological submanifold with 4 faces per vertex,
Euler's formula constrains the topology. For quadrilateral faces:
V - E + F = chi(K), with 4F = 2E and 4V = 2E, giving chi = 0.
So K is topologically a torus or Klein bottle.

For non-quadrilateral faces with 4 per vertex, other topologies are possible
but chi <= 0 (higher genus surfaces).

### 9. Summary

The smoothing exists because:
1. The 4-face + Lagrangian condition forces a symplectic direct sum R^4 = V_1 ⊕ V_2
   at each vertex (opposite edges span complementary symplectic 2-planes)
2. This decomposition forces Maslov index exactly 0 (algebraic, not just generic)
3. Zero Maslov + transverse crossing → Lagrangian surgery (Polterovich 1991)
4. Edge creases are smoothed by generating function interpolation
5. All smoothings are Hamiltonian isotopies, composable to give global K_t
6. The 3-face case is impossible (isotropic dimension bound), explaining
   why 4 is the right valence

## Corrections from v1

- **v1 claimed** Maslov index vanishes by "alternating orientations relative
  to J." **Replaced** with the precise symplectic direct sum argument: the
  edge-sharing constraint forces omega to be block diagonal in the
  (e_1, e_3, e_2, e_4) basis, giving mu = 0 algebraically.
- **v1 stated** the 3-face obstruction as "nonzero Maslov index."
  **Corrected:** 3-face Lagrangian vertices don't have nonzero Maslov index —
  they're impossible (3 isotropic vectors can't span 3D in R^4).
- **v1 lacked** numerical verification. **Added** via verify-p8-maslov-v2.py:
  998/998 valid configurations give mu = 0; comparison with non-edge-sharing
  quadruples (55% have mu = 0) confirms the constraint is essential.

## Key References from futon6 corpus

- PlanetMath: "symplectic manifold" (SymplecticManifold) — symplectic structures
- PlanetMath: "Darboux's theorem" (DarbouxsTheoremSymplecticGeometry) — local coordinates
- PlanetMath: "concepts in symplectic geometry" (ConceptsInSymplecticGeometry) — overview
- PlanetMath: "symplectic complement" / "coisotropic subspace" — Lagrangian subspaces
- PlanetMath: "Kähler manifold is symplectic" (AKahlerManifoldIsSymplectic) — complex structure
