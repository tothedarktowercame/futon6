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

**Yes.** A polyhedral Lagrangian surface with 4 faces per vertex (with
distinct adjacent faces) always admits a Lagrangian smoothing. The vertex
spanning property ({e_1,...,e_4} spans R^4) is automatic — it follows from
the Lagrangian and distinct-face conditions alone (Section 3), with no
additional hypotheses needed.

**Confidence: Medium-high.** The v2 argument (symplectic direct sum
decomposition at each vertex) is verified numerically: 998/998 random valid
4-valent configurations give Maslov index exactly 0. The decomposition proof
is algebraic (not just heuristic). Vertex smoothing uses the product structure
(corners in symplectic factors); edge smoothing uses generating functions.
The Hamiltonian isotopy property is established via the Weinstein neighborhood
theorem (edges) and the vanishing flux in simply-connected R^2 (vertices).

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
basis — see the nondegeneracy hypothesis below.)

**Lemma (vertex spanning).** At every vertex of a polyhedral Lagrangian
surface with 4 faces per vertex (with distinct adjacent faces sharing
1-dimensional edges), the 4 edge vectors {e_1, e_2, e_3, e_4} span R^4.

*Proof.* Suppose for contradiction that the 4 edge vectors lie in a
3-dimensional subspace H ⊂ R^4. Then all 4 faces L_i = span(e_{i-1,i},
e_{i,i+1}) ⊂ H.

**Step 1.** The restricted form omega|_H has rank 2: since omega is
non-degenerate on R^4, its restriction to a codimension-1 subspace has a
1-dimensional kernel ell = ker(omega|_H) ⊂ H.

**Step 2.** Every Lagrangian 2-plane L ⊂ H contains ell. Proof: L is
2-dimensional with omega|_L = 0. If L were transverse to ell (i.e.,
L ∩ ell = {0}), then L would project isomorphically onto a 2-plane in
H/ell ≅ R^2. But omega|_H descends to a non-degenerate 2-form on H/ell
(since ell = ker(omega|_H)), and a 2-plane in R^2 on which a
non-degenerate 2-form vanishes must be {0}. Contradiction. So ell ⊂ L.

**Step 3 (key).** Each edge e_i is the intersection of two adjacent faces:
e_1 = L_1 ∩ L_2, e_2 = L_2 ∩ L_3, etc. Since L_i and L_{i+1} are
distinct 2-planes (distinct faces), their intersection is exactly
1-dimensional: dim(L_i ∩ L_{i+1}) = 1, so L_i ∩ L_{i+1} = span(e_i).

By Step 2, ell ⊂ L_i and ell ⊂ L_{i+1}, so ell ⊂ L_i ∩ L_{i+1} =
span(e_i). Since both ell and span(e_i) are 1-dimensional subspaces and
one contains the other, ell = span(e_i).

Applying this to all four edges: span(e_1) = ell = span(e_2) = span(e_3) =
span(e_4). That is, all four edge vectors are proportional.

But then L_1 = span(e_4, e_1) = span(e_1) is 1-dimensional, contradicting
the fact that L_1 is a 2-plane. ∎

(Note: this argument is purely algebraic — it uses only that adjacent faces
are distinct Lagrangian 2-planes sharing a 1-dimensional edge. No
topological submanifold condition is needed.)

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

### 5a. Vertex smoothing via the product structure

The key observation: the V_1 ⊕ V_2 decomposition (Section 3) gives K near
each vertex v a product structure, enabling a direct smoothing that bypasses
both crease smoothing and Lagrangian surgery.

**Product decomposition of K near v.** Choose Darboux coordinates adapted to
V_1 ⊕ V_2: let V_1 have coordinates (x, y) with omega_1 = dx ∧ dy, and V_2
have coordinates (u, v) with omega_2 = du ∧ dv (so omega = omega_1 + omega_2).
Orient the edge vectors so that:

    e_1 = (1,0,0,0), e_2 = (0,0,1,0), e_3 = (0,1,0,0), e_4 = (0,0,0,1)

(Here e_1, e_3 ∈ V_1 and e_2, e_4 ∈ V_2, consistent with the decomposition
in Section 3. This can always be arranged by a linear symplectomorphism.)

The 4 faces in these coordinates are:

    L_1 = span(e_4, e_1) = {y = 0, u = 0}, sector x > 0, v > 0
    L_2 = span(e_1, e_2) = {y = 0, v = 0}, sector x > 0, u > 0
    L_3 = span(e_2, e_3) = {x = 0, v = 0}, sector y > 0, u > 0
    L_4 = span(e_3, e_4) = {x = 0, u = 0}, sector y > 0, v > 0

Define the "corner" curves in each factor:

    C_1 = {(x, 0) : x ≥ 0} ∪ {(0, y) : y ≥ 0}  ⊂ V_1  (positive x and y axes)
    C_2 = {(u, 0) : u ≥ 0} ∪ {(0, v) : v ≥ 0}  ⊂ V_2  (positive u and v axes)

**Claim:** K ∩ B(v, r) = C_1 × C_2 (product of the two corners) for small r.

*Verification:* A point (p, q) with p ∈ C_1 and q ∈ C_2 has p = (x, 0) or
(0, y) and q = (u, 0) or (0, v), giving 4 cases — exactly the 4 faces
L_1, ..., L_4 listed above. ∎

**Smoothing.** Replace each corner C_j with a smooth curve C_j^{sm} ⊂ V_j
that agrees with C_j outside a ball of radius delta around the origin:

    C_1^{sm}: smooth curve in V_1, = {(x, 0)} for x > delta, = {(0, y)} for y > delta,
              smooth through the origin (e.g., the curve (cos theta, sin theta)
              reparameterized to match the axes outside the transition)
    C_2^{sm}: analogous in V_2

Explicitly: parameterize C_1^{sm} as gamma_1(t) for t ∈ R, where
gamma_1(t) = (t, 0) for t ≥ delta, gamma_1(t) = (0, -t) for t ≤ -delta,
and gamma_1 is a smooth embedded curve for t ∈ [-delta, delta]. (Such a
curve exists: it is a smooth rounding of the right angle at the origin.)
Define gamma_2(t) analogously for C_2^{sm}.

**Lagrangian property of the product:** The smoothed surface
K^{sm} = C_1^{sm} × C_2^{sm} is a product of smooth curves in the
symplectic factors. A curve in a 2-dimensional symplectic manifold is
always Lagrangian (its dimension is 1 = half of 2, and omega restricted
to a 1-submanifold is zero for dimensional reasons). The product of
Lagrangian submanifolds in (V_1, omega_1) × (V_2, omega_2) is Lagrangian
in (V_1 ⊕ V_2, omega_1 + omega_2): for tangent vectors (u_1, u_2) and
(w_1, w_2) to C_1^{sm} × C_2^{sm}, we have

    omega((u_1, u_2), (w_1, w_2)) = omega_1(u_1, w_1) + omega_2(u_2, w_2) = 0 + 0 = 0

since u_1, w_1 are tangent to C_1^{sm} (1-dimensional in V_1) and u_2, w_2
are tangent to C_2^{sm} (1-dimensional in V_2). ∎

**Smoothness:** K^{sm} = C_1^{sm} × C_2^{sm} is the image of the smooth
map (t_1, t_2) → (gamma_1(t_1), gamma_2(t_2)), which is a smooth immersion
(the tangent vectors d gamma_1/dt_1 and d gamma_2/dt_2 are nonzero and lie
in complementary subspaces V_1 and V_2). So K^{sm} is a smooth Lagrangian
surface, including at v = gamma_1(0) × gamma_2(0). ∎

**Agreement:** K^{sm} = C_1 × C_2 = K outside the region where either
factor was modified (|p| ≤ delta in V_1 or |q| ≤ delta in V_2), so K^{sm}
agrees with K outside a neighborhood of v of radius O(delta).

### 6. Edge smoothing (crease smoothing along edges between vertices)

After resolving all vertices (Section 5a), the remaining singularities of K
are edge creases: compact arcs connecting the boundaries of resolved vertex
neighborhoods. Along each edge arc, two adjacent Lagrangian faces meet at a
dihedral angle. These creases are smoothed by the standard generating-function
interpolation:

**Lemma (edge crease smoothing).** Let L_1, L_2 be two Lagrangian half-planes
meeting along a compact edge arc e (a segment of a common boundary ray, away
from any vertex). Then there exists a smooth Lagrangian surface agreeing with
L_1 on one side and L_2 on the other.

*Proof.* Choose Darboux coordinates (x_1, y_1, x_2, y_2) with e along the
x_1-axis. Each L_i is locally the graph y = grad S_i(x) for quadratic S_i.
Define S_eps(x) = chi(x_1/eps) S_1(x) + (1 - chi(x_1/eps)) S_2(x) for a
smooth cutoff chi. The graph y = grad S_eps is Lagrangian (graph of the
exact 1-form dS_eps). For any fixed eps > 0, this gives a smooth Lagrangian
surface replacing the crease along e. ∎

(The C^1 control issues noted in earlier drafts do not arise here: the
crease smoothing is applied only along edge arcs that are INTERIOR to edges
(between vertex neighborhoods), not at vertices. Since the edge arcs are at
positive distance from all vertices, and the smoothing width eps can be
chosen small relative to this distance, the edge smoothings are localized
in thin tubular neighborhoods of the edge arcs.)

### 7. Global smoothing

**Step 7a: Resolve vertices.** For each 4-valent vertex v_i, choose a ball
B_i of radius r_i centered at v_i, where r_i is small enough that:
- B_i contains no other vertex v_j (j != i), and
- B_i intersects only the edges and faces incident to v_i.

Such radii exist because the vertex set is finite and discrete in R^4.
Within each B_i, apply the product smoothing (Section 5a) using the
V_1 ⊕ V_2 decomposition from Section 3. This replaces K ∩ B_i = C_1 × C_2
with the smooth Lagrangian surface C_1^{sm} × C_2^{sm}.

**Commutativity:** Since the balls {B_i} are pairwise disjoint, the
vertex smoothings have disjoint support and commute.

**Step 7b: Smooth edges.** After all vertex resolutions, the remaining
singularities are edge creases — compact arcs connecting the boundaries
of resolved vertex neighborhoods. Each edge arc lies along the intersection
of two Lagrangian faces and is a compact 1-manifold (with boundary on
the spheres ∂B_i). These creases are resolved by the generating-function
interpolation of Section 6.

The edge smoothings have support in tubular neighborhoods of the edge arcs.
These neighborhoods can be chosen to be disjoint from each other (since the
edges are disjoint away from vertices, which have already been resolved) and
from the vertex balls (since the edge arcs start at ∂B_i, outside B_i).
Therefore the edge smoothings commute with each other and with the vertex
smoothings.

**Compatibility at ∂B_i.** The vertex smoothing (Section 5a) agrees with
the original polyhedral K outside a neighborhood of v_i of radius delta < r_i.
So on ∂B_i, the surface is still polyhedral (two flat faces meeting at an
edge). The edge crease smoothing (Section 6) begins at ∂B_i, where the
surface is already flat. Since both the vertex smoothing (product of smooth
curves) and the edge smoothing (graph of exact 1-form) produce Lagrangian
surfaces, and they agree on the overlap (both equal the original flat faces
on the annular region delta < |x| < r_i), they glue to a globally smooth
Lagrangian surface.

**Step 7c: Global Hamiltonian isotopy.** Each smoothing is parameterized by
a width parameter t (delta for vertices, eps for edges). As t → 0, the
smoothing region shrinks and K_t → K = K_0 in C^0, giving the topological
isotopy on [0, 1]. It remains to show K_t is a *Hamiltonian* isotopy
for t > 0.

**Edge smoothings are Hamiltonian.** The family of generating functions
S_t(x) = chi(x_1/t) S_1(x) + (1 - chi(x_1/t)) S_2(x) defines a smooth
1-parameter family of Lagrangian graphs y = grad S_t(x) in T^*R^2. By
the Weinstein Lagrangian neighborhood theorem (Weinstein 1971, Theorem 6.1),
a smooth family of exact Lagrangian submanifolds in an exact symplectic
manifold (here T^*R^2 with lambda = y dx) is generated by a Hamiltonian:
the 1-form i_{d/dt}(omega)|_{K_t} is exact (it equals d(S_t|_{K_t})),
so the isotopy is Hamiltonian with generating function H_t = dS_t/dt
evaluated on the Lagrangian.

**Vertex smoothings are Hamiltonian.** The vertex smoothing is a product:
K_t^{vertex} = C_1^{sm}(t) × C_2^{sm}(t) in V_1 × V_2. Each C_j^{sm}(t)
is a smooth 1-parameter family of curves in (V_j, omega_j) ≅ (R^2, dx∧dy).
In a 2-dimensional exact symplectic manifold, EVERY isotopy of compact
(or compactly supported) Lagrangian submanifolds is Hamiltonian. This is
because V_j = R^2 is simply connected, so the flux homomorphism
Flux: pi_1(Symp) → H^1(L; R) is trivial (no non-Hamiltonian symplectic
isotopies exist). Concretely: the velocity field d/dt of the isotopy
C_j^{sm}(t) is a symplectic vector field along C_j^{sm}(t); contracting
with omega_j gives a closed 1-form on C_j^{sm}(t), which is exact because
H^1(C_j^{sm}(t); R) = 0 (curves in R^2 are contractible). The primitive
is the Hamiltonian H_j(t). The product isotopy on V_1 × V_2 is Hamiltonian
with generating function H_1(t) + H_2(t) (see McDuff-Salamon, 3rd ed.,
Exercise 3.18: product of Hamiltonian isotopies is Hamiltonian).

**Composition.** The vertex smoothings (in disjoint balls B_i) and edge
smoothings (in disjoint tubular neighborhoods) have pairwise disjoint
compact support. Each is a compactly supported Hamiltonian isotopy. Their
composition is a compactly supported Hamiltonian isotopy (McDuff-Salamon,
*Introduction to Symplectic Topology*, 3rd ed., Proposition 3.17: the
group Ham_c(M, omega) of compactly supported Hamiltonian diffeomorphisms
is a group under composition).

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
   at each vertex (vertex spanning is proved algebraically; opposite edges span
   complementary symplectic 2-planes)
2. This decomposition forces Maslov index exactly 0 (algebraic, not just generic)
3. Near each vertex, K = C_1 × C_2 (product of corners). Smoothing each corner
   gives K^{sm} = C_1^{sm} × C_2^{sm}, a smooth Lagrangian surface (product of
   curves in symplectic factors). No Lagrangian surgery needed at vertices.
4. Edge creases (between vertices) are smoothed by generating function interpolation
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

## References

- L. Polterovich, "The surgery of Lagrange submanifolds," GAFA 1 (1991),
  198-210. [Lagrangian surgery at transverse double points with Maslov index 0]
- F. Lalonde, J.-C. Sikorav, "Sous-variétés lagrangiennes et lagrangiennes
  exactes des fibrés cotangents," Comment. Math. Helv. 66 (1991), 18-33.
  [Lagrangian surgery, complementary to Polterovich]
- D. McDuff, D. Salamon, *Introduction to Symplectic Topology*, 3rd ed.,
  Oxford University Press, 2017, Proposition 3.17. [Composition of compactly
  supported Hamiltonian isotopies]

## Key References from futon6 corpus

- PlanetMath: "symplectic manifold" (SymplecticManifold) — symplectic structures
- PlanetMath: "Darboux's theorem" (DarbouxsTheoremSymplecticGeometry) — local coordinates
- PlanetMath: "concepts in symplectic geometry" (ConceptsInSymplecticGeometry) — overview
- PlanetMath: "symplectic complement" / "coisotropic subspace" — Lagrangian subspaces
- PlanetMath: "Kähler manifold is symplectic" (AKahlerManifoldIsSymplectic) — complex structure
