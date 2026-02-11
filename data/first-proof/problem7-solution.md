# Problem 7: Uniform Lattice with 2-Torsion as Fundamental Group of Rationally Acyclic Manifold

## Problem Statement

Suppose Gamma is a uniform lattice in a real semi-simple group, and Gamma
contains some 2-torsion. Is it possible for Gamma to be the fundamental group
of a compact manifold without boundary whose universal cover is acyclic over Q?

## Answer

**Yes.** It is possible.

**Confidence: Medium-low.** The argument relies on surgery theory and the
Farrell-Jones conjecture, which are known for lattices in semi-simple groups
but whose explicit computations are delicate.

## Solution

### 1. Setup and key definitions

Let G be a connected real semi-simple Lie group, K < G a maximal compact
subgroup, and X = G/K the associated symmetric space (a non-positively curved
Riemannian manifold, contractible). Let Gamma < G be a uniform (cocompact)
lattice containing an element g of order 2.

A compact manifold M with pi_1(M) = Gamma and rationally acyclic universal
cover M_tilde means: M is a "rational model" for B Gamma (the classifying
space). Equivalently:

    H_*(M; Q) = H_*(Gamma; Q)  (group homology with rational coefficients)

### 2. The torsion-free case (background)

If Gamma were torsion-free, the answer is immediate: Gamma acts freely and
properly on X, and the quotient M = X / Gamma is a closed manifold with
pi_1 = Gamma and universal cover X (contractible, hence rationally acyclic).
These are the classical locally symmetric spaces.

### 3. Obstruction from torsion: the orbifold issue

With 2-torsion, Gamma does NOT act freely on X. The element g of order 2 has a
nonempty fixed-point set X^g (a totally geodesic submanifold of X). The quotient
X / Gamma is a closed orbifold, not a manifold. Its orbifold fundamental group
is Gamma, but the topological fundamental group of the underlying space may
differ.

However, we are not required to use X / Gamma. We seek ANY closed manifold M
with pi_1(M) = Gamma and rationally acyclic universal cover.

### 4. Gamma is a rational Poincare duality group

**Key fact:** Since Gamma is a uniform lattice in G, the rational group
cohomology H^*(Gamma; Q) is isomorphic to H^*(X/Gamma; Q) (the rational
cohomology of the orbifold). Since X/Gamma is a closed orbifold of dimension
d = dim(X), the rational cohomology satisfies Poincare duality:

    H^k(Gamma; Q) = H^{d-k}(Gamma; Q)  (with appropriate orientation character)

Therefore Gamma is a rational Poincare duality group of formal dimension d.

### 5. The Farrell-Jones conjecture and surgery

The closed manifold M we seek must have dimension d (to match the PD dimension)
and satisfy:
- pi_1(M) = Gamma
- The classifying map f: M -> B Gamma induces an isomorphism H_*(M; Q) -> H_*(Gamma; Q)

The existence of such M is a surgery-theoretic question.

**Step 5a: Normal map.** Start with a degree-1 normal map
f: M_0 -> B Gamma from some closed d-manifold M_0 that induces the correct
map on fundamental groups. Such a map exists by standard surgery theory (for
d >= 5) using the fact that Gamma is finitely presented and has a finite
rational classifying space.

**Step 5b: Surgery obstruction.** The obstruction to surgering f into a
rational homology equivalence (while preserving pi_1 = Gamma) lives in the
Wall group L_d(Z[Gamma]).

**Step 5c: Farrell-Jones isomorphism.** The Farrell-Jones conjecture is
KNOWN for lattices in semi-simple Lie groups (Bartels-Lück 2012, building on
Farrell-Jones 1993 for non-positively curved manifolds). This gives:

    L_d(Z[Gamma]) = H_d^{Gamma}(E_{Fin} Gamma; L^{<-infty>})

where E_{Fin} Gamma is the classifying space for proper Gamma-actions (which
can be taken to be the symmetric space X with its Gamma-action).

**Step 5d: Vanishing of the rational surgery obstruction.** The rational
surgery obstruction is computed by:

    L_d(Z[Gamma]) ⊗ Q = H_d^{Gamma}(E_{Fin} Gamma; L ⊗ Q)

By the Atiyah-Hirzebruch spectral sequence and the fact that L_*(Z) ⊗ Q = Q
concentrated in degrees 0 mod 4, this equivariant homology can be computed
from the rational cohomology of the orbifold X/Gamma and the fixed-point data
of finite subgroups.

For suitable choices of G and Gamma (in particular, for hyperbolic lattices
SO(d,1) with d >= 5 and d odd), the rational surgery obstruction vanishes,
allowing the construction of M.

### 6. Concrete example sketch

Take G = SO(2k+1, 1) for k >= 3, so the symmetric space is hyperbolic
(2k+1)-space H^{2k+1} (dimension 2k+1 >= 7, odd). Let Gamma < G be a
cocompact arithmetic lattice containing an isometric involution (giving
2-torsion). Such lattices exist by arithmetic construction.

The symmetric space H^{2k+1} is contractible. The orbifold H^{2k+1}/Gamma has
odd dimension, so the signature obstruction (which lives in L_{even}) doesn't
interfere. The surgery theory in odd dimensions is more flexible: the Wall
group L_{2k+1}(Z[Gamma]) is related to the Whitehead group, and for lattices
satisfying Farrell-Jones, this is computable.

The rational surgery obstruction vanishes (by parity), giving a closed manifold
M of dimension 2k+1 with pi_1(M) = Gamma and a rational equivalence
M -> B Gamma. The universal cover M_tilde is then rationally acyclic.

### 7. Why 2-torsion is not an obstruction over Q

The critical point: Smith theory (which would obstruct the construction over
Z/2 coefficients) does NOT apply over Q. Specifically:

- Over Z/2: If Z/2 acts on a mod-2 acyclic space, the fixed-point set is
  mod-2 acyclic (Smith's theorem). This forces fixed points, contradicting
  a free action.

- Over Q: If Z/2 acts on a rationally acyclic space, there is NO constraint
  on fixed points. The action can be free. This is because 2 is invertible
  in Q, so the transfer argument gives isomorphisms without fixed-point
  contributions.

Since Gamma acts FREELY on M_tilde (it's the universal cover of M with
pi_1 = Gamma), and the Z/2-subgroup acts freely, there is no Smith-theoretic
obstruction to M_tilde being rationally acyclic.

### 8. Summary

1. Gamma is a rational PD group of dimension d = dim(G/K) (orbifold PD)
2. The surgery-theoretic obstruction to realizing Gamma as pi_1 of a
   rationally aspherical closed d-manifold lives in L_d(Z[Gamma])
3. The Farrell-Jones conjecture (known for these lattices) computes
   L_d(Z[Gamma]) from equivariant topology of the symmetric space
4. For suitable choices (odd-dimensional hyperbolic lattices), the rational
   surgery obstruction vanishes
5. Smith theory doesn't obstruct over Q (unlike over Z/2)
6. The resulting manifold M has pi_1 = Gamma with rationally acyclic M_tilde

## Key References from futon6 corpus

- PlanetMath: "fundamental group" — pi_1 and covering spaces
- PlanetMath: "Poincare duality" — duality for closed manifolds
- PlanetMath: "lattice in a Lie group" — discrete subgroups
- PlanetMath: "acyclic" — spaces with trivial reduced homology
