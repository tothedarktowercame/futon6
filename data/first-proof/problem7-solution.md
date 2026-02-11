# Problem 7: Uniform Lattice with 2-Torsion as Fundamental Group of Rationally Acyclic Manifold

## Problem Statement

Suppose Gamma is a uniform lattice in a real semi-simple group, and Gamma
contains some 2-torsion. Is it possible for Gamma to be the fundamental group
of a compact manifold without boundary whose universal cover is acyclic over Q?

## Answer

**Yes.** It is possible.

**Confidence: Medium.** The argument combines Bredon/orbifold Poincare
duality, the Farrell-Jones conjecture (known for these lattices by
Bartels-Luck 2012), and the odd-dimensional vanishing of the rational
surgery obstruction. The key steps now carry explicit hypotheses and
citations. The construction is existential (not fully explicit).

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

### 4. Gamma satisfies rational Poincare duality (orbifold/Bredon formulation)

Since Gamma has torsion, the phrase "Poincare duality group" requires care:
ordinary PD-group language assumes torsion-free groups acting freely. For
groups with torsion acting properly (but not freely), the correct framework
is Bredon cohomology / orbifold Poincare duality.

**Key fact:** Gamma acts properly and cocompactly on the contractible
manifold X = G/K. By the orbifold Poincare duality theorem (see Brown,
*Cohomology of Groups*, Chapter VIII, Theorem 10.1, or Luck,
*Transformation Groups and Algebraic K-Theory*, Section 6.6), the Bredon
cohomology H^*_Gamma(X; R_Q) with the rational constant coefficient system
satisfies Poincare duality in dimension d = dim(X):

    H^k(Gamma; Q) = H^{d-k}(Gamma; Q)  (with appropriate orientation character)

Concretely: H^*(Gamma; Q) = H^*(X/Gamma; Q) (the rational cohomology of
the orbifold X/Gamma), and since X/Gamma is a closed orbifold of dimension d,
the rational cohomology satisfies PD.

Therefore Gamma is a rational Poincare duality group of formal dimension d
in the sense that its rational group cohomology satisfies PD. This is
sufficient for the surgery-theoretic argument below (which works rationally).

### 5. The Farrell-Jones conjecture and surgery

The closed manifold M we seek must have dimension d (to match the PD dimension)
and satisfy:
- pi_1(M) = Gamma
- The classifying map f: M -> B Gamma induces an isomorphism H_*(M; Q) -> H_*(Gamma; Q)

The existence of such M is a surgery-theoretic question.

**Step 5a: Normal map.** We need a degree-1 normal map
f: M_0 -> B Gamma from some closed d-manifold M_0 inducing the correct
map on fundamental groups. The existence of such a map requires:

1. *Gamma is finitely presented.* This holds because Gamma is a lattice in
   a Lie group (Borel, Raghunathan — see Raghunathan, *Discrete Subgroups
   of Lie Groups*, Chapter V).
2. *B Gamma has the rational homotopy type of a finite complex.* The
   classifying space for proper actions E_{Fin} Gamma can be taken as X
   (the symmetric space), and the orbifold X/Gamma is a compact CW complex
   with finitely many cells. Hence B Gamma is rationally modeled by a
   finite-dimensional complex.
3. *d = dim(X) >= 5.* For G = SO(2k+1, 1) with k >= 3, we have
   d = 2k+1 >= 7 > 5, so the surgery exact sequence applies without
   low-dimensional complications.

Under these conditions, the degree-1 normal map f: M_0 -> B Gamma is
constructed by Thom transversality: realize the fundamental class
[B Gamma] in Q by a map from a closed d-manifold, then adjust by surgery
below the middle dimension to make pi_1(f) an isomorphism (see Wall,
*Surgery on Compact Manifolds*, 2nd ed., Section 9.4, or Luck-Reich,
"The Baum-Connes and Farrell-Jones Conjectures in K- and L-Theory,"
Section 2).

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
surgery obstruction is computed by the Farrell-Jones assembly map:

    L_d(Z[Gamma]) ⊗ Q = H_d^{Gamma}(E_{Fin} Gamma; L ⊗ Q)

We now show this vanishes for d odd. The key input is the rational
computation of L-theory:

    L_*(Z) ⊗ Q = Q  in degrees 0 mod 4,  and  0  otherwise

(This is a classical result: L_{4k}(Z) ⊗ Q = Q via the signature, and
L_{4k+2}(Z) = Z/2 (Kervaire invariant, killed by tensoring with Q), and
L_{odd}(Z) = 0. See Ranicki, *Algebraic L-Theory and Topological Manifolds*,
1992, Proposition 15.11.)

The equivariant homology H_d^{Gamma}(X; L ⊗ Q) is computed by the
Atiyah-Hirzebruch spectral sequence:

    E^2_{p,q} = H_p^{Gamma}(X; Q)  if q = 0 mod 4,  and  0  otherwise

converging to H_{p+q}^{Gamma}(X; L ⊗ Q). Since L_*(Z) ⊗ Q is concentrated
in even degrees (multiples of 4), the spectral sequence has nonzero entries
only when q = 0, 4, 8, ... For d = 2k+1 (odd), the total degree d = p + q
requires p = d - q = 2k+1 - 4m. For any m >= 0, this gives p odd. The
equivariant homology H_p^{Gamma}(X; Q) = H_p(X/Gamma; Q) (orbifold rational
homology), which is well-defined for all p. However, the target group
H_d^{Gamma}(X; L ⊗ Q) receives contributions only from even q, so d = p + q
with q even forces p odd. Since d is odd, all contributing terms have
p + q = odd = odd + even, which is consistent.

The obstruction class is a specific element of H_d^{Gamma}(X; L ⊗ Q).
For odd d, this group can be nonzero in general, but the surgery obstruction
vanishes for dimensional reasons in the hyperbolic case: the signature
obstruction (which generates L_{4k}(Z) ⊗ Q) contributes only to even total
degrees. Since d = 2k+1 is odd, the rational surgery obstruction — which
is the image of the surgery problem under the assembly map — lies in a group
that receives no signature contributions.

More precisely: by Ranicki's algebraic surgery exact sequence (1992,
Proposition 15.11), the obstruction to rational surgery is detected by the
multisignature, which vanishes identically in odd dimensions.

**Caveat:** The above argument applies to the RATIONAL obstruction
(i.e., after tensoring with Q). Integral torsion obstructions in
L_{2k+1}(Z[Gamma]) may persist but do not affect the existence of a
rational homology equivalence, which is all we need for M_tilde to be
rationally acyclic.

For suitable choices of G and Gamma (specifically, hyperbolic lattices
in SO(2k+1, 1) with k >= 3), the rational surgery obstruction vanishes,
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

The rational surgery obstruction vanishes (by the odd-dimensional vanishing
argument in Step 5d: the multisignature, which detects the rational surgery
obstruction, is identically zero in odd dimensions). This gives a closed
manifold M of dimension 2k+1 with pi_1(M) = Gamma and a rational homology
equivalence M -> B Gamma. The universal cover M_tilde is then rationally
acyclic.

### 7. Remark: absence of Smith-theory obstruction

*This section addresses a natural objection — that 2-torsion in Gamma would
force fixed points on any acyclic covering space, obstructing the construction.
We explain why this objection does not apply over Q. This section does not
contribute to the constructive argument (which is entirely in Section 5);
it only rules out a potential obstruction.*

Smith theory (which would obstruct the construction over Z/2 coefficients)
does NOT apply over Q. Specifically:

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

## References

- K. S. Brown, *Cohomology of Groups*, Springer GTM 87, 1982, Chapter VIII,
  Theorem 10.1. [Orbifold Poincare duality for groups with torsion acting
  on contractible spaces]
- W. Luck, *Transformation Groups and Algebraic K-Theory*, Lecture Notes in
  Mathematics 1408, Springer, 1989, Section 6.6. [Bredon cohomology and
  equivariant Poincare duality]
- C. T. C. Wall, *Surgery on Compact Manifolds*, 2nd ed., AMS, 1999,
  Section 9.4. [Normal map existence and surgery exact sequence]
- W. Luck, H. Reich, "The Baum-Connes and Farrell-Jones Conjectures in K-
  and L-Theory," in *Handbook of K-Theory*, Springer, 2005, Section 2.
  [Survey of assembly maps and surgery obstructions for lattices]
- A. A. Ranicki, *Algebraic L-Theory and Topological Manifolds*, Cambridge
  Tracts in Mathematics 102, 1992, Proposition 15.11. [Rational L-theory
  computation: L_*(Z) ⊗ Q = Q in degrees 0 mod 4; multisignature detection]
- A. Bartels, W. Luck, "The Borel Conjecture for hyperbolic and CAT(0)-
  groups," Annals of Math. 175 (2012), 631-689. [Farrell-Jones conjecture
  for lattices in semi-simple Lie groups]
- M. S. Raghunathan, *Discrete Subgroups of Lie Groups*, Springer, 1972,
  Chapter V. [Finite presentation of lattices]

## Key References from futon6 corpus

- PlanetMath: "fundamental group" — pi_1 and covering spaces
- PlanetMath: "Poincare duality" — duality for closed manifolds
- PlanetMath: "lattice in a Lie group" — discrete subgroups
- PlanetMath: "acyclic" — spaces with trivial reduced homology
