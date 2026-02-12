# Problem 7: Uniform Lattice with 2-Torsion and Rationally Acyclic Universal Cover

## Problem Statement

Suppose `Gamma` is a uniform lattice in a real semisimple Lie group, and
`Gamma` contains an element of order `2`. Is it possible that `Gamma` is the
fundamental group of a closed manifold whose universal cover is acyclic over
`Q`?

## Status in This Writeup

**Answer: conditional yes.**

- Obligation (E2) — placing `Gamma` in `FH(Q)` — is **discharged
  unconditionally** for a concrete reflection-lattice family.
- Obligation (S) — upgrading from finite CW complex to closed manifold —
  is **open**. We describe the available geometric data and three candidate
  approaches, each with unresolved obstacles.

## 1. Baseline Geometry

Let `G` be connected real semisimple, `K < G` maximal compact, and
`X = G/K` contractible. For a uniform lattice `Gamma < G` with torsion,
`X/Gamma` is a compact orbifold (not a manifold).

So the torsion-free argument `M = X/Gamma` does not apply directly.

## 2. Cohomological Structure via Bredon Framework

Since `Gamma` acts properly and cocompactly on the contractible space `X`,
the Bredon cohomology `H^*_Gamma(X; R_Q)` (with the rational constant
coefficient system) satisfies Poincare duality by the orbifold PD theorem
(Brown, *Cohomology of Groups*, Chapter VIII; Luck, *Transformation Groups
and Algebraic K-Theory*, Section 6.6).

Concretely, `H^*(Gamma; Q) = H^*(X/Gamma; Q)` satisfies `H^k = H^{d-k}`
where `d = dim(X)`.

**Note on terminology.** With torsion present, saying "`Gamma` is a rational
PD group" requires this Bredon/orbifold interpretation. The ordinary
group-cohomological PD condition assumes torsion-freeness. Throughout this
writeup, "rational Poincare duality" for `Gamma` with torsion refers to the
Bredon-equivariant formulation above.

## 3. Obligation E2: Finite-CW Realization (`Gamma in FH(Q)`)

### 3a. Fowler's Criterion

A theorem of Fowler (arXiv:1204.4667, Main Theorem) gives a concrete
criterion: if a finite group `G` acts on a finite CW complex `Y` such that
for every nontrivial subgroup `H < G` and every connected component `C` of
the fixed set `Y^H`, the Euler characteristic `chi(C) = 0`, then the
orbifold extension group `pi_1((EG x Y)/G)` lies in `FH(Q)`.

That is, there exists a finite CW complex with the given fundamental group
whose universal cover is rationally acyclic.

### 3b. Concrete Instantiation via Reflection Lattice

The E2 obligation is discharged by the following construction (details in
`problem7r-s2b-candidate-construction.md`):

1. **Arithmetic lattice with reflection.** Take an arithmetic uniform lattice
   `Gamma_0 < Isom(H^n)` containing a reflection `tau`, as provided by
   Douba-Vargas Pallete (arXiv:2506.23994, Remark 5). Choose `n` **even**
   (e.g., `n = 4` or `n = 6`).

2. **Congruence cover.** Let `pi = Gamma_0(I)` be a sufficiently deep
   principal congruence subgroup. Then `M = pi \ H^n` is a closed hyperbolic
   manifold, and `tau` induces an involution `tau_bar` on `M`.

3. **Extension.** Set `G = <tau_bar> = Z/2` acting on `Bpi := M`.
   The orbifold extension gives `1 -> pi -> Gamma -> Z/2 -> 1`, where
   `Gamma` is a cocompact lattice (finite extension of cocompact `pi`) with
   order-2 torsion.

4. **Fixed-set Euler check.** The fixed set `Fix(tau_bar)` is a (possibly
   disconnected) closed, embedded, totally geodesic hypersurface
   (arXiv:2506.23994). Each component has dimension `n-1`. Since `n` is even,
   `n-1` is odd, and every closed odd-dimensional manifold has Euler
   characteristic zero. So `chi(C) = 0` for every fixed component `C`.

5. **Fowler application.** The only nontrivial subgroup of `Z/2` is itself.
   All fixed components have zero Euler characteristic. By Fowler's Main
   Theorem, `Gamma in FH(Q)`.

**E2 status: discharged** for this lattice family.

## 4. Obligation S: From Finite Complex to Closed Manifold (Open)

Problem 7 asks for a **closed manifold** `M` with `pi1(M) = Gamma` and
`H_*(M_tilde; Q) = 0` for `* > 0`. Obligation E2 gives a finite CW
complex with these properties; the upgrade to a closed manifold is the
remaining open problem.

### Available geometric data

The construction in Section 3b provides:

- `pi`: a torsion-free cocompact lattice in `Isom(H^n)`, with `n` even
  `>= 6`.
- `M = H^n/pi`: a closed hyperbolic `n`-manifold. `M` is a closed manifold
  with `pi_1(M) = pi` and contractible universal cover.
- `tau`: an involution on `M` with fixed set `F` — a closed, totally
  geodesic `(n-1)`-manifold.
- `Gamma = pi rtimes Z/2`: the target group (cocompact lattice with
  order-2 torsion).
- `Y`: a finite CW complex with `pi_1(Y) = Gamma` and `Y~` rationally
  acyclic (from Fowler's theorem).
- `H^n/Gamma`: a compact orbifold with `pi_1^{orb} = Gamma`, mirror
  singularity along the image of `F`.

The torsion-free quotient `M` already solves the problem for `pi`. The
difficulty is entirely in passing from `pi` to `Gamma` — from the
torsion-free lattice to its Z/2-extension.

### Approach I: Wall surgery via the FH(Q) complex

**Idea.** Use `Y` (the FH(Q) complex) as the target of a surgery problem.
Promote `Y` to a rational Poincare complex, find a degree-1 normal map to
it, and show the surgery obstruction vanishes.

**Obstacles.**

1. *Poincare complex structure.* FH(Q) gives a finite CW complex `Y` with
   rationally acyclic universal cover, but not a Poincare complex. Since
   `Y~` is rationally contractible (by rational Hurewicz from rational
   acyclicity + simple connectivity), the Serre spectral sequence collapses
   and `H_*(Y; Q) = H_*(Gamma; Q)`, which satisfies rational PD
   (Section 2). But promoting homology-level PD to a chain-level Poincare
   complex structure on `Y` has not been done.

2. *Degree-1 normal map.* Even with Poincare complex structure, a degree-1
   normal map `f: M_0 -> Y` requires the Spivak normal fibration of `Y` to
   admit a topological reduction. No construction of this reduction has been
   given.

3. *Surgery obstruction.* The obstruction `sigma(f) in L_n(Z[Gamma]) tensor Q`
   is not known to vanish. The Farrell-Jones conjecture (which holds for
   `Gamma`) identifies this L-group with equivariant L-homology, but the
   resulting computation has not been completed. See `problem7r-s3b-obstruction.md`
   for the FJ reduction framework and a conjectural (but unverified)
   localization of the obstruction.

**Status.** This approach has three successive obstacles, each open.
See `problem7r-s3a-setup.md` for the detailed analysis.

### Approach II: Equivariant surgery on (M, tau)

**Idea.** Work directly with the closed manifold `M` and the involution
`tau`. Eliminate the fixed set `F` by equivariant surgery to obtain `M'`
with a free Z/2-action. Then `M'/(Z/2)` is a closed manifold with
`pi_1 = Gamma`.

**Potential advantages.** Bypasses the FH(Q) complex entirely — no need
for Poincare complex structure or Spivak reduction. Works with the concrete
geometric data (M, tau, F).

**Obstacles.**

1. *Equivariant surgery obstruction.* The obstruction to eliminating
   a codimension-1 fixed set by equivariant surgery is nontrivial
   (Dovermann-Schultz, *Equivariant Surgery Theories*, Springer LNM 1443).
   No computation has been done for this specific lattice action.

2. *pi_1 control.* Equivariant surgery modifying `M` near `F` must
   preserve the property that the quotient has `pi_1 = Gamma`. This
   requires tracking the effect of equivariant handle operations on the
   orbifold fundamental group.

3. *Rational acyclicity.* The modified `M'` must still have rationally
   acyclic universal cover. If the equivariant surgery only changes `M`
   rationally trivially (e.g., by rational cobordism), this is preserved.

**Status.** Not explored beyond this sketch.

### Approach III: Orbifold resolution

**Idea.** The quotient `H^n/Gamma` is a compact orbifold with mirror
singularity. Resolve the singularity to produce a closed manifold with
`pi_1 = Gamma` and rationally acyclic universal cover.

**Obstacles.**

1. *pi_1 preservation.* Standard orbifold resolution (e.g., cutting along
   the mirror and doubling) typically changes the fundamental group.
   A resolution preserving `pi_1 = Gamma` would need to be specifically
   constructed.

2. *Rational acyclicity.* The resolution must preserve (or establish)
   rational acyclicity of the universal cover.

**Status.** Not explored beyond this sketch.

### Structural observation: dimension-parity tension

The E2 obligation requires `n` **even** (so the fixed set has odd dimension
and Euler characteristic zero). But the surgery obstruction computation
(Approach I) and the AHSS structure of `L_n(Z[Gamma]) tensor Q` have better
vanishing properties when `n` is **odd** (the 4-periodicity of rational
L-theory forces `p` odd in the contributing `E^2_{p,q}` terms). No
vanishing result for odd `p` has been proved for these lattices, but the
structural favorability is real. This tension between E2 and S is a central
difficulty.

An alternative would be to find a lattice construction where E2 works in
odd dimension (fixed set with even dimension but zero Euler characteristic).
This has not been attempted.

## 5. Remark: Absence of Smith-Theory Obstruction

A natural objection is that the order-2 element in `Gamma` would force fixed
points on any rationally acyclic covering space via Smith theory. This does
**not** apply here: Smith theory over `Z/p` constrains mod-p homology of
fixed sets, but the construction targets `Q`-acyclicity. Over `Q`, the
transfer homomorphism shows that fixed sets can be rationally trivial without
contradicting Smith's theorem.

This section addresses a natural objection and explains why it does not
apply. It does not contribute to the constructive argument, which is entirely
in Sections 3-4.

## 6. Theorem

**Theorem.** Let `Gamma` be the cocompact lattice extension
`1 -> pi -> Gamma -> Z/2 -> 1` constructed in Section 3b from an arithmetic
reflection lattice in `Isom(H^n)` with `n` even, `n >= 6`. Then:

(a) **(Unconditional)** `Gamma in FH(Q)`: there exists a finite CW complex
    `Y` with `pi_1(Y) = Gamma` and `H_*(Y_tilde; Q) = 0` for `* > 0`.

(b) **(Open)** Whether there exists a **closed manifold** `M` with
    `pi_1(M) = Gamma` and `H_*(M_tilde; Q) = 0` for `* > 0` remains
    unresolved. Three approaches to the manifold-upgrade problem are
    described in Section 4; none has been completed.

## 7. Path to Full Closure

Resolving obligation S requires completing any one of the three approaches
in Section 4. The most concrete next steps:

1. **For Approach I (Wall surgery):** resolve the Poincare complex structure
   (obstacle 1), ideally by finding a reference for chain-level PD promotion
   of finite complexes with rationally contractible universal covers.

2. **For Approach II (equivariant surgery):** compute the equivariant
   surgery obstruction to eliminating the fixed set F in (M, tau) for the
   specific lattice. This requires engaging with the Dovermann-Schultz
   framework.

3. **For all approaches:** investigate whether an odd-dimensional E2
   construction exists (lattice with 2-torsion where the fixed set has
   even dimension but vanishing Euler characteristic), which would ease the
   surgery obstruction computation.

## References

- J. Fowler, *Finiteness Properties of Rational Poincare Duality Groups*,
  arXiv:1204.4667.
- G. Avramidi, *Rational Manifold Models for Duality Groups*,
  arXiv:1506.06293.
- A. Douba, F. Vargas Pallete, *On Reflections of Congruence Hyperbolic
  Manifolds*, arXiv:2506.23994.
- A. Bartels, F. T. Farrell, W. Luck, *The Farrell-Jones Conjecture for
  Cocompact Lattices in Virtually Connected Lie Groups*, arXiv:1101.0469.
- A. Bartels, W. Luck, *The Farrell-Jones Conjecture for Arbitrary
  Lattices in Virtually Connected Lie Groups*, arXiv:1401.0876.
- D. Crowley, W. Luck, T. Macko, *Surgery Theory: Foundations*,
  arXiv:0905.0104.
- K. H. Dovermann, R. Schultz, *Equivariant Surgery Theories and Their
  Periodicity Properties*, Springer LNM 1443, 1990.
- A. Ranicki, *Algebraic L-Theory and Topological Manifolds*, Cambridge
  Tracts in Mathematics 102, 1992.
- K. S. Brown, *Cohomology of Groups*, Springer GTM 87.
- W. Luck, *Transformation Groups and Algebraic K-Theory*, Springer LNM
  1408.
- C. T. C. Wall, *Surgery on Compact Manifolds*, 2nd ed., AMS.
