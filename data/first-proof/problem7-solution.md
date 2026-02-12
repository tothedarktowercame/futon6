# Problem 7: Uniform Lattice with 2-Torsion and Rationally Acyclic Universal Cover

## Problem Statement

Suppose `Gamma` is a uniform lattice in a real semisimple Lie group, and
`Gamma` contains an element of order `2`. Is it possible that `Gamma` is the
fundamental group of a closed manifold whose universal cover is acyclic over
`Q`?

## Status in This Writeup

**Answer: conditional yes.** Obligation (E2) — placing `Gamma` in `FH(Q)` —
is discharged unconditionally for a concrete reflection-lattice family.
Obligation (S) — upgrading from finite CW complex to closed manifold — is
conditional on three items: (G1) that the FH(Q) complex admits rational
Poincare complex structure, (G2) that its Spivak normal fibration admits a
topological reduction, and (S) that the rational surgery obstruction
vanishes. The S-branch analysis is outlined below.

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

## 4. The Remaining Gap: Obligation S (Manifold Upgrade)

Problem 7 asks for a **closed manifold** `M` with `pi1(M) = Gamma` and
`H_*(M_tilde; Q) = 0` for `* > 0`. Obligation E2 gives a finite CW complex
with these properties. The upgrade to a closed manifold requires:

1. A surgery setup producing a degree-1 normal map to a finite Poincare
   complex representing `Gamma`.
2. Vanishing (or controlled membership) of the surgery obstruction in the
   appropriate L-group.

### 4a. Surgery Setup Interface (p7r-s3a)

`Gamma` is finitely presented (it is a lattice in a Lie group). The finite
CW complex `Y` from E2 has `pi_1(Y) = Gamma` and `Y~` rationally acyclic.

**Gap (G1): `Y` is not automatically a Poincare complex.** FH(Q) gives a
finite CW complex with rationally acyclic universal cover, but does not
directly endow `Y` with a (rational) Poincare complex structure (fundamental
class + cap product duality). A separate argument is needed:

- `Y~` is simply connected (universal cover) and rationally acyclic, so by
  rational Hurewicz, `pi_*(Y~) tensor Q = 0` for all `* > 0`, meaning `Y~`
  is rationally contractible.
- The Serre spectral sequence for `Y~ -> Y -> BGamma` then collapses,
  giving `H_*(Y; Q) = H_*(Gamma; Q)`.
- `Gamma` has rational PD in dimension `n` (Section 2, Bredon framework).
- So `H_*(Y; Q)` satisfies rational Poincare duality.

However, this shows PD on homology groups, not that `Y` admits a geometric
fundamental class with cap product duality at the chain level. Promoting
this to a rational Poincare complex structure on `Y` requires either:
(a) an explicit chain-level duality construction (e.g., via the equivariant
    diagonal on `Y~`), or
(b) a reference showing that finite CW complexes with rationally
    contractible universal cover and rational-PD fundamental group inherit
    rational Poincare complex structure.

**This gap is currently open.** We proceed conditionally on `Y` admitting
rational Poincare complex structure.

For the surgery exact sequence to apply (Wall, *Surgery on Compact
Manifolds*, Chapter 9), we need (assuming G1 resolved):

- `Gamma` finitely presented: **yes** (lattice).
- A finite `d`-dimensional rational Poincare complex: **open** (G1).
- `d >= 5`: **yes** when `n >= 6` (guaranteed by choosing `G = SO(n,1)` with
  `n = 6` or any even `n >= 6`).

**Gap (G2): Existence of degree-1 normal map.** For Wall's machinery, a
degree-1 normal map `f: M_0 -> Y` requires the Spivak normal fibration of
`Y` to admit a topological reduction (lift to a stable vector bundle). The
previous version claimed this follows by transfer from the double cover
`Y_pi ~ M`, but that descent is nontrivial: the stable normal bundle of `M`
does not automatically descend to a topological reduction of the Spivak
fibration of `Y` without an explicit equivariant lifting argument.

**This gap is currently open.** It requires either:
(a) a direct construction of the topological reduction of `nu_Y`, or
(b) an equivariant bordism argument showing the Z/2-equivariant normal
    structure on `M` descends to `Y`, or
(c) bypassing Spivak entirely by working within Avramidi's rational surgery
    framework (arXiv:1506.06293), which may handle the normal map
    construction differently for rational PD inputs.

**pi_1 preservation.** Assuming a degree-1 normal map exists, surgery below
the middle dimension preserves `pi_1` by general position when `d >= 5`
(Wall, Proposition 1.2). See `problem7r-s3a-setup.md` for the detailed
interface specification.

### 4b. Obstruction Computation (p7r-s3b)

The surgery obstruction lives in `L_d(Z[Gamma])`. By the Farrell-Jones
isomorphism (Bartels-Farrell-Luck, arXiv:1101.0469, for cocompact lattices
in virtually connected Lie groups), this reduces to an equivariant homology
computation:

`L_d(Z[Gamma]) tensor Q  ~=  H_d^{Gamma}(E_{VCyc}Gamma; L tensor Q)`

where `E_{VCyc}Gamma` is the classifying space for the family of virtually
cyclic subgroups.

The Atiyah-Hirzebruch spectral sequence for the equivariant homology
`H_*^{Gamma}(E_{VCyc}Gamma; L tensor Q)` has `E^2_{p,q}` terms involving
`H_p^{Gamma}(E_{VCyc}Gamma; L_q(Z) tensor Q)`. The rational L-theory
spectrum satisfies `L_q(Z) tensor Q = Q` for `q = 0 mod 4` and `0`
otherwise. The spectral sequence collapses rationally (Ranicki, *Algebraic
L-Theory and Topological Manifolds*, Proposition 15.11).

For `d = n = 6`: the contributing terms are `E^2_{p,q}` with `p + q = 6`
and `q = 0 mod 4`, giving `(p,q) = (6,0)` and `(2,4)`. These terms are
potentially nonzero and the obstruction class lies in their sum.

**This does not automatically vanish.** The earlier claim of "vanishing by
parity" was too coarse. Instead:

- For `d` odd (achievable if we work with `d = n + 1` via a product
  `Gamma x Z` acting on `X x R`, or by choosing an odd-dimensional lattice
  variant), the relevant `(p,q)` pairs with `p + q` odd and `q = 0 mod 4`
  force `p` odd, and the obstruction is valued in `H_{odd}` of the
  classifying space. This is heuristically favorable for vanishing, but a
  proof that `H_{odd}` terms vanish for the specific lattices under
  consideration has not been supplied and should not be assumed.

- For `d = 6` as stated, the obstruction is potentially nonzero and requires
  either:
  (a) An explicit computation showing the assembly map sends the obstruction
      class to zero in this case, or
  (b) A dimension shift: work with `Gamma x Z` acting on `X x R` to move to
      odd total dimension `7` where the obstruction vanishes, then extract
      the closed `6`-manifold via restriction.

**Current status.** The obstruction computation for the specific
Douba-Vargas Pallete lattice in dimension 6 is not yet complete. The proof
is **conditional on vanishing of the rational surgery obstruction
`sigma(f) in L_6(Z[Gamma]) tensor Q`** for the degree-1 normal map `f`
from Section 4a.

See `problem7r-s3b-obstruction.md` for the detailed computation attempt and
remaining gaps.

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

## 6. Theorem (Conditional)

**Theorem.** Let `Gamma` be the cocompact lattice extension
`1 -> pi -> Gamma -> Z/2 -> 1` constructed in Section 3b from an arithmetic
reflection lattice in `Isom(H^n)` with `n` even, `n >= 6`. Then:

(a) **(Unconditional)** `Gamma in FH(Q)`: there exists a finite CW complex
    `Y` with `pi_1(Y) = Gamma` and `H_*(Y_tilde; Q) = 0` for `* > 0`.

(b) **(Conditional on G1 + G2 + S)** If:
    - (G1) `Y` admits rational Poincare complex structure,
    - (G2) the Spivak normal fibration of `Y` admits a topological
      reduction, and
    - (S) the rational surgery obstruction `sigma in L_n(Z[Gamma]) tensor Q`
      vanishes for the resulting degree-1 normal map,
    then there exists a **closed manifold** `M` with
    `pi_1(M) = Gamma` and `H_*(M_tilde; Q) = 0` for `* > 0`.

The answer to Problem 7 is **yes** assuming conditions (G1), (G2), and (S).

## 7. Path to Full Closure

To remove the conditional:

1. **Dimension selection.** Determine whether an odd manifold dimension can
   be achieved for this lattice family (which would give automatic vanishing
   of the rational obstruction via the `L_q` periodicity argument).

2. **Direct computation.** For `n = 6`: compute the assembly image of
   `sigma` in `L_6(Z[Gamma]) tensor Q` using the Farrell-Jones reduction
   and the specific structure of `Gamma` as a `Z/2`-extension of a
   hyperbolic lattice.

3. **Closed-manifold subgroup.** Apply Crowley-Luck-Macko
   (arXiv:0905.0104, Theorems A/B) to determine whether the assembly image
   lies in the closed-manifold realization subgroup for the given decorations
   and dimension.

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
- A. Ranicki, *Algebraic L-Theory and Topological Manifolds*, Cambridge
  Tracts in Mathematics 102, 1992.
- K. S. Brown, *Cohomology of Groups*, Springer GTM 87.
- W. Luck, *Transformation Groups and Algebraic K-Theory*, Springer LNM
  1408.
- C. T. C. Wall, *Surgery on Compact Manifolds*, 2nd ed., AMS.
