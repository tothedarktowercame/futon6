# Problem 7 Reduced: `p7r-s3b` Obstruction Computation and Closure

Date: 2026-02-12

## Target Node

`p7r-s3b`: compute or cite the rational surgery obstruction for the
concrete lattice family from `p7r-s2b`, and determine whether it vanishes.

## Input from upstream

From `p7r-s3a` (open — two gaps remain):

- `Gamma = pi rtimes Z/2`, where `pi` is torsion-free cocompact in
  `Isom(H^n)`, `n` even `>= 6`. Concrete choice: `n = 6`.
- Surgery prerequisites partially verified: finite presentation and
  dimension confirmed; rational Poincare complex structure (G1) and
  degree-1 normal map existence (G2) are open.
- **Conditional assumption for this document**: we assume G1 and G2 are
  resolved, so that a degree-1 normal map `f: M_0 -> Y` exists and the
  surgery obstruction `sigma(f) in L_6(Z[Gamma]) tensor Q` is well-defined.
- Farrell-Jones conjecture holds for `Gamma` (Bartels-Farrell-Luck,
  arXiv:1101.0469).

Additional geometric data:

- `M = H^6/pi` is a closed hyperbolic 6-manifold.
- The involution `tau` acts on `M` with fixed set `F = H^5/pi_F`, a closed
  totally geodesic hypersurface (5-manifold).
- `pi_F = Stab_pi(H^5)` is the subgroup of `pi` preserving the fixed
  hyperplane.

## Step 1: Farrell-Jones reduction

The FJ isomorphism gives:

```
L_6(Z[Gamma]) tensor Q  ~=  H_6^{Gamma}(E_{VCyc}Gamma; L tensor Q)
```

where `E_{VCyc}Gamma` is the classifying space for the family of virtually
cyclic subgroups.

### Comparison: `E_{Fin}` vs `E_{VCyc}`

The map `E_{Fin}Gamma -> E_{VCyc}Gamma` induces:

```
H_6^{Gamma}(E_{Fin}Gamma; L tensor Q) -> H_6^{Gamma}(E_{VCyc}Gamma; L tensor Q)
```

The cofiber contributes UNil terms from infinite virtually cyclic subgroups
of `Gamma`. The relevant infinite virtually cyclic subgroups are:

- Type I (infinite cyclic `Z`): these are subgroups of the torsion-free `pi`.
  For type I, UNil contributions to L-theory are zero
  (Ranicki, *Algebraic L-Theory and Topological Manifolds*).

- Type II (infinite dihedral `D_infty = Z rtimes Z/2`): these arise from
  infinite cyclic subgroups of `pi` whose normalizer in `Gamma` contains
  the Z/2 factor. The UNil terms for L-theory of the infinite dihedral group
  satisfy: `UNil_n(Z; Z, Z)` is a 2-primary torsion group for all `n`
  (Connolly-Davis, *Surgery obstruction groups for... infinite dihedral
  group*, 2004; see also Connolly-Ranicki, 2005).

**Key fact.** Since UNil terms are 2-torsion, they vanish after
rationalization:

```
UNil_n(Z; Z, Z) tensor Q = 0  for all n.
```

**Consequence.** The comparison map is a rational isomorphism:

```
H_6^{Gamma}(E_{Fin}Gamma; L tensor Q) ~= H_6^{Gamma}(E_{VCyc}Gamma; L tensor Q)
```

So we may compute with `E_{Fin}Gamma` instead of `E_{VCyc}Gamma`.

## Step 2: Model for `E_{Fin}Gamma`

`H^6` (hyperbolic 6-space) with the proper, cocompact `Gamma`-action is a
model for `E_{Fin}Gamma`: every finite subgroup of `Gamma` has a
non-empty contractible fixed-point set in `H^6`, and the action is proper.

So:

```
L_6(Z[Gamma]) tensor Q ~= H_6^{Gamma}(H^6; L tensor Q)
```

Since `pi` acts freely on `H^6`, we can quotient by `pi` first:

```
H_6^{Gamma}(H^6; L tensor Q) = H_6^{Z/2}(M; L tensor Q)
```

where `M = H^6/pi` is the closed hyperbolic 6-manifold and `Z/2 = Gamma/pi`
acts on `M` by the involution `tau` with fixed set `F`.

**This is the key simplification:** the L-theoretic computation for `Gamma`
reduces to a Z/2-equivariant L-homology computation on the manifold `M`.

## Step 3: Transfer argument

The transfer (restriction) map:

```
res: L_6(Z[Gamma]) tensor Q -> L_6(Z[pi]) tensor Q
```

corresponds to the "forget equivariance" map:

```
H_6^{Z/2}(M; L tensor Q) -> H_6(M; L(Z) tensor Q)
```

For `pi` (torsion-free, satisfying FJ), the assembly map gives:

```
L_6(Z[pi]) tensor Q ~= H_6(M; L(Z) tensor Q) ~= H_6(M; Q) + H_2(M; Q)
```

(using the AHSS collapse: `L_q(Z) tensor Q = Q` for `q = 0 mod 4`, else
`0`; contributing terms `(p,q) = (6,0)` and `(2,4)` for total degree 6).

**Intended argument.** The restriction of `sigma(f)` to `pi`:

```
res(sigma(f)) in L_6(Z[pi]) tensor Q
```

should be the surgery obstruction of the **restricted** surgery problem on
the double cover `Y_pi` (the cover of `Y` corresponding to `pi < Gamma`).

If the normal map `f: M_0 -> Y` is constructed compatibly with the covering
structure — meaning that the restriction of `f` to the double cover gives
a normal map `f_pi: M_{0,pi} -> Y_pi` that is normally cobordant to the
identity on `M` — then `res(sigma(f)) = 0`.

**Gap (G3): compatibility of the normal map with the cover.** The previous
version asserted `res(sigma(f)) = 0` from `Y_pi ~ M`, but this requires:

1. That `Y_pi` is rationally homotopy equivalent to `M` (established: both
   have rationally contractible universal cover and `pi_1 = pi`, so they
   agree in rational homotopy type).
2. That the specific normal map `f` restricts on the double cover to a
   normal map whose surgery obstruction is zero. For an **arbitrary** choice
   of `f`, this need not hold — only a normal map constructed compatibly
   with the covering data will have this property.

If `f` is constructed via approach (c) of G2 (equivariant bordism descent
from `M`), then the restriction to `Y_pi` is by construction cobordant to
the identity on `M`, and `res(sigma(f)) = 0` follows.

**Conditional conclusion.** If the normal map `f` is constructed compatibly
with the covering (which is part of resolving G2), then
`sigma(f) in ker(res)`.

## Step 4: Conjectural structure of `ker(res)`

**Status: the following analysis is a sketch, not a proof. The claimed
decomposition relies on several intermediate steps that are not yet
verified with references. It should be treated as a heuristic guide to
the expected obstruction structure, not as an established result.**

### Sketch of the decomposition

The equivariant L-homology `H_6^{Z/2}(M; L tensor Q)` should decompose
(via the isotropy stratification of the Z/2-action on `M`) into:

- **Free stratum**: contributions from `M \ F` where Z/2 acts freely.
  This should map (up to factor 2) to `L_6(Z[pi])^{Z/2} tensor Q`
  under the transfer.

- **Fixed stratum**: contributions from the fixed set `F` with its
  Z/2-isotropy. This should map to zero under the transfer.

If this decomposition holds, then `ker(res) = (fixed stratum contribution)`.

### Unverified claims in the decomposition

**(U1) Clean free/fixed splitting.** The decomposition of equivariant
L-homology into free and fixed strata uses the isotropy filtration of the
Z/2-action on `M`. For genuine equivariant homology theories (as opposed to
Borel equivariant), this filtration gives a long exact sequence, but the
claimed direct-sum splitting requires the connecting maps to vanish. This
needs verification (e.g., via analysis of the relevant extension groups).

**(U2) Equivariant Thom isomorphism.** The fixed stratum contribution
involves the equivariant Thom isomorphism for the normal bundle `nu_F`
(the 1-dimensional sign representation). The expected formula is:

```
fixed contribution to H_6^{Z/2}(M; L tensor Q)
    ~= H_5(F; L^{R^-}(Z[Z/2]) tensor Q)
```

where `L^{R^-}` denotes L-theory twisted by the sign representation. The
Thom isomorphism in equivariant L-theory is established in principle
(see Luck, *Transformation Groups and Algebraic K-Theory*), but the
specific coefficient computation needs careful verification.

**(U3) Twisted L-theory coefficients.** The computation uses:

- `Q[Z/2] = Q x Q` (trivial and sign representations)
- Trivial factor: `L_q(Q) = Q` for `q = 0 mod 4`, else `0`.
- Sign factor: `L_q^{w}(Q) = Q` for `q = 2 mod 4`, else `0`
  (where `w` is the nontrivial orientation character).

The claim about `L_q^w(Q)` — that twisting the orientation character shifts
the nonvanishing degrees by 2 — is standard for symmetric L-theory over
fields of characteristic 0, but needs an explicit reference (e.g., Ranicki,
*Algebraic L-Theory*, Chapter 13, or Lueck-Schick for the equivariant
version).

### Conjectural AHSS computation

Assuming (U1)-(U3), the AHSS for `H_5(F; L^{R^-}(Z[Z/2]) tensor Q)`
collapses rationally with contributing terms for `p + q = 5`:

| (p, q) | Coefficient | E^2 term |
|---------|-------------|----------|
| (5, 0)  | Q (trivial factor) | `H_5(F; Q) = Q` (fundamental class) |
| (3, 2)  | Q (sign factor) | `H_3(F; Q)` |
| (1, 4)  | Q (trivial factor) | `H_1(F; Q)` |

This would give:

```
ker(res) ~= Q + H_3(F; Q) + H_1(F; Q)  (CONJECTURAL)
```

where `F` is the closed hyperbolic 5-manifold (the fixed set of `tau` on
`M`).

**This formula is a target, not a theorem.** Verification of (U1)-(U3)
would establish it. Without that verification, the claimed localization
is unreliable and the actual obstruction space could be different.

## Step 5: Assessment of `sigma(f)`

Conditional on resolving the upstream gaps (G1, G2, G3) and the unverified
claims (U1-U3):

1. `sigma(f) in ker(res)` (by the transfer argument, Step 3 — requires G3).
2. `ker(res) ~= Q + H_3(F; Q) + H_1(F; Q)` (conjectural, Step 4 — requires
   U1-U3).

If both hold, then `sigma(f)` is a specific element of the conjectural
target space `Q + H_3(F; Q) + H_1(F; Q)`.

### Does `sigma(f) = 0`?

This is **not established**. The obstruction might be nonzero. Its value
depends on:

- The `Q` component (from `H_5(F; Q)`): this is a scalar, related to the
  "equivariant signature defect" at the fixed set. For a reflection on a
  hyperbolic manifold, this is related to the eta-invariant of `F`.

- The `H_3(F; Q)` component: depends on the topology of `F` and the
  specific geometry of the normal map.

- The `H_1(F; Q)` component: depends on `b_1(F)`, the first Betti number
  of the fixed hypersurface.

### Lower bound on the obstruction space

For a generic closed hyperbolic 5-manifold `F`:

- `H_1(F; Q)` can be nonzero (depends on the lattice `pi_F`).
- `H_3(F; Q) ~= H^2(F; Q) ~= H_3(F; Q)` by Poincare duality; can be
  nonzero.

So `ker(res)` is generically nonzero, and `sigma(f)` could land in a
nontrivial subspace.

## Step 6: The dimension-parity tension

The core difficulty is a tension between the two proof obligations:

| Obligation | Requires | Reason |
|------------|----------|--------|
| E2 (Fowler criterion) | `n` **even** | Fixed set has dim `n-1` odd, forcing `chi = 0` |
| S (surgery obstruction) | `n` **odd** preferred | `L_n` rational AHSS: odd total degree forces odd `p` in `E^2_{p,q}` with `q = 0 mod 4`, giving better vanishing |

For `n` odd, the surgery obstruction would live in `H_n^{Z/2}(M; L tensor Q)`
with `n` odd. The free-stratum contribution would involve
`H_n(M; Q) + H_{n-4}(M; Q) + ...` at odd degrees, and the 4-periodic
structure of rational L-theory would force many terms to vanish. The
fixed-stratum Thom isomorphism would shift to `H_{n-1}(F; ...)` with `n-1`
even, and the contributing AHSS terms would also be more constrained.

In contrast, for `n = 6` (even), both the free stratum and the fixed
stratum have nonzero contributions, and no parity argument eliminates them.

**This tension is the fundamental remaining obstacle in Problem 7.**

## Step 7: Resolution strategies

### Strategy A: Direct computation for specific lattice

Choose a specific arithmetic reflection lattice (from Douba-Vargas Pallete)
with `n = 6` and compute:

1. The Betti numbers `b_1(F)`, `b_3(F)` of the fixed hypersurface `F`.
2. The specific value of `sigma(f)` in `Q + H_3(F; Q) + H_1(F; Q)`.

This requires explicit knowledge of the lattice and its fixed set. For
arithmetic hyperbolic manifolds, Betti numbers can sometimes be computed
using automorphic forms (Bergeron-Venkatesh, arXiv:1212.3847).

**Difficulty:** high. Requires number-theoretic input.

### Strategy B: Lattice selection with `b_*(F) = 0`

Find a specific congruence subgroup where the fixed hypersurface `F` has
`H_1(F; Q) = H_3(F; Q) = 0`. Then `ker(res) = Q`, and only the scalar
component (equivariant signature defect) remains.

By results of Bergeron-Venkatesh, the ratio `b_k(F)/vol(F)` tends to the
`L^2`-Betti number `b_k^{(2)}(H^5)` as the congruence level deepens. For
`H^5`:

- `b_k^{(2)}(H^5) = 0` for `k != 5/2` (non-integer, so all `L^2`-Betti
  numbers vanish for odd-dimensional hyperbolic space).

So `b_k(F)/vol(F) -> 0` for all `k`. But this gives sublinear growth, not
vanishing. Individual Betti numbers can still be nonzero.

**Difficulty:** moderate. Might be achievable with careful lattice selection,
but not guaranteed.

### Strategy C: Equivariant signature defect vanishing

Even if `ker(res) != 0`, show that the specific surgery obstruction
`sigma(f)` is zero by computing the equivariant signature defect.

The `Q` component of `sigma(f)` is related to:

```
sigma_equivariant = Sig_tau(M) - (correction from normal map)
```

where `Sig_tau(M)` is the equivariant signature of the Z/2-action on `M`
(Atiyah-Singer equivariant signature theorem). If the normal map is
"equivariantly trivial" (i.e., comes from an equivariant identification),
this defect might vanish.

**Difficulty:** moderate. Requires equivariant index theory computation.

### Strategy D: Odd-dimensional E2 alternative

Revisit obligation E2 with an odd-dimensional construction:

Find a uniform lattice `Gamma' < Isom(H^m)` with `m` odd, `m >= 7`,
containing an order-2 element, such that the fixed set of the Z/2-action
has `chi(C) = 0` for all components `C` despite having even dimension
`m - 1`.

This requires fixed-set components that are even-dimensional manifolds with
zero Euler characteristic (e.g., products with `S^1`, or flat manifolds).
Such constructions are non-trivial but not impossible.

If achieved, the S branch (surgery obstruction) benefits from the
odd-dimensional parity vanishing.

**Difficulty:** high. Requires new geometric construction.

### Strategy E: Weaken to conditional statement

Accept the conditional and state the theorem as:

"For the specific lattice family constructed in Section 3b, the answer to
Problem 7 is yes if `sigma(f) = 0` in `Q + H_3(F; Q) + H_1(F; Q)`. This
obstruction is localized to the equivariant fixed-point contribution and
vanishes under the transfer to the torsion-free cover."

This is already a substantial narrowing of the problem: from an arbitrary
element of `L_6(Z[Gamma]) tensor Q` to a specific element of a much smaller
space determined by the topology of the fixed set `F`.

**Difficulty:** none (already achieved).

## Summary and node status

### What is established

1. The Farrell-Jones reduction to equivariant L-homology (Step 1): solid.
2. The `E_{Fin}` vs `E_{VCyc}` comparison via UNil rationalization (Step 1):
   solid (Connolly-Davis).
3. The model `H^6` for `E_{Fin}Gamma` and reduction to `H_6^{Z/2}(M; L tensor Q)`
   (Step 2): solid.
4. The dimension-parity tension between E2 and S (Step 6): structural
   observation, solid.

### What is conditional or conjectural

1. `sigma(f) in ker(res)` — requires compatible normal map construction
   (G3, depends on G2).
2. `ker(res) ~= Q + H_3(F; Q) + H_1(F; Q)` — conjectural, depends on
   unverified claims U1-U3 (free/fixed splitting, equivariant Thom
   isomorphism, twisted L-theory coefficients).
3. Whether `sigma(f) = 0` — not addressed.
4. Whether an odd-dimensional E2 construction exists — not addressed.

### Node status

**`p7r-s3b`: open.** The FJ reduction framework (Steps 1-2) is solid. The
transfer argument (Step 3) and localization (Step 4) are sketched but rest
on unproven intermediate lemmas (G3, U1-U3). The obstruction target space
`Q + H_3(F; Q) + H_1(F; Q)` is a plausible but unverified conjecture.

The proof remains conditional on `sigma(f) = 0` (plus G1, G2, G3 from
upstream). The FJ reduction and dimension-parity analysis represent genuine
progress; the localization formula needs verification before it can be
relied upon.

## References

- A. Bartels, F. T. Farrell, W. Luck, arXiv:1101.0469.
- F. Connolly, J. F. Davis, *L-theory of the infinite dihedral group*,
  Forum Math. 16 (2004), 687-699.
- F. Connolly, A. Ranicki, *On the calculation of UNil*, Adv. Math. 195
  (2005), 205-258.
- A. Ranicki, *Algebraic L-Theory and Topological Manifolds*, Cambridge
  Tracts 102, 1992.
- N. Bergeron, A. Venkatesh, *The asymptotic growth of torsion homology
  for arithmetic groups*, arXiv:1212.3847.
- M. Atiyah, I. Singer, *The index of elliptic operators III*, Ann. Math.
  87 (1968), 546-604.
- K. H. Dovermann, R. Schultz, *Equivariant Surgery Theories and Their
  Periodicity Properties*, Springer LNM 1443, 1990.
- C. T. C. Wall, *Surgery on Compact Manifolds*, 2nd ed., AMS, 1999.
