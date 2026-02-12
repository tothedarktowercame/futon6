# Problem 7 Reduced: `p7r-s3a` Approach I — Wall Surgery Setup

Date: 2026-02-12

## Scope

This document analyzes **Approach I** (Wall surgery via the FH(Q) complex) for
the manifold-upgrade problem (obligation S) described in `problem7-solution.md`,
Section 4. Two alternative approaches — equivariant surgery on `(M, tau)` and
orbifold resolution — are sketched in the solution document but not analyzed in
detail here.

**Status: OPEN.** This approach has three successive unresolved obstacles
(Poincare complex structure, degree-1 normal map, surgery obstruction). None
has been overcome.

## Input from upstream nodes

From `p7r-s2b` (discharged):

- `Gamma` is a cocompact lattice extension `1 -> pi -> Gamma -> Z/2 -> 1`,
  where `pi = Gamma_0(I)` is a torsion-free principal congruence subgroup of
  an arithmetic uniform lattice `Gamma_0 < Isom(H^n)` containing a
  reflection, with `n` **even**, `n >= 6`.
- `Gamma in FH(Q)`: there exists a finite CW complex `Y` with
  `pi_1(Y) = Gamma` and `Y~` rationally acyclic.
- `Y` has formal dimension `n` (the virtual cohomological dimension of
  `Gamma` equals `vcd(pi) = n`).

## The surgery problem

### Goal

Produce a closed `n`-manifold `M` with `pi_1(M) = Gamma` and
`H_*(M_tilde; Q) = 0` for `* > 0`.

### Strategy (Approach I)

Use Wall's surgery exact sequence to upgrade the FH(Q) complex `Y` to a closed
manifold. This requires:

1. Establishing that `Y` is a finite rational Poincare complex of dimension `n`.
2. Constructing a degree-1 normal map `f: M_0 -> Y` from a closed manifold.
3. Performing surgery on `f` below the middle dimension, preserving
   `pi_1 = Gamma`.
4. Showing the surgery obstruction `sigma(f) in L_n(Z[Gamma])` vanishes
   (rationally).

Each of steps 1, 2, and 4 presents an unresolved obstacle.

## Verification of surgery prerequisites

### P1. `Gamma` is finitely presented

**Satisfied.** `Gamma` is a cocompact lattice in a Lie group, hence finitely
presented (Selberg, Borel-Serre; see Raghunathan, *Discrete Subgroups of Lie
Groups*, Theorem 6.15).

### P2. Finite Poincare complex of dimension `n`

**GAP (G1): not yet established.**

FH(Q) gives a finite CW complex `Y` with `pi_1(Y) = Gamma` and `Y~`
rationally acyclic. This does **not** automatically give `Y` the structure
of a rational Poincare complex (which requires a fundamental class and
chain-level cap product duality).

**Partial justification.** The homology groups of `Y` do satisfy rational PD:

1. `Y~` is simply connected and rationally acyclic, so by rational Hurewicz,
   `pi_*(Y~) tensor Q = 0` for all `* > 0`. Thus `Y~` is rationally
   contractible.
2. The Serre spectral sequence for `Y~ -> Y -> BGamma` collapses (since
   `H_q(Y~; Q) = 0` for `q > 0`), giving `H_*(Y; Q) = H_*(Gamma; Q)`.
3. `Gamma` has rational PD in dimension `n` (Bredon framework: `Gamma` acts
   properly cocompactly on contractible `H^n`).
4. Therefore `H_*(Y; Q)` satisfies rational Poincare duality in dimension `n`.

**What remains.** This shows PD on the level of homology groups. Promoting
this to a rational Poincare complex structure on `Y` (fundamental class with
chain-level cap product duality) requires an additional argument. Possible
approaches:

**(a)** Use the equivariant diagonal on `Y~` to construct the chain-level
duality map, leveraging the rational contractibility of `Y~`.

**(b)** Show that any finite CW complex with rationally contractible
universal cover and rational-PD fundamental group automatically inherits
rational Poincare complex structure.

**(c)** Bypass `Y` and work within Avramidi's rational surgery framework
(arXiv:1506.06293), which may construct the rational Poincare complex
directly from the group-theoretic PD data without relying on the specific
CW complex from Fowler.

**Gap status: open.** The homology-level PD is established; chain-level
promotion needs a proof or reference.

### P3. Dimension `d >= 5`

**Satisfied.** We choose `n = 6` (or any even `n >= 6`). In all cases
`n >= 5`.

### P4. Existence of degree-1 normal map

**GAP (G2): not yet established.**

For any finite Poincare complex `Y` of dimension `n >= 5` (assuming G1),
the Spivak normal fibration theorem guarantees a stable spherical fibration
`nu_Y` over `Y`. A degree-1 normal map exists iff `nu_Y` lifts to a stable
vector bundle (topological reduction).

**Previous (incorrect) argument.** The prior version claimed that since the
double cover `Y_pi` is homotopy equivalent to the closed manifold `M`, the
stable normal bundle of `M` transfers to `Y`. This is insufficient:

1. `Y_pi` has the same rational homology as `M` (by the spectral sequence
   argument in P2), but `Y_pi` need not be homotopy equivalent to `M` —
   only rationally homotopy equivalent (since `Y~` is rationally but not
   necessarily integrally contractible).
2. Even given a topological reduction on `Y_pi`, descent to `Y` through a
   Z/2-quotient requires an explicit equivariant lifting of the stable
   bundle structure through the covering map `Y_pi -> Y`. This equivariant
   compatibility is not automatic.

**Possible approaches:**

**(a)** Construct the topological reduction directly by showing that the
rational normal invariant `[Y, G/Top] tensor Q` is nonempty (which holds
when rational Pontryagin classes can be defined on `Y`, using the PD
structure).

**(b)** Work within Avramidi's rational surgery framework, which may
construct the degree-1 normal map as part of the rational surgery setup
without requiring a separate Spivak/topological-reduction step.

**(c)** Use the equivariant bordism structure of `M` with its Z/2-action
to directly construct an equivariant normal map that descends to `Y`.

**Gap status: open.** The descent argument needs to be replaced by one of
the approaches above.

### P5. `pi_1` preservation through surgery

**Satisfied.** Surgery below the middle dimension preserves `pi_1` by
general position: if `n >= 5`, then surgeries in dimensions `<= n/2 - 1`
(i.e., `<= 2` for `n = 6`) do not affect the fundamental group. The
resulting manifold `M` after surgery has `pi_1(M) = pi_1(Y) = Gamma`.

Reference: Wall, *Surgery on Compact Manifolds*, Proposition 1.2 and the
discussion in Section 1.5 on "surgery below the middle dimension."

### P6. Rational acyclicity of universal cover is preserved

**Satisfied.** If `f: M -> Y` is a rational homotopy equivalence (the
output of successful rational surgery), then the induced map on universal
covers `f_tilde: M_tilde -> Y_tilde` is also a rational homotopy
equivalence. Since `Y_tilde` is rationally acyclic (by FH(Q)), so is
`M_tilde`.

## Interface to p7r-s3b (obstruction computation)

**This interface is reached only if obstacles 1 and 2 above are resolved.**
Currently, both are open.

If P2 and P4 are established, the remaining obligation is:

**Obstruction vanishing.** Show that the rational surgery obstruction

`sigma(f) in L_n(Z[Gamma]) tensor Q`

vanishes for the degree-1 normal map `f: M_0 -> Y` from P4, with `n = 6`
(or the chosen even dimension `n >= 6`).

The obstruction computation is analyzed in `problem7r-s3b-obstruction.md`.

## Dimension selection analysis

The obstruction computation depends on the parity of `n`:

| n | d = n | L_d rat. behavior | Prospects |
|---|-------|-------------------|-----------|
| 6 | even  | `L_6 tensor Q` has `(p,q)=(6,0),(2,4)` terms | Potentially nonzero |
| 8 | even  | `L_8 tensor Q` has `(p,q)=(8,0),(4,4),(0,8)` terms | Potentially nonzero |
| 7 | odd   | `L_7 tensor Q`: `p+q=7`, `q=0 mod 4` forces `p` odd | Likely zero |

For `n` odd, the rational obstruction is much more likely to vanish due to
the 4-periodicity of rational L-theory. However, our construction requires
`n` **even** (for the fixed-set Euler argument in E2).

### Possible resolution: dimension shift

If the dimension-6 obstruction proves intractable, consider:

1. **Product with S^1.** Replace `Gamma` by `Gamma x Z`, acting on
   `H^n x R`. The manifold dimension becomes `n + 1 = 7` (odd), and the
   obstruction lives in `L_7(Z[Gamma x Z]) tensor Q`, which has better
   vanishing prospects. If the `S^1`-direction surgery obstruction vanishes,
   we get a closed 7-manifold `M` with `pi_1(M) = Gamma x Z` — but this
   changes `pi_1`, so it does not directly solve Problem 7 for `Gamma`.

2. **Odd-dimensional lattice variant.** Seek a different construction where
   `Gamma` is a cocompact lattice with 2-torsion in `Isom(H^m)` for some
   `m` odd, using a route that does not require even dimension for E2.
   This would require revisiting `p7r-s2b` with a different strategy.

3. **Direct obstruction computation.** Use the specific structure of
   `Gamma = pi rtimes Z/2` and the Farrell-Jones reduction to compute
   `L_6(Z[Gamma]) tensor Q` explicitly. This is the approach pursued in
   `p7r-s3b`.

## Obstacle summary for Approach I

| Prerequisite | Status | Obstacle |
|---|---|---|
| P1 (finitely presented) | Verified | — |
| P2 (Poincare complex) | **Open** | Chain-level promotion from homology PD not proved |
| P3 (dimension >= 5) | Verified | — |
| P4 (degree-1 normal map) | **Open** | No construction; prior descent argument retracted |
| P5 (pi_1 preservation) | Verified (if P4) | — |
| P6 (rational acyclicity) | Verified (if P4) | — |
| Obstruction vanishing | **Open** | See `p7r-s3b-obstruction.md` |

**Approach I status: OPEN.** Three successive obstacles (P2, P4, obstruction
vanishing) are unresolved. Each blocks the next; none has been overcome.

See `problem7-solution.md`, Section 4 for the two alternative approaches
(equivariant surgery and orbifold resolution), which bypass the FH(Q) complex
entirely.

## References

- C. T. C. Wall, *Surgery on Compact Manifolds*, 2nd ed., AMS, 1999.
- G. Avramidi, *Rational Manifold Models for Duality Groups*,
  arXiv:1506.06293.
- A. Bartels, F. T. Farrell, W. Luck, arXiv:1101.0469.
- M. Spivak, *Spaces satisfying Poincare duality*, Topology 6 (1967).
- M. S. Raghunathan, *Discrete Subgroups of Lie Groups*, Springer, 1972.
- A. Ranicki, *Algebraic L-Theory and Topological Manifolds*, Cambridge, 1992.
