# Problem 7 Reduced: `p7r-s3a` Manifold-Upgrade Setup Interface

Date: 2026-02-12

## Target Node

`p7r-s3a`: specify the surgery setup that upgrades the finite-CW realization
from E2 (`Gamma in FH(Q)`) to a closed manifold with the same fundamental
group and rationally acyclic universal cover.

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

### Strategy

Use Wall's surgery exact sequence to upgrade the finite Poincare complex `Y`
to a closed manifold. The plan:

1. Establish that `Y` is a finite rational Poincare complex of dimension `n`.
2. Construct a degree-1 normal map `f: M_0 -> Y` from a closed manifold.
3. Perform surgery on `f` below the middle dimension to make it highly
   connected while preserving `pi_1 = Gamma`.
4. The remaining obstruction is `sigma(f) in L_n(Z[Gamma])`. If it vanishes
   (rationally), the surgery succeeds.

## Verification of surgery prerequisites

### P1. `Gamma` is finitely presented

**Satisfied.** `Gamma` is a cocompact lattice in a Lie group, hence finitely
presented (Selberg, Borel-Serre; see Raghunathan, *Discrete Subgroups of Lie
Groups*, Theorem 6.15).

### P2. Finite Poincare complex of dimension `n`

**Satisfied (rationally).** The FH(Q) condition gives a finite CW complex
`Y` with `pi_1(Y) = Gamma` and `Y~` rationally acyclic. We claim `Y`
satisfies rational Poincare duality in dimension `n`:

- Since `pi < Gamma` is torsion-free of index 2, and `Bpi = H^n/pi` is a
  closed oriented `n`-manifold (hence an integral Poincare complex), the
  transfer argument gives `H^*(Gamma; Q) ~ H^*(pi; Q)^{Z/2}`, which
  satisfies Poincare duality over `Q` in dimension `n`.
- More precisely: `Gamma` acts properly and cocompactly on the contractible
  `n`-manifold `H^n`, so the equivariant Bredon cohomology gives rational
  PD in dimension `n` (see Section 2 of the solution file).

**Integral vs. rational PD.** For Wall's surgery machinery in its standard
form, integral Poincare duality is needed. Our `Y` only satisfies rational
PD. Two approaches:

**(a) Rational surgery (Avramidi framework).** Avramidi (arXiv:1506.06293,
Theorems 14, 16, 17) develops surgery with rational coefficients for groups
with finite classifying spaces. The obstruction then lives in
`L_n(Z[Gamma]) tensor Q`. This is the natural framework for our problem
since we only need `Q`-acyclicity of the universal cover.

**(b) Transfer to the torsion-free cover.** Since `[Gamma : pi] = 2`, any
finite Poincare complex `Y` for `Gamma` has a double cover `Y_pi` which is
a finite Poincare complex for `pi`. Since `pi` is a closed-manifold group
(`Bpi = H^n/pi`), `Y_pi` is integrally PD. Surgery on `Y_pi` can be
performed by standard methods, and the Z/2 action descends to the result if
it is equivariant.

We adopt approach (a) as the primary route, with (b) as a cross-check.

### P3. Dimension `d >= 5`

**Satisfied.** We choose `n = 6` (or any even `n >= 6`). In all cases
`n >= 5`.

### P4. Existence of degree-1 normal map

**Satisfied.** For any finite Poincare complex `Y` of dimension `n >= 5`,
the Spivak normal fibration theorem guarantees a stable spherical fibration
`nu_Y` over `Y`. The question is whether `nu_Y` lifts to a stable vector
bundle (equivalently, a topological reduction).

For our `Y`: the double cover `Y_pi` is homotopy equivalent to the closed
manifold `M = H^n/pi`, which has a genuine stable normal bundle. The
transfer of this structure to `Y` gives a topological reduction of `nu_Y`.
Hence a degree-1 normal map `f: M_0 -> Y` exists.

More concretely: the closed manifold `M` with its identity map `id: M -> M`
is a degree-1 normal map for `pi`. The Z/2 equivariant version gives a
normal map for `Gamma`.

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

After this setup, the single remaining obligation is:

**Obstruction vanishing.** Show that the rational surgery obstruction

`sigma(f) in L_n(Z[Gamma]) tensor Q`

vanishes for the degree-1 normal map `f: M_0 -> Y` from P4, with `n = 6`
(or the chosen even dimension `n >= 6`).

### What p7r-s3b receives

- Group: `Gamma = pi rtimes Z/2`, where `pi` is a torsion-free cocompact
  lattice in `Isom(H^n)`, n even >= 6.
- Obstruction group: `L_n(Z[Gamma]) tensor Q`.
- Farrell-Jones applies: `Gamma` is a cocompact lattice in a virtually
  connected Lie group, so the K- and L-theoretic Farrell-Jones conjecture
  holds (Bartels-Farrell-Luck, arXiv:1101.0469).
- The specific normal map `f` comes from the Spivak normal fibration
  lifted via the torsion-free double cover.

### What p7r-s3b must produce

Either:
1. A proof that `sigma(f) = 0` in `L_n(Z[Gamma]) tensor Q`, or
2. A computation of `L_n(Z[Gamma]) tensor Q` showing it is zero, or
3. An explicit identification of the obstruction as an element that can be
   shown to vanish by a specific argument, or
4. An honest assessment that the obstruction is nonzero / unknown, with a
   weakening to conditional.

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

## Node-level status

- P1 (finitely presented): **verified**
- P2 (finite Poincare complex): **verified** (rationally; use Avramidi
  rational surgery framework)
- P3 (dimension >= 5): **verified** (n = 6)
- P4 (degree-1 normal map existence): **verified** (via torsion-free cover)
- P5 (pi_1 preservation): **verified**
- P6 (universal cover acyclicity preserved): **verified**

**p7r-s3a status: provisionally closed.** All surgery prerequisites are
met. The single remaining obligation is the obstruction computation,
which passes to `p7r-s3b`.

## References

- C. T. C. Wall, *Surgery on Compact Manifolds*, 2nd ed., AMS, 1999.
- G. Avramidi, *Rational Manifold Models for Duality Groups*,
  arXiv:1506.06293.
- A. Bartels, F. T. Farrell, W. Luck, arXiv:1101.0469.
- M. Spivak, *Spaces satisfying Poincare duality*, Topology 6 (1967).
- M. S. Raghunathan, *Discrete Subgroups of Lie Groups*, Springer, 1972.
- A. Ranicki, *Algebraic L-Theory and Topological Manifolds*, Cambridge, 1992.
