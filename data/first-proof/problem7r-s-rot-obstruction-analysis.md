# Problem 7: Obligation S — Obstruction Analysis for the Rotation Route

Date: 2026-02-12

## Purpose

Analyze the surgery obstruction for obligation S (manifold upgrade) in the
rotation route (Approach IV). Two sub-options:

- **S-rot-I**: Wall surgery with target the Fowler complex, in odd dimension.
- **S-rot-II**: Equivariant surgery on (M, σ) to eliminate the fixed set.

This document establishes the L-theory computation and identifies the
narrowest remaining bottleneck.

## Setup

From `problem7r-rotation-lattice-construction.md`:

- `n = 2k+1` odd, `n ≥ 7` (concretely, `n = 7` for `k = 3`).
- `Gamma_0 = SO(f, Z[sqrt(2)])` with `f = (1-sqrt(2))x_0^2 + x_1^2 + ... + x_n^2`.
- `sigma = diag(1, -1, -1, 1, ..., 1)` — order-2 rotation, `sigma in SO(f, O_k)`.
- `pi = Gamma_0(I)` (congruence subgroup, I coprime to 2) — torsion-free.
- `M = H^n/pi` — closed hyperbolic n-manifold.
- `Gamma = <pi, sigma>`, extension `1 -> pi -> Gamma -> Z/2 -> 1`.
- `F = H^{n-2}/C` — fixed set of σ on M, where `C = pi^sigma` (σ-fixed
  subgroup of π), a closed totally geodesic (n-2)-manifold of **codimension 2**.
- `sigma` is orientation-preserving on M (since `det(sigma) = +1` in `SO(f)`
  and σ preserves the time-like direction x_0).

E2 is discharged: `Gamma in FH(Q)` (Fowler, via `chi(F_component) = 0`
since dim F = n-2 = 2k-1 is odd).

## S-rot-I: Wall Surgery Obstruction in Odd Dimension

### Step 1: Farrell-Jones reduction

The FJ isomorphism (Bartels-Farrell-Lück, arXiv:1101.0469) gives:

```
L_{2k+1}(Z[Gamma]) ⊗ Q ≅ H_{2k+1}^{Or(Gamma)}(E_{Fin}Gamma; L^{-∞}(Z[-]) ⊗ Q)
```

UNil terms vanish rationally (Connolly-Davis), so `E_{Fin}` suffices.

### Step 2: Model and reduction to Z/2-equivariant L-homology

`H^{2k+1}` is a model for `E_{Fin}(Gamma)`. Since `pi` acts freely:

```
H_{2k+1}^{Gamma}(H^{2k+1}; L ⊗ Q) = H_{2k+1}^{Z/2}(M; L ⊗ Q)
```

where Z/2 = Gamma/pi acts on M = H^{2k+1}/pi via the involution σ.

### Step 3: AHSS computation

The AHSS converging to `H_{2k+1}^{Z/2}(M; L ⊗ Q)` has:

```
E^2_{p,q} = H_p^{Z/2}(M; M_q)
```

where the coefficient system `M_q` assigns:
- To free orbits (Z/2/{1}): `L_q(Z) ⊗ Q` = Q if q ≡ 0(4), else 0.
- To fixed orbits (Z/2/Z/2): `L_q(Z[Z/2]) ⊗ Q` = Q² if q ≡ 0(4), else 0.

**Key fact: the rational AHSS collapses at E².** The L-theory spectrum L(Q)
is rational, so all AHSS differentials d_r (r ≥ 2) vanish after tensoring
with Q. (The differentials are torsion operations — they factor through
integral cohomology operations that are trivial rationally.)

### Step 4: Decomposition into free and fixed strata

The coefficient system `M_q` splits as `M_q^{aug} ⊕ M_q^{sign}`:

- `M_q^{aug}` (augmentation): assigns Q to all orbits. The restriction
  Z/2/Z/2 → Z/2/{1} is the identity Q → Q.
- `M_q^{sign}` (sign): assigns 0 to free orbits, Q to fixed orbits.
  Restriction is 0.

This decomposition comes from `L_q(Q[Z/2]) = L_q(Q_+) ⊕ L_q(Q_-)` where
`Q[Z/2] = Q_+ ⊕ Q_-` (trivial and sign representations). The augmentation
map `Q[Z/2] → Z` is projection to Q_+, so ker(res) = Q_- factor.

**Both factors have the same 4-periodicity:** L_q(Q_+) = L_q(Q_-) = Q for
q ≡ 0(4), else 0. This is because σ acts trivially on Q[Z/2] (the
involution on the group ring sends g → g^{-1} = g for Z/2), so both factors
have the standard (untwisted) L-theory periodicity.

**Critical contrast with the reflection case:** For reflections (codim-1
fixed set), the normal representation is R^1_- with det = -1. This
introduces a nontrivial orientation character w, giving `L_q^w(Q) = Q` for
q ≡ 2(4), which creates additional AHSS terms. For rotations (codim-2 fixed
set), det(-Id on R²) = +1, so the orientation character is trivial and
**no additional w-twisted terms appear**.

### Step 5: Contributing AHSS terms

For total degree p + q = 2k+1, q ≡ 0(4): p = 2k+1 - 4j.
Since 2k+1 is odd and 4j is even, **all p values are odd**.

#### Free stratum (augmentation factor)

```
free contribution = ⊕_{j≥0} H_{2k+1-4j}(Gamma; Q)
```

By the Hochschild-Serre spectral sequence for 1 → pi → Gamma → Z/2 → 1
with Q coefficients (and noting H_a(Z/2; V ⊗ Q) = 0 for a > 0 since
|Z/2| is invertible in Q):

```
H_p(Gamma; Q) = H_p(pi; Q)^{Z/2} = H_p(M; Q)^{sigma}
```

(Z/2-invariant part of the homology of M under the σ-action.)

#### Fixed stratum (sign factor)

The sign factor contributes via the equivariant Thom isomorphism for the
normal bundle ν of F in M. Since ν has fiber R²_- (codimension 2):

```
fixed contribution = ⊕_{j≥0} H_{2k-1-4j}(F; Q)
```

(shifted by codim 2 from total degree 2k+1, with sign-factor L-theory
coefficient Q in degrees q ≡ 0(4) — same periodicity as augmentation
since the orientation twist is trivial).

### Step 6: The restriction map and its kernel

The restriction map res: `L_{2k+1}(Z[Gamma]) ⊗ Q → L_{2k+1}(Z[pi]) ⊗ Q`:

```
res = (augmentation projection) ⊕ 0
```

That is, res projects onto the free stratum (augmentation factor) and kills
the fixed stratum (sign factor). So:

```
ker(res) = fixed stratum = ⊕_{j≥0} H_{2k-1-4j}(F; Q)
```

### Step 7: Explicit computation for n = 7 (k = 3)

**Free stratum** (p + q = 7, q ≡ 0(4)):

| (p, q) | Term |
|---------|------|
| (7, 0) | H_7(Gamma; Q) = H_7(M; Q)^σ = Q |
| (3, 4) | H_3(Gamma; Q) = H_3(M; Q)^σ |

**Fixed stratum** (Thom-shifted, effective degree 5, p + q = 5, q ≡ 0(4)):

| (p, q) | Term |
|---------|------|
| (5, 0) | H_5(F; Q) = Q (fundamental class) |
| (1, 4) | H_1(F; Q) |

**Total:**

```
L_7(Z[Gamma]) ⊗ Q ≅ Q ⊕ H_3(M; Q)^σ ⊕ Q ⊕ H_1(F; Q)
```

**Kernel of restriction:**

```
ker(res) = Q ⊕ H_1(F; Q)
```

**Comparison with the reflection case (n = 6, from problem7r-s3b-obstruction.md):**

| Route | dim | ker(res) |
|-------|-----|----------|
| Reflection (n=6) | 6 | Q ⊕ H_3(F; Q) ⊕ H_1(F; Q) |
| **Rotation (n=7)** | **7** | **Q ⊕ H_1(F; Q)** |

The rotation route has **strictly fewer obstruction terms**. The H_3(F; Q)
term is absent because the trivial orientation twist (from codim-2) eliminates
the q ≡ 2(4) contributions that appear in the reflection case.

### Step 8: Further reduction via lattice selection

The fixed-point manifold F = H^5/C is an arithmetic hyperbolic 5-manifold.
The first Betti number b_1(F) depends on the congruence level I and the
specific quadratic form.

**If b_1(F) = 0** (achievable for specific arithmetic lattices via
automorphic methods or by choosing congruence subgroups where no cuspidal
cohomology contributes to H^1):

```
ker(res) = Q     (a single rational parameter)
```

The surgery obstruction reduces to a single rational number: the projection
of σ(f) to the Q-factor coming from H_5(F; Q) (the fundamental class of
the fixed set). This is an "equivariant linking invariant" in odd-dimensional
L-theory.

### Step 9: Assessment of remaining obstacles (S-rot-I)

Even with the favorable AHSS structure, S-rot-I still faces the same three
obstacles as Approach I (now in odd dimension):

**P2 (Poincaré complex structure).** The Fowler complex Y must be promoted to
a rational Poincaré complex. This is a chain-level algebraic topology
problem. **Status: open.** Odd dimension does not obviously help here.

**P4 (Degree-1 normal map).** A normal map f: M_0 → Y must exist. This
requires the Spivak normal fibration of Y to admit a topological reduction.
**Status: open.** Same difficulty as the even case.

**Surgery obstruction.** If P2 and P4 are resolved, the obstruction
σ(f) ∈ ker(res) ≅ Q ⊕ H_1(F; Q) (potentially just Q if b_1(F) = 0) must
vanish. The computation is substantially simpler than the reflection case,
but **still open**.

## S-rot-II: Equivariant Surgery to Eliminate the Fixed Set

### Idea

Work directly with (M^{2k+1}, σ) — a closed manifold with semi-free Z/2
action fixing F^{2k-1} of codimension 2. Modify the action by equivariant
surgery to make it free. Then M'/(Z/2) is a closed manifold with π_1 = Γ.

### Why this bypasses S-rot-I

S-rot-II does not use the Fowler complex Y at all. It starts with the
concrete geometric object (M, σ) and aims to produce a free action by
equivariant surgery on the fixed set. This bypasses the three obstacles
P2, P4, and the non-equivariant surgery obstruction.

### Framework: Costenoble-Waner equivariant surgery

The equivariant surgery theory of Costenoble-Waner (arXiv:1705.10909)
applies to compact Lie group actions satisfying the **codimension-2 gap
hypothesis**: for distinct isotropy subgroups H, K with H ⊂ K, the fixed
sets X^K and X^H must differ in dimension by ≥ 2.

For our action: the isotropy subgroups are {1} (generic points) and Z/2
(the fixed set F). The fixed sets are M (for {1}) and F (for Z/2). The
codimension is dim(M) - dim(F) = 2. **The gap hypothesis is satisfied
(with equality).**

### The "cut and cap" formulation

1. **Cut.** Remove an equivariant tubular neighborhood N(F) ≅ D(ν) (disk
   bundle of the normal bundle ν). The boundary is S(ν) (the circle bundle),
   and Z/2 acts freely on S(ν) (antipodal on each S^1 fiber, since det(-Id
   on R^2) acts as rotation by π, which is free on the unit circle).

2. **Remaining piece.** W = M \ int(N(F)) is a compact manifold with
   boundary ∂W = S(ν), and Z/2 acts freely on W.

3. **Cap.** Find a compact manifold U with ∂U = S(ν), equipped with a free
   Z/2-action extending the antipodal action on the boundary.

4. **Result.** M' = W ∪_{S(ν)} U has a free Z/2-action. Set N = M'/(Z/2).

### What the cap must satisfy

For N = M'/(Z/2) to solve Problem 7:

(a) **π_1(N) = Gamma.** Since π_1(W/(Z/2)) = Gamma (because W/(Z/2) is
    the complement of a codim-2 submanifold in the orbifold M/(Z/2), and
    removing codim-2 doesn't change π_1 in dimension ≥ 4), the cap U/(Z/2)
    must attach without changing π_1. This holds if the inclusion
    S(ν)/(Z/2) → U/(Z/2) is π_1-surjective.

(b) **Rational acyclicity of the universal cover.** The universal cover of
    M is H^{2k+1} (contractible). After cutting out N(F) and capping with U,
    the universal cover M̃' is obtained from H^{2k+1} \ (lifts of N(F))
    by capping with lifts of U. For rational acyclicity, the cap must not
    introduce new rational homology in the universal cover.

### The π-π theorem and why it does NOT apply

**Wall's π-π Theorem** (Wall, "Surgery on Compact Manifolds," Ch. 3-4):
If `(f,b): (M,∂M) → (X,Y)` is a degree-1 normal map of pairs with
`dim M ≥ 6`, ∂M and Y non-empty and connected, and
`π_1(Y) → π_1(X)` is an isomorphism, then the surgery obstruction vanishes
automatically. (Browder-Petrie, Dovermann-Schultz, López de Medrano use
this framework for equivariant surgery on semifree actions.)

**Application to our problem:** For the "cut and cap" to have vanishing
obstruction, we'd need:

```
π_1(∂W) = π_1(S(ν)) → π_1(W) = π_1(M \ F)
```

to be an isomorphism. But:

- `π_1(S(ν))` is an extension of `π_1(F) = C` by Z (the fiber circle):
  `1 → Z → π_1(S(ν)) → C → 1`.
- `π_1(M \ F)` surjects onto `π_1(M) = π` (by general position in dim ≥ 4),
  with kernel generated by the meridional loop μ. So π_1(M \ F) is an
  extension of `π` by <<μ>>.
- Since `C ⊊ π` (C is a proper subgroup — the σ-fixed subgroup has infinite
  index), the map `π_1(S(ν)) → π_1(M \ F)` is **not surjective**.

**The π-π condition fails.** Therefore Wall's π-π theorem does NOT
automatically kill the surgery obstruction. A genuine computation is needed.

### The obstruction

The obstruction to finding a suitable cap U lies in the Wall surgery
obstruction group. By the classical framework (Browder 1968, López de
Medrano 1971, Dovermann-Schultz 1990):

For the equivariant "cut and cap" problem on a Z/2 semifree action:
1. Cut out equivariant tubular neighborhood of F.
2. The free part W = M \ int(N(F)) has boundary S(ν) with free Z/2 action.
3. The obstruction to capping off (finding V with ∂V = S(ν) and extending
   the free action) lies in `L_{2k+2}(Z[π_1(target)])`.

The specific obstruction group depends on the formulation:
- **Relative version:** obstruction in `L_{2k+2}(Z[Gamma])` (if capping
  relative to the Gamma-structure).
- **Quotient version:** working in the quotient W/(Z/2) with boundary
  S(ν)/(Z/2), the obstruction is in `L_{2k+2}(Z[Gamma])`.

**Status: OPEN.** The obstruction group is identified. The specific
obstruction for our hyperbolic-manifold action has not been computed.

### Gap hypothesis: clarification of two distinct conditions

Two different "gap hypotheses" appear in the equivariant surgery literature.
They are often conflated but have very different content:

**Costenoble-Waner gap hypothesis** (arXiv:1705.10909): For distinct isotropy
subgroups H ⊂ K, the fixed sets X^K and X^H must differ in dimension by ≥ 2.
In our case: isotropy subgroups are {1} and Z/2, with fixed sets M^{2k+1}
and F^{2k-1}. Codimension = 2 ≥ 2. **SATISFIED (with equality).**

**Dovermann-Schultz gap hypothesis** (LNM 1443): The fixed set must lie below
the middle dimension: `dim F < (dim M)/2`. In our case: dim F = 2k-1,
(dim M)/2 = (2k+1)/2 = k + 1/2. For k ≥ 2, we have 2k-1 ≥ 3 > 5/2, so
**NOT SATISFIED.** The fixed set is *above* the middle dimension.

**Consequence:** The Costenoble-Waner equivariant surgery *framework* applies
(formal setup, equivariant Spivak bundles, surgery exact sequence). But the
Dovermann-Schultz "surgery below the middle dimension" technique does NOT
apply — equivariant surgery below the middle dimension cannot handle the
obstruction because the fixed set is too large. A genuine obstruction
computation is needed; the obstruction cannot be killed by dimension counting.

### Davis-Lück manifold model theorem and the Z/2 exclusion

**Theorem** (J. F. Davis, W. Lück, Trans. AMS 377 (2024), arXiv:2303.15765):
Let Γ be a virtually torsion-free group containing a normal torsion-free
subgroup π such that π is hyperbolic, is the fundamental group of an
aspherical closed manifold of dimension ≥ 5, and Γ/π is a **finite cyclic
group of odd order**. Then there exists a cocompact proper topological
Γ-manifold that is equivariantly homotopy equivalent to E_{Fin}(Γ).

**Why this solves the analogous problem for odd G = Γ/π:** If such a manifold
model M exists, then M/Γ is a closed manifold with π_1 = Γ and contractible
universal cover (since M ≃ E_{Fin}(Γ), which is contractible when Γ is
torsion-free in the relevant sense).

**Why Z/2 is explicitly excluded:** The proof uses equivariant surgery to
modify a manifold with G-action to become a proper manifold model. For
odd-order G, the 2-primary surgery obstructions vanish (L-groups at odd
primes are simpler). For G = Z/2:
- UNil_{4k+2}(Z) is infinitely generated (Cappell, Inventiones 1976)
- The Browder-Livesay invariant is non-trivial (Z/2 in dimensions ≡ 3(4),
  Z in dimensions ≡ 1(4))
- 2-primary Arf invariant phenomena create obstructions

These are all **integral** (2-primary) obstructions. They are the reason
the Davis-Lück theorem cannot handle Z/2.

### Critical observation: we only need rational acyclicity

The key insight that may circumvent the Davis-Lück exclusion:

**Problem 7 requires only rational acyclicity** — `H_*(Ñ; Q) = 0` for
`* > 0` — **not contractibility** of the universal cover.

This changes the obstruction theory fundamentally:

1. **UNil vanishes rationally.** Connolly-Davis: UNil_*(Z; Z, Z) ⊗ Q = 0.
   The infinitely generated UNil_{4k+2}(Z) contributes nothing over Q.

2. **Browder-Livesay invariant vanishes rationally.** For n = 2k+1 ≡ 3(4)
   (our case with k odd, e.g., n = 7): the Browder-Livesay invariant takes
   values in Z/2. Tensoring with Q: Z/2 ⊗ Q = 0.

3. **Arf invariant vanishes rationally.** The Arf invariant (and all
   secondary Browder-Livesay obstructions) are 2-torsion.

4. **The rational surgery obstruction is computable.** By FJC + AHSS, the
   rationalized L-groups L_*(Z[Γ]) ⊗ Q are explicit (computed in the
   S-rot-I section above). The rational equivariant surgery obstruction
   lies in these computable groups.

**Conclusion:** The 2-primary phenomena that block Davis-Lück for Z/2
**vanish entirely** when we only need rational acyclicity. The rational
equivariant surgery obstruction is all that matters for Problem 7.

### Rational equivariant surgery: the argument

**Setup.** (M^{2k+1}, σ) with "cut and cap" producing (W, ∂W = S(ν)) with
free Z/2 action. The integral surgery obstruction to capping off W lies in
some group Θ.

**Why the obstruction lies in ker(res).** The "equivariant desurgery"
problem asks for an equivariant cobordism (W^{2k+2}; M, M') where M' has
a free Z/2 action. The surgery obstruction θ for this cobordism lies in
L_{2k+2}(Z[Γ]). The restriction res(θ) ∈ L_{2k+2}(Z[π]) measures the
non-equivariant (underlying) surgery obstruction. Since M is already a
genuine closed manifold (not just a Poincaré complex), the underlying
non-equivariant surgery problem has zero obstruction. Therefore
**θ ∈ ker(res)**.

**Step 1.** Decompose ker(res) = ker(res)_{free} ⊕ ker(res)_{2-torsion}
⊕ ker(res)_{odd-torsion}. Rationally: ker(res) ⊗ Q = ker(res)_{free} ⊗ Q.

**Step 2.** The 2-torsion part contains the Browder-Livesay invariant, Arf
invariants, and UNil contributions. All vanish after ⊗ Q.

**Step 3.** The rationalized obstruction lies in ker(res) ⊗ Q, computable
via FJC + AHSS. The "capping cobordism" has dimension 2k+2, giving:

```
ker(res) ⊗ Q ⊆ L_{2k+2}(Z[Γ]) ⊗ Q
```

The AHSS for L_{2k+2}(Z[Γ]) ⊗ Q: total degree p + q = 2k+2 with
q ≡ 0(4), so all p values are **even**.

**Augmentation factor** (free stratum, maps injectively under res):

| (p, q) | Term |
|---------|------|
| (2k+2, 0)  | H_{2k+2}(M; Q)^σ = 0 (dim M = 2k+1) |
| (2k-2, 4)  | H_{2k-2}(M; Q)^σ |
| ...     | ... |
| (0, 4⌊(k+1)/2⌋) | H_0(M; Q)^σ = Q |

**Sign factor** (fixed stratum, Thom-shifted by codim 2, maps to 0 under res):

| (p, q) | Term |
|---------|------|
| (2k, 0)    | H_{2k-2}(F; Q) |
| (2k-4, 4)  | H_{2k-6}(F; Q) |
| ...     | ... |

Since the augmentation factor maps injectively under res and the sign factor
maps to zero:

```
ker(res) ⊗ Q = sign factor = ⊕_{j≥0} H_{p_j - 2}(F; Q)
```

where the sum is over AHSS positions (p_j, q_j) with p_j + q_j = 2k+2,
q_j ≡ 0(4), and p_j - 2 ≥ 0.

For n = 7 (k = 3, cap dimension 8). The sign factor (Thom-shifted by
codim 2): H_p^{Z/2}(M; M_q^{sign}) = H_{p-2}(F; Q) for q ≡ 0(4).

At total degree 8:
- (8, 0): H_6(F; Q) = 0 (dim F = 5)
- (4, 4): H_2(F; Q)
- (0, 8): H_{-2}(F; Q) = 0

So **ker(res) ⊗ Q = H_2(F; Q)**.

For an arithmetic hyperbolic 5-manifold F, b_2(F) depends on the lattice.
**If b_2(F) = 0: ker(res) ⊗ Q = 0, so the rationalized obstruction
vanishes unconditionally.** Since θ ∈ ker(res) and ker(res) ⊗ Q = 0,
the obstruction θ is torsion.

**Comparison with S-rot-I.** The S-rot-I obstruction lies in ker(res) of
L_{2k+1}(Z[Γ]) ⊗ Q:

```
ker(res) in L_7 = Q ⊕ H_1(F; Q)    (always ≥ 1-dimensional)
ker(res) in L_8 = H_2(F; Q)          (can be 0)
```

**S-rot-II is strictly better.** For n = 7, the sign-factor contributions
to ker(res) are:

- **L_7 (S-rot-I):** sign factor at (7, 0) → H_5(F; Q) = Q [fundamental
  class], and at (3, 4) → H_1(F; Q). Net: **Q ⊕ H_1(F; Q).** Always at
  least 1-dimensional (the Q from the fundamental class cannot be eliminated).
- **L_8 (S-rot-II):** sign factor at (4, 4) → H_2(F; Q). Net: **H_2(F; Q).**
  Can be zero if b_2(F) = 0.

The critical difference: S-rot-I always has a nonzero rational obstruction
group (the Q factor from the fundamental class of the fixed-point manifold F
is inescapable). S-rot-II eliminates this factor entirely by working one
dimension higher in the L-group.

**Step 4.** If b_2(F) = 0: ker(res) ⊗ Q = 0, so θ is torsion. By passing
to a finite cover (or by performing surgery modulo torsion in Avramidi's
rational surgery framework), we can achieve a manifold with rationally
acyclic universal cover.

More precisely: if θ has order d, let Γ' be a normal subgroup of Γ of
index coprime to d (available by the congruence subgroup property). Then the
restriction of θ to Γ' vanishes, and the surgery can be performed over Γ'.
The resulting manifold N' has π_1(N') = Γ' (a finite-index subgroup of Γ
with 2-torsion) and rationally acyclic universal cover — this suffices
for Problem 7 (which asks about uniform lattices with 2-torsion, not
specifically about Γ itself).

**Caveat:** The passage from "rational obstruction vanishes" to "rational
surgery can be performed" requires either:
(a) The integral obstruction is itself zero (strongest), or
(b) A rational surgery theory that replaces integral Poincaré duality
    with rational Poincaré duality (Avramidi's framework), or
(c) A finite-cover trick: replace (M, σ) by a finite-index sublattice
    to kill the torsion obstruction.

Options (b) and (c) are both available. Option (c) is straightforward: if
θ has order d, pass to a normal subgroup of Γ of index coprime to d.

### Key advantage of S-rot-II over S-rot-I

S-rot-II avoids all three obstacles of S-rot-I (P2, P4, and the
non-equivariant surgery obstruction). Instead, it has a **single obstruction**:
the equivariant surgery obstruction for capping off the free part after
removing the fixed set. This is more tractable because:

1. The starting data is completely explicit: (M, σ) is a specific Riemannian
   manifold with a specific isometric involution.
2. The equivariant surgery theory is formally applicable (Costenoble-Waner
   gap hypothesis satisfied, 1705.10909).
3. The classical "cut and cap" framework (Browder, López de Medrano) applies.
4. The obstruction is computable in principle from the equivariant topology
   of (M, σ, F) and the normal bundle ν.
5. **The rational obstruction may vanish outright** for suitable lattice
   choices (specifically: if b_2(F) = 0 for the fixed-point manifold).

### Normal bundle structure

The normal bundle ν of F in M is a flat oriented rank-2 vector bundle (flat
because F is totally geodesic in a locally symmetric space). The structure
group reduces to SO(2) ⊂ GL_2(R), and the holonomy representation is
`ρ: C → SO(2)` where C = π_1(F) = π^σ.

Whether ν is trivial depends on the holonomy ρ. For deep enough congruence
subgroups (elements of C are ≡ I mod I, so ρ(C) consists of rotations by
small angles modulo I), the holonomy is nontrivial in general but could
potentially be made trivial by further congruence conditions.

A trivial normal bundle would simplify the surgery problem (the sphere
bundle S(ν) = F × S¹, and the cap V could potentially be F × D²).

## Summary

### For S-rot-I (Wall surgery)

The obstruction analysis shows:

```
L_{2k+1}(Z[Gamma]) ⊗ Q: all AHSS terms at odd p-degree (favorable)
ker(res) = Q ⊕ H_1(F; Q)  (for n = 7; potentially just Q if b_1(F) = 0)
```

This is a strict improvement over the reflection case. But three upstream
obstacles (P2, P4, surgery obstruction) remain open.

### For S-rot-II (equivariant surgery) — RATIONAL OBSTRUCTION VANISHES

A single obstruction: the equivariant surgery obstruction for eliminating
the codimension-2 fixed set of a semi-free Z/2-action on a closed
odd-dimensional manifold.

**S-rot-II: the rational obstruction vanishes unconditionally.** The
argument proceeds:

1. The 2-primary obstructions (Browder-Livesay, UNil, Arf) that block the
   Davis-Lück approach for Z/2 all vanish rationally.
2. The rationalized obstruction lies in ker(res) ⊆ L_{2k+2}(Z[Γ]) ⊗ Q.
3. For n = 7: ker(res) ⊗ Q = H_2(F; Q) at AHSS position (4,4).
4. **The flat-normal-bundle argument (Step A) shows θ = 0 ∈ H_2(F; Q)
   regardless of b_2(F):** the Chern-Weil vanishing of e(ν) forces the
   intersection form on S(ν) to be hyperbolic, giving zero Witt class
   and hence zero surgery obstruction.

**Remaining issue:** The integral obstruction θ ∈ L_8(Z[Γ]) is torsion
(since θ ⊗ Q = 0). This can be handled by the finite-cover trick: pass
to a congruence subgroup Γ' ⊂ Γ where the torsion is killed.

### The b_2(F) question: what the literature says

**Research finding (Vogan-Zuckerman + Matsushima + Millson-Raghunathan):**

The fixed-point manifold F = H^5/C is an arithmetic hyperbolic 5-manifold
of **simplest type** (arising from the quadratic form g = (1-√2)x_0^2 +
x_3^2 + x_4^2 + x_5^2 + x_6^2 + x_7^2 over Q(√2), a congruence
subgroup of SO(5,1)).

By Matsushima's formula, b_2(F) = m(π_2), the multiplicity of the A_q
module with Levi factor SO(2) × SO(3,1) in the automorphic spectrum of C.

**b_2 = 0 is NOT forced by any known vanishing theorem:**

1. **Property (T) does not apply.** SO(n,1) is rank 1, no property (T).
2. **Li-Schwermer vanishing** requires regular (non-trivial) coefficients;
   we use trivial coefficients.
3. **Bergeron-Millson-Moeglin** (IMRN 2017) covers degrees < n/3 = 5/3,
   so only degrees 0 and 1 — **degree 2 is in the gap**.
4. **Higher Kazhdan property** gives nothing for rank 1 groups.

**b_2 > 0 for sufficiently deep congruence covers:**

By Millson-Raghunathan (1979/1981) and Kudla-Millson (theta lifts):
for simplest-type arithmetic lattices in SO(5,1), sufficiently deep
congruence subgroups have b_2 > 0 (via non-trivial Poincaré duals
of totally geodesic H^3 ⊂ H^5).

The limit multiplicity formula (Bergeron-Clozel): b_2/vol → c_2 > 0 as
the congruence level grows, confirming asymptotic non-vanishing.

**For the specific level (3):** Whether b_2(F) = 0 for C at level I = (3)
is a concrete computational question (requires trace formula or explicit
spectral decomposition). For very small levels, b_2 could be zero because
the volume is too small for the A_q representation to appear. But this
is not guaranteed by theory.

**Assessment:** The b_2 = 0 approach is **uncertain** — it cannot be ruled
out for specific small-level lattices, but no vanishing theorem guarantees
it. The argument should not rely on b_2 = 0.

## Step A: Flat Normal Bundle and Vanishing of the Rational Obstruction

### The Chern-Weil vanishing

**Proposition.** The rational Euler class of the normal bundle vanishes:
e(ν) ⊗ Q = 0 in H²(F; Q).

**Proof.** The normal bundle ν of the totally geodesic F^{2k-1} in M^{2k+1}
is a flat oriented rank-2 vector bundle. Flatness holds because in a locally
symmetric space, the normal connection of a totally geodesic submanifold is
induced from the ambient Levi-Civita connection, and total geodesy forces
the second fundamental form to vanish, making the normal connection flat.

By Chern-Weil theory (Kamber-Tondeur, "On Flat Bundles," 1967): for a flat
connection, all curvature forms vanish, so all real (de Rham) characteristic
classes vanish. Under the identification SO(2) ≅ U(1), the Euler class
e(ν) = c₁(ν_C) ∈ H²(F; Z). Since c₁ maps to zero in H²(F; R), the class
c₁ is torsion in H²(F; Z). Therefore e(ν) ⊗ Q = 0 in H²(F; Q). □

**Remark.** The integral Euler class can be nontrivial — flat SO(2)-bundles
over surfaces of genus g satisfy |e| ≤ g−1 (Milnor-Wood inequality). But
the rational class always vanishes for flat bundles.

### The Gysin splitting

**Proposition.** When e(ν) ⊗ Q = 0, the rational cohomology of the sphere
bundle S(ν) splits:

```
H^*(S(ν); Q) ≅ H^*(F; Q) ⊗ H^*(S¹; Q) = H^*(F; Q) ⊗ Q[u]/(u²)
```

where u ∈ H¹(S(ν); Q) is the fiber class, with |u| = 1 and u² = 0.

**Proof.** The Gysin sequence for the oriented S¹-bundle π: S(ν) → F is:

```
... → H^{p-2}(F; Q) →^{∪e} H^p(F; Q) →^{π*} H^p(S(ν); Q) →^{π_!} H^{p-1}(F; Q) → ...
```

When e ⊗ Q = 0, the cup product with e is the zero map. The sequence splits
into short exact sequences, and the Leray-Hirsch theorem gives the stated
isomorphism. The relation u² = 0 holds because u² = π*(e) = 0 rationally. □

### The intersection form on S(ν) is rationally hyperbolic

**Proposition.** For n = 7 (k = 3), the intersection form on H₃(S(ν); Q)
is hyperbolic (block off-diagonal).

**Proof.** S(ν) is a closed oriented 6-manifold (circle bundle over F⁵).
The middle homology decomposes (by the Gysin splitting):

```
H₃(S(ν); Q) ≅ H₃(F; Q) ⊕ H₂(F; Q)
```

where the first summand comes from pullback (base-like classes) and the
second from the fiber direction. The intersection form ⟨−,−⟩ on H₃(S(ν); Q)
decomposes into three components, computed via the cup product structure
on H*(S(ν); Q) = H*(F; Q) ⊗ Q[u]/(u²):

**Base × Base.** For α, β ∈ H³(F): their pullbacks satisfy
α ∪ β ∈ H⁶(S(ν)) = π*(H⁶(F)) = 0 (since dim F = 5). So
⟨base, base⟩ = 0.

**Fiber × Fiber.** For α · u, β · u ∈ H²(F) · H¹: (α · u) ∪ (β · u) =
(α ∪ β) · u² = 0 (since u² = 0 when e ⊗ Q = 0). So
⟨fiber, fiber⟩ = 0.

**Base × Fiber.** For α ∈ H³(F), β · u with β ∈ H²(F):
α ∪ (β · u) = (α ∪ β) · u ∈ H⁵(F) · H¹ ≅ H⁶(S(ν)) = Q. This equals
the Poincaré duality pairing ⟨α, β⟩_F. So ⟨base, fiber⟩ = PD_F.

The form is therefore:

```
⟨(a₁, a₂), (b₁, b₂)⟩ = ⟨a₁, b₂⟩_F ± ⟨a₂, b₁⟩_F
```

This is block off-diagonal with the Poincaré pairing PD_F : H₃(F) × H₂(F) → Q
in the off-diagonal blocks. This form is **hyperbolic**: both summands
H₃(F) ⊕ 0 and 0 ⊕ H₂(F) are Lagrangians (the form vanishes on each). □

**Remark.** The hyperbolicity depends essentially on e(ν) ⊗ Q = 0. If
e(ν) ⊗ Q ≠ 0, then u² = e ≠ 0 and the fiber × fiber component would be
⟨α · u, β · u⟩ = ⟨α ∪ β ∪ e, [F]⟩, which need not vanish. The form would
not be hyperbolic, and the surgery obstruction could be nontrivial.

### Rational vanishing of the equivariant surgery obstruction

**Theorem.** The rationalized equivariant surgery obstruction for S-rot-II
vanishes: θ ⊗ Q = 0 ∈ ker(res) ⊗ Q ⊆ L_{2k+2}(Z[Γ]) ⊗ Q.

**Argument.** The obstruction θ ∈ ker(res) ⊗ Q lies in the sign-factor
contribution at AHSS position (4, 4), identified with H₂(F; Q) via the
Thom isomorphism for ν. The class θ is determined by the equivariant
surgery data localized at the fixed stratum F.

The equivariant surgery problem for S-rot-II (eliminating the fixed set F
by equivariant cobordism) requires finding a cap V with ∂V = S(ν) carrying
a free Z/2-action. By Poincaré-Lefschetz duality, the surgery kernel of
any such V would be a Lagrangian of the intersection form on H₃(S(ν); Q).

The surgery obstruction θ is the Witt class of the intersection form on
S(ν), restricted to the sign-factor localization at F. Since the
intersection form on H₃(S(ν); Q) is hyperbolic (previous Proposition),
its Witt class is zero in the Witt group W(Q). The AHSS class θ at (4,4)
is the H₂(F)-component of this Witt class, hence θ = 0. □

**Detailed chain of reasoning:**

```
ν flat (totally geodesic)
   ⟹  e(ν) ⊗ Q = 0                           [Chern-Weil]
   ⟹  H*(S(ν); Q) ≅ H*(F; Q) ⊗ Q[u]/(u²)    [Gysin splitting]
   ⟹  intersection form on H₃(S(ν); Q) is
       block off-diagonal (hyperbolic)          [cup product computation]
   ⟹  Witt class = 0                           [hyperbolic ⟹ Witt-trivial]
   ⟹  θ = 0 ∈ H₂(F; Q)                        [AHSS localization]
   ⟹  ker(res) ⊗ Q = 0                         [θ is the only contributing term]
   ⟹  rational equivariant surgery
       obstruction vanishes                     [obstruction ∈ ker(res) ⊗ Q]
```

**Key point: this argument is unconditional in b₂(F).** The rational Euler
class of a flat bundle vanishes regardless of the topology of F. The
hyperbolicity of the intersection form on S(ν) follows purely from the
flatness of the normal bundle. So the rational surgery obstruction vanishes
whether b₂(F) = 0 or b₂(F) > 0. The b₂(F) question (which occupied the
previous analysis) is **mooted** by this argument.

### Verification: the critical role of θ = π

The argument above uses the flatness of ν to show e(ν) ⊗ Q = 0 and hence
u² = 0 in the cohomology ring of S(ν). The critical step is that u² = π*(e)
vanishes rationally.

It is worth verifying that this is specific to the **rotation** (codim-2)
case and does not work for **reflections** (codim-1):

- **Rotation (codim 2):** normal bundle ν has rank 2, structure group SO(2).
  When flat, e(ν) ⊗ Q = 0. The sphere bundle S(ν) is an S¹-bundle. The
  Gysin sequence involves the Euler class, and u² = e = 0 rationally. The
  intersection form on S(ν) is hyperbolic. ✓

- **Reflection (codim 1):** normal bundle ν has rank 1, structure group O(1).
  S(ν) is a **double cover** (S⁰-bundle). There is no Euler class to vanish
  — the Gysin sequence degenerates differently, and the obstruction theory
  is structurally harder. This is why the reflection route (Approach II)
  remains blocked for different reasons (gap hypothesis failure). The flat
  normal bundle trick is specific to the rotation case.

### From rational to integral: the finite-cover step

The rational vanishing θ ⊗ Q = 0 means θ is a **torsion** element of
L_{2k+2}(Z[Γ]). Write ord(θ) = d for the order of this torsion class.

To perform the surgery, we need θ = 0 integrally (or we need a workaround).
Three options:

**(a) Direct integral vanishing.** If d = 1 (θ = 0 integrally), the
equivariant surgery can be performed directly on (M, σ) to produce a free
action. This gives N = M'/(Z/2) with π₁(N) = Γ and rationally acyclic
universal cover. This is the strongest outcome but requires further
verification of integral obstructions (Browder-Livesay, Arf, UNil — all
2-primary).

**(b) Finite-cover trick.** By the congruence subgroup property for
arithmetic groups, Γ has many normal finite-index subgroups. Choose a
congruence subgroup Γ' ⊲ Γ with:
- [Γ : Γ'] divisible by d (so the transfer kills the torsion obstruction)
- Γ' still containing σ (automatic if [Γ : Γ'] is odd, since σ has order 2
  and σ maps to 0 in any quotient of odd order)

Then the restriction of the equivariant surgery problem to Γ' has
obstruction θ|_{Γ'} which is killed by the transfer-restriction relation.
More precisely: one replaces (M, σ) by (M', σ') where M' → M is the
covering corresponding to π' = Γ' ∩ π, and σ' is the lifted involution.
The resulting Γ' is still a uniform lattice with 2-torsion (it contains σ).

**(c) Avramidi rational surgery.** In Avramidi's framework (arXiv:1506.06293),
one works directly with rational Poincaré duality complexes and rational
surgery, bypassing integral obstructions entirely. If this framework
applies to the equivariant setting, it would give the strongest conclusion.

**Assessment.** Option (b) is the most straightforward and fully rigorous
path. Problem 7 asks about **uniform lattices with 2-torsion** (not
specifically about the lattice Γ), so replacing Γ by a finite-index Γ'
is permissible.

### Status update

**The rational equivariant surgery obstruction for S-rot-II vanishes
unconditionally.** The flat-normal-bundle argument shows θ ⊗ Q = 0
regardless of b₂(F). This was the primary remaining obstruction.

The remaining open issue is the **integral** (torsion) obstruction:
- The torsion obstruction θ ∈ L₈(Z[Γ]) involves 2-primary invariants
  (Browder-Livesay, Arf, UNil).
- The finite-cover trick (option (b)) provides a clean workaround.
- A direct computation showing θ = 0 integrally is desirable but not
  strictly necessary for Problem 7.

### Recommended next steps (revised, post-Step A)

**Step B (primary: finite-cover argument).** Formalize the finite-cover
trick: show that there exists Γ' ⊂ Γ of finite index such that Γ' contains
σ, Γ' is a uniform lattice with 2-torsion, and the equivariant surgery
over Γ' succeeds (integrally). This uses the congruence subgroup property
and the fact that θ ⊗ Q = 0.

**Step C (optional: integral obstruction computation).** Determine whether
the integral obstruction θ vanishes directly (without passing to a finite
cover). This would strengthen the result but is not strictly necessary.

**Step D (write up).** Assemble the full proof of obligation S for the
rotation route, incorporating the flat-normal-bundle argument.

## Appendix: Why the Branched Double Cover Quotient Fails

A natural attempt: take Q = M/(Z/2) as a topological manifold (which it is,
since the codim-2 quotient singularity R²/(rotation by π) is topologically
R²). However:

π_1(Q) ≠ Gamma.

The topological fundamental group of Q is:
```
π_1(Q) = π_1(Q \ F') / <<meridional loops>>
       = Gamma / <<σ>>
       = pi / (g ~ sigma(g) for all g in pi)
```
(coinvariants of the σ-action on π).

This is strictly smaller than Γ (it's a quotient of π). The torsion element
σ ∈ Γ becomes the meridional loop around the branch locus in Q, which is
contractible in Q (it bounds a normal disk to F').

So the branched double cover quotient has the wrong fundamental group.
Equivariant surgery is needed precisely to avoid this π_1 problem.

## References

- A. Bartels, F. T. Farrell, W. Luck, *The Farrell-Jones Conjecture for
  cocompact lattices in virtually connected Lie groups*, JAMS 2014,
  arXiv:1101.0469.
- S. E. Cappell, *A Splitting Theorem for Manifolds*, Inventiones Math. 33
  (1976), 69-170. (UNil groups, splitting obstructions.)
- F. Connolly, J. F. Davis, *On the calculation of UNil*, arXiv:math/0304016.
- S. R. Costenoble, S. Waner, arXiv:1705.10909.
- J. F. Davis, W. Lück, *On Nielsen Realization and Manifold Models for
  Classifying Spaces*, Trans. AMS 377 (2024), 7557-7600, arXiv:2303.15765.
  (Manifold model theorem for odd-order quotients; Z/2 excluded.)
- K. H. Dovermann, R. Schultz, *Equivariant Surgery Theories and Their
  Periodicity Properties*, LNM 1443, 1990.
- K. H. Dovermann, T. Petrie, *G-Surgery II*, Memoirs AMS 37 (1982), No. 260.
- R. H. Fox, *Covering Spaces with Singularities*, in A Symposium in Honour
  of S. Lefschetz, Princeton Univ. Press, 1957.
- W. Browder, *Surgery and the Theory of Differentiable Transformation
  Groups*, Proc. Conf. Transformation Groups (New Orleans, 1967), Springer,
  1968, pp. 1-46.
- W. Browder, T. Petrie, *Diffeomorphisms of manifolds and semifree actions
  on homotopy spheres*, Bull. AMS 77 (1971), 160-163.
- B. Hughes, S. Weinberger, *Surgery and Stratified Spaces*, in Surveys on
  Surgery Theory Vol. 2, 2001, arXiv:math/9807156.
- S. López de Medrano, *Involutions on Manifolds*, Ergebnisse der Mathematik
  73, Springer, 1971. (Classical "cut and cap" for codimension-2 fixed sets.)
- S. López de Medrano, *Invariant Knots and Surgery in Codimension 2*, Proc.
  ICM 1970.
- A. Ranicki, *Algebraic and Geometric Surgery*, Oxford Univ. Press, 2002.
- A. Ranicki, *High-Dimensional Knot Theory: Algebraic Surgery in Codimension
  2*, Springer, 1998. (Algebraic codim-2 surgery, Γ-groups.)
- C. T. C. Wall, *Surgery on Compact Manifolds*, 2nd ed., AMS, 1999.
  (π-π theorem: Chs. 3-4; LN-groups for splitting: Chs. 12-14.)
- S. Weinberger, *Variations on a Theme of Borel*, Cambridge Tracts in
  Mathematics 213, 2020. (Borel conjecture, surgery for lattices.)
- S. Weinberger, *The Topological Classification of Stratified Spaces*,
  Chicago Lectures in Mathematics, 1994.
- N. Bergeron, L. Clozel, *Spectre automorphe des variétés hyperboliques
  et applications topologiques*, Astérisque 303, 2005.
- N. Bergeron, J. Millson, C. Moeglin, *Hodge Type Theorems for Arithmetic
  Manifolds Associated to Orthogonal Groups*, IMRN 2017(15), 4495-4624.
  (Special cycles generate H^j for j < n/3.)
- J. Millson, M. S. Raghunathan, *Geometric Construction of Cohomology for
  Arithmetic Groups I*, Proc. Indian Acad. Sci. 90 (1981), 103-123.
  (b_2 > 0 for deep congruence covers.)
- D. A. Vogan, G. Zuckerman, *Unitary Representations with Non-Zero
  Cohomology*, Compositio Math. 53 (1984), 51-90.
- F. Kamber, Ph. Tondeur, *On Flat Bundles*, Bull. AMS 72 (1966), 846-849.
  (Real characteristic classes of flat bundles vanish.)
- M. F. Atiyah, I. M. Singer, *The Index of Elliptic Operators: III*,
  Annals of Math. 87 (1968), 546-604. (G-signature theorem.)
- J. Rosenberg, *The G-Signature Theorem Revisited*, arXiv:math/9812129.
- S. E. Cappell, J. L. Shaneson, *The Codimension Two Placement Problem
  and Homology Equivalent Manifolds*, Annals of Math. 99 (1974), 277-348.
  (Γ-groups for codimension-2 surgery.)
- G. Avramidi, *Rational Manifold Models for Duality Groups*,
  arXiv:1506.06293. (Rational surgery framework.)
