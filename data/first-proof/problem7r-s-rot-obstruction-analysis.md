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

### Key advantage of S-rot-II over S-rot-I

S-rot-II avoids all three obstacles of S-rot-I (P2, P4, and the
non-equivariant surgery obstruction). Instead, it has a **single obstruction**:
the equivariant surgery obstruction for capping off the free part after
removing the fixed set. This is more tractable because:

1. The starting data is completely explicit: (M, σ) is a specific Riemannian
   manifold with a specific isometric involution.
2. The equivariant surgery theory is formally applicable (gap hypothesis
   satisfied, Costenoble-Waner 1705.10909).
3. The classical "cut and cap" framework (Browder, López de Medrano) applies.
4. The obstruction is computable in principle from the equivariant topology
   of (M, σ, F) and the normal bundle ν.

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

## Summary: Narrowest Bottleneck

### For S-rot-I (Wall surgery)

The obstruction analysis shows:

```
L_{2k+1}(Z[Gamma]) ⊗ Q: all AHSS terms at odd p-degree (favorable)
ker(res) = Q ⊕ H_1(F; Q)  (for n = 7; potentially just Q if b_1(F) = 0)
```

This is a strict improvement over the reflection case. But three upstream
obstacles (P2, P4, surgery obstruction) remain open.

### For S-rot-II (equivariant surgery)

A single obstruction: the equivariant surgery obstruction for eliminating
the codimension-2 fixed set of a semi-free Z/2-action on a closed
odd-dimensional manifold.

**S-rot-II is the narrower bottleneck.** It replaces three open obstacles
with one.

### Recommended next step

Compute the equivariant surgery obstruction for the specific action
(M^{2k+1}, σ) with:
- M a closed hyperbolic manifold
- σ an isometric involution fixing F^{2k-1} (totally geodesic, codim 2)
- Normal bundle: ν = R²_- (sign representation)

Literature to consult:
- Wall, Surgery on Compact Manifolds, Chapter 14 (surgery on involutions)
- Dovermann-Schultz, LNM 1443 (equivariant surgery periodicity)
- Costenoble-Waner, arXiv:1705.10909 (equivariant Spivak bundle, surgery)
- López de Medrano, Involutions on Manifolds (classical surgery on involutions)
- Weinberger, The Topological Classification of Stratified Spaces (surgery
  on stratified spaces, relevant for orbifold quotients)

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

- A. Bartels, F. T. Farrell, W. Luck, arXiv:1101.0469.
- W. Browder, *Surgery and the Theory of Differentiable Transformation
  Groups*, Proc. Conf. Transformation Groups (New Orleans, 1967), Springer,
  1968, pp. 1-46.
- W. Browder, T. Petrie, *Diffeomorphisms of manifolds and semifree actions
  on homotopy spheres*, Bull. AMS 77 (1971), 160-163.
- F. Connolly, J. F. Davis, *L-theory of the infinite dihedral group*,
  Forum Math. 16 (2004), 687-699.
- S. R. Costenoble, S. Waner, arXiv:1705.10909.
- K. H. Dovermann, R. Schultz, *Equivariant Surgery Theories and Their
  Periodicity Properties*, LNM 1443, 1990.
- R. H. Fox, *Covering Spaces with Singularities*, in A Symposium in Honour
  of S. Lefschetz, Princeton Univ. Press, 1957.
- S. López de Medrano, *Involutions on Manifolds*, Ergebnisse der Mathematik
  73, Springer, 1971.
- A. Ranicki, *Algebraic and Geometric Surgery*, Oxford Univ. Press, 2002.
- C. T. C. Wall, *Surgery on Compact Manifolds*, 2nd ed., AMS, 1999.
  (π-π theorem: Chapters 3-4.)
- S. Weinberger, *The Topological Classification of Stratified Spaces*,
  Chicago Lectures in Mathematics, 1994.
