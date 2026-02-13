# Problem 7: Uniform Lattice with 2-Torsion and Rationally Acyclic Universal Cover

## Problem Statement

Suppose `Gamma` is a uniform lattice in a real semisimple Lie group, and
`Gamma` contains an element of order `2`. Is it possible that `Gamma` is the
fundamental group of a closed manifold whose universal cover is acyclic over
`Q`?

Here "acyclic over Q" means H_i(M_tilde; Q) = 0 for all i > 0 (and
H_0(M_tilde; Q) = Q).

## Status in This Writeup

**Answer in this writeup: conditional/partial.**

- Obligation (E2) — placing `Gamma` in `FH(Q)` — is **discharged** for
  the rotation-lattice family (Fowler criterion, codim-2 fixed set with
  chi = 0).
- Obligation (S) — upgrading from finite CW complex to closed manifold — is
  **not fully validated in this document**. We present a strong rotation-route
  candidate chain (normal-bundle triviality and codim-2 surgery framework),
  but treat the final manifold-upgrade verification as an explicit remaining
  validation item.

## Assumptions Used (explicit)

1. `Gamma` is a cocompact lattice with finite isotropy on `X = G/K`; the
   relevant duality is Bredon/orbifold rational duality.
2. Fowler's criterion is applied only after checking the fixed-set
   Euler-vanishing hypothesis for nontrivial finite subgroups.
3. The manifold-upgrade step (S) requires an additional surgery theorem chain;
   unless each hypothesis is checked in full, S is treated as conditional.

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

## 4. Obligation S: From Finite Complex to Closed Manifold

Obligation S is not claimed as fully closed in this file; this section records
the open obstacles and the strongest current candidate route (S-rot-II).

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

### Approach II: Equivariant surgery on (M, tau) — BLOCKED for reflections

**Idea.** Work directly with the closed manifold `M` and the involution
`tau`. Eliminate the fixed set `F` by equivariant surgery to obtain `M'`
with a free Z/2-action. Then `M'/(Z/2)` is a closed manifold with
`pi_1 = Gamma`.

**Blocking obstruction.** The equivariant surgery framework of
Costenoble-Waner (arXiv:1705.10909) requires a **codimension-2 gap
hypothesis**: fixed sets of distinct isotropy subgroups must differ in
dimension by at least 2 (Condition 3.4(3)). For our Z/2-action on `M^n`
with codimension-1 fixed set `F^{n-1}`, this gap condition **fails**.
The Dovermann-Schultz framework (Springer LNM 1443) similarly requires
gap conditions.

**Status.** Blocked for the reflection construction (codimension-1 fixed
sets). However, the approach becomes viable for **rotational involutions**
(codimension-2 fixed sets) — see Approach IV below.

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

The E2 obligation (for reflections) requires `n` **even** (so the fixed set
has odd dimension and Euler characteristic zero). But the surgery obstruction
computation (Approach I) and the AHSS structure of `L_n(Z[Gamma]) tensor Q`
have better vanishing properties when `n` is **odd**. This tension between
E2 and S is a central difficulty for the reflection construction.

**Why reflections cannot work in odd dimension.** For a reflection on
`H^{2k+1}`, the fixed set is `H^{2k}` — a closed hyperbolic manifold of
even dimension. By the Gauss-Bonnet theorem, such manifolds have `chi != 0`.
So Fowler's criterion fails. This is not an artifact of the construction; it
is forced by Riemannian geometry.

### Approach IV: Rotation route (resolves the parity tension)

**Idea.** Replace the reflection (codimension-1 involution) with a
**rotational involution** (codimension-2 fixed set) in **odd** ambient
dimension. An order-2 isometry of `H^{2k+1}` that fixes `H^{2k-1}`
(codimension 2) is a "rotation by pi" in a normal 2-plane.

**Why this resolves the tension.** For `n = 2k+1` odd:

- Fixed set `H^{2k-1}` has dimension `2k-1` (odd), so `chi = 0`. Fowler
  criterion is satisfied: `Gamma in FH(Q)`.
- Surgery obstruction lives in `L_{2k+1}(Z[Gamma]) tensor Q`, which has
  favorable parity (odd total degree forces odd `p` in AHSS terms).
- The codimension-2 gap hypothesis (required by equivariant surgery theories,
  Costenoble-Waner arXiv:1705.10909) is **satisfied**, so Approach II
  (equivariant surgery) becomes available as a method for the S-branch.

**Lattice construction: DISCHARGED.** Details in
`problem7r-rotation-lattice-construction.md`. Summary:

1. **Quadratic form.** Let `k = Q(sqrt(2))`, `O_k = Z[sqrt(2)]`. Define
   `f = (1 - sqrt(2))x_0^2 + x_1^2 + ... + x_n^2` in `n+1` variables.
   Under the two real embeddings of `k`, `f` has signatures `(n, 1)` and
   `(n+1, 0)`. So `SO(f)` gives `SO(n, 1)` at one place and a compact
   group at the other.

2. **Uniform lattice.** `Gamma_0 = SO(f, O_k)` is a cocompact arithmetic
   lattice in `SO(n, 1)` (Borel-Harish-Chandra; cocompactness by Godement
   criterion since `f` is anisotropic over `k`).

3. **Order-2 rotation.** `sigma = diag(1, -1, -1, 1, ..., 1)` is in
   `SO(f, O_k)`: it preserves `f` (negates `x_1, x_2` in the `x_1^2+x_2^2`
   summand), has determinant `+1`, and has integer entries. Its fixed set on
   `H^n` is `H^{n-2}` (codimension 2).

4. **Congruence subgroup.** `pi = Gamma_0(I)` for ideal `I` coprime to 2:
   torsion-free (Minkowski), `sigma notin pi`, and `M = H^n/pi` is a closed
   hyperbolic manifold. The extension `1 -> pi -> Gamma -> Z/2 -> 1` with
   `Gamma = <pi, sigma>` is a cocompact lattice with order-2 torsion.

5. **Fowler application.** Fixed set has dimension `n-2 = 2k-1` (odd), so
   `chi = 0`. By Fowler's Main Theorem: `Gamma in FH(Q)`.

**E2 status for rotation route: DISCHARGED.**

**Obligation S resolution.** The manifold upgrade proceeds
via two sub-options available on the rotation route:

- **S-rot-I (Wall surgery in odd dimension).** Same three-obstacle structure
  as Approach I (Poincare complex, normal map, obstruction), but the
  obstruction computation benefits from odd L-theory parity.
- **S-rot-II (Equivariant surgery on (M, sigma)).** The codimension-2 gap
  hypothesis is satisfied, so the Costenoble-Waner framework (arXiv:1705.10909)
  applies. The equivariant surgery obstruction is computable in principle.

**This is the most promising remaining path for Problem 7.** The lattice
existence bottleneck is resolved. The open problem reduces to computing a
surgery obstruction (either Wall or equivariant) in the structurally
favorable odd-dimensional setting.

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

## 6. Theorem (scoped)

**Theorem.** Let `Gamma` be a cocompact lattice extension
`1 -> pi -> Gamma -> Z/2 -> 1` constructed from an arithmetic lattice in
`Isom(H^n)` containing an order-2 isometry, via either:

- **(Reflection route)** Section 3b: reflection lattice in `Isom(H^n)`,
  `n` even, `n >= 6`.
- **(Rotation route)** Approach IV: rotation lattice in `Isom(H^n)`,
  `n` odd, `n >= 7`. See `problem7r-rotation-lattice-construction.md`.

Then:

(a) **(Unconditional)** `Gamma in FH(Q)`: there exists a finite CW complex
    `Y` with `pi_1(Y) = Gamma` and `H_*(Y_tilde; Q) = 0` for `* > 0`.
    (Both routes discharge E2 via Fowler's criterion.)

(b) **(Rotation route, conditional S-step)** For the rotation lattice
    (Approach IV, `n = 7`, congruence ideal `I = (3)` in `Z[sqrt(2)]`):
    if the S-rot-II theorem chain hypotheses are fully verified (including
    trivial-holonomy normal-bundle identification and the exact obstruction
    vanishing in the relevant L-group), then one obtains a closed manifold
    `N` with `pi_1(N) = Gamma` and `H_*(N_tilde; Q) = 0` for `* > 0`.
    In this file we treat this as a candidate conditional closure path.

## 7. Path to Full Closure

Resolving obligation S requires computing a surgery obstruction. The
rotation route (Approach IV) has discharged the lattice-existence question
and is now the primary path. See `problem7-hypothetical-wirings.md` for
full wiring diagrams.

**Active path: Approach IV (rotation route).** The lattice construction
is complete (see `problem7r-rotation-lattice-construction.md`). E2 is
discharged. Two sub-options for obligation S:

1. **S-rot-II (Equivariant surgery — candidate vanishing path).**
   The Costenoble-Waner codimension-2 gap hypothesis is satisfied. The "cut
   and cap" method (Browder, López de Medrano) applies.

   **Candidate result to validate:** equivariant surgery obstruction
   theta = 0 (integrally), via the two-layer argument below.

   The argument has two layers:

   - **Rational vanishing (Step A: flat-normal-bundle argument).** The
     normal bundle ν of $F \in M$ is flat (totally geodesic embedding). By
     Chern-Weil, e(ν)⊗Q = 0. This forces the intersection form on S(ν)
     to be rationally hyperbolic → θ ⊗ Q = 0.

   - **Integral vanishing (trivial holonomy).** For the congruence ideal
     I with Norm(I) > 2 (e.g., I = (3)): the integrality constraint on
     rotation matrices over $\mathbb{Z}[\sqrt{2}]$, combined with the congruence condition
     g ≡ I mod I, forces the holonomy representation ρ: C → SO(2) to be
     trivial. So ν is a trivial bundle, e(ν) = 0 in H²(F; Z), and the
     circle bundle S(ν) = F × S¹ is a product. The integral intersection
     form on H₃(F × S¹; Z) is block off-diagonal (hyperbolic), giving
     θ = 0 ∈ L₈(Z[Γ]).

   **If theta = 0:** The equivariant "cut and cap" surgery succeeds. Remove
   the tubular neighborhood N(F) from M, obtaining W = M \ int(N(F)) with
   ∂W = S(ν) = F × S¹ and free Z/2-action on W. Since θ = 0, a cap V
   exists with ∂V = F × S¹ and free Z/2-action. Set M' = W ∪ V. Then
   N = M'/(Z/2) is a closed manifold with pi_1(N) = Gamma and rationally
   acyclic universal cover.

   See `problem7r-s-rot-obstruction-analysis.md` for full computation.

2. **S-rot-I (Wall surgery in odd dimension).** Fallback. Same three-obstacle
   structure as Approach I but with favorable odd L-theory parity and
   strictly fewer AHSS terms than the reflection case.

**Deprioritized paths:**

3. **Approach I (Wall surgery, reflection route).** Three successive open
   obstacles with structural headwinds (even L-theory parity).

4. **Approach III (orbifold resolution).** No technique identified.

5. **Approach II (equivariant surgery, reflection route).** Blocked by
   codimension-2 gap hypothesis (Costenoble-Waner, arXiv:1705.10909).

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
- S. R. Costenoble, S. Waner, *The Equivariant Spivak Normal Bundle and
  Equivariant Surgery for Compact Lie Groups*, arXiv:1705.10909.
- J. F. Davis, W. Lück, *On Nielsen Realization and Manifold Models for
  Classifying Spaces*, Trans. AMS 377 (2024), 7557-7600, arXiv:2303.15765.
- G. Avramidi, *Smith Theory, L2 Cohomology, Isometries of Locally Symmetric
  Manifolds, and Moduli Spaces of Curves*, arXiv:1106.1704.
- M. Belolipetsky, A. Lubotzky, *Finite Groups and Hyperbolic Manifolds*,
  arXiv:math/0406607.
- A. Borel, Harish-Chandra, *Arithmetic Subgroups of Algebraic Groups*,
  Annals of Mathematics 75 (1962), 485-535.
- J. Millson, M. S. Raghunathan, *Geometric Construction of Cohomology for
  Arithmetic Groups I*, Proc. Indian Acad. Sci. 90 (1981), 103-123.
- A. Ranicki, *Algebraic L-Theory and Topological Manifolds*, Cambridge
  Tracts in Mathematics 102, 1992.
- K. S. Brown, *Cohomology of Groups*, Springer GTM 87.
- W. Luck, *Transformation Groups and Algebraic K-Theory*, Springer LNM
  1408.
- C. T. C. Wall, *Surgery on Compact Manifolds*, 2nd ed., AMS.
