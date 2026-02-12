# Problem 7: Proof Outline (with Identified Gaps)

Date: 2026-02-12

## Status

**Sections 1–2 (lattice construction and E2) are complete.** These are
self-contained and depend only on standard arithmetic group theory and
Fowler's theorem.

**Section 3 (equivariant surgery) has three identified gaps** that prevent
it from being a finished proof. The geometric inputs (trivial normal bundle,
hyperbolic intersection form) are established; the gaps are in the surgery-
theoretic infrastructure connecting those inputs to the conclusion. See
Section 5 for a precise list.

## Target Theorem

**Theorem (conditional on gaps in Section 5).** There exists a uniform
lattice Γ in a real semisimple Lie group such that Γ contains an element of
order 2 and Γ is the fundamental group of a closed manifold whose universal
cover is rationally acyclic.

Concretely: for n = 7, the arithmetic lattice Γ = ⟨Γ₀(I), σ⟩ in SO(7,1)
constructed below (with I = (3) in Z[√2]) is the fundamental group of a
closed 7-manifold N with H̃*(Ñ; Q) = 0.

## Overview of the Argument

The argument has three parts:

1. **Lattice construction** (Section 1). Build an arithmetic uniform lattice
   Γ₀ in SO(n,1) for n = 7, containing an order-2 rotation σ with
   codimension-2 fixed set. Extract a torsion-free congruence subgroup
   π = Γ₀(I) and set Γ = ⟨π, σ⟩.

2. **E2: Γ ∈ FH(Q)** (Section 2). The fixed set of σ on M = H⁷/π has
   odd dimension (5), hence Euler characteristic zero. Fowler's Main
   Theorem gives a finite CW complex Y with π₁(Y) = Γ and rationally
   acyclic universal cover.

3. **S: closed manifold** (Section 3). Equivariant surgery ("cut and cap")
   on (M, σ) should produce a closed manifold N with π₁(N) = Γ and
   rationally acyclic universal cover. The geometric inputs are established:
   the congruence condition forces the normal bundle of the fixed set to be
   trivial, making the intersection form on the sphere bundle integrally
   hyperbolic. **Three gaps remain** in the surgery-theoretic bridge from
   these inputs to the conclusion (see Section 5).


---

## Section 1. Lattice Construction

### 1.1. Quadratic form

Let k = Q(√2) with ring of integers O_k = Z[√2]. Define the quadratic form
in 8 variables:

```
f(x₀, x₁, ..., x₇) = (1 - √2)x₀² + x₁² + x₂² + x₃² + x₄² + x₅² + x₆² + x₇²
```

The field k has two real embeddings: σ₁(√2) = +√2 and σ₂(√2) = −√2.

- Under σ₁: coefficient of x₀² is 1 − √2 < 0, so f^σ₁ has signature (7, 1).
- Under σ₂: coefficient of x₀² is 1 + √2 > 0, so f^σ₂ has signature (8, 0).

Thus G = SO(f) is an algebraic group over k with G(k_σ₁) = SO(7, 1) and
G(k_σ₂) compact.

### 1.2. Uniform lattice

The group Γ₀ = SO(f, O_k) is an arithmetic subgroup of SO(7, 1). By
Borel–Harish-Chandra, Γ₀ is a lattice. Since f^σ₂ is positive definite,
G is anisotropic over k (any k-rational isotropic vector for f would be
isotropic for the definite form f^σ₂). By the Godement compactness criterion,
Γ₀ is a **uniform** (cocompact) lattice.

### 1.3. Order-2 rotation

Define the involution:

```
σ = diag(1, −1, −1, 1, 1, 1, 1, 1)
```

acting on (x₀, x₁, ..., x₇) by negating x₁ and x₂.

**Claim:** σ ∈ SO(f, O_k).

*Proof.* The form splits as f = f₀ + f₁₂ + f_rest where f₁₂ = x₁² + x₂².
Since σ negates x₁, x₂ and f₁₂(−x₁, −x₂) = f₁₂(x₁, x₂), we have
f(σ(x)) = f(x). The determinant is det(σ) = 1·(−1)²·1⁵ = +1, and all
entries lie in {±1} ⊂ O_k. □

**Fixed set.** σ fixes the subspace {x₁ = x₂ = 0}, which is 6-dimensional.
The restriction of f to this subspace has signature (5, 1) under σ₁, so the
fixed set of σ on H⁷ is a totally geodesic copy of **H⁵** — codimension 2.

### 1.4. Congruence subgroup and extension

Let I = (3) in O_k = Z[√2]. Define the principal congruence subgroup:

```
π = Γ₀(I) = ker(SO(f, O_k) → SO(f, O_k/I))
```

**Properties of π:**

(i) *Torsion-free.* By Minkowski's lemma, since I is coprime to 2 and
Norm(I) = 9 > 2, the kernel π contains no nontrivial elements of finite
order.

(ii) *σ ∉ π.* The element σ reduces to a nontrivial element mod I because
its (1,1)-entry is −1, and −1 ≢ 1 (mod 3) in Z[√2]/(3).

(iii) *Normal, finite index.* As a congruence kernel, π ⊲ Γ₀ with
[Γ₀ : π] < ∞.

(iv) *M = H⁷/π is a closed hyperbolic 7-manifold.* Since π is torsion-free
and cocompact (finite-index in cocompact Γ₀), the quotient is a closed
manifold with contractible universal cover H⁷.

Set **Γ = ⟨π, σ⟩**. Since σ² = Id and σ ∉ π, we have the extension:

```
1 → π → Γ → Z/2 → 1
```

where Γ is a uniform lattice in SO(7,1) containing the order-2 element σ.


---

## Section 2. E2: Γ ∈ FH(Q)

### 2.1. Fixed-set Euler characteristic

The involution σ acts on M = H⁷/π (well-defined since π ⊲ Γ). The fixed
set Fix(σ, M) is a (possibly disconnected) closed, totally geodesic
submanifold. Each component has dimension 5 (= 7 − 2).

Every closed odd-dimensional manifold has Euler characteristic zero (by
Poincaré duality: the alternating sum of Betti numbers cancels in pairs).
So **χ(C) = 0** for every connected component C of the fixed set.

### 2.2. Fowler's criterion

**Theorem** (Fowler, arXiv:1204.4667, Main Theorem). Let G be a finite group
acting on a finite CW complex B. If for every nontrivial subgroup H ≤ G and
every connected component C of B^H, the Euler characteristic χ(C) = 0, then
the orbifold extension group π₁((EG × B)/G) lies in FH(Q).

**Application.** Take G = Z/2 = ⟨σ⟩ acting on B = M (a finite CW complex
with π₁(M) = π). The only nontrivial subgroup of Z/2 is itself. Every
component of M^{Z/2} has χ = 0 (Section 2.1). Therefore:

> **Γ ∈ FH(Q):** there exists a finite CW complex Y with π₁(Y) = Γ and
> H̃*(Ỹ; Q) = 0.

This completes the E2 obligation.


---

## Section 3. S: Closed Manifold via Equivariant Surgery

### 3.1. Setup

From Sections 1–2:

- M⁷ = H⁷/π is a closed hyperbolic 7-manifold.
- σ acts on M as an orientation-preserving involution (det σ = +1 in SO(f)).
- The action is **semi-free**: the isotropy subgroups are {1} (free orbits)
  and Z/2 (the fixed set F).
- F = Fix(σ, M) is a closed totally geodesic 5-manifold, of codimension 2.
- The normal bundle ν of F in M is an oriented rank-2 vector bundle.

### 3.2. Strategy: cut and cap

The goal is to modify (M, σ) by equivariant surgery to produce M' with a
**free** Z/2-action. Then N = M'/(Z/2) is a closed manifold with π₁(N) = Γ.

**Step 1 (Cut).** Remove an equivariant tubular neighborhood N(F) ≅ D(ν)
(the disk bundle of ν). The boundary is S(ν) (the circle bundle of ν).
The Z/2-action on each fiber S¹ is rotation by π (i.e., the antipodal map),
which is **free**. So Z/2 acts freely on ∂W = S(ν).

Set W = M \ int(N(F)). Then W is a compact 7-manifold with boundary
∂W = S(ν), and Z/2 acts freely on W.

**Step 2 (Cap).** Find a compact 7-manifold V with ∂V = S(ν), equipped with
a free Z/2-action extending the one on S(ν).

**Step 3 (Assemble).** Set M' = W ∪_{S(ν)} V. The Z/2-action on M' is free.
Set N = M'/(Z/2).

### 3.3. What the cap must satisfy

For N to solve Problem 7, the cap V must be chosen so that M' = W ∪ V and
N = M'/(Z/2) have the right properties. This requires careful π₁ control.

**The meridian issue.** Removing a codimension-2 submanifold from a manifold
does NOT preserve π₁ — it adds a meridional generator. Specifically:

- π₁(M \ F) surjects onto π₁(M) = π, with kernel normally generated by the
  meridian μ (a small loop linking F once in the normal plane).
- π₁(W) = π₁(M \ F) (homotopy equivalence), so π₁(W) is LARGER than π.
- The covering W → W/(Z/2) is a regular double cover, giving the sequence
  1 → π₁(W) → π₁(W/(Z/2)) → Z/2 → 1.

So **π₁(W/(Z/2)) ≠ Γ** in general — it is a Z/2-extension of the larger
group π₁(M \ F), not of π.

**Required π₁ control.** For π₁(N) = Γ, the cap V must kill the meridian:

(a) The inclusion S(ν) ↪ V must send the meridian μ ∈ π₁(S(ν)) to the
    identity in π₁(V). Since S(ν) = F × S¹ (trivial ν), this means
    π₁(V) receives π₁(F) = C but kills Z = ⟨μ⟩.

(b) If (a) holds, Van Kampen gives π₁(M') = π₁(W) *_{C × Z} π₁(V),
    and killing μ yields π₁(M') = π₁(W)/⟨⟨μ⟩⟩ = π₁(M) = π.

(c) The Z/2-action on M' then gives 1 → π → π₁(N) → Z/2 → 1, and this
    extension should be Γ (since the Z/2-action on π₁(M') = π is
    conjugation by σ, the same as the original action).

**Gap (G1).** Whether the equivariant surgery machinery produces a cap V
satisfying condition (a) is not proved here. The surgery theory guarantees
a cap with controlled algebraic topology, but the specific π₁ property
(killing the meridian while preserving the C-factor) requires verification
against the actual theorem hypotheses. See Section 5.

**Required rational acyclicity control.**

(d) The universal cover of N is obtained from M̃' = H⁷ \ (lifts of N(F))
    ∪ (lifts of V). For rational acyclicity, the lifts of V must not
    introduce new rational homology.

**Gap (G3).** This homology control is a condition on the surgery output
that requires verification. See Section 5.

### 3.4. The surgery obstruction

The obstruction to finding a suitable cap V lies in the L-group
L₈(Z[Γ]). (The dimension is 2k + 2 = 8 for k = 3, one more than the
manifold dimension, because the "cap" is a cobordism problem.)

**Why the obstruction lies in ker(res).** The restriction map
res: L₈(Z[Γ]) → L₈(Z[π]) measures the underlying non-equivariant surgery
obstruction. Since M is already a genuine closed manifold, the non-equivariant
problem has zero obstruction. So the equivariant surgery obstruction
**θ ∈ ker(res)**.

### 3.5. Trivial holonomy

**Proposition.** For I = (3) in Z[√2], the holonomy representation
ρ: π₁(F) → SO(2) of the normal bundle ν is trivial.

*Proof.* The normal bundle ν is flat (F is totally geodesic in the locally
symmetric space M). Its holonomy representation ρ sends g ∈ C = π^σ
(the σ-centralizer in π) to the rotation by which g acts on the normal
2-plane to F.

**Identification of ρ with the matrix block.** In the locally symmetric
space M = H⁷/π, the normal connection of the totally geodesic F is induced
from the ambient Levi-Civita connection. Since the isometric action of
Γ₀ on H⁷ preserves the orthogonal decomposition T_x H⁷ = T_x(H⁵) ⊕ ν_x
at each fixed point x ∈ H⁵, the parallel transport along a deck
transformation g ∈ C in the normal bundle is exactly the action of g on
the normal 2-plane. In the matrix representation, this is the (x₁, x₂)-block
of g. (This uses: (i) totally geodesic ⟹ the normal connection equals
the ambient connection restricted to the normal bundle, and (ii) isometries
preserve parallel transport.)

An element g ∈ C ⊂ π = Γ₀(I) satisfies g ≡ I₈ (mod I). In particular,
the (x₁, x₂)-block of g is a 2×2 rotation matrix R(θ) with entries
cos θ, sin θ ∈ Z[√2], satisfying R(θ) ≡ I₂ (mod I).

The rotation matrices over Z[√2] with integer entries are exactly:

| θ    | (cos θ, sin θ) |
|------|----------------|
| 0    | (1, 0)         |
| π/2  | (0, 1)         |
| π    | (−1, 0)        |
| 3π/2 | (0, −1)        |

(These correspond to the 4th roots of unity in Z[√2][i] = Z[√2, ζ₄].)

The congruence condition R(θ) ≡ I₂ (mod (3)) requires cos θ ≡ 1 and
sin θ ≡ 0 modulo (3). For each non-identity rotation:

- θ = π/2: cos θ = 0 ≢ 1 (mod 3).
- θ = π: cos θ = −1 ≢ 1 (mod 3), since Norm(3) = 9 > 2.
- θ = 3π/2: cos θ = 0 ≢ 1 (mod 3).

Only θ = 0 satisfies the congruence condition. Therefore ρ is trivial. □

**Corollary.** ν is the trivial bundle: ν ≅ F × R².

*Proof.* A flat bundle with trivial holonomy is trivial. □

### 3.6. The intersection form on S(ν) is integrally hyperbolic

Since ν is trivial, S(ν) = F × S¹ (a product). The middle cohomology of
the 6-manifold S(ν) in degree 3 decomposes by the Künneth theorem:

```
H³(F × S¹; Z) ≅ H³(F; Z) ⊕ H²(F; Z)
```

where the first summand is the pullback from F ("base classes") and the
second is the product with the generator u ∈ H¹(S¹; Z) ("fiber classes").

**Proposition.** The intersection form ⟨−,−⟩ on H₃(S(ν); Z) is hyperbolic.

*Proof.* The cup product structure on H*(F × S¹; Z) = H*(F; Z) ⊗ H*(S¹; Z)
gives three components:

**Base × Base.** For α, β ∈ H³(F): α ∪ β ∈ H⁶(F × S¹). Since
H⁶(F; Z) = 0 (dim F = 5) and H⁶(F × S¹) ≅ H⁵(F) ⊗ H¹(S¹) = Z, the
class α ∪ β = 0 in H⁶(F; Z) pulled back to F × S¹ is zero.
So **⟨base, base⟩ = 0**.

**Fiber × Fiber.** For αu, βu with α, β ∈ H²(F): (αu) ∪ (βu) =
(α ∪ β) · u² = 0 (since u² = 0 in H*(S¹; Z)).
So **⟨fiber, fiber⟩ = 0**.

**Base × Fiber.** For α ∈ H³(F) and βu with β ∈ H²(F):
α ∪ (βu) = (α ∪ β) · u ∈ H⁵(F) ⊗ H¹(S¹) = H⁶(F × S¹) = Z.
This equals the Poincaré duality pairing ⟨α, β⟩_F on F.

The form is therefore block off-diagonal:

```
    ⎛  0    PD_F ⎞
    ⎝ ±PD_F   0  ⎠
```

where PD_F: H₃(F) × H₂(F) → Z is the Poincaré duality pairing on F.
Both H₃(F) ⊕ 0 and 0 ⊕ H₂(F) are Lagrangians (the form vanishes on each
summand), so the form is **hyperbolic**. □

### 3.7. The surgery obstruction

**Established geometric fact.** The intersection form on H₃(S(ν); Z) is
integrally hyperbolic (Section 3.6). This is a direct computation from
ν trivial, using only the Künneth theorem and Poincaré duality.

**Claimed conclusion.** The equivariant surgery obstruction vanishes:
θ = 0 ∈ L₈(Z[Γ]).

**Argument (conditional on Gap G2).** The obstruction θ lies in
ker(res) ⊆ L₈(Z[Γ]) (Section 3.4). The standard Browder–Quinn stratified
surgery framework (Appendix A) identifies the fixed-stratum component of θ
with the Witt class of the intersection form on S(ν).

Since the intersection form on H₃(S(ν); Z) is integrally hyperbolic, its
Witt class is zero. If the identification holds as stated, θ = 0.

**Gap (G2): the identification.** The argument requires that:

(i) The full surgery obstruction θ ∈ L₈(Z[Γ]) is determined by its
    fixed-stratum component (i.e., the free-stratum component vanishes
    because M is already a manifold — this is the ker(res) argument).

(ii) The fixed-stratum component equals the Witt class of the intersection
     form on S(ν). This is the Browder–Quinn identification.

(iii) No additional 2-primary components of L₈(Z[Γ]) (such as UNil
      contributions, Browder–Livesay invariants, or Arf invariants) survive
      outside the Witt class identification.

Point (i) is standard. Point (ii) is sketched in Appendix A with literature
references, but the exact theorem statement (with hypotheses verified for
our setting) is not given. Point (iii) is the most delicate: the integral
hyperbolicity of the intersection form should kill the 2-primary obstructions
(since a hyperbolic form over Z has trivial Arf invariant and trivial
mod-2 reduction), but this requires checking that the Browder–Quinn
decomposition captures ALL components of the obstruction, not just the
rational part. See Section 5.

### 3.8. Conclusion (conditional)

**If** the three gaps (G1, G2, G3) are closed — specifically, if the
equivariant surgery produces a cap V that kills the meridian (G1), if the
Browder–Quinn identification holds integrally in this setting (G2), and if
rational acyclicity is preserved (G3) — **then:**

(a) θ = 0 implies the equivariant cobordism problem has a solution: there
    exists a cap V with ∂V = S(ν) = F × S¹, carrying a free Z/2-action.

(b) M' = W ∪_{S(ν)} V is a closed 7-manifold with free Z/2-action.

(c) If V kills the meridian μ (Gap G1), then π₁(M') = π by Van Kampen
    (Section 3.3(a–b)), and π₁(N) = Γ via the extension
    1 → π → π₁(N) → Z/2 → 1 (Section 3.3(c)).

(d) If rational acyclicity is preserved through the surgery (Gap G3), then
    H̃*(Ñ; Q) = 0.

The geometric inputs (Sections 3.5–3.6) are unconditional. The gaps are in
the surgery-theoretic infrastructure. See Section 5 for discussion.


---

## Section 4. Summary of the Argument Chain

```
(a) Lattice construction                                         [PROVED]
    Γ = ⟨π, σ⟩ with π = Γ₀((3)) < SO(f, Z[√2]), σ = diag(1,−1,−1,1,...,1).
    Γ is a uniform lattice in SO(7,1) with 2-torsion.

(b) E2 discharge                                                 [PROVED]
    Fixed set F has dim 5 (odd), χ = 0. Fowler Main Theorem: Γ ∈ FH(Q).

(c) Normal bundle is trivial                                     [PROVED]
    Congruence condition (Norm(I) > 2, I coprime to 2)
       ⟹  holonomy ρ trivial              [integrality of rotation angles]
       ⟹  ν ≅ F × R²                      [trivial flat bundle]

(d) Intersection form is hyperbolic                              [PROVED]
    ν trivial ⟹ S(ν) = F × S¹
              ⟹ intersection form on H₃(F × S¹; Z) is hyperbolic

(e) Surgery obstruction vanishes                                 [GAP G2]
    Requires: Browder–Quinn identification (Witt class = θ) holds
    integrally in this setting, capturing all components including
    2-primary.

(f) Cut and cap succeeds with correct π₁                        [GAP G1]
    Requires: surgery produces cap V that kills the meridian μ.
    If so, Van Kampen gives π₁(M') = π, then π₁(N) = Γ.

(g) Rational acyclicity preserved                                [GAP G3]
    Requires: surgery does not introduce rational homology in
    universal cover.
```


---

## Section 5. Identified Gaps

Three gaps prevent this from being a complete proof. They are all in the
surgery-theoretic infrastructure of Section 3, not in the geometric
computations (Sections 1–2 and 3.5–3.6).

### Gap G1: π₁ control through cut-and-cap

**The problem.** Removing F (codimension 2) from M adds a meridional
generator μ to π₁. The quotient W/(Z/2) has fundamental group strictly
larger than Γ. For π₁(N) = Γ, the cap V must kill μ while preserving the
π₁(F)-factor.

**What is needed.** A theorem or construction guaranteeing that the
equivariant surgery cap V satisfies: the inclusion S(ν) ↪ V sends
⟨μ⟩ = Z to 1 ∈ π₁(V) and sends π₁(F) = C surjectively onto π₁(V).

**Discussion.** In the López de Medrano framework for semi-free involutions
(Ergebnisse 73, Ch. V), the standard cap for a trivial circle bundle
S(ν) = F × S¹ is V = F × D², which satisfies π₁(V) = π₁(F) = C and kills
the S¹ factor. This is the "obvious" cap. The question is whether the
surgery obstruction θ = 0 guarantees that this obvious cap (or one with the
same π₁ properties) can be equipped with a compatible free Z/2-action. This
is plausible but unproved here.

### Gap G2: Obstruction identification (integral)

**The problem.** The argument claims θ = 0 ∈ L₈(Z[Γ]) from the
hyperbolicity of the intersection form on S(ν). This requires two things:

(a) The Browder–Quinn identification: the fixed-stratum component of θ
    equals the Witt class of this intersection form. Appendix A sketches
    the literature chain but does not give a precise theorem with
    hypotheses verified for our setting.

(b) Integral completeness: the Witt class captures the FULL integral
    obstruction, not just the rational part. Specifically, 2-primary
    obstructions (UNil, Browder–Livesay, Arf) must be shown to vanish or
    to be subsumed by the Witt class vanishing.

**What is needed.** Either:
- A precise citation giving: for semi-free Z/2 actions on closed manifolds
  with codimension-2 fixed set and trivial normal bundle, the equivariant
  surgery obstruction θ ∈ L_{n+1}(Z[Γ]) is determined by the Witt class
  of the intersection form on S(ν), integrally. Or:
- Separate treatment of the 2-primary components showing they vanish
  independently.

**Discussion.** The integral hyperbolicity is stronger than rational
hyperbolicity. A hyperbolic form over Z has trivial Arf invariant, and
the mod-2 intersection form is also hyperbolic. This should kill the
2-primary obstructions. But the precise mechanism (is it through the
Browder–Quinn machine, or through a separate Arf/Browder-Livesay
computation?) needs to be spelled out.

### Gap G3: Rational acyclicity through surgery

**The problem.** The surgery replaces a neighborhood of F in the universal
cover H⁷ with lifts of the cap V. The universal cover of M' is
H⁷ \ (equivariant lifts of N(F)) ∪ (equivariant lifts of V). For rational
acyclicity, these lifts must not create new rational homology.

**What is needed.** Verification that the surgery preserves
H̃*(−; Q) = 0 for the universal cover. This should follow from the
surgery being performed on a contractible ambient space (H⁷), but the
argument needs to be written out, not just asserted.

**Discussion.** If the cap V has H̃*(Ṽ; Q) = 0 (where Ṽ is the universal
cover of V), then the Mayer–Vietoris sequence for M̃' = (H⁷ \ lifts of
N(F)) ∪ lifts of V should give the desired acyclicity. The condition
H̃*(Ṽ; Q) = 0 is a constraint on the surgery output.

### Assessment

The gaps are ordered by expected difficulty:

- **G1** (π₁ control): likely closable by citing López de Medrano's
  explicit cap construction for trivial normal bundles.
- **G3** (rational acyclicity): likely closable by a Mayer–Vietoris
  argument once the cap's topology is specified.
- **G2** (obstruction identification): requires the most work — either
  locating a precise theorem in the Browder–Quinn/Ranicki literature
  or building a bespoke argument for the 2-primary components.


---

## Appendix A. The Browder–Quinn Identification (Sketch)

This appendix provides the identification used in Section 3.7: the
equivariant surgery obstruction θ at the fixed stratum equals the Witt class
of the intersection form on the sphere bundle S(ν).

**Caveat.** This appendix is a sketch, not a self-contained proof. The
identification of θ with the Witt class of the intersection form on S(ν)
is assembled from several results in the equivariant surgery literature.
The exact hypotheses of each cited theorem are not fully verified here for
our specific setting (semi-free Z/2 on a closed hyperbolic 7-manifold with
codimension-2 totally geodesic fixed set and trivial normal bundle). This
verification is Gap G2.

### A.1. Stratified surgery exact sequence

For a semi-free G-action (G = Z/2) on a closed manifold M^n with fixed set
F^{n−c} of codimension c ≥ 2, the equivariant surgery theory of
Browder–Quinn (1975) and Hughes–Weinberger (arXiv:math/9807156) provides a
**stratified surgery exact sequence** analogous to the classical Wall surgery
exact sequence:

```
... → S^{str}(M, F) → N^{str}(M, F) → L^{str}(M, F) → ...
```

where S^{str} is the structure set of equivariant manifold structures,
N^{str} is the set of equivariant normal invariants, and L^{str} is the
obstruction group.

**Hypothesis check (incomplete).** The Browder–Quinn theory requires the
stratification to satisfy certain transversality and local triviality
conditions. Our setting (isometric involution on a Riemannian manifold
with smooth fixed set) should satisfy these, but the verification is not
given here.

### A.2. Decomposition into strata

The key feature of the Browder–Quinn theory is that the obstruction group
L^{str} decomposes into contributions from each stratum. For a semi-free
action with two strata (free part M \ F and fixed part F), the obstruction
maps to:

```
L^{str}(M, F) → L_n(Z[π₁(M/G)]) ⊕ L_{n−c+1}(Z[π₁(F)])
```

The first factor is the free-stratum (augmentation) contribution, and the
second is the fixed-stratum (sign-factor) contribution, shifted in dimension
by the codimension c and accounting for the normal data.

**Important caveat.** The map above is part of a longer exact sequence, not
an isomorphism in general. Whether L^{str} is determined by the two factors
on the right depends on the connecting maps. For the "cut and cap" problem,
the free-stratum contribution should vanish (this is the ker(res) argument,
Section 3.4), leaving the fixed-stratum factor. But whether the fixed-
stratum factor is the COMPLETE obstruction (vs. a quotient or subgroup of
it) requires checking the exact sequence. This is part of Gap G2.

### A.3. The fixed-stratum obstruction

The fixed-stratum surgery obstruction is identified via the **equivariant
Thom isomorphism** for the normal bundle ν:

```
θ_F ∈ L_{n−c+1}(Z[π₁(F)]) (shifted by normal data)
```

In our case (n = 7, c = 2), this is θ_F ∈ L₆(Z[C]) where C = π₁(F) = π^σ.

The classical identification (Browder 1968, López de Medrano 1971, Ranicki
1998) for codimension-2 fixed sets of semi-free actions relates θ_F to the
**boundary surgery problem** on S(ν):

> The obstruction θ_F is the surgery obstruction for the boundary
> manifold S(ν) — equivalently, the Witt class of the (−1)^{(n−c)/2}-
> symmetric intersection form on H_{(n−c)/2}(S(ν); Z).

In our case, n − c = 5 (odd), and the relevant form is on H₃(S(ν); Z),
which is a skew-symmetric (= symplectic) intersection form on the
6-manifold S(ν).

### A.4. Connection to the AHSS

The Farrell–Jones isomorphism (Bartels–Farrell–Lück, arXiv:1101.0469,
which holds for cocompact lattices in SO(n,1)) identifies:

```
L₈(Z[Γ]) ⊗ Q ≅ H₈^{Or(Γ)}(E_{Fin}Γ; L^{−∞}(Z[−]) ⊗ Q)
```

The Atiyah–Hirzebruch spectral sequence (AHSS) computing this equivariant
homology group has E²-terms:

```
E²_{p,q} = H_p^{Z/2}(M; M_q)
```

where the coefficient system M_q decomposes into an augmentation factor
(the free-stratum contribution, mapping injectively under res) and a sign
factor (the fixed-stratum contribution, mapping to zero under res).

The sign factor, Thom-shifted by codimension 2, contributes:

```
ker(res) ⊗ Q at total degree 8 = H₂(F; Q) at AHSS position (4, 4)
```

The identification of this AHSS class with the Witt class of the
intersection form on S(ν) proceeds through three steps:

**Step 1.** The sign-factor contribution at (4, 4) is computed by the
equivariant Thom isomorphism for ν, which shifts the fixed-stratum
L-homology by the codimension. This is standard (Lück–Reich 2005, §7).

**Step 2.** The fixed-stratum L-homology class at F is the surgery
obstruction θ_F localized at the fixed-point stratum (Section A.3). By
the Browder–Quinn decomposition, this is the Witt class of the form on
S(ν).

**Step 3.** The composite identification takes the AHSS class at (4, 4)
through the Thom isomorphism to the L-class at F, and then through the
Browder–Quinn localization to the Witt class on S(ν).

### A.5. The identification in the trivial-ν case

When ν is trivial (our case, Section 3.5), the identification simplifies
dramatically. The Thom isomorphism becomes canonical (no twisting), and
S(ν) = F × S¹ is a product. The intersection form on H₃(F × S¹; Z) is
manifestly hyperbolic (Section 3.6 gives an explicit computation requiring
only the Künneth theorem and Poincaré duality on F). The Witt class is
therefore zero by direct computation, without needing the full Browder–Quinn
machinery for the vanishing — only for the identification that this Witt
class IS the surgery obstruction.

### A.6. Literature chain

The three-step identification is supported by:

1. **Browder (1968).** Surgery and the Theory of Differentiable
   Transformation Groups. Introduces the codimension-2 framework for
   semi-free actions, identifies the surgery obstruction with the linking
   form on the boundary of the tubular neighborhood.

2. **López de Medrano (1971).** Involutions on Manifolds. Develops the
   "cut and cap" method for semi-free involutions. Shows the obstruction
   to capping is determined by the surgery kernel on the sphere bundle
   boundary.

3. **Ranicki (1998).** High-Dimensional Knot Theory: Algebraic Surgery
   in Codimension 2. Provides the algebraic framework (Γ-groups) for
   codimension-2 surgery. Identifies the obstruction with the Witt class
   of the Seifert form, which for sphere bundles of normal bundles
   reduces to the intersection form on S(ν).

4. **Browder–Quinn (1975), Hughes–Weinberger (2001).** Stratified surgery
   exact sequence, decomposition of L-groups by strata. The fixed-stratum
   contribution is localized at F and determined by the normal data.

5. **Bartels–Farrell–Lück (2014).** FJ conjecture for cocompact lattices.
   Provides the AHSS computation of L*(Z[Γ]) ⊗ Q.

**Confidence level.** The geometric input (trivial ν → hyperbolic form) is
unconditional and elementary. The surgery-theoretic identification (Witt
class = θ) is assembled from standard results, but the assembly for our
specific setting has not been fully verified. The novel content is the
observation that trivial holonomy (forced by the congruence condition) makes
ν trivial. This is a geometric observation, not a new theoretical
development. The gap is in the bridge from this observation to the L-group
vanishing.


---

## Appendix B. Conditions on the Congruence Ideal

The proof uses the congruence ideal I = (3) in Z[√2]. The two conditions
on I are:

1. **I coprime to 2.** Ensures π = Γ₀(I) is torsion-free (Minkowski's
   lemma) and that σ ∉ π (since −1 ≢ 1 mod I when 2 is invertible mod I).

2. **Norm(I) > 2.** Ensures the holonomy of the normal bundle is trivial
   (Section 3.5). This is because the only rotation angles (cos θ, sin θ)
   with integer entries in Z[√2] that satisfy cos θ ≡ 1, sin θ ≡ 0 mod I
   with Norm(I) > 2 are (1, 0).

The ideal I = (3) satisfies both: 3 is coprime to 2 in Z[√2], and
Norm(3) = 9 > 2. (The element 3 ∈ Z[√2] has norm N_{Q(√2)/Q}(3) = 9.)

**Other valid choices:** Any prime ideal I = (p) with p an odd rational prime
that remains inert or splits in Z[√2], and Norm(I) > 2. For instance,
I = (5), I = (7), etc. The choice I = (3) is the simplest.


---

## References

- A. Bartels, F. T. Farrell, W. Lück, *The Farrell–Jones Conjecture for
  Cocompact Lattices in Virtually Connected Lie Groups*, JAMS 27 (2014),
  339–388; arXiv:1101.0469.
- A. Borel, Harish-Chandra, *Arithmetic Subgroups of Algebraic Groups*,
  Annals of Mathematics 75 (1962), 485–535.
- W. Browder, *Surgery and the Theory of Differentiable Transformation
  Groups*, Proc. Conf. Transformation Groups (New Orleans, 1967), Springer,
  1968, 1–46.
- W. Browder, F. Quinn, *A Surgery Theory for G-Manifolds and Stratified
  Sets*, Manifolds — Tokyo 1973, University of Tokyo Press, 1975, 27–36.
- S. E. Cappell, J. L. Shaneson, *The Codimension Two Placement Problem
  and Homology Equivalent Manifolds*, Annals of Mathematics 99 (1974),
  277–348.
- F. Connolly, J. F. Davis, *On the Calculation of UNil*,
  arXiv:math/0304016.
- S. R. Costenoble, S. Waner, *The Equivariant Spivak Normal Bundle and
  Equivariant Surgery for Compact Lie Groups*, arXiv:1705.10909.
- J. Fowler, *Finiteness Properties of Rational Poincaré Duality Groups*,
  arXiv:1204.4667.
- B. Hughes, S. Weinberger, *Surgery and Stratified Spaces*, Surveys on
  Surgery Theory Vol. 2, Annals of Mathematics Studies 149, 2001;
  arXiv:math/9807156.
- F. Kamber, Ph. Tondeur, *On Flat Bundles*, Bull. AMS 72 (1966), 846–849.
- S. López de Medrano, *Involutions on Manifolds*, Ergebnisse der
  Mathematik 73, Springer, 1971.
- W. Lück, H. Reich, *The Baum–Connes and the Farrell–Jones Conjectures in
  K- and L-Theory*, Handbook of K-Theory, 2005, 703–842.
- J. Millson, M. S. Raghunathan, *Geometric Construction of Cohomology for
  Arithmetic Groups I*, Proc. Indian Acad. Sci. 90 (1981), 103–123.
- A. Ranicki, *High-Dimensional Knot Theory: Algebraic Surgery in
  Codimension 2*, Springer Monographs in Mathematics, 1998.
- C. T. C. Wall, *Surgery on Compact Manifolds*, 2nd ed., Mathematical
  Surveys and Monographs 69, AMS, 1999.
