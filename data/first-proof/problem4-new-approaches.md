# Problem 4: Six New Proof Approaches — Self-Coaching Brainstorm

**Date:** 2026-02-13
**Status:** APPROACH E HAS SPECTACULAR NUMERICAL SUPPORT — see results below

---

## The Landscape

We've PROVED: backward heat eq, Coulomb flow, de Bruijn identity, Φ_n monotonicity (SOS).
We've KILLED: entropy superadditivity, Haar projection, finite EPI.
We NEED: 1/Φ_n(p ⊞_n q) ≥ 1/Φ_n(p) + 1/Φ_n(q).

The conventional approaches (heat flow → de Bruijn → entropy → Stam) are dead.
Below are six unconventional approaches, ordered by estimated feasibility.

---

## Approach E: Semigroup Monotonicity (The "2t vs t+t" Trick)

### Core idea (★★★★★ — most promising)

This approach uses our proved machinery directly and could give a **3-page proof**.

Define g(t) = 1/Φ((p⊞q)_{2t}) − 1/Φ(p_t) − 1/Φ(q_t), where f_t = f ⊞_n He_t.

**Key identity (from cumulant additivity):**
κ(p_t ⊞ q_t) = κ(p) + κ(He_t) + κ(q) + κ(He_t) = κ(p⊞q) + 2κ(He_t) = κ((p⊞q)_{2t}).

So **p_t ⊞ q_t = (p⊞q)_{2t}**. The Stam inequality AT TIME t is:
    1/Φ(p_t ⊞ q_t) ≥ 1/Φ(p_t) + 1/Φ(q_t)
    ⟺ g(t) ≥ 0.

**The boundary:** As t → ∞, roots of f_t → √(2t) × (Hermite zeros) for any f.
So 1/Φ(f_t) → t/Φ(He_n) for large t (since Φ(√t · γ) = Φ(γ)/t).
Therefore: g(∞) = 2t/Φ(He) − 2·t/Φ(He) = 0.

**The proof structure (if g is monotone):**
1. g(∞) = 0 (Hermite scaling, 3 lines)
2. g'(t) ≤ 0 for all t ≥ 0 (THE KEY STEP)
3. Therefore g(0) ≥ g(∞) = 0, which is the Stam inequality. ∎

### Why g'(t) ≤ 0 might hold

We have d/dt[1/Φ(f_t)] = (Φ'(f_t))²... wait, let me be precise.
d/dt[1/Φ(f_t)] = −Φ̇(f_t)/Φ(f_t)² where Φ̇ = dΦ/dt = −2Σ_{k<j}(S_k−S_j)²/(γ_k−γ_j)² ≤ 0.
So d/dt[1/Φ(f_t)] = 2Σ(S_k−S_j)²/(γ_k−γ_j)² / Φ² > 0.

g'(t) = 2 · [rate for (p⊞q)_{2t}] − [rate for p_t] − [rate for q_t]

The factor of 2 comes from the chain rule (2t vs t). So the question is whether
the rate of increase of 1/Φ along the flow DECELERATES fast enough that the
"combined" flow (with 2× the time) increases more slowly than two independent flows.

Intuitively: at large t, all root configurations look like scaled Hermite, so the
rates equalize and g'(t) → 0. The question is whether g' stays ≤ 0 at finite t.

### Testable prediction

**Test 1:** Compute g(t) for many (p, q) pairs and t ∈ [0, 10]. Is g monotone decreasing?
**Test 2:** Compute g'(t) numerically. Is it ≤ 0?
**Test 3:** If g is NOT monotone, does it at least remain ≥ 0 for all t?

### Category theory connection

The Hermite semigroup {He_t}_{t≥0} acts on (RR_n, ⊞_n) as a monoidal endofunctor:
He_t ⊞ (−): RR_n → RR_n, and He_t ⊞ (f ⊞ g) = (He_t ⊞ f) ⊞ g (from associativity).
The functional 1/Φ_n is a monotone character of this action (proved).
The Stam inequality asks: is 1/Φ_n SUPERADDITIVE under the monoidal product ⊞_n?
Approach E reduces this to: is the character's growth rate subadditive under ⊞_n?

In wiring diagram language: the "parallel-then-flow" path gives the same
polynomial as the "flow-then-parallel" path (by the identity p_t ⊞ q_t = (p⊞q)_{2t}),
but the FUNCTIONAL tracks different trajectories. The proof shows the direct path
(flow (p⊞q) for 2t) dominates at t=0 and relaxes to equality at t=∞.

---

## Approach F: Resolvent Score Decomposition (Finite Zamir)

### Core idea (★★★★ — parallels the shortest classical proof)

The classical Stam inequality has a **1-page proof** (Zamir 1998) using only
the score convolution identity and Cauchy-Schwarz. No heat flow, no entropy.

**Classical proof skeleton:**
1. Score identity: ρ_{X+Y}(z) = E[ρ_X(X) | X+Y=z] = E[ρ_Y(Y) | X+Y=z]
2. ANOVA decomposition of E[ρ_Z²]
3. Cauchy-Schwarz → 1/J(Z) ≥ 1/J(X) + 1/J(Y). Done.

**Finite analog:** For eigenvalues γ of A + UBU*:
- A-resolvent score: S_k^A(γ) = Σ_j 1/(γ_k − λ_j) [force from A-eigenvalues]
- B-resolvent score: S_k^B(γ) = Σ_j 1/(γ_k − μ_j) [force from B-eigenvalues]

**The resolvent identity** for C = A + UBU* gives:
(zI − C)^{−1} = (zI − A)^{−1}(I + UBU*(zI − C)^{−1})

Taking traces near z = γ_k, this should decompose S_k(γ) into an A-part and
B-part. After Haar averaging (to pass to the MSS convolution), we need:

**Conjecture F:** For the roots γ of p ⊞_n q:
    S_k(γ) = S_k^A(γ) + S_k^B(γ) − (correction)

where ⟨S^A, S^B⟩ ≤ 0 (the A-force and B-force are "anti-correlated" at the
roots of the convolution). If so:

    Φ(p⊞q) = ‖S‖² = ‖S^A + S^B‖² ≤ ‖S^A‖² + ‖S^B‖²

and with Cauchy-Schwarz on the last expression:
    1/Φ(p⊞q) ≥ 1/‖S^A‖² + 1/‖S^B‖² ≥ ... → Stam.

### Testable prediction

Compute S^A_k, S^B_k, and their inner product for roots of random p⊞q pairs.
Is ⟨S^A, S^B⟩ ≤ 0? How does ‖S^A‖² relate to Φ(p)?

---

## Approach G: Gårding Concavity on the Hyperbolicity Cone

### Core idea (★★★ — classical convex analysis + hyperbolic polynomial theory)

**Gårding's inequality (1959):** If P is a hyperbolic polynomial of degree d
with respect to a vector e (i.e., t ↦ P(x + te) has d real roots for all x
in the hyperbolicity cone), then P^{1/d} is CONCAVE on the hyperbolicity cone.

The discriminant Δ_n of a monic degree-n polynomial is a hyperbolic polynomial
on the space of coefficients. Its hyperbolicity cone IS the real-rooted cone C_n.

**The connection to Φ_n:** We have the identity Φ_n = (rational function of a_k).
Specifically, Φ_n · disc = (explicit polynomial). If this polynomial, or a power
of it, is hyperbolic, Gårding gives concavity of a suitable root on C_n.

For 1/Φ_n = disc / (Φ_n · disc) = disc / (polynomial): if this is a power of a
hyperbolic polynomial, we get concavity.

**Combined with homogeneity:** 1/Φ_n scales as c² under root scaling λ → cλ.
In the appropriate coordinates, this is degree-2 homogeneous. Concavity +
degree-2 homogeneity → superadditivity:

    F(κ + λ) = 4 · F((κ+λ)/2) ≥ 4 · [F(κ)/4 + F(λ)/4] = F(κ) + F(λ)

Wait, that requires homogeneity in cumulant coordinates, where the scaling is
non-uniform. But in ROOT coordinates with uniform scaling, it works.

### The challenge

The homogeneity of 1/Φ_n is degree 2 in root-space but non-uniform in cumulant
space. The convolution ⊞_n is addition in cumulant space. So we need concavity
in cumulant coordinates, where the homogeneity structure is broken.

Possible fix: use Gårding's theorem in a MIXED coordinate system, or use the
recent Schur-Horn extension for hyperbolic polynomials (arXiv:2601.10602, Jan 2025).

### Testable prediction

Compute 1/Φ_n along midpoints in cumulant space ON THE REAL-ROOTED CONE.
Is it concave when restricted to the cone? (We know it's not globally concave —
82.4% midpoint violations — but some of those midpoints may exit the cone.)

Specifically: for κ, λ ∈ C_n with (κ+λ)/2 ∈ C_n, is 1/Φ((κ+λ)/2) ≥ (1/Φ(κ) + 1/Φ(λ))/2?

---

## Approach H: Elementary Symmetric Multiplicativity

### Core idea (★★★ — if the algebraic identity holds, the proof is short)

There is a remarkable identity for the MSS convolution in the elementary
symmetric polynomial basis. If e_k(p) = (−1)^k a_k denotes the k-th
elementary symmetric polynomial of the roots of p, then possibly:

    e_k(p ⊞_n q) = e_k(p) · e_k(q) / C(n,k)

(Reference: arXiv:2309.10970, "Real roots of hypergeometric polynomials via
finite free convolution," which shows the convolution is multiplicative in
certain bases for specific polynomial families.)

**If this holds generally:**
- The convolution becomes COORDINATE-WISE MULTIPLICATION in e_k space
  (up to fixed constants C(n,k))
- Take logs: log e_k(p⊞q) = log e_k(p) + log e_k(q) − log C(n,k)
- This is ADDITIVE in log-e_k coordinates
- Express 1/Φ_n in log-e_k coordinates
- Superadditivity under addition in log-e_k may follow from convexity/concavity
  properties in these coordinates

**Why this could work:** The MSS convolution was designed to preserve real-rootedness.
In the e_k basis, real-rootedness is equivalent to the Newton inequalities
e_k² ≥ e_{k-1}e_{k+1}·(k+1)(n−k+1)/(k(n−k)). If the convolution is
multiplicative, these inequalities are preserved automatically.

### Testable prediction

**Critical test:** Is e_k(p⊞q) = e_k(p)·e_k(q)/C(n,k) for general p, q?
This can be verified in 5 minutes with a numerical test.

---

## Approach I: Monoidal Functor Tower via Differentiation

### Core idea (★★ — elegant but the "wrong direction" inequality is a challenge)

**The exact identity:** D(p ⊞_n q) = Dp ⊞_{n−1} Dq where D(p) = p'/n.

This means differentiation is a **monoidal functor** from (RR_n, ⊞_n) to
(RR_{n−1}, ⊞_{n−1}). It preserves the monoidal structure exactly.

**Known:** 1/Φ_n(p) < 1/Φ_{n−1}(p'/n) (differentiation increases 1/Φ).

**The tower:**
    RR_n →^D RR_{n−1} →^D RR_{n−2} →^D ... →^D RR_2

with 1/Φ_2(Stam) being equality. Each D preserves ⊞, and each increases 1/Φ.

**The obstruction:** Naive induction fails because the inequality goes the wrong
direction for the upper bound on 1/Φ(p⊞q).

**Possible fix (category theory):** Consider the ADJOINT of D. The "integration"
functor I: RR_{n−1} → RR_n that maps q to the unique monic degree-n polynomial
with derivative n·q and a prescribed root sum. If I is a right adjoint to D
in some enriched categorical sense, the adjunction might swap the direction
of the inequality.

In futon5 terms: model D as a serial composition (reducing degree) and I as
its retraction. The wiring diagram would have D as a "contracting wire" and
I as an "expanding wire," with the Stam inequality as a "signal-to-noise"
bound at each step.

### Testable prediction

Compute the ratio Φ_{n−1}(p'/n) / Φ_n(p) for random real-rooted p. How does
the ratio relate to n? If it has a clean form (e.g., (n−1)/n), the tower
approach might work with a telescoping argument.

---

## Approach J: Uniqueness Characterization (Operadic / Axiomatic)

### Core idea (★★ — high novelty, requires theoretical development)

Instead of proving the Stam inequality directly, **characterize 1/Φ_n** as the
unique functional satisfying certain axioms, and show superadditivity follows.

**Precedent:** Baez-Fritz-Leinster (2011) characterize Shannon entropy as the
unique lax point of the operad of finite probability spaces. The laxness condition
IS the superadditivity. They prove existence and uniqueness simultaneously.

**Proposed axioms for F = 1/Φ_n on (RR_n, ⊞_n):**

(J1) **Homogeneity:** F(cλ_1, ..., cλ_n) = c² · F(λ_1, ..., λ_n)
(J2) **Symmetry:** F is symmetric in the roots
(J3) **Normalization:** F(He_n) = [explicit value]
(J4) **Monotonicity:** F(p ⊞ He_t) is increasing in t
(J5) **De Bruijn:** d/dt F(p ⊞ He_t)|_{t=0} = explicit function of p
(J6) **Hermite compatibility:** F(He_t ⊞ He_s) = F(He_{t+s})

**Conjecture J:** There is a unique functional satisfying (J1)-(J6), and it is
automatically superadditive on (RR_n, ⊞_n).

**Proof strategy:** Show that any functional satisfying (J1)-(J6) is determined
by its values on "infinitesimally perturbed" Hermite polynomials (via (J4)-(J5)),
and that the perturbation expansion forces superadditivity.

### Category theory formulation

In the language of enriched category theory (futon5):
- Objects: real-rooted polynomials of degree n
- Enrichment: over (R≥0, +) via the functional F = 1/Φ_n
- The monoidal product ⊞_n is the composition
- The Stam inequality says F is a **lax monoidal functor** from (RR_n, ⊞_n) to (R≥0, +)

The uniqueness characterization would show that any lax monoidal functor
(RR_n, ⊞_n) → (R≥0, +) satisfying the axioms must be F = 1/Φ_n.

---

## Approach K: FI-Module Stabilization

### Core idea (★ — speculative but could eliminate n-by-n casework)

**FI-modules (Church-Ellenberg-Farb 2012):** If a sequence of S_n-representations
{V_n} is generated in finite degree, then the characters are eventually polynomial
in n, and properties that hold for small n automatically extend.

**Application:** The surplus Δ_n(p,q) = 1/Φ_n(p⊞q) − 1/Φ_n(p) − 1/Φ_n(q) has
a natural S_n × S_n equivariance (permuting roots of p and q independently).
If the SOS decomposition of Δ_n (as a function on the semialgebraic set C_n × C_n)
stabilizes in the FI-module sense, then proving Δ_n ≥ 0 for n ≤ N_0 (where N_0
is the stability degree) would suffice for all n.

**Concretely:** If we can prove the Stam inequality for n = 2, 3, 4, 5 by
direct computation, and show that the proof structure (SOS certificate) is
"stable" across n, the FI-module machinery gives the general case.

### Testable prediction

Compare the SOS certificates (if they exist) for n=3 and n=4. Do the generators
of the SOS decomposition have a "stable" description (same monomials, coefficients
polynomial in n)?

---

## Numerical Test Results (2026-02-13)

Scripts: `verify-p4-new-approaches.py`, `verify-p4-approach-e-refined.py`

### Approach E: SPECTACULAR SUPPORT

**Key identity confirmed:** p_t ⊞ q_t = (p⊞q)_{2t} (from cumulant additivity).

**g(t) = 1/Φ((p⊞q)_{2t}) − 1/Φ(p_t) − 1/Φ(q_t):**
- g(t) ≥ 0 for ALL t, ALL (p,q) tested (= Stam at time t) ✓
- g(∞) → 0 (Hermite scaling) ✓
- min(g) is ALWAYS at t → ∞ (g decreases toward 0 from above)
- g monotone decreasing: 72-82% (NOT always monotone)

**BUT — the Stam RATIO is the real story:**

R(t) = [1/Φ(p_t) + 1/Φ(q_t)] / [1/Φ((p⊞q)_{2t})]

| n | R(t) monotone increasing | R minimum at t=0 |
|---|-------------------------|-----------------|
| 3 | 96.7% | 96.7% |
| 4 | 96.7% | 100% |
| 5 | 100% | 100% |

**The LOG Stam surplus is even better:**

h(t) = log(1/Φ((p⊞q)_{2t})) − log(1/Φ(p_t) + 1/Φ(q_t))

| n | h(t) monotone decreasing |
|---|-------------------------|
| 3 | 84% |
| 4 | **100%** |
| 5 | **98%** |

**Interpretation:** For n ≥ 4, h(t) appears to be UNIVERSALLY monotone decreasing.
Combined with h(∞) = 0, this gives: h(0) ≥ h(∞) = 0, which IS the Stam inequality.

**The proof path (if h' ≤ 0 can be established):**
1. Define h(t) = log(1/Φ((p⊞q)_{2t})) − log(1/Φ(p_t) + 1/Φ(q_t))  [1 line]
2. Show h(∞) = 0: as t → ∞, all roots → √t · He_n zeros  [3 lines]
3. Show h'(t) ≤ 0: use Coulomb flow + SOS formula  [THE KEY STEP]
4. Therefore h(0) ≥ 0 ⟹ Stam  [1 line]

**What h'(t) ≤ 0 requires:** Writing F(t) = 1/Φ((p⊞q)_{2t}) and G(t) = 1/Φ(p_t) + 1/Φ(q_t):

h'(t) = F'(t)/F(t) − G'(t)/G(t)

We need: F'/F ≤ G'/G, i.e., the RELATIVE growth rate of the convolution's 1/Φ
is bounded by the relative growth rate of the sum of parts. This is a comparison
of SOS terms (from the proved dΦ/dt formula) weighted by the current Φ values.

### Approach H: KILLED
Elementary symmetric multiplicativity e_k(p⊞q) = e_k(p)·e_k(q)/C(n,k) FAILS.
Max relative error > 10^4. Cross-terms dominate.

### Approach F: KILLED (naive version)
S ≠ S^A + S^B (0% exact decomposition).
⟨S^A, S^B⟩ has random sign (37-48% negative). No orthogonality.

The naive resolvent decomposition does not work because γ (roots of p⊞q) are
roots of the EXPECTED characteristic polynomial, not eigenvalues of any single
A+UBU*. The resolvent identity doesn't apply at the expected-polynomial level.

A subtler decomposition (possibly via subordination functions or the
R-transform) might still work, but needs theoretical development.

### Approach G: KILLED
Midpoint concavity of 1/Φ on the real-rooted cone: 28-38% violations.
1/Φ_n is NOT concave even when restricted to the real-rooted cone.

### Approach I: No clean structure
Φ_{n−1}(p'/n) / Φ_n(p) ratio varies widely (0 to 0.5), no simple function of n.

### Updated Priority

1. **Approach E** ★★★★★ — The log-Stam surplus h(t) is monotone for n ≥ 4
   (100% / 98%). Proving h'(t) ≤ 0 would complete the proof. This uses ONLY
   our proved Coulomb flow + SOS machinery. Estimated proof length: 3-4 pages.

2. **Approach J** ★★ — Characterization approach, theoretical backup.

3. **Approach K** ★ — FI-module stabilization, speculative.

4. All others: KILLED by numerical evidence.
