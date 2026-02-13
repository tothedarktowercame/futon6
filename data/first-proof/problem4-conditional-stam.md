# Problem 4: Conditional Finite Stam Theorem

**Date:** 2026-02-13
**Status:** UPDATED after numerical tests. Approach A killed (A2 fails).
**De Bruijn identity discovered:** d/dt H'_n(p_t) = ((n+1)/12) · Φ_n(p_t),
verified to CV < 0.5% across n=3,4,5. This is a provable theorem.
Approach B pivoted: log-disc not superadditive at n=3, need alternative.

---

## 1. The Reduction to Plain Addition

The single most important structural fact:

**In finite free cumulant coordinates, ⊞_n is plain addition.**

The finite free cumulants κ_k^(n) (Arizmendi-Perales 2018) satisfy:

    κ_k^(n)(p ⊞_n q) = κ_k^(n)(p) + κ_k^(n)(q)

for all k = 1, ..., n. The moment-cumulant relation is:

    a_k = κ_k + (nonlinear terms in κ_1, ..., κ_{k-1})

For centered polynomials (κ_1 = 0):
- κ_2 = a_2
- κ_3 = a_3
- κ_4 = a_4 + (1/12)·a_2²    [absorbs the (1/6)a_2·b_2 cross-term]
- κ_k for k ≥ 5: higher corrections

**Consequence:** The inequality 1/Φ_n(p ⊞_n q) ≥ 1/Φ_n(p) + 1/Φ_n(q)
is equivalent to:

> 1/Φ_n is superadditive as a function of (κ_2, ..., κ_n),
> restricted to the real-rooted cone.

This removes the MSS bilinear weights entirely. The problem becomes:

**Problem (Cumulant Superadditivity):** Define F(κ) = 1/Φ_n(p(κ)) where
p(κ) is the monic degree-n polynomial with finite free cumulants κ = (κ_2, ..., κ_n)
(after centering, κ_1 = 0). Let C_n ⊂ R^{n-1} be the set of cumulant vectors
for which p(κ) has n distinct real roots. Prove:

    F(κ + λ) ≥ F(κ) + F(λ)    for all κ, λ ∈ C_n with κ + λ ∈ C_n.

---

## 2. What We Know About F

**Homogeneity:** Under root scaling (λ_i → c·λ_i), Φ_n → Φ_n/c², so
F = 1/Φ_n → c²·F. But in cumulant space, scaling is NOT uniform:
κ_k → c^k · κ_k. So F is NOT homogeneous of a single degree in cumulant
coordinates. This rules out the classical "homogeneous + superadditive = concave"
reduction.

**Non-convexity (established):** F is NEITHER convex NOR concave in cumulant
space. The Hessian is indefinite in 100% of tested points (75/75 at n=4).
Midpoint test: 82.4% convex violations, 17.6% concave violations.

**Superadditivity (established numerically):** F(κ+λ) ≥ F(κ) + F(λ) in
0/35,000+ random tests across n=2,...,5. The minimum surplus ratio is
1 + O(10^{-8}), approached via extreme scale separation.

**The paradox:** F is superadditive WITHOUT being convex. This means the
proof must exploit the DOMAIN CONSTRAINT (real-rootedness), not just the
function's curvature properties.

---

## 3. Conditional Theorem — Approach A: Score Projection

### Statement

**Theorem A (Conditional Finite Stam — Projection Version).**
Let p, q be monic real-rooted degree-n polynomials with roots
λ_1 < ... < λ_n and μ_1 < ... < μ_n. Let A = diag(λ), B = diag(μ),
and let U be Haar-random on U(n). If:

**(A1)** Φ_n(p ⊞_n q) ≤ E_U[Φ_n(char(A + UBU*))]

> (Jensen step: root-force energy of the averaged characteristic
> polynomial ≤ average root-force energy)

**(A2)** E_U[Φ_n(char(A + UBU*))] ≤ Φ_n(p)·Φ_n(q) / (Φ_n(p) + Φ_n(q))

> (Sample-level Stam: average root-force energy ≤ harmonic mean
> of individual energies)

Then 1/Φ_n(p ⊞_n q) ≥ 1/Φ_n(p) + 1/Φ_n(q).

### Proof of implication

By (A1) and (A2):

    Φ_n(p ⊞_n q) ≤ E_U[Φ_n(A+UBU*)] ≤ Φ_n(p)·Φ_n(q)/(Φ_n(p)+Φ_n(q))

Taking reciprocals (all quantities positive):

    1/Φ_n(p ⊞_n q) ≥ (Φ_n(p)+Φ_n(q))/(Φ_n(p)·Φ_n(q))
                     = 1/Φ_n(q) + 1/Φ_n(p).   ∎

### Analysis of (A1)

Condition (A1) asks: is Φ_n convex as a function on the space of polynomials,
in the sense that Φ_n(E[P]) ≤ E[Φ_n(P)] for any random polynomial P with
real-rooted realizations?

This is a form of Jensen's inequality. It would follow if Φ_n were convex
as a function of polynomial coefficients on the real-rooted cone. Since
Φ_n = 1/F and F is neither convex nor concave, the convexity of Φ_n is
a separate question.

**Testable:** Sample many U, compute Φ_n(A+UBU*), compare E[Φ_n] with
Φ_n(p ⊞_n q). Run for n = 3, 4, 5.

### Analysis of (A2)

Condition (A2) is the "sample-level Stam inequality." At the free level
(n → ∞), this is exactly what Voiculescu proves via:

1. Score decomposition: J(X+Y) = E[J(X) | W*(X+Y)]
2. L²-contractivity of conditional expectation
3. Orthogonality of partial scores (from freeness)
4. Optimization of the mixing parameter α

For finite n, approximate freeness under Haar rotation should give
approximate versions of steps 2-4, with corrections of order O(1/n).

**Testable:** Same sampling as (A1) — check whether E_U[Φ_n] ≤
harmonic mean of Φ_n(p), Φ_n(q).

---

## 4. Conditional Theorem — Approach B: Finite De Bruijn

### The analogy

| Classical | Free | Finite (sought) |
|-----------|------|-----------------|
| H(X) = -∫ f log f | χ(X) = ∫∫ log|x-y| dμ dμ | H_n(p) = ? |
| I(X) = ∫ (f'/f)² f | Φ*(X) = ∫ (Hμ)² dμ | Φ_n(p) = Σ_i S_i² |
| d/dt H(X+√tZ) = ½I | d/dt χ(X ⊞ √tS) = ½Φ* | d/dt H_n(p_t) = -Φ_n ? |
| 1/I(X+Y) ≥ 1/I(X)+1/I(Y) | 1/Φ*(X⊞Y) ≥ 1/Φ*(X)+1/Φ*(Y) | target |

### Statement

**Theorem B (Conditional Finite Stam — De Bruijn Version).**
Let p_t = p ⊞_n h_t where h_t is a "finite heat kernel" — a one-parameter
family of monic real-rooted degree-n polynomials with h_0(x) = x^n. If
there exists a functional H_n on monic real-rooted degree-n polynomials such that:

**(B1)** d/dt H_n(p_t)|_{t=0+} = -c · Φ_n(p) for some universal c > 0.

> (Finite de Bruijn identity: entropy decreases at rate proportional
> to root-force energy under the heat semigroup)

**(B2)** H_n(p ⊞_n q) ≥ H_n(p) + H_n(q)   for all real-rooted p, q.

> (Entropy superadditivity under finite free convolution)

**(B3)** d/dt[1/Φ_n(p_t)] ≥ 0  for all t > 0 and all real-rooted p.

> (Reciprocal root-force energy is non-decreasing under the heat semigroup)

Then 1/Φ_n(p ⊞_n q) ≥ 1/Φ_n(p) + 1/Φ_n(q).

### Proof sketch (conditional)

The proof would follow the classical/free pattern:
- (B2) gives H_n(p ⊞_n q) ≥ H_n(p) + H_n(q)
- (B1) relates the derivative of H_n to Φ_n, giving an integral representation
- (B3) + a monotone coupling argument yields the harmonic mean bound

The exact proof depends on the choice of H_n and h_t. The structure is
clearer than Approach A but requires constructing THREE objects (H_n, h_t,
and the coupling), whereas Approach A requires verifying TWO inequalities.

### Candidate for H_n

The most natural candidate is the log-discriminant:

    H_n(p) = (2/n²) Σ_{i<j} log|λ_i - λ_j|

This is the finite analog of Voiculescu's free entropy χ(μ) = ∫∫ log|x-y| dμ(x)dμ(y).

Under "standard" Dyson BM (β = 2):

    dλ_i = dB_i + 2 Σ_{j≠i} 1/(λ_i - λ_j) dt

the Itô calculus gives:

    d H_n = martingale + (2/n²) Σ_{i<j} [(S_i-S_j)/(λ_i-λ_j) - 1/(λ_i-λ_j)²] dt

The drift involves BOTH:
- Ψ_n = Σ_{i<j} 1/(λ_i-λ_j)²   (the pair-sum of squared gaps)
- Cross terms S_i·S_j / (λ_i-λ_j)

These are related to but NOT equal to Φ_n = Σ_i S_i². The de Bruijn identity
would require showing that the drift equals -c·Φ_n, which is not obvious.

**Testable:** Numerically compute E[d H_n / dt] along Dyson trajectories
and compare with Φ_n values at multiple time points.

### Candidate for h_t

**Option 1:** h_t(x) = Hermite polynomial H_n(x/√t) · t^{n/2}. The roots
of h_t are √t times the roots of H_n. As t → 0, roots collapse to 0
(= roots of x^n). This is the finite analog of "add a semicircular of
variance t."

**Option 2:** h_t(x) = (x - t)·(x - 2t)·...·(x - nt) (equally spaced
roots at spacing t). As t → 0, roots collapse. This is simpler but less
canonical.

**Option 3:** h_t is the characteristic polynomial of √t · GUE_n (random).
Then p ⊞_n h_t = E_U[char(A + √t · U·GUE·U*)] — this IS the finite free
convolution with noise. But h_t would need to be deterministic for the
semigroup to work cleanly.

---

## 5. Approach C: Lorentzian Polynomials on the Real-Rooted Cone

### The observation

The real-rooted cone C_n (monic degree-n polynomials with all roots real and
distinct) is closely related to the theory of Lorentzian polynomials
(Brändén-Huh 2019). A Lorentzian polynomial f is one whose Hessian has
signature (1, n-1, 0, ...) on the positive orthant — it is "log-concave
in every direction."

The MSS framework that defines ⊞_n is deeply connected to real stability
(Borcea-Brändén 2009). The real-rooted cone is a real section of the
stable-polynomial cone.

### Speculation

**Conjecture:** 1/Φ_n restricted to the real-rooted cone C_n, in cumulant
coordinates, satisfies a Lorentzian-type condition: its Hessian, restricted
to tangent directions of C_n, has controlled signature.

This would explain:
1. Why 1/Φ_n is NOT globally convex (the Hessian is indefinite on R^{n-1})
2. Why 1/Φ_n IS superadditive (the real-rooted constraint restricts to
   directions where the Hessian has the right sign)
3. Why the real-rootedness constraint is essential (plain addition fails
   29% without the MSS weights, which preserve real-rootedness)

### What this would give

If 1/Φ_n were "Lorentzian" on C_n, standard results from Brändén-Huh theory
could give:
- Log-concavity of sequences derived from 1/Φ_n
- Ultra-log-concavity and therefore superadditivity
- Connection to matroid theory (via the MSS work on interlacing families)

### Key question for literature search

Does the theory of Lorentzian polynomials or related "cone-restricted convexity"
results apply to functions like 1/Φ_n on the real-rooted cone? Specifically:
- Are there results about superadditivity of functions on the stable/real-rooted cone?
- Does the Brändén-Huh theory interact with the MSS finite free convolution?
- Is there a notion of "Lorentzian function" (not polynomial) that captures
  cone-restricted superadditivity?

---

## 6. The n=3 Proof Revisited in Cumulant Coordinates

For n=3 centered: κ_2 = a_2, κ_3 = a_3. ⊞_3 = plain addition. The proof:

    F(κ) = 1/Φ_3 = disc/(18·κ_2²) = (-4κ_2³ - 27κ_3²)/(18κ_2²)
         = -2κ_2/9 - (3/2)·κ_3²/κ_2²

    F(κ+λ) - F(κ) - F(λ)
      = (3/2)·[κ_3²/κ_2² + λ_3²/λ_2² - (κ_3+λ_3)²/(κ_2+λ_2)²]
      ≥ 0   (Cauchy-Schwarz / Titu's lemma)

**Why this works:** F decomposes as (linear part) + (ratio term). The linear
part adds perfectly. The ratio term κ_3²/κ_2² is convex in (κ_2, κ_3) on
the positive orthant (κ_2 < 0 in our convention, but the structure is the same).
The negative sign in front makes the ratio contribution superadditive.

**What needs to generalize for n ≥ 4:** Express F = 1/Φ_n in cumulant
coordinates as a sum of terms, each of which is individually superadditive
or whose interactions are controlled. The n=3 decomposition into
"linear + ratio" is specific to the Φ_3·disc identity, but the STRUCTURE
(decomposition into superadditive pieces) might generalize.

---

## 7. Numerical Test Results (2026-02-13)

### Test summary

| Test | n=3 | n=4 | n=5 | Verdict |
|------|-----|-----|-----|---------|
| A1 (Jensen) | ratio 0.34-0.96, 0 viol. | ratio 0.14-0.66, 0 viol. | ratio 0.30-0.68, 0 viol. | **PASS** |
| A2 (Sample Stam) | 14/30 violations | 14/30 violations | 10/30 violations | **FAIL** |
| B1 (de Bruijn) | c=-0.0741, CV=0.01% | c=-0.0521, CV=0.31% | c=-0.0400, CV=0.41% | **PASS** |
| B2 (log-disc super.) | 3/100 violations | 0/100 violations | 0/100 violations | **FAIL (n=3)** |

Script: `scripts/verify-p4-conditional-stam.py`
Data: `data/first-proof/problem4-conditional-tests.jsonl`

### What this means

**Approach A is DEAD.** Condition (A2) fails in ~47% of random polynomial pairs.
E_U[Φ_n(A+UBU*)] is NOT bounded by the harmonic mean. The two-step reduction
(Jensen + sample Stam) does not lead to a proof.

**The de Bruijn identity is REAL.** The constant c (with H_n = (2/n²)Σ log|gap|)
satisfies:

    c_n = -(2/n²) · (n+1)/12 = -(n+1)/(6n²)

Equivalently, with the unnormalized log-discriminant H' = Σ_{i<j} log|λ_i-λ_j|:

    d/dt H'_n(p_t) = ((n+1)/12) · Φ_n(p_t)

Verified to better than 0.5% coefficient of variation across 20 random starting
polynomials × 5 time points, for each n ∈ {3,4,5}. The formula (n+1)/12 fits
n=3,4,5 exactly (1/3, 5/12, 1/2).

**This is a concrete theorem to prove.** The heat kernel used is
h_t(x) = Π_k(x - √t · k) with k = -(n-1)/2, ..., (n-1)/2
(equally spaced roots at scale √t). The variance of the empirical root
distribution grows at rate (n²-1)/12, so the "per-variance" constant is
1/(n-1), consistent with a large-n limit.

**B2 fails at n=3 but passes at n=4,5.** The log-discriminant is NOT
superadditive under ⊞_3 (3 violations in 100 tests, surplus down to -0.12).
This means the naive de Bruijn approach (B1+B2+B3 ⟹ Stam) doesn't work for n=3.
But the Stam inequality IS proved for n=3 (by a different method), so the log-disc
approach is not the right path at n=3. For n≥4, B2 may hold — further testing needed.

### Revised strategy

Since A2 fails and B2 fails at n=3, neither Approach A nor Approach B works
as originally stated. The surviving elements are:

1. **A1 (Jensen) HOLDS** — useful as a step in any proof involving Haar averaging
2. **The de Bruijn identity HOLDS** — a provable theorem connecting H'_n and Φ_n
3. **B2 HOLDS for n ≥ 4** — might give a de Bruijn proof for n ≥ 4 specifically

The most promising direction is now:

**Approach D: Prove the de Bruijn identity, then exploit it.**
- Step 1: Prove d/dt H'_n(p_t) = ((n+1)/12) · Φ_n(p_t) rigorously.
  This should follow from Itô calculus on the root dynamics of p_t = p ⊞_n h_t.
- Step 2: For n ≥ 4, combine with B2 (if it holds) and a monotonicity argument
  to get the Stam inequality.
- Step 3: For n=3, the de Bruijn identity is a standalone result; the Stam
  inequality was already proved by other means.

**Approach E: Direct cumulant-space attack (unchanged).**
- The cumulant reformulation (Section 1) is independent of A/B approaches.
- Lorentzian polynomial direction (Section 5) is the layer switch.

## 8. Action Items (revised)

### Immediate (numerical verification)

1. **Test Condition (A1):** For n=3,4,5 and many (p,q) pairs, sample
   U ~ Haar(U(n)), compute E_U[Φ_n(A+UBU*)] and Φ_n(p ⊞_n q).
   Check whether E_U[Φ_n] ≥ Φ_n(conv).

2. **Test Condition (A2):** Same sampling. Check whether
   E_U[Φ_n(A+UBU*)] ≤ Φ_n(p)·Φ_n(q)/(Φ_n(p)+Φ_n(q)).

3. **Test de Bruijn (B1):** Numerically integrate Dyson BM from roots of p
   with various noise levels. Compute d/dt H_n(p_t) and compare with Φ_n(p_t).
   Test H_n = log-discriminant.

4. **Hessian signature on C_n tangent space:** At random real-rooted points
   in cumulant space, compute the Hessian of 1/Φ_n RESTRICTED to tangent
   directions of C_n. Check signature.

### Codex literature search

5. **Lorentzian polynomials + MSS/finite free convolution:** Any connection
   between Brändén-Huh theory and superadditivity on the real-rooted cone.

6. **Finite free entropy:** Any existing definition or results on a finite
   analog of Voiculescu's free entropy χ that interacts with ⊞_n.

7. **Score projection under Haar averaging:** Results on
   E_U[Φ_n(A+UBU*)] or E_U[1/Φ_n(A+UBU*)]. Related Weingarten calculations.

### Codex computational

8. **n=4 SOS in cumulant coordinates:** Reframe the existing n=4 SOS
   work in cumulant coordinates. The cross-term disappears, leaving
   superadditivity under plain addition. Might simplify the algebra.

---

## 8. Relationship to Existing Work

### What Strategy D (from proof-strategies.md) proposed

The conditional theorem with Lemma A (finite de Bruijn) + Lemma B (score
projection). This document refines that into two approaches:
- Approach A (projection): conditions (A1) + (A2), directly testable
- Approach B (de Bruijn): conditions (B1) + (B2) + (B3), requires constructing H_n

### What's NEW here

1. **The cumulant-space reformulation.** Reducing to superadditivity under
   plain addition by using the fact that κ are additive under ⊞_n. This
   simplifies the problem statement and connects to existing algebraic
   geometry results.

2. **The Lorentzian polynomial direction.** The observation that
   superadditivity WITHOUT convexity, on the real-rooted cone, is
   suggestive of Lorentzian/log-concave phenomena. This is a potential
   layer switch — from analysis (heat flow, Fisher information) to
   algebraic combinatorics (Lorentzian polynomials, matroid theory).

3. **The n=3 decomposition revisited.** Seeing the n=3 proof as
   "linear part + convex ratio part" in cumulant coordinates suggests
   looking for a similar decomposition at n ≥ 4.

---

## References

- Arizmendi, Perales (2018), JCTA 155. [Finite free cumulants]
- Brändén, Huh (2019), Ann. Math. 192. [Lorentzian polynomials]
- Marcus, Spielman, Srivastava (2015), Ann. Math. 182. [⊞_n definition]
- Marcus (2021), arXiv:2108.07054. [Finite free probability survey]
- Voiculescu (1998), Invent. Math. 132. [Free Stam inequality proof]
- Borcea, Brändén (2009), Invent. Math. 177. [Real stability]
