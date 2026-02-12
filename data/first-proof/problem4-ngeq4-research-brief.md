# Library Research Brief: Problem 4 — Proof for n >= 4

**Purpose:** Targeted research to find a proof of the superadditivity
inequality 1/Phi_n(p ⊞_n q) >= 1/Phi_n(p) + 1/Phi_n(q) for n >= 4.
The cases n=2 (equality) and n=3 (Cauchy-Schwarz) are PROVED. The n >= 4
case is numerically verified but structurally different.

**Priority:** HIGH — this is the remaining open piece of Problem 4.

**Predecessor:** `problem4-research-brief.md` (general strategy search),
`problem4-research-findings.md` (MO mining results),
`problem4-proof-strategy-skeleton.md` (route analysis).

---

## What is proved

**n = 2:** Equality. 1/Phi_2 = disc(p)/2 = (a_1^2 - 4a_2)/2 is linear
in the coefficients, and ⊞_2 preserves this linear combination exactly.
Surplus = 0 always.

**n = 3:** Strict inequality via three steps:
1. **Centering reduction:** WLOG a_1 = b_1 = 0 (Phi_n is translation-
   invariant, ⊞_n commutes with translation via A + αI in the random
   matrix model).
2. **Key identity:** Phi_3 · disc(p) = 18 · a_2^2 for centered cubics.
   Therefore 1/Phi_3 = -2a_2/9 - 3a_3^2/(2a_2^2).
3. **⊞_3 simplification:** For centered cubics, all cross-terms vanish:
   c_2 = a_2 + b_2, c_3 = a_3 + b_3 (plain coefficient addition).
4. **Titu's lemma:** surplus = (3/2)[u²/s² + v²/t² - (u+v)²/(s+t)²] >= 0.

Verification: `scripts/verify-p4-n3-proof.py` (symbolic SymPy proof).

## Why n >= 4 is harder — three structural obstacles

**Obstacle 1:** The identity Phi_n · disc = const · a_2^2 FAILS for n >= 4.
At n=4, Phi_4 · disc depends on a_3 and a_4. There is no clean two-variable
formula for 1/Phi_4.

**Obstacle 2:** ⊞_n has cross-terms even for centered polynomials when n >= 4.
At n=4: c_4 = a_4 + (1/6)·a_2·b_2 + b_4. The cross-term (1/6)·a_2·b_2
does not vanish when a_1 = b_1 = 0.

**Obstacle 3:** The cross-term is ESSENTIAL. Plain coefficient addition
(without the (1/6)·a_2·b_2 term) fails superadditivity ~29% of the time
for centered quartics. The specific MSS bilinear weights are necessary.

## Computational evidence for n >= 4

**Stress test** (`scripts/verify-p4-stress-test.py`): Five strategies tested
— random sampling, near-degenerate roots, extreme coefficient ratios,
Nelder-Mead optimization, differential evolution. Results across 35K+ tests:

| Strategy | n=4 | n=5 | n=6 | n=7 | n=8 |
|----------|-----|-----|-----|-----|-----|
| Random (min ratio) | 1.002 | 1.002 | 1.020 | 1.176 | 1.167 |
| Extreme coeffs | 1+2.5e-8 | 1+1.7e-8 | 1+9.3e-8 | — | — |
| Adversarial opt | 1+O(eps) | 1+O(eps) | 1+O(eps) | — | — |
| Global opt | 1+2e-7 | 1+1e-5 | — | — | — |

0 genuine violations. The adversarial infimum appears to be exactly 1.0
for all n >= 3 (approached via extreme scale separation between p and q,
but never reached). This means a "prove for n > N" strategy based on
growing surplus does NOT work — the infimum is 1 at every degree.

---

## Research Questions (search these)

### Q4.7: Finite-n analog of free entropy

Voiculescu's free entropy χ(μ) = ∫∫ log|x-y| dμ(x)dμ(y) satisfies
χ(μ ⊞ ν) >= χ(μ) + χ(ν) (superadditivity). Our inequality has the
exact same form. Is there a finite-n entropy functional that:
(a) equals or approximates 1/Phi_n for finite root configurations, and
(b) inherits superadditivity from the free case?

Search terms:
- "finite free entropy" OR "discrete free entropy"
- "finite free" "logarithmic energy" superadditive
- "free entropy" "characteristic polynomial" OR "finite dimensional"
- Voiculescu entropy "finite" convolution OR "finite free"
- "microstate free entropy" finite dimensional approximation

What we're looking for: Any finite-dimensional version of χ that is
superadditive under ⊞_n. Even if it's not exactly 1/Phi_n, showing the
relationship would suggest the proof technique.

### Q4.8: Free Fisher information and Phi_n

The free Fisher information Φ*(μ) = ∫ (Hμ(x))² μ(dx) where Hμ is the
Hilbert transform of μ. For n roots at positions λ_1,...,λ_n, the
discrete analog is exactly Phi_n = Σ_i (Σ_{j≠i} 1/(λ_i - λ_j))².
So Phi_n IS the finite-dimensional free Fisher information!

The known result in free probability: 1/Φ*(μ) is CONCAVE along free
Brownian motion (not convex!). But the superadditivity
1/Φ*(μ ⊞ ν) >= 1/Φ*(μ) + 1/Φ*(ν) is a DIFFERENT statement from
concavity. Does superadditivity of 1/Φ* hold in the free case?

Search terms:
- "free Fisher information" "reciprocal" superadditive OR convolution
- "free Fisher information" "free convolution" inequality
- 1/Phi "free Fisher" superadditive
- "Cramer-Rao" free probability OR "free Fisher"
- "free information inequality" Voiculescu OR Nica OR Speicher

What we're looking for: Whether 1/Φ*(μ ⊞ ν) >= 1/Φ*(μ) + 1/Φ*(ν)
is a known result, conjecture, or false in free probability. If TRUE
in the limit, the finite-n version might follow from quantitative
convergence estimates.

### Q4.9: MSS bilinear weights and superadditivity

The MSS coefficient formula c_k = Σ_{i+j=k} w(n,i,j)·a_i·b_j has
weights w(n,i,j) = (n-i)!(n-j)!/(n!(n-k)!). These are falling-factorial
binomial coefficients. At n=3 centered, the weights conspire to make
⊞_3 = plain addition (all cross-terms vanish). At n=4, the cross-term
w(4,2,2) = 1/6 is essential for superadditivity.

Is there a direct algebraic proof via the weight structure? Specifically:
can we express 1/Phi_n(conv) - 1/Phi_n(p) - 1/Phi_n(q) as a sum of
squares or as a manifestly non-negative expression involving the
MSS weights?

Search terms:
- "falling factorial" "bilinear" convolution superadditive OR "sum of squares"
- Marcus Spielman Srivastava "coefficient" weights OR formula
- "finite free" convolution "bilinear" algebraic identity
- "characteristic polynomial" "expected" "bilinear" "positive"
- "mixed characteristic polynomial" coefficient structure

What we're looking for: Any algebraic identity that makes the surplus
manifestly non-negative. The falling-factorial weights have rich
combinatorial structure (connections to ballot problems, lattice paths,
symmetric groups) that might yield such an identity.

### Q4.10: Weingarten calculus approach

Since p ⊞_n q = E_Q[det(xI - A - QBQ*)] with Q Haar unitary, the
coefficients of the convolution involve integrals over U(n). The
Weingarten calculus provides explicit formulas for such integrals:

    E_Q[Q_{i1j1}...Q_{ikjk} Q*_{i'1j'1}...Q*_{i'kj'k}]
    = Σ_{σ,τ ∈ S_k} δ(i,σ(i')) δ(j,τ(j')) Wg(σ·τ^{-1}, n)

Can this be used to compute E_Q[1/Phi_n(A+QBQ*)] or to bound
1/Phi_n of the expected polynomial directly?

Search terms:
- "Weingarten" eigenvalue functional OR "characteristic polynomial"
- "Haar unitary" "expected" eigenvalue function
- "HCIZ integral" "Coulomb" OR energy OR "root separation"
- "Harish-Chandra Itzykson Zuber" polynomial OR determinant
- "unitary integral" "discriminant" OR "resultant"

What we're looking for: Explicit formulas for E_Q[f(eigs(A+QBQ*))]
where f is a symmetric function of eigenvalues. Even partial results
(e.g., E_Q[Σ (eig_i - eig_j)^{-2}]) would be valuable.

### Q4.11: The tight cases — scale separation

The adversarial optimizer finds that the tightest cases (ratio closest
to 1) occur when one polynomial has roots at scale s >> 1 and the other
at scale 1/s. As s → ∞, the ratio → 1. This is an extreme
scale-separation regime.

In this regime, ⊞_n should behave like "add a large deterministic shift
to small fluctuations." Is there a perturbative analysis of ⊞_n in
the scale-separated limit? What does 1/Phi_n(p ⊞_n q) look like when
spread(p) >> spread(q)?

Search terms:
- "free convolution" "perturbation" OR asymptotic OR "small variance"
- "subordination" "free convolution" perturbative
- "free convolution" "degenerate" OR "degenerate limit"
- Marcus Spielman "asymptotic" finite free OR "large degree"
- "free convolution" "semicircle" perturbation

What we're looking for: Asymptotic formulas for ⊞_n when one input
is highly spread and the other is concentrated. Understanding the
tight cases would reveal why the surplus is always positive and
suggest the right proof structure.

### Q4.12: Induction on degree

Can the n=k case be reduced to the n=k-1 case? For monic polynomials,
there's a natural map from degree n to degree n-1 via differentiation
(p → p'/n, whose roots interlace with p's roots by Rolle's theorem).
Does ⊞_n interact well with differentiation?

In fact, Marcus (2021) shows that differentiation "commutes" with ⊞_n
in a specific sense: (p ⊞_n q)' relates to p' ⊞_{n-1} q' (up to
normalization). If Phi_n can be expressed in terms of Phi_{n-1} of the
derivative, an inductive proof might work.

Search terms:
- "finite free convolution" derivative OR differentiation
- Marcus "polynomial convolution" derivative interlacing
- "free convolution" "Rolle" OR interlacing induction
- ⊞_n OR "boxplus" derivative OR "d/dx"
- "finite free" induction degree

What we're looking for: Any result showing that ⊞_n preserves
structure under differentiation, and that Phi_n relates to Phi_{n-1}
of the derivative. This would enable an inductive proof starting
from the proved n=3 base case.

---

## Key References (updated)

- Marcus, Spielman, Srivastava (2015), "Interlacing Families II,"
  Annals of Math 182(1), 327-350. arXiv:1504.00350.

- Arizmendi, Perales (2018), "Cumulants for finite free convolution,"
  JCTA 155, 244-266. arXiv:1611.06598.

- Marcus (2021), "Polynomial convolutions and (finite) free probability,"
  arXiv:2108.07054. [Survey — check Section 5.2 on majorization and
  any results on differentiation/induction.]

- Marcus, Spielman, Srivastava (2018), arXiv:1811.06382.
  [Further structure, barrier method generalization.]

- Voiculescu (1993), "The analogues of entropy and of Fisher's
  information in free probability theory, I," CMP 155(1), 71-92.
  [Free entropy superadditivity — the infinite-n analog.]

- Nica, Speicher (2006), "Lectures on the Combinatorics of Free
  Probability," Cambridge UP. [Standard reference for free Fisher
  information Φ* and its properties.]

## Critical Connection: Phi_n = discrete free Fisher information

The most important conceptual observation from our work:

    Phi_n(p) = Σ_i (Σ_{j≠i} 1/(λ_i - λ_j))²

is the DISCRETE VERSION of the free Fisher information

    Φ*(μ) = ∫ (Hμ(x))² dμ(x)

where Hμ is the Hilbert transform. In the limit n → ∞ with empirical
measure μ_n → μ, we have Phi_n/n → Φ*(μ) (up to normalization).

The inequality 1/Phi_n(p ⊞_n q) >= 1/Phi_n(p) + 1/Phi_n(q) is then
the FINITE VERSION of 1/Φ*(μ ⊞ ν) >= 1/Φ*(μ) + 1/Φ*(ν).

**Q4.8 is therefore the single highest-priority question:** does
1/Φ* satisfy superadditivity under free convolution? If yes, the
finite-n result should follow from convergence estimates. If no,
then our inequality is a genuinely finite-dimensional phenomenon
and requires a different proof.

## Web Research Findings (2026-02-12)

### Q4.8 RESULT: The free Stam inequality EXISTS and is PROVED

**Key finding:** Voiculescu (1998) proved the **free Stam inequality**:

    1/Φ*(X+Y) >= 1/Φ*(X) + 1/Φ*(Y)

when X, Y are freely independent. Equality holds iff X, Y are semicircular
(the free analog of Gaussian). This is EXACTLY our inequality in the
n → ∞ limit!

References:
- Voiculescu (1993), "Analogues of entropy and Fisher's information, I,"
  CMP 155, 71-92. [Defines Φ*, proves free Cramer-Rao]
- Voiculescu (1998), "Analogues of entropy and Fisher's information, V:
  Noncommutative Hilbert transforms," Invent. Math. 132.
  [Proves the free Stam inequality]

**Implication for P4:** Our inequality 1/Phi_n(p ⊞_n q) >= 1/Phi_n(p) +
1/Phi_n(q) is the FINITE ANALOG of a known theorem. The proof strategy
should be:
1. Understand Voiculescu's proof technique for the free Stam inequality
2. Find the finite-n analog of each step
3. Use quantitative convergence (Phi_n/n → Φ*) to check consistency

### Q4.12 RESULT: Differentiation DOES interact with ⊞_n

**Key finding from Marcus (2021) arXiv:2108.07054:**
- Differentiation preserves real-rootedness under finite free convolution
- The relationship between convolutions of different grades is a key tool
  in analytic proofs, which "induct on the degrees of the polynomials"
- Bounds on roots at degree n can be obtained by "reducing to the equal
  degree case by differentiating sufficiently many times"
- Section 5.2 contains majorization relations on convolutions

**Additional from arXiv:2505.01705v2 (Finite-free Convolution: Infinitesimal
Distributions, 2025):**
- Proposition 5.4 analyzes how infinitesimal distributions change under
  polynomial differentiation
- H-transform formalism connects derivatives to convolution structure
- No energy functional or superadditivity result in this paper

**Implication for P4:** The induction-via-differentiation route (Q4.12)
has concrete structural support. Marcus uses degree reduction via
differentiation as a proof technique for finite free convolution results.
The missing piece: connecting Phi_n to Phi_{n-1} through differentiation.

### Q4.10 RESULT: Strong Weingarten/HCIZ toolbox available

The local MO/MSE mining (see problem4-ngeq4-library-findings.md) found
6 relevant MO threads on Weingarten calculus and HCIZ integrals. Web
search confirms these are well-developed tools. The gap: no one has
applied them specifically to compute E_Q[1/Phi_n(A + QBQ*)] or to
prove superadditivity of 1/Phi_n under the expected-characteristic-
polynomial convolution.

### Updated Priority Ranking (revised after Codex Strategy A results)

1. **Q4.8 (FREE STAM — HIGHEST)**: The free Stam inequality IS our
   inequality in the limit. **Voiculescu's actual proof mechanism** is
   conditional expectation + L²-contractivity on conjugate variables,
   NOT Dyson BM monotonicity (see Codex results below). The finite-n
   proof should finitize the projection/orthogonality mechanism, not
   the heat flow.

2. **Q4.12 (INDUCTION via DIFFERENTIATION)**: Differentiation commutes
   EXACTLY with ⊞_n (proved algebraically + numerically). But naive
   induction is blocked: 1/Phi_n < 1/Phi_{n-1}(p'/n) always (wrong
   direction). May work with a different functional or corrected chain.

3. **Q4.9 (MSS WEIGHTS)**: Algebraic route via MO 287724. Concrete but
   requires finding a nonnegative decomposition.

4. **Q4.10 (WEINGARTEN/HCIZ)**: Strong toolbox. Codex Step 3 confirms
   HCIZ gives principled route to E_U[f(eigs(A+UBU*))], but no Jensen
   direction without convexity. May feed into the projection approach.

5. **Q4.11 (SCALE SEPARATION)**: Diagnostic, not proof-generating.

6. **Q4.7 (FINITE ENTROPY)**: Subsumed by Q4.8 — the free Stam inequality
   is the infinite-n version of what we want.

---

## Search Strategy Notes

1. **MathOverflow strongly preferred.** This is research-level.

2. **Author priority:** Speicher, Nica, Voiculescu (free Fisher info);
   Marcus, Spielman, Srivastava (⊞_n structure); Arizmendi, Perales
   (finite cumulants); Biane, Capitaine (free convolution asymptotics).

3. **The derivative/induction angle (Q4.12) is fresh** — it wasn't in
   the earlier research brief. Prioritize this if you find anything.

4. **Return format**: For each hit, return:
   - Post title, URL/ID, author, date
   - Key claim or technique (1-2 sentences)
   - How it connects to the n>=4 gap (direct / analogous / tangential)
   - Confidence that it helps (high / medium / low)

5. **Read Marcus (2021) arXiv:2108.07054 carefully** — it's a survey
   by one of the creators of ⊞_n and may contain exactly the structural
   results we need (differentiation, energy functionals, open problems).

---

## Codex Strategy A Results (2026-02-12, commit cd8396f)

Six-step research pipeline via `scripts/run-research-codex-p4-stam.py`.
Full results in `data/first-proof/problem4-stam-codex-results.jsonl`.

### Step 1 — Voiculescu's proof mechanism (plausible)

**CRITICAL FINDING: Voiculescu does NOT use Dyson/heat-flow monotonicity.**

The free Stam inequality is proved via **conjugate variables and L²
projection**, not free Brownian motion:

1. Define conjugate variable J(X) with Φ*(X) = ‖J(X)‖²
2. Conditional expectation: J(X+Y) = E_{W*(X+Y)}[αJ(X:Y) + (1-α)J(Y:X)]
3. L²-contractivity: Φ*(X+Y) ≤ ‖αJ(X:Y) + (1-α)J(Y:X)‖²
4. Freeness: J(X:Y) = J(X), J(Y:X) = J(Y), and mixed inner product = 0
5. Expand: Φ*(X+Y) ≤ α²Φ*(X) + (1-α)²Φ*(Y)
6. Optimize α = Φ*(Y)/(Φ*(X)+Φ*(Y)) → harmonic mean → Stam

**Implication:** The finite-n proof should look for:
- A finite conjugate variable (root-force field S_i is the candidate)
- A finite conditional-expectation / projection surrogate for ⊞_n
- Finite orthogonality or controlled cross-term bound

This completely reframes Strategy A away from Dyson monotonicity.

### Step 2 — Dyson Itô calculus for Φ_n (established)

Explicit formulas derived. Key identities:
- Φ_n = 2 Σ_{i<j} (λ_i - λ_j)^{-2} (triple cross-terms cancel)
- ∂_k Φ_n = -4U_k, ∂_{kk} Φ_n = 12V_k
- Full drift formula for L(1/Φ_n) under Dyson generator

**Result: L(1/Φ_n) is NOT sign-definite.**
Counterexample: n=3, λ=(-1,0,1) gives L(1/Φ_n) = -1/27 < 0.
Confirms our numerical finding that E[1/Φ_n] is not monotone along Dyson BM.

### Step 3 — Connection to ⊞_n (plausible)

- A+UBU* is NOT a stopped Dyson process (hard obstruction: tr((X-A)^k)
  is deterministic for orbital sum, random for Dyson)
- HCIZ gives principled route to E_U[f(eigs(A+UBU*))] but is
  computationally difficult for general n
- No finite-n subordination in full generality; Marcus's finite free
  cumulants provide algebraic (not analytic) linearization
- Jensen direction between E_U[1/Φ_n] and 1/Φ_n(p ⊞_n q) unknown
  without convexity/concavity

### Step 4 — Jensen gap analysis (TIMED OUT at 180s)

Parse error. This was the most computationally demanding step. The Jensen
gap question is now somewhat secondary since the Dyson route itself is
not the right mechanism.

### Step 5 — Finite Stam proof attempt (gap)

Attempted Dyson/Itô and convexity routes; both fail:
- Dyson route: no closed SDE/PDE for roots of the averaged polynomial
  (Φ_n acts on roots of E_U[det(...)], not on sample eigenvalues)
- Convexity route: 1/Φ_n has no global concavity certificate for n≥4

**Exact obstruction identified:** The missing ingredient is the
finite-dimensional analog of {free score, subordination, de Bruijn}.
These three pieces work together in Voiculescu's framework and have no
known finite-n counterparts.

### Step 6 — Synthesis (gap, but informative)

**Proof viability:** 25-40% unconditional; ~70% conditional theorem;
>90% for verified small-n cases.

**Four precise blockers:**

| Blocker | What's needed | Likely true? |
|---------|--------------|-------------|
| A: Finite de Bruijn | H_n with d/dt H_n(p_t) = -Φ_n(p_t) along finite heat flow | medium-high |
| B: Finite score/projection | Inequality playing role of free-score projection for ⊞_n | medium |
| C: Monotonicity step | d/dt[1/Φ_n(r_t) - 1/Φ_n(p_t) - 1/Φ_n(q_t)] ≥ 0 | medium (if A+B) |
| D: n=4 algebraic positivity | Δ_4(p,q) ≥ 0 via SOS/CAD after normalization | high |

**Recommended program:**
- Track A: Conditional finite Stam theorem with explicit lemma dependencies
- Track B: Exact n=4 positivity certificate (symbolic computation)

**Fallback positions:** conditional theorem, n=4 exact proof, asymptotic
liminf result, restricted-class proofs (one factor near-Gaussian).

### Local exploration results (same session)

**Strategy B (induction via differentiation):**
- Differentiation commutes EXACTLY with ⊞_n: (p ⊞_n q)'/n = (p'/n) ⊞_{n-1} (q'/n)
  (algebraic proof + numerical verification, error < 10^{-13})
- BUT: naive induction chain FAILS at Step A — 1/Phi_n(p) < 1/Phi_{n-1}(p'/n)
  in 100% of tests (wrong direction)
- Steps B and C of the chain hold 100%

**Strategy A (convexity in cumulants):**
- Cumulant additivity confirmed exact: κ_4 = a_4 - (1/12)a_2² under ⊞_4
- 1/Phi_4 is NEITHER convex NOR concave in cumulant space (82.4%/17.6% split)
- Hessian is INDEFINITE in 100% of trials (75/75)
- Superadditivity HOLDS despite non-convexity (0 violations)
- Dyson BM E[1/Φ_n] is NOT monotone in t (44-54% decreasing steps)
- 1D ray superadditivity holds (0 violations in 62 trials)

**Key insight:** 1/Φ_n is superadditive in cumulant space WITHOUT being
convex. The proof must exploit structure beyond definiteness — possibly the
real-rootedness constraint on the domain.
