# GPL-H Transfer Lemma: Synthesis Handoff (student-dispatch pattern)

**Date:** 2026-02-12
**From:** Claude (advisor)
**To:** Codex (student-explorer)
**Pattern:** agent/student-dispatch

---

## 1. Context

All three attack directions (A: Strongly Rayleigh, B: Hyperbolic barrier,
C: Fixed-block interlacing) have been probed. None closes GPL-H directly.
This note synthesizes what was learned and focuses the next attempt.

### Where each direction stopped

| Direction | Verdict | Failure point |
|-----------|---------|--------------|
| A (SR on edges) | NARROWS GAP | No SR -> grouped-min theorem exists |
| B (Hyperbolic barrier) | FAILS | Frobenius averaging not universal (Barbell = 1.0) |
| C (Avg char poly) | FAILS | Common interlacing doesn't hold for {p_v} |

### The transfer lemma (Direction A's residual)

Direction A got closest. The remaining gap is a single lemma:

> **Transfer lemma (proposed shape):** There exists universal kappa such that
> for any Case-2b state (I_0, S_t, M_t) satisfying H1-H4 and the UST-forest
> SR measure mu on cross-edges:
>
> (i) `||Y_t(v)|| <= kappa * E_mu[||Z_t(v; F)||]` for all v in R_t
>
> (ii) `min_v E_mu[||Z_t(v; F)||] < 1/kappa`
>
> where Z_t(v; F) is the star load induced by forest sample F.

### Why the current shape doesn't work

Codex's stress test (n<=40, 271 UST step rows) reveals the two requirements
fight each other:

| kappa | transfer holds | barrier holds | both hold |
|-------|---------------|--------------|-----------|
| 1.6 | 89.3% | 63.8% | **53.1%** |
| 2.0 | 96.7% | 44.3% | **41.0%** |
| 2.4 | 99.3% | 32.8% | **32.1%** |

Combined feasibility *decreases* as kappa grows. The regime where transfer is
easy (large kappa) is where the barrier target E[min] < 1/kappa becomes
impossible. Worst observed kappa_row = 2.7011.

Hard failures concentrate in dense late-step rows (K_n, ER p=0.5, n=36-40,
eps=0.25-0.3).

---

## 2. Dead ends with reasons (from all three directions)

### DE-A1: Product SR measures for transfer
Product Bernoulli (p=0.5 or degree-normalized) are SR but give poor transfer.
Bernoulli p=0.5 has many degenerate rows (E[min sampled] ~ 0). Degree-normalized
has outliers up to ratio 31.96. **Only UST-forest is viable.**

### DE-A2: Uniform kappa comparability (current theorem shape)
The two constraints (bounded kappa vs barrier target 1/kappa) pull in opposite
directions. Combined feasibility peaks at ~53% and falls. **The theorem shape
needs modification.**

### DE-B1: Frobenius averaging (self-concordance)
avg_v ||Y_t(v)||_F^2 < 1 holds in 75/77 instances but fails on Barbell at
exactly 1.0. Not universal. **Dead end for a proof.**

### DE-B2: Hyperbolicity cone convexity
The average direction (1/r_t)*1 may not lie in Lambda_+. Cone convexity gives
no per-coordinate guarantees. **Structural mismatch.**

### DE-B3: Hessian spreading arguments
Rank gap ~ 1.013 means Y_t(v) atoms are effectively rank-1 — no eigenvalue
spreading to exploit. Frobenius ~ operator norm. **No headroom.**

### DE-C1: Common interlacing for {p_v}
Sampled pairwise interlacing rates drop to ~0.758. Not automatic for the fixed
grouped family. **Unjustified hypothesis.**

### DE-C2: Averaged polynomial root as surrogate
Active-only largest_root(pbar) can reach/exceed 1.0 in hard steps.
**Not robust.**

---

## 3. Directions to explore

### Direction D: Near-rank-1 reformulation

**Source:** Direction B discovered that Y_t(v) atoms have rank gap ~ 1.013 —
they are effectively rank-1. Write Y_t(v) ~ sigma_v * q_v q_v^T where
sigma_v = ||Y_t(v)|| and q_v is the top eigenvector. Then:

    min_v sigma_v = min_v q_v^T Y_t(v) q_v

**Idea:** The GPL-H problem reduces to: among rank-1 projections
{sigma_v q_v q_v^T}, is there one with sigma_v < 1? This IS a rank-1 problem,
where MSS/KS/SR machinery works best. The question becomes whether the fixed
graph grouping prevents using rank-1 tools, or whether near-rank-1 structure
makes the grouping benign.

**Concrete approach:**
1. At each Case-2b step, compute the top eigenvector q_v of each Y_t(v).
2. Form the rank-1 "shadow" family: A_v := sigma_v * q_v q_v^T.
3. Apply MSS to the shadow family {A_v}. The average polynomial
   p_shadow(x) = (1/r_t) sum_v det(xI - A_v) = (1/r_t) sum_v (x - sigma_v) * x^{n-1}
   has largest root = (1/r_t) sum_v sigma_v = average score.
4. If average score < 1 (which follows from L1 drift averaging when rank gap ~ 1),
   then some v has sigma_v < 1.

**Key question:** Does the rank gap staying near 1 extend to larger n? The
tested range is n<=24. If rank gap grows with n, this direction fails.

**Diagnostic to run:** Compute rank gap for n up to 48, all graph families.
If rank gap stays <= 1.05, this direction is viable.

### Direction E: Graph-adaptive transfer constant

**Source:** Direction A's UST data.

**Idea:** Instead of universal kappa, allow kappa to depend on a graph parameter
that is bounded under H1-H4. Candidates:
- spectral gap of the cross-graph (between S_t and R_t)
- maximum cross-degree (bounded by D_0/eps from H2)
- effective resistance diameter of I_0

**Concrete approach:**
1. For each stress-test row, record both kappa_row and candidate graph parameters.
2. Find a graph parameter P such that kappa_row <= f(P) for some explicit f,
   AND the barrier target E[min] < 1/f(P) is feasible.
3. The transfer lemma becomes: ||Y_t(v)|| <= f(P) * E_mu[||Z_t(v;F)||] with
   f(P) bounded by H1-H4.

**Key question:** Does such a parameter exist? The hard cases are dense
late-step rows — what graph parameter distinguishes them?

### Direction F: Bypass transfer via greedy ordering

**Source:** The greedy trajectory itself.

**Idea:** Don't try to prove the transfer lemma for ALL steps simultaneously.
Instead, show that the greedy algorithm (pick v with min score) makes progress
at EACH step: either score < theta (success) or the barrier potential decreases
by a definite amount (progress). If the potential starts bounded and decreases
each step, after enough steps we must hit success.

**Concrete approach:**
1. Define potential Psi_t = tr(B_t) = sum_i 1/(eps - lambda_i(M_t)).
2. When the greedy picks v with score theta_t, the barrier update gives:
   Psi_{t+1} <= Psi_t * (some function of theta_t).
3. If theta_t is large (close to 1), the step is "hard" but the potential
   decreases fast. If theta_t is small, that's the success step.
4. Show the potential can't decrease T times without hitting theta_t < theta.

**Key question:** Does the potential decrease monotonically along the greedy?
The barrier potential tr(B_t) INCREASES as M_t grows (eigenvalues approach eps).
So the potential increases each step — the question is whether it increases
slowly enough that the budget (H4: enough remaining vertices) forces a good step.

**What not to try:** Don't try to make the potential decrease. It increases.
The argument should be: it can't increase too fast without violating H4.

---

## 4. Report format

For each direction explored, return:

```markdown
### Direction [D/E/F]: [name]

**What was tried:**
[1-3 sentences on the specific argument attempted]

**What happened:**
[Computation or derivation results]

**Exact failure point (if applicable):**
[The specific step that doesn't hold, with why]

**Partial results (if any):**
[New bounds, structural insights, or intermediate lemmas]

**Surprises:**
[Anything unexpected]

**New dead ends discovered:**
[Technique or approach tried and ruled out]

**Verdict:** CLOSES GPL-H / NARROWS GAP TO [description] / FAILS AT [step]
```

---

## 5. Success criteria

Same as the original dispatch, with emphasis:

- **Full closure of GPL-H** from any direction. theta = 0.99 is fine.
- **Conditional closure** under a checkable additional hypothesis.
- **New quantitative bound** that improves on trace averaging (already: Frobenius
  averaging at 75/77, need: universal or with identified residual).
- **Rank-gap universality:** Confirm or refute that rank gap stays ~ 1 for
  n up to 48. This alone would be a significant structural result.
- **Potential-based argument skeleton:** Even a heuristic version of Direction F
  that identifies the right potential function is valuable.

Priority order: **D > F > E**. Direction D exploits the strongest structural
insight (near-rank-1) and has the most concrete diagnostic test.

---

## Files to read

| File | What it contains |
|------|-----------------|
| `data/first-proof/problem6-gpl-h-attack-dispatch.md` | Original 3-direction dispatch (context, dead ends DE1-DE5) |
| `data/first-proof/problem6-direction-a-report.md` | Direction A results (UST transfer, stress test) |
| `data/first-proof/problem6-direction-b-report.md` | Direction B results (Frobenius, rank gap) |
| `data/first-proof/problem6-direction-c-report.md` | Direction C results (interlacing failure) |
| `data/first-proof/problem6-proof-attempt.md` | Full proof attempt, GPL-H at lines 719-748 |
| `scripts/verify-p6-gpl-h.py` | Numerical GPL-H verifier |
| `scripts/verify-p6-gpl-h-direction-a.py` | Direction A probe script |
| `scripts/verify-p6-gpl-h-direction-b.py` | Direction B probe script |
| `scripts/verify-p6-gpl-h-direction-c.py` | Direction C probe script |
