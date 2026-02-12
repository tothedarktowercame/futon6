# Problem 6: Direction B Probe Report (Hyperbolic Barrier / Self-Concordance)

Date: 2026-02-12
Author: Claude

## Direction B: Hyperbolic barrier in hyperbolicity cone

**What was tried:**
The multivariate polynomial P(x) = det(εI − M_t − Σ_v x_v C_t(v)) is
hyperbolic with respect to x=0 (since εI − M_t ≻ 0 by H3). The hyperbolicity
cone Λ_+ = {x ≥ 0 : P(x) > 0} is convex, and the BSS barrier potential
Φ = tr(εI − M_t)^{-1} is its standard barrier function. The goal was to use
properties of this cone — convexity, self-concordance, Frobenius/local norm
structure, and mixed discriminants — to prove GPL-H.

Three sub-approaches were investigated:

1. **Self-concordance local norm.** For the barrier F = -log P, the local norm
   of the direction e_v at x=0 is ||Y_t(v)||_F (Frobenius norm). A move
   x = 0 → x = e_v is self-concordantly feasible iff ||Y_t(v)||_F < 1.
   Since ||Y_t(v)|| ≤ ||Y_t(v)||_F, this is a sufficient condition for
   score < 1. Tested whether avg_v ||Y_t(v)||_F^2 < 1 (Frobenius averaging).

2. **Cone convexity.** The cone Λ_+ is convex and contains s·1 for small
   s > 0. If the cone extends far enough in some coordinate direction e_v,
   then ||Y_t(v)|| < 1. Investigated whether convexity + budget constraints
   force this.

3. **Hessian anisotropy.** The Hessian ∂²(-log P)/∂x_u ∂x_v = tr(Y_t(u)Y_t(v))
   captures pairwise vertex interactions. Investigated whether off-diagonal
   structure reveals spreading not captured by diagonal (drift) averaging.

Script: `scripts/verify-p6-gpl-h-direction-b.py`

---

**What happened:**

### Sub-approach 1: Frobenius averaging

The Frobenius averaging bound — if avg_v ||Y_t(v)||_F^2 < 1 then some v has
||Y_t(v)|| < 1 — holds in **75 out of 77 tested Case-2b instances** (n ≤ 24).

The two failures are both Barbell graphs hitting avg ||Y||_F^2 = 1.0000:
- Barbell_8 at eps=0.25
- Barbell_10 at eps=0.20

This is strictly better than trace averaging (which fails widely, with
trace bounds > 30 in many cases), but NOT universal.

### Sub-approach 2: Cone convexity

The convexity of the hyperbolicity cone does not help directly. The
"average direction" ā = (1/r_t)·1 may not be in Λ_+ (this is equivalent to
||Σ_v Y_t(v)/r_t|| < 1, which is the budget condition DE3 that fails in
dense cases). Convexity of the cone says: if a point is in the cone, so are
all convex combinations with the origin. But this gives no per-coordinate
guarantees.

### Sub-approach 3: Hessian anisotropy

The rank gap (drift/score ratio) averages **1.013** across all instances —
meaning eigenvalues of Y_t(v) are overwhelmingly concentrated in a single
direction (near rank-1). This means:
- ||Y_t(v)||_F ≈ ||Y_t(v)|| (Frobenius ≈ operator norm)
- The Frobenius bound doesn't gain much over the operator norm bound
- Off-diagonal Hessian terms are small relative to diagonal

The near-rank-1 structure is itself a useful structural insight: the barrier-
normalized grouped updates Y_t(v) behave as if they're nearly rank-1, even
though they're sums of multiple rank-1 edge atoms.

---

**Exact failure point:**

The self-concordance / Frobenius approach fails because:

1. **The Frobenius norm is not strictly smaller than 1 universally.**
   Barbell graphs at critical eps values have avg ||Y_t(v)||_F^2 = 1.0,
   which means some vertex v has ||Y_t(v)||_F ≥ 1. Since ||Y_t(v)|| ≤
   ||Y_t(v)||_F, we can't conclude ||Y_t(v)|| < 1 from this.

2. **The rank gap is ≈ 1**, meaning ||Y_t(v)||_F ≈ ||Y_t(v)||. So the
   Frobenius approach doesn't gain significant headroom over direct score
   control. The intermediate position (between trace and operator norm)
   collapses because the atoms are effectively rank-1.

3. **The barrier function framework IS the BSS greedy framework** — just
   viewed through the lens of hyperbolic polynomials. The hyperbolicity
   cone formulation doesn't introduce new information beyond what the
   barrier potential Φ_t = tr(B_t) already captures.

The fundamental issue: hyperbolic polynomial tools (Brändén, Zhang-Zhang)
give bounds for OPTIMIZED partitions/selections. The hyperbolicity cone is
convex but the "good vertex" question is about extreme rays of the positive
orthant intersecting the cone — a combinatorial question that convexity alone
doesn't resolve for fixed graph structure.

---

**Partial results:**

1. **Frobenius averaging nearly universal:** 75/77 instances. This is a new
   quantitative result — the self-concordance local norm is < 1 for almost
   all tested states. The Barbell failures are edge cases.

2. **Near-rank-1 structure discovered:** avg rank gap ≈ 1.013. The barrier-
   normalized grouped updates Y_t(v) = B_t^{1/2} C_t(v) B_t^{1/2} are
   effectively rank-1 despite being sums of multiple edge atoms. This
   means the top eigenvalue dominates, and eigenvalue spreading (which
   would make Frobenius < operator norm) is minimal.

   **Implication:** If the Y_t(v) are effectively rank-1, then the GPL-H
   problem is closer to the standard rank-1 MSS setting than it appears.
   The difficulty is not rank — it's the fixed grouping.

3. **Hessian off-diagonal structure:** The cross-terms tr(Y_t(u) Y_t(v))
   are computable and small relative to diagonal terms. This suggests
   vertex contributions are approximately independent in the spectral
   sense, which could be exploitable by an independence-based argument
   (like Direction A's strongly Rayleigh approach).

---

**Surprises:**

1. The rank gap being ≈ 1 was unexpected. A priori, each Y_t(v) is a sum
   of up to D₀/ε rank-1 atoms, so could have rank up to D₀/ε ≈ 40-60.
   In practice, the top eigenvalue dominates completely. This means the
   "trace-to-norm gap" (which seemed like the core difficulty) is actually
   small — the real difficulty is elsewhere.

2. The Frobenius bound failing at exactly 1.0 on Barbell graphs (not > 1,
   but = 1 to numerical precision) suggests a structural phenomenon rather
   than a quantitative looseness. The Barbell bridge vertex may be
   a degenerate case where all edge contributions align.

3. The BSS barrier and the hyperbolicity cone are literally the same
   mathematical object — the formulation adds no new information. This
   was suspected but now confirmed computationally: every quantity
   accessible via the hyperbolic polynomial view (barrier function,
   self-concordance, cone membership) reduces to a function of B_t and
   {C_t(v)} that was already available.

---

**New dead ends discovered:**

1. **Self-concordance local norm averaging** (Frobenius bound) is not
   universal — fails on Barbell at critical eps. Dead end for a proof.

2. **Cone convexity → per-coordinate guarantee** does not work because the
   budget condition (average direction in cone) can fail (DE3).

3. **Hessian-based spreading arguments** are ineffective because rank gap
   ≈ 1 — there's no significant eigenvalue spreading to exploit.

4. **Brändén's rank-r KS extension** would give bounds of the form
   (√(r·μ) + √R)², but with near-rank-1 atoms this reduces to the
   standard MSS bound, which doesn't help for fixed partitions.

---

**Structural insight for Direction A (strongly Rayleigh):**

The near-rank-1 discovery is significant for Direction A. If Y_t(v) is
effectively rank-1, write Y_t(v) ≈ σ_v · q_v q_v^T where σ_v = ||Y_t(v)||
and q_v is the top eigenvector. Then:

    min_v σ_v = min_v q_v^T Y_t(v) q_v

The question reduces to: among the rank-1 projections {σ_v q_v q_v^T},
is there one with σ_v < 1? This IS essentially a rank-1 problem, and
rank-1 is where MSS/KS/SR machinery works best.

The key remaining question: does the fixed graph grouping prevent us
from using rank-1 tools? Or does the near-rank-1 structure mean the
grouping is benign?

---

**Verdict:** FAILS AT Frobenius averaging universality
(Barbell counterexample at avg ||Y||_F^2 = 1.0).

Direction B does not close GPL-H but provides two useful structural
insights: (1) the Frobenius bound is NEARLY universal (75/77), and
(2) the atoms Y_t(v) are effectively rank-1, which means the core
difficulty is the fixed grouping, not the rank. This sharpens the
target for Direction A.

## Quantitative summary

| Metric | Value |
|--------|-------|
| Case-2b instances tested | 77 |
| Steps analyzed | 145 |
| Worst min score | 0.744 (< 1, GPL-H consistent) |
| Worst avg ||Y||_F^2 | 1.000 (Frobenius bound fails) |
| Frobenius bound holds | 75/77 (97%) |
| Avg rank gap | 1.013 (near rank-1) |
| Worst trace bound | ~35 (trace averaging hopeless) |

## Files

- `scripts/verify-p6-gpl-h-direction-b.py` — diagnostic script
- `data/first-proof/problem6-direction-b-report.md` — this report
