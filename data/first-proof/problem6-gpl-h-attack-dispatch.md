# GPL-H Attack Dispatch (student-dispatch pattern)

**Date:** 2026-02-12
**From:** Claude (advisor)
**To:** Codex (student-explorer)
**Pattern:** agent/student-dispatch

---

## 1. Context

Problem 6 asks: does there exist universal c₀ > 0 such that every weighted
graph G has a vertex subset S with |S| ≥ c₀·ε·n and L_{G[S]} ≤ ε·L?

The proof reduces cleanly to a single open bridge called **GPL-H** (Grouped
Paving Lemma — Hypothesis form). Everything else is proved:

- Leverage threshold lemma: S must be independent in the heavy subgraph G_H.
- Turán bound: ∃I independent in G_H with |I| ≥ εn/3.
- Cases 1, 2a: closed (c₀ = 1/3).
- Core regularization (Step 0): extract I₀ ⊆ I with |I₀| ≥ |I|/2 and
  leverage-degree ℓ_v ≤ 12/ε.
- Reduction proposition: **L2\* alone implies linear-size ε-light set.**

### GPL-H statement

For universal c_step > 0 and θ ∈ (0,1), given any Case-2b state
(I₀, S_t, M_t) with t ≤ c_step·ε·n:

Define:
```
B_t = (εI - M_t)^{-1}
C_t(v) = Σ_{u ∈ S_t, u~v} X_{uv}
Y_t(v) = B_t^{1/2} C_t(v) B_t^{1/2}      for v ∈ R_t := I₀ \ S_t
```

Under hypotheses:
- **H1**: τ_{uv} ≤ ε for all edges internal to I₀
- **H2**: ℓ_v ≤ D₀/ε (D₀ = 12) for all v ∈ I₀
- **H3**: M_t ≤ εI (barrier valid)
- **H4**: |R_t| ≥ η·ε·n

**Conclude:** min_{v ∈ R_t} ‖Y_t(v)‖ ≤ θ.

### Key identities

- X_{uv} = w_{uv} L^{+/2} b_{uv} b_{uv}^T L^{+/2}, rank-1 PSD, ‖X_{uv}‖ = τ_{uv}
- Σ_e X_e = I (projection onto range of L)
- M_t = Σ_{f⊆S_t} X_f
- If score_t(v) ≤ θ < 1, then M_{t+1} = M_t + C_t(v) ≤ (1−θ)M_t + θεI ≤ εI

### What's proved about the Y_t(v) family

- **L1 (drift averaging):** avg_{v∈R_t} drift_t(v) ≤ (tD/r_t)·tr(B_t²).
  Guarantees ∃v with small drift. Does NOT control score.
- **Trace averaging for score:** min_v ‖Y_t(v)‖ ≤ (tD/r_t)·tr(B_t).
  Not universal because tr(B_t) grows with n and barrier proximity.
- **Tested and falsified:** Σ_v Y_t(v) ≤ ρI with ρ < 1. The sum ‖Σ_v Y_t(v)‖
  is often ≫ 1 on dense graphs, yet min_v ‖Y_t(v)‖ stays < 1.

### Empirical signal

313 baseline + 2040 randomized trajectories (n ≤ 48): worst θ = 0.667.
Exhaustive check n ≤ 14: worst 0.476. No counterexample anywhere.

---

## 2. Dead ends with reasons

### DE1: Trace-to-norm (6 variants)
All approaches using ‖M‖ ≤ tr(M) lose a factor of rank(Y_v) ≤ D₀/ε.
The trace-only greedy ceiling is proved: tr(M_{S_t}) ≤ D·t²/(m₀−t), giving
sublinear set size. **This eliminates the entire technique class.**

### DE2: Star domination
Replacing Z_u Z_v with (Z_u+Z_v)/2 converts p² dependence to p, destroying
concentration headroom. Star graph counterexample: ‖A_v‖ = 1/2 for all v,
yet problem is trivially solvable (take all leaves). **Structural mismatch.**

### DE3: Budget condition Σ_v Y_t(v) ≤ ρI
Falsified numerically. The global budget ‖Σ_v Y_t(v)‖ can be ≫ 1 while
min_v ‖Y_t(v)‖ < 1. So the argument cannot go through a global PSD budget
bound on the sum.

### DE4: Recursive decoupling + MSS
Bipartition I' into A, B, subsample one side, apply MSS to cross-terms.
Reproduces the subsampling bound exactly: c₀ = O(√ε). The quadratic-vs-linear
scaling mismatch is fundamental to all subsample-and-control approaches.

### DE5: Direct MSS on the Y_t(v) family
MSS gives bounds (√μ + √R)² where R = max ‖atom‖. Since R can be up to ε
(light edges) and the bound is ≥ R, there's no room. MSS applied to the
grouped atoms directly doesn't help because it optimizes over partitions and
we have a fixed one.

### Structural obstruction theorem (from DE1-DE5)
**Any technique whose operator-norm control factors through trace averaging,
star domination, global budget Σ_v Y_t(v) ≤ ρI, or subsample-and-concentrate
is structurally blocked for GPL-H.** The proof must control λ_max(Y_t(v))
directly, exploiting anisotropic/cancellation structure beyond scalar traces.

---

## 3. Directions to explore

Three attack paths, one per dispatch. Each adapts a technique from a solved
analogous problem. **Explore these as open-ended investigations, not
verifications.** You may discover that a direction partially works, needs
modification, or reveals new structure.

### Direction A: Strongly Rayleigh on vertex indicators

**Source:** Anari-Gharan (arXiv:1411.4613, 1412.1143). Their KS-for-strongly-
Rayleigh theorem: if μ is a strongly Rayleigh distribution on edge subsets
with bounded marginals and small individual atoms, then some realization has
bounded spectral norm.

**Adaptation idea:**
1. Define a strongly Rayleigh distribution μ on subsets of R_t. Candidate:
   the DPP with kernel K related to leverage scores on I₀, or the uniform
   spanning tree restricted to I₀.
2. For each vertex v, define the "vertex load" Y_t(v). Under μ, these loads
   are correlated through the SR structure.
3. Apply the Anari-Gharan bound: if individual edge atoms ‖A_{uv}^{(t)}‖ are
   small (≤ ε·‖B_t‖², from H1) and marginals are bounded (from H2), then
   some v has ‖Y_t(v)‖ ≤ θ.

**Key question:** Does the SR structure on **edges** induce enough control on
**vertex-star sums** Y_t(v)? The Anari-Gharan theorem bounds the spectral norm
of a random edge-sum; we need the spectral norm of a random vertex-star-sum.
This is a nontrivial transfer — investigate whether the proof technique
(real stability of the generating polynomial) survives the vertex-star
aggregation.

**What not to try:** Don't try to make the vertex indicators themselves SR —
vertex indicators in a graph are NOT strongly Rayleigh in general (SR is about
negative correlations, but vertex neighborhoods induce positive correlations).
The SR structure must live on **edges**, with vertex stars as derived objects.

### Direction B: Hyperbolic barrier in hyperbolicity cone

**Source:** Brändén (arXiv:1809.03255). Extends MSS from rank-1 to arbitrary-
rank PSD atoms using hyperbolic polynomials. The hyperbolicity cone generalizes
the PSD cone, and the barrier function extends naturally.

**Adaptation idea:**
1. Define the multivariate polynomial:
   ```
   p(x) = det(εI − M_t − Σ_{v∈R_t} x_v C_t(v))
   ```
   where x_v ≥ 0 are formal variables, one per remaining vertex.
2. This polynomial is hyperbolic with respect to the direction e = (0,...,0)
   (the zero assignment), since εI − M_t ≻ 0 by H3.
3. The hyperbolicity cone Λ_+(p, e) consists of directions x where roots
   stay real and positive.
4. Apply Brändén's machinery: the mixed characteristic polynomial
   E_v[det(xI − Y_t(v))] should have bounded largest root if individual
   atoms satisfy the rank-adjusted Weaver condition.
5. The largest root of the average characteristic polynomial bounds
   min_v ‖Y_t(v)‖ via the interlacing/common-interlacing lemma.

**Key question:** What are the quantitative bounds? Brändén's KS_r bound for
rank-r atoms has the form (√(r·μ) + √R)². With r ≤ D₀/ε (leverage degree
bound), this could give a bound that depends on 1/ε — not universal.
Investigate whether the specific structure of GPL-H (the atoms come from a
graph Laplacian, not arbitrary PSD matrices) gives better rank control.

**What not to try:** Don't try to apply the scalar MSS theorem directly to
Y_t(v) — each Y_t(v) has rank > 1, so the rank-1 MSS machinery (Sherman-
Morrison for potential updates) doesn't apply. The whole point of this
direction is to use the hyperbolic polynomial generalization.

### Direction C: Fixed-block interlacing (average characteristic polynomial)

**Source:** MSS interlacing families (arXiv:1306.3969) + Xie-Xu fixed-block
subset selection (arXiv:1903.06350).

**Adaptation idea:**
1. For each v ∈ R_t, define the characteristic polynomial:
   ```
   p_v(x) = det(xI − Y_t(v))
   ```
2. Form the average: p̄(x) = (1/r_t) Σ_v p_v(x).
3. **MSS largest-root lemma:** If the polynomials {p_v} have a common
   interlacing, then some p_v has largest root ≤ largest root of p̄.
4. Therefore: if largest_root(p̄) < 1, then ∃v with ‖Y_t(v)‖ < 1.
5. Compute or bound largest_root(p̄) using H1-H4.

**Key questions:**
- Do the {p_v(x)} have a common interlacing? This requires a polynomial q of
  degree dim−1 such that q interlaces each p_v. Investigate whether the graph
  structure provides this.
- Can we bound the coefficients of p̄(x) = (1/r_t) Σ_v det(xI − Y_t(v))
  using the trace bounds from L1 and the individual-atom bounds from H1?
- The Xie-Xu "fixed block" machinery treats S_t as pre-fixed and applies
  interlacing to candidate additions. Does their framework directly apply
  if we view each v ∈ R_t as a candidate column-block?

**What not to try:** Don't try to show the average MATRIX (1/r_t)Σ_v Y_t(v)
has norm < 1 — this is DE3 (falsified). The characteristic polynomial average
is fundamentally different from the matrix average: the average of det(xI−A_i)
carries spectral-spreading information that the average of A_i does not.

---

## 4. Report format

For each direction explored, return:

```markdown
### Direction [A/B/C]: [name]

**What was tried:**
[1-3 sentences on the specific mathematical argument attempted]

**What happened:**
[The argument proceeds to step X, where it encounters Y]

**Exact failure point (if applicable):**
[The specific inequality/identity/lemma that doesn't hold, with why]

**Partial results (if any):**
[Any new bounds, structural insights, or intermediate lemmas proved]

**Surprises:**
[Anything unexpected — a direction that partially works in a different way,
a new connection, a reformulation that might help]

**New dead ends discovered:**
[Any new technique or approach that was tried and ruled out]

**Verdict:** CLOSES GPL-H / NARROWS GAP TO [description] / FAILS AT [step]
```

---

## 5. Success criteria

Progress means any of:
- **Full closure:** A proof of GPL-H from any direction. θ = 0.99 is fine.
- **Conditional closure:** GPL-H under an additional hypothesis that's checkable
  and plausibly true (e.g., "if the graph has expansion ≥ δ within I₀").
- **Quantitative narrowing:** A new bound on min_v ‖Y_t(v)‖ that's better than
  the trace averaging bound, even if not < 1.
- **Structural insight:** A reformulation of GPL-H that connects it to a known
  open problem or reveals new structure.
- **New dead end with proof:** A rigorous argument that one of the three
  directions cannot work, with the exact structural reason.

Any of these advances the project. A well-documented failure is valuable.

---

## Files to read

| File | What it contains |
|------|-----------------|
| `data/first-proof/problem6-proof-attempt.md` | Full proof attempt, GPL-H at lines 719-748, all dead ends |
| `data/first-proof/problem6-library-research.md` | Literature survey + MO/MSE threads on closest results |
| `data/first-proof/problem6-method-wiring-library.md` | D1-D10 technique diagrams |
| `scripts/verify-p6-gpl-h.py` | Numerical verifier (run with `--nmax 20` to stay safe on memory) |
| `data/first-proof/p6-p7-process-patterns.md` | Process patterns from the proof journey |
