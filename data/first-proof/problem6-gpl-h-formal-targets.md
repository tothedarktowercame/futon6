# GPL-H Formal Closure: Student Dispatch (targets G1–G3)

**Date:** 2026-02-12
**From:** Claude (advisor)
**To:** Codex (student-explorer)
**Pattern:** agent/student-dispatch

---

## 1. Context

GPL-H is 95-98% closed. Three independent empirical certificates (max
score, avg trace, avg charpoly root) have zero violations across 569
nontrivial rows at n ≤ 96. The gap is purely formal.

Previous directions explored:

| Direction | Verdict |
|-----------|---------|
| A (Strongly Rayleigh) | NARROWS GAP — no SR→grouped-min theorem |
| B (Hyperbolic barrier) | FAILS — Frobenius not universal |
| C (Interlacing) | FAILS — common interlacing doesn't hold |
| D (Near-rank-1) | FAILS — rank gap grows to 2.2 at n≤48 |
| KK extremal | PROVES for K_k — score ≤ 5/6 at c_step = 1/3 |
| Phase structure | PROVES Phase 1 — 64% of instances trivially |
| Double-counting | PROVES when m_0 ≈ n — dbar ≤ 0.74 < 1 |

The proof NOW reduces to a single structural claim about Phase 2.

### The two-phase decomposition (proved)

At each greedy step t ≤ εn/3:

**Phase 1 (∃ zero-score vertex):** GPL-H holds trivially (score = 0 < 1).

**Phase 2 (all vertices active):** Need min score < 1. The ratio
certificate gives min ≤ dbar/gbar ≤ dbar (since gbar ≥ 1). So
**dbar < 1 suffices**.

### The double-counting bound (proved)

When S_t has average leverage degree ≤ avg(I_0) and m_0 ≈ n:

    dbar ≤ 2(n-1)t / (m_0 · ε · (m_0 - t))

At t = εn/3 and m_0 = n: dbar ≤ 2/(3(1 - ε/3)) < 1 for ε < 1.

### The empirical constraint (verified, not proved)

In ALL 609 Phase 2 step-rows (n ≤ 96): **m_0/n ≥ 0.952**.

Phase 2 is ONLY reached when I_0 is nearly all of V.

---

## 2. Dead ends from this round

### DE-G1: Scalar trace bound for dbar
The bound dbar ≤ tr(Q)/(|A_t|·(ε − ‖M_t‖)) gives dbar ≥ 1 always,
because it ignores anisotropy between crossing edges and barrier.
**Cannot close GPL-H from scalar H1-H4 bounds alone.**

### DE-G2: Turán bound on m_0
The Turán lower bound |I_0| ≥ εn/(2+ε) is too loose. For ε = 0.3:
|I_0| ≥ 0.13n, but we need |I_0| ≥ 0.87n for the double-counting
to give dbar < 1. **Turán alone gives factor-of-7 gap.**

### DE-G3: Leverage budget for all-of-S_t
The bound Σ_{u∈S_t} ℓ_u ≤ Σ_{e∈E(I_0)} τ_e ≤ n−1 is correct but
too loose (proportional to n, not to t). Need proportional-to-t bound.
**Works for m_0 ≈ n but not for small m_0.**

---

## 3. Formal targets to explore

### Target G1: Domination implies large I_0

**Claim:** If the I_0-subgraph (all edges have τ ≤ ε) is dominatable by
t ≤ εn/3 vertices, then |I_0| ≥ cn for universal c ∈ (0, 1).

**Why it's plausible:** Domination by t vertices requires average degree
≥ (m_0 − t)/t in the I_0-subgraph. At t = εn/3: degree ≥ 3(m_0/n − ε/3)/ε.
For m_0 = 0.5n: degree ≥ 3(0.5 − 0.1)/0.3 = 4. This is moderate and
doesn't obviously constrain m_0.

BUT: the I_0-subgraph has ONLY light edges (τ ≤ ε). The total leverage
of these edges is bounded: Σ τ ≤ ε·|E|. And the degree-leverage product
gives: max degree ≤ leverage-degree / min(τ). For the subgraph to be
dominatable AND have only light edges, there's a tension.

**Concrete approach:**
1. Assume Phase 2 reached at step t ≤ εn/3 with m_0 < cn.
2. Use the domination condition: Σ_{u∈S_t} deg(u, R_t) ≥ m_0 − t.
3. Use the light-edge constraint: Σ_{e∈E(I_0)} τ_e ≤ ε·|E(I_0)|.
4. Use the leverage budget: Σ_{e∈E(I_0)} τ_e ≤ n − 1.
5. Derive a contradiction from m_0 < cn.

**Key identity to exploit:** I_0 is independent in G_H = {e : τ_e > ε}.
So |V \ I_0| ≤ n − m_0, and each vertex in V \ I_0 has a heavy neighbor
in I_0. The heavy-edge budget: |E_H| ≤ (n−1)/ε. Each heavy edge removes
at most one vertex from I_0, so n − m_0 ≤ |E_H| ≤ n/ε. This gives
m_0 ≥ n(1 − 1/ε), which is NEGATIVE for ε < 1.

Better: each vertex in V \ I_0 has degree ≥ 1 in G_H. The total heavy
degree is 2|E_H| ≤ 2n/ε. Among V \ I_0 vertices, total heavy degree
≥ n − m_0 (each needs ≥ 1 heavy edge). Among I_0 vertices, total heavy
degree ≤ 2n/ε − (n − m_0). This bounds the heavy degree of I_0
vertices, which constrains how dense the I_0 subgraph can be.

**Diagnostic:** For each Phase 2 instance, compute: (a) number of heavy
edges, (b) heavy degree of I_0 vertices, (c) density of I_0-subgraph.
Look for the structural reason m_0/n ≥ 0.95.

### Target G2: Average leverage degree of greedy selection

**Claim:** If S_t is selected by the min-score greedy from I_0, then
(1/t) Σ_{u∈S_t} ℓ_u ≤ 2 + o(1).

**Why it's plausible:** The global average leverage degree is < 2 (since
Σ_e τ_e = n−1, so Σ_v ℓ_v = 2(n−1), avg = 2(n−1)/n). The greedy
selects zero-score vertices (Phase 1), which are NOT adjacent to S_t.
These are "spectrally isolated" vertices that plausibly have
below-average leverage degree.

**Empirical support:** In 609 Phase 2 rows, the ratio avg_lev(S_t) /
avg_lev(I_0) ranges from 0.68 to 1.045 (mean 0.999). The greedy
selects near-average vertices.

**Concrete approach:**
1. Show that the greedy Phase 1 selection is an independent set in
   the I_0-subgraph (PROVED — zero-score vertices have no I_0-edges to S_t).
2. For an independent set S in a graph G with Σ_e τ_e = L:
   Σ_{u∈S} ℓ_u ≤ L (each edge contributes to at most one endpoint in S).
3. If L ≤ n−1 and |S| = t: average ℓ per selected vertex ≤ (n−1)/t.
4. But we need avg ≤ 2, which requires t ≥ (n−1)/2. Since t ≤ εn/3,
   this FAILS for small ε.
5. Alternative: use L_int = Σ_{e∈E(I_0)} τ_e (internal leverage only).
   For K_k: L_int = k−1 and avg ℓ = 2(k−1)/k ≈ 2.
   The question is whether L_int / m_0 ≤ 1 (average is controlled).

**Key obstacle:** The bound Σ_{u∈S_t} ℓ_u ≤ L_int is correct but
doesn't scale with t. We need: the FRACTION of total leverage in S_t
is ≈ t/m_0 (proportional selection). This is a uniformity statement.

**Possible tool:** Expander mixing lemma or discrepancy bounds for
independent sets in the I_0-subgraph.

### Target G3: Direct determinant positivity

**Claim:** At each nontrivial step with M_t = 0:
Σ_{v∈A_t} det(I − Y_t(v)) > 0.

This implies p_avg(1) > 0, so some v has ‖Y_t(v)‖ < 1.

**Why it's plausible:** Empirically, the largest root of p_avg is ≤ 0.500
(huge margin above the threshold). The sum of determinants is strongly
positive.

**Proved for t = 1:** Σ_v det(I − Y_1(v)) = Σ_v (1 − τ/ε) > 0 by strict H1.

**Concrete approach for general t:**
1. When M_t = 0: det(I − Y_t(v)) = det(I − C_t(v)/ε) = ε^{−n} det(εI − C_t(v)).
2. C_t(v) = Σ_{u∈S_t, u~v} X_{u,v} is a sum of rank-1 PSD matrices.
3. det(εI − C_t(v)) can be computed using the matrix determinant lemma
   for rank-1 updates: det(A − uuᵀ) = det(A)(1 − uᵀA⁻¹u).
4. For d edges from v to S_t: det(εI − C_t(v)) = ε^n · Π_{i=1}^{d}(1 − σ_i)
   where σ_i are related to the sequential Schur complements.
5. The sum Σ_v may telescope or simplify via a spanning-tree-type identity.

**Possible tool:** Matrix-tree theorem. The quantities det(εI − C_t(v))
involve Laplacian submatrices. Kirchhoff's theorem connects determinants
of Laplacian submatrices to spanning tree counts. There may be a
combinatorial identity for Σ_v det(εI − C_t(v)).

**Key question:** Does Σ_v det(εI − C_t(v)) have a closed form in terms
of graph invariants (spanning trees, effective resistances)?

---

## 4. Report format

For each target explored, return:

```markdown
### Target G[1/2/3]: [name]

**What was tried:**
[1-3 sentences on the specific argument attempted]

**What happened:**
[Computation or derivation results]

**Exact failure point (if applicable):**
[The specific step that doesn't hold, with why]

**Partial results (if any):**
[New bounds, structural insights, or intermediate lemmas]

**Verdict:** CLOSES GPL-H / NARROWS GAP TO [desc] / FAILS AT [step]
```

---

## 5. Success criteria

- **Full closure of GPL-H** from any target. θ = 0.99 is fine.
- **Conditional closure** under m_0/n ≥ c, with proof that c holds
  in the relevant regime (Phase 2 + Case 2b).
- **New structural bound** on m_0 or leverage degree that tightens
  the Turán bound in Case 2b.
- **Determinant identity** for Σ_v det(εI − C_t(v)) that can be
  bounded from below.

Priority order: **G1 > G3 > G2**. G1 is the most concrete (prove a
lower bound on m_0) and has the strongest empirical support (m_0/n ≥ 0.95).
G3 bypasses m_0 entirely via a direct algebraic argument.

---

## 6. Files to read

| File | What it contains |
|------|-----------------|
| `data/first-proof/problem6-gpl-h-closure-attempt.md` | Full closure state (this round's analysis) |
| `data/first-proof/problem6-kk-extremal-analysis.md` | K_k formula, phase structure, all-scores bound |
| `data/first-proof/problem6-proof-attempt.md` | Full proof, GPL-H at L719-748, ratio cert at L771-975 |
| `scripts/verify-p6-gpl-h.py` | Base GPL-H verifier (graph generators, spectral comp) |
| `scripts/verify-p6-allscores-bound.py` | All-scores < 1 check (n ≤ 96) |
| `scripts/verify-p6-avg-charpoly.py` | Average charpoly root check |
| `scripts/verify-p6-double-counting.py` | Leverage degree of greedy selection |
| `scripts/verify-p6-phase2-m0.py` | Phase 2 m_0/n analysis |
| `scripts/verify-p6-trace-budget.py` | Trace budget and anisotropy |
| `scripts/verify-p6-phase-structure.py` | Phase 1/2 structure |
| `scripts/verify-p6-fresh-dbar-bound.py` | Leverage structure analysis |

## 7. Key quantities reference

| Symbol | Definition | Empirical bound |
|--------|-----------|----------------|
| `score(v) = ‖Y_t(v)‖` | Barrier-normalized edge load | ≤ 0.675 (n≤96) |
| `dbar` | avg tr(Y_t(v)) over active v | ≤ 0.683 |
| `gbar` | avg (tr/‖·‖) over active v | ≥ 1.0 (proved) |
| `m_0 = \|I_0\|` | Size of regularized independent set | ≥ 0.95n in Phase 2 |
| `ℓ_u` | Leverage degree of u in I_0-subgraph | avg ≈ 1.92 |
| `L_int` | Total internal leverage Σ_{e∈E(I_0)} τ_e | ≈ m_0 − 1 |
| `avg charpoly root` | Largest root of avg det(xI − Y_v) | ≤ 0.500 |
