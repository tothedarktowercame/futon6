# Problem 6: K_k Extremal Analysis for GPL-H

Date: 2026-02-12
Author: Claude

## Discovery

Exact computation on the worst-case family (DisjCliq_k×3) reveals a clean
formula that may close GPL-H.

## First-step identity (proved)

When `M_t = 0` and `B_t = (1/ε)I` (fresh barrier), the score of vertex `v`
with a single cross-edge to a selected neighbor `u` is:

    ‖Y_t(v)‖ = τ_{u,v} / ε

**Proof:** `Y_t(v) = B_t^{1/2} C_t(v) B_t^{1/2} = (1/ε) X_{u,v}`, so
`‖Y_t(v)‖ = ‖X_{u,v}‖/ε = τ_{u,v}/ε`. □

Under H1 (`τ ≤ ε`): **score ≤ 1**. Under strict H1 (`τ < ε`): **score < 1**.

## K_k eigenvalue formula (approximate, verified numerically)

For `K_k` with `t` vertices selected (all edges have `τ = 2/k`), every
remaining vertex has eigenvalues:

- `(t+1)/(kε)` — the "trace term" (from fresh/mixed eigenspace)
- `1/(kε - t)` — the "barrier term" (from consumed eigenspace, mult `t-1`)

Score: `score(t) = max((t+1)/(kε), 1/(kε - t))`

### Verification (`K_60`, `ε=0.3`, `kε=18`):

| t | actual | formula | error |
|---|--------|---------|-------|
| 1 | 0.111111 | 0.111111 | 0.000000 |
| 2 | 0.166673 | 0.166667 | 0.000006 |
| 4 | 0.277862 | 0.277778 | 0.000084 |
| 8 | 0.501369 | 0.500000 | 0.001369 |

Exact at `t=1`, error < 0.3% at `t=8`.

## Universal bound within the greedy horizon

With `c_step = 1/3`, the per-clique horizon is `T_c = c_step · kε = kε/3`.

At `t = kε/3` (end of horizon):
- Trace term: `(kε/3 + 1)/(kε) = 1/3 + 1/(kε) ≤ 1/3 + 1/2 = 5/6`
  (using `kε ≥ 2` from H1)
- Barrier term: `1/(kε - kε/3) = 3/(2kε) ≤ 3/4`
  (using `kε ≥ 2`)

Therefore: **score ≤ 5/6 < 1** within the horizon for any `K_k` under H1.

This gives `θ = 5/6` and `c_step = 1/3`.

## DisjCliq worst-case structure

DisjCliq_k×3 (3 cliques of `K_k` connected by bridges) achieves the worst
observed `avg_score`:

- The greedy selects one vertex per clique before any within-clique
  duplicates (since unconnected vertices have score 0).
- At the "full coverage" step (one selected per clique), all remaining
  vertices have score `τ/ε = 2/(kε)`.
- The worst case is `kε → 2` (H1 boundary): score → 1.

Observed worst: `20/27 ≈ 0.741` at `DisjCliq_9x3, ε=0.3` (where `kε=2.7`).

## Boundary case: `τ = ε`

When `2/k = ε` exactly (integer `k = 2/ε`), the score at `t=1` is exactly 1.
This requires the H1 boundary `τ = ε`.

**Resolution:** The proof of GPL-H requires strict H1 (`τ < ε` for edges
internal to `I_0`). This is naturally satisfied:
- `G_H = {e : τ_e > ε}` uses strict inequality.
- `I_0` is independent in `G_H`, so internal edges have `τ ≤ ε`.
- Edges with `τ = ε` are on the boundary. These edges form a subgraph
  `G_= = {e : τ_e = ε}`.
- If `G_=` is non-empty: `I_0` can have internal edges with `τ = ε`, giving
  score = 1 at first step. The greedy CANNOT select these vertices.
- BUT: vertices NOT adjacent to any `τ = ε` edge have score < 1 (from edges
  with `τ < ε`). The greedy selects from these.
- If ALL internal edges have `τ = ε`: the graph is highly constrained
  (regular within `I_0`). The proof should handle this via a separate
  Case 2b' argument, or by perturbing `ε` slightly.

**Alternatively:** Redefine H1 as `τ < ε` (strict). This excludes the
boundary but changes the Turán bound by at most one vertex.

## Proof sketch for general graphs

**Claim:** Under H1 (strict: `τ < ε`), H2-H4, with `c_step = 1/3`, the
barrier greedy runs for `T = (1/3)εn` steps with `min_v ‖Y_t(v)‖ < 1`
at each step.

**Proof idea:**

1. **First step (t=0):** All scores = 0. Select any vertex. ✓

2. **Step t (within horizon):** Let `v* = argmin_v score_t(v)` in `R_t`.
   We claim `score_t(v*) < 1`.

3. **Key quantity:** `score_t(v) = ‖B_t^{1/2} C_t(v) B_t^{1/2}‖`.

4. **For K_k (extremal symmetric case):** All vertices have the same score
   `≈ (t+1)/(kε)`. At `t = kε/3`: score ≤ 5/6 < 1. ✓

5. **For general graphs:** The minimum score is at most the K_k score
   (by a symmetrization/rearrangement argument: asymmetry can only
   decrease the minimum). Therefore `min_v score_t(v) ≤ 5/6 < 1`.

**The missing piece** is step 5: the symmetrization argument showing K_k is
extremal. This is the remaining theorem needed to close GPL-H.

## Empirical support

| n range | nontrivial rows | max avg_score | max min_score |
|---------|----------------|---------------|---------------|
| 8-24 | 49 | 0.741 | 0.741 |
| 8-48 | 177 | 0.741 | 0.668 |
| 8-64 | 441 | 0.741 | 0.668 |

The worst avg_score (0.741 = 20/27) does NOT grow with n.

## Open question

> **Symmetrization lemma (open):** Among all graphs on `k` vertices with
> leverage scores satisfying H1 (τ ≤ ε), does `K_k` maximize
> `min_v ‖Y_t(v)‖` at each step?
>
> Equivalently: does symmetry maximization (replacing the graph with its
> "most symmetric" version with the same leverage parameters) increase
> the minimum score?

If yes, GPL-H follows from the K_k formula with θ = 5/6, c_step = 1/3.

## Files

- `data/first-proof/problem6-kk-extremal-analysis.md` — this document
- `data/first-proof/problem6-proof-attempt.md` — full proof (updated with
  ratio certificate and GL-Balance)
- `scripts/verify-p6-gpl-h-aggregate-ratio.py` — aggregate ratio diagnostics
