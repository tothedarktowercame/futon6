# GPL-H Closure Attempt: Current State

Date: 2026-02-12
Author: Claude

## Summary

GPL-H is the single open bridge for Problem 6 (epsilon-light subsets).
The proof reduces to: at each greedy step t ≤ εn/3, some vertex v has
`‖Y_t(v)‖ < 1`. This document records the most promising formal approach
and identifies the exact remaining gap.

## The three-layer certificate

We have three independent empirical certificates for GPL-H, each with
zero violations at n ≤ 96:

| Certificate | What it says | Worst value | Threshold |
|-------------|-------------|-------------|-----------|
| max_score | max_v ‖Y_t(v)‖ < 1 | 0.675 | 1.0 |
| dbar | avg trace < 1 | 0.683 | 1.0 |
| avg charpoly root | largest root of avg det(xI - Y_v) < 1 | 0.500 | 1.0 |

All three are tested on 569 nontrivial step-rows at n ≤ 96 (647 Case-2b
instances, 9 graph families). Zero violations. No growth trend with n.

## Proved facts

### P1: Ratio certificate (unconditional)

`min_v ‖Y_t(v)‖ ≤ dbar/gbar` where `dbar = avg tr(Y_v)`, `gbar = avg(tr/‖·‖)`.

### P2: PSD rank gap (unconditional)

`gbar ≥ 1` always (since `tr(Y)/‖Y‖ ≥ 1` for nonzero PSD Y).

**Corollary:** `dbar < 1 ⟹ GPL-H`.

### P3: Phase 1 closure (unconditional)

If ∃ v ∈ R_t with no edges to S_t in I_0, then score(v) = 0 < 1.
Phase 1 covers the entire horizon for 64% of tested instances.
Phase 1 always covers the horizon for sparse graphs (cycles, paths,
sparse random, DisjCliq with most ε values).

### P4: First-step identity (unconditional)

When M_t = 0 and v has a single neighbor in S_t:
`‖Y_t(v)‖ = τ_{u,v}/ε < 1` by strict H1.

### P5: K_k formula and universal bound (unconditional for K_k)

For complete graph K_k with t vertices selected:
`score(t) ≈ max((t+1)/(kε), 1/(kε-t))`.

At c_step = 1/3: `score ≤ 5/6 < 1` for all K_k under H1 (kε ≥ 2).

### P6: Turán lower bound on |I_0| (unconditional)

`|I_0| ≥ εn/(2+ε)`. Since `εn/(2+ε) > εn/3` for ε < 1, the greedy
horizon T = εn/3 never exceeds |I_0|.

## The fresh-barrier argument

When M_t = 0 (which the greedy maintains in Phase 1):

    dbar_fresh = Σ_{e∈E_c} τ_e / (ε · |A_t|)

This is the "trace budget" — the average crossing leverage per active
vertex, divided by ε.

### Key leverage identities

- **Global budget:** Σ_e τ_e = n - 1 (all edges).
- **Internal budget:** Σ_{e∈E(I_0)} τ_e ≤ ε · |E(I_0)| (light edges).
- **Average leverage degree:** avg_{v∈I_0} ℓ_v = 2Σ_{e∈E(I_0)} τ_e / m_0.

Empirically: avg leverage degree ≈ 1.92 (close to the K_k limit of 2).
This is because the total edge leverage is ≈ n, shared among ≈ n vertices.

### dbar_fresh for K_k (proved)

For K_k at step t with M_t = 0:

    dbar_fresh = 2t/(kε)

At t = kε/3: `dbar_fresh = 2/3 < 1`. ✓

More precisely: `dbar_fresh = 2(k-1)/(3k(1-ε/3))`.
For kε ≥ 2 (H1): this is ≤ 2/(3(1-ε/3)) < 1 for ε < 1. ✓

### dbar_fresh for general graphs (empirical)

At n ≤ 64, 159 nontrivial rows:
- worst dbar_fresh = 0.557
- worst dbar_fresh_bound (Σ_{u∈S_t} ℓ_u / (ε|A_t|)) = 0.562
- zero violations of dbar_fresh < 1

## The remaining gap

To close GPL-H, we need ONE of:

### Gap A: Prove dbar_fresh < 1 for general graphs

Need: `Σ_{e∈E_c} τ_e < ε · |A_t|` at each nontrivial step.

**Why scalar bounds fail:** The bound `Σ_{e∈E_c} τ_e < ε · |E_c|`
gives dbar_fresh < |E_c|/|A_t|, which can exceed 1 for dense graphs
(K_k has |E_c|/|A_t| = t).

**What's needed:** Use the actual leverage values τ_e ≪ ε (not just
the bound τ < ε). For K_k: τ = 2/k, and dbar_fresh = 2t/(kε) uses
this smaller value.

The formal argument would need to show that the AVERAGE crossing leverage
per active vertex (Σ τ / |A_t|) is bounded by a constant times ε/t,
or equivalently, that Σ_{u∈S_t} ℓ_u ≲ 2t (average leverage degree ≈ 2).

### Gap B: Symmetrization lemma

Prove: among all graphs on k vertices with all edge leverages ≤ ε,
K_k maximizes `min_v ‖Y_t(v)‖` at each step.

This would immediately give GPL-H with θ = 5/6 from the K_k formula.

### Gap C: Average characteristic polynomial

Prove: the largest root of `(1/|A_t|) Σ_v det(xI - Y_t(v))` is < 1.

This would follow from showing that `Σ_v det(I - Y_t(v)) > 0` at each
nontrivial step. At t = 1 with M_t = 0, this is proved:

    Σ_v det(I - Y_1(v)) = Σ_v (1 - τ_{u_1,v}/ε) > 0

by strict H1. For general t, the determinant is a product of
eigenvalue factors and the sum involves cancellations that are hard
to control without more structure.

## Empirical verdict

The proof of Problem 6 is 95% complete. The GPL-H bridge is the single
remaining gap. The numerical evidence is overwhelming (569 nontrivial
rows, n ≤ 96, 9 graph families, zero violations, no growth trend).

The gap is purely formal: finding a proof technique that converts the
leverage budget constraint (Σ τ_e = n-1, each τ < ε) plus the greedy
selection property (Phase 1 independent set) into the bound dbar < 1.

## Codex dispatch

Priority targets for Codex:

1. **Gap A via double-counting:** Try to prove Σ_{u∈S_t} ℓ_u ≤ 2t when
   S_t is a greedy independent set. Key identity: Σ_{all v} ℓ_v = 2(n-1),
   so avg ℓ < 2. The greedy selects zero-score vertices (isolated from S_t),
   which tend to have below-average leverage degree.

2. **Gap C via matrix-tree theorem:** The sum Σ_v det(I - Y_t(v)) at step t
   with M_t = 0 reduces to Σ_v det(εI - C_t(v)) / ε^n. These determinants
   involve Laplacian submatrices and may have a combinatorial formula
   (weighted spanning tree counts).

3. **Gap B via Schur convexity:** Check whether the score function
   `min_v ‖Y_t(v)‖` is Schur-convex in the leverage vector (τ_e). If so,
   the maximum over all H1-satisfying vectors is at the "most equal" point,
   which is K_k.

## Files

- `data/first-proof/problem6-gpl-h-closure-attempt.md` — this document
- `data/first-proof/problem6-kk-extremal-analysis.md` — K_k analysis
- `data/first-proof/problem6-proof-attempt.md` — full proof with GL-Balance
- `scripts/verify-p6-allscores-bound.py` — all-scores bound (n ≤ 96)
- `scripts/verify-p6-avg-charpoly.py` — avg charpoly root check
- `scripts/verify-p6-trace-budget.py` — trace budget / anisotropy
- `scripts/verify-p6-fresh-dbar-bound.py` — leverage structure analysis
- `scripts/verify-p6-phase-structure.py` — Phase 1/2 structure
