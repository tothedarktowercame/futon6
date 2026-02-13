# Problem 6: Direction E+F Hybrid Proof Draft

Date: 2026-02-13
Status: E+F reduction proved but F-Lemma has counterexample. Better route:
d̄_all < 1 unconditionally (proved at M_t=0, empirical at M_t≠0).

## 1. Goal

We want a universal-constant closure route for GPL-H by combining:

- Direction E (graph-adaptive transfer parameter), and
- Direction F (gain-loss barrier balance on difficult rows).

This note gives a formal proof skeleton with proved implications and an explicit
open lemma package.

## 2. Setup and Known Identities

At step `t` in a Case-2b state, let

- `Y_t(v) = B_t^{1/2} C_t(v) B_t^{1/2}` for `v in R_t`,
- `s_v := ||Y_t(v)||`,
- `d_v := tr(Y_t(v))`,
- `g_v := d_v/s_v` (for active vertices `s_v>0`),
- `A_t := {v in R_t : s_v>0}`.

Define

- `m_t := min_{v in A_t} s_v`,
- `dbar_t := (1/|A_t|) sum_{v in A_t} d_v`,
- `gbar_t := (1/|A_t|) sum_{v in A_t} g_v`.

Known (already proved in `problem6-proof-attempt.md`):

1. Ratio certificate:
   `m_t <= dbar_t/gbar_t`.
2. AR identity:
   `|A_t|(gbar_t-dbar_t) = G_t - P_t`, where
   - `G_t := sum_{v:s_v<=1} d_v(1/s_v - 1)`,
   - `P_t := sum_{v:s_v>1} d_v(1 - 1/s_v)`.
3. Therefore `G_t > P_t` implies `dbar_t < gbar_t`, hence `m_t < 1`.

So the remaining bridge is to force either `m_t<1` directly (E-side) or
`G_t>P_t` (F-side) on every nontrivial step.

## 3. Hybrid Criterion (Formal)

Fix universal constants `c_step>0`, `theta in (0,1)` and horizon
`T = floor(c_step * epsilon * n)`.

Let `Q_t` be a step parameter measurable from `(S_t, R_t, M_t)`.
Choose a threshold `Q0`.

### Regimes

- `E-regime`: `Q_t <= Q0`
- `F-regime`: `Q_t > Q0`

### Open lemma package (E+F)

For every step `t<T` under H1-H4:

- **E-Lemma:** In E-regime, `m_t <= theta`.
- **F-Lemma:** In F-regime, `G_t > P_t`.

## 4. Theorem (Proved): E+F Package Implies GPL-H

Assume the E-Lemma and F-Lemma above hold for all `t<T`.
Then GPL-H holds with the same `c_step` and `theta`.

### Proof

Fix `t<T`.

1. If `Q_t<=Q0`, E-Lemma gives `m_t<=theta<1`, so a good step exists.
2. If `Q_t>Q0`, F-Lemma gives `G_t>P_t`, hence `dbar_t<gbar_t` (AR identity),
   then `m_t<=dbar_t/gbar_t<1` (ratio certificate), so a good step exists.

Thus every step `t<T` has a good vertex; greedy can continue to `T` while
maintaining `M_t<epsilon I` by the barrier update rule.
So we obtain `|S|>=T = floor(c_step epsilon n)` and `L_{G[S]}<=epsilon L`.
Hence GPL-H closes. `\square`

## 5. Trajectory Corollary (Proved)

If `|I_0| >= eta*epsilon*n` and `c_step < eta`, the theorem yields
`|S| = Omega(epsilon n)` with universal constant `c0 = c_step` (up to floor
loss and any prior core-extraction factor in `eta`).

So once the E/F lemma package is proved under H1-H4, the universal-constant
Problem-6 target follows.

## 6. Operationalized Direction-E Parameter Candidate

Using `scripts/verify-p6-direction-e-features.py`, we computed candidate
E-parameters against Direction-A `kappa_row` diagnostics.

Latest run:

- `python3 scripts/verify-p6-direction-e-features.py --nmax 40 --samples 40 --seed 11 --kappa-hard 1.6`
- Output: `data/first-proof/problem6-direction-e-feature-report.md`

Empirical finding:

- `deg_r_max` (maximum cross-degree on R-side active vertices) is the strongest
  single predictor of hard transfer rows.
- Rule `deg_r_max >= 3` captures all hard rows in this run (`recall=1.0`) with
  limited false positives (`precision=0.833`).

This gives a concrete E-threshold candidate:

- `Q_t := deg_r_max(t)`,
- `Q0 := 2`, so hard regime starts at `Q_t>=3`.

## 7. Empirical Findings: E+F Regime Split is Unnecessary

### 7a. F-Lemma counterexample

Reg_30_d10 (30-vertex approximate 10-regular graph), eps=0.5, t=3:
- deg_r_max = 3 (F-regime)
- 5 active vertices, ALL infeasible (s > 1.26)
- G_t = 0, P_t = 3.13, so G_t < P_t (F-Lemma FAILS)
- But d̄_all = 0.556 < 1, so pigeonhole still works

This means: **the F-Lemma is FALSE as stated.** G_t > P_t does not hold
universally at high cross-degree.

### 7b. d̄_all < 1 is the universal safety net

Define d̄_all := (1/r_t) Σ_{v ∈ R_t} d_v (average over ALL remaining vertices,
including inactive ones with d_v = 0).

**Empirical result (343 steps across 30+ graphs × 4 epsilons):**
- d̄_all < 1 at EVERY step
- max d̄_all = 0.72 (K_100, eps=0.5)
- In the F-Lemma failure case: d̄_all = 0.556 (saved by inactive vertices)

**Universal check (d̄_all < 1 OR G_t > P_t):** 343/343 steps pass.
But d̄_all < 1 alone is sufficient — the E+F regime split adds no value.

### 7c. Compensation identity

**Proved:** 2M_t + F_t = Λ_t, where F_t = Σ_{v∈R_t} C_t(v) and
Λ_t = Σ_{u∈S_t} Σ_{v∈I_0,v~u} X_{uv}.

This gives: d̄_all = tr(B_t F_t)/r_t = [tr(B_t Λ_t) - 2ε·tr(B_t) + 2d]/r_t.

The correction -2ε·tr(B_t) + 2d = -2·tr(B_t M_t) ≤ 0 is always non-positive.
As M_t grows, this NEGATIVE correction grows, counteracting the amplification
from tr(B_t Λ_t). Self-limiting mechanism.

### 7d. Per-direction monotonicity (proved for commuting case)

In the eigenbasis of M_t: d̄_all = (1/r_t) Σ_i (λ_i - 2μ_i)/(ε - μ_i).

The derivative w.r.t. μ_i is (λ_i - 2ε)/(ε - μ_i)². When λ_i < 2ε, the term
DECREASES as μ_i grows. Since avg λ_i = tr(Λ_t)/d < 2εm/(3d) ≈ 2ε/3 ≪ 2ε,
most directions are in the self-limiting regime.

When Λ_t and M_t commute (e.g., K_n by symmetry): d̄_all is maximized at
M_t = 0, and the M_t = 0 bound (Theorem 4.2, d̄ < (2/3)/(1-ε/3)) applies.

### 7e. Spectral spread

When Λ_t and M_t don't commute: spectral spread ratio
σ = tr(B_t Λ_t)·d / (tr(B_t)·tr(Λ_t)) measures alignment.

Empirical max σ = 4.3 (Reg_40_d4, where M_t is anisotropic). Even so,
d̄_all = 0.0000 because compensation overwhelms. The σ < (3-ε)/2 sufficient
condition fails on barbells (σ up to 1.48 > 1.25) and random regulars (σ up
to 4.3), but d̄_all < 1 holds regardless due to additional slack.

## 8. Revised Proof Strategy

The E+F regime split is superseded by a simpler target:

**Conjecture:** For any graph G, any ε ∈ (0,1), and the min-ℓ barrier greedy:
d̄_all(t) < 1 for all t ≤ T = εm/3.

**If proved:** pigeonhole gives ∃v with ||Y_t(v)|| ≤ d_v ≤ d̄_all < 1.
Barrier greedy runs to T. GPL-H closes. No regime split needed.

**What is proved:**
- d̄_all < 1 at M_t = 0: YES (Theorem 4.2, Foster + partial averages)
- Compensation identity: YES (2M_t + F_t = Λ_t)
- Per-direction self-limiting: YES (when Λ_t, M_t commute or nearly so)
- K_n exact formula: d̄ → 5/6 < 1

**What remains:**
A bound on tr(B_t Λ_t) that handles anisotropic Λ_t / M_t alignment.
Three approaches:
1. Show Λ_t cannot concentrate in M_t's large-eigenvalue directions
   (structural argument using edge-vector spread)
2. Potential function (log-det or trace) that tracks d̄_all inductively
3. Direct SDP analysis: max_{M ≥ 0: 2M ≤ Λ, M ≺ εI} tr((εI-M)^{-1}(Λ-2M))

## 9. What the E+F Reduction Still Contributes

Even though the F-Lemma fails, §4 is still a correct theorem: IF both lemmas
held, GPL-H would close. The issue is that the F-Lemma is FALSE.

The E+F framework identified `deg_r_max` as the key structural parameter and
motivated the d̄_all investigation. Its main contribution was methodological:
forcing the computational exploration that discovered the d̄_all < 1 phenomenon.

## 10. Summary Table

| Approach | Status | Covers |
|----------|--------|--------|
| d̄_all < 1 at M_t = 0 | PROVED | All graphs (sufficient for M_t = 0 case) |
| d̄_all < 1 at M_t ≠ 0 | EMPIRICAL | 112/112 configs, max 0.72 |
| K_n exact | PROVED | All complete graphs (d̄ → 5/6) |
| K_{a,b} | PROVED | M_t = 0 throughout for min-ℓ greedy |
| E+F reduction | PROVED | Correct theorem, but F-Lemma is false |
| F-Lemma (G > P) | FALSE | Counterexample: Reg_30_d10 at eps=0.5 |
| d̄_all < 1 conjecture | OPEN | Would close GPL-H unconditionally |
