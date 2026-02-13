# Problem 6: Direction E+F Hybrid Proof Draft

Date: 2026-02-13
Status: Reduction-to-lemmas proved; one hybrid bridge package remains open.

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

## 7. What Remains to Finish the Proof

To upgrade this draft to a full theorem, prove:

1. **E-Lemma at low cross-degree** (`deg_r_max<=2`):
   a uniform score bound `m_t<=theta_E<1` (or direct `dbar_t<1`) in this regime.
2. **F-Lemma at high cross-degree** (`deg_r_max>=3`):
   a deterministic gain-loss inequality `G_t>P_t` from H1-H4.

Once these two are proved, Section 4 closes GPL-H unconditionally.

## 8. Why This Is a Real Proof Step (Not Just Plan)

What is already fully proved here:

- exact logical reduction from GPL-H to a two-regime E/F lemma package,
- exact implication chain from those lemmas to stepwise good-vertex existence,
- trajectory closure to linear-size `epsilon`-light sets.

So the open part is now theorem-localized to two explicit inequalities tied to a
single concrete regime separator (`deg_r_max`).
