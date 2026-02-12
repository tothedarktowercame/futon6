# Problem 6: Direction A Probe Report (Strongly Rayleigh on Edge Indicators)

Date: 2026-02-12
Author: Codex

## Direction A: Strongly Rayleigh on edge indicators -> vertex-star loads

**What was tried:**
Built a dedicated probe to test the Direction A transfer question:
can an SR distribution on cross-edge indicators induce usable control on
`Y_t(v)=B_t^{1/2} C_t(v) B_t^{1/2}` star loads?

Added script:
- `scripts/verify-p6-gpl-h-direction-a.py`

At each Case-2b step (along the existing barrier-greedy trajectory), the script:
1. Constructs cross-edge atoms `A_e^{(t)}` between `S_t` and `R_t`.
2. Computes deterministic active star norms (`full_score_v = ||Y_t(v)||`).
3. Samples SR-like edge laws and induced sampled star norms:
   - `bern_p50`: product Bernoulli `p=0.5` (SR baseline)
   - `bern_deg`: product Bernoulli `p_e = 1/deg_R(v)` (SR degree-normalized)
   - `ust_forest`: Wilson uniform spanning forest on cross-graph components (SR)
4. Measures transfer ratios, e.g. `full_score_v / E[sampled_score_v]`.

Commands run:
1. `python3 scripts/verify-p6-gpl-h-direction-a.py --nmax 20 --eps 0.12 0.15 0.2 0.25 0.3 --samples 120 --c-step 0.5`
2. `python3 scripts/verify-p6-gpl-h-direction-a.py --nmax 32 --eps 0.1 0.12 0.15 0.2 0.25 0.3 --samples 80 --c-step 0.5`
3. `python3 scripts/verify-p6-gpl-h-direction-a.py --nmax 32 --eps 0.1 0.12 0.15 0.2 0.25 0.3 --samples 60 --c-step 0.5`

---

**What happened:**

Across the largest run (`n<=32`):
- Case-2b instances analyzed: 97
- Step-level transfer rows: 390

Key signal by measure:

1. `bern_p50` (product SR baseline)
- Highly degenerate for min-star signal (many rows with `E[min sampled] ~ 0`).
- Nondegenerate rows still show poor transfer (`median full_min / E[min] ~ 25`,
  90% ~ 60).
- Verdict: not viable for GPL-H bridge.

2. `bern_deg` (product SR degree-normalized)
- Better than p=0.5 but still unstable in hard steps.
- Nondegenerate `full_min / E[min]`: median ~1, 90% ~2.86, max ~31.96.
- Per-vertex worst transfer can exceed factor 2.6.
- Verdict: better heuristic, still structurally unreliable.

3. `ust_forest` (SR via spanning-tree/forest law)
- Best behavior by far.
- No degenerate rows (`E[min sampled] > 0` in all rows).
- `full_min / E[min]`: median 1.00, 90% ~1.03, max ~1.77.
- Per-vertex worst ratio `full_score_v / E[sampled_score_v]`: median 1.00,
  90% ~1.39, max ~1.75.
- Verdict: strong empirical transfer hint (constant-factor comparability) but not proof.

---

**Exact failure point (current):**

Direction A still fails at a theorem-level transfer step:

1. Available SR matrix concentration theorems control global sums like
   `||sum_e xi_e A_e||`.
2. GPL-H needs a *grouped minimum* statement
   `min_v ||sum_{e in star(v)} A_e|| <= theta`.
3. There is no current inequality that converts global SR concentration into
   this grouped-min norm control for fixed stars.

Even with the promising `ust_forest` empirical ratios, we still lack a proof of:

`||Y_t(v)|| <= kappa * E_mu ||Z_t(v; F)||` (uniform `kappa`) and, separately,
a theorem that forces `min_v E||Z_t(v;F)|| < 1/kappa` under H1-H4.

So the missing bridge is still exactly the edge-SR -> vertex-star transfer.

---

**Partial results:**

1. Implemented a reusable Direction A harness over real Case-2b trajectories.
2. Identified a sharp empirical separation between SR laws:
   - product SR measures are weak for this transfer,
   - UST-forest SR is qualitatively much closer to the needed behavior.
3. Quantitative narrowing:
   under UST-forest, transfer distortion stayed below ~1.8 in tested range.

This suggests a concrete subtarget:

> Prove a universal constant `kappa` for the UST-forest (or nearby SR law)
> such that star loads are comparable in operator norm to their SR-induced
> counterparts on GPL-H states.

---

**Surprises:**

1. UST-forest transfer is substantially tighter than product SR baselines.
2. Many hard rows are near equality (`ratio ~ 1`) under UST, indicating nontrivial
   structural alignment between cross-edge trees and vertex-star loads.

---

**New dead ends discovered:**

1. Product SR `p=0.5` is effectively unusable for grouped-min transfer.
2. Degree-normalized product SR improves but still exhibits large outliers,
   so it is not a robust bridge mechanism.

---

**Verdict:** NARROWS GAP TO edge-SR -> star-transfer lemma.

Direction A did not close GPL-H in this cycle, but produced a concrete,
better-focused next theorem target and a computational basis for prioritizing
UST-style SR structure over product SR baselines.
