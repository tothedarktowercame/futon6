# Problem 6 Cycle 5 Codex Verification

Date: 2026-02-13
Agent: Codex
Base handoff: `data/first-proof/problem6-codex-cycle5-handoff.md`

Artifacts:
- Script: `scripts/verify-p6-cycle5-codex.py`
- Results JSON: `data/first-proof/problem6-codex-cycle5-results.json`

## Executive Summary

Cycle 5 confirms the relaxed route is viable, with one important correction:

1. `rho_1 < 1` holds on all tested vertex-induced runs (max `rho_1 = 0.493976`).
2. `alpha < 1` holds on all tested vertex-induced runs (max `alpha = 0.493976`).
3. `dbar0 <= 2/3` is **not** true at all steps (max `dbar0 = 0.755163`).
4. `c_needed >= 1` is **not** true at all finite-`x` steps (minimum finite `c_needed = 0.957162`).
5. Despite (3)-(4), there is still a large uniform window for closure:
   - `max rho_1 = 0.493976`
   - `min finite c_needed = 0.957162`
   - so any `c0` in `(0.493976, 0.957162)` works uniformly on tested data.

This means the "houseboat" relaxation (`rho_1 < 1` instead of `< 1/2`) remains strongly supported, but the auxiliary claim "any `c < 1` suffices" is not universally true without additional conditions on `(x, dbar0)`.

---

## Task 1: Assembly With `c < 1`

### Formula check
At each step we used
\[
\bar d \le \bar d^0 \frac{1-x+cx}{1-x},\qquad x=\|M\|/\varepsilon,
\]
which is equivalent to
\[
c < c_{\text{needed}}:=\frac{(1-x)(1/\bar d^0-1)}{x} \quad (x>0).
\]
For `x=0`, the condition is simply `dbar0 < 1`.

### Empirical outcomes
Across 60 runs / 678 steps:

- `dbar0<=2/3 all`: **False**
- `max dbar0`: **0.755163** (ExpanderProxy_Reg_100_d6, eps=0.5, t=16)
- `c_needed>=1 all`: **False**
- `min finite c_needed`: **0.957162** (Reg_100_d50, eps=0.5, t=16)
- `max x`: **0.333333** (K_100, eps=0.3, t=10)

Witness with finite `c_needed < 1`:
- Graph: `Reg_100_d50`, `eps=0.5`, `t=16`
- `dbar0=0.701508`, `x=0.307740`, `c_needed=0.957162`, `rho1=0.219556`

So the statement "confirm `c_needed >= 1` everywhere" is refuted.

### Uniform `c0` question
Let `c0` satisfy both:
1. `rho1 < c0`
2. `dbar0 * (1-x+c0*x)/(1-x) < 1`

From the scan:
- `max rho1 = 0.4939759`
- `min finite c_needed = 0.9571623`

Hence a uniform `c0 < 1` exists, with feasible open interval:
\[
0.4939759 < c_0 < 0.9571623.
\]
A safe midpoint used in the JSON summary is `c0 = 0.7255691`.

---

## Task 2: Symbolic `rho_1 < 1` / `alpha < 1`

### Key identity and reduction
\[
\rho_1 = \frac{\operatorname{tr}(MF)}{\|M\|\operatorname{tr}(F)}
\le \frac{\operatorname{tr}(P_MF)}{\operatorname{tr}(F)} = \alpha,
\]
because `M <= ||M|| P_M` for PSD `M`.
Thus it is enough to show `alpha < 1` (when `tr(F)>0`).

### Vertex-induced proof sketch for `alpha < 1`
Let `b_uv = e_u - e_v` and `z_uv = L^{+/2} b_uv`.

- `col(M)` is the span of internal-edge vectors `z_ab` with `a,b in S`.
- Every corresponding incidence vector `b_ab` has support contained in `S`.
- For any cross edge `(u,v)` with `u in S`, `v in R`, vector `b_uv` has nonzero coordinate on `v \notin S`, so `b_uv` is not in the span of internal `b_ab` vectors.
- On `im(L)`, map `L^{+/2}` is invertible; therefore linear non-membership is preserved:
  `z_uv \notin col(M)`.
- Hence each cross edge contributes strictly positive perpendicular mass:
  `||P_{M^\perp} z_uv||^2 > 0`.

Since
\[
F=\sum_{(u,v)\in E_{cross}} z_{uv}z_{uv}^\top,
\]
we get
\[
\operatorname{tr}((I-P_M)F)=\sum_{cross} ||P_{M^\perp} z_{uv}||^2 > 0
\]
whenever cross edges exist (`tr(F)>0`). Therefore
\[
\alpha = 1 - \frac{\operatorname{tr}((I-P_M)F)}{\operatorname{tr}(F)} < 1,
\]
and so `rho_1 < 1`.

### Numerical confirmation
- `rho1<1 all`: True
- `alpha<1 all (vertex-induced)`: True
- maxima coincide at K_n extremal trajectory (`0.493976`).

---

## Task 3: Expanded Graph Family Scan

Families run (eps in `{0.1,0.2,0.3,0.5}`):
- `K_100`, `K_200`, `K_500` (K_200/K_500 analytic mode)
- `Reg_100_d3`, `Reg_100_d10`, `Reg_100_d50`
- `ER_100_p0.1`, `ER_100_p0.3`, `ER_100_p0.5`
- `Tree_prufer_100`, `Path_100`, `Star_100`
- `K_50_50`, `BipRand_50_50_p0.2`
- `ExpanderProxy_Reg_100_d6`

Key answers:
1. Tightest relaxed margin (`c_needed-rho1`) is still large:
   - minimum `0.51` (K_500, eps=0.3, t=50).
2. Family with `c_needed < 1` exists:
   - `Reg_100_d50`, eps=0.5, t=16 (`c_needed=0.957162`).
3. `x_t` stays far from 1 in this scan:
   - max `x = 1/3`.
4. `dbar0 <= 2/3` at all steps is false:
   - multiple witnesses (regular/bipartite/ER/expander at larger t).

---

## Task 4: Low-Rank Scan

Per-rank aggregate (`alpha` over all step rows):
- rank 0: `alpha_max=0`
- rank 1: `alpha_max=0.25`
- rank 2: `alpha_max=1/3`
- ...
- highest observed: rank 82 with `alpha_max=0.4939759` (K_500 trajectory)

No observed approach to 1 under vertex-induced greedy trajectories.

Rank-1 exact formula check:
\[
\alpha = \frac{\sum_{cross} \langle z_{uv}, z_e\rangle^2}{\|z_e\|^2 \sum_{cross}\|z_{uv}\|^2}
\]
(where `M = z_e z_e^T` rank-1) matched numerically with
- max absolute discrepancy: `7.633e-17`.

---

## Task 5: Edge-Partition Relaxation

Random edge partitions (`p=0.1..0.9`, 16 trials each `p`) were run for `n<=100` graphs.

Result:
- `alpha < 1` is **not** true for arbitrary edge partitions.
- Many cases hit `alpha ~= 1` (995 cases at tolerance level).
- Typical witness: full-rank internal part (`rank(M)=n-1`) gives `P_M=Pi`, so for `F` in `im(Pi)`, `alpha=1`.

Conclusion: vertex-induced structure is essential; the relaxed `alpha<1` argument does not extend to arbitrary edge partitions.

---

## Task 6: Random Selection Probe

For each run we sampled random subsets `S` of size horizon `T` (80 trials in generic mode).

Observed maxima:
- max random `alpha_mean`: `0.46875` (K_100, eps=0.5)
- max random `alpha_max`: `0.485211` (ER_100_p0.3, eps=0.5)
- max random `rho1_max`: `0.46875` (K_100, eps=0.5)

So random selection also stayed below 1/2 in this scan, though this remains empirical.

---

## Bottom Line

- The Cycle 5 relaxation target (`rho_1 < 1`) is strongly supported and admits a clean vertex-induced proof path via `alpha<1`.
- The specific intermediate claim "`dbar0 <= 2/3` at all steps" is false.
- The stronger claim "any `c<1` suffices" is also false in full generality of scanned steps; finite-step `c_needed` can dip below 1.
- Nevertheless, there is still a large and uniform feasible `c0` interval in tested data, so the relaxed assembly strategy remains viable.
