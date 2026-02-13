# Problem 6 Cycle 4 Codex Verification

Date: 2026-02-13
Agent: Codex
Commit base: `ddd2244` (handoff), run on current `master`
Artifacts:
- Script: `scripts/verify-p6-cycle4-codex.py`
- Results: `data/first-proof/problem6-codex-cycle4-results.json`

## Executive Answer

- **Task 1 (literature):** no direct prior result found that matches the exact epsilon-light induced-subgraph condition `L_S \preceq \varepsilon L` with linear-size `|S| \ge c\varepsilon n` and universal `c`.
- **Task 3 (BSS adaptation):** plausible as a technique source, but the direct BSS existence/counting proof does not transfer cleanly because updates in vertex-greedy are not from a fixed rank-1 family.
- **Task 2 (effective-resistance symbolic route):** decomposition is structurally promising, but the proof currently breaks at controlling the cross term globally.
- **Task 4 (interlacing probe):** average-polynomial/interlacing checks are inconsistent in this setting; no robust tighter certificate beyond `\bar d` was obtained.
- **Task 5 (alpha probes):** strong numeric support for the per-vertex route (`\alpha_v < 1/2` everywhere, with huge margin), but this is still numerical evidence, not a formal proof.

---

## Task 1: Literature Search

### Query goal
Determine whether the epsilon-light subset problem or the key matrix inequality is already known under another name.

### Findings by source

1. **BSS edge sparsification (barrier method):** Batson-Spielman-Srivastava, 2012
   - Link: https://arxiv.org/abs/0808.0163
   - Relevance: canonical barrier potential method and Theorem 3.1 style step-selection argument.
   - Outcome: handles edge/rank-1 updates from a fixed family; does **not** directly solve vertex-induced constraint `L_S \preceq \varepsilon L`.

2. **Effective-resistance edge sampling:** Spielman-Srivastava, 2011
   - Link: https://arxiv.org/abs/0803.0929
   - Relevance: leverage/effective-resistance framework used in our notation.
   - Outcome: edge sparsifiers, not induced vertex subsets.

3. **Kadison-Singer / interlacing families:** Marcus-Spielman-Srivastava, 2015
   - Links:
     - https://arxiv.org/abs/1304.4132
     - https://arxiv.org/abs/1306.3969
   - Relevance: mixed characteristic polynomials, interlacing existence arguments.
   - Outcome: powerful partition/discrepancy machinery, but no direct statement equivalent to our induced-subgraph epsilon-light objective.

4. **Constructive/algorithmic KS-style sparsification:** Lee-Sun, 2017
   - Link: https://arxiv.org/abs/1707.03019
   - Relevance: algorithmic barrier framework for linear-size spectral sparsifiers.
   - Outcome: again edge/PSD-sum setting, not induced vertex subset closure.

5. **Vertex sparsifiers (cut/spectral via Schur complements):**
   - Moitra, 2010: https://arxiv.org/abs/1006.4536
   - Goranci-Henzinger-Peng, 2019: https://arxiv.org/abs/1906.10530
   - Relevance: "vertex sparsification" terminology check.
   - Outcome: these preserve terminal behavior using Schur complements, different objective than induced `L_S \preceq \varepsilon L`.

6. **Subspace/PSD sparsification:**
   - Allen-Zhu-Liao-Orecchia, 2018: https://arxiv.org/abs/1807.06455
   - Harvey, 2013: https://arxiv.org/abs/1301.2434
   - Relevance: sparse sums of PSD matrices and subspace guarantees.
   - Outcome: closest technical ecosystem, but still no direct theorem resolving our specific induced-vertex condition.

### MathOverflow check

- Searched local MO dump (`se-data/mathoverflow.net/Posts.xml`) for literal terms like `vertex sparsifier`, `spectral sparsifier`, `induced subgraph`, `effective resistance`, `interlacing`, `Kadison`.
- Result: no direct post matching the exact epsilon-light induced-subgraph problem statement was found.

### Task 1 conclusion

No known off-the-shelf theorem was found that makes the remaining gap moot. This still looks **greenfield** (or at least not under a standard, easily discoverable label).

---

## Task 2: Symbolic Route via Effective Resistance

Target: prove
\[
\alpha = \frac{\operatorname{tr}(P_M F)}{\operatorname{tr}(F)} < \frac12
\]
using `z_{uv} = z_u - z_v` and resistance structure.

### What works

- Decomposition
  \[
  \|P_M(z_u-z_v)\|^2 = \|P_M z_u\|^2 + \|P_M z_v\|^2 - 2\langle P_M z_u, P_M z_v\rangle
  \]
  is exact and naturally separates `u \in S`, `v \in R` effects.
- Per-vertex projections on `R` are numerically tiny (Task 5), suggesting weak alignment of unselected vertices with `\operatorname{col}(M)`.

### Where it breaks

- The mixed term `-2\langle P_M z_u, P_M z_v\rangle` is not sign-controlled in a way that gives a uniform global `< 1/2` bound by current arguments.
- Crude inequalities (Cauchy/triangle) lose too much and do not recover the needed constant.

### Task 2 conclusion

Good structural decomposition, but no formal closure yet; blocker is a sharp global control of the cross term.

---

## Task 3: BSS Potential Adaptation

Question: can BSS-style barrier counting prove `\bar d < 1` directly for vertex selection?

### Positive alignment with BSS

- We still have an upper-barrier inverse `B=(\varepsilon I-M)^{-1}` and step quality metric `\operatorname{tr}(BC_v)`.
- The greedy criterion is exactly selecting low-score candidates from current remainder.

### Core mismatch with direct BSS transfer

1. **Update family is not fixed across time.**
   In BSS, candidate rank-1 updates are fixed vectors/atoms. Here each `C_t(v)` depends on current `S_t`, so all candidates mutate after every step.
2. **Updates are not rank-1 atoms.**
   Each selected vertex contributes a multi-edge PSD block whose structure changes with trajectory.
3. **Counting/averaging lemma does not port verbatim.**
   BSS existence proofs average over a fixed pool with fixed total mass constraints; our pool is trajectory-dependent.

### Task 3 conclusion

BSS remains the best technique template, but a direct theorem-level transplant of Theorem 3.1 is currently blocked by non-stationary vertex-block updates.

---

## Task 4: Numerical Interlacing Probe

Suite: `K_40`, `K_80`, `ER_60(p=0.5,seed=42)`, `Barbell_40`, `Star_40`, `Grid_8x5`; `\varepsilon \in \{0.2,0.3,0.5\}`.

### Results summary

- Steps with finite root/trace ratio: **48**
- `Q` real-rooted failures: **35** (out of 117 total steps)
- Random partition interlacing pass rate: **703/1170 = 0.6009**
- Max `\lambda_{\max}(Q)/\bar d`: **1.0000000000000009** (numerical equality)
- Count with `\lambda_{\max}(Q) \le \bar d`: **48/48** among finite-ratio steps

### Interpretation

- The expected strict improvement from interlacing did **not** materialize in this probe.
- Empirically, where computable/reliable, largest root is essentially equal to the trace bound, not better.

---

## Task 5: Effective-Resistance / Alpha Probes

### Global summary

- Vertex rows: **6768**
- Cross-edge rows: **18721**
- `\max \alpha_v`: **0.0186538** (Barbell_40, `\varepsilon=0.5`, `t=13`)
- `\max \alpha_{uv}`: **0.4930677** (ER_60, `\varepsilon=0.5`, `t=10`, edge `(3,33)`)
- `\alpha_v < 1/2` violations: **none**

### Correlations

- corr(`\alpha_v`, min effective resistance to `S`): **-0.1739**
- corr(`\alpha_v`, mean effective resistance to `S`): **0.2009**
- corr(`\alpha_{uv}`, `\tau_{uv}`): **-0.6680**

### Horizon `x = ||M||/\varepsilon`

- K_n follows expected `x \approx t/(n\varepsilon)` behavior at horizon.
- Non-K_n can exceed `1/3` at horizon:
  - Barbell_40: `x=0.375` (`\varepsilon=0.2`), `x=0.35` (`\varepsilon=0.5`)

### Task 5 conclusion

The per-vertex alignment route is numerically very strong (`\alpha_v` far below `1/2`), but converting that into a formal theorem still needs a symbolic bridge from vertex-level to edge-level aggregate.

---

## Final Status for Cycle 4

- **Task 1:** complete (no known direct prior closure found).
- **Task 2:** partial (identified exact blocker).
- **Task 3:** partial (identified exact non-transfer point from BSS).
- **Task 4:** complete numerical probe (no robust interlacing gain).
- **Task 5:** complete numerical probe (strong empirical support for low alignment).

Most actionable next move remains: prove a trajectory-stable inequality converting tiny `\alpha_v` behavior into `\alpha < 1/2` at the cross-edge aggregate level.
