# Problem 6 Cycle 5b Codex Verification

Date: 2026-02-13  
Agent: Codex  
Base handoff: `data/first-proof/problem6-codex-cycle5b-handoff.md`  
Artifacts:
- Results JSON: `data/first-proof/problem6-codex-cycle5b-results.json`
- Extra stress data: `data/first-proof/problem6-codex-cycle5b-randomstress.json`
- Extra subset probe: `data/first-proof/problem6-codex-cycle5b-subsetprobe.json`
- Extra exhaustive small-n scan: `data/first-proof/problem6-codex-cycle5b-exhaustive-small.json`

## Executive Answer

- I did **not** find a known theorem that directly solves the Cycle 5b sub-lemma
  \
  `dbar0 = tr(F)/(r eps) < 1`
  \
  under the exact induced-subgraph setup used in Problem 6.
- I translated the sub-lemma into a clean UST statement:
  \
  `tr(F) = E[ number of UST edges crossing (S,R) ]`.
  \
  This is correct and useful, but I did not find a literature bound that directly implies `< r eps` in full generality.
- I obtained a **conditional proof route**: if `S` is taken as the prefix of vertices with smallest leverage degree in `I0`, then for `t <= floor(eps m/3)` one gets `dbar0 < 1`.
- Numeric evidence remains strongly supportive:
  - prior Cycle 5: max `dbar0 = 0.755` over 678 steps,
  - Cycle 5b random stress: max `dbar0 = 0.827` over 560 runs,
  - no `dbar0 > 1` found.
- Caveat: a knife-edge equality `dbar0 ~= 1` can occur when leverage hits `tau_e = eps` exactly (strict inequality needs strict-light slack tracked explicitly).

## Task 1: Literature Search (targeted)

### What I looked for

1. Alternate names of the problem (`L_S <= eps L` for induced vertex subset).  
2. Known cut-leverage / UST-cut expectation upper bounds that would imply `dbar0 < 1`.  
3. Matrix inequality shortcut for `tr(MF) <= (1/2)||M|| tr(F)` style claims.  
4. Schur-complement/effective-resistance tools that might imply the missing bound.

### Findings

No direct closure found. Closest ecosystems are:

- Effective-resistance sparsification (edge sampling), not induced vertex subsets:
  - Spielman-Srivastava, *Graph Sparsification by Effective Resistances* (2008), https://arxiv.org/abs/0803.0929
- Barrier/potential methods (edge atoms), template only:
  - Batson-Spielman-Srivastava, *Twice-Ramanujan Sparsifiers* (2008), https://arxiv.org/abs/0808.0163
- UST transfer-current / determinantal machinery:
  - Burton-Pemantle, *... via Transfer-Impedances* (1993), https://arxiv.org/abs/math/0404048
  - Lyons, *Determinantal probability measures* (2002), https://arxiv.org/abs/math/0204325
- Vertex sparsifier literature is about terminal/Schur-complement objectives, not induced `L_S <= eps L`:
  - Moitra et al., *Vertex Sparsifiers and Abstract Rounding Algorithms* (2010), https://arxiv.org/abs/1006.4536
  - Durfee-Gao-Goranci-Peng, *Fully Dynamic Spectral Vertex Sparsifiers...* (2019), https://arxiv.org/abs/1906.10530
  - Goranci-Henzinger-Peng, *Dynamic Effective Resistances and Approximate Schur Complement...* (2018), https://arxiv.org/abs/1802.09111

Secondary/local MO references recovered from local dump:

- https://mathoverflow.net/questions/81251/number-of-spanning-trees-which-contain-a-given-edge
- https://mathoverflow.net/questions/100209/probability-of-an-edge-appearing-in-a-spanning-tree

These support standard identities (edge-in-UST probability via effective resistance / Kirchhoff formulations), but do not give the needed cut bound directly.

## Task 3 + Task 5: Sub-lemma proof attempt

Let `I0 = S disjoint-union R`, `m=|I0|`, `t=|S|`, `r=|R|`, and

\[
\bar d^0 = \frac{\operatorname{tr}(F)}{r\varepsilon}
= \frac{1}{r\varepsilon}\sum_{e\in\delta(S,R)}\tau_e.
\]

### 1) UST translation (successful, but incomplete for closure)

Using `tau_e = P[e in T]` for `T` a random spanning tree,

\[
\operatorname{tr}(F)=\sum_{e\in\delta(S,R)}\tau_e
=\mathbb{E}\big[|\delta_T(S,R)|\big],
\]

so

\[
\bar d^0 < 1 \iff
\mathbb{E}\big[|\delta_T(S,R)|\big] < r\varepsilon.
\]

This is a clean reformulation. The unresolved step is a universal upper bound on that expected cut-tree-edge count in terms of `r eps`.

### 2) Foster decomposition identity (successful bookkeeping)

For edges internal to `I0` split as `S`-internal (`tau`), cross (`F`), and `R`-internal (`L_R`):

\[
\tau + F + L_R \le m-\kappa,
\]

by induced-subgraph Foster (with `\kappa` components of `G[I0]`). This is correct, but still needs a lower bound on `L_R` (or equivalent) to force `F < r eps`.

### 3) Conditional closure via ordered leverage-degree prefix (successful, conditional)

Define leverage degree on `I0`:

\[
\ell_v := \sum_{u\in I_0,\ u\sim v}\tau_{uv}.
\]

By induced Foster:

\[
\sum_{v\in I_0}\ell_v \le 2(m-1).
\]

If `S` is the size-`t` prefix of vertices sorted by nondecreasing `\ell_v`, partial averages give

\[
\sum_{u\in S}\ell_u \le \frac{2t(m-1)}{m}.
\]

Also

\[
F = \sum_{u\in S}\ell_u - 2\tau \le \sum_{u\in S}\ell_u.
\]

Hence

\[
\bar d^0 = \frac{F}{r\varepsilon}
\le \frac{2t(m-1)}{m\varepsilon(m-t)}.
\]

For `t <= floor(eps m/3)`:

\[
\bar d^0
\le \frac{2(m-1)}{3(m-t)} < 1.
\]

So this route proves `dbar0 < 1` **for that ordered-prefix selection rule**.

### Remaining logical gap

Current proof chain still needs one of:

1. show barrier-greedy trajectory is controlled by (or dominated by) that low-`\ell` prefix route, or
2. explicitly switch the construction to the low-`\ell` rule in the final theorem assembly.

Without that bridge, we have a strong conditional proof but not a full trajectory-independent closure.

## Numerical cross-checks (Cycle 5b additions)

### A) Broad random stress (`problem6-codex-cycle5b-randomstress.json`)

- 560 runs (ER/regular/bipartite/tree families, `n` up to 120, `eps in {0.1,0.2,0.3,0.5}`)
- max observed `dbar0 = 0.827327...`
- no `dbar0 >= 1`

### B) Random subset probe inside `I0` (`problem6-codex-cycle5b-subsetprobe.json`)

- 24 graph/eps rows (subset scan not restricted to greedy order)
- max observed `dbar0 = 0.734161...`
- no `dbar0 >= 1`

### C) Exhaustive small-n subset scan (`problem6-codex-cycle5b-exhaustive-small.json`)

- 72 rows across small graphs (`n<=20`), all subsets up to horizon
- no `dbar0 > 1`
- worst case: `dbar0 ~= 1` at `(K_10, eps=0.2, t=1)`

Interpretation: this is a knife-edge case where leverage can numerically hit `tau_e = eps`. If strict `<1` is required, strict-lightness (`tau_e < eps`) or an explicit slack convention should be encoded.

## Final Status for Cycle 5b

- **Literature shortcut found?** No.
- **Exact sub-lemma fully closed from known results?** Not yet.
- **Conditional proof route obtained?** Yes (ordered low-leverage prefix gives `dbar0 < 1`).
- **Empirical status of target inequality?** Strong support; no `dbar0 > 1` found in Cycle 5 + 5b scans.
- **Most actionable next proof step:** connect barrier-greedy trajectory to the ordered low-`\ell` prefix bound (or adopt that rule in the formal construction).
