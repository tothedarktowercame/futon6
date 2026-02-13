# Problem 6 Cycle 9 Codex Handoff

Date: 2026-02-13
Agent: Claude
Script: `scripts/verify-p6-cycle9-codex.py`

## Context

Cycle 9 formalized five new lemmas (8-12), proved the Sparse Dichotomy
theorem, and conjectured the Strong Dichotomy. See:
- `data/first-proof/problem6-bridge-b-formalization.md` (Lemmas 8-12, theorems)
- `data/first-proof/problem6-solution.md` (Section 6, items n-p)

## What the script verifies

The script `verify-p6-cycle9-codex.py` replays the modified leverage-order
barrier greedy on the full base + adversarial test suite, and at each step:

### Task 1: Lemma 8 (Rayleigh-Monotonicity Matrix Bound)
- Checks eigenvalues of Pi_{I_0}: all should be <= 1.
- Checks eigenvalues of (I - M_t - F_t): all should be >= 0 (i.e., F_t <= I - M_t).
- For each eigenvector u_j of M_t: checks f_j = u_j^T F_t u_j <= 1 - lambda_j.
- Checks ||C_t(v)|| <= 1 for each v in R_t.

Priority: **HIGHEST** — this is the foundational matrix bound enabling the
dichotomy. Zero violations expected (these are proved lemmas, so violations
would indicate a bug in the script or numerical issues).

### Task 2: Lemma 11 (Rank)
- For each v in R_t with deg_S(v) >= 1: computes rank(Y_t(v)) via singular
  values and checks it equals deg_S(v).

Priority: **HIGH** — proved lemma, zero violations expected.

### Task 3: Lemma 12 (Projection Pigeonhole)
- For each eigenvector u_j of M_t: computes min_{v in R_t} u_j^T C_t(v) u_j
  and checks it <= (1 - lambda_j) / r_t.

Priority: **HIGH** — proved lemma. Reports worst-case ratio for tightness
analysis. If ratio approaches 1.0 for the dangerous eigenspace, the
pigeonhole is tight and the sub-gap may be hard to close algebraically.

### Task 4: Sparse Dichotomy
- Computes Delta(G[I_0]) (max degree of induced subgraph).
- Checks whether Delta < 3/eps - 1 (condition for Sparse Dichotomy).
- Reports the fraction of runs where Sparse Dichotomy applies.

Priority: **MEDIUM** — already proved. This just measures coverage.

### Task 5: Strong Dichotomy
- At each step: checks (A) exists v in R_t with deg_S(v) = 0, OR (B) dbar_t < 1.
- Records any counterexamples (steps where both fail).

Priority: **HIGHEST** — this is the main conjecture. A counterexample
would be a step where all R-vertices have deg_S >= 1 AND dbar >= 1.
Expected: 0 counterexamples (based on C8 data). A counterexample would
force us to strengthen the attack path.

### Task 6: Dense-Case Probes
- At steps where isolation fails (all R-vertices dominated): decomposes the
  barrier contribution by eigenspace (dangerous vs safe).
- For the vertex minimizing dangerous-direction contribution: checks if
  normY < 1 (feasible despite no isolation).
- Reports the max dbar at non-isolation steps.

Priority: **HIGH** — this is the key data for closing the sub-gap.
If max dbar at non-isolation steps is always < 1, the Strong Dichotomy
is confirmed and the sub-gap reduces to proving that domination implies
bounded amplification.

## Running

```bash
cd /home/joe/code/futon6
python scripts/verify-p6-cycle9-codex.py --seed 42 --eps 0.1 0.2 0.3 0.5
```

Optional: `--skip-adversarial` to run only base suite (faster).

## Expected outputs

- `data/first-proof/problem6-codex-cycle9-results.json` (detailed results)
- `data/first-proof/problem6-codex-cycle9-verification.md` (summary)

## What to look for in results

1. **All lemma violations should be 0.** If not, there's a numerical issue
   or a bug. Check the tolerance (1e-9) and report exact excess values.

2. **Strong Dichotomy counterexamples**: the critical field. If 0: the
   conjecture holds on this suite. If > 0: report the graph, eps, step,
   deg0_count (should be 0), and dbar (should be >= 1).

3. **Dense-case probes**: when isolation fails, what's the max dbar?
   If always < 1: pigeonhole closes the gap at those steps. Report the
   distribution of dbar at non-isolation steps.

4. **Lemma 12 worst ratio**: how close is the pigeonhole to tight?
   If the ratio (min_v u_max^T C_t(v) u_max) / ((1-lambda_max)/r_t)
   approaches 1.0, the bound is tight and there's no slack.

5. **Sparse Dichotomy coverage**: what fraction of runs have
   Delta(G[I_0]) < 3/eps - 1? This measures how much of the test suite
   is covered by the proved theorem vs. the conjectured extension.
