# Problem 6 Cycle 7 Codex Verification

Date: 2026-02-13
Agent: Codex
Base handoff: `data/first-proof/problem6-codex-cycle7-handoff.md`

Artifacts:
- Script: `scripts/verify-p6-cycle7-codex.py`
- Results JSON: `data/first-proof/problem6-codex-cycle7-results.json`

## Executive Summary

- Base suite runs: 116 (847 total steps)
- Adversarial suite runs: 32 (264 total steps)
- BMI empirical status: dbar < 1 on all tested steps (base=False, adversarial=True)
- Worst base dbar: 1.738589 at `Reg_100_d10` eps=0.5 t=16
- Base steps with dbar >= 1: 12
- Worst adversarial dbar: 0.925926 at `Adv_Barbell_40_40_b3` eps=0.2 t=5
- Adversarial steps with dbar >= 1: 0

## Task 1: Eigenvalue-Level BMI Computation

For every modified-greedy step, the script records:
- eigenvalues lambda_j of M_t on im(L)
- projections pi_j = u_j^T Pi_{I0} u_j
- contributions (pi_j-lambda_j)/(eps-lambda_j) and f_j/(eps-lambda_j)
- decomposition check pi_j-lambda_j = f_j + l_j

Consistency checks: max |dbar - sum_j f_j/(eps-lambda_j)/r| = 3.686e-14 (base), 5.551e-16 (adv)
Proposed pi-lambda expression mismatch: base=9.917e+00, adv=9.900e+00
Max decomposition error |(pi-lambda)-(f+l)| = 9.648e-14 (base), 2.365e-14 (adv)
pi_j <= 1 violations: base=0, adv=0

## Task 2: Direct BMI Proof Probes

Three upper-bound attempts were evaluated per step:
1. uniform pi_j <= 1 bound
2. LP-style bound using sum pi_j = tr(Pi_{I0}) and rank cap
3. Cauchy-Schwarz bound

Approach 1 proves all steps: False (failures=847)
Approach 2 proves all steps: False (failures=649)
Approach 3 proves all steps: False (failures=649)
These remain diagnostic bounds; none closes BMI universally on this scan.

## Task 3: K_n Extremality

Compared each step to K_m reference
dbar_Km(t) = (t-1)/(m eps - t) + (t+1)/(m eps)
with m=|I0| for the run.

Base suite max ratio dbar/dbar_Km: 2.279226 (horizon max 2.225604)
Base ratio violations (dbar/dbar_Km > 1): 97
Adversarial suite max ratio dbar/dbar_Km: 1.211612 (horizon max 1.211612)
Adversarial ratio violations (dbar/dbar_Km > 1): 25
Extremality (<=1) on all tested steps: base=False, adv=False

## Task 4: Stress Test (Adversarial Families)

Added families:
- near-bipartite complete K_{a,b} with uneven splits
- expander + pendant attachments
- weighted bipartite leverage concentration
- barbell variants with sparse bridges
- random graph with planted dense subgraph

Adversarial dbar<1 across all steps: True

## Task 5: BSS Potential Probe

Tracked Phi_t = tr((eps I - M_t)^{-1}) on im(L), and checked two identities:
- naive handoff candidate: Phi_{t+1}-Phi_t ?= tr(B_{t+1} C_t(v_t))
- resolvent identity: Phi_{t+1}-Phi_t = tr(B_{t+1} C_t(v_t) B_t)

Max naive-delta abs error: base=1.163e+02, adv=1.567e+01
Max resolvent-delta abs error: base=6.928e-13, adv=1.060e-13
Max telescoping abs error: base=0.000e+00, adv=0.000e+00
Result: the naive formula is false in general; the resolvent identity is numerically exact.

## Bottom Line

Cycle 7 now provides the requested eigenspace-level BMI dataset and comparative diagnostics for all five tasks.
Empirically, dbar<1 holds on base=False and adversarial=True scans.
A full analytic BMI closure still needs a stronger inequality than the current three direct bound probes.
