# Problem 6 Cycle 8 Codex Verification

Date: 2026-02-13
Agent: Codex
Base handoff: `data/first-proof/problem6-codex-cycle8-handoff.md`

Artifacts:
- Script: `scripts/verify-p6-cycle8-codex.py`
- Results JSON: `data/first-proof/problem6-codex-cycle8-results.json`

## Executive Summary

- Strict-threshold base runs: 116 (731 scanned steps)
- Strict-threshold adversarial runs: 32 (232 scanned steps)
- GPL-V on strict runs (all steps have some feasible vertex): base=True, adversarial=True
- Strict base dbar>=1 steps: 7 (worst 1.641043 at Reg_100_d10 eps=0.5)
- Strict skip runs: base=5, adversarial=0 (max skips before one selection = 2)

## Task 1: Full R-Scan

At each selected step, the scan records all vertices in R_t with:
- normY(v), traceY(v)
- deg_S(v), ell_v^{I0}, ell_v^{S_t}
- high-eigenspace overlap and cross-degree bound diagnostics

Key strict-base minimum feasible count over all steps: 12 (fraction 1.0000)

## Task 2: Threshold Reconciliation

- Cases compared: 116
- Different I0 size: 0
- Different horizon: 0
- Different skip totals: 0
- Strict has more skips: 0, loose has more skips: 0

## Task 3: Eigenspace Overlap

- Mean high-overlap fraction (feasible): 0.005524857125511627
- Mean high-overlap fraction (infeasible): 0.14644837589529303
- Mean high-norm fraction (feasible): 0.012531768929174032
- Mean high-norm fraction (infeasible): 0.563394899802198
- Focus steps captured (skip or dbar>=1): 12

## Task 4: Cross-Degree Bound

Tested bound: normY(v) <= deg_S(v) * max_{e incident to v and S} z_e^T B_t z_e
- Vertex rows checked: 78619
- Violations: 0
- Max ratio norm/bound: 1.000000
- Ratio quantiles: {'q50': 0.0, 'q90': 0.9999999999999998, 'q95': 1.0000000000000002, 'q99': 1.0000000000000018}

## Task 5: Partial Averages With Skips

- Skip runs (strict): 5
- Max skips before one selection: 2
- Skip-adjusted dbar0 bound violations: 0
- Worst skip-adjusted violation: None

## Task 6: Alternative Existence Arguments

- Pr[normY<1] > 0 at all steps: True
- Feasible-fraction quantiles: {'q10': 1.0, 'q25': 1.0, 'q50': 1.0, 'q75': 1.0, 'q90': 1.0}
- deg_S=0 fraction quantiles: {'q10': 0.0, 'q25': 0.2403515728257996, 'q50': 0.7241379310344828, 'q75': 1.0, 'q90': 1.0}
- Mean E[normY]: 0.145087
- Mean E[traceY]: 0.221550
- Mean E[normY]/E[traceY]: 0.492620

## Bottom Line

With strict threshold (tau >= eps), No-Skip is not universal, but GPL-V remains empirically robust on this suite.
The full R-scan/eigenspace data is now available for proving existence of SOME feasible vertex directly.
