# P4 n=4 Case 3c: Task B (Smale Alpha) Status

Date: 2026-02-13

## Goal

Certify the four known Case 3c critical points from PHC as genuine simple roots
of the 4x4 gradient system `grad(-N)=0`.

## Method

Script: `scripts/verify-p4-n4-case3c-alpha.py`

Input seeds:
- `data/first-proof/problem4-case3c-phc-certified.json` (`case3c_points`)

For each seed:
1. Refine with high-precision Newton (mpmath, 120 dps).
2. Compute `beta = ||J^{-1}F||_2`.
3. Use the standard alpha-theory upper bound
   `gamma <= mu * D^(3/2) / (2 ||x||_1)`,
   with `mu = max(1, ||F||_BW * ||J^{-1}Delta||)` and a conservative
   Frobenius upper bound for `||J^{-1}Delta||`.
4. Check `alpha_upper = beta * gamma_upper < alpha_0`, with
   `alpha_0 = 0.157671`.

## Result

- 4/4 roots satisfy `alpha_upper < 0.157671`.
- Residual norms are about `1e-116`.
- All 4 refined roots are in the feasibility domain.
- All 4 have `-N = 1678.549826372544892... > 0`.

Artifacts:
- `data/first-proof/problem4-case3c-alpha-certification.json`
- `data/first-proof/problem4-case3c-certified-v2.json`

## Caveat (important)

This Task B result certifies the listed four roots. It does **not** by itself
prove global exhaustiveness of all real in-domain roots, so it does not close
the remaining PHC failed-path gap on its own.
