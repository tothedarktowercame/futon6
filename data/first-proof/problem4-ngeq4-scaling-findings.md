# Problem 4 (n>=4) Scaling Findings

Date: 2026-02-11

## Inputs
- Research brief: `data/first-proof/problem4-ngeq4-research-brief.md`
- Script: `scripts/verify-p4-scaling.py`
- Raw run output: `data/first-proof/problem4-scaling-research-run.txt`

## Run profile (bounded)
- Command:
  - `python3 -u scripts/verify-p4-scaling.py --adversarial-ns 3,4,5,6,7,8 --adversarial-starts-small 40 --adversarial-starts-large 20 --nm-maxiter 2000 --scale-trials 1000 --random-trials-small 1500 --random-trials-large 600`
- Seed: default (`42`)

## Key quantitative results
1. Adversarial minima (`n=3..8`) remain at ratio ~`1.000000000000` (to numerical precision).
2. Scale-separation test (`n=4..8`) produces best ratios slightly below 1 by tiny eps (`~1e-11` to `1e-8`), consistent with floating-point/optimizer tolerance near the boundary.
3. Random percentile floors rise with degree:
   - `n=4` min ~`1.0011`
   - `n=8` min ~`1.2908`
   - `n=15` min ~`2.4370`
   So typical random instances get farther from the tight boundary as `n` increases.

## Structural pattern in tight cases
- The optimizer repeatedly returns near-arithmetic root patterns in both polynomials.
- Gap CVs are low and matched across p and q (e.g. for `n=8`, both ~`0.0885`).
- Tightness correlates with coordinated spacing structure and, in scale-separation sweeps, very large relative scales.

## Implications for proof strategy
- Supports brief claim that infimum is likely `1` for every tested degree (tight but not violated).
- Reinforces that a proof approach depending on a uniform strict gap above `1` for large `n` is unlikely to work.
- Suggests pursuing asymptotic/tangent analysis around scale-separated or near-arithmetic extremizers, rather than average-case arguments.

## Immediate next research actions
1. Derive first-order/second-order asymptotics of the surplus in the scale-separated regime from MSS coefficient weights.
2. Test whether near-arithmetic root families admit an explicit symbolic surplus decomposition for `n=4`.
3. Connect these extremal families to finite-free cumulant coordinates to search for a sum-of-squares form.
