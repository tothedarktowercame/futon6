# P4 Task B Status: n=3 Matching-Average Cauchy-Schwarz

Date: 2026-02-13

Script:
- `scripts/verify-p4-n3-matching-cs.py`

Run:
- `python3 scripts/verify-p4-n3-matching-cs.py --trials 2500 --seed 20260213`

## What was tested

For centered cubics
- `p(x)=x^3-sx-u`, `q(x)=x^3-tx-v` with three distinct real roots,
- build all 6 matching polynomials `m_sigma(x)`,
- set `c(x) = (1/6) sum_sigma m_sigma(x)` and verify `c = p boxplus_3 q`.

At each root `zeta_k` of `c`, define:
- `alpha_sigma,k = m_sigma''(zeta_k)/(2 m_sigma'(zeta_k))`,
- `w_sigma,k = m_sigma'(zeta_k) / sum_tau m_tau'(zeta_k)`.

Then check
- exact score identity: `S_k(c) = sum_sigma w_sigma,k alpha_sigma,k`,
- Cauchy/Jensen bounds based on this representation.

## Numerical findings (2500 trials)

1. **Matching-average identity is exact numerically**
   - `max ||matching-average coeffs - MSS coeffs||_inf = 1.776e-14`.

2. **Rootwise score decomposition is exact numerically**
   - `max |S_k(c) - sum_sigma w_sigma,k alpha_sigma,k| = 6.661e-16`.

3. **Weight geometry at n=3 appears favorable**
   - all tested rootpoints had `w_sigma,k >= 0` (7500/7500),
   - all tested rootpoints had matching derivative signs aligned.
   - So Jensen/variance at each root is valid in this tested regime.

4. **But direct C-S/Jensen bound is generally weaker than Stam**
   - Construct route:
     `Phi(c) <= B1 := sum_k sum_sigma w_sigma,k alpha_sigma,k^2`,
     hence `1/Phi(c) >= 1/B1`.
   - Compare `1/B1` to Stam RHS `1/Phi(p)+1/Phi(q)`.
   - This route implied Stam in only `826/2500` trials.
   - Ratio stats `(1/B1)/(1/Phi(p)+1/Phi(q))`: min `0.0199`, mean `0.9475`.

5. **Generic ratio-of-expectations C-S is much weaker**
   - implied Stam in `25/2500` trials.

6. **Stam itself remained true numerically**
   - `0/2500` violations.

## Interpretation

Task B’s matching-average decomposition succeeds at the identity level:
- score at roots of `c` is exactly a weighted average of matching-root scores.

However, the naive Cauchy/Jensen step does **not** recover Stam sharply:
- it produces a valid but often subcritical lower bound on `1/Phi(c)`.

So the remaining gap is not the decomposition itself, but **tight control of the
Jensen slack across roots** (or a different inequality that couples roots and
matchings more efficiently than per-root second-moment bounds).

