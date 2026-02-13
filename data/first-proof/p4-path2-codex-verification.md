# P4 Path 2 Codex Verification

Date: 2026-02-13  
Agent: Codex  
Handoff: `data/first-proof/p4-path2-codex-handoff.md`

Artifacts:
- Script: `scripts/explore-p4-path2-codex.py`
- Results JSON: `data/first-proof/p4-path2-codex-results.json`

## Executive Summary

I ran the full Path 2 algebra pipeline and found one normalization issue plus a strong partial certificate.

1. The literal handoff formula for `K_T2R_num` is not the same as the exact normalized T2+R numerator used by the verified T2R pipeline.
2. After correction, the canonical reduced polynomial satisfies all expected structure:
   - degree profile `0/2/4/6`,
   - exact `d6 = -1296*r*L*p^2*q^2*(p+q)^2`,
   - stable `(u,v)` coefficient factoring patterns.
3. A strong candidate certificate was found:
   - `K_red = A(p^2,q^2) + p*q*B(p^2,q^2)`,
   - across 200k feasible samples: `B < 0` always, `A > |p*q*B|` always, min ratio `A/|pqB| = 2.297...`,
   - yielding `K_red > 0` in all samples.

No full symbolic closure yet for `A >= |pqB|`, but this is now a focused algebraic target.

## Task 1: Compute K, r² divisibility, d6 structure

### Normalization check

Three numerators were computed:

- `K_handoff_literal` from the handoff literal formula:
  
  `2*L*f2p*f2q*f2c + r*(1+3x)(1+3y)*f1c*R_surplus_num`
- `K_corr` from exact T2/R assembly:
  
  `2*r*L*f2p*f2q*f2c + (1+3x)(1+3y)*f1c*R_surplus_num`
- `K_exact_red` from full normalized T2+R derivation (`prove-p4` pipeline), divided by `r^2`.

Exact identity established:

- `8*K_corr == 9*K_exact_red` (symbolically true)

So `K_corr` and the canonical `K_exact_red` are equivalent up to positive scalar `9/8`.

### Term counts

- `K_handoff_literal`: 652 terms
- `K_corr`: 487 terms
- `K_exact_red`: 487 terms

### r² divisibility

- Canonical full numerator `K_exact` is divisible by `r^2`.
- `K_red = K_exact / r^2` is the correct reduced object for analysis.

### d6 factorization

For `K_red`, verified exactly:

`d6 = -1296*r*L*p^2*q^2*(p+q)^2`

## Task 2: (u,v) analysis

Used `p=(u+v)/2`, `q=(u-v)/2` on canonical `K_red`.

Monomials present:
- `u^0v^0, u^0v^2, u^0v^4, u^1v^1, u^2v^0, u^2v^2, u^2v^4, u^3v^1, u^4v^0, u^4v^2, u^6v^0`

For each coefficient, divisibility by `W` and `L` is recorded in:

- `data/first-proof/p4-path2-codex-results.json` → `task2.uv_divisibility`

(These are now programmatically available for targeted case splits.)

## Task 3: Algebraic certificate attempt

Derived exact decomposition:

`K_red = A(p^2,q^2) + p*q*B(p^2,q^2)`

with coefficient identities:

- `B_{0,1} = 2*A_{0,2}`
- `B_{1,0} = 2*A_{2,0}`
- `B_{1,1} = 2*A_{1,2} = 2*A_{2,1}`

This exposes a strong structured coupling of even/odd blocks.

Numerical evidence (200k feasible samples):

- `B > 0`: 0 cases (`B_max = -8.73e-06`)
- `K_red < 0`: 0 cases (`K_min = 2.71e-08`)
- `A/|p*q*B|` minimum: `2.297...`

Interpretation:

- If one proves `B <= 0` and `A >= |p*q*B|` symbolically on feasible domain, closure follows immediately.

## Task 4: Structural discoveries used

Confirmed the line identity behind `L<=0 => 3x+3y>=2`:

- `L(x, 2/3 - x) = 4*(r+1)*(3x-1)^2`

which is exactly the required boundary-square structure.

`L`-split numerics were included in the same 200k scan:

- `K_red` remained nonnegative in both `L<=0` and `L>0` regions.

## Task 5: Notes on failed routes / what not repeated

I did not rerun global Gram SOS attempts or the previously failed perfect-square/Hilbert routes. The work stayed on:

- corrected normalization,
- exact block identities,
- focused certificate target (`A + pqB`).

## Bottom Line

- The handoff literal `K` expression appears mis-normalized; the corrected/canonical `K_red` is consistent with prior verified structure.
- All required structural checks (degree profile, `d6`, `(u,v)` factoring patterns) now pass cleanly on the corrected object.
- A strong partial certificate is now isolated:
  - prove `B <= 0` and `A >= |pqB|` symbolically under feasible constraints.

That is the shortest remaining route from this branch of Path 2 to a full algebraic proof.
