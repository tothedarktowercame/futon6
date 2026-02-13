# P4 n=4 SOS Handoff Execution Results

## Step 1
- Phi4*disc identity exact symbolic: `True`
- Max numeric abs error (200 tests): `8.941e-08`
- Centered MSS E2 additive: `True`
- Centered MSS E3 additive: `True`
- Centered MSS E4 cross 1/6: `True`

## Step 2
- Surplus numerator degree: `16`
- Surplus numerator terms: `659`
- Swap symmetry: `True`
- (u,v)->(-u,-v) symmetry: `True`

## Step 3
- cvxpy available: `False`
- scs available: `False`
- status: `blocked_no_sdp_solver`

## Step 4 (u=v=0) symbolic
- factorization checks: `N=True`, `D=True`, `reduced=True`
- A2 decomposition exact: `True`
- A0 decomposition exact: `True`
- A1 square-completion exact: `True`
- symbolic conclusion surplus>=0: `True`

## Step 4 (u=v=0) numeric sanity
- random valid samples: `79577`
- random min surplus: `1.129470e-07`
- random negative count: `0`
- global-opt min surplus: `1.643405e-10`

## Sign Convention Note
- Verified E4 cross term is `+(1/6)e2(p)e2(q)`.
- With `s=-e2(p), t=-e2(q)`, this is `+st/6` in E4.
