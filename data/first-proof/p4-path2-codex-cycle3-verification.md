# P4 Path 2 Cycle 3 (Codex): Same-Sign Claim 3 Status

Date: 2026-02-13  
Script: `scripts/explore-p4-path2-codex-cycle3.py`  
Results: `data/first-proof/p4-path2-codex-cycle3-results.json`

## Scope
This run addresses the Cycle 3 gap from `p4-path2-codex-handoff-3.md`:
prove `K_red(p,q) >= 0` on the feasible same-sign region `p,q >= 0`.

## Exact algebraic reductions (proved symbolically)
1. `K_red = A(P,Q) + p q B(P,Q)` with `P=p^2`, `Q=q^2` (exact identity check is `True`).
2. On same-sign points (`p,q>=0`):
   `K_same(P,Q) = A(P,Q) + sqrt(PQ) B(P,Q)`.
3. Candidate equivalent route:
   if `A >= 0`, `B <= 0`, and `A^2 - P Q B^2 >= 0`, then
   `K_same >= 0`.

## Edge sub-claim (3a): endpoint signs closed exactly
For edge quadratics
- `K(0,q) = a00 + a01 Q + a02 Q^2`, `Q in [0,Qmax]`
- `K(p,0) = a00 + a10 P + a20 P^2`, `P in [0,Pmax]`

the endpoint values at `Qmax`, `Pmax` factor exactly as in the JSON output.
The remaining linear factors are
- `Fq = 3 r^2 y - 7 r^2 + 3 r x - 7 r + 3 x - 3`
- `Fp = 3 r^2 y - 3 r^2 + 3 r y - 7 r + 3 x - 7`

and satisfy exact upper-bound identities:
- `Fq = (-4 r^2 - 4 r) + 3 r^2 (y-1) + 3 (r+1)(x-1) <= -4 r^2 - 4 r < 0`
- `Fp = (-4 r - 4) + 3 r^2 (y-1) + 3 r (y-1) + 3 (x-1) <= -4 r - 4 < 0`

So the edge endpoint signs are algebraically fixed.

## Numerical stress (same-sign + edges)
From `120,000` same-sign feasible samples (`r` log-uniform on `[1e-8,1e8]`):
- `K_negative = 0`
- `A_negative = 0`
- `B_positive = 0`
- `S_negative = 0` where `S` is evaluated stably as `K*(A - sqrt(PQ)B)`.

From `120,000` edge-trajectory samples:
- `q_edge_negative = 0`
- `p_edge_negative = 0`.

## What this closes vs. what remains
Closed in this cycle:
1. Exact same-sign reduction `K = A + sqrt(PQ)B`.
2. Exact endpoint sign lemmas for edge factors (`Fq<0`, `Fp<0`).
3. Strong large-range stress support for `K>=0` and for the `A/B/S` route.

Not yet closed symbolically:
1. A full analytic proof that `A(P,Q) >= 0` on the entire feasible rectangle.
2. A full analytic proof that `A^2 - P Q B^2 >= 0` on the same domain.

So Claim 3 is now narrowed to two concrete polynomial-inequality subclaims in `(r,x,y,P,Q)`.
