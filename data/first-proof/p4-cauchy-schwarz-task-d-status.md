# P4 Cauchy-Schwarz Handoff: Task D Status (n=3)

Date: 2026-02-13

## Scope

Implemented the prioritized Task D from
`data/first-proof/CODEX-P4-CAUCHY-SCHWARZ-HANDOFF.md`:

- explicit `n=3` matching-average expansion,
- centered coefficient-addition reduction,
- centered cubic `1/Phi_3` formula in `(e2,e3)`,
- 4-variable surplus polynomial and SOS-ready decomposition.

Script:
- `scripts/verify-p4-n3-cauchy-sos.py`

## Main symbolic outputs

1. Matching-average (explicit over 6 permutations) reproduces the `n=3` MSS formulas:
   - `E1 = e1(p) + e1(q)`
   - `E2 = e2(p) + (2/3)e1(p)e1(q) + e2(q)`
   - `E3 = e3(p) + (1/3)e2(p)e1(q) + (1/3)e1(p)e2(q) + e3(q)`

2. Centered specialization (`e1(p)=e1(q)=0`) gives coefficient addition:
   - `E2 = e2(p) + e2(q)`
   - `E3 = e3(p) + e3(q)`

3. Centered cubic identity verified symbolically:
   - `1/Phi_3(e2,e3) = -2*e2/9 - 3*e3^2/(2*e2^2)`
   - equivalent to `Phi_3 * disc = 18*e2^2`.

4. Surplus with `p(x)=x^3-sx-u`, `q(x)=x^3-tx-v`, `s,t>0`:
   - `surplus = (3/2)[u^2/s^2 + v^2/t^2 - (u+v)^2/(s+t)^2]`
   - numerator (after clearing positive denominator) has decomposition
     `N = (s^2 v - t^2 u)^2 + s t (s v + t u)^2 + s t (s v - t u)^2`.

This is an explicit SOS-style nonnegativity certificate on `s,t>0`.

## Equality condition (n=3 centered)

From the above decomposition (or the two-step Titu chain), with `s,t>0`:

- `surplus = 0` iff `u = v = 0`.

The proof text was updated accordingly in:
- `data/first-proof/problem4-solution.md`

## Remaining open issues by priority

1. Task B (next): move from the `n=3` coefficient certificate to a direct
   Cauchy-Schwarz argument on the matching-average score representation.
2. Task A: formalize the equi-spaced Pythagorean equality family for general `n`.
3. Task C: categorical/coend lens only if it yields new algebraic inequalities.

