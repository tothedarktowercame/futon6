# P4 n=4 Symmetric Quartic Certificate (30-Term Numerator)

Date: 2026-02-13

## Scope

This closes the handoff subtask:
- prove nonnegativity of the 30-term symmetric-quartic surplus numerator
  on the cone
  - `s,t > 0`
  - `0 < 4a < s^2`
  - `0 < 4b < t^2`.

Script:
- `scripts/prove-p4-n4-symmetric-30term.py`

## Setup

For `e3 = 0`, write
- `p(x) = x^4 - s x^2 + a`
- `q(x) = x^4 - t x^2 + b`.

The symmetric-case Stam surplus has denominator
`9st(12a+s^2)(12b+t^2)(s+t)(12a+12b+s^2+4st+t^2)`,
which is strictly positive on the cone.

So it is enough to prove `N(s,t,a,b) >= 0`, where `N` is the 30-term polynomial
from `CODEX-P4-N4-SOS-HANDOFF.md`.

## Reduction

Normalize:
- `r = t/s > 0`
- `x = 4a/s^2 in (0,1)`
- `y = 4b/t^2 in (0,1)`.

Then:
- `N = s^10 r^4 G(r,x,y)`,
so sign(`N`) = sign(`G`).

Shift box coordinates:
- `p = 3x - 1 in (-1,2)`
- `q = 3y - 1 in (-1,2)`.

In these variables:
- `G = A(p,q) r^2 + B(p,q) r + C(p,q)`,
with
- `A = p^2 q^2/2 + 2p^2 q + 2p^2 + p q^3/2 + 3p q^2 + q^3 + 6q^2`
- `B = 2p^2 q + 6p^2 + 2p q^2 - 4p q + 6q^2`
- `C = p^3 q/2 + p^3 + p^2 q^2/2 + 3p^2 q + 6p^2 + 2p q^2 + 2q^2`.

## Positivity proof

1. `A` as quadratic in `p`:
   - leading coefficient `(q+2)^2/2 > 0`,
   - discriminant
     `DeltaA = q^2(q+6)(q^3 - 2q^2 - 32q - 32)/4 <= 0` on `q in [-1,2]`.
   - Therefore `A >= 0`.

2. `B` as quadratic in `p`:
   - leading coefficient `2(q+3) > 0`,
   - discriminant
     `DeltaB = 4q^2(q^2 - 16q - 32) <= 0` on `q in [-1,2]`.
   - Therefore `B >= 0`.

3. `C` as quadratic in `q`:
   - leading coefficient `(p+2)^2/2 > 0`,
   - discriminant
     `DeltaC = p^2(p+6)(p^3 - 2p^2 - 32p - 32)/4 <= 0` on `p in [-1,2]`.
   - Therefore `C >= 0`.

Since `r > 0`, we get `G = A r^2 + B r + C >= 0`, hence `N >= 0`.

So the symmetric `n=4` Stam surplus is nonnegative.

## Verification output

`python3 scripts/prove-p4-n4-symmetric-30term.py` reports:
- symbolic reduction checks passed,
- discriminant checks passed,
- random sanity check over 200000 feasible samples:
  - minimum observed numerator `~ 4.54e-06` (no negatives).

