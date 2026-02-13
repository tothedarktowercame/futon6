# P4 n=4 Full Case: Normalized Numerator Structure

Date: 2026-02-13

Script:
- `scripts/analyze-p4-n4-full-normalized.py`

Run:
- `python3 scripts/analyze-p4-n4-full-normalized.py --trials 25000 --seed 20260213`

## Goal

Extend the symmetric (`e3=0`) 30-term certificate toward the full centered
`n=4` Stam numerator in variables `(s,t,u,v,a,b)`.

## Normalization

Using
- `r = t/s`,
- `x = 4a/s^2`,
- `y = 4b/t^2`,
- `p = u/s^(3/2)`,
- `q = v/s^(3/2)`,

the full surplus numerator satisfies:
- `N(s,t,u,v,a,b) = s^16 * K(r,x,y,p,q)`.

This removes one scale and makes the feasible cone bounded in `(x,y,p,q)` once
`r` is fixed.

## Structural findings

1. Full numerator complexity:
   - `N` has 659 monomials (total degree 16).
   - `K` has 659 monomials (total degree 18).

2. Symmetry:
   - swap symmetry `(s,a,u) <-> (t,b,v)` holds.
   - global parity `(u,v) -> (-u,-v)` holds.

3. `(p,q)` block decomposition:
   - degree 0 block: 102 terms
   - degree 2 block: 200 terms
   - degree 4 block: 201 terms
   - degree 6 block: 114 terms
   - degree 8 block: 42 terms

## Key obstruction

The first perturbation block
- `K2 = A2(r,x,y) p^2 + B2(r,x,y) p q + C2(r,x,y) q^2`
is **not** PSD over sampled base points:
- `min A2 < 0`, `min C2 < 0`,
- and `max(B2^2 - 4A2C2) > 0`.

So a naive proof by nonnegativity of each `(p,q)`-degree block fails.

## Positivity status

Despite the `K2` obstruction, the full `K` remained positive on all sampled
feasible points (`25000`/`25000`, minimum about `2.0e-06`).

## Conclusion

The symmetric certificate is the base layer, but full-case closure requires
**coupled control across blocks** (`K2,K4,K6,K8`) rather than separate
blockwise positivity.

Natural next step:
- derive inequalities that bound negative regions of `K2` by positive `K4+K6+K8`
  on the feasible cone.

