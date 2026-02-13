#!/usr/bin/env python3
"""Quick check: what is the r-degree of K0+K2+K4 and K8?

If K0+K2+K4 is low-degree in r (e.g. quadratic), the same
discriminant-bounding technique from the symmetric case might extend.
"""

import sympy as sp


def disc_poly(e2, e3, e4):
    return (256*e4**3 - 128*e2**2*e4**2 + 144*e2*e3**2*e4
            + 16*e2**4*e4 - 27*e3**4 - 4*e2**3*e3**2)


def phi4_disc(e2, e3, e4):
    return (-8*e2**5 - 64*e2**3*e4 - 36*e2**2*e3**2
            + 384*e2*e4**2 - 432*e3**2*e4)


print("Building full surplus and decomposing (takes ~2 min)...")

e2, e3, e4 = sp.symbols('e2 e3 e4')
s, t, u, v, a, b = sp.symbols('s t u v a b', positive=True)
r, x, y, p, q = sp.symbols('r x y p q', real=True)

inv_phi = sp.cancel(disc_poly(e2, e3, e4) / phi4_disc(e2, e3, e4))
inv_p = inv_phi.subs({e2: -s, e3: u, e4: a})
inv_q = inv_phi.subs({e2: -t, e3: v, e4: b})
inv_c = inv_phi.subs({e2: -(s+t), e3: u+v, e4: a+b+s*t/6})

surplus = sp.together(inv_c - inv_p - inv_q)
N, D = sp.fraction(surplus)
N = sp.expand(N)

# Normalize
subs = {t: r*s, a: x*s**2/4, b: y*r**2*s**2/4,
        u: p*s**sp.Rational(3,2), v: q*s**sp.Rational(3,2)}
K = sp.expand(sp.expand(N.subs(subs)) / s**16)

# Decompose by (p,q)-degree
poly_pq = sp.Poly(K, p, q)
blocks = {}
for monom, coeff in poly_pq.as_dict().items():
    i, j = monom
    d = i + j
    blocks[d] = sp.expand(blocks.get(d, sp.Integer(0)) + coeff * p**i * q**j)

# Check r-degree of each block and key sums
print(f"\n{'='*60}")
print("r-degree analysis")
print('='*60)

for d in sorted(blocks.keys()):
    poly_r = sp.Poly(blocks[d], r)
    print(f"K{d}: degree in r = {poly_r.degree()}")

# K0+K2+K4
S024 = sp.expand(blocks[0] + blocks[2] + blocks[4])
poly_r_024 = sp.Poly(S024, r)
print(f"\nK0+K2+K4: degree in r = {poly_r_024.degree()}")
print(f"K0+K2+K4: terms = {len(sp.Poly(S024, r, x, y, p, q).as_dict())}")

# Extract r-coefficients of K0+K2+K4
print(f"\nK0+K2+K4 as polynomial in r:")
for k in range(poly_r_024.degree(), -1, -1):
    coeff_k = sp.expand(poly_r_024.nth(k))
    if coeff_k != 0:
        terms = len(sp.Poly(coeff_k, x, y, p, q).as_dict())
        print(f"  r^{k}: {terms} terms")

# K8
poly_r_8 = sp.Poly(blocks[8], r)
print(f"\nK8: degree in r = {poly_r_8.degree()}")
print(f"K8: terms = {len(sp.Poly(blocks[8], r, x, y, p, q).as_dict())}")
print(f"K8 as polynomial in r:")
for k in range(poly_r_8.degree(), -1, -1):
    coeff_k = sp.expand(poly_r_8.nth(k))
    if coeff_k != 0:
        terms = len(sp.Poly(coeff_k, x, y, p, q).as_dict())
        print(f"  r^{k}: {terms} terms")

# K6
poly_r_6 = sp.Poly(blocks[6], r)
print(f"\nK6: degree in r = {poly_r_6.degree()}")
print(f"K6 as polynomial in r:")
for k in range(poly_r_6.degree(), -1, -1):
    coeff_k = sp.expand(poly_r_6.nth(k))
    if coeff_k != 0:
        terms = len(sp.Poly(coeff_k, x, y, p, q).as_dict())
        print(f"  r^{k}: {terms} terms")

# Also check: is K0+K2+K4 quadratic in r?
if poly_r_024.degree() == 2:
    print(f"\n*** K0+K2+K4 is QUADRATIC in r! Same technique may apply. ***")
    A = sp.expand(poly_r_024.nth(2))
    B = sp.expand(poly_r_024.nth(1))
    C = sp.expand(poly_r_024.nth(0))
    print(f"  Coefficient of r^2: {len(sp.Poly(A, x, y, p, q).as_dict())} terms")
    print(f"  Coefficient of r^1: {len(sp.Poly(B, x, y, p, q).as_dict())} terms")
    print(f"  Coefficient of r^0: {len(sp.Poly(C, x, y, p, q).as_dict())} terms")
elif poly_r_024.degree() <= 4:
    print(f"\n*** K0+K2+K4 is degree {poly_r_024.degree()} in r. ***")
    print("Low enough for coefficient analysis.")
