#!/usr/bin/env python3
"""Algebraic decomposition of R_surplus_num for proof.

Key insight from gram2: c31/c40 = c13/c04 = 2, so the quartic has form:
  Q4 = c40·p²(p+q)² + δ·p²q² + c04·q²(p+q)²
where δ = c22 - c40 - c04.

Since c40, c04 ≥ 0 on feasible, Q4 ≥ 0 if δ ≥ 0.

Also: c00 has factor W = 3r²(y-1)+3(x-1)-4r, and W < 0 on feasible interior.
The sign of c00 depends on (3x+3y-2). Question: does L ≤ 0 force 3x+3y ≥ 2?

Plan:
1. Verify the quartic decomposition Q4 = (c40p²+c04q²)(p+q)² + δp²q²
2. Test sign of δ on feasible, especially when L ≤ 0
3. Test whether L ≤ 0 ⟹ 3x+3y ≥ 2
4. Look for clean decomposition of R_surplus_num as sum of non-negative terms
"""

import numpy as np
import sympy as sp
from sympy import symbols, expand, Poly, factor, cancel, simplify
import time


def pr(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


def main():
    t0 = time.time()
    r, x, y, p, q = symbols('r x y p q')

    pr("Building R_surplus_num...")

    C_p = x - 1; f1p = 1 + 3*x; f2p = 2*(1-x) - 9*p**2
    C_q = r**2*(y - 1); f1q = r**2*(1 + 3*y); f2q = 2*r**3*(1-y) - 9*q**2
    Sv = 1 + r; Av12 = 3*x + 3*y*r**2 + 2*r
    C_c = expand(Av12/3 - Sv**2)
    f1c = expand(Sv**2 + Av12)
    f2c = expand(2*Sv**3 - 2*Sv*Av12/3 - 9*(p+q)**2)

    R_num = expand(C_c*f1c*f2p*f2q - C_p*f1p*f2c*f2q - C_q*f1q*f2c*f2p)

    # Extract (p,q)-monomial coefficients
    poly_pq = Poly(R_num, p, q)
    coeffs = {}
    for monom, coeff in poly_pq.as_dict().items():
        coeffs[monom] = expand(coeff)

    c00 = coeffs.get((0,0), sp.Integer(0))
    c20 = coeffs.get((2,0), sp.Integer(0))
    c11 = coeffs.get((1,1), sp.Integer(0))
    c02 = coeffs.get((0,2), sp.Integer(0))
    c40 = coeffs.get((4,0), sp.Integer(0))
    c31 = coeffs.get((3,1), sp.Integer(0))
    c22 = coeffs.get((2,2), sp.Integer(0))
    c13 = coeffs.get((1,3), sp.Integer(0))
    c04 = coeffs.get((0,4), sp.Integer(0))
    pr(f"  Built [{time.time()-t0:.1f}s]")

    L = -27*r*x*y + 3*r*x + 9*r*y**2 - 3*r*y + 2*r + 9*x**2 - 27*x*y - 3*x + 3*y + 2
    W = 3*r**2*y - 3*r**2 - 4*r + 3*x - 3

    # ================================================================
    pr(f"\n{'='*72}")
    pr("PART 1: QUARTIC DECOMPOSITION")
    pr('='*72)

    # Q4 = c40·p⁴ + c31·p³q + c22·p²q² + c13·pq³ + c04·q⁴
    # Since c31 = 2c40 and c13 = 2c04 (verified):
    # Q4 = c40(p⁴+2p³q) + c22·p²q² + c04(q⁴+2pq³)
    # Note: p⁴+2p³q = p²(p²+2pq) = p²(p+q)² - p²q²
    # So: Q4 = c40(p²(p+q)² - p²q²) + c22·p²q² + c04(q²(p+q)² - p²q²)
    # = (c40·p² + c04·q²)(p+q)² + (c22 - c40 - c04)·p²q²

    delta = expand(c22 - c40 - c04)
    delta_f = factor(delta)
    pr(f"  δ = c22 - c40 - c04 = {delta_f}")
    n_delta = len(Poly(delta, r, x, y).as_dict())
    pr(f"  ({n_delta} terms)")

    # So Q4 = (c40p²+c04q²)(p+q)² + δ·p²q²
    # c40 = 81r⁴(1-y)(3y+1) ≥ 0, c04 = 81(1-x)(3x+1) ≥ 0
    # If δ ≥ 0: Q4 ≥ 0

    # Check sign of δ
    delta_fn = sp.lambdify((r, x, y), delta, 'numpy')
    L_fn = sp.lambdify((r, x, y), L, 'numpy')

    rng = np.random.default_rng(42)
    delta_neg_all = 0
    delta_neg_Lneg = 0
    n_all = 0
    n_Lneg = 0

    for _ in range(500000):
        rv = float(np.exp(rng.uniform(np.log(0.05), np.log(20.0))))
        xv = float(rng.uniform(0.001, 0.999))
        yv = float(rng.uniform(0.001, 0.999))
        n_all += 1
        dv = delta_fn(rv, xv, yv)
        Lv = L_fn(rv, xv, yv)
        if dv < 0:
            delta_neg_all += 1
        if Lv <= 0:
            n_Lneg += 1
            if dv < 0:
                delta_neg_Lneg += 1

    pr(f"\n  δ < 0: {delta_neg_all}/{n_all} overall")
    pr(f"  δ < 0: {delta_neg_Lneg}/{n_Lneg} when L ≤ 0")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("PART 2: DOES L ≤ 0 FORCE 3x+3y ≥ 2?")
    pr('='*72)

    rng2 = np.random.default_rng(123)
    counter = 0
    for _ in range(1000000):
        rv = float(np.exp(rng2.uniform(np.log(0.05), np.log(20.0))))
        xv = float(rng2.uniform(0.001, 0.999))
        yv = float(rng2.uniform(0.001, 0.999))
        Lv = L_fn(rv, xv, yv)
        if Lv <= 0 and 3*xv + 3*yv < 2:
            counter += 1
    pr(f"  L ≤ 0 AND 3x+3y < 2: {counter}/1000000")

    if counter == 0:
        pr("  *** L ≤ 0 implies 3x+3y ≥ 2 (1M samples) ***")
        # Try to prove: L ≤ 0 ⟹ x+y ≥ 2/3
        # Substitute y = 2/3-x-ε with ε > 0, show L > 0
        # L(x, 2/3-x) already computed: 18(x-1/3)² + 6 at r=1
        # At general r: L on x+y=2/3 line
        eps = sp.Symbol('eps', positive=True)
        y_sub = sp.Rational(2,3) - x
        L_on_line = expand(L.subs(y, y_sub))
        pr(f"\n  L at x+y=2/3: {factor(L_on_line)}")
        L_coeffs = Poly(L_on_line, x).all_coeffs()
        pr(f"  As polynomial in x: coefficients = {[factor(c) for c in L_coeffs]}")

        # Check discriminant
        if len(L_coeffs) == 3:
            a_coeff, b_coeff, c_coeff = L_coeffs
            disc_line = expand(b_coeff**2 - 4*a_coeff*c_coeff)
            pr(f"  Discriminant = {factor(disc_line)}")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("PART 3: QUADRATIC DECOMPOSITION")
    pr('='*72)

    # Q2 = c20·p² + c11·pq + c02·q²
    # Similarly to quartic, check if c11 = 2α·something
    # c20 - c02 = 6r(-ry+r+x-1)W² (from gram2)
    pr(f"  c20 - c02 = {factor(expand(c20 - c02))}")

    # If c20 ≥ 0 and c02 ≥ 0, we have a more refined decomposition
    # Q2 = min(c20,c02)(p²+q²) + (c20-c02)(p² or q²) + c11·pq
    #     = min(c20,c02)(p+q)² + (c20-c02-c11/2+something)...

    # Let me try: Q2 = α(p+q)² + β(p-q)² + γpq
    # = (α+β)p² + (2α-2β+γ)pq + (α+β)q²
    # Need: α+β = c20, 2α-2β+γ = c11... but this requires c20 = c02.

    # Alternative: Q2 = c20p² + c11pq + c02q²
    # = c20(p + c11/(2c20) q)² + (c02 - c11²/(4c20))q²
    # PSD iff c20 ≥ 0 and 4c20·c02 - c11² ≥ 0

    # Check c20 sign
    c20_fn = sp.lambdify((r, x, y), c20, 'numpy')
    c02_fn = sp.lambdify((r, x, y), c02, 'numpy')
    c11_fn = sp.lambdify((r, x, y), c11, 'numpy')

    rng3 = np.random.default_rng(456)
    c20_neg = 0; c02_neg = 0; disc_neg = 0; n_check = 0
    c20_neg_Lneg = 0; c02_neg_Lneg = 0; disc_neg_Lneg = 0; n_Lneg2 = 0

    for _ in range(500000):
        rv = float(np.exp(rng3.uniform(np.log(0.05), np.log(20.0))))
        xv = float(rng3.uniform(0.001, 0.999))
        yv = float(rng3.uniform(0.001, 0.999))
        n_check += 1
        c20v = c20_fn(rv, xv, yv)
        c02v = c02_fn(rv, xv, yv)
        c11v = c11_fn(rv, xv, yv)
        Lv = L_fn(rv, xv, yv)
        disc = c11v**2 - 4*c20v*c02v
        if c20v < 0: c20_neg += 1
        if c02v < 0: c02_neg += 1
        if disc > 0: disc_neg += 1
        if Lv <= 0:
            n_Lneg2 += 1
            if c20v < 0: c20_neg_Lneg += 1
            if c02v < 0: c02_neg_Lneg += 1
            if disc > 0: disc_neg_Lneg += 1

    pr(f"\n  Overall ({n_check} samples):")
    pr(f"    c20 < 0: {c20_neg}")
    pr(f"    c02 < 0: {c02_neg}")
    pr(f"    disc > 0 (not PSD): {disc_neg}")

    pr(f"\n  When L ≤ 0 ({n_Lneg2} samples):")
    pr(f"    c20 < 0: {c20_neg_Lneg}")
    pr(f"    c02 < 0: {c02_neg_Lneg}")
    pr(f"    disc > 0 (not PSD): {disc_neg_Lneg}")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("PART 4: FULL R_surplus_num DECOMPOSITION ATTEMPT")
    pr('='*72)

    # R_surplus_num = c00 + Q2·(p,q) + Q4·(p,q)
    # = c00 + c20p² + c11pq + c02q² + (c40p²+c04q²)(p+q)² + δp²q²
    #
    # On feasible (p,q aren't too large):
    # f2p > 0 → p² < 2(1-x)/9
    # f2q > 0 → q² < 2r³(1-y)/9
    #
    # Can we prove c00 + Q2 + Q4 ≥ 0 when L ≤ 0?
    #
    # The quartic part Q4 = (c40p²+c04q²)(p+q)² + δp²q²
    # If we can show Q2 ≥ 0 as quadratic form (PSD) AND c00 ≥ 0 AND Q4 ≥ 0,
    # then we're done for L ≤ 0.

    # From Part 2: L ≤ 0 ⟹ 3x+3y ≥ 2 ⟹ c00 ≥ 0 (via the factor (3x+3y-2))
    # From Part 3: need Q2 PSD and Q4 ≥ 0 when L ≤ 0

    # Better: group differently. Write:
    # R_surplus_num = c00 + (c20p²+c11pq+c02q²) + (c40p²+c04q²)(p+q)² + δp²q²

    # Actually, consider decomposing into (p+q)² terms:
    # R_surplus_num = c00 + α₂(p+q)² + β₂(p²+q²-2pq)/4·... + (c40p²+c04q²)(p+q)² + δp²q²
    # This is getting complicated. Let me try a different tack.

    # ================================================================
    # NEW APPROACH: Use the structure of each coefficient
    # ================================================================

    pr("\n  Approach: R_surplus_num as function of u=p+q and w=pq")
    pr("  Since Q4 has c31=2c40, c13=2c04, rewrite in (u,w) variables")

    # p²+q² = u²-2w, p⁴+q⁴ = (u²-2w)² - 2w², p³q+pq³ = w(u²-2w)
    # p⁴+2p³q+2pq³+q⁴ = (p²+q²)²+2pq(p²+q²) = (u²-2w)²+2w(u²-2w)
    #                    = (u²-2w)(u²-2w+2w) = u²(u²-2w) = u²(p²+q²)
    # Hmm, let me just substitute directly

    u, w = symbols('u w')
    # p = (u + sqrt(u²-4w))/2, q = (u - sqrt(u²-4w))/2
    # p²+q² = u²-2w, pq = w, p²q² = w²
    # p⁴ = ((u²-2w)+2w+u√(u²-4w))/... this is messy

    # Better: express each monomial type in terms of u=p+q and w=pq
    # p^a q^b + p^b q^a = symmetric, expressible in (e1=u, e2=w)
    # We have:
    # p⁰q⁰ = 1
    # p²+q² = u²-2w
    # pq = w
    # p⁴+q⁴ = u⁴-4u²w+2w²
    # p³q+pq³ = w(u²-2w) = wu²-2w²
    # p²q² = w²

    # R_surplus_num = c00 + c20(u²-2w) + c11·w + c02·???
    # Wait, p²q⁰ ≠ q²p⁰ in general. The coefficients are NOT symmetric in p,q.
    # c20 is coeff of p², c02 is coeff of q². c20 ≠ c02 in general.

    # So the (u,w) substitution doesn't directly help because R_surplus_num
    # is NOT symmetric in p,q.

    # Let me try yet another approach: factor out common terms

    # From the factored forms:
    # c40 = -81r⁴(y-1)(3y+1), c04 = -81(x-1)(3x+1)
    # These factor cleanly. What about c20 and c02?

    pr(f"\n  c20 factored: {factor(c20)}")
    pr(f"  c02 factored: {factor(c02)}")
    pr(f"  c11 factored: {factor(c11)}")

    # c11 = -36r³(x-1)(y-1)(3ry+r+3x+1) — nice!
    # c20 = 6r⁴(y-1)·(something)
    # c02 = -6r(x-1)·(something)

    # Pull out the common (x-1)(y-1) where possible
    # c00 = -4r⁴(x-1)(y-1)(3x+3y-2)W/3
    # c11 = -36r³(x-1)(y-1)(3ry+r+3x+1)
    # c40 = -81r⁴(y-1)(3y+1) = 81r⁴(1-y)(3y+1)
    # c04 = -81(x-1)(3x+1) = 81(1-x)(3x+1)

    # c20: factor(c20) already computed
    # Let's extract the (y-1) factor from c20
    c20_over = cancel(c20 / (r**4*(y-1)))
    pr(f"\n  c20/(r⁴(y-1)) = {factor(c20_over)}")

    c02_over = cancel(c02 / (r*(x-1)))
    pr(f"  c02/(r(x-1)) = {factor(c02_over)}")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("PART 5: R_surplus_num AT SPECIFIC p,q VALUES")
    pr('='*72)

    # Check R_surplus_num along the line p=q (symmetric case)
    R_sym = R_num.subs(q, p)
    R_sym_poly = Poly(R_sym, p)
    pr(f"  R_surplus_num(p,p): degree {R_sym_poly.degree()} in p")
    for i, c in enumerate(R_sym_poly.all_coeffs()):
        if c != 0:
            deg = R_sym_poly.degree() - i
            pr(f"    p^{deg}: {factor(c)}")

    # Check R_surplus_num along p=-q (anti-symmetric case)
    R_anti = R_num.subs(q, -p)
    R_anti_poly = Poly(R_anti, p)
    pr(f"\n  R_surplus_num(p,-p): degree {R_anti_poly.degree()} in p")
    for i, c in enumerate(R_anti_poly.all_coeffs()):
        if c != 0:
            deg = R_anti_poly.degree() - i
            pr(f"    p^{deg}: {factor(c)}")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("PART 6: CHECK COMPLETE SQUARE STRUCTURE IN (p+q)")
    pr('='*72)

    # Rewrite R_surplus_num in terms of u=p+q and v=p-q (or p,q individually)
    # Group by powers of (p+q):
    # R = c00 + [c20·p²+c11·pq+c02·q²] + [c40p⁴+c31p³q+c22p²q²+c13pq³+c04q⁴]
    #
    # p = (u+v)/2, q = (u-v)/2 where u=p+q, v=p-q
    # p² = (u²+2uv+v²)/4, q² = (u²-2uv+v²)/4, pq = (u²-v²)/4
    #
    # R_surplus_num in (u,v):
    v = symbols('v')
    R_uv = R_num.subs([(p, (u+v)/2), (q, (u-v)/2)])
    R_uv_exp = expand(R_uv)
    R_uv_poly = Poly(R_uv_exp, u, v)

    pr("  R_surplus_num in (u=p+q, v=p-q):")
    for monom in sorted(R_uv_poly.as_dict().keys()):
        c = R_uv_poly.as_dict()[monom]
        if c != 0:
            n_t = len(Poly(c, r, x, y).as_dict())
            pr(f"    u^{monom[0]}v^{monom[1]}: {n_t} terms, factored={factor(c)}")

    pr(f"\n  Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
