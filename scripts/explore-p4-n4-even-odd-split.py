#!/usr/bin/env python3
"""Explore the even-odd split of K in (p,q).

Since K only has even total (p,q)-degree, and all monomials p^i q^j
have i,j of the same parity, we can decompose:

  K = A(r,x,y,p²,q²) + pq · B(r,x,y,p²,q²)

where A collects monomials with both i,j even, and B collects
monomials with both i,j odd (divided by pq).

For K >= 0 on the feasible domain, we need:
  A + pq·B >= 0  for all |pq| <= sqrt(Pmax*Qmax)

This requires A >= sqrt(p²q²)|B|, which is equivalent to A >= 0 and A² >= p²q²·B².

Key questions:
1. What is A? What is B?
2. Is A >= 0 on the feasible domain?
3. Is A² - p²q²·B² >= 0 on the feasible domain?
4. Does the even-odd split simplify the problem?
"""

import numpy as np
import sympy as sp
from sympy import Rational


def pr(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def disc_poly(e2, e3, e4):
    return (256*e4**3 - 128*e2**2*e4**2 + 144*e2*e3**2*e4
            + 16*e2**4*e4 - 27*e3**4 - 4*e2**3*e3**2)


def phi4_disc(e2, e3, e4):
    return (-8*e2**5 - 64*e2**3*e4 - 36*e2**2*e3**2
            + 384*e2*e4**2 - 432*e3**2*e4)


def build_K_and_split():
    pr("Building full surplus K...")
    s, t, u, v, a, b = sp.symbols('s t u v a b', positive=True)
    r, x, y, p, q = sp.symbols('r x y p q', real=True)

    inv_phi = lambda e2, e3, e4: sp.cancel(disc_poly(e2, e3, e4) / phi4_disc(e2, e3, e4))
    inv_p = inv_phi(-s, u, a)
    inv_q = inv_phi(-t, v, b)
    inv_c = inv_phi(-(s+t), u+v, a+b+s*t/6)

    surplus = sp.together(inv_c - inv_p - inv_q)
    N, D = sp.fraction(surplus)
    N = sp.expand(N)

    subs = {t: r*s, a: x*s**2/4, b: y*r**2*s**2/4,
            u: p*s**Rational(3,2), v: q*s**Rational(3,2)}
    K = sp.expand(sp.expand(N.subs(subs)) / s**16)
    pr(f"  K: {len(sp.Poly(K, r,x,y,p,q).as_dict())} terms")

    # Split into A (both exponents even) and B (both exponents odd, divided by pq)
    poly_pq = sp.Poly(K, p, q)
    A_expr = sp.Integer(0)
    B_expr = sp.Integer(0)

    for monom, coeff in poly_pq.as_dict().items():
        i, j = monom
        if i % 2 == 0 and j % 2 == 0:
            A_expr = sp.expand(A_expr + coeff * p**i * q**j)
        elif i % 2 == 1 and j % 2 == 1:
            # Factor out pq: p^i q^j = pq * p^(i-1) q^(j-1)
            B_expr = sp.expand(B_expr + coeff * p**(i-1) * q**(j-1))
        else:
            pr(f"  WARNING: mixed parity monomial p^{i}*q^{j}!")

    # Verify: K = A + pq*B
    check = sp.expand(A_expr + p*q*B_expr - K)
    pr(f"  K = A + pq*B? residual zero: {check == 0}")

    # Analyze A
    A_poly = sp.Poly(A_expr, p, q)
    A_terms = len(A_poly.as_dict())
    A_max_deg = max(sum(m) for m in A_poly.as_dict().keys())
    pr(f"\n  A: {A_terms} terms, max (p,q)-degree {A_max_deg}")
    pr(f"  A monomials (p,q):")
    for m in sorted(A_poly.as_dict().keys()):
        i, j = m
        c = sp.expand(A_poly.as_dict()[m])
        nt = len(sp.Poly(c, r, x, y).as_dict())
        pr(f"    p^{i}*q^{j}: {nt} terms")

    # Analyze B
    B_poly = sp.Poly(B_expr, p, q)
    B_terms = len(B_poly.as_dict())
    B_max_deg = max(sum(m) for m in B_poly.as_dict().keys())
    pr(f"\n  B: {B_terms} terms, max (p,q)-degree {B_max_deg}")
    pr(f"  B monomials (p,q):")
    for m in sorted(B_poly.as_dict().keys()):
        i, j = m
        c = sp.expand(B_poly.as_dict()[m])
        nt = len(sp.Poly(c, r, x, y).as_dict())
        pr(f"    p^{i}*q^{j}: {nt} terms")

    # Now substitute P=p², Q=q² in A, and express B in terms of P,Q
    P, Q = sp.symbols('P Q', positive=True)
    A_PQ = sp.Integer(0)
    for monom, coeff in A_poly.as_dict().items():
        i, j = monom  # both even
        A_PQ = sp.expand(A_PQ + coeff * P**(i//2) * Q**(j//2))

    B_PQ = sp.Integer(0)
    for monom, coeff in B_poly.as_dict().items():
        i, j = monom  # p^i q^j where original was p^(i+1) q^(j+1)
        # so P-exponent is i//2 if i is even, but i could be odd here
        # B_expr has p^(i-1) q^(j-1) terms where (i,j) were both odd
        # so the exponents in B_poly are (i-1, j-1) for original (i,j) both odd
        # meaning B_poly monomials have EVEN exponents (since i-1 is even when i odd)
        B_PQ = sp.expand(B_PQ + coeff * P**(i//2) * Q**(j//2))

    pr(f"\n  A(P,Q) = A(p²,q²): {len(sp.Poly(A_PQ, P, Q).as_dict())} terms")
    pr(f"  B(P,Q) = B(p²,q²): {len(sp.Poly(B_PQ, P, Q).as_dict())} terms")

    return (r, x, y, p, q, P, Q), K, A_expr, B_expr, A_PQ, B_PQ


def sample_feasible(rng, n_samples):
    out = []
    tries = 0
    while len(out) < n_samples and tries < 30 * n_samples:
        tries += 1
        rv = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
        xv = float(rng.uniform(1e-4, 1 - 1e-4))
        yv = float(rng.uniform(1e-4, 1 - 1e-4))
        pmax2 = 2*(1-xv)/9
        qmax2 = 2*(rv**3)*(1-yv)/9
        if pmax2 <= 0 or qmax2 <= 0:
            continue
        pv = float(rng.uniform(-0.95*np.sqrt(pmax2), 0.95*np.sqrt(pmax2)))
        qv = float(rng.uniform(-0.95*np.sqrt(qmax2), 0.95*np.sqrt(qmax2)))
        g3 = (6 + 14*rv + 14*rv**2 + 6*rv**3
              - 6*xv - 6*xv*rv - 6*yv*rv**2 - 6*yv*rv**3
              - 27*(pv+qv)**2)
        if g3 <= 1e-6:
            continue
        out.append((rv, xv, yv, pv, qv))
    return np.array(out)


def main():
    syms, K, A_expr, B_expr, A_PQ, B_PQ = build_K_and_split()
    r, x, y, p, q, P, Q = syms

    # Lambdify
    A_fn = sp.lambdify((r, x, y, p, q), A_expr, 'numpy')
    B_fn = sp.lambdify((r, x, y, p, q), B_expr, 'numpy')
    K_fn = sp.lambdify((r, x, y, p, q), K, 'numpy')

    pr(f"\n{'='*72}")
    pr("NUMERICAL TESTS")
    pr('='*72)

    rng = np.random.default_rng(20260213)
    samples = sample_feasible(rng, 100000)
    pr(f"Feasible samples: {len(samples)}")

    rv = samples[:, 0]
    xv = samples[:, 1]
    yv = samples[:, 2]
    pv = samples[:, 3]
    qv = samples[:, 4]

    A_vals = A_fn(rv, xv, yv, pv, qv)
    B_vals = B_fn(rv, xv, yv, pv, qv)
    K_vals = K_fn(rv, xv, yv, pv, qv)
    pq_vals = pv * qv

    n = len(samples)
    pr(f"\n  K:  min={np.min(K_vals):.4e}, neg={np.sum(K_vals < -1e-12)}/{n}")
    pr(f"  A:  min={np.min(A_vals):.4e}, neg={np.sum(A_vals < -1e-12)}/{n}")
    pr(f"  B:  min={np.min(B_vals):.4e}, max={np.max(B_vals):.4e}")
    pr(f"  pq*B: min={np.min(pq_vals*B_vals):.4e}, max={np.max(pq_vals*B_vals):.4e}")

    # Check: is A always >= 0?
    A_neg = np.sum(A_vals < -1e-12)
    pr(f"\n  A >= 0 on feasible? {A_neg == 0} (neg: {A_neg}/{n})")

    if A_neg == 0:
        pr("  *** A >= 0 confirmed! K >= 0 reduces to A² >= p²q²B² ***")

    # Check: A² - p²q²*B²
    margin = A_vals**2 - (pv**2 * qv**2) * B_vals**2
    pr(f"  A²-p²q²B²: min={np.min(margin):.4e}, neg={np.sum(margin < -1e-10)}/{n}")

    # Check: A/|pq*B| ratio (where B != 0)
    mask = np.abs(pq_vals * B_vals) > 1e-15
    if np.sum(mask) > 0:
        ratio = A_vals[mask] / np.abs(pq_vals[mask] * B_vals[mask])
        pr(f"  A/|pq*B| ratio: min={np.min(ratio):.4f}, median={np.median(ratio):.4f}")

    # Check sign of A at p=q=0 (should be K0)
    A_at_0 = A_fn(rv, xv, yv, np.zeros_like(rv), np.zeros_like(rv))
    pr(f"\n  A(p=0,q=0) = K0: min={np.min(A_at_0):.4e}, neg={np.sum(A_at_0 < -1e-12)}/{n}")

    # Block decomposition of A: which (p,q)-degree blocks does A have?
    pr(f"\n{'='*72}")
    pr("A AS A FUNCTION OF (P,Q) = (p²,q²)")
    pr('='*72)

    # At the equality point (r=1, x=y=1/3, p=q=0)
    eq_vals = {r: sp.Integer(1), x: sp.Rational(1,3), y: sp.Rational(1,3)}
    A_PQ_eq = A_PQ.subs(eq_vals)
    B_PQ_eq = B_PQ.subs(eq_vals)
    pr(f"  A(P,Q) at equality (r=1,x=y=1/3): {sp.expand(A_PQ_eq)}")
    pr(f"  B(P,Q) at equality (r=1,x=y=1/3): {sp.expand(B_PQ_eq)}")

    # Check: at equality, Pmax = Qmax = 2*(2/3)/9 = 4/27
    Pmax_eq = sp.Rational(4, 27)
    pr(f"  Pmax = Qmax = {Pmax_eq} at equality")
    pr(f"  max P*Q = {Pmax_eq**2}")

    # Key insight check: for max |pq| = sqrt(Pmax*Qmax),
    # does A >= sqrt(PQ)*|B| at equality extremes?
    pr(f"\n  At equality, checking A(P,Q) >= sqrt(PQ)*|B(P,Q)| at P=Q=Pmax:")
    A_corner = A_PQ_eq.subs({P: Pmax_eq, Q: Pmax_eq})
    B_corner = B_PQ_eq.subs({P: Pmax_eq, Q: Pmax_eq})
    pr(f"    A = {float(A_corner):.6f}")
    pr(f"    B = {float(B_corner):.6f}")
    pr(f"    sqrt(PQ)|B| = {float(Pmax_eq * abs(B_corner)):.6f}")
    pr(f"    A >= sqrt(PQ)|B|? {float(A_corner) >= float(Pmax_eq * abs(B_corner))}")

    # PART 2: Is A globally non-negative in (P,Q)?
    # A(r,x,y,P,Q) viewed as polynomial in (P,Q) with non-negative variables
    pr(f"\n{'='*72}")
    pr("IS A GLOBALLY NON-NEG IN (P,Q)?")
    pr('='*72)

    # Sample with P,Q going beyond feasible
    A_PQ_fn = sp.lambdify((r, x, y, P, Q), A_PQ, 'numpy')
    rng2 = np.random.default_rng(42)
    neg_a = 0
    n_test = 50000
    for _ in range(n_test):
        rv_i = float(np.exp(rng2.uniform(np.log(0.1), np.log(10.0))))
        xv_i = float(rng2.uniform(0.01, 0.99))
        yv_i = float(rng2.uniform(0.01, 0.99))
        # P, Q can be anything non-negative
        Pv = float(rng2.uniform(0, 2.0))  # well beyond Pmax ~ 0.2
        Qv = float(rng2.uniform(0, 2.0 * rv_i**3))
        try:
            val = float(A_PQ_fn(rv_i, xv_i, yv_i, Pv, Qv))
            if np.isfinite(val) and val < -1e-10:
                neg_a += 1
                if neg_a <= 3:
                    pr(f"  A<0: r={rv_i:.3f} x={xv_i:.3f} y={yv_i:.3f} "
                       f"P={Pv:.4f} Q={Qv:.4f} A={val:.4e}")
        except:
            pass
    pr(f"  Global test: A<0 at {neg_a}/{n_test} points")

    # PART 3: Structure of A² - PQ*B² on feasible domain
    pr(f"\n{'='*72}")
    pr("A² - PQ*B² ANALYSIS")
    pr('='*72)

    # For the feasible domain, P < Pmax(x), Q < Qmax(r,y)
    # So PQ < Pmax*Qmax = (2(1-x)/9)*(2r³(1-y)/9) = 4r³(1-x)(1-y)/81
    # Check if A² - PQ*B² >= 0 on feasible:
    margin_vals = A_vals**2 - (pv**2 * qv**2) * B_vals**2
    pr(f"  A²-p²q²B²: min={np.min(margin_vals):.4e}, neg={np.sum(margin_vals < -1e-10)}/{n}")

    # At what fraction of feasible boundary (p²→Pmax or q²→Qmax) is the margin tightest?
    boundary_samples = []
    for _ in range(20000):
        rv_i = float(np.exp(rng2.uniform(np.log(0.1), np.log(10.0))))
        xv_i = float(rng2.uniform(0.01, 0.99))
        yv_i = float(rng2.uniform(0.01, 0.99))
        Pmax_i = 2*(1-xv_i)/9
        Qmax_i = 2*rv_i**3*(1-yv_i)/9
        # Sample at boundary: p² = 0.99*Pmax, q² = 0.99*Qmax
        pv_i = np.sqrt(0.99*Pmax_i)
        qv_i = np.sqrt(0.99*Qmax_i)
        # Check both signs of p and q
        for sp_sign, sq_sign in [(1,1), (1,-1), (-1,1), (-1,-1)]:
            boundary_samples.append((rv_i, xv_i, yv_i, sp_sign*pv_i, sq_sign*qv_i))

    boundary_arr = np.array(boundary_samples)
    A_bnd = A_fn(boundary_arr[:,0], boundary_arr[:,1], boundary_arr[:,2],
                 boundary_arr[:,3], boundary_arr[:,4])
    B_bnd = B_fn(boundary_arr[:,0], boundary_arr[:,1], boundary_arr[:,2],
                 boundary_arr[:,3], boundary_arr[:,4])
    pq_bnd = boundary_arr[:,3] * boundary_arr[:,4]
    K_bnd = A_bnd + pq_bnd * B_bnd
    margin_bnd = A_bnd**2 - (boundary_arr[:,3]**2 * boundary_arr[:,4]**2) * B_bnd**2

    pr(f"\n  At boundary (p²≈Pmax, q²≈Qmax):")
    pr(f"    K min: {np.min(K_bnd):.4e}, neg: {np.sum(K_bnd < -1e-10)}/{len(boundary_arr)}")
    pr(f"    A min: {np.min(A_bnd):.4e}, neg: {np.sum(A_bnd < -1e-10)}/{len(boundary_arr)}")
    pr(f"    margin min: {np.min(margin_bnd):.4e}, neg: {np.sum(margin_bnd < -1e-10)}/{len(boundary_arr)}")

    # PART 4: Alternative decomposition using Cauchy-Schwarz on the blocks
    pr(f"\n{'='*72}")
    pr("CAUCHY-SCHWARZ BETWEEN K0+K2+K4 AND K8 TO BOUND K6")
    pr('='*72)
    pr("If K6² <= 4*(K0+K2+K4)*K8, then K = (K0+K2+K4) + K6 + K8 >= 0")
    pr("because (√(K0+K2+K4) + sign·√K8)² >= 0 gives K0+K2+K4+K8 >= ±2√((K0+K2+K4)·K8) >= |K6|")

    # Rebuild blocks
    poly_pq = sp.Poly(K, p, q)
    blocks = {}
    for monom, coeff in poly_pq.as_dict().items():
        i, j = monom
        d = i + j
        blocks[d] = sp.expand(blocks.get(d, sp.Integer(0)) + coeff * p**i * q**j)

    block_fns = {d: sp.lambdify((r, x, y, p, q), blocks[d], 'numpy')
                 for d in sorted(blocks.keys())}

    K024 = block_fns[0](rv, xv, yv, pv, qv) + block_fns[2](rv, xv, yv, pv, qv) + block_fns[4](rv, xv, yv, pv, qv)
    K6_v = block_fns[6](rv, xv, yv, pv, qv)
    K8_v = block_fns[8](rv, xv, yv, pv, qv)

    # Check: K6² <= 4*(K0+K2+K4)*K8?
    lhs = K6_v**2
    rhs = 4 * K024 * K8_v
    cs_holds = np.sum(lhs <= rhs + 1e-10)
    cs_ratio = lhs / np.maximum(rhs, 1e-30)
    pr(f"  K6² <= 4*(K024)*K8: {cs_holds}/{n}")
    pr(f"  K6²/(4·K024·K8): max={np.max(cs_ratio):.4f}, median={np.median(cs_ratio):.6f}")

    if cs_holds < n:
        # Where does it fail?
        fail_mask = lhs > rhs + 1e-10
        pr(f"  Fails at {np.sum(fail_mask)} points")
        fail_ratio = cs_ratio[fail_mask]
        pr(f"  Max ratio at failures: {np.max(fail_ratio):.6f}")
        # Sample a failure
        fail_idx = np.where(fail_mask)[0][0]
        pr(f"  Example failure: r={rv[fail_idx]:.4f} x={xv[fail_idx]:.4f} y={yv[fail_idx]:.4f}")
        pr(f"    p={pv[fail_idx]:.6f} q={qv[fail_idx]:.6f}")
        pr(f"    K024={K024[fail_idx]:.4e} K6={K6_v[fail_idx]:.4e} K8={K8_v[fail_idx]:.4e}")
        pr(f"    K6²={lhs[fail_idx]:.4e} 4*K024*K8={rhs[fail_idx]:.4e}")

    # Also check AM-GM variant: does K024 + K8 >= 2*|K6| ?
    # This is weaker but might hold since K6 is "small" relative to K024
    am_gm = K024 + K8_v - 2*np.abs(K6_v)
    pr(f"\n  K024 + K8 - 2|K6|: min={np.min(am_gm):.4e}, neg={np.sum(am_gm < -1e-10)}/{n}")

    # Check: does K024 >= |K6| + |K8| (absolute domination)?
    abs_dom = K024 - np.abs(K6_v) - K8_v
    pr(f"  K024 - |K6| - K8: min={np.min(abs_dom):.4e}, neg={np.sum(abs_dom < -1e-10)}/{n}")

    # Since K8 >= 0, the question is really: K024 + K6 >= 0?
    K0246 = K024 + K6_v
    pr(f"\n  K0+K2+K4+K6: min={np.min(K0246):.4e}, neg={np.sum(K0246 < -1e-10)}/{n}")
    if np.sum(K0246 < -1e-10) == 0:
        pr("  *** K0+K2+K4+K6 >= 0 on feasible! K = (K0+K2+K4+K6) + K8 >= 0 ***")
    else:
        # How big is K8 at the points where K0246 < 0?
        neg_mask = K0246 < -1e-10
        pr(f"  At K0+K2+K4+K6 < 0: K8 min={np.min(K8_v[neg_mask]):.4e}")
        pr(f"  K8 > |K0246| at those points? {np.all(K8_v[neg_mask] > -K0246[neg_mask])}")


if __name__ == '__main__':
    main()
