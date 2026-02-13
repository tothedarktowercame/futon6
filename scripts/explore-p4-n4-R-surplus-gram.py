#!/usr/bin/env python3
"""Try Gram matrix SOS for R_surplus_num when L <= 0.

R_surplus_num is 92 terms, degree 4 in (p,q), 9 (p,q)-monomial types.
Try: R_surplus_num = v^T G v where v = [1, p, q, p^2, pq, q^2]^T (6x6 Gram).

Also try domain-constrained: R_surplus_num = σ₀ + σ₁(Pmax-p²) + σ₂(Qmax-q²)
"""

import numpy as np
import sympy as sp
from sympy import Rational, symbols, expand, Poly, factor
from scipy.optimize import minimize as scipy_minimize
import time


def pr(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


def main():
    t0 = time.time()
    r, x, y, p, q = symbols('r x y p q')

    pr("Building R_surplus_num...")

    # R components
    C_p = x - 1; f1p = 1 + 3*x; f2p = 2*(1-x) - 9*p**2
    C_q = r**2*(y - 1); f1q = r**2*(1 + 3*y); f2q = 2*r**3*(1-y) - 9*q**2
    Sv = 1 + r; Av12 = 3*x + 3*y*r**2 + 2*r
    C_c = expand(Av12/3 - Sv**2)
    f1c = expand(Sv**2 + Av12)
    f2c = expand(2*Sv**3 - 2*Sv*Av12/3 - 9*(p+q)**2)

    R_num = expand(C_c*f1c*f2p*f2q - C_p*f1p*f2c*f2q - C_q*f1q*f2c*f2p)
    pr(f"  R_surplus_num: {len(Poly(R_num, r,x,y,p,q).as_dict())} terms [{time.time()-t0:.1f}s]")

    # Extract (p,q)-monomial coefficients
    poly_pq = Poly(R_num, p, q)
    coeffs = {}
    for monom, coeff in poly_pq.as_dict().items():
        coeffs[monom] = expand(coeff)

    pr("  Monomials in R_surplus_num:")
    for m in sorted(coeffs.keys()):
        n_t = len(Poly(coeffs[m], r, x, y).as_dict())
        pr(f"    p^{m[0]}q^{m[1]}: {n_t} terms")

    # Lambdify coefficients
    coeff_fns = {}
    for m, c in coeffs.items():
        coeff_fns[m] = sp.lambdify((r, x, y), c, 'numpy')

    L = -27*r*x*y + 3*r*x + 9*r*y**2 - 3*r*y + 2*r + 9*x**2 - 27*x*y - 3*x + 3*y + 2
    L_fn = sp.lambdify((r, x, y), L, 'numpy')

    # ================================================================
    pr(f"\n{'='*72}")
    pr("6x6 GRAM MATRIX TEST FOR R_surplus_num")
    pr('='*72)
    pr("v = [1, p, q, p^2, pq, q^2]")

    v_monoms = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2)]
    n_v = len(v_monoms)

    # Build v⊗v monomial mapping
    generated = {}
    for a_idx in range(n_v):
        for b_idx in range(a_idx, n_v):
            pa, qa = v_monoms[a_idx]
            pb, qb = v_monoms[b_idx]
            m = (pa+pb, qa+qb)
            if m not in generated:
                generated[m] = []
            generated[m].append((a_idx, b_idx))

    R_monoms = set(coeffs.keys())
    pr(f"  v⊗v generates {len(generated)} monomials")
    pr(f"  R_surplus_num has {len(R_monoms)} monomials")
    missing = R_monoms - set(generated.keys())
    extra = set(generated.keys()) - R_monoms
    pr(f"  Missing: {missing}")
    pr(f"  v⊗v = 0: {len(extra)} ({sorted(extra)})")

    n_gram = n_v*(n_v+1)//2
    n_constr = len(generated)
    pr(f"  Gram entries: {n_gram}, constraints: {n_constr}")
    pr(f"  Free parameters: {n_gram - n_constr}")

    def var_idx(i, j, n):
        if i > j: i, j = j, i
        return i * n - i * (i - 1) // 2 + (j - i)

    # Test at sample points
    rng = np.random.default_rng(42)
    all_monoms = sorted(set(generated.keys()) | R_monoms)

    # Build constraint matrix (constant, doesn't depend on r,x,y)
    A_rows = []
    for m in all_monoms:
        row = np.zeros(n_gram)
        for (a_i, b_i) in generated.get(m, []):
            mult = 1 if a_i == b_i else 2
            row[var_idx(a_i, b_i, n_v)] += mult
        A_rows.append(row)
    A_mat = np.array(A_rows)

    def test_gram(rv_i, xv_i, yv_i, n_trials=30):
        """Try to find PSD Gram matrix at a point."""
        b_vals = []
        for m in all_monoms:
            if m in coeff_fns:
                b_vals.append(float(coeff_fns[m](rv_i, xv_i, yv_i)))
            else:
                b_vals.append(0.0)
        b_vec = np.array(b_vals)

        g_part, _, rank, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        _, S_vals, Vh = np.linalg.svd(A_mat, full_matrices=True)
        tol = max(A_mat.shape) * S_vals[0] * 1e-10
        rank = np.sum(S_vals > tol)
        null_basis = Vh[rank:].T
        null_dim = null_basis.shape[1]

        def neg_min_eig(alpha):
            g = g_part + null_basis @ alpha
            G = np.zeros((n_v, n_v))
            for ii in range(n_v):
                for jj in range(ii, n_v):
                    G[ii, jj] = g[var_idx(ii, jj, n_v)]
                    G[jj, ii] = G[ii, jj]
            return -np.linalg.eigvalsh(G)[0]

        best = np.inf
        best_alpha = None
        for trial in range(n_trials):
            a0 = np.random.randn(null_dim) * 0.001 if trial > 0 else np.zeros(null_dim)
            res = scipy_minimize(neg_min_eig, a0, method='Nelder-Mead',
                                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16})
            if res.fun < best:
                best = res.fun
                best_alpha = res.x

        return -best, best_alpha  # returns min eigenvalue

    # Test on L <= 0 points
    pr(f"\n  Testing Gram on L<=0 points...")
    psd_count_Lneg = 0
    n_test_Lneg = 0
    min_eig_Lneg = np.inf

    for _ in range(2000):
        rv_i = float(np.exp(rng.uniform(np.log(0.2), np.log(5.0))))
        xv_i = float(rng.uniform(0.05, 0.95))
        yv_i = float(rng.uniform(0.05, 0.95))
        Lv = L_fn(rv_i, xv_i, yv_i)
        if Lv > 0:
            continue

        n_test_Lneg += 1
        min_eig, _ = test_gram(rv_i, xv_i, yv_i, n_trials=15)
        if min_eig < min_eig_Lneg:
            min_eig_Lneg = min_eig
        if min_eig >= -1e-8:
            psd_count_Lneg += 1

        if n_test_Lneg >= 200:
            break

    pr(f"  L<=0 Gram PSD: {psd_count_Lneg}/{n_test_Lneg}")
    pr(f"  Min eigenvalue: {min_eig_Lneg:.4e}")

    if psd_count_Lneg == n_test_Lneg:
        pr("  *** R_surplus_num is SOS in (p,q) for ALL L<=0 test points! ***")
        pr("  By Hilbert 1888 (bivariate quartic), this is guaranteed if globally non-neg.")
    else:
        pr(f"  NOT globally SOS ({n_test_Lneg - psd_count_Lneg} failures)")
        pr("  Need domain-constrained SOS")

    # Test on ALL points (including L > 0)
    pr(f"\n  Testing Gram on ALL feasible points...")
    psd_count_all = 0
    n_test_all = 0

    for _ in range(1000):
        rv_i = float(np.exp(rng.uniform(np.log(0.2), np.log(5.0))))
        xv_i = float(rng.uniform(0.05, 0.95))
        yv_i = float(rng.uniform(0.05, 0.95))

        n_test_all += 1
        min_eig, _ = test_gram(rv_i, xv_i, yv_i, n_trials=10)
        if min_eig >= -1e-8:
            psd_count_all += 1

        if n_test_all >= 200:
            break

    pr(f"  All-points Gram PSD: {psd_count_all}/{n_test_all}")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("FACTOR THE QUARTIC FORM (degree 4 part)")
    pr('='*72)

    # Degree 4 part of R_surplus_num
    d4_part = sp.Integer(0)
    for m, c in coeffs.items():
        if m[0] + m[1] == 4:
            d4_part = expand(d4_part + c * p**m[0] * q**m[1])

    d4_factored = factor(d4_part)
    pr(f"  d4 part: {d4_factored}")

    # Factor each coefficient
    for m in sorted(coeffs.keys()):
        if m[0]+m[1] == 4:
            f = factor(coeffs[m])
            pr(f"    p^{m[0]}q^{m[1]}: {f}")

    # Check if d4 is a perfect square or product of quadratics
    # d4 = -27·(quartic)
    # The quartic might be (αp²+βpq+γq²)² or (αp²+βpq+γq²)(δp²+εpq+ζq²)
    # Check discriminant structure

    # Extract the quartic coefficients as functions of (r,x,y)
    c40 = coeffs.get((4,0), sp.Integer(0))
    c31 = coeffs.get((3,1), sp.Integer(0))
    c22 = coeffs.get((2,2), sp.Integer(0))
    c13 = coeffs.get((1,3), sp.Integer(0))
    c04 = coeffs.get((0,4), sp.Integer(0))

    pr(f"\n  Quartic coefficients:")
    pr(f"    c40 = {factor(c40)}")
    pr(f"    c31 = {factor(c31)}")
    pr(f"    c22 = {factor(c22)}")
    pr(f"    c13 = {factor(c13)}")
    pr(f"    c04 = {factor(c04)}")

    # Check: is the quartic c40*p^4 + c31*p^3*q + c22*p^2*q^2 + c13*p*q^3 + c04*q^4
    # a perfect square? Compare with (ap^2+bpq+cq^2)^2 = a^2p^4 + 2abp^3q + (b^2+2ac)p^2q^2 + 2bcpq^3 + c^2q^4
    # So: c40 = a^2, c04 = c^2, c31 = 2ab, c13 = 2bc
    # a = sqrt(c40), c = sqrt(c04), b = c31/(2a) = c13/(2c)
    # Check: b^2+2ac = c22?

    pr(f"\n  Testing if quartic is a perfect square:")
    # c40 = -27·(9r^4y^2-6r^4y-3r^4) = -27·3r^4(3y^2-2y-1) = -81r^4(3y+1)(y-1)
    # c04 = -27·(9x^2-6x-3) = -81(3x+1)(x-1)
    # For perfect square: need c40 ≥ 0 and c04 ≥ 0
    # But (y-1) < 0 on feasible, so c40 = 81r^4(3y+1)(1-y) > 0 ✓
    # And (x-1) < 0 on feasible, so c04 = 81(3x+1)(1-x) > 0 ✓

    pr(f"    c40 = 81r⁴(3y+1)(1-y) (positive on feasible)")
    pr(f"    c04 = 81(3x+1)(1-x) (positive on feasible)")
    pr(f"    √c40 = 9r²√((3y+1)(1-y))")
    pr(f"    √c04 = 9√((3x+1)(1-x))")

    # c31 = 2·√c40·b = -27·(18r^4y^2-12r^4y-6r^4) = -27·6r^4(3y^2-2y-1) = -162r^4(3y+1)(y-1)
    # = 162r^4(3y+1)(1-y) = 2·c40
    pr(f"    c31 = {factor(c31)}")
    pr(f"    c31/c40 = {cancel(c31/c40)} (should be 2b/a for perfect square)")

    # Similarly c13/c04
    pr(f"    c13 = {factor(c13)}")
    pr(f"    c13/c04 = {cancel(c13/c04)}")

    # c31 = 2·c40 means b = a (coefficient of p²) → quartic starts as a²(p²+pq)²?
    # No: c31 = 2ab and c40 = a². c31/c40 = 2b/a = 2 → b = a.
    # Similarly c13/c04 = 2b/c = 2 → b = c.
    # So if a = b = c: quartic = a²p⁴ + 2a²p³q + (a²+2a²)p²q² + 2a²pq³ + a²q⁴
    # = a²(p⁴+2p³q+3p²q²+2pq³+q⁴) = a²((p²+pq)²+2p²q²+q⁴)???
    # No: (p²+pq+q²)² = p⁴+2p³q+3p²q²+2pq³+q⁴. Yes!

    # So the quartic might be a²(p²+pq+q²)² where a²=c40.
    # Check: c22 should equal 3·c40 (= a²·3)
    c22_expected = 3*c40
    check = expand(c22 - c22_expected)
    pr(f"\n    c22 - 3·c40 = {factor(check)}")
    if check == 0:
        pr("    *** Quartic = c40·(p²+pq+q²)² — PERFECT SQUARE! ***")
    else:
        pr(f"    Not a perfect square of (p²+pq+q²)")
        # Check: maybe (p²+αpq+q²)² for some α
        # (p²+αpq+q²)² = p⁴+2αp³q+(α²+2)p²q²+2αpq³+q⁴
        # Need: c31 = 2α·c40 → α = c31/(2c40) = 2/2 = 1 ← already checked above
        # And c22 = (α²+2)·c40 = 3·c40 ← checked above

        # Try (p²+pq+q²)² + correction:
        pr(f"    Correction: {factor(check)}")
        # If correction is L-dependent, this connects to the overall structure

    # ================================================================
    pr(f"\n{'='*72}")
    pr("QUADRATIC FORM ANALYSIS OF DEGREE-2 PART")
    pr('='*72)

    c20 = coeffs.get((2,0), sp.Integer(0))
    c11 = coeffs.get((1,1), sp.Integer(0))
    c02 = coeffs.get((0,2), sp.Integer(0))

    pr(f"  c20 (coeff of p²) = {factor(c20)}")
    pr(f"  c11 (coeff of pq) = {factor(c11)}")
    pr(f"  c02 (coeff of q²) = {factor(c02)}")

    # For the quadratic form c20·p² + c11·pq + c02·q² to be PSD:
    # Need c20 ≥ 0, c02 ≥ 0, and 4·c20·c02 ≥ c11²
    pr(f"\n  c11/c20 = {cancel(c11/c20)}")
    pr(f"  c11/c02 = {cancel(c11/c02)}")
    pr(f"  c11² - 4·c20·c02 = {factor(expand(c11**2 - 4*c20*c02))}")

    pr(f"\n  Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
