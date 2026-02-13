#!/usr/bin/env python3
"""R_surplus Gram SOS analysis — fast version.

Focus on:
1. Algebraic structure of quartic and quadratic parts (instant)
2. Fast Gram PSD test (fewer trials, SDP-like approach)
3. Test if R_surplus is always SOS in (p,q) at each (r,x,y)
"""

import numpy as np
import sympy as sp
from sympy import Rational, symbols, expand, Poly, factor, cancel, sqrt
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

    # ================================================================
    pr(f"\n{'='*72}")
    pr("PART 1: QUARTIC FORM ANALYSIS (degree 4 in p,q)")
    pr('='*72)

    c40 = coeffs.get((4,0), sp.Integer(0))
    c31 = coeffs.get((3,1), sp.Integer(0))
    c22 = coeffs.get((2,2), sp.Integer(0))
    c13 = coeffs.get((1,3), sp.Integer(0))
    c04 = coeffs.get((0,4), sp.Integer(0))

    pr(f"\n  Factored quartic coefficients:")
    pr(f"    c40 = {factor(c40)}")
    pr(f"    c31 = {factor(c31)}")
    pr(f"    c22 = {factor(c22)}")
    pr(f"    c13 = {factor(c13)}")
    pr(f"    c04 = {factor(c04)}")

    # Perfect square test: (ap^2 + bpq + cq^2)^2
    # → a^2 p^4 + 2ab p^3q + (b^2+2ac) p^2q^2 + 2bc pq^3 + c^2 q^4
    # c40 = a^2, c04 = c^2, c31 = 2ab, c13 = 2bc
    # If perfect square: c31/c40 = 2b/a, c13/c04 = 2b/c

    ratio_31_40 = cancel(c31/c40)
    ratio_13_04 = cancel(c13/c04)
    pr(f"\n  Perfect square test (ap² + bpq + cq²)²:")
    pr(f"    c31/c40 = {ratio_31_40}  (= 2b/a)")
    pr(f"    c13/c04 = {ratio_13_04}  (= 2b/c)")

    # c22 vs (b^2+2ac) where a^2=c40, c^2=c04, b=a·(c31/c40)/2
    # If c31/c40 = 2 and c13/c04 = 2, then b=a, b=c, so a=b=c
    # quartic = a^2(p^2+pq+q^2)^2, check c22 = 3a^2 = 3·c40
    c22_check = expand(c22 - 3*c40)
    pr(f"    c22 - 3·c40 = {factor(c22_check)}")

    if c22_check == 0:
        pr("    *** QUARTIC = c40·(p²+pq+q²)² — PERFECT SQUARE! ***")
        pr(f"    c40 = {factor(c40)}")
        # Check c40 ≥ 0 on feasible
        c40_factored = factor(c40)
        pr(f"    (c40 ≥ 0 on feasible since x ∈ (0,1), y ∈ (0,1))")
    else:
        pr(f"    NOT a perfect square: residual = {factor(c22_check)}")
        # Try (ap² + bpq + cq²)(dp² + epq + fq²)
        # More general: check discriminant

        # Compute the catalecticant / Hankel matrix of the quartic
        # For binary quartic c40 t^4 + c31 t^3 + c22 t^2 + c13 t + c04
        # with t = p/q, the catalecticant is:
        # | c40  c31/2  c22/3 |
        # | c31/2  c22/3  c13/2 |
        # | c22/3  c13/2  c04  |
        # PSD catalecticant → SOS as sum of two squares

        pr(f"\n  Quartic residual from perfect square:")
        pr(f"    = {factor(c22_check)}")

        # Factor to understand sign
        # Also check: quartic might be c40·(p²+pq+q²)² + correction
        # where correction involves L

    # ================================================================
    pr(f"\n{'='*72}")
    pr("PART 2: QUADRATIC FORM (degree 2 in p,q)")
    pr('='*72)

    c20 = coeffs.get((2,0), sp.Integer(0))
    c11 = coeffs.get((1,1), sp.Integer(0))
    c02 = coeffs.get((0,2), sp.Integer(0))

    pr(f"  c20 = {factor(c20)}")
    pr(f"  c11 = {factor(c11)}")
    pr(f"  c02 = {factor(c02)}")

    # PSD condition: c20 ≥ 0, c02 ≥ 0, 4c20·c02 ≥ c11²
    disc = expand(c11**2 - 4*c20*c02)
    pr(f"\n  Discriminant c11² - 4·c20·c02 = {factor(disc)}")
    pr(f"  ({len(Poly(disc, r, x, y).as_dict())} terms)")

    # Check: is the quadratic form proportional to (p+q)²?
    # (p+q)² = p² + 2pq + q², so need c11 = 2·c20 = 2·c02
    ratio_11_20 = cancel(c11/c20)
    ratio_11_02 = cancel(c11/c02)
    pr(f"\n  c11/c20 = {ratio_11_20}  (= 2 if proportional to (p+q)²)")
    pr(f"  c11/c02 = {ratio_11_02}")

    # Check: maybe c20·p² + c11·pq + c02·q² = α(p+q)² + β(p-q)²
    # = (α+β)p² + 2(α-β)pq + (α+β)q²
    # Need c20 = c02 for this
    pr(f"  c20 - c02 = {factor(expand(c20 - c02))}")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("PART 3: CONSTANT TERM (degree 0)")
    pr('='*72)

    c00 = coeffs.get((0,0), sp.Integer(0))
    pr(f"  c00 = R_surplus_num(p=q=0)")
    c00_factored = factor(c00)
    pr(f"  = {c00_factored}")
    n_c00 = len(Poly(c00, r, x, y).as_dict())
    pr(f"  ({n_c00} terms before factoring)")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("PART 4: FAST NUMERICAL GRAM TEST")
    pr('='*72)

    # Lambdify all coefficients
    coeff_fns = {}
    for m, c in coeffs.items():
        coeff_fns[m] = sp.lambdify((r, x, y), c, 'numpy')

    L = -27*r*x*y + 3*r*x + 9*r*y**2 - 3*r*y + 2*r + 9*x**2 - 27*x*y - 3*x + 3*y + 2
    L_fn = sp.lambdify((r, x, y), L, 'numpy')

    # 6x6 Gram: v = [1, p, q, p², pq, q²]
    v_monoms = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2)]
    n_v = len(v_monoms)

    # v⊗v monomial mapping
    generated = {}
    for a_idx in range(n_v):
        for b_idx in range(a_idx, n_v):
            pa, qa = v_monoms[a_idx]
            pb, qb = v_monoms[b_idx]
            m = (pa+pb, qa+qb)
            if m not in generated:
                generated[m] = []
            generated[m].append((a_idx, b_idx))

    def var_idx(i, j, n):
        if i > j: i, j = j, i
        return i * n - i * (i - 1) // 2 + (j - i)

    n_gram = n_v*(n_v+1)//2  # 21
    all_monoms = sorted(set(generated.keys()) | set(coeffs.keys()))

    # Build constraint matrix (independent of r,x,y)
    A_rows = []
    for m in all_monoms:
        row = np.zeros(n_gram)
        for (a_i, b_i) in generated.get(m, []):
            mult = 1 if a_i == b_i else 2
            row[var_idx(a_i, b_i, n_v)] += mult
        A_rows.append(row)
    A_mat = np.array(A_rows)

    # Precompute SVD for null space (constant across all (r,x,y))
    _, S_vals, Vh = np.linalg.svd(A_mat, full_matrices=True)
    tol = max(A_mat.shape) * S_vals[0] * 1e-10
    rank = np.sum(S_vals > tol)
    null_basis = Vh[rank:].T
    null_dim = null_basis.shape[1]
    pr(f"  Constraint rank: {rank}, null dim: {null_dim}")

    # Precompute index map for building G from g
    idx_pairs = []
    for ii in range(n_v):
        for jj in range(ii, n_v):
            idx_pairs.append((ii, jj, var_idx(ii, jj, n_v)))

    def test_gram_fast(rv_i, xv_i, yv_i, n_trials=5):
        """Try to find PSD Gram matrix at a point — fast version."""
        b_vals = []
        for m in all_monoms:
            if m in coeff_fns:
                b_vals.append(float(coeff_fns[m](rv_i, xv_i, yv_i)))
            else:
                b_vals.append(0.0)
        b_vec = np.array(b_vals)

        g_part, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)

        def neg_min_eig(alpha):
            g = g_part + null_basis @ alpha
            G = np.zeros((n_v, n_v))
            for ii, jj, vi in idx_pairs:
                G[ii, jj] = g[vi]
                G[jj, ii] = g[vi]
            return -np.linalg.eigvalsh(G)[0]

        best = neg_min_eig(np.zeros(null_dim))
        best_alpha = np.zeros(null_dim)

        for trial in range(n_trials):
            a0 = np.random.randn(null_dim) * (0.01 if trial < 3 else 0.1)
            res = scipy_minimize(neg_min_eig, a0, method='Nelder-Mead',
                                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best:
                best = res.fun
                best_alpha = res.x

        min_eig = -best

        # Also get the Gram matrix for inspection
        g_best = g_part + null_basis @ best_alpha
        G_best = np.zeros((n_v, n_v))
        for ii, jj, vi in idx_pairs:
            G_best[ii, jj] = g_best[vi]
            G_best[jj, ii] = g_best[vi]

        return min_eig, G_best

    # --- Test on L <= 0 points ---
    pr(f"\n  Testing Gram on L<=0 points (fast, 50 points)...")
    rng = np.random.default_rng(42)
    psd_Lneg = 0
    n_Lneg = 0
    worst_eig_Lneg = np.inf
    worst_point_Lneg = None

    for _ in range(500):
        rv_i = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
        xv_i = float(rng.uniform(0.01, 0.99))
        yv_i = float(rng.uniform(0.01, 0.99))
        if L_fn(rv_i, xv_i, yv_i) > 0:
            continue

        n_Lneg += 1
        min_eig, G = test_gram_fast(rv_i, xv_i, yv_i, n_trials=5)
        if min_eig < worst_eig_Lneg:
            worst_eig_Lneg = min_eig
            worst_point_Lneg = (rv_i, xv_i, yv_i)
        if min_eig >= -1e-8:
            psd_Lneg += 1

        if n_Lneg % 10 == 0:
            pr(f"    ...{n_Lneg} points, {psd_Lneg} PSD, worst_eig={worst_eig_Lneg:.4e}")

        if n_Lneg >= 50:
            break

    pr(f"\n  L<=0 Gram PSD: {psd_Lneg}/{n_Lneg}")
    pr(f"  Worst min eigenvalue: {worst_eig_Lneg:.6e}")
    if worst_point_Lneg:
        pr(f"  Worst point: r={worst_point_Lneg[0]:.4f}, x={worst_point_Lneg[1]:.4f}, y={worst_point_Lneg[2]:.4f}")

    if psd_Lneg == n_Lneg:
        pr("  *** R_surplus_num is SOS in (p,q) for ALL L<=0 test points! ***")
    else:
        pr(f"  NOT globally SOS on L<=0 ({n_Lneg - psd_Lneg} failures)")

    # --- Test on ALL feasible points ---
    pr(f"\n  Testing Gram on ALL feasible points (50 points)...")
    psd_all = 0
    n_all = 0
    worst_eig_all = np.inf
    worst_point_all = None

    for _ in range(500):
        rv_i = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
        xv_i = float(rng.uniform(0.01, 0.99))
        yv_i = float(rng.uniform(0.01, 0.99))

        n_all += 1
        min_eig, G = test_gram_fast(rv_i, xv_i, yv_i, n_trials=5)
        if min_eig < worst_eig_all:
            worst_eig_all = min_eig
            worst_point_all = (rv_i, xv_i, yv_i)
        if min_eig >= -1e-8:
            psd_all += 1

        if n_all % 10 == 0:
            pr(f"    ...{n_all} points, {psd_all} PSD, worst_eig={worst_eig_all:.4e}")

        if n_all >= 50:
            break

    pr(f"\n  All-points Gram PSD: {psd_all}/{n_all}")
    pr(f"  Worst min eigenvalue: {worst_eig_all:.6e}")

    if psd_all == n_all:
        pr("  *** R_surplus_num is SOS in (p,q) for ALL test points! ***")

    # --- Show Gram at equality point ---
    pr(f"\n  Gram matrix at equality point r=1, x=y=1/3:")
    min_eig_eq, G_eq = test_gram_fast(1.0, 1/3, 1/3, n_trials=10)
    pr(f"  Min eigenvalue: {min_eig_eq:.6e}")
    pr(f"  Gram matrix:\n{np.array2string(G_eq, precision=4, suppress_small=True)}")
    eigs = np.linalg.eigvalsh(G_eq)
    pr(f"  Eigenvalues: {np.array2string(eigs, precision=6)}")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("PART 5: R_surplus STRUCTURE WHEN L > 0 (where R might be negative)")
    pr('='*72)

    # When L > 0, T2 > 0 and R might be negative
    # But T2 + R > 0 always (partition theorem)
    # Question: what does R_surplus look like when L > 0?
    pr("  Sampling R_surplus values for L > 0 region...")

    R_surplus_fn = sp.lambdify((r, x, y, p, q), R_num, 'numpy')

    rng2 = np.random.default_rng(123)
    neg_count = 0
    n_Lpos = 0
    min_R = np.inf

    for _ in range(200000):
        rv = float(np.exp(rng2.uniform(np.log(0.1), np.log(10.0))))
        xv = float(rng2.uniform(0.01, 0.99))
        yv = float(rng2.uniform(0.01, 0.99))
        if L_fn(rv, xv, yv) <= 0:
            continue

        f2p_max = 2*(1 - xv)
        f2q_max = 2*rv**3*(1 - yv)
        p_max = np.sqrt(max(f2p_max/9, 0))
        q_max = np.sqrt(max(f2q_max/9, 0))

        pv = float(rng2.uniform(-p_max, p_max))
        qv = float(rng2.uniform(-q_max, q_max))

        # Check f2c > 0
        f2c_v = 2*(1+rv)**3 - 2*(1+rv)*(3*xv + 3*yv*rv**2 + 2*rv)/3 - 9*(pv+qv)**2
        if f2c_v <= 0:
            continue

        n_Lpos += 1
        Rv = float(R_surplus_fn(rv, xv, yv, pv, qv))
        if Rv < min_R:
            min_R = Rv
        if Rv < -1e-10:
            neg_count += 1

    pr(f"  L>0 feasible samples: {n_Lpos}")
    pr(f"  R_surplus < 0: {neg_count}/{n_Lpos}")
    pr(f"  Min R_surplus: {min_R:.6e}")

    if neg_count > 0:
        pr(f"  R_surplus CAN be negative when L > 0 (expected: T2 compensates)")
    else:
        pr("  *** R_surplus >= 0 EVEN when L > 0! ***")

    pr(f"\n  Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
