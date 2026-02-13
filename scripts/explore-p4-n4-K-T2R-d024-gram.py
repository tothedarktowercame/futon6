#!/usr/bin/env python3
"""Prove K_T2R/r^2 >= 0 via block decomposition + Gram SOS.

Key insight: K_T2R/r^2 = d0+d2+d4+d6 where:
  - d0+d2+d4 >= 0 on ALL 100k feasible samples (degree 4 in p,q)
  - d6 has only 30 terms (degree 6 in p,q)
  - By Hilbert 1888, non-negative bivariate quartic = SOS automatically

Strategy:
1. Check if d0+d2+d4 is GLOBALLY non-negative in (p,q) for feasible (r,x,y)
2. If yes: find 6x6 Gram matrix G(r,x,y) with d0+d2+d4 = v^T G v
3. Analyze d6 and show |d6| <= d0+d2+d4 on feasible domain
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
    s, t, u, v, a, b = symbols('s t u v a b')
    r, x, y, p, q = symbols('r x y p q')

    pr("Building K_T2R/r^2...")

    # T2+R numerator: 8a(s^2-4a)^2 - su^2(s^2+60a)
    def T2R_num(ss, uu, aa):
        return 8*aa*(ss**2 - 4*aa)**2 - ss*uu**2*(ss**2 + 60*aa)

    def T2R_den(ss, uu, aa):
        return 2*(ss**2 + 12*aa)*(2*ss**3 - 8*ss*aa - 9*uu**2)

    S = s + t
    U = u + v
    A = a + b + s*t/6

    num_c = T2R_num(S, U, A)
    den_c = T2R_den(S, U, A)
    num_p = T2R_num(s, u, a)
    den_p = T2R_den(s, u, a)
    num_q = T2R_num(t, v, b)
    den_q = T2R_den(t, v, b)

    surplus_num = expand(num_c*den_p*den_q - num_p*den_c*den_q - num_q*den_c*den_p)
    pr(f"  Surplus numerator computed [{time.time()-t0:.1f}s]")

    subs_norm = {t: r*s, a: x*s**2/4, b: y*r**2*s**2/4,
                 u: p*s**Rational(3,2), v: q*s**Rational(3,2)}
    num_sub = expand(surplus_num.subs(subs_norm))
    K_T2R = expand(num_sub / s**16)
    K_red = sp.cancel(K_T2R / r**2)
    K_red = expand(K_red)
    pr(f"  K_T2R/r^2: {len(Poly(K_red, r,x,y,p,q).as_dict())} terms [{time.time()-t0:.1f}s]")

    # Block decomposition
    poly_pq = Poly(K_red, p, q)
    blocks = {}
    for monom, coeff in poly_pq.as_dict().items():
        i, j = monom
        d = i + j
        blocks[d] = expand(blocks.get(d, sp.Integer(0)) + coeff * p**i * q**j)

    for d in sorted(blocks.keys()):
        n_terms = len(Poly(blocks[d], r, x, y, p, q).as_dict())
        pr(f"  degree {d}: {n_terms} terms")

    # d0+d2+d4
    K024 = expand(blocks.get(0, 0) + blocks.get(2, 0) + blocks.get(4, 0))
    K6 = blocks.get(6, 0)
    pr(f"\n  d0+d2+d4: {len(Poly(K024, r,x,y,p,q).as_dict())} terms")
    pr(f"  d6: {len(Poly(K6, r,x,y,p,q).as_dict())} terms")

    # Check: K_red = K024 + K6
    check = expand(K_red - K024 - K6)
    pr(f"  K_red = K024 + K6? {check == 0}")

    # Lambdify
    K024_fn = sp.lambdify((r, x, y, p, q), K024, 'numpy')
    K6_fn = sp.lambdify((r, x, y, p, q), K6, 'numpy')
    K_fn = sp.lambdify((r, x, y, p, q), K_red, 'numpy')

    # ================================================================
    # TEST 1: Is d0+d2+d4 GLOBALLY non-negative in (p,q)?
    # ================================================================
    pr(f"\n{'='*72}")
    pr("TEST: GLOBAL NON-NEGATIVITY OF d0+d2+d4 IN (p,q)")
    pr('='*72)

    rng = np.random.default_rng(42)
    global_neg = 0
    global_pts = 0
    min_global = np.inf

    for _ in range(2000):
        rv = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
        xv = float(rng.uniform(0.01, 0.99))
        yv = float(rng.uniform(0.01, 0.99))

        # Minimize d0+d2+d4 over ALL (p,q) in R^2 (not just feasible)
        def neg_K024(pq):
            try:
                val = float(K024_fn(rv, xv, yv, pq[0], pq[1]))
                return val if np.isfinite(val) else 1e30
            except:
                return 1e30

        best_val = np.inf
        # Try multiple starting points
        for pv0, qv0 in [(0,0), (0.1,0.1), (-0.1,0.1), (0.3,0), (0,0.3),
                          (0.5, 0.5), (-0.5, -0.5), (1.0, 0), (0, 1.0)]:
            res = scipy_minimize(neg_K024, [pv0, qv0], method='Nelder-Mead',
                                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun

        if best_val < min_global:
            min_global = best_val
        if best_val < -1e-8:
            global_neg += 1
        global_pts += 1

    pr(f"  Tested {global_pts} feasible (r,x,y) points")
    pr(f"  Global minimum of d0+d2+d4 over all (p,q): {min_global:.4e}")
    pr(f"  Points where global min < 0: {global_neg}/{global_pts}")

    if global_neg > 0:
        pr("  *** d0+d2+d4 is NOT globally non-negative in (p,q) ***")
        pr("  Need domain-constrained SOS approach")
    else:
        pr("  *** d0+d2+d4 appears GLOBALLY non-negative in (p,q)! ***")
        pr("  By Hilbert 1888: non-negative bivariate quartic is SOS")

    # ================================================================
    # TEST 2: What are the leading (p,q) coefficients?
    # ================================================================
    pr(f"\n{'='*72}")
    pr("LEADING COEFFICIENTS OF d0+d2+d4")
    pr('='*72)

    # Extract (p,q)-monomial coefficients
    poly024 = Poly(K024, p, q)
    coeffs_024 = {}
    for monom, coeff in poly024.as_dict().items():
        coeffs_024[monom] = expand(coeff)

    pr("  (p,q)-monomials in d0+d2+d4:")
    for m in sorted(coeffs_024.keys()):
        c = coeffs_024[m]
        c_str = str(c)
        if len(c_str) > 100:
            c_str = c_str[:100] + "..."
        pr(f"    p^{m[0]}q^{m[1]}: {c_str}")

    # Factor the leading quartic coefficients
    pr(f"\n  Factoring leading coefficients [{time.time()-t0:.1f}s]...")
    for m_name, m_key in [("p^4", (4,0)), ("p^2*q^2", (2,2)), ("q^4", (0,4)),
                           ("p^3*q", (3,1)), ("p*q^3", (1,3))]:
        if m_key in coeffs_024:
            try:
                f = factor(coeffs_024[m_key])
                f_str = str(f)
                if len(f_str) > 200:
                    f_str = f_str[:200] + "..."
                pr(f"    {m_name}: {f_str}")
            except Exception as e:
                pr(f"    {m_name}: factor failed: {e}")
        else:
            pr(f"    {m_name}: absent")

    # ================================================================
    # TEST 3: d6 structure
    # ================================================================
    pr(f"\n{'='*72}")
    pr("d6 STRUCTURE (30 terms)")
    pr('='*72)

    poly6 = Poly(K6, p, q)
    coeffs_6 = {}
    for monom, coeff in poly6.as_dict().items():
        coeffs_6[monom] = expand(coeff)

    pr("  (p,q)-monomials in d6:")
    for m in sorted(coeffs_6.keys()):
        c = coeffs_6[m]
        c_str = str(c)
        if len(c_str) > 120:
            c_str = c_str[:120] + "..."
        pr(f"    p^{m[0]}q^{m[1]}: {c_str}")

    pr(f"\n  Factoring d6 coefficients...")
    for m in sorted(coeffs_6.keys()):
        try:
            f = factor(coeffs_6[m])
            f_str = str(f)
            if len(f_str) > 200:
                f_str = f_str[:200] + "..."
            pr(f"    p^{m[0]}q^{m[1]}: {f_str}")
        except:
            pr(f"    p^{m[0]}q^{m[1]}: factor failed")

    # ================================================================
    # TEST 4: 6x6 Gram matrix for d0+d2+d4 (if globally non-neg)
    # ================================================================
    if global_neg == 0:
        pr(f"\n{'='*72}")
        pr("6x6 GRAM MATRIX FOR d0+d2+d4")
        pr('='*72)
        pr("v = [1, p, q, p^2, pq, q^2]")

        v_monoms = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2)]
        n_v = len(v_monoms)

        # Build monomial-to-Gram-entry mapping
        generated = {}
        for a_idx in range(n_v):
            for b_idx in range(a_idx, n_v):
                pa, qa = v_monoms[a_idx]
                pb, qb = v_monoms[b_idx]
                m = (pa+pb, qa+qb)
                if m not in generated:
                    generated[m] = []
                generated[m].append((a_idx, b_idx))

        K024_monoms = set(poly024.as_dict().keys())
        pr(f"  v⊗v generates {len(generated)} monomials")
        pr(f"  d0+d2+d4 has {len(K024_monoms)} monomials")
        missing = K024_monoms - set(generated.keys())
        extra = set(generated.keys()) - K024_monoms
        pr(f"  Missing from v⊗v: {missing}")
        pr(f"  v⊗v monomials = 0: {len(extra)}")

        n_gram_vars = n_v * (n_v + 1) // 2
        n_constraints = len(generated)
        pr(f"  Gram matrix: {n_v}x{n_v}, {n_gram_vars} upper triangle entries")
        pr(f"  Constraints: {n_constraints} ({len(K024_monoms)} non-zero + {len(extra)} zero)")
        pr(f"  Free parameters: {n_gram_vars - n_constraints}")

        # Lambdify coefficients
        coeff_fns = {}
        for m, c in poly024.as_dict().items():
            coeff_fns[m] = sp.lambdify((r, x, y), expand(c), 'numpy')

        def var_idx(i, j, n):
            if i > j: i, j = j, i
            return i * n - i * (i - 1) // 2 + (j - i)

        # Test at sampled points
        psd_count = 0
        n_test = 500
        pr(f"\n  Testing {n_test} feasible (r,x,y) points...")

        for idx in range(n_test):
            rv_i = float(np.exp(rng.uniform(np.log(0.2), np.log(5.0))))
            xv_i = float(rng.uniform(0.05, 0.95))
            yv_i = float(rng.uniform(0.05, 0.95))

            # Build linear system
            all_monoms = sorted(set(generated.keys()) | K024_monoms)
            A_rows = []
            b_vals = []
            for m in all_monoms:
                row = np.zeros(n_gram_vars)
                for (a_i, b_i) in generated.get(m, []):
                    mult = 1 if a_i == b_i else 2
                    row[var_idx(a_i, b_i, n_v)] += mult
                A_rows.append(row)
                if m in coeff_fns:
                    b_vals.append(float(coeff_fns[m](rv_i, xv_i, yv_i)))
                else:
                    b_vals.append(0.0)

            A_mat = np.array(A_rows)
            b_vec = np.array(b_vals)

            g_part, _, rank, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            _, S_vals, Vh = np.linalg.svd(A_mat, full_matrices=True)
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
            for trial in range(10):
                a0 = np.random.randn(null_dim) * 0.01
                res = scipy_minimize(neg_min_eig, a0, method='Nelder-Mead',
                                     options={'maxiter': 2000, 'xatol': 1e-12})
                if res.fun < best:
                    best = res.fun

            min_eig = -best
            if min_eig >= -1e-8:
                psd_count += 1

        pr(f"  6x6 Gram PSD: {psd_count}/{n_test}")
        if psd_count == n_test:
            pr("  *** d0+d2+d4 is SOS in (p,q) at ALL test points! ***")
        elif psd_count >= 0.95 * n_test:
            pr(f"  d0+d2+d4 is SOS at {psd_count}/{n_test} — mostly works")
        else:
            pr(f"  d0+d2+d4 is NOT SOS in (p,q) ({n_test-psd_count} failures)")

    # ================================================================
    # TEST 5: Ratio |d6| / (d0+d2+d4) on feasible domain
    # ================================================================
    pr(f"\n{'='*72}")
    pr("RATIO |d6| / (d0+d2+d4) ON FEASIBLE DOMAIN")
    pr('='*72)

    samples = []
    tries = 0
    while len(samples) < 100000 and tries < 3000000:
        tries += 1
        rv = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
        xv = float(rng.uniform(1e-4, 1-1e-4))
        yv = float(rng.uniform(1e-4, 1-1e-4))
        pmax2 = 2*(1-xv)/9
        qmax2 = 2*(rv**3)*(1-yv)/9
        pv = float(rng.uniform(-0.95*np.sqrt(pmax2), 0.95*np.sqrt(pmax2)))
        qv = float(rng.uniform(-0.95*np.sqrt(qmax2), 0.95*np.sqrt(qmax2)))
        S_i = 1 + rv
        U_i = pv + qv
        A_i = xv/4 + yv*rv**2/4 + rv/6
        f2_c = 2*S_i**3 - 8*S_i*A_i - 9*U_i**2
        if f2_c <= 1e-6:
            continue
        samples.append((rv, xv, yv, pv, qv))
    samples = np.array(samples)
    n_samp = len(samples)
    pr(f"  Got {n_samp} feasible samples")

    rv = samples[:, 0]
    xv = samples[:, 1]
    yv = samples[:, 2]
    pv = samples[:, 3]
    qv = samples[:, 4]

    K024_vals = K024_fn(rv, xv, yv, pv, qv)
    K6_vals = K6_fn(rv, xv, yv, pv, qv)
    K_vals = K_fn(rv, xv, yv, pv, qv)

    pr(f"  d0+d2+d4: min={np.min(K024_vals):.4e}, neg={np.sum(K024_vals < -1e-12)}/{n_samp}")
    pr(f"  d6: min={np.min(K6_vals):.4e}, max={np.max(K6_vals):.4e}")
    pr(f"  |d6|/d0+d2+d4 where d024>0:")

    pos_mask = K024_vals > 1e-12
    if np.any(pos_mask):
        ratios = np.abs(K6_vals[pos_mask]) / K024_vals[pos_mask]
        pr(f"    max ratio: {np.max(ratios):.4f}")
        pr(f"    95th percentile: {np.percentile(ratios, 95):.4f}")
        pr(f"    median: {np.median(ratios):.4f}")
        pr(f"    ratio > 1: {np.sum(ratios > 1)}/{np.sum(pos_mask)}")
        if np.max(ratios) < 1:
            pr("  *** |d6| < d0+d2+d4 everywhere on feasible! ***")
            pr("  This means d0+d2+d4 >= 0 alone would suffice to prove K >= 0")

    # ================================================================
    # TEST 6: Alternative decomposition — can we group differently?
    # ================================================================
    pr(f"\n{'='*72}")
    pr("ALTERNATIVE GROUPINGS")
    pr('='*72)

    # Test d0+d4 and d2+d6 separately
    d0_vals = K024_fn(rv, xv, yv, np.zeros_like(pv), np.zeros_like(qv))
    d2_vals = K024_vals - d0_vals - (K024_vals - K024_fn(rv, xv, yv, pv, qv) + K6_vals)

    # Actually let me compute them properly
    from sympy import Integer
    block_fns = {}
    for d in sorted(blocks.keys()):
        block_fns[d] = sp.lambdify((r, x, y, p, q), blocks[d], 'numpy')

    d0_v = block_fns[0](rv, xv, yv, pv, qv)
    d2_v = block_fns[2](rv, xv, yv, pv, qv)
    d4_v = block_fns[4](rv, xv, yv, pv, qv)
    d6_v = block_fns[6](rv, xv, yv, pv, qv)

    pr(f"  d0: min={np.min(d0_v):.4e}, neg={np.sum(d0_v < -1e-12)}/{n_samp}")
    pr(f"  d2: min={np.min(d2_v):.4e}, neg={np.sum(d2_v < -1e-12)}/{n_samp}")
    pr(f"  d4: min={np.min(d4_v):.4e}, neg={np.sum(d4_v < -1e-12)}/{n_samp}")
    pr(f"  d6: min={np.min(d6_v):.4e}, neg={np.sum(d6_v < -1e-12)}/{n_samp}")

    # d0+d4 vs d2+d6
    d04 = d0_v + d4_v
    d26 = d2_v + d6_v
    pr(f"\n  d0+d4: min={np.min(d04):.4e}, neg={np.sum(d04 < -1e-12)}/{n_samp}")
    pr(f"  d2+d6: min={np.min(d26):.4e}, neg={np.sum(d26 < -1e-12)}/{n_samp}")

    # d0+d2 and d4+d6
    d02 = d0_v + d2_v
    d46 = d4_v + d6_v
    pr(f"  d0+d2: min={np.min(d02):.4e}, neg={np.sum(d02 < -1e-12)}/{n_samp}")
    pr(f"  d4+d6: min={np.min(d46):.4e}, neg={np.sum(d46 < -1e-12)}/{n_samp}")

    # ================================================================
    # TEST 7: Can we express K_T2R/r^2 as a product or simple form?
    # ================================================================
    pr(f"\n{'='*72}")
    pr("QUICK FACTORING ATTEMPTS ON d6")
    pr('='*72)

    try:
        d6_factored = factor(K6)
        d6_str = str(d6_factored)
        if len(d6_str) > 500:
            pr(f"  d6 factored: ({len(d6_str)} chars)")
            # Check for common factors
            for tf_name, tf in [("r", r), ("r^2", r**2), ("p", p), ("q", q),
                                ("pq", p*q), ("(x-1)", x-1), ("(y-1)", y-1),
                                ("(1+3x)", 1+3*x), ("(1+3y)", 1+3*y)]:
                q_d, r_d = sp.div(K6, tf, r, x, y, p, q)
                if expand(r_d) == 0:
                    pr(f"    {tf_name} divides d6!")
        else:
            pr(f"  d6 factored: {d6_factored}")
    except Exception as e:
        pr(f"  Factor failed: {e}")

    pr(f"\n  Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
