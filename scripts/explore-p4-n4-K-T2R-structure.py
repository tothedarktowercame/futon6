#!/usr/bin/env python3
"""Analyze the structure of K_T2R — the polynomial we need to prove non-negative.

K_T2R = numerator of T2+R surplus. 487 terms, max (p,q)-degree 6, r^2 | K_T2R.

Key questions:
1. Is K_T2R/r^2 globally non-negative in (p,q)? (Original K was NOT)
2. What is the block structure of K_T2R/r^2?
3. Does K_T2R/r^2 have a nice Gram matrix representation?
4. Can we prove K_T2R/r^2 >= 0 via a simpler certificate?
"""

import numpy as np
import sympy as sp
from sympy import Rational, symbols, expand, cancel, together, fraction, factor, Poly
import time


def pr(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


def main():
    t0 = time.time()
    s, t, u, v, a, b = symbols('s t u v a b')
    r, x, y, p, q = symbols('r x y p q')

    pr("Building K_T2R...")

    # T2+R numerator: 8a(s^2-4a)^2 - su^2(s^2+60a)
    # T2+R denominator: 2*(s^2+12a)*(2s^3-8sa-9u^2)
    def T2R_num(ss, uu, aa):
        return 8*aa*(ss**2 - 4*aa)**2 - ss*uu**2*(ss**2 + 60*aa)

    def T2R_den(ss, uu, aa):
        return 2*(ss**2 + 12*aa)*(2*ss**3 - 8*ss*aa - 9*uu**2)

    # Build surplus numerator
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
    pr(f"  Surplus numerator: {len(Poly(surplus_num, s,t,u,v,a,b).as_dict())} terms [{time.time()-t0:.1f}s]")

    # Normalize
    subs_norm = {t: r*s, a: x*s**2/4, b: y*r**2*s**2/4,
                 u: p*s**Rational(3,2), v: q*s**Rational(3,2)}
    num_sub = expand(surplus_num.subs(subs_norm))
    K_T2R = expand(num_sub / s**16)
    n_terms = len(Poly(K_T2R, r,x,y,p,q).as_dict())
    pr(f"  K_T2R: {n_terms} terms [{time.time()-t0:.1f}s]")

    # Divide by r^2
    K_red = sp.cancel(K_T2R / r**2)
    K_red = expand(K_red)
    n_red = len(Poly(K_red, r,x,y,p,q).as_dict())
    pr(f"  K_T2R/r^2: {n_red} terms [{time.time()-t0:.1f}s]")

    # Lambdify for fast numerical checks
    K_fn = sp.lambdify((r, x, y, p, q), K_red, 'numpy')

    pr(f"\n{'='*72}")
    pr("GLOBAL NON-NEGATIVITY CHECK FOR K_T2R/r^2")
    pr('='*72)

    rng = np.random.default_rng(42)

    # Test at increasing multiples of feasible domain
    for scale_name, scale in [("1x feasible", 1.0), ("1.5x", 1.5), ("2x", 2.0), ("5x", 5.0)]:
        neg_count = 0
        n_pts = 30000
        min_val = np.inf
        for _ in range(n_pts):
            rv = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
            xv = float(rng.uniform(0.01, 0.99))
            yv = float(rng.uniform(0.01, 0.99))
            pmax = np.sqrt(max(0, 2*(1-xv)/9))
            qmax = np.sqrt(max(0, 2*rv**3*(1-yv)/9))
            pv = float(rng.uniform(-scale*pmax, scale*pmax))
            qv = float(rng.uniform(-scale*qmax, scale*qmax))
            try:
                val = float(K_fn(rv, xv, yv, pv, qv))
                if np.isfinite(val):
                    if val < min_val:
                        min_val = val
                    if val < -1e-10:
                        neg_count += 1
            except:
                pass
        pr(f"  {scale_name}: neg={neg_count}/{n_pts}, min={min_val:.4e}")

    # Block decomposition of K_red
    pr(f"\n{'='*72}")
    pr("BLOCK DECOMPOSITION OF K_T2R/r^2")
    pr('='*72)

    poly_pq = Poly(K_red, p, q)
    blocks = {}
    for monom, coeff in poly_pq.as_dict().items():
        i, j = monom
        d = i + j
        blocks[d] = expand(blocks.get(d, sp.Integer(0)) + coeff * p**i * q**j)

    block_fns = {}
    for d in sorted(blocks.keys()):
        n_block = len(Poly(blocks[d], r, x, y, p, q).as_dict())
        pr(f"  degree {d}: {n_block} terms")
        block_fns[d] = sp.lambdify((r, x, y, p, q), blocks[d], 'numpy')

    # Sample feasible domain
    pr(f"\n  Feasible domain sampling (100k)...")
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
    n = len(samples)
    pr(f"  Got {n} samples")

    rv = samples[:, 0]
    xv = samples[:, 1]
    yv = samples[:, 2]
    pv = samples[:, 3]
    qv = samples[:, 4]

    # Check each block
    for d in sorted(blocks.keys()):
        vals = block_fns[d](rv, xv, yv, pv, qv)
        pr(f"  K_T2R_d{d}: min={np.min(vals):.4e}, neg={np.sum(vals < -1e-12)}/{n}")

    # Partial sums
    cumul = np.zeros(n)
    for d in sorted(blocks.keys()):
        cumul += block_fns[d](rv, xv, yv, pv, qv)
        pr(f"  Sum through d{d}: min={np.min(cumul):.4e}, neg={np.sum(cumul < -1e-12)}/{n}")

    # Full K_T2R/r^2
    K_vals = K_fn(rv, xv, yv, pv, qv)
    pr(f"\n  K_T2R/r^2 on feasible: min={np.min(K_vals):.4e}, neg={np.sum(K_vals < -1e-12)}/{n}")

    # Key: is K_T2R/r^2 GLOBALLY non-negative?
    # This is the critical question. If yes, we have a direct SOS proof.
    pr(f"\n{'='*72}")
    pr("GRAM MATRIX FOR K_T2R/r^2 (degree 6 in p,q)")
    pr('='*72)
    pr("For degree-6 polynomial in (p,q), use v = [1, p, q, p^2, pq, q^2, p^3, p^2q, pq^2, q^3]")
    pr("Gram matrix is 10x10 with many zero constraints from missing monomials")

    # Count v-monomial products
    v_monoms = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), (3,0), (2,1), (1,2), (0,3)]
    n_v = len(v_monoms)
    generated = {}
    for a_idx in range(n_v):
        for b_idx in range(a_idx, n_v):
            pa, qa = v_monoms[a_idx]
            pb, qb = v_monoms[b_idx]
            m = (pa+pb, qa+qb)
            if m not in generated:
                generated[m] = []
            generated[m].append((a_idx, b_idx))

    K_monoms = set(poly_pq.as_dict().keys())
    pr(f"\n  v⊗v generates {len(generated)} monomials")
    pr(f"  K_T2R/r^2 has {len(K_monoms)} monomials")
    missing_from_gram = K_monoms - set(generated.keys())
    extra_in_gram = set(generated.keys()) - K_monoms
    pr(f"  K monomials not in v⊗v: {missing_from_gram}")
    pr(f"  v⊗v monomials = 0: {len(extra_in_gram)}")

    n_gram_vars = n_v * (n_v + 1) // 2
    n_constraints = len(generated)
    pr(f"  Gram matrix: {n_v}x{n_v}, {n_gram_vars} upper triangle entries")
    pr(f"  Constraints: {n_constraints} ({len(K_monoms)} non-zero + {len(extra_in_gram)} zero)")
    pr(f"  Free parameters: {n_gram_vars - n_constraints}")

    # Quick numerical Gram test at a few points
    if len(missing_from_gram) == 0:
        pr(f"\n  Testing 10x10 Gram at 100 feasible (r,x,y) points...")
        from scipy.optimize import minimize as scipy_minimize

        coeff_fns = {}
        for m, c in poly_pq.as_dict().items():
            coeff_fns[m] = sp.lambdify((r, x, y), expand(c), 'numpy')

        def var_idx(i, j, n):
            if i > j: i, j = j, i
            return i * n - i * (i - 1) // 2 + (j - i)

        psd_count = 0
        for idx in range(100):
            rv_i = float(np.exp(rng.uniform(np.log(0.3), np.log(3.0))))
            xv_i = float(rng.uniform(0.1, 0.9))
            yv_i = float(rng.uniform(0.1, 0.9))

            # Build constraint system
            n_vars = n_gram_vars
            all_monoms = sorted(set(generated.keys()) | K_monoms)
            A_rows = []
            b_vals = []
            for m in all_monoms:
                row = np.zeros(n_vars)
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
            _, S, Vh = np.linalg.svd(A_mat, full_matrices=True)
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
            for trial in range(20):
                a0 = np.random.randn(null_dim) * 0.1
                res = scipy_minimize(neg_min_eig, a0, method='Nelder-Mead',
                                     options={'maxiter': 5000, 'xatol': 1e-14})
                if res.fun < best:
                    best = res.fun

            min_eig = -best
            if min_eig >= -1e-8:
                psd_count += 1

        pr(f"  10x10 Gram PSD: {psd_count}/100")
        if psd_count == 100:
            pr(f"  *** K_T2R/r^2 appears to be globally SOS in (p,q)! ***")
        else:
            pr(f"  K_T2R/r^2 is NOT globally SOS in (p,q) ({100-psd_count} failures)")

    pr(f"\n  Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
