#!/usr/bin/env python3
"""Quick check: is K globally non-negative in (p,q), or only on feasible domain?

Also: what does the K8 block look like? Can we prove K8 >= 0 analytically?
And: can domain-constrained SOS (adding multiplier*constraint to make globally PSD) work?

Uses EXACT SAME surplus construction as the proven-fast block-coupling script.
"""

import sys
import numpy as np
import sympy as sp
from sympy import Rational

# Force unbuffered output
def pr(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def disc_poly(e2, e3, e4):
    return (256*e4**3 - 128*e2**2*e4**2 + 144*e2*e3**2*e4
            + 16*e2**4*e4 - 27*e3**4 - 4*e2**3*e3**2)


def phi4_disc(e2, e3, e4):
    return (-8*e2**5 - 64*e2**3*e4 - 36*e2**2*e3**2
            + 384*e2*e4**2 - 432*e3**2*e4)


def build_K():
    """Build normalized surplus K and decompose by (p,q)-degree."""
    pr("Building full surplus and decomposing (takes ~2 min)...")
    s, t, u, v, a, b = sp.symbols('s t u v a b', positive=True)
    r, x, y, p, q = sp.symbols('r x y p q', real=True)

    inv_phi = lambda e2, e3, e4: sp.cancel(disc_poly(e2, e3, e4) / phi4_disc(e2, e3, e4))
    inv_p = inv_phi(-s, u, a)
    inv_q = inv_phi(-t, v, b)
    inv_c = inv_phi(-(s+t), u+v, a+b+s*t/6)

    surplus = sp.together(inv_c - inv_p - inv_q)
    N, D = sp.fraction(surplus)
    N = sp.expand(N)
    pr(f"  N has {len(sp.Poly(N, s,t,u,v,a,b).as_dict())} terms")

    subs = {t: r*s, a: x*s**2/4, b: y*r**2*s**2/4,
            u: p*s**Rational(3,2), v: q*s**Rational(3,2)}
    K = sp.expand(sp.expand(N.subs(subs)) / s**16)
    pr(f"  K has {len(sp.Poly(K, r, x, y, p, q).as_dict())} terms")

    # Decompose by (p,q)-degree
    poly_pq = sp.Poly(K, p, q)
    blocks = {}
    for monom, coeff in poly_pq.as_dict().items():
        i, j = monom
        d = i + j
        blocks[d] = sp.expand(blocks.get(d, sp.Integer(0)) + coeff * p**i * q**j)

    pr(f"  Blocks: {sorted(blocks.keys())}")
    return (r, x, y, p, q), blocks, K


def sample_feasible_rxy(rng, n_samples):
    """Sample (r,x,y) only."""
    out = []
    for _ in range(n_samples):
        rv = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
        xv = float(rng.uniform(1e-3, 1 - 1e-3))
        yv = float(rng.uniform(1e-3, 1 - 1e-3))
        out.append((rv, xv, yv))
    return np.array(out)


def main():
    syms, blocks, K_expr = build_K()
    r, x, y, p, q = syms

    # Lambdify K and blocks
    K_fn = sp.lambdify((r, x, y, p, q), K_expr, 'numpy')
    block_fns = {d: sp.lambdify((r, x, y, p, q), blocks[d], 'numpy')
                 for d in sorted(blocks.keys())}

    pr(f"\n{'='*72}")
    pr("PART 1: GLOBAL NON-NEGATIVITY")
    pr('='*72)

    rng = np.random.default_rng(42)

    for scale_name, scales in [("2x feasible", 2.0), ("5x feasible", 5.0),
                                ("10x feasible", 10.0), ("100x feasible", 100.0)]:
        neg_count = 0
        min_val = np.inf
        worst = None
        n_pts = 30000

        for _ in range(n_pts):
            rv = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
            xv = float(rng.uniform(0.01, 0.99))
            yv = float(rng.uniform(0.01, 0.99))
            pmax = np.sqrt(max(0, 2*(1-xv)/9))
            qmax = np.sqrt(max(0, 2*rv**3*(1-yv)/9))
            pv = float(rng.uniform(-scales * pmax, scales * pmax))
            qv = float(rng.uniform(-scales * qmax, scales * qmax))
            try:
                val = float(K_fn(rv, xv, yv, pv, qv))
                if np.isfinite(val):
                    if val < min_val:
                        min_val = val
                        worst = (rv, xv, yv, pv, qv, pmax, qmax)
                    if val < -1e-10:
                        neg_count += 1
            except:
                pass

        pr(f"  {scale_name}: neg={neg_count}/{n_pts}, min={min_val:.4e}")
        if worst and min_val < 0:
            rv, xv, yv, pv, qv, pmax, qmax = worst
            pr(f"    worst: r={rv:.3f} x={xv:.3f} y={yv:.3f} "
               f"p={pv:.3f}({pv/pmax:.1f}×max) q={qv:.3f}({qv/qmax:.1f}×max)")

    pr(f"\n{'='*72}")
    pr("PART 2: K8 STRUCTURE")
    pr('='*72)

    # Extract K8 coefficients
    K8 = blocks.get(8, sp.Integer(0))
    if K8 != 0:
        poly_K8 = sp.Poly(K8, p, q)
        pr(f"  K8 monomials:")
        for monom in sorted(poly_K8.as_dict().keys()):
            i, j = monom
            coeff = sp.expand(poly_K8.as_dict()[monom])
            n_terms = len(sp.Poly(coeff, r, x, y).as_dict())
            pr(f"    p^{i}*q^{j}: {n_terms} terms, coeff = {coeff}")

    pr(f"\n{'='*72}")
    pr("PART 3: K6 STRUCTURE (first few)")
    pr('='*72)
    K6 = blocks.get(6, sp.Integer(0))
    if K6 != 0:
        poly_K6 = sp.Poly(K6, p, q)
        for monom in sorted(poly_K6.as_dict().keys())[:3]:
            i, j = monom
            coeff = sp.expand(poly_K6.as_dict()[monom])
            n_terms = len(sp.Poly(coeff, r, x, y).as_dict())
            pr(f"    p^{i}*q^{j}: {n_terms} terms")

    pr(f"\n{'='*72}")
    pr("PART 4: DOMAIN-CONSTRAINED SOS")
    pr('='*72)
    pr("Testing: K + c1*p² + c2*q² - c1*Pmax - c2*Qmax >= 0 globally?")

    # Extract A2, C2 (coefficients of p², q² in K2)
    K2 = blocks.get(2, sp.Integer(0))
    poly_K2 = sp.Poly(K2, p, q)
    A2_expr = poly_K2.as_dict().get((2, 0), sp.Integer(0))
    C2_expr = poly_K2.as_dict().get((0, 2), sp.Integer(0))

    A2_fn = sp.lambdify((r, x, y), sp.expand(A2_expr), 'numpy')
    C2_fn = sp.lambdify((r, x, y), sp.expand(C2_expr), 'numpy')
    K0_fn = sp.lambdify((r, x, y), blocks[0], 'numpy')

    # For each (r,x,y), find optimal c1, c2 and check if modified K is globally non-neg
    rxy_samples = sample_feasible_rxy(rng, 2000)
    feasible_count = 0
    infeasible_count = 0

    for idx in range(len(rxy_samples)):
        rv, xv, yv = rxy_samples[idx]
        a2 = float(A2_fn(rv, xv, yv))
        c2 = float(C2_fn(rv, xv, yv))
        k0 = float(K0_fn(rv, xv, yv))
        Pmax = 2 * (1 - xv) / 9
        Qmax = 2 * rv**3 * (1 - yv) / 9

        # Need c1 >= max(0, -a2), c2_ >= max(0, -c2)
        c1_min = max(0.0, -a2)
        c2_min = max(0.0, -c2)

        # Remaining constant: K0 - c1*Pmax - c2_*Qmax >= 0
        # With minimal c1, c2_:
        remaining = k0 - c1_min * Pmax - c2_min * Qmax
        if remaining >= -1e-10:
            feasible_count += 1
        else:
            infeasible_count += 1

    pr(f"  With minimal multipliers: feasible={feasible_count}/{len(rxy_samples)}")
    pr(f"  (feasible means K0 absorbs the shift cost)")

    if infeasible_count > 0:
        pr(f"  NOTE: {infeasible_count} points infeasible with constant multipliers.")
        pr(f"  Need polynomial multipliers or different approach.")

    # PART 5: Actually test full 9x9 Gram numerically at a few points
    pr(f"\n{'='*72}")
    pr("PART 5: 9×9 GRAM MATRIX (quick test at 10 points)")
    pr('='*72)

    # Extract ALL (p,q)-monomial coefficients
    full_poly = sp.Poly(K_expr, p, q)
    all_coeffs = {}
    for monom, coeff in full_poly.as_dict().items():
        all_coeffs[monom] = sp.lambdify((r, x, y), sp.expand(coeff), 'numpy')

    # v-monomials
    v_monoms = [(0,0), (2,0), (1,1), (0,2), (4,0), (3,1), (2,2), (1,3), (0,4)]
    n = len(v_monoms)

    # Map: (p,q)-monomial → list of (i,j) pairs in v⊗v
    pq_to_vv = {}
    for i in range(n):
        for j in range(i, n):
            pi, qi = v_monoms[i]
            pj, qj = v_monoms[j]
            pm, qm = pi + pj, qi + qj
            mult = 1 if i == j else 2
            if (pm, qm) not in pq_to_vv:
                pq_to_vv[(pm, qm)] = []
            pq_to_vv[(pm, qm)].append((i, j, mult))

    # Check coverage
    covered = set(pq_to_vv.keys())
    needed = set(all_coeffs.keys())
    pr(f"  v-monomial products cover: {len(covered)} monomials")
    pr(f"  K needs: {len(needed)} monomials")
    pr(f"  Missing: {needed - covered}")
    pr(f"  Extra (must be zero): {covered - needed}")

    # For a quick test: at a few (r,x,y) points, solve the Gram PSD problem
    from scipy.optimize import minimize as scipy_minimize

    def test_gram_at_point(rv, xv, yv):
        # Build constraint system
        n_vars = n * (n + 1) // 2  # 45

        def var_idx(i, j):
            if i > j: i, j = j, i
            return i * n - i * (i - 1) // 2 + (j - i)

        # Build A matrix and b vector
        all_monoms_ordered = sorted(needed | (covered - needed))
        A_rows = []
        b_vals = []
        for m in all_monoms_ordered:
            row = np.zeros(n_vars)
            for (i, j, mult) in pq_to_vv.get(m, []):
                row[var_idx(i, j)] += mult
            A_rows.append(row)
            if m in all_coeffs:
                b_vals.append(float(all_coeffs[m](rv, xv, yv)))
            else:
                b_vals.append(0.0)

        A = np.array(A_rows)
        b = np.array(b_vals)

        # Particular solution + null space
        g_part, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        _, S, Vh = np.linalg.svd(A, full_matrices=True)
        null_basis = Vh[rank:].T  # shape (n_vars, null_dim)
        null_dim = null_basis.shape[1]

        def neg_min_eig(alpha):
            g = g_part + null_basis @ alpha
            G = np.zeros((n, n))
            for ii in range(n):
                for jj in range(ii, n):
                    G[ii, jj] = g[var_idx(ii, jj)]
                    G[jj, ii] = G[ii, jj]
            return -np.linalg.eigvalsh(G)[0]

        best = np.inf
        for trial in range(10):
            a0 = np.random.randn(null_dim) * 0.1
            res = scipy_minimize(neg_min_eig, a0, method='Nelder-Mead',
                                 options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14})
            if res.fun < best:
                best = res.fun

        return -best, rank, null_dim

    rng2 = np.random.default_rng(999)
    for idx in range(10):
        rv = float(np.exp(rng2.uniform(np.log(0.3), np.log(3.0))))
        xv = float(rng2.uniform(0.1, 0.9))
        yv = float(rng2.uniform(0.1, 0.9))
        min_eig, rank, null_dim = test_gram_at_point(rv, xv, yv)
        status = "PSD" if min_eig >= -1e-8 else "NOT PSD"
        pr(f"  r={rv:.3f} x={xv:.3f} y={yv:.3f}: "
           f"rank={rank}, null_dim={null_dim}, min_eig={min_eig:.4e} [{status}]")


if __name__ == '__main__':
    main()
