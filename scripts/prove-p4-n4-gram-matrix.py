#!/usr/bin/env python3
"""Approach B: Parametric SOS for K0+K2+K4 via 4×4 Gram matrix.

K0+K2+K4 has degree 4 in (p,q) with only even total degrees.
Using monomial vector v = [1, p², pq, q²]^T, we can write:

   K0+K2+K4 = v^T G(r,x,y) v

where G is a 4×4 symmetric matrix with entries determined by the
polynomial coefficients, EXCEPT for one free parameter g33.

The constraint: coeff(p²q²) = 2*G24 + G33, so G24 = (c_p2q2 - G33)/2.

If we can choose g33 = g33(r,x,y) such that G is PSD on the feasible
domain, then K0+K2+K4 >= 0 is proved as a sum of squares in (p,q).

This script:
1. Extracts (p,q)-coefficients from K0+K2+K4 symbolically
2. Constructs the 4×4 Gram matrix
3. For sampled feasible (r,x,y), finds the optimal g33 and checks PSD
4. Analyzes whether a universal g33 choice exists
"""

import numpy as np
import sympy as sp
from sympy import Rational


def disc_poly(e2, e3, e4):
    return (256*e4**3 - 128*e2**2*e4**2 + 144*e2*e3**2*e4
            + 16*e2**4*e4 - 27*e3**4 - 4*e2**3*e3**2)


def phi4_disc(e2, e3, e4):
    return (-8*e2**5 - 64*e2**3*e4 - 36*e2**2*e3**2
            + 384*e2*e4**2 - 432*e3**2*e4)


def build_K024_coefficients():
    """Build K0+K2+K4 and extract (p,q)-coefficients."""
    print("Building full surplus and extracting K0+K2+K4 coefficients...")

    s, t, u, v, a, b = sp.symbols('s t u v a b', positive=True)
    r, x, y, p, q = sp.symbols('r x y p q', real=True)

    inv_phi = lambda e2, e3, e4: sp.cancel(disc_poly(e2, e3, e4) / phi4_disc(e2, e3, e4))

    inv_p = inv_phi(-s, u, a)
    inv_q = inv_phi(-t, v, b)
    inv_c = inv_phi(-(s+t), u+v, a+b+s*t/6)

    surplus = sp.together(inv_c - inv_p - inv_q)
    N, D = sp.fraction(surplus)
    N = sp.expand(N)

    # Normalize
    subs = {t: r*s, a: x*s**2/4, b: y*r**2*s**2/4,
            u: p*s**Rational(3,2), v: q*s**Rational(3,2)}
    Nn = sp.expand(N.subs(subs))
    K = sp.expand(Nn / s**16)

    # Decompose by total (p,q)-degree
    poly_pq = sp.Poly(K, p, q)
    blocks = {}
    for monom, coeff in poly_pq.as_dict().items():
        i, j = monom
        d = i + j
        blocks[d] = sp.expand(blocks.get(d, sp.Integer(0)) + coeff * p**i * q**j)

    # Build K0+K2+K4
    K024 = sp.expand(blocks[0] + blocks[2] + blocks[4])
    print(f"  K0+K2+K4: {len(sp.Poly(K024, r, x, y, p, q).as_dict())} terms")

    # Extract coefficients of each (p,q)-monomial
    poly = sp.Poly(K024, p, q)
    coeffs = {}
    for monom, coeff in poly.as_dict().items():
        i, j = monom
        coeffs[(i, j)] = sp.expand(coeff)

    # Print what we have
    print(f"\n  (p,q)-monomial coefficients:")
    for (i, j) in sorted(coeffs.keys()):
        n_terms = len(sp.Poly(coeffs[(i,j)], r, x, y).as_dict())
        print(f"    p^{i}*q^{j}: {n_terms} terms in (r,x,y)")

    return (r, x, y, p, q), coeffs, blocks


def analyze_gram_matrix(syms, coeffs):
    """Construct and analyze the 4x4 Gram matrix."""
    r, x, y, p, q = syms
    g33 = sp.Symbol('g33', real=True)

    # Gram matrix entries from K0+K2+K4 = v^T G v
    # v = [1, p², pq, q²]^T
    #
    # v*v^T monomials:
    # (0,0): 1       → G[0,0]
    # (0,1): p²      → G[0,1] + G[1,0] = 2*G[0,1]
    # (0,2): pq      → 2*G[0,2]
    # (0,3): q²      → 2*G[0,3]
    # (1,1): p⁴      → G[1,1]
    # (1,2): p³q     → 2*G[1,2]
    # (1,3): p²q²    → 2*G[1,3] + G[2,2]  ← THE FREE PARAMETER
    # (2,3): pq³     → 2*G[2,3]
    # (3,3): q⁴      → G[3,3]

    def get_coeff(i, j):
        return coeffs.get((i, j), sp.Integer(0))

    # Read off coefficients
    c_1    = get_coeff(0, 0)  # constant term = K0
    c_p2   = get_coeff(2, 0)  # coefficient of p²
    c_pq   = get_coeff(1, 1)  # coefficient of pq
    c_q2   = get_coeff(0, 2)  # coefficient of q²
    c_p4   = get_coeff(4, 0)  # coefficient of p⁴
    c_p3q  = get_coeff(3, 1)  # coefficient of p³q
    c_p2q2 = get_coeff(2, 2)  # coefficient of p²q²
    c_pq3  = get_coeff(1, 3)  # coefficient of pq³
    c_q4   = get_coeff(0, 4)  # coefficient of q⁴

    print("\nGram matrix G (parametric in g33):")
    print("  G[0,0] = K0 (constant coefficient)")
    print("  G[0,1] = c_p2 / 2")
    print("  G[0,2] = c_pq / 2")
    print("  G[0,3] = c_q2 / 2")
    print("  G[1,1] = c_p4")
    print("  G[1,2] = c_p3q / 2")
    print("  G[1,3] = (c_p2q2 - g33) / 2")
    print("  G[2,2] = g33  ← FREE PARAMETER")
    print("  G[2,3] = c_pq3 / 2")
    print("  G[3,3] = c_q4")

    # Construct G symbolically
    G = sp.Matrix([
        [c_1,        c_p2/2,            c_pq/2,              c_q2/2],
        [c_p2/2,     c_p4,              c_p3q/2,             (c_p2q2 - g33)/2],
        [c_pq/2,     c_p3q/2,           g33,                 c_pq3/2],
        [c_q2/2,     (c_p2q2 - g33)/2,  c_pq3/2,            c_q4]
    ])

    # Lambdify each entry for numerical evaluation
    G_entries = {}
    for i in range(4):
        for j in range(i, 4):
            G_entries[(i, j)] = sp.lambdify((r, x, y, g33), G[i, j], 'numpy')

    return G, G_entries


def sample_feasible(rng, n_samples):
    """Sample from the normalized feasible cone (r,x,y only)."""
    out = []
    tries = 0
    while len(out) < n_samples and tries < 10 * n_samples:
        tries += 1
        rv = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
        xv = float(rng.uniform(1e-3, 1 - 1e-3))
        yv = float(rng.uniform(1e-3, 1 - 1e-3))
        out.append((rv, xv, yv))
    return np.array(out)


def find_optimal_g33(G_entries, rv, xv, yv, n_grid=200):
    """For a single (r,x,y), find the g33 that maximizes min-eigenvalue of G."""
    def build_G_numeric(g33_val):
        G = np.zeros((4, 4))
        for i in range(4):
            for j in range(i, 4):
                G[i, j] = float(G_entries[(i, j)](rv, xv, yv, g33_val))
                if j > i:
                    G[j, i] = G[i, j]
        return G

    # First get a sense of scale
    G0 = build_G_numeric(0.0)
    diag_scale = max(abs(G0[i, i]) for i in range(4)) + 1e-30

    # Search over a range of g33 values
    # g33 should be comparable to the p²q² coefficient magnitude
    c_p2q2_val = 2 * G0[1, 3] + G0[2, 2]  # since G24 = (c_p2q2 - g33)/2, at g33=0: G24 = c_p2q2/2

    g33_range = np.linspace(-5 * diag_scale, 5 * diag_scale, n_grid)

    best_g33 = 0.0
    best_min_eig = -np.inf

    for g33_val in g33_range:
        G = build_G_numeric(g33_val)
        eigs = np.linalg.eigvalsh(G)
        min_eig = eigs[0]
        if min_eig > best_min_eig:
            best_min_eig = min_eig
            best_g33 = g33_val

    # Refine with golden section search
    lo = best_g33 - 10 * diag_scale / n_grid
    hi = best_g33 + 10 * diag_scale / n_grid
    for _ in range(50):
        m1 = lo + (hi - lo) * 0.382
        m2 = lo + (hi - lo) * 0.618
        e1 = np.linalg.eigvalsh(build_G_numeric(m1))[0]
        e2 = np.linalg.eigvalsh(build_G_numeric(m2))[0]
        if e1 < e2:
            lo = m1
        else:
            hi = m2
    best_g33 = (lo + hi) / 2
    best_min_eig = np.linalg.eigvalsh(build_G_numeric(best_g33))[0]

    return best_g33, best_min_eig, build_G_numeric(best_g33)


def main():
    syms, coeffs, blocks = build_K024_coefficients()
    r, x, y, p, q = syms

    G_sym, G_entries = analyze_gram_matrix(syms, coeffs)

    # Sample feasible (r,x,y) points
    print(f"\n{'='*72}")
    print("Numerical PSD check of 4×4 Gram matrix")
    print('='*72)

    rng = np.random.default_rng(42)
    samples = sample_feasible(rng, 5000)
    print(f"Feasible (r,x,y) samples: {len(samples)}")

    psd_count = 0
    not_psd_count = 0
    min_eig_overall = np.inf
    worst_point = None
    g33_values = []
    min_eigs = []

    for idx in range(len(samples)):
        rv, xv, yv = samples[idx]
        best_g33, best_min_eig, G_num = find_optimal_g33(G_entries, rv, xv, yv)

        g33_values.append(best_g33)
        min_eigs.append(best_min_eig)

        if best_min_eig >= -1e-10:
            psd_count += 1
        else:
            not_psd_count += 1
            if best_min_eig < min_eig_overall:
                min_eig_overall = best_min_eig
                worst_point = (rv, xv, yv, best_g33, best_min_eig, G_num)

        if (idx + 1) % 1000 == 0:
            print(f"  {idx+1}/{len(samples)}: PSD={psd_count}, not-PSD={not_psd_count}")

    g33_values = np.array(g33_values)
    min_eigs = np.array(min_eigs)

    print(f"\n{'='*72}")
    print("RESULTS")
    print('='*72)
    print(f"  PSD: {psd_count}/{len(samples)}")
    print(f"  Not PSD: {not_psd_count}/{len(samples)}")
    print(f"  Min eigenvalue (best g33): min={np.min(min_eigs):.6e}, "
          f"median={np.median(min_eigs):.6e}")
    print(f"  Optimal g33: min={np.min(g33_values):.6e}, "
          f"max={np.max(g33_values):.6e}, "
          f"median={np.median(g33_values):.6e}")

    if worst_point is not None:
        rv, xv, yv, g33v, eig_v, G_num = worst_point
        print(f"\n  Worst point: r={rv:.4f}, x={xv:.4f}, y={yv:.4f}")
        print(f"  Best g33={g33v:.6e}, min eigenvalue={eig_v:.6e}")
        print(f"  G matrix:\n{G_num}")
        print(f"  All eigenvalues: {np.linalg.eigvalsh(G_num)}")

    if not_psd_count == 0:
        print(f"\n*** K0+K2+K4 is SOS in (p,q) for all {len(samples)} sampled (r,x,y)! ***")
        print("*** The 4×4 Gram matrix approach WORKS. ***")

        # Analyze g33 structure
        print(f"\n{'='*72}")
        print("g33 structure analysis")
        print('='*72)
        rv_all = samples[:, 0]
        xv_all = samples[:, 1]
        yv_all = samples[:, 2]

        # Look for g33 as a function of r, x, y
        print("  Correlation of optimal g33 with r, x, y:")
        print(f"    corr(g33, r) = {np.corrcoef(g33_values, rv_all)[0,1]:.4f}")
        print(f"    corr(g33, x) = {np.corrcoef(g33_values, xv_all)[0,1]:.4f}")
        print(f"    corr(g33, y) = {np.corrcoef(g33_values, yv_all)[0,1]:.4f}")

        # Check if g33 = 0 works universally
        print(f"\n  Testing fixed g33 choices:")
        for g33_test in [0.0]:
            min_eig_fixed = np.inf
            count_psd = 0
            for idx in range(len(samples)):
                rv, xv, yv = samples[idx]
                G = np.zeros((4, 4))
                for i in range(4):
                    for j in range(i, 4):
                        G[i, j] = float(G_entries[(i, j)](rv, xv, yv, g33_test))
                        if j > i:
                            G[j, i] = G[i, j]
                eigs = np.linalg.eigvalsh(G)
                if eigs[0] >= -1e-10:
                    count_psd += 1
                min_eig_fixed = min(min_eig_fixed, eigs[0])
            print(f"    g33={g33_test}: PSD={count_psd}/{len(samples)}, "
                  f"min_eig={min_eig_fixed:.6e}")

    # Also check: is the full K representable as SOS via 9×9 Gram matrix?
    # First, extract all coefficients
    print(f"\n{'='*72}")
    print("Full K: extract all (p,q)-coefficients for 9×9 Gram analysis")
    print('='*72)

    K_full = sp.Integer(0)
    for d in sorted(blocks.keys()):
        K_full = sp.expand(K_full + blocks[d])

    poly_full = sp.Poly(K_full, p, q)
    full_coeffs = {}
    for monom, coeff in poly_full.as_dict().items():
        i, j = monom
        full_coeffs[(i, j)] = sp.expand(coeff)

    for (i, j) in sorted(full_coeffs.keys()):
        n_terms = len(sp.Poly(full_coeffs[(i,j)], r, x, y).as_dict())
        print(f"  p^{i}*q^{j}: {n_terms} terms in (r,x,y)")


if __name__ == '__main__':
    main()
