#!/usr/bin/env python3
"""Check global non-negativity of K and attempt 9×9 Gram matrix SOS.

Key questions:
1. Is K(r,x,y,p,q) >= 0 for ALL (p,q) in R², or only on feasible domain?
2. If globally non-neg: can we certify via 9×9 Gram matrix?
3. If not: what about K + multipliers * constraints?
"""

import numpy as np
import sympy as sp
from sympy import Rational
from scipy.optimize import minimize


def disc_poly(e2, e3, e4):
    return (256*e4**3 - 128*e2**2*e4**2 + 144*e2*e3**2*e4
            + 16*e2**4*e4 - 27*e3**4 - 4*e2**3*e3**2)


def phi4_disc(e2, e3, e4):
    return (-8*e2**5 - 64*e2**3*e4 - 36*e2**2*e3**2
            + 384*e2*e4**2 - 432*e3**2*e4)


def build_all():
    """Build K and extract coefficients."""
    print("Building full surplus (takes ~2 min)...")
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

    # Extract all (p,q)-monomial coefficients
    poly = sp.Poly(K, p, q)
    coeffs = {}
    for monom, coeff in poly.as_dict().items():
        i, j = monom
        coeffs[(i, j)] = sp.expand(coeff)

    print(f"  K has {len(poly.as_dict())} terms total")
    print(f"  {len(coeffs)} distinct (p,q)-monomials")

    return (r, x, y, p, q), coeffs, K


def check_global_nonnegativity(syms, K):
    """Check if K >= 0 for ALL (p,q), not just feasible."""
    r, x, y, p, q = syms
    K_fn = sp.lambdify((r, x, y, p, q), K, 'numpy')

    print(f"\n{'='*72}")
    print("GLOBAL NON-NEGATIVITY CHECK")
    print('='*72)

    rng = np.random.default_rng(42)
    min_val = np.inf
    worst_pt = None
    neg_count = 0
    n_total = 0

    # Test 1: Large p,q (outside feasible)
    print("\n[1] Large p,q (outside feasible domain):")
    for _ in range(50000):
        rv = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
        xv = float(rng.uniform(0.01, 0.99))
        yv = float(rng.uniform(0.01, 0.99))
        # p,q much larger than feasible bound
        pmax = np.sqrt(2*(1-xv)/9)
        qmax = np.sqrt(2*rv**3*(1-yv)/9)
        pv = float(rng.uniform(-5*pmax, 5*pmax))
        qv = float(rng.uniform(-5*qmax, 5*qmax))
        val = float(K_fn(rv, xv, yv, pv, qv))
        n_total += 1
        if val < min_val:
            min_val = val
            worst_pt = (rv, xv, yv, pv, qv)
        if val < -1e-10:
            neg_count += 1

    print(f"  Samples: {n_total}, negative: {neg_count}")
    print(f"  Min value: {min_val:.6e}")
    if worst_pt:
        rv, xv, yv, pv, qv = worst_pt
        pmax = np.sqrt(2*(1-xv)/9)
        qmax = np.sqrt(2*rv**3*(1-yv)/9)
        print(f"  Worst: r={rv:.3f} x={xv:.3f} y={yv:.3f} "
              f"p={pv:.4f} (max={pmax:.4f}) q={qv:.4f} (max={qmax:.4f})")
        print(f"  p/pmax={pv/pmax:.2f}, q/qmax={qv/qmax:.2f}")

    # Test 2: Very large p,q (extreme regime)
    print("\n[2] Very large p,q (extreme regime):")
    neg_count_2 = 0
    min_val_2 = np.inf
    for _ in range(50000):
        rv = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
        xv = float(rng.uniform(0.01, 0.99))
        yv = float(rng.uniform(0.01, 0.99))
        pv = float(rng.uniform(-10, 10))
        qv = float(rng.uniform(-10, 10))
        val = float(K_fn(rv, xv, yv, pv, qv))
        if val < min_val_2:
            min_val_2 = val
        if val < -1e-10:
            neg_count_2 += 1

    print(f"  Samples: 50000, negative: {neg_count_2}")
    print(f"  Min value: {min_val_2:.6e}")

    # Test 3: Systematic edge exploration
    print("\n[3] Systematic p=±pmax*k, q=±qmax*k for k in [0.5, 5]:")
    for k in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        neg_k = 0
        min_k = np.inf
        for _ in range(10000):
            rv = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
            xv = float(rng.uniform(0.01, 0.99))
            yv = float(rng.uniform(0.01, 0.99))
            pmax = np.sqrt(2*(1-xv)/9)
            qmax = np.sqrt(2*rv**3*(1-yv)/9)
            pv = k * pmax * float(rng.choice([-1, 1]))
            qv = k * qmax * float(rng.choice([-1, 1]))
            val = float(K_fn(rv, xv, yv, pv, qv))
            if val < min_k:
                min_k = val
            if val < -1e-10:
                neg_k += 1
        print(f"  k={k:.1f}: neg={neg_k}/10000, min={min_k:.6e}")

    is_globally_nonneg = (neg_count == 0 and neg_count_2 == 0)
    return is_globally_nonneg


def build_gram_system(syms, coeffs):
    """Build the linear system for the 9×9 Gram matrix.

    v = [1, p², pq, q², p⁴, p³q, p²q², pq³, q⁴]
    v has 9 entries, G is 9×9 symmetric → 45 upper-triangle entries.
    K has 21 (p,q)-monomial constraints.
    Free parameters: 45 - 21 = 24.
    """
    r, x, y, p, q = syms

    # v-monomials: (p-exp, q-exp)
    v_monoms = [(0,0), (2,0), (1,1), (0,2), (4,0), (3,1), (2,2), (1,3), (0,4)]
    n = len(v_monoms)  # 9

    # For each pair (i,j) with i <= j, the product monomial is v[i]*v[j]
    # Map from (p,q)-monomial → list of (i,j,multiplicity) contributing to it
    pq_to_ij = {}
    for i in range(n):
        for j in range(i, n):
            pi, qi = v_monoms[i]
            pj, qj = v_monoms[j]
            pm, qm = pi + pj, qi + qj
            mult = 1 if i == j else 2  # symmetric: G[i,j] + G[j,i] = 2*G[i,j]
            if (pm, qm) not in pq_to_ij:
                pq_to_ij[(pm, qm)] = []
            pq_to_ij[(pm, qm)].append((i, j, mult))

    print(f"\n{'='*72}")
    print("9×9 GRAM MATRIX STRUCTURE")
    print('='*72)
    print(f"  v-monomials: {v_monoms}")
    print(f"  G is {n}×{n} symmetric → {n*(n+1)//2} upper-triangle entries")
    print(f"  Product monomials: {len(pq_to_ij)}")

    # Check which product monomials match K's monomials
    k_monoms = set(coeffs.keys())
    prod_monoms = set(pq_to_ij.keys())
    missing = k_monoms - prod_monoms
    extra = prod_monoms - k_monoms

    print(f"  K monomials: {len(k_monoms)}")
    print(f"  Missing from Gram: {missing}")
    print(f"  Extra in Gram (must be zero): {extra}")

    if missing:
        print("  ERROR: Some K monomials cannot be represented!")
        return None, None, None, None

    # Build constraint matrix: for each K-monomial, sum of G[i,j]*mult = coeff
    # For extra monomials: sum of G[i,j]*mult = 0
    n_vars = n * (n + 1) // 2  # 45
    all_monoms = sorted(k_monoms | extra)
    n_constraints = len(all_monoms)

    print(f"  Constraints: {n_constraints} ({len(k_monoms)} from K + {len(extra)} zero)")
    print(f"  Variables: {n_vars}")
    print(f"  Free parameters: {n_vars - n_constraints}")

    # Variable indexing: G[i,j] with i <= j → index k
    def var_idx(i, j):
        if i > j:
            i, j = j, i
        return i * n - i * (i - 1) // 2 + (j - i)

    # Lambdify each coefficient
    coeff_fns = {}
    for m in k_monoms:
        coeff_fns[m] = sp.lambdify((r, x, y), coeffs[m], 'numpy')

    return v_monoms, pq_to_ij, var_idx, coeff_fns


def solve_gram_at_point(v_monoms, pq_to_ij, var_idx, coeff_fns, rv, xv, yv):
    """At a specific (r,x,y), find if 9×9 Gram can be PSD."""
    n = len(v_monoms)
    n_vars = n * (n + 1) // 2

    # Build constraint matrix A and rhs b: A @ g = b
    k_monoms = set(coeff_fns.keys())
    all_monoms = sorted(k_monoms | (set(pq_to_ij.keys()) - k_monoms))

    A_rows = []
    b_vals = []
    for m in all_monoms:
        row = np.zeros(n_vars)
        for (i, j, mult) in pq_to_ij.get(m, []):
            row[var_idx(i, j)] += mult
        A_rows.append(row)
        if m in coeff_fns:
            b_vals.append(float(coeff_fns[m](rv, xv, yv)))
        else:
            b_vals.append(0.0)

    A = np.array(A_rows)
    b = np.array(b_vals)

    # Find particular solution and null space
    # Use least squares for particular solution
    g_part, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)

    # Null space
    U, S, Vh = np.linalg.svd(A)
    null_mask = S < 1e-10 * S[0] if len(S) > 0 else np.zeros(len(S), dtype=bool)
    # Also include dimensions beyond rank
    null_dim = n_vars - rank
    null_basis = Vh[rank:].T  # columns are null space basis vectors

    if null_dim == 0:
        # No freedom — check if g_part gives PSD
        G = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                G[i, j] = g_part[var_idx(i, j)]
                G[j, i] = G[i, j]
        eigs = np.linalg.eigvalsh(G)
        return eigs[0], G, g_part

    # Optimize: maximize min eigenvalue over null space parameters
    def neg_min_eig(alpha):
        g = g_part + null_basis @ alpha
        G = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                G[i, j] = g[var_idx(i, j)]
                G[j, i] = G[i, j]
        eigs = np.linalg.eigvalsh(G)
        return -eigs[0]

    # Try multiple starting points
    best_min_eig = -np.inf
    best_G = None

    for trial in range(5):
        alpha0 = np.random.randn(null_dim) * 0.01
        res = minimize(neg_min_eig, alpha0, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
        min_eig = -res.fun
        if min_eig > best_min_eig:
            best_min_eig = min_eig
            g = g_part + null_basis @ res.x
            G = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    G[i, j] = g[var_idx(i, j)]
                    G[j, i] = G[i, j]
            best_G = G

    return best_min_eig, best_G, None


def main():
    syms, coeffs, K = build_all()
    r, x, y, p, q = syms

    # Step 1: Check global non-negativity
    is_global = check_global_nonnegativity(syms, K)

    if is_global:
        print("\n*** K appears GLOBALLY non-negative! ***")
        print("*** Attempting 9×9 Gram matrix SOS... ***")
    else:
        print("\n*** K is NOT globally non-negative. ***")
        print("*** Need domain-constrained approach. ***")

    # Step 2: Build Gram system
    result = build_gram_system(syms, coeffs)
    if result[0] is None:
        return
    v_monoms, pq_to_ij, var_idx, coeff_fns = result

    # Step 3: Test Gram PSD at sampled points
    print(f"\n{'='*72}")
    print("9×9 GRAM PSD CHECK AT SAMPLED POINTS")
    print('='*72)

    rng = np.random.default_rng(123)
    n_test = 200  # fewer points since optimization is expensive

    psd_count = 0
    not_psd_count = 0
    min_eig_all = []
    worst_point = None
    worst_eig = 0.0

    for idx in range(n_test):
        rv = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
        xv = float(rng.uniform(0.01, 0.99))
        yv = float(rng.uniform(0.01, 0.99))

        best_eig, best_G, _ = solve_gram_at_point(
            v_monoms, pq_to_ij, var_idx, coeff_fns, rv, xv, yv)

        min_eig_all.append(best_eig)
        if best_eig >= -1e-8:
            psd_count += 1
        else:
            not_psd_count += 1
            if best_eig < worst_eig:
                worst_eig = best_eig
                worst_point = (rv, xv, yv, best_G)

        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{n_test}: PSD={psd_count}, not-PSD={not_psd_count}")

    min_eig_all = np.array(min_eig_all)
    print(f"\nRESULTS:")
    print(f"  PSD: {psd_count}/{n_test}")
    print(f"  Not PSD: {not_psd_count}/{n_test}")
    print(f"  Min eigenvalue: min={np.min(min_eig_all):.6e}, "
          f"median={np.median(min_eig_all):.6e}")

    if worst_point:
        rv, xv, yv, G = worst_point
        print(f"\n  Worst: r={rv:.4f}, x={xv:.4f}, y={yv:.4f}")
        print(f"  Eigenvalues: {np.linalg.eigvalsh(G)}")

    if psd_count == n_test:
        print(f"\n*** 9×9 Gram matrix is PSD at all {n_test} points! ***")
        print("*** K is SOS in (p,q) — proof complete modulo algebraic certificate. ***")
    elif is_global:
        print(f"\n*** K is globally non-neg but Gram fails — optimizer may need tuning ***")


if __name__ == '__main__':
    main()
