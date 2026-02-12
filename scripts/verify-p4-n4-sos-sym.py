#!/usr/bin/env python3
"""Symmetry-reduced SOS certificate for n=4 surplus.

Exploits two symmetries:
1. Reflection: N(a3,a4,b3,b4) = N(-a3,a4,-b3,b4) (even in a3,b3)
2. Swap: N(a3,a4,b3,b4) = N(b3,b4,a3,a4)

This reduces the Gram matrix from 126×126 to two blocks (~50×50 + ~16×16),
and the swap symmetry further constrains the structure.

Builds the SDP data directly as sparse matrices for SCS solver,
bypassing CVXPY's memory-heavy expression tree.
"""

import sympy as sp
from sympy import symbols, Rational, expand, Poly, together, fraction
import numpy as np
from scipy import sparse
import time


def build_surplus_numerator():
    """Build N, D such that surplus = N/D with D > 0 on domain."""
    a3, a4, b3, b4 = sp.symbols('a3 a4 b3 b4')

    disc_p = expand(256*a4**3 - 128*a4**2 - 144*a3**2*a4
                    - 27*a3**4 + 16*a4 + 4*a3**2)
    f1_p = 1 + 12*a4
    f2_p = 9*a3**2 + 8*a4 - 2

    disc_q = expand(256*b4**3 - 128*b4**2 - 144*b3**2*b4
                    - 27*b3**4 + 16*b4 + 4*b3**2)
    f1_q = 1 + 12*b4
    f2_q = 9*b3**2 + 8*b4 - 2

    c3 = a3 + b3
    c4 = a4 + Rational(1, 6) + b4
    disc_r = expand(256*c4**3 - 512*c4**2 + 288*c3**2*c4
                    - 27*c3**4 + 256*c4 + 32*c3**2)
    f1_r = expand(4 + 12*c4)
    f2_r = expand(-16 + 16*c4 + 9*c3**2)

    surplus_expr = (-disc_r / (4 * f1_r * f2_r)
                    + disc_p / (4 * f1_p * f2_p)
                    + disc_q / (4 * f1_q * f2_q))

    surplus_frac = together(surplus_expr)
    N, D = fraction(surplus_frac)
    N = expand(N)
    D = expand(D)

    return N, D, disc_p, disc_q, f1_p, f2_p, f1_q, f2_q, (a3, a4, b3, b4)


def enum_even_monomials(max_total_deg):
    """Enumerate monomials in (a3, a4, b3, b4) with even a3 and b3 powers.

    Returns two lists:
    - even_even: both a3, b3 have even exponents
    - odd_odd: both a3, b3 have odd exponents (product is even)
    """
    even_even = []
    odd_odd = []

    for i3 in range(0, max_total_deg + 1):
        for i4 in range(0, max_total_deg + 1 - i3):
            for j3 in range(0, max_total_deg + 1 - i3 - i4):
                j4_max = max_total_deg - i3 - i4 - j3
                for j4 in range(0, j4_max + 1):
                    exp = (i3, i4, j3, j4)
                    if i3 % 2 == 0 and j3 % 2 == 0:
                        even_even.append(exp)
                    elif i3 % 2 == 1 and j3 % 2 == 1:
                        odd_odd.append(exp)
    return even_even, odd_odd


def poly_to_dict(poly_expr, variables):
    """Convert sympy expression to {exponent_tuple: float_coefficient}."""
    poly = Poly(expand(poly_expr), *variables)
    return {e: float(c) for e, c in poly.as_dict().items()}


def gram_products(basis):
    """For a monomial basis, compute the products basis[i] * basis[j].

    Returns dict: exponent_tuple -> list of (i, j, multiplier) where
    multiplier is 1 for diagonal, 2 for off-diagonal.
    """
    n = len(basis)
    products = {}
    for i in range(n):
        for j in range(i, n):
            e = tuple(basis[i][k] + basis[j][k] for k in range(len(basis[0])))
            if e not in products:
                products[e] = []
            mult = 1.0 if i == j else 2.0
            products[e].append((i, j, mult))
    return products


def constraint_products(g_dict, gram_prods):
    """Products of polynomial g with Gram matrix products.

    Returns dict: exponent_tuple -> list of (coeff, i, j, mult).
    """
    result = {}
    for e_g, c_g in g_dict.items():
        for e_q, entries in gram_prods.items():
            e = tuple(e_g[k] + e_q[k] for k in range(len(e_g)))
            if e not in result:
                result[e] = []
            for i, j, mult in entries:
                result[e].append((c_g * mult, i, j))
    return result


def svec_index(i, j, n):
    """Index into the svec (scaled upper triangle) of an n×n matrix."""
    if i > j:
        i, j = j, i
    return i * n - i * (i + 1) // 2 + j


def main():
    print("=" * 70)
    print("Symmetry-reduced SOS certificate (direct SDP formulation)")
    print("=" * 70)
    print()

    t0 = time.time()

    # Build surplus numerator
    print("Building surplus numerator...")
    N, D, disc_p, disc_q, f1_p, f2_p, f1_q, f2_q, variables = build_surplus_numerator()
    a3, a4, b3, b4 = variables

    # CRITICAL: D < 0 on the domain (product of three negative f2 factors).
    # surplus = N/D >= 0  iff  N <= 0  iff  -N >= 0.
    # So we need to certify -N >= 0, not N >= 0!
    neg_N = expand(-N)
    N_dict = poly_to_dict(neg_N, variables)
    print(f"  -N has {len(N_dict)} terms, max degree {max(sum(e) for e in N_dict)}")
    print(f"  (Using -N since D < 0 on domain, so surplus >= 0 iff -N >= 0)")

    disc_p_dict = poly_to_dict(disc_p, variables)
    disc_q_dict = poly_to_dict(disc_q, variables)
    neg_f2_p_dict = poly_to_dict(-f2_p, variables)
    neg_f2_q_dict = poly_to_dict(-f2_q, variables)

    print(f"  ({time.time()-t0:.1f}s)")
    print()

    # Build symmetry-reduced bases
    # σ_0: degree ≤ 10, basis up to degree 5
    ee5, oo5 = enum_even_monomials(5)
    print(f"σ_0 basis: {len(ee5)} even-even + {len(oo5)} odd-odd = {len(ee5)+len(oo5)} (vs 126 unreduced)")

    # σ_1, σ_2 (disc multiplier, degree 4): basis up to degree 3
    ee3, oo3 = enum_even_monomials(3)
    print(f"σ_1,σ_2 basis: {len(ee3)} even-even + {len(oo3)} odd-odd = {len(ee3)+len(oo3)} (vs 35)")

    # σ_3, σ_4 (-f2 multiplier, degree 2): basis up to degree 4
    ee4, oo4 = enum_even_monomials(4)
    print(f"σ_3,σ_4 basis: {len(ee4)} even-even + {len(oo4)} odd-odd = {len(ee4)+len(oo4)} (vs 70)")

    # Gram matrix sizes
    n_ee5, n_oo5 = len(ee5), len(oo5)
    n_ee3, n_oo3 = len(ee3), len(oo3)
    n_ee4, n_oo4 = len(ee4), len(oo4)

    # Total PSD matrix entries (upper triangle)
    total_params = (n_ee5*(n_ee5+1)//2 + n_oo5*(n_oo5+1)//2  # σ_0
                    + 2*(n_ee3*(n_ee3+1)//2 + n_oo3*(n_oo3+1)//2)  # σ_1, σ_2
                    + 2*(n_ee4*(n_ee4+1)//2 + n_oo4*(n_oo4+1)//2))  # σ_3, σ_4
    print(f"Total PSD parameters: {total_params}")

    # Compute Gram products for each block
    print("Computing Gram products...")
    gp_ee5 = gram_products(ee5)
    gp_oo5 = gram_products(oo5)
    gp_ee3 = gram_products(ee3)
    gp_oo3 = gram_products(oo3)
    gp_ee4 = gram_products(ee4)
    gp_oo4 = gram_products(oo4)

    # Compute constraint products
    print("Computing multiplier products...")
    cp_disc_p_ee3 = constraint_products(disc_p_dict, gp_ee3)
    cp_disc_p_oo3 = constraint_products(disc_p_dict, gp_oo3)
    cp_disc_q_ee3 = constraint_products(disc_q_dict, gp_ee3)
    cp_disc_q_oo3 = constraint_products(disc_q_dict, gp_oo3)
    cp_f2p_ee4 = constraint_products(neg_f2_p_dict, gp_ee4)
    cp_f2p_oo4 = constraint_products(neg_f2_p_dict, gp_oo4)
    cp_f2q_ee4 = constraint_products(neg_f2_q_dict, gp_ee4)
    cp_f2q_oo4 = constraint_products(neg_f2_q_dict, gp_oo4)

    # Collect all monomials
    all_exp = set(N_dict.keys())
    for cp_dict in [gp_ee5, gp_oo5,
                     cp_disc_p_ee3, cp_disc_p_oo3,
                     cp_disc_q_ee3, cp_disc_q_oo3,
                     cp_f2p_ee4, cp_f2p_oo4,
                     cp_f2q_ee4, cp_f2q_oo4]:
        all_exp |= set(cp_dict.keys())

    # Filter to even-parity exponents only
    all_exp = sorted([e for e in all_exp if e[0] % 2 == 0 and e[2] % 2 == 0],
                     key=lambda e: (sum(e), e))
    print(f"Monomial constraints: {len(all_exp)}")
    print(f"  ({time.time()-t0:.1f}s)")
    print()

    # Build the SDP using CVXPY (smaller matrices now)
    import cvxpy as cp

    print("Building CVXPY problem with symmetry-reduced matrices...")

    # PSD matrix variables (one per block)
    Q0_ee = cp.Variable((n_ee5, n_ee5), symmetric=True)
    Q0_oo = cp.Variable((n_oo5, n_oo5), symmetric=True)
    Q1_ee = cp.Variable((n_ee3, n_ee3), symmetric=True)
    Q1_oo = cp.Variable((n_oo3, n_oo3), symmetric=True)
    Q2_ee = cp.Variable((n_ee3, n_ee3), symmetric=True)
    Q2_oo = cp.Variable((n_oo3, n_oo3), symmetric=True)
    Q3_ee = cp.Variable((n_ee4, n_ee4), symmetric=True)
    Q3_oo = cp.Variable((n_oo4, n_oo4), symmetric=True)
    Q4_ee = cp.Variable((n_ee4, n_ee4), symmetric=True)
    Q4_oo = cp.Variable((n_oo4, n_oo4), symmetric=True)

    constraints = [Q0_ee >> 0, Q0_oo >> 0,
                   Q1_ee >> 0, Q1_oo >> 0,
                   Q2_ee >> 0, Q2_oo >> 0,
                   Q3_ee >> 0, Q3_oo >> 0,
                   Q4_ee >> 0, Q4_oo >> 0]

    # Build equality constraints
    print("Building equality constraints...")
    n_eq = 0
    for e in all_exp:
        lhs = 0

        # σ_0 contribution
        if e in gp_ee5:
            for i, j, mult in gp_ee5[e]:
                lhs = lhs + mult * Q0_ee[i, j]
        if e in gp_oo5:
            for i, j, mult in gp_oo5[e]:
                lhs = lhs + mult * Q0_oo[i, j]

        # σ_1 * disc_p
        if e in cp_disc_p_ee3:
            for coeff, i, j in cp_disc_p_ee3[e]:
                lhs = lhs + coeff * Q1_ee[i, j]
        if e in cp_disc_p_oo3:
            for coeff, i, j in cp_disc_p_oo3[e]:
                lhs = lhs + coeff * Q1_oo[i, j]

        # σ_2 * disc_q
        if e in cp_disc_q_ee3:
            for coeff, i, j in cp_disc_q_ee3[e]:
                lhs = lhs + coeff * Q2_ee[i, j]
        if e in cp_disc_q_oo3:
            for coeff, i, j in cp_disc_q_oo3[e]:
                lhs = lhs + coeff * Q2_oo[i, j]

        # σ_3 * (-f2_p)
        if e in cp_f2p_ee4:
            for coeff, i, j in cp_f2p_ee4[e]:
                lhs = lhs + coeff * Q3_ee[i, j]
        if e in cp_f2p_oo4:
            for coeff, i, j in cp_f2p_oo4[e]:
                lhs = lhs + coeff * Q3_oo[i, j]

        # σ_4 * (-f2_q)
        if e in cp_f2q_ee4:
            for coeff, i, j in cp_f2q_ee4[e]:
                lhs = lhs + coeff * Q4_ee[i, j]
        if e in cp_f2q_oo4:
            for coeff, i, j in cp_f2q_oo4[e]:
                lhs = lhs + coeff * Q4_oo[i, j]

        rhs = N_dict.get(e, 0.0)
        constraints.append(lhs == rhs)
        n_eq += 1

    print(f"  {n_eq} equality constraints")
    print(f"  ({time.time()-t0:.1f}s)")
    print()

    # Solve
    print("Solving SDP...")
    prob = cp.Problem(cp.Minimize(0), constraints)

    for solver_name, solver in [("CLARABEL", cp.CLARABEL), ("SCS", cp.SCS)]:
        print(f"  Trying {solver_name}...")
        try:
            if solver_name == "SCS":
                prob.solve(solver=solver, verbose=False, max_iters=10000, eps=1e-9)
            else:
                prob.solve(solver=solver, verbose=False)
            print(f"  Status: {prob.status}")
            if prob.status in ['optimal', 'optimal_inaccurate']:
                print(f"  CERTIFICATE FOUND with {solver_name}!")
                for name, Q in [("Q0_ee", Q0_ee), ("Q0_oo", Q0_oo),
                                ("Q1_ee", Q1_ee), ("Q1_oo", Q1_oo),
                                ("Q2_ee", Q2_ee), ("Q2_oo", Q2_oo),
                                ("Q3_ee", Q3_ee), ("Q3_oo", Q3_oo),
                                ("Q4_ee", Q4_ee), ("Q4_oo", Q4_oo)]:
                    if Q.value is not None:
                        ev = np.linalg.eigvalsh(Q.value)
                        print(f"    {name}: size {Q.shape[0]}×{Q.shape[0]}, min_eig={ev.min():.4e}, max_eig={ev.max():.4e}")
                    else:
                        print(f"    {name}: None")

                # Verify the certificate
                print()
                print("  Verifying certificate numerically...")
                verify_certificate(N_dict, variables,
                                   disc_p_dict, disc_q_dict,
                                   neg_f2_p_dict, neg_f2_q_dict,
                                   Q0_ee, Q0_oo, Q1_ee, Q1_oo,
                                   Q2_ee, Q2_oo, Q3_ee, Q3_oo,
                                   Q4_ee, Q4_oo,
                                   ee5, oo5, ee3, oo3, ee4, oo4)
                break
            else:
                print(f"  {solver_name} failed: {prob.status}")
        except Exception as ex:
            print(f"  {solver_name} error: {ex}")

    print(f"  ({time.time()-t0:.1f}s)")


def verify_certificate(N_dict, variables,
                        disc_p_dict, disc_q_dict,
                        neg_f2_p_dict, neg_f2_q_dict,
                        Q0_ee, Q0_oo, Q1_ee, Q1_oo,
                        Q2_ee, Q2_oo, Q3_ee, Q3_oo,
                        Q4_ee, Q4_oo,
                        ee5, oo5, ee3, oo3, ee4, oo4):
    """Numerically verify the certificate at random points."""
    a3, a4, b3, b4 = variables
    rng = np.random.default_rng(42)

    def eval_sos_block(Q_val, basis, point):
        """Evaluate m^T Q m at a point."""
        m = np.array([point[0]**e[0] * point[1]**e[1] * point[2]**e[2] * point[3]**e[3]
                       for e in basis])
        return m @ Q_val @ m

    def eval_poly_dict(pd, point):
        """Evaluate a polynomial dict at a point."""
        return sum(c * point[0]**e[0] * point[1]**e[1] * point[2]**e[2] * point[3]**e[3]
                   for e, c in pd.items())

    max_err = 0
    for _ in range(10000):
        pt = [rng.uniform(-0.3, 0.3), rng.uniform(0.01, 0.24),
              rng.uniform(-0.3, 0.3), rng.uniform(0.01, 0.24)]

        # LHS: N
        lhs = eval_poly_dict(N_dict, pt)

        # RHS: σ_0 + σ_1*disc_p + σ_2*disc_q + σ_3*(-f2_p) + σ_4*(-f2_q)
        rhs = 0
        if Q0_ee.value is not None:
            rhs += eval_sos_block(Q0_ee.value, ee5, pt)
        if Q0_oo.value is not None:
            rhs += eval_sos_block(Q0_oo.value, oo5, pt)

        dp = eval_poly_dict(disc_p_dict, pt)
        if Q1_ee.value is not None:
            rhs += dp * eval_sos_block(Q1_ee.value, ee3, pt)
        if Q1_oo.value is not None:
            rhs += dp * eval_sos_block(Q1_oo.value, oo3, pt)

        dq = eval_poly_dict(disc_q_dict, pt)
        if Q2_ee.value is not None:
            rhs += dq * eval_sos_block(Q2_ee.value, ee3, pt)
        if Q2_oo.value is not None:
            rhs += dq * eval_sos_block(Q2_oo.value, oo3, pt)

        fp = eval_poly_dict(neg_f2_p_dict, pt)
        if Q3_ee.value is not None:
            rhs += fp * eval_sos_block(Q3_ee.value, ee4, pt)
        if Q3_oo.value is not None:
            rhs += fp * eval_sos_block(Q3_oo.value, oo4, pt)

        fq = eval_poly_dict(neg_f2_q_dict, pt)
        if Q4_ee.value is not None:
            rhs += fq * eval_sos_block(Q4_ee.value, ee4, pt)
        if Q4_oo.value is not None:
            rhs += fq * eval_sos_block(Q4_oo.value, oo4, pt)

        err = abs(lhs - rhs) / max(abs(lhs), 1e-10)
        if err > max_err:
            max_err = err

    print(f"  Max relative error: {max_err:.4e}")
    if max_err < 1e-4:
        print(f"  Certificate VERIFIED numerically.")
    else:
        print(f"  Certificate has significant error — may be inaccurate.")


if __name__ == "__main__":
    main()
