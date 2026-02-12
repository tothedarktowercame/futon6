#!/usr/bin/env python3
"""SOS / Positivstellensatz certificate search for n=4 surplus.

Attempts to find a certificate: N = σ_0 + σ_1*disc_p + σ_2*disc_q + ...
where σ_i are SOS polynomials and disc_p, disc_q are the discriminant
constraints (non-negative on the real-rooted domain).

Uses CVXPY for the semidefinite program.
"""

import sympy as sp
from sympy import symbols, Rational, expand, Poly
import numpy as np
import time
from itertools import product as iter_product


def build_surplus_numerator():
    """Build the surplus numerator N such that surplus = N/D, D > 0 on the cone."""
    a3, a4, b3, b4 = sp.symbols('a3 a4 b3 b4')

    # p: centered, a2=-1
    disc_p = expand(256*a4**3 - 128*a4**2 - 144*a3**2*a4
                    - 27*a3**4 + 16*a4 + 4*a3**2)
    f1_p = 1 + 12*a4
    f2_p = 9*a3**2 + 8*a4 - 2

    # q: centered, b2=-1
    disc_q = expand(256*b4**3 - 128*b4**2 - 144*b3**2*b4
                    - 27*b3**4 + 16*b4 + 4*b3**2)
    f1_q = 1 + 12*b4
    f2_q = 9*b3**2 + 8*b4 - 2

    # r = p ⊞ q: c2=-2, c3=a3+b3, c4=a4+1/6+b4
    c3 = a3 + b3
    c4 = a4 + Rational(1, 6) + b4
    disc_r = expand(256*c4**3 - 512*c4**2 + 288*c3**2*c4
                    - 27*c3**4 + 256*c4 + 32*c3**2)
    f1_r = expand(4 + 12*c4)
    f2_r = expand(-16 + 16*c4 + 9*c3**2)

    # surplus = -disc_r/(4*f1_r*f2_r) + disc_p/(4*f1_p*f2_p) + disc_q/(4*f1_q*f2_q)
    # Common denominator: 4^3 * f1_p * f2_p * f1_q * f2_q * f1_r * f2_r
    # But each factor f1*f2 < 0 on the domain, and there are 3 such factors.

    # Let's compute the surplus as a fraction and extract numerator.
    surplus_expr = (-disc_r / (4 * f1_r * f2_r)
                    + disc_p / (4 * f1_p * f2_p)
                    + disc_q / (4 * f1_q * f2_q))

    # Get numerator and denominator
    surplus_frac = sp.together(surplus_expr)
    N, D = sp.fraction(surplus_frac)
    N = expand(N)
    D = expand(D)

    return N, D, disc_p, disc_q, f1_p, f2_p, f1_q, f2_q, (a3, a4, b3, b4)


def poly_to_dict(poly_expr, variables):
    """Convert sympy expression to {exponent_tuple: coefficient} dict."""
    poly = Poly(poly_expr, *variables)
    return dict(poly.as_dict())


def enumerate_monomials(n_vars, max_degree):
    """Enumerate all monomials up to max_degree in n_vars variables.
    Returns list of exponent tuples."""
    if n_vars == 0:
        return [()]
    if n_vars == 1:
        return [(d,) for d in range(max_degree + 1)]

    result = []
    for d in range(max_degree + 1):
        for rest in enumerate_monomials(n_vars - 1, max_degree - d):
            result.append((d,) + rest)
    return result


def multiply_poly_dicts(p1, p2):
    """Multiply two polynomial dicts."""
    result = {}
    for e1, c1 in p1.items():
        for e2, c2 in p2.items():
            e = tuple(a + b for a, b in zip(e1, e2))
            result[e] = result.get(e, 0) + c1 * c2
    return result


def main():
    print("=" * 70)
    print("SOS / Positivstellensatz certificate search")
    print("=" * 70)
    print()

    t0 = time.time()

    # Build surplus numerator
    print("Building surplus numerator...")
    N, D, disc_p, disc_q, f1_p, f2_p, f1_q, f2_q, variables = build_surplus_numerator()
    a3, a4, b3, b4 = variables

    # Get polynomial dictionaries
    N_dict = poly_to_dict(N, variables)
    print(f"  N has {len(N_dict)} terms")
    max_deg = max(sum(e) for e in N_dict.keys())
    print(f"  Max total degree: {max_deg}")

    # Check sign of D numerically
    import random
    random.seed(42)
    pos_count = 0
    for _ in range(1000):
        vals = {a3: random.uniform(-0.2, 0.2), a4: random.uniform(0.01, 0.24),
                b3: random.uniform(-0.2, 0.2), b4: random.uniform(0.01, 0.24)}
        d_val = float(D.subs(vals))
        if d_val > 0:
            pos_count += 1
    print(f"  D > 0 in {pos_count}/1000 samples (should be 1000)")

    # Check N >= 0 at a few points
    n_neg = 0
    for _ in range(1000):
        vals = {a3: random.uniform(-0.2, 0.2), a4: random.uniform(0.01, 0.24),
                b3: random.uniform(-0.2, 0.2), b4: random.uniform(0.01, 0.24)}
        n_val = float(N.subs(vals))
        if n_val < -1e-10:
            n_neg += 1
    print(f"  N < 0 in {n_neg}/1000 samples (should be 0)")
    print(f"  ({time.time()-t0:.1f}s)")
    print()

    # Get constraint polynomial dicts
    disc_p_dict = poly_to_dict(disc_p, variables)
    disc_q_dict = poly_to_dict(disc_q, variables)
    f1_p_dict = poly_to_dict(f1_p, variables)
    f2_p_dict = poly_to_dict(-f2_p, variables)  # -f2 > 0 on domain
    f1_q_dict = poly_to_dict(f1_q, variables)
    f2_q_dict = poly_to_dict(-f2_q, variables)  # -f2 > 0 on domain

    print("Constraint degrees:")
    print(f"  disc_p: degree {max(sum(e) for e in disc_p_dict)}")
    print(f"  disc_q: degree {max(sum(e) for e in disc_q_dict)}")
    print(f"  f1_p: degree {max(sum(e) for e in f1_p_dict)}")
    print(f"  -f2_p: degree {max(sum(e) for e in f2_p_dict)}")
    print()

    # ===== SDP Setup =====
    # Try: N = σ_0 + σ_1*disc_p + σ_2*disc_q
    # where σ_i are SOS polynomials.
    #
    # σ_0 has degree ≤ max_deg = 10, basis up to degree 5
    # σ_1 has degree ≤ 10 - 4 = 6, basis up to degree 3
    # σ_2 has degree ≤ 10 - 4 = 6, basis up to degree 3

    import cvxpy as cp

    print("Setting up SDP...")
    n_vars = 4

    # Monomial bases
    basis_5 = enumerate_monomials(n_vars, 5)  # for σ_0
    basis_3 = enumerate_monomials(n_vars, 3)  # for σ_1, σ_2

    n5 = len(basis_5)
    n3 = len(basis_3)
    print(f"  Basis sizes: σ_0={n5}×{n5}, σ_1=σ_2={n3}×{n3}")

    # SDP variables: PSD matrices Q0, Q1, Q2
    Q0 = cp.Variable((n5, n5), symmetric=True)
    Q1 = cp.Variable((n3, n3), symmetric=True)
    Q2 = cp.Variable((n3, n3), symmetric=True)

    constraints = [Q0 >> 0, Q1 >> 0, Q2 >> 0]

    # For each monomial in N, build the equation:
    # N[e] = Σ_{i,j: basis_5[i]+basis_5[j]=e} Q0[i,j]
    #       + Σ_{e'} disc_p[e'] * Σ_{i,j: basis_3[i]+basis_3[j]=e-e'} Q1[i,j]
    #       + Σ_{e'} disc_q[e'] * Σ_{i,j: basis_3[i]+basis_3[j]=e-e'} Q2[i,j]

    # Precompute: for each exponent e, which (i,j) pairs in basis contribute
    def build_gram_contributions(basis):
        """For each exponent tuple e, list (i,j) pairs with basis[i]+basis[j]=e."""
        contrib = {}
        n = len(basis)
        for i in range(n):
            for j in range(i, n):
                e = tuple(a + b for a, b in zip(basis[i], basis[j]))
                if e not in contrib:
                    contrib[e] = []
                contrib[e].append((i, j))
        return contrib

    print("  Building Gram contributions...")
    contrib_5 = build_gram_contributions(basis_5)
    contrib_3 = build_gram_contributions(basis_3)

    # Multiply disc_p_dict with contrib_3 to get contributions for σ_1*disc_p
    def constraint_contributions(g_dict, gram_contrib):
        """For each final exponent e, the contribution from σ*g."""
        result = {}  # e -> list of (coeff, i, j) meaning coeff * Q[i,j]
        for e_g, c_g in g_dict.items():
            for e_q, pairs in gram_contrib.items():
                e = tuple(a + b for a, b in zip(e_g, e_q))
                if e not in result:
                    result[e] = []
                for i, j in pairs:
                    result[e].append((float(c_g), i, j))
        return result

    print("  Building constraint contributions...")
    contrib_disc_p = constraint_contributions(disc_p_dict, contrib_3)
    contrib_disc_q = constraint_contributions(disc_q_dict, contrib_3)

    # Collect all monomials that appear
    all_exponents = set(N_dict.keys())
    for c in [contrib_5, contrib_disc_p, contrib_disc_q]:
        all_exponents |= set(c.keys())

    print(f"  Total monomial constraints: {len(all_exponents)}")
    print(f"  ({time.time()-t0:.1f}s)")

    # Build constraints: for each exponent e, the sum must equal N[e]
    print("  Building SDP constraints...")
    n_constraints = 0
    for e in sorted(all_exponents, key=lambda x: (sum(x), x)):
        lhs = 0

        # σ_0 contribution
        if e in contrib_5:
            for i, j in contrib_5[e]:
                if i == j:
                    lhs = lhs + Q0[i, j]
                else:
                    lhs = lhs + 2 * Q0[i, j]  # symmetric: Q[i,j] + Q[j,i] = 2*Q[i,j]

        # σ_1 * disc_p contribution
        if e in contrib_disc_p:
            for coeff, i, j in contrib_disc_p[e]:
                if i == j:
                    lhs = lhs + coeff * Q1[i, j]
                else:
                    lhs = lhs + 2 * coeff * Q1[i, j]

        # σ_2 * disc_q contribution
        if e in contrib_disc_q:
            for coeff, i, j in contrib_disc_q[e]:
                if i == j:
                    lhs = lhs + coeff * Q2[i, j]
                else:
                    lhs = lhs + 2 * coeff * Q2[i, j]

        # RHS: coefficient of e in N
        rhs = float(N_dict.get(e, 0))

        constraints.append(lhs == rhs)
        n_constraints += 1

    print(f"  Total equality constraints: {n_constraints}")
    print(f"  ({time.time()-t0:.1f}s)")
    print()

    # Solve
    print("Solving SDP (σ_0 + σ_1*disc_p + σ_2*disc_q)...")
    prob = cp.Problem(cp.Minimize(0), constraints)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
        print(f"  Status: {prob.status}")
        if prob.status in ['optimal', 'optimal_inaccurate']:
            print(f"  CERTIFICATE FOUND!")
            # Check Q matrices are PSD
            eigvals_0 = np.linalg.eigvalsh(Q0.value)
            eigvals_1 = np.linalg.eigvalsh(Q1.value)
            eigvals_2 = np.linalg.eigvalsh(Q2.value)
            print(f"  Q0 min eigenvalue: {eigvals_0.min():.6e}")
            print(f"  Q1 min eigenvalue: {eigvals_1.min():.6e}")
            print(f"  Q2 min eigenvalue: {eigvals_2.min():.6e}")
        else:
            print(f"  No certificate found with this ansatz.")
    except Exception as ex:
        print(f"  Solver error: {ex}")
    print(f"  ({time.time()-t0:.1f}s)")
    print()

    # If basic certificate fails, try adding more constraints
    if prob.status not in ['optimal', 'optimal_inaccurate']:
        print("Trying extended certificate:")
        print("  N = σ_0 + σ_1*disc_p + σ_2*disc_q + σ_3*(-f2_p) + σ_4*(-f2_q)")
        print("  (adding domain constraints -f2 > 0)")

        # σ_3 has degree ≤ 10 - 2 = 8, basis up to degree 4
        basis_4 = enumerate_monomials(n_vars, 4)
        n4 = len(basis_4)
        print(f"  Extra basis size: σ_3=σ_4={n4}×{n4}")

        Q3 = cp.Variable((n4, n4), symmetric=True)
        Q4 = cp.Variable((n4, n4), symmetric=True)

        Q0b = cp.Variable((n5, n5), symmetric=True)
        Q1b = cp.Variable((n3, n3), symmetric=True)
        Q2b = cp.Variable((n3, n3), symmetric=True)

        constraints2 = [Q0b >> 0, Q1b >> 0, Q2b >> 0, Q3 >> 0, Q4 >> 0]

        contrib_4 = build_gram_contributions(basis_4)
        contrib_f2p = constraint_contributions(f2_p_dict, contrib_4)
        contrib_f2q = constraint_contributions(f2_q_dict, contrib_4)

        all_exp2 = set(N_dict.keys())
        for c in [contrib_5, contrib_disc_p, contrib_disc_q, contrib_f2p, contrib_f2q]:
            all_exp2 |= set(c.keys())

        print(f"  Total monomial constraints: {len(all_exp2)}")

        for e in sorted(all_exp2, key=lambda x: (sum(x), x)):
            lhs = 0

            if e in contrib_5:
                for i, j in contrib_5[e]:
                    mult = 1 if i == j else 2
                    lhs = lhs + mult * Q0b[i, j]

            if e in contrib_disc_p:
                for coeff, i, j in contrib_disc_p[e]:
                    mult = 1 if i == j else 2
                    lhs = lhs + mult * coeff * Q1b[i, j]

            if e in contrib_disc_q:
                for coeff, i, j in contrib_disc_q[e]:
                    mult = 1 if i == j else 2
                    lhs = lhs + mult * coeff * Q2b[i, j]

            if e in contrib_f2p:
                for coeff, i, j in contrib_f2p[e]:
                    mult = 1 if i == j else 2
                    lhs = lhs + mult * coeff * Q3[i, j]

            if e in contrib_f2q:
                for coeff, i, j in contrib_f2q[e]:
                    mult = 1 if i == j else 2
                    lhs = lhs + mult * coeff * Q4[i, j]

            rhs = float(N_dict.get(e, 0))
            constraints2.append(lhs == rhs)

        prob2 = cp.Problem(cp.Minimize(0), constraints2)
        try:
            prob2.solve(solver=cp.CLARABEL, verbose=False)
            print(f"  Status: {prob2.status}")
            if prob2.status in ['optimal', 'optimal_inaccurate']:
                print(f"  CERTIFICATE FOUND!")
                eigvals = [np.linalg.eigvalsh(Q.value).min()
                           for Q in [Q0b, Q1b, Q2b, Q3, Q4]]
                print(f"  Min eigenvalues: {[f'{v:.6e}' for v in eigvals]}")
            else:
                print(f"  No certificate with this ansatz either.")
        except Exception as ex:
            print(f"  Solver error: {ex}")
        print(f"  ({time.time()-t0:.1f}s)")
        print()

        # Try with f1 constraints too
        if prob2.status not in ['optimal', 'optimal_inaccurate']:
            print("Trying full certificate:")
            print("  N = σ_0 + σ_1*disc_p + σ_2*disc_q + σ_3*(-f2_p) + σ_4*(-f2_q) + σ_5*f1_p + σ_6*f1_q")

            # σ_5 has degree ≤ 10 - 1 = 9, must be even → degree 8
            # Basis up to degree 4
            Q5 = cp.Variable((n4, n4), symmetric=True)
            Q6 = cp.Variable((n4, n4), symmetric=True)

            Q0c = cp.Variable((n5, n5), symmetric=True)
            Q1c = cp.Variable((n3, n3), symmetric=True)
            Q2c = cp.Variable((n3, n3), symmetric=True)
            Q3c = cp.Variable((n4, n4), symmetric=True)
            Q4c = cp.Variable((n4, n4), symmetric=True)

            constraints3 = [Q0c >> 0, Q1c >> 0, Q2c >> 0,
                            Q3c >> 0, Q4c >> 0, Q5 >> 0, Q6 >> 0]

            # f1_p and f1_q contributions
            # f1_p has degree 1, σ_5*f1_p has degree 8+1=9
            # But we need total degree ≤ 10, and f1 has degree 1,
            # so σ_5 can have degree up to 9 (odd). But SOS must be even.
            # σ_5 degree 8 → σ_5*f1_p has degree 9. That's ≤ 10. OK.
            # Actually, the monomial degree mismatch: σ_5*f1_p gives
            # monomials up to degree 9, not 10. So this can only match
            # N terms up to degree 9. N has terms up to degree 10, so
            # we need σ_0 to handle the degree 10 part.

            contrib_f1p = constraint_contributions(f1_p_dict, contrib_4)
            contrib_f1q = constraint_contributions(f1_q_dict, contrib_4)

            all_exp3 = set(N_dict.keys())
            for c in [contrib_5, contrib_disc_p, contrib_disc_q,
                       contrib_f2p, contrib_f2q, contrib_f1p, contrib_f1q]:
                all_exp3 |= set(c.keys())

            print(f"  Total monomial constraints: {len(all_exp3)}")

            for e in sorted(all_exp3, key=lambda x: (sum(x), x)):
                lhs = 0

                if e in contrib_5:
                    for i, j in contrib_5[e]:
                        mult = 1 if i == j else 2
                        lhs = lhs + mult * Q0c[i, j]

                if e in contrib_disc_p:
                    for coeff, i, j in contrib_disc_p[e]:
                        mult = 1 if i == j else 2
                        lhs = lhs + mult * coeff * Q1c[i, j]

                if e in contrib_disc_q:
                    for coeff, i, j in contrib_disc_q[e]:
                        mult = 1 if i == j else 2
                        lhs = lhs + mult * coeff * Q2c[i, j]

                if e in contrib_f2p:
                    for coeff, i, j in contrib_f2p[e]:
                        mult = 1 if i == j else 2
                        lhs = lhs + mult * coeff * Q3c[i, j]

                if e in contrib_f2q:
                    for coeff, i, j in contrib_f2q[e]:
                        mult = 1 if i == j else 2
                        lhs = lhs + mult * coeff * Q4c[i, j]

                if e in contrib_f1p:
                    for coeff, i, j in contrib_f1p[e]:
                        mult = 1 if i == j else 2
                        lhs = lhs + mult * coeff * Q5[i, j]

                if e in contrib_f1q:
                    for coeff, i, j in contrib_f1q[e]:
                        mult = 1 if i == j else 2
                        lhs = lhs + mult * coeff * Q6[i, j]

                rhs = float(N_dict.get(e, 0))
                constraints3.append(lhs == rhs)

            prob3 = cp.Problem(cp.Minimize(0), constraints3)
            try:
                prob3.solve(solver=cp.CLARABEL, verbose=False)
                print(f"  Status: {prob3.status}")
                if prob3.status in ['optimal', 'optimal_inaccurate']:
                    print(f"  CERTIFICATE FOUND!")
                    for name, Q in [("Q0", Q0c), ("Q1", Q1c), ("Q2", Q2c),
                                     ("Q3", Q3c), ("Q4", Q4c), ("Q5", Q5), ("Q6", Q6)]:
                        ev = np.linalg.eigvalsh(Q.value).min()
                        print(f"    {name} min eigenvalue: {ev:.6e}")
                else:
                    print(f"  No certificate with full constraints either.")
                    print(f"  May need cross-terms (σ*disc_p*(-f2_p)) or higher degree.")
            except Exception as ex:
                print(f"  Solver error: {ex}")
            print(f"  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
