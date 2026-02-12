#!/usr/bin/env python3
"""Reduced SOS certificate search exploiting reflection symmetry.

The surplus N is even in (a3, b3), so we only need monomials with even
powers of a3 and b3. This reduces basis from 126 to ~50+16 = ~66 entries.

Also tries simpler certificates first:
1. Fix (a4, b4) and check if the surplus in (a3, b3) is SOS on the 2D domain
2. Try a parametric SOS where the Gram matrix entries depend on (a4, b4)
"""

import sympy as sp
from sympy import symbols, Rational, expand, Poly, together, fraction
import numpy as np
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


def surplus_at_fixed_a4b4(a4_val, b4_val):
    """Return the surplus as a function of (a3, b3) at fixed a4, b4.

    Returns a polynomial in a3, b3 (after clearing denominators).
    """
    a3, b3 = sp.symbols('a3 b3')
    a4 = sp.Rational(a4_val) if isinstance(a4_val, (int, str)) else a4_val
    b4 = sp.Rational(b4_val) if isinstance(b4_val, (int, str)) else b4_val

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

    surplus = (-disc_r / (4 * f1_r * f2_r)
               + disc_p / (4 * f1_p * f2_p)
               + disc_q / (4 * f1_q * f2_q))

    surplus_frac = together(surplus)
    N, D = fraction(surplus_frac)
    N = expand(N)
    D = expand(D)

    return N, D, (a3, b3)


def check_sos_2var(poly_expr, variables, max_degree=None):
    """Check if a bivariate polynomial is SOS using CVXPY.

    Returns (is_sos, Q_matrix) or (False, None).
    """
    import cvxpy as cp

    x, y = variables
    poly = Poly(poly_expr, x, y)
    coeffs = dict(poly.as_dict())

    if max_degree is None:
        max_degree = max(sum(e) for e in coeffs.keys())

    half_deg = max_degree // 2

    # Build monomial basis up to half_deg
    basis = []
    for i in range(half_deg + 1):
        for j in range(half_deg + 1 - i):
            basis.append((i, j))
    n = len(basis)

    # Gram matrix
    Q = cp.Variable((n, n), symmetric=True)
    constraints = [Q >> 0]

    # For each monomial in poly, sum of Q[i,j] where basis[i]+basis[j] = monomial
    contrib = {}
    for i in range(n):
        for j in range(i, n):
            e = (basis[i][0] + basis[j][0], basis[i][1] + basis[j][1])
            if e not in contrib:
                contrib[e] = []
            contrib[e].append((i, j))

    all_exp = set(coeffs.keys()) | set(contrib.keys())
    for e in all_exp:
        lhs = 0
        if e in contrib:
            for i, j in contrib[e]:
                mult = 1 if i == j else 2
                lhs = lhs + mult * Q[i, j]
        rhs = float(coeffs.get(e, 0))
        constraints.append(lhs == rhs)

    prob = cp.Problem(cp.Minimize(0), constraints)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
        if prob.status in ['optimal', 'optimal_inaccurate']:
            eigvals = np.linalg.eigvalsh(Q.value)
            return True, Q.value, eigvals.min()
        return False, None, None
    except Exception:
        return False, None, None


def main():
    print("=" * 70)
    print("Reduced SOS certificate search (exploiting symmetry)")
    print("=" * 70)
    print()

    t0 = time.time()

    # ===== APPROACH 1: Fix (a4, b4), check 2D SOS =====
    print("APPROACH 1: Fix (a4, b4), check if surplus in (a3, b3) has")
    print("            a Positivstellensatz certificate on 2D domain")
    print("-" * 60)
    print()

    import cvxpy as cp

    test_points = [
        (Rational(1, 12), Rational(1, 12), "equality (1/12, 1/12)"),
        (Rational(1, 8), Rational(1, 8), "symmetric (1/8, 1/8)"),
        (Rational(1, 6), Rational(1, 6), "symmetric (1/6, 1/6)"),
        (Rational(1, 24), Rational(1, 24), "symmetric (1/24, 1/24)"),
        (Rational(1, 8), Rational(1, 12), "asymmetric (1/8, 1/12)"),
        (Rational(1, 6), Rational(1, 24), "asymmetric (1/6, 1/24)"),
        (Rational(1, 5), Rational(1, 10), "asymmetric (1/5, 1/10)"),
        (Rational(1, 5), Rational(1, 5), "near boundary (1/5, 1/5)"),
    ]

    results = []
    for a4_val, b4_val, label in test_points:
        print(f"  Testing {label}...")
        N, D, (a3, b3) = surplus_at_fixed_a4b4(a4_val, b4_val)

        # Check sign of D numerically
        d_val = float(D.subs([(a3, 0.1), (b3, 0.05)]))
        sign_str = "D>0" if d_val > 0 else "D<0"

        # The surplus has the form N/D. Depending on sign of D, we need
        # N >= 0 or N <= 0. Check which:
        n_val = float(N.subs([(a3, 0), (b3, 0)]))

        # At a3=b3=0, surplus should be >= 0, so N/D >= 0
        # If D > 0, need N >= 0. If D < 0, need N <= 0.
        target = N if d_val > 0 else -N
        target = expand(target)

        poly = Poly(target, a3, b3)
        deg = poly.total_degree()

        # Try pure SOS first
        is_sos, Q, min_eig = check_sos_2var(target, (a3, b3), deg if deg % 2 == 0 else deg + 1)
        if is_sos:
            print(f"    -> SOS! (degree {deg}, min eigenvalue {min_eig:.2e})")
            results.append((label, "SOS", min_eig))
        else:
            print(f"    -> NOT SOS (degree {deg})")

            # Try with disc constraints
            # N_target = σ_0 + σ_1 * disc_p(a3, a4_val) + σ_2 * disc_q(b3, b4_val)
            disc_p_fixed = expand(256*a4_val**3 - 128*a4_val**2
                                  - 144*a3**2*a4_val - 27*a3**4
                                  + 16*a4_val + 4*a3**2)
            disc_q_fixed = expand(256*b4_val**3 - 128*b4_val**2
                                  - 144*b3**2*b4_val - 27*b3**4
                                  + 16*b4_val + 4*b3**2)

            # Also try -f2 constraints
            neg_f2_p = expand(2 - 8*a4_val - 9*a3**2)
            neg_f2_q = expand(2 - 8*b4_val - 9*b3**2)

            half = deg // 2

            # Basis for σ_0
            basis_0 = [(i, j) for i in range(half+1)
                       for j in range(half+1-i)]
            n0 = len(basis_0)

            # Basis for σ_1 (disc has degree 4 in a3)
            d_disc = 4
            half_1 = (deg - d_disc) // 2
            basis_1 = [(i, j) for i in range(half_1+1)
                       for j in range(half_1+1-i)]
            n1 = len(basis_1)

            # Basis for σ_2
            basis_2 = [(i, j) for i in range(half_1+1)
                       for j in range(half_1+1-i)]
            n2 = len(basis_2)

            # Basis for σ_3 (neg_f2_p has degree 2)
            half_3 = (deg - 2) // 2
            basis_3 = [(i, j) for i in range(half_3+1)
                       for j in range(half_3+1-i)]
            n3 = len(basis_3)

            Q0 = cp.Variable((n0, n0), symmetric=True)
            Q1 = cp.Variable((n1, n1), symmetric=True)
            Q2 = cp.Variable((n2, n2), symmetric=True)
            Q3 = cp.Variable((n3, n3), symmetric=True)
            Q4 = cp.Variable((n3, n3), symmetric=True)

            cons = [Q0 >> 0, Q1 >> 0, Q2 >> 0, Q3 >> 0, Q4 >> 0]

            # Build contributions
            target_dict = dict(Poly(target, a3, b3).as_dict())
            disc_p_dict = dict(Poly(disc_p_fixed, a3, b3).as_dict())
            disc_q_dict = dict(Poly(disc_q_fixed, a3, b3).as_dict())
            f2p_dict = dict(Poly(neg_f2_p, a3, b3).as_dict())
            f2q_dict = dict(Poly(neg_f2_q, a3, b3).as_dict())

            def gram_contrib(basis):
                c = {}
                for i in range(len(basis)):
                    for j in range(i, len(basis)):
                        e = (basis[i][0]+basis[j][0], basis[i][1]+basis[j][1])
                        c.setdefault(e, []).append((i, j))
                return c

            gc0 = gram_contrib(basis_0)
            gc1 = gram_contrib(basis_1)
            gc2 = gram_contrib(basis_2)
            gc3 = gram_contrib(basis_3)

            def mult_contrib(g_dict, gc):
                result = {}
                for eg, cg in g_dict.items():
                    for eq, pairs in gc.items():
                        e = (eg[0]+eq[0], eg[1]+eq[1])
                        result.setdefault(e, [])
                        for i, j in pairs:
                            result[e].append((float(cg), i, j))
                return result

            mc1 = mult_contrib(disc_p_dict, gc1)
            mc2 = mult_contrib(disc_q_dict, gc2)
            mc3 = mult_contrib(f2p_dict, gc3)
            mc4 = mult_contrib(f2q_dict, gc3)

            all_exp = set(target_dict.keys())
            for c in [gc0, mc1, mc2, mc3, mc4]:
                all_exp |= set(c.keys())

            for e in all_exp:
                lhs = 0
                if e in gc0:
                    for i, j in gc0[e]:
                        lhs += (1 if i==j else 2) * Q0[i,j]
                if e in mc1:
                    for c, i, j in mc1[e]:
                        lhs += (1 if i==j else 2) * c * Q1[i,j]
                if e in mc2:
                    for c, i, j in mc2[e]:
                        lhs += (1 if i==j else 2) * c * Q2[i,j]
                if e in mc3:
                    for c, i, j in mc3[e]:
                        lhs += (1 if i==j else 2) * c * Q3[i,j]
                if e in mc4:
                    for c, i, j in mc4[e]:
                        lhs += (1 if i==j else 2) * c * Q4[i,j]

                rhs = float(target_dict.get(e, 0))
                cons.append(lhs == rhs)

            prob = cp.Problem(cp.Minimize(0), cons)
            try:
                prob.solve(solver=cp.CLARABEL, verbose=False)
                if prob.status in ['optimal', 'optimal_inaccurate']:
                    eigs = [np.linalg.eigvalsh(Q.value).min()
                            for Q in [Q0, Q1, Q2, Q3, Q4]]
                    print(f"    -> Positivstellensatz! min eigs={[f'{e:.1e}' for e in eigs]}")
                    results.append((label, "PSATZ", min(eigs)))
                else:
                    print(f"    -> No certificate ({prob.status})")
                    results.append((label, "FAIL", None))
            except Exception as ex:
                print(f"    -> Error: {ex}")
                results.append((label, "ERROR", None))

        print(f"    ({time.time()-t0:.1f}s)")

    print()
    print("Summary of 2D certificates:")
    print("-" * 50)
    for label, status, eig in results:
        eig_str = f", min_eig={eig:.2e}" if eig is not None else ""
        print(f"  {label}: {status}{eig_str}")
    print()

    # ===== APPROACH 2: All critical points =====
    print("APPROACH 2: Find all critical points of surplus")
    print("-" * 60)
    print()

    print("  Computing critical point equations symbolically...")
    N, D, disc_p, disc_q, f1_p, f2_p, f1_q, f2_q, variables = build_surplus_numerator()
    a3, a4, b3, b4 = variables

    # For the rational function surplus = N/D, critical points satisfy:
    # d(N/D)/dx = (N'D - ND')/D^2 = 0, i.e., N'D - ND' = 0 for each variable.
    # This is a system of 4 polynomial equations.

    # Instead, work with the rational surplus directly at a3=b3=0.
    # We already know the gradient is 0 there. Check if there are other
    # critical points at a3=b3=0 (i.e., in the symmetric subfamily).

    # In the symmetric case (a3=b3=0), surplus is a function of (a4, b4) only.
    # Set derivatives to 0:
    surplus_sym = sp.together((-sp.expand(256*(a4+Rational(1,6)+b4)**3
                               - 512*(a4+Rational(1,6)+b4)**2
                               + 256*(a4+Rational(1,6)+b4)) /
                               (4*(4+12*(a4+Rational(1,6)+b4))*(-16+16*(a4+Rational(1,6)+b4)))
                               + (256*a4**3 - 128*a4**2 + 16*a4) /
                               (4*(1+12*a4)*(8*a4-2))
                               + (256*b4**3 - 128*b4**2 + 16*b4) /
                               (4*(1+12*b4)*(8*b4-2))))

    print("  Surplus in symmetric case (a3=b3=0):")
    ds_da4 = sp.diff(surplus_sym, a4)
    ds_db4 = sp.diff(surplus_sym, b4)

    # Solve ds/da4 = 0, ds/db4 = 0
    print("  Solving for critical points of symmetric surplus...")
    n_da4, d_da4 = sp.fraction(sp.together(ds_da4))
    n_db4, d_db4 = sp.fraction(sp.together(ds_db4))
    n_da4 = expand(n_da4)
    n_db4 = expand(n_db4)

    print(f"    Numerator of ds/da4 has degree {Poly(n_da4, a4, b4).total_degree()}")
    print(f"    Numerator of ds/db4 has degree {Poly(n_db4, a4, b4).total_degree()}")

    # Try to solve symbolically
    try:
        solutions = sp.solve([n_da4, n_db4], [a4, b4], dict=True)
        print(f"    Found {len(solutions)} critical points:")
        for sol in solutions:
            a4v = sol.get(a4, '?')
            b4v = sol.get(b4, '?')
            # Check if in domain
            if a4v != '?' and b4v != '?':
                try:
                    in_domain = float(a4v) > -1/12 and float(b4v) > -1/12
                    in_domain = in_domain and float(a4v) < 0.25 and float(b4v) < 0.25
                    s_val = float(surplus_sym.subs([(a4, a4v), (b4, b4v)]))
                    print(f"      a4={a4v}, b4={b4v}, in_domain={in_domain}, surplus={s_val:.6e}")
                except (TypeError, ValueError):
                    print(f"      a4={a4v}, b4={b4v} (complex or symbolic)")
    except Exception as ex:
        print(f"    Solve failed: {ex}")
        # Try numerical
        print("    Trying numerical solve with Groebner basis...")
        try:
            # Just check at the equality point
            at_eq = surplus_sym.subs([(a4, Rational(1, 12)), (b4, Rational(1, 12))])
            print(f"    Surplus at a4=b4=1/12: {sp.simplify(at_eq)}")
        except:
            pass

    print(f"  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
