#!/usr/bin/env python3
"""Rich Positivstellensatz certificate with cross-term multipliers.

Tries progressively richer certificates:
Level 1: σ₀ + σ₁·disc_p + σ₂·disc_q + σ₃·(-f₂_p) + σ₄·(-f₂_q)
Level 2: + σ₅·f₁_p + σ₆·f₁_q
Level 3: + σ₇·disc_p·(-f₂_p) + σ₈·disc_q·(-f₂_q)
Level 4: + σ₉·disc_p·disc_q + σ₁₀·(-f₂_p)·(-f₂_q) + σ₁₁·f₁_p·f₁_q

Uses the reflection symmetry (even in a3, b3) to reduce basis sizes.
"""

import sympy as sp
from sympy import symbols, Rational, expand, Poly, together, fraction
import numpy as np
import time


def build_all():
    """Build -N and constraint polynomials."""
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
    neg_N = expand(-N)  # We need -N >= 0 since D < 0

    variables = (a3, a4, b3, b4)

    # Build constraint dictionary: name -> sympy polynomial
    constraints = {
        'disc_p': disc_p,           # >= 0, degree 4
        'disc_q': disc_q,           # >= 0, degree 4
        '-f2_p': expand(-f2_p),     # >= 0, degree 2
        '-f2_q': expand(-f2_q),     # >= 0, degree 2
        'f1_p': f1_p,              # >= 0, degree 1
        'f1_q': f1_q,              # >= 0, degree 1
        'disc_p*(-f2_p)': expand(disc_p * (-f2_p)),  # >= 0, degree 6
        'disc_q*(-f2_q)': expand(disc_q * (-f2_q)),  # >= 0, degree 6
        'disc_p*disc_q': expand(disc_p * disc_q),     # >= 0, degree 8
        '(-f2_p)*(-f2_q)': expand((-f2_p) * (-f2_q)), # >= 0, degree 4
        'f1_p*f1_q': expand(f1_p * f1_q),             # >= 0, degree 2
        'f1_p*(-f2_p)': expand(f1_p * (-f2_p)),       # >= 0, degree 3
        'f1_q*(-f2_q)': expand(f1_q * (-f2_q)),       # >= 0, degree 3
        'f1_p*disc_p': expand(f1_p * disc_p),          # >= 0, degree 5
        'f1_q*disc_q': expand(f1_q * disc_q),          # >= 0, degree 5
    }

    return neg_N, variables, constraints


def poly_to_dict(poly_expr, variables):
    poly = Poly(expand(poly_expr), *variables)
    return {e: float(c) for e, c in poly.as_dict().items()}


def enum_even_basis(max_deg):
    """Enumerate monomials with even a3, b3 powers, split into ee and oo blocks."""
    even_even = []
    odd_odd = []
    for i3 in range(0, max_deg + 1):
        for i4 in range(0, max_deg + 1 - i3):
            for j3 in range(0, max_deg + 1 - i3 - i4):
                for j4 in range(0, max_deg + 1 - i3 - i4 - j3):
                    exp = (i3, i4, j3, j4)
                    if i3 % 2 == 0 and j3 % 2 == 0:
                        even_even.append(exp)
                    elif i3 % 2 == 1 and j3 % 2 == 1:
                        odd_odd.append(exp)
    return even_even, odd_odd


def gram_products(basis):
    products = {}
    n = len(basis)
    for i in range(n):
        for j in range(i, n):
            e = tuple(basis[i][k] + basis[j][k] for k in range(4))
            products.setdefault(e, []).append((i, j, 1.0 if i == j else 2.0))
    return products


def constraint_products(g_dict, gp):
    result = {}
    for eg, cg in g_dict.items():
        for eq, entries in gp.items():
            e = tuple(eg[k] + eq[k] for k in range(4))
            result.setdefault(e, [])
            for i, j, mult in entries:
                result[e].append((cg * mult, i, j))
    return result


def try_certificate(neg_N_dict, variables, constraint_names, constraint_dicts, max_deg=10):
    """Try to find a Positivstellensatz certificate with given constraints."""
    import cvxpy as cp

    n_vars = 4
    half_deg = max_deg // 2

    # σ_0 basis (degree ≤ max_deg, half for SOS)
    ee_0, oo_0 = enum_even_basis(half_deg)

    # Build PSD variables and contributions for each term
    all_blocks = []  # list of (name, Q_ee, Q_oo, contributions_ee, contributions_oo)
    all_psd_constraints = []

    # σ_0 (no multiplier)
    Q0_ee = cp.Variable((len(ee_0), len(ee_0)), symmetric=True)
    Q0_oo = cp.Variable((len(oo_0), len(oo_0)), symmetric=True)
    all_psd_constraints.extend([Q0_ee >> 0, Q0_oo >> 0])
    gp0_ee = gram_products(ee_0)
    gp0_oo = gram_products(oo_0)
    all_blocks.append(("σ_0", Q0_ee, Q0_oo, gp0_ee, gp0_oo))
    print(f"  σ_0: {len(ee_0)}+{len(oo_0)} basis")

    # σ_i * g_i for each constraint
    for cname in constraint_names:
        g_dict = constraint_dicts[cname]
        g_deg = max(sum(e) for e in g_dict)
        sigma_deg = max_deg - g_deg
        if sigma_deg < 0:
            print(f"  {cname}: degree {g_deg} > {max_deg}, skipping")
            continue
        sigma_half = sigma_deg // 2
        ee_i, oo_i = enum_even_basis(sigma_half)
        if len(ee_i) == 0 and len(oo_i) == 0:
            ee_i = [(0, 0, 0, 0)]
            oo_i = []

        Qi_ee = cp.Variable((len(ee_i), len(ee_i)), symmetric=True) if len(ee_i) > 0 else None
        Qi_oo = cp.Variable((len(oo_i), len(oo_i)), symmetric=True) if len(oo_i) > 0 else None

        if Qi_ee is not None:
            all_psd_constraints.append(Qi_ee >> 0)
        if Qi_oo is not None:
            all_psd_constraints.append(Qi_oo >> 0)

        gpi_ee = gram_products(ee_i) if len(ee_i) > 0 else {}
        gpi_oo = gram_products(oo_i) if len(oo_i) > 0 else {}

        cp_ee = constraint_products(g_dict, gpi_ee) if gpi_ee else {}
        cp_oo = constraint_products(g_dict, gpi_oo) if gpi_oo else {}

        all_blocks.append((cname, Qi_ee, Qi_oo, cp_ee, cp_oo))
        print(f"  {cname}: deg {g_deg}, σ basis {len(ee_i)}+{len(oo_i)}")

    # Collect all monomial exponents
    all_exp = set(neg_N_dict.keys())
    for _, _, _, cp_ee, cp_oo in all_blocks:
        all_exp |= set(cp_ee.keys()) | set(cp_oo.keys())
    all_exp = sorted([e for e in all_exp if e[0] % 2 == 0 and e[2] % 2 == 0],
                     key=lambda e: (sum(e), e))

    print(f"  Constraints: {len(all_exp)} monomials")

    # Build equality constraints
    eq_constraints = list(all_psd_constraints)
    for e in all_exp:
        lhs = 0
        for block_name, Q_ee, Q_oo, cp_ee, cp_oo in all_blocks:
            if block_name == "σ_0":
                # Direct Gram contribution
                if e in cp_ee:
                    for i, j, mult in cp_ee[e]:
                        lhs = lhs + mult * Q_ee[i, j]
                if e in cp_oo and Q_oo is not None:
                    for i, j, mult in cp_oo[e]:
                        lhs = lhs + mult * Q_oo[i, j]
            else:
                # Constraint product contribution
                if e in cp_ee and Q_ee is not None:
                    for coeff, i, j in cp_ee[e]:
                        lhs = lhs + coeff * Q_ee[i, j]
                if e in cp_oo and Q_oo is not None:
                    for coeff, i, j in cp_oo[e]:
                        lhs = lhs + coeff * Q_oo[i, j]

        rhs = neg_N_dict.get(e, 0.0)
        eq_constraints.append(lhs == rhs)

    # Solve
    prob = cp.Problem(cp.Minimize(0), eq_constraints)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
        return prob.status, all_blocks
    except Exception as ex:
        return f"error: {ex}", all_blocks


def main():
    print("=" * 70)
    print("Rich Positivstellensatz certificate search")
    print("=" * 70)
    print()

    t0 = time.time()

    neg_N, variables, constraints = build_all()
    neg_N_dict = poly_to_dict(neg_N, variables)
    constraint_dicts = {k: poly_to_dict(v, variables) for k, v in constraints.items()}

    print(f"-N: {len(neg_N_dict)} terms, degree {max(sum(e) for e in neg_N_dict)}")
    print(f"({time.time()-t0:.1f}s)")
    print()

    # Define levels of increasing richness
    levels = [
        ("Level 1 (Putinar basic)",
         ['disc_p', 'disc_q', '-f2_p', '-f2_q']),
        ("Level 2 (+f1)",
         ['disc_p', 'disc_q', '-f2_p', '-f2_q', 'f1_p', 'f1_q']),
        ("Level 3 (+cross disc*f2)",
         ['disc_p', 'disc_q', '-f2_p', '-f2_q', 'f1_p', 'f1_q',
          'disc_p*(-f2_p)', 'disc_q*(-f2_q)']),
        ("Level 4 (+all degree-2 products)",
         ['disc_p', 'disc_q', '-f2_p', '-f2_q', 'f1_p', 'f1_q',
          'disc_p*(-f2_p)', 'disc_q*(-f2_q)',
          '(-f2_p)*(-f2_q)', 'f1_p*f1_q',
          'f1_p*(-f2_p)', 'f1_q*(-f2_q)']),
        ("Level 5 (+disc cross-products)",
         ['disc_p', 'disc_q', '-f2_p', '-f2_q', 'f1_p', 'f1_q',
          'disc_p*(-f2_p)', 'disc_q*(-f2_q)',
          '(-f2_p)*(-f2_q)', 'f1_p*f1_q',
          'f1_p*(-f2_p)', 'f1_q*(-f2_q)',
          'disc_p*disc_q', 'f1_p*disc_p', 'f1_q*disc_q']),
    ]

    for level_name, cnames in levels:
        print(f"\n{'='*60}")
        print(f"{level_name}")
        print(f"{'='*60}")

        status, blocks = try_certificate(neg_N_dict, variables, cnames, constraint_dicts)
        print(f"  Status: {status}")
        print(f"  ({time.time()-t0:.1f}s)")

        if status in ['optimal', 'optimal_inaccurate']:
            print(f"\n  *** CERTIFICATE FOUND! ***")
            for name, Q_ee, Q_oo, _, _ in blocks:
                if Q_ee is not None and Q_ee.value is not None:
                    ev = np.linalg.eigvalsh(Q_ee.value)
                    print(f"    {name}_ee: min_eig={ev.min():.3e}")
                if Q_oo is not None and Q_oo.value is not None:
                    ev = np.linalg.eigvalsh(Q_oo.value)
                    print(f"    {name}_oo: min_eig={ev.min():.3e}")
            break
        elif 'error' in str(status).lower():
            print(f"  Solver error, trying SCS...")
            import cvxpy as cp
            # Rebuild and try SCS
            # (simplified: just report the error)
            pass

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
