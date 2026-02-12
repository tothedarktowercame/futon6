#!/usr/bin/env python3
"""Higher-degree SOS certificate (degree 12 relaxation).

The degree-10 certificate fails because -N has an interior zero (at the
equality point). Higher-degree relaxations can sometimes succeed by
allowing cancellations at degrees 11-12.

Certificate: -N = σ₀ + σ₁·disc_p + σ₂·disc_q + σ₃·(-f₂_p) + σ₄·(-f₂_q) + σ₅·f₁_p + σ₆·f₁_q
where all terms have degree ≤ cert_degree, and coefficients at degrees > 10 must cancel.
"""

import sympy as sp
from sympy import symbols, Rational, expand, Poly, together, fraction
import numpy as np
import time


def build_neg_N():
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

    surplus_frac = together(-disc_r/(4*f1_r*f2_r) + disc_p/(4*f1_p*f2_p) + disc_q/(4*f1_q*f2_q))
    N, D = fraction(surplus_frac)
    neg_N = expand(-N)

    vars_ = (a3, a4, b3, b4)
    constraints = {
        'disc_p': disc_p,
        'disc_q': disc_q,
        '-f2_p': expand(-f2_p),
        '-f2_q': expand(-f2_q),
        'f1_p': f1_p,
        'f1_q': f1_q,
    }
    return neg_N, vars_, constraints


def poly_to_dict(expr, variables):
    poly = Poly(expand(expr), *variables)
    return {e: float(c) for e, c in poly.as_dict().items()}


def enum_even_basis(max_deg):
    even_even = []
    odd_odd = []
    for i3 in range(max_deg + 1):
        for i4 in range(max_deg + 1 - i3):
            for j3 in range(max_deg + 1 - i3 - i4):
                for j4 in range(max_deg + 1 - i3 - i4 - j3):
                    exp = (i3, i4, j3, j4)
                    if i3 % 2 == 0 and j3 % 2 == 0:
                        even_even.append(exp)
                    elif i3 % 2 == 1 and j3 % 2 == 1:
                        odd_odd.append(exp)
    return even_even, odd_odd


def try_sos(neg_N_dict, constraint_dicts, constraint_names, cert_degree, target_degree=10):
    """Try SOS certificate at given degree.

    cert_degree: max degree of σ_0 and σ_i * g_i
    target_degree: degree of -N (coefficients above this must cancel)
    """
    import cvxpy as cp

    half = cert_degree // 2

    # σ_0 basis (degree ≤ half)
    ee_0, oo_0 = enum_even_basis(half)

    # Build blocks
    blocks = []  # (name, Q_ee, Q_oo, gram_prods_ee, gram_prods_oo)
    psd_constraints = []

    def gram_prods(basis):
        p = {}
        for i in range(len(basis)):
            for j in range(i, len(basis)):
                e = tuple(basis[i][k] + basis[j][k] for k in range(4))
                p.setdefault(e, []).append((i, j, 1.0 if i == j else 2.0))
        return p

    def mult_gram_prods(g_dict, gp):
        result = {}
        for eg, cg in g_dict.items():
            for eq, entries in gp.items():
                e = tuple(eg[k] + eq[k] for k in range(4))
                result.setdefault(e, [])
                for i, j, mult in entries:
                    result[e].append((cg * mult, i, j))
        return result

    # σ_0
    Q0_ee = cp.Variable((len(ee_0), len(ee_0)), symmetric=True)
    Q0_oo = cp.Variable((len(oo_0), len(oo_0)), symmetric=True) if len(oo_0) > 0 else None
    psd_constraints.append(Q0_ee >> 0)
    if Q0_oo is not None:
        psd_constraints.append(Q0_oo >> 0)
    gp0_ee = gram_prods(ee_0)
    gp0_oo = gram_prods(oo_0) if len(oo_0) > 0 else {}
    blocks.append(("σ_0", Q0_ee, Q0_oo, gp0_ee, gp0_oo))
    print(f"  σ_0: {len(ee_0)}+{len(oo_0)} basis")

    # Multiplier blocks
    for cname in constraint_names:
        g_dict = constraint_dicts[cname]
        g_deg = max(sum(e) for e in g_dict)
        sigma_max = cert_degree - g_deg
        if sigma_max < 0:
            continue
        sigma_half = sigma_max // 2
        ee_i, oo_i = enum_even_basis(sigma_half)
        if not ee_i:
            ee_i = [(0, 0, 0, 0)]

        Qi_ee = cp.Variable((len(ee_i), len(ee_i)), symmetric=True)
        Qi_oo = cp.Variable((len(oo_i), len(oo_i)), symmetric=True) if len(oo_i) > 0 else None
        psd_constraints.append(Qi_ee >> 0)
        if Qi_oo is not None:
            psd_constraints.append(Qi_oo >> 0)

        gpi_ee = gram_prods(ee_i)
        gpi_oo = gram_prods(oo_i) if len(oo_i) > 0 else {}
        cp_ee = mult_gram_prods(g_dict, gpi_ee)
        cp_oo = mult_gram_prods(g_dict, gpi_oo) if gpi_oo else {}

        blocks.append((cname, Qi_ee, Qi_oo, cp_ee, cp_oo))
        print(f"  {cname}: deg {g_deg}, σ basis {len(ee_i)}+{len(oo_i)}")

    # Collect all even-exponent monomials up to cert_degree
    all_exp = set(neg_N_dict.keys())
    for _, _, _, cp_ee, cp_oo in blocks:
        all_exp |= set(cp_ee.keys()) | set(cp_oo.keys())
    all_exp = sorted([e for e in all_exp if e[0] % 2 == 0 and e[2] % 2 == 0],
                     key=lambda e: (sum(e), e))

    n_target = sum(1 for e in all_exp if sum(e) <= target_degree)
    n_cancel = sum(1 for e in all_exp if sum(e) > target_degree)
    print(f"  Constraints: {len(all_exp)} total ({n_target} matching + {n_cancel} cancellation)")

    # Build equality constraints
    eq_constraints = list(psd_constraints)
    for e in all_exp:
        lhs = 0
        for block_name, Q_ee, Q_oo, gp_ee, gp_oo in blocks:
            if block_name == "σ_0":
                if e in gp_ee:
                    for i, j, mult in gp_ee[e]:
                        lhs = lhs + mult * Q_ee[i, j]
                if e in gp_oo and Q_oo is not None:
                    for i, j, mult in gp_oo[e]:
                        lhs = lhs + mult * Q_oo[i, j]
            else:
                if e in gp_ee:
                    for coeff, i, j in gp_ee[e]:
                        lhs = lhs + coeff * Q_ee[i, j]
                if e in gp_oo and Q_oo is not None:
                    for coeff, i, j in gp_oo[e]:
                        lhs = lhs + coeff * Q_oo[i, j]

        rhs = neg_N_dict.get(e, 0.0) if sum(e) <= target_degree else 0.0
        eq_constraints.append(lhs == rhs)

    prob = cp.Problem(cp.Minimize(0), eq_constraints)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
        status = prob.status
    except Exception as ex:
        status = f"error: {ex}"

    return status, blocks


def main():
    print("=" * 70)
    print("Higher-degree SOS certificate search")
    print("=" * 70)
    print()

    t0 = time.time()

    neg_N, variables, constraints = build_neg_N()
    neg_N_dict = poly_to_dict(neg_N, variables)
    constraint_dicts = {k: poly_to_dict(v, variables) for k, v in constraints.items()}
    print(f"-N: {len(neg_N_dict)} terms, degree 10")
    print(f"({time.time()-t0:.1f}s)")
    print()

    multiplier_sets = [
        ("Basic (disc + f2)", ['disc_p', 'disc_q', '-f2_p', '-f2_q']),
        ("With f1", ['disc_p', 'disc_q', '-f2_p', '-f2_q', 'f1_p', 'f1_q']),
    ]

    for cert_deg in [12, 14]:
        for set_name, cnames in multiplier_sets:
            print(f"\n{'='*60}")
            print(f"Degree {cert_deg}, {set_name}")
            print(f"{'='*60}")

            status, blocks = try_sos(neg_N_dict, constraint_dicts, cnames,
                                     cert_degree=cert_deg, target_degree=10)
            print(f"  Status: {status}")
            print(f"  ({time.time()-t0:.1f}s)")

            if status in ['optimal', 'optimal_inaccurate']:
                print(f"\n  *** CERTIFICATE FOUND at degree {cert_deg}! ***")
                for name, Q_ee, Q_oo, _, _ in blocks:
                    if Q_ee is not None and Q_ee.value is not None:
                        ev = np.linalg.eigvalsh(Q_ee.value)
                        print(f"    {name}_ee: {Q_ee.shape[0]}×{Q_ee.shape[0]}, min_eig={ev.min():.3e}")
                    if Q_oo is not None and Q_oo.value is not None:
                        ev = np.linalg.eigvalsh(Q_oo.value)
                        print(f"    {name}_oo: {Q_oo.shape[0]}×{Q_oo.shape[0]}, min_eig={ev.min():.3e}")
                return

    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print("\nNo certificate found at available degrees.")
    print("Options: try degree 16+, use MOSEK solver, or different proof strategy.")


if __name__ == "__main__":
    main()
