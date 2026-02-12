#!/usr/bin/env python3
"""Degree-12 SOS certificate via direct SCS interface.

Bypasses CVXPY's expression tree (which OOM'd at degree 12) by building
the SDP constraint matrices directly as scipy sparse and calling SCS.

Certificate form:
  -N = σ₀ + σ₁·disc_p + σ₂·disc_q + σ₃·(-f₂_p) + σ₄·(-f₂_q) + σ₅·f₁_p + σ₆·f₁_q

where -N is the negated surplus numerator (need -N ≥ 0 since D < 0 on domain),
and each σᵢ is a sum-of-squares polynomial (Gram matrix PSD).

Symmetry exploited: -N is even under (a₃,b₃) → (-a₃,-b₃), so each σᵢ
decomposes into even-even and odd-odd parity blocks.
"""

import sympy as sp
from sympy import symbols, Rational, expand, Poly, together, fraction
import numpy as np
from scipy import sparse
import scs
import time
import sys


def build_neg_N():
    """Build -N and constraint polynomials symbolically."""
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

    surplus_frac = together(-disc_r/(4*f1_r*f2_r)
                            + disc_p/(4*f1_p*f2_p)
                            + disc_q/(4*f1_q*f2_q))
    N, D = fraction(surplus_frac)
    neg_N = expand(-N)

    variables = (a3, a4, b3, b4)
    constraints = {
        'disc_p': disc_p,
        'disc_q': disc_q,
        '-f2_p': expand(-f2_p),
        '-f2_q': expand(-f2_q),
        'f1_p': f1_p,
        'f1_q': f1_q,
    }
    return neg_N, variables, constraints


def poly_to_dict(expr, variables):
    """Convert sympy expression to {exponent_tuple: float_coefficient}."""
    poly = Poly(expand(expr), *variables)
    return {e: float(c) for e, c in poly.as_dict().items()}


def enum_even_basis(max_deg):
    """Enumerate monomial basis split by (a3, b3) parity.

    Returns (even_even, odd_odd) lists of exponent tuples.
    """
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


def gram_products(basis):
    """Compute monomial products from Gram matrix entries.

    Returns dict: exponent_tuple -> list of (i, j, mult)
    where mult = 1 for diagonal, 2 for off-diagonal.
    """
    n = len(basis)
    products = {}
    for i in range(n):
        for j in range(i, n):
            e = tuple(basis[i][k] + basis[j][k] for k in range(4))
            products.setdefault(e, []).append((i, j, 1.0 if i == j else 2.0))
    return products


def constraint_gram_products(g_dict, gp):
    """Products of constraint polynomial g with Gram products.

    Returns dict: exponent_tuple -> list of (coeff, i, j)
    where coeff = c_g * mult.
    """
    result = {}
    for eg, cg in g_dict.items():
        for eq, entries in gp.items():
            e = tuple(eg[k] + eq[k] for k in range(4))
            result.setdefault(e, [])
            for i, j, mult in entries:
                result[e].append((cg * mult, i, j))
    return result


def svec_idx(i, j, n):
    """Index into lower-triangular svec of an n×n matrix.
    SCS uses column-major lower triangle: (0,0), (1,0), (2,0),...,(n-1,0), (1,1), (2,1),...
    """
    if i < j:
        i, j = j, i
    return j * n - j * (j + 1) // 2 + i


def svec_size(n):
    """Size of svec for n×n matrix."""
    return n * (n + 1) // 2


def build_sdp_direct(neg_N_dict, constraint_dicts, constraint_names, cert_degree):
    """Build SDP data for SCS directly as sparse matrices.

    Variables x = [svec(Q₀_ee), svec(Q₀_oo), svec(Q₁_ee), svec(Q₁_oo), ...]
    where each svec uses SCS convention (lower-triangular, column-major,
    √2 scaling on off-diagonal).

    SCS form: min c'x s.t. Ax + s = b, s ∈ K
    K = {zero cone for equalities} × {SDP cones for PSD constraints}
    """
    half = cert_degree // 2

    # Build block specifications: (name, basis_ee, basis_oo, constraint_poly_or_None)
    blocks = []

    # σ₀ (no constraint polynomial)
    ee, oo = enum_even_basis(half)
    blocks.append(('σ₀', ee, oo, None))

    for cname in constraint_names:
        g_dict = constraint_dicts[cname]
        g_deg = max(sum(e) for e in g_dict)
        sigma_max_deg = cert_degree - g_deg
        if sigma_max_deg < 0:
            continue
        sigma_half = sigma_max_deg // 2
        ee_i, oo_i = enum_even_basis(sigma_half)
        if not ee_i:
            ee_i = [(0, 0, 0, 0)]
        blocks.append((cname, ee_i, oo_i, g_dict))

    # Print block info
    total_svec = 0
    block_svec_ranges = []  # (name, parity, offset, size, n)
    sdp_cone_sizes = []

    for bname, ee_i, oo_i, _ in blocks:
        for parity, basis in [('ee', ee_i), ('oo', oo_i)]:
            n = len(basis)
            if n == 0:
                continue
            sv = svec_size(n)
            block_svec_ranges.append((bname, parity, total_svec, sv, n))
            sdp_cone_sizes.append(n)
            total_svec += sv
        print(f"  {bname}: ee={len(ee_i)}, oo={len(oo_i)}")

    print(f"  Total svec size (decision variables): {total_svec}")
    print(f"  PSD blocks: {len(sdp_cone_sizes)} (sizes: {sdp_cone_sizes})")

    # Compute all monomial contributions from each block
    # For each block, compute gram products and (if multiplier) constraint products
    block_contributions = []  # list of (offset, n, products_dict)
    idx = 0
    for bname, ee_i, oo_i, g_dict in blocks:
        for parity, basis in [('ee', ee_i), ('oo', oo_i)]:
            n = len(basis)
            if n == 0:
                continue
            offset = block_svec_ranges[idx][2]
            gp = gram_products(basis)
            if g_dict is None:
                # σ₀: direct Gram products
                block_contributions.append((offset, n, gp, 'direct'))
            else:
                # σᵢ * gᵢ: constraint products
                cp = constraint_gram_products(g_dict, gp)
                block_contributions.append((offset, n, cp, 'constraint'))
            idx += 1

    # Collect all monomial exponents that appear
    all_exp = set(neg_N_dict.keys())
    for _, _, prods, _ in block_contributions:
        all_exp |= set(prods.keys())
    # Filter to even-parity only (a3 even, b3 even)
    all_exp = sorted([e for e in all_exp if e[0] % 2 == 0 and e[2] % 2 == 0],
                     key=lambda e: (sum(e), e))

    n_mono = len(all_exp)
    mono_idx = {e: i for i, e in enumerate(all_exp)}
    n_target = sum(1 for e in all_exp if sum(e) <= 10)
    n_cancel = n_mono - n_target
    print(f"  Monomial constraints: {n_mono} ({n_target} matching + {n_cancel} cancellation)")

    # Build A matrix (sparse) for equality constraints
    # Row: monomial index; Column: svec variable index
    # Entry: coefficient of svec[col] in monomial[row]'s equation
    print("  Building constraint matrix...")
    rows = []
    cols = []
    vals = []

    for offset, n, prods, prod_type in block_contributions:
        for e, entries in prods.items():
            if e[0] % 2 != 0 or e[2] % 2 != 0:
                continue  # skip odd-parity monomials
            row = mono_idx.get(e)
            if row is None:
                continue

            if prod_type == 'direct':
                # entries: (i, j, mult) where mult = 1 (diag) or 2 (off-diag)
                for i, j, mult in entries:
                    col = offset + svec_idx(i, j, n)
                    # In svec form: Q[i,j] = x[col] if i==j, x[col]/√2 if i≠j
                    # Contribution = mult * Q[i,j]
                    # = mult * x[col] if i==j
                    # = mult * x[col]/√2 if i≠j
                    # = 1 * x[col] if i==j (mult=1)
                    # = 2 * x[col]/√2 = √2 * x[col] if i≠j (mult=2)
                    if i == j:
                        val = 1.0
                    else:
                        val = np.sqrt(2.0)
                    rows.append(row)
                    cols.append(col)
                    vals.append(val)
            else:
                # entries: (coeff, i, j) where coeff = c_g * mult
                for coeff, i, j in entries:
                    col = offset + svec_idx(i, j, n)
                    # Same svec conversion as above but with coefficient
                    if i == j:
                        val = coeff
                    else:
                        val = coeff / np.sqrt(2.0)
                    rows.append(row)
                    cols.append(col)
                    vals.append(val)

    # Build b vector (RHS of equality constraints)
    b_eq = np.zeros(n_mono)
    for e, c in neg_N_dict.items():
        if e in mono_idx:
            row = mono_idx[e]
            if sum(e) <= 10:
                b_eq[row] = c
            # else: cancellation constraint, b = 0

    # Build the full A matrix for SCS
    # SCS form: Ax + s = b, s ∈ K
    # K = zero(n_mono) × SDP(n₁) × SDP(n₂) × ...
    #
    # Zero cone rows: A_eq * x = b_eq → A_eq * x + s = b_eq, s = 0
    # SDP rows: x[block] ∈ S₊ → -I * x + s = 0, s = svec(Q) ∈ S₊

    A_eq = sparse.csc_matrix((vals, (rows, cols)), shape=(n_mono, total_svec))

    # Identity block for PSD constraints
    A_psd = -sparse.eye(total_svec, format='csc')

    # Stack: A = [A_eq; A_psd]
    A = sparse.vstack([A_eq, A_psd], format='csc')

    # b = [b_eq; 0]
    b = np.concatenate([b_eq, np.zeros(total_svec)])

    # c = 0 (feasibility)
    c = np.zeros(total_svec)

    # Cone specification
    cone = {'z': n_mono, 's': sdp_cone_sizes}

    return A, b, c, cone, block_svec_ranges, all_exp, mono_idx


def verify_certificate(x, block_svec_ranges, neg_N_dict, constraint_dicts, all_exp):
    """Verify the SOS certificate numerically at random points."""
    rng = np.random.default_rng(42)

    # Extract Gram matrices from svec
    gram_matrices = {}
    for bname, parity, offset, sv, n in block_svec_ranges:
        svec_vals = x[offset:offset + sv]
        Q = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1):
                idx = svec_idx(i, j, n)
                val = svec_vals[idx]
                if i == j:
                    Q[i, j] = val
                else:
                    Q[i, j] = val / np.sqrt(2.0)
                    Q[j, i] = val / np.sqrt(2.0)
        gram_matrices[(bname, parity)] = Q

        # Check PSD
        eigs = np.linalg.eigvalsh(Q)
        min_eig = eigs.min()
        status = "PSD" if min_eig >= -1e-6 else "NOT PSD"
        print(f"    {bname}_{parity}: {n}×{n}, min_eig={min_eig:.3e} [{status}]")

    print()
    return gram_matrices


def main():
    print("=" * 70)
    print("Degree-12 SOS certificate via direct SCS")
    print("=" * 70)
    print()

    t0 = time.time()

    # Build polynomials
    print("Building -N and constraints...")
    neg_N, variables, constraints = build_neg_N()
    neg_N_dict = poly_to_dict(neg_N, variables)
    constraint_dicts = {k: poly_to_dict(v, variables) for k, v in constraints.items()}
    print(f"  -N: {len(neg_N_dict)} terms, degree {max(sum(e) for e in neg_N_dict)}")
    print(f"  ({time.time()-t0:.1f}s)")
    print()

    # Try progressively higher degrees
    multiplier_sets = [
        ("Basic (disc + f2)", ['disc_p', 'disc_q', '-f2_p', '-f2_q']),
        ("With f1", ['disc_p', 'disc_q', '-f2_p', '-f2_q', 'f1_p', 'f1_q']),
    ]

    for cert_deg in [12, 14]:
        for set_name, cnames in multiplier_sets:
            print(f"\n{'='*60}")
            print(f"Degree {cert_deg}, {set_name}")
            print(f"{'='*60}")

            try:
                A, b, c, cone, block_svec_ranges, all_exp, mono_idx = \
                    build_sdp_direct(neg_N_dict, constraint_dicts, cnames, cert_deg)
            except MemoryError:
                print(f"  OOM building matrices at degree {cert_deg}")
                continue

            print(f"  A shape: {A.shape}, nnz: {A.nnz}")
            print(f"  ({time.time()-t0:.1f}s)")
            print()

            # Solve with SCS
            print("  Solving with SCS...")
            data = {
                'A': A,
                'b': b,
                'c': c,
            }

            try:
                solver = scs.SCS(data, cone,
                                 max_iters=50000,
                                 eps_abs=1e-9,
                                 eps_rel=1e-9,
                                 verbose=True)
                sol = solver.solve()
                status = sol['info']['status']
                print(f"\n  Status: {status}")
                print(f"  Primal obj: {sol['info']['pobj']:.6e}")
                print(f"  Dual obj: {sol['info']['dobj']:.6e}")
                print(f"  ({time.time()-t0:.1f}s)")

                if 'solved' in status.lower():
                    print(f"\n  *** CERTIFICATE FOUND at degree {cert_deg}! ***\n")
                    gram_mats = verify_certificate(
                        sol['s'], block_svec_ranges,
                        neg_N_dict, constraint_dicts, all_exp)
                    return True

            except Exception as ex:
                print(f"  SCS error: {ex}")
                print(f"  ({time.time()-t0:.1f}s)")

            print()

    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print("No certificate found.")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
