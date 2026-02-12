#!/usr/bin/env python3
"""Algebraic verification and computation for the n=4 finite Stam inequality.

Stage 1: Symbolically verify Phi_4 * disc = -4*(a2^2+12*a4)*(2*a2^3-8*a2*a4+9*a3^2)
Stage 2: Compute the general surplus numerator in 4 variables (a3, a4, b3, b4)
         after centering (a1=b1=0) and scaling (a2=b2=-1)
Stage 3: Analyze the numerator structure for Positivstellensatz feasibility
"""

import sympy as sp
from sympy import symbols, poly, Rational, factor, expand, simplify, resultant
from sympy import Matrix, sqrt, collect, Poly, degree
import numpy as np
from itertools import combinations
from math import factorial
import time


def stage1_verify_identity():
    """Symbolically verify Phi_4 * disc = -4*(a2^2+12*a4)*(2*a2^3-8*a2*a4+9*a3^2)."""
    print("=" * 70)
    print("STAGE 1: Symbolic verification of Phi_4 * disc identity")
    print("=" * 70)
    print()

    x = sp.Symbol('x')
    a2, a3, a4 = sp.symbols('a2 a3 a4')

    # The centered monic quartic
    p = x**4 + a2*x**2 + a3*x + a4
    print("p(x) =", p)

    # Discriminant of a depressed quartic x^4 + px^2 + qx + r:
    # disc = 256*r^3 - 128*p^2*r^2 + 144*p*q^2*r - 27*q^4 + 16*p^4*r - 4*p^3*q^2
    disc = (256*a4**3 - 128*a2**2*a4**2 + 144*a2*a3**2*a4
            - 27*a3**4 + 16*a2**4*a4 - 4*a2**3*a3**2)
    print("disc =", disc)

    # The claimed identity
    f1 = a2**2 + 12*a4
    f2 = 2*a2**3 - 8*a2*a4 + 9*a3**2
    claimed = -4 * f1 * f2
    print("Claimed Phi_4 * disc =", sp.expand(claimed))
    print()

    # To verify: we need Phi_4 symbolically.
    # Phi_4 = sum_i S_i^2 where S_i = sum_{j!=i} 1/(lam_i - lam_j)
    # Equivalently: Phi_4 = (1/4) * sum_i (p''(lam_i)/p'(lam_i))^2
    #
    # Key identity: Phi_4 = 2 * sum_{i<j} 1/(lam_i - lam_j)^2
    #
    # We can compute this via: Phi_4 = (sum_i S_i)^2 - 2*sum_{i<j} S_i*S_j
    # But more directly: use the formula involving resultants/power sums.
    #
    # The approach: Phi_n * disc = resultant-based expression.
    # Phi_n = 2 * sum_{i<j} 1/(lam_i - lam_j)^2
    # disc = prod_{i<j} (lam_i - lam_j)^2
    #
    # So Phi_n * disc = 2 * sum_{i<j} prod_{(k,l)!=(i,j)} (lam_k - lam_l)^2
    #
    # This is a symmetric polynomial in the roots, hence expressible in terms
    # of the coefficients. We compute it via the resultant approach.

    # Alternative approach: compute via p'(x) and resultant.
    # S_i = p''(lam_i) / (2 * p'(lam_i))
    # Phi_4 = sum_i S_i^2 = (1/4) * sum_i [p''(lam_i) / p'(lam_i)]^2
    #
    # sum_i f(lam_i) for roots lam_i of p(x) can be computed as:
    # sum_i f(lam_i) = -Res(p(x), f(x) * p'(x) / p(x)) ... no, that's circular.
    #
    # Better: use the fact that for polynomial g(x),
    # sum_i g(lam_i) / p'(lam_i) = coefficient extraction from g(x)/p(x)
    # (partial fraction decomposition).
    #
    # Specifically: sum_i g(lam_i)/p'(lam_i) = [g(x)/p(x)]_{polynomial part removed}
    # evaluated as the sum of residues.
    #
    # For our case: sum_i [p''(lam_i)]^2 / [p'(lam_i)]^2
    # This is not directly a "sum of g(lam_i)/p'(lam_i)" form.
    # We need: sum_i [p''(lam_i)/p'(lam_i)]^2 = sum_i p''(lam_i)^2 / p'(lam_i)^2
    #
    # Let's use a different approach: work with 4 explicit root variables.

    print("Computing via explicit roots (symmetric polynomial reduction)...")
    print("(This may take a minute with SymPy)")
    t0 = time.time()

    l1, l2, l3, l4 = sp.symbols('l1 l2 l3 l4')
    roots = [l1, l2, l3, l4]

    # Phi_4 = 2 * sum_{i<j} 1/(li - lj)^2
    phi4_roots = 2 * sum(1/(roots[i] - roots[j])**2
                         for i in range(4) for j in range(i+1, 4))

    # disc = prod_{i<j} (li - lj)^2
    disc_roots = sp.prod([(roots[i] - roots[j])**2
                          for i in range(4) for j in range(i+1, 4)])

    # Product
    product_roots = sp.expand(phi4_roots * disc_roots)
    print(f"  Phi_4 * disc expanded in root variables ({time.time()-t0:.1f}s)")

    # Now express in terms of elementary symmetric polynomials
    # e1 = l1+l2+l3+l4 = 0 (centered)
    # e2 = sum_{i<j} li*lj = a2
    # e3 = sum_{i<j<k} li*lj*lk = -a3 (sign convention: p = x^4 + 0*x^3 + a2*x^2 + a3*x + a4)
    # e4 = l1*l2*l3*l4 = a4
    #
    # Actually for p(x) = (x-l1)(x-l2)(x-l3)(x-l4) = x^4 - e1*x^3 + e2*x^2 - e3*x + e4
    # So: a2 = e2, a3 = -e3, a4 = e4, and e1 = 0 (centered).

    e1 = l1 + l2 + l3 + l4
    e2 = sp.Add(*[roots[i]*roots[j] for i in range(4) for j in range(i+1, 4)])
    e3 = sp.Add(*[roots[i]*roots[j]*roots[k]
                   for i in range(4) for j in range(i+1, 4) for k in range(j+1, 4)])
    e4 = l1*l2*l3*l4

    # Substitute e1 = 0 by setting l4 = -l1-l2-l3
    print("  Substituting l4 = -l1-l2-l3 (centering)...")
    product_centered = product_roots.subs(l4, -l1-l2-l3)
    product_centered = sp.expand(product_centered)
    print(f"  Centered product expanded ({time.time()-t0:.1f}s)")

    # Now express e2, e3, e4 in centered form
    e2_centered = e2.subs(l4, -l1-l2-l3)
    e3_centered = e3.subs(l4, -l1-l2-l3)
    e4_centered = e4.subs(l4, -l1-l2-l3)

    e2_c = sp.expand(e2_centered)
    e3_c = sp.expand(e3_centered)
    e4_c = sp.expand(e4_centered)

    print(f"  e2 = {e2_c}")
    print(f"  e3 = {e3_c}")
    print(f"  e4 = {e4_c}")

    # The claimed formula in terms of e2, e3, e4 (where a2=e2, a3=-e3, a4=e4):
    claimed_roots = -4 * (e2_c**2 + 12*e4_c) * (2*e2_c**3 - 8*e2_c*e4_c + 9*e3_c**2)
    claimed_roots = sp.expand(claimed_roots)
    print(f"  Claimed formula expanded ({time.time()-t0:.1f}s)")

    # Check difference
    print("  Computing difference...")
    diff = sp.expand(product_centered - claimed_roots)
    print(f"  Difference computed ({time.time()-t0:.1f}s)")

    # Collect and simplify
    if diff == 0:
        print()
        print("  *** IDENTITY VERIFIED SYMBOLICALLY ***")
        print("  Phi_4(p) * disc(p) = -4*(a2^2+12*a4)*(2*a2^3-8*a2*a4+9*a3^2)")
        print()
        return True
    else:
        # Try harder simplification
        print(f"  Direct comparison: diff has {len(diff.as_ordered_terms())} terms")
        print("  Attempting simplification...")
        diff_simplified = sp.simplify(diff)
        if diff_simplified == 0:
            print()
            print("  *** IDENTITY VERIFIED SYMBOLICALLY (after simplification) ***")
            return True
        else:
            print(f"  Simplified diff = {diff_simplified}")
            print("  IDENTITY NOT YET VERIFIED — may need manual reduction")
            return False


def stage2_general_surplus():
    """Compute the general surplus numerator for n=4 with a2=b2=-1."""
    print()
    print("=" * 70)
    print("STAGE 2: General surplus numerator (a2=b2=-1)")
    print("=" * 70)
    print()

    a3, a4, b3, b4 = sp.symbols('a3 a4 b3 b4')

    # With a2 = b2 = -1:
    # 1/Phi_4 = disc / (4 * f1 * |f2|) = -disc / (4 * f1 * f2)
    # since f2 < 0 on real-rooted cone, |f2| = -f2, so disc/(4*f1*|f2|) = -disc/(4*f1*f2)

    # For p: a2 = -1
    disc_p = 256*a4**3 - 128*a4**2 + 144*(-1)*a3**2*a4 - 27*a3**4 + 16*a4 - 4*(-1)*a3**2
    disc_p = sp.expand(disc_p)
    f1_p = 1 + 12*a4       # (-1)^2 + 12*a4
    f2_p = -2 - 8*(-1)*a4 + 9*a3**2  # 2*(-1)^3 - 8*(-1)*a4 + 9*a3^2
    f2_p = -2 + 8*a4 + 9*a3**2

    # 1/Phi_4(p) = -disc_p / (4 * f1_p * f2_p)
    inv_phi_p = -disc_p / (4 * f1_p * f2_p)

    print("1/Phi_4(p) = -disc_p / (4 * f1_p * f2_p)")
    print(f"  disc_p = {disc_p}")
    print(f"  f1_p = {f1_p}")
    print(f"  f2_p = {f2_p}")
    print()

    # For q: b2 = -1
    disc_q = 256*b4**3 - 128*b4**2 + 144*(-1)*b3**2*b4 - 27*b3**4 + 16*b4 - 4*(-1)*b3**2
    disc_q = sp.expand(disc_q)
    f1_q = 1 + 12*b4
    f2_q = -2 + 8*b4 + 9*b3**2

    inv_phi_q = -disc_q / (4 * f1_q * f2_q)

    # For convolution: c2 = -2, c3 = a3+b3, c4 = a4 + 1/6 + b4
    c2 = -2
    c3 = a3 + b3
    c4 = a4 + Rational(1, 6) + b4

    disc_r = (256*c4**3 - 128*c2**2*c4**2 + 144*c2*c3**2*c4
              - 27*c3**4 + 16*c2**4*c4 - 4*c2**3*c3**2)
    disc_r = sp.expand(disc_r)
    f1_r = c2**2 + 12*c4
    f1_r = sp.expand(f1_r)
    f2_r = 2*c2**3 - 8*c2*c4 + 9*c3**2
    f2_r = sp.expand(f2_r)

    inv_phi_r = -disc_r / (4 * f1_r * f2_r)

    print("Convolution: c2=-2, c3=a3+b3, c4=a4+1/6+b4")
    print(f"  disc_r = {disc_r}")
    print(f"  f1_r = {f1_r}")
    print(f"  f2_r = {f2_r}")
    print()

    # Surplus Delta = inv_phi_r - inv_phi_p - inv_phi_q
    # = -disc_r/(4*f1_r*f2_r) + disc_p/(4*f1_p*f2_p) + disc_q/(4*f1_q*f2_q)
    #
    # Common denominator: 4 * f1_p * f2_p * f1_q * f2_q * f1_r * f2_r
    #
    # Numerator = -disc_r * f1_p * f2_p * f1_q * f2_q
    #           + disc_p * f1_q * f2_q * f1_r * f2_r
    #           + disc_q * f1_p * f2_p * f1_r * f2_r
    #
    # Wait: surplus = inv_phi_r - inv_phi_p - inv_phi_q
    # = -disc_r/(4*f1_r*f2_r) - (-disc_p/(4*f1_p*f2_p)) - (-disc_q/(4*f1_q*f2_q))
    # = -disc_r/(4*f1_r*f2_r) + disc_p/(4*f1_p*f2_p) + disc_q/(4*f1_q*f2_q)
    #
    # Hmm, but we want Delta >= 0, i.e., inv_phi_r >= inv_phi_p + inv_phi_q.
    # Since f2 < 0, we have inv_phi = -disc/(4*f1*f2) = disc/(4*f1*(-f2)) > 0.
    # So inv_phi_r - inv_phi_p - inv_phi_q
    # = disc_r/(4*f1_r*(-f2_r)) - disc_p/(4*f1_p*(-f2_p)) - disc_q/(4*f1_q*(-f2_q))
    #
    # Let g1 = -f2_p > 0, g2 = -f2_q > 0, g3 = -f2_r > 0 on real-rooted cone.
    # Delta = disc_r/(4*f1_r*g3) - disc_p/(4*f1_p*g1) - disc_q/(4*f1_q*g2)
    #
    # Common denominator: 4 * f1_p * g1 * f1_q * g2 * f1_r * g3
    # All positive on real-rooted cone, so sign of Delta = sign of numerator.

    print("Computing surplus numerator...")
    print("(Common denominator: 4 * f1_p * (-f2_p) * f1_q * (-f2_q) * f1_r * (-f2_r))")
    print("(All factors positive on real-rooted cone)")
    print()

    g1 = -f2_p  # = 2 - 8*a4 - 9*a3^2
    g2 = -f2_q  # = 2 - 8*b4 - 9*b3^2
    g3 = -f2_r

    # Numerator of Delta (same sign as Delta on real-rooted cone):
    # N = disc_r * f1_p * g1 * f1_q * g2
    #   - disc_p * f1_q * g2 * f1_r * g3
    #   - disc_q * f1_p * g1 * f1_r * g3

    print("Expanding numerator term by term...")
    t0 = time.time()

    term1 = sp.expand(disc_r * f1_p * g1 * f1_q * g2)
    print(f"  Term 1 done ({time.time()-t0:.1f}s, {len(term1.as_ordered_terms())} terms)")

    term2 = sp.expand(disc_p * f1_q * g2 * f1_r * g3)
    print(f"  Term 2 done ({time.time()-t0:.1f}s, {len(term2.as_ordered_terms())} terms)")

    term3 = sp.expand(disc_q * f1_p * g1 * f1_r * g3)
    print(f"  Term 3 done ({time.time()-t0:.1f}s, {len(term3.as_ordered_terms())} terms)")

    N = sp.expand(term1 - term2 - term3)
    print(f"  Full numerator N ({time.time()-t0:.1f}s, {len(N.as_ordered_terms())} terms)")
    print()

    # Analyze structure
    p_N = sp.Poly(N, a3, a4, b3, b4)
    print(f"Numerator polynomial:")
    print(f"  Variables: a3, a4, b3, b4")
    print(f"  Total degree: {p_N.total_degree()}")
    print(f"  Number of terms: {len(p_N.as_dict())}")
    print()

    # Check symmetry: N(a3,a4,b3,b4) should equal N(b3,b4,a3,a4) (swap p,q)
    N_swapped = N.subs([(a3, b3), (a4, b4), (b3, a3), (b4, a4)])
    sym_diff = sp.expand(N - N_swapped)
    print(f"  Symmetric under (p,q) swap: {sym_diff == 0}")

    # Check reflection: N(-a3,a4,-b3,b4) should equal N(a3,a4,b3,b4)
    N_reflected = N.subs([(a3, -a3), (b3, -b3)])
    ref_diff = sp.expand(N - N_reflected)
    print(f"  Invariant under a3->-a3, b3->-b3: {ref_diff == 0}")

    # Check: at a3=b3=0, should reduce to symmetric case
    N_sym = N.subs([(a3, 0), (b3, 0)])
    N_sym = sp.expand(N_sym)
    N_sym_poly = sp.Poly(N_sym, a4, b4)
    print(f"  Symmetric case (a3=b3=0): degree {N_sym_poly.total_degree()}, "
          f"{len(N_sym_poly.as_dict())} terms")
    print(f"  N|_sym = {N_sym}")
    print()

    # Print the full numerator for the record
    print("Full numerator N(a3, a4, b3, b4):")
    print(N)
    print()

    # Degree breakdown by variable
    for var in [a3, a4, b3, b4]:
        max_deg = max(m[list(p_N.gens).index(var)] for m in p_N.as_dict().keys())
        print(f"  Max degree in {var}: {max_deg}")

    print()
    return N, (a3, a4, b3, b4)


def stage3_numerical_check(N_expr, variables):
    """Numerically verify the surplus numerator on random real-rooted quartics."""
    print()
    print("=" * 70)
    print("STAGE 3: Numerical verification of surplus numerator")
    print("=" * 70)
    print()

    a3, a4, b3, b4 = variables
    N_func = sp.lambdify((a3, a4, b3, b4), N_expr, 'numpy')

    violations = 0
    valid = 0
    min_val = float('inf')

    for trial in range(5000):
        # Generate random centered quartic with a2=-1 that has real roots
        # For real-rootedness with a2=-1: need disc > 0
        a3_val = np.random.uniform(-0.5, 0.5)
        a4_val = np.random.uniform(0, 0.3)
        b3_val = np.random.uniform(-0.5, 0.5)
        b4_val = np.random.uniform(0, 0.3)

        # Check real-rootedness of p
        p_roots = np.roots([1, 0, -1, a3_val, a4_val])
        if not np.all(np.isreal(p_roots)):
            continue

        # Check real-rootedness of q
        q_roots = np.roots([1, 0, -1, b3_val, b4_val])
        if not np.all(np.isreal(q_roots)):
            continue

        # Check real-rootedness of convolution
        c4_val = a4_val + 1.0/6 + b4_val
        r_roots = np.roots([1, 0, -2, a3_val + b3_val, c4_val])
        if not np.all(np.isreal(r_roots)):
            continue

        valid += 1
        try:
            val = float(N_func(a3_val, a4_val, b3_val, b4_val))
        except Exception:
            continue

        if val < min_val:
            min_val = val

        if val < -1e-6:
            violations += 1

    print(f"Valid trials: {valid}")
    print(f"Violations (N < -1e-6): {violations}")
    print(f"Minimum N value: {min_val:.6e}")
    print(f"Consistent with N >= 0 on real-rooted cone: {violations == 0}")


if __name__ == "__main__":
    t_start = time.time()

    verified = stage1_verify_identity()

    N, variables = stage2_general_surplus()

    stage3_numerical_check(N, variables)

    print()
    print(f"Total time: {time.time() - t_start:.1f}s")
