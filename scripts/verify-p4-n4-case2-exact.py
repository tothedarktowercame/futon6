#!/usr/bin/env python3
"""Exact algebraic proof: no critical points with b₃=0, a₃≠0 in domain.

Uses resultant elimination to reduce the gradient system to univariate,
then checks all real roots against domain constraints.

Case 2 structure:
- At b₃=0, the parity symmetry (-N)(a₃,a₄,b₃,b₄) = (-N)(-a₃,a₄,-b₃,b₄) gives:
  * g₁ = ∂(-N)/∂a₃ is odd in a₃ → divisible by a₃
  * g₃ = ∂(-N)/∂b₃ is odd in a₃ → divisible by a₃
  * g₂ = ∂(-N)/∂a₄ is even in a₃ → polynomial in u=a₃²
  * g₄ = ∂(-N)/∂b₄ is even in a₃ → polynomial in u=a₃²
- After dividing g₁,g₃ by a₃ and substituting u=a₃²:
  4 equations h₁,h₂,h₃,h₄ in 3 unknowns (u,a₄,b₄)
- Resultant elimination: u → a₄ → univariate in b₄
"""

import sympy as sp
from sympy import (Rational, expand, symbols, diff, resultant, factor,
                   Poly, real_roots, nroots, RootOf, Interval, oo,
                   solve, groebner, together, fraction, sqrt, S)
import time
import sys

sys.stdout.reconfigure(line_buffering=True)


def build_neg_N():
    a3, a4, b3, b4 = sp.symbols('a3 a4 b3 b4')
    # Centered (a₂ = b₂ = -1) degree-4 polynomials
    disc_p = expand(256*a4**3 - 128*a4**2 - 144*a3**2*a4 - 27*a3**4 + 16*a4 + 4*a3**2)
    f1_p = 1 + 12*a4
    f2_p = 9*a3**2 + 8*a4 - 2
    disc_q = expand(256*b4**3 - 128*b4**2 - 144*b3**2*b4 - 27*b3**4 + 16*b4 + 4*b3**2)
    f1_q = 1 + 12*b4
    f2_q = 9*b3**2 + 8*b4 - 2
    c3 = a3 + b3
    c4 = a4 + Rational(1, 6) + b4
    disc_r = expand(256*c4**3 - 512*c4**2 + 288*c3**2*c4 - 27*c3**4 + 256*c4 + 32*c3**2)
    f1_r = expand(4 + 12*c4)
    f2_r = expand(-16 + 16*c4 + 9*c3**2)
    surplus_frac = together(-disc_r/(4*f1_r*f2_r) + disc_p/(4*f1_p*f2_p) + disc_q/(4*f1_q*f2_q))
    num, den = fraction(surplus_frac)
    neg_N = expand(-num)
    return neg_N, (a3, a4, b3, b4)


def main():
    print("=" * 70)
    print("CASE 2: Exact algebraic proof for b₃=0, a₃≠0")
    print("=" * 70)
    t0 = time.time()

    print("\nStep 1: Build -N and gradient...")
    neg_N, (a3, a4, b3, b4) = build_neg_N()
    g1 = expand(diff(neg_N, a3))
    g2 = expand(diff(neg_N, a4))
    g3 = expand(diff(neg_N, b3))
    g4 = expand(diff(neg_N, b4))
    print(f"  Done ({time.time()-t0:.1f}s)")

    print("\nStep 2: Restrict to b₃=0...")
    g1_0 = expand(g1.subs(b3, 0))
    g2_0 = expand(g2.subs(b3, 0))
    g3_0 = expand(g3.subs(b3, 0))
    g4_0 = expand(g4.subs(b3, 0))
    print(f"  g1|_{{b3=0}} terms: {len(g1_0.as_ordered_terms())}")
    print(f"  g2|_{{b3=0}} terms: {len(g2_0.as_ordered_terms())}")
    print(f"  g3|_{{b3=0}} terms: {len(g3_0.as_ordered_terms())}")
    print(f"  g4|_{{b3=0}} terms: {len(g4_0.as_ordered_terms())}")

    # Verify parity: g1, g3 should be odd in a3; g2, g4 should be even
    print("\n  Parity check:")
    print(f"    g1 odd in a3: {expand(g1_0 + g1_0.subs(a3, -a3)) == 0}")
    print(f"    g3 odd in a3: {expand(g3_0 + g3_0.subs(a3, -a3)) == 0}")
    print(f"    g2 even in a3: {expand(g2_0 - g2_0.subs(a3, -a3)) == 0}")
    print(f"    g4 even in a3: {expand(g4_0 - g4_0.subs(a3, -a3)) == 0}")
    print(f"  ({time.time()-t0:.1f}s)")

    print("\nStep 3: Factor out a₃ and substitute u = a₃²...")
    u = sp.Symbol('u')

    # g1_0 = a3 * h1(a3², a4, b4)
    p1 = Poly(g1_0, a3)
    # Divide by a3
    h1_poly, rem = sp.div(g1_0, a3, a3)
    assert expand(rem) == 0, f"g1 not divisible by a3! remainder = {rem}"
    h1_a3 = expand(h1_poly)  # h1 in terms of a3, a4, b4 (even in a3)
    print(f"  g1/a3 terms: {len(h1_a3.as_ordered_terms())}")

    # g3_0 = a3 * h3(a3², a4, b4)
    h3_poly, rem = sp.div(g3_0, a3, a3)
    assert expand(rem) == 0, f"g3 not divisible by a3! remainder = {rem}"
    h3_a3 = expand(h3_poly)
    print(f"  g3/a3 terms: {len(h3_a3.as_ordered_terms())}")

    # Substitute u = a3² in all four equations
    # h1_a3 and h3_a3 are even in a3, so they're polynomials in a3²
    # g2_0 and g4_0 are also even in a3

    def sub_u(expr, a3_sym, u_sym):
        """Replace a3² → u in an expression that is even in a3."""
        p = Poly(expr, a3_sym)
        result = S.Zero
        for monom, coeff in p.as_dict().items():
            power = monom[0]
            assert power % 2 == 0, f"Odd power {power} found in supposedly even polynomial"
            result += coeff * u_sym**(power // 2)
        return expand(result)

    h1 = sub_u(h1_a3, a3, u)
    h3 = sub_u(h3_a3, a3, u)
    h2 = sub_u(g2_0, a3, u)
    h4 = sub_u(g4_0, a3, u)

    print(f"\n  h1(u,a4,b4) terms: {len(h1.as_ordered_terms())}")
    print(f"  h2(u,a4,b4) terms: {len(h2.as_ordered_terms())}")
    print(f"  h3(u,a4,b4) terms: {len(h3.as_ordered_terms())}")
    print(f"  h4(u,a4,b4) terms: {len(h4.as_ordered_terms())}")

    # Degrees in each variable
    for name, h in [('h1', h1), ('h2', h2), ('h3', h3), ('h4', h4)]:
        du = Poly(h, u).degree() if h.has(u) else 0
        da = Poly(h, a4).degree() if h.has(a4) else 0
        db = Poly(h, b4).degree() if h.has(b4) else 0
        dt = Poly(h, u, a4, b4).total_degree()
        print(f"    {name}: deg_u={du}, deg_a4={da}, deg_b4={db}, total={dt}")
    print(f"  ({time.time()-t0:.1f}s)")

    print("\nStep 4: Resultant elimination — eliminate u...")
    # Take resultant of h1 and h2 w.r.t. u → R12(a4, b4)
    # Take resultant of h1 and h3 w.r.t. u → R13(a4, b4)
    # (Using h1 with h2 and h3 since h1 likely has lowest degree in u)

    print("  Computing resultant(h1, h2, u)...")
    t1 = time.time()
    R12 = resultant(h1, h2, u)
    R12 = expand(R12)
    print(f"    R12 terms: {len(R12.as_ordered_terms())} ({time.time()-t1:.1f}s)")

    print("  Computing resultant(h1, h3, u)...")
    t1 = time.time()
    R13 = resultant(h1, h3, u)
    R13 = expand(R13)
    print(f"    R13 terms: {len(R13.as_ordered_terms())} ({time.time()-t1:.1f}s)")

    # Check degrees
    for name, R in [('R12', R12), ('R13', R13)]:
        da = Poly(R, a4).degree() if R.has(a4) else 0
        db = Poly(R, b4).degree() if R.has(b4) else 0
        dt = Poly(R, a4, b4).total_degree()
        print(f"    {name}: deg_a4={da}, deg_b4={db}, total={dt}")

    # Try to factor
    print("  Factoring R12...")
    t1 = time.time()
    R12_f = factor(R12)
    print(f"    R12 factored: {len(str(R12_f))} chars ({time.time()-t1:.1f}s)")

    print("  Factoring R13...")
    t1 = time.time()
    R13_f = factor(R13)
    print(f"    R13 factored: {len(str(R13_f))} chars ({time.time()-t1:.1f}s)")
    print(f"  ({time.time()-t0:.1f}s)")

    print("\nStep 5: Resultant elimination — eliminate a₄...")
    # Compute resultant(R12, R13, a4) → univariate in b4
    print("  Computing resultant(R12, R13, a4)...")
    t1 = time.time()
    R_final = resultant(R12, R13, a4)
    R_final = expand(R_final)
    print(f"    R_final terms: {len(R_final.as_ordered_terms())} ({time.time()-t1:.1f}s)")
    print(f"    R_final degree in b4: {Poly(R_final, b4).degree()}")

    print("  Factoring R_final...")
    t1 = time.time()
    R_final_f = factor(R_final)
    print(f"    Factored ({time.time()-t1:.1f}s)")

    # Extract irreducible factors
    factors = sp.factorint(Poly(R_final, b4))
    if not factors:
        # Try factoring as expression
        from sympy import Mul, Pow
        R_final_f_expanded = sp.factor_list(R_final, b4)
        print(f"    factor_list: {R_final_f_expanded[0]} * ...")
        factors_list = R_final_f_expanded[1]
    else:
        factors_list = list(factors.items())

    print(f"\nStep 6: Analyze real roots of R_final(b₄)...")

    # Find all real roots in the domain range b4 ∈ [-1/12, 1/4]
    b4_lo = Rational(-1, 12)
    b4_hi = Rational(1, 4)

    # Get numerical roots
    print("  Finding numerical roots...")
    t1 = time.time()
    try:
        all_roots = nroots(R_final, n=20)
        real_roots_list = sorted([complex(r).real for r in all_roots if abs(complex(r).imag) < 1e-10])
        domain_roots = [r for r in real_roots_list if float(b4_lo) - 0.001 <= r <= float(b4_hi) + 0.001]
        print(f"    Total roots: {len(all_roots)}")
        print(f"    Real roots: {len(real_roots_list)}")
        print(f"    Roots in [b4_lo, b4_hi]: {len(domain_roots)}")
        for r in domain_roots:
            print(f"      b4 = {r:.15f}")
    except Exception as e:
        print(f"    nroots failed: {e}")
        # Try RealRoots
        try:
            rr = real_roots(Poly(R_final, b4))
            domain_roots_exact = [r for r in rr if b4_lo <= r <= b4_hi]
            print(f"    Real roots (exact): {len(rr)}")
            print(f"    In domain: {len(domain_roots_exact)}")
        except Exception as e2:
            print(f"    real_roots also failed: {e2}")

    print(f"    ({time.time()-t1:.1f}s)")

    print("\nStep 7: Back-substitute to find all critical points...")
    # For each b4 root in domain, solve R12(a4, b4) = R13(a4, b4) = 0 for a4
    # Then solve for u from h1(u, a4, b4) = 0
    # Check u > 0 (since u = a3²) and domain constraints

    if 'domain_roots' in dir():
        for b4_val in domain_roots:
            b4_rat = Rational(b4_val).limit_denominator(10**12)
            print(f"\n  b4 ≈ {float(b4_rat):.12f}:")

            # Solve R12(a4, b4_val) = 0 for a4
            R12_b4 = R12.subs(b4, b4_rat)
            try:
                a4_roots = nroots(R12_b4, n=15)
                a4_real = sorted([complex(r).real for r in a4_roots if abs(complex(r).imag) < 1e-8])
                a4_domain = [r for r in a4_real if float(Rational(-1, 12)) - 0.001 <= r <= 0.25 + 0.001]
                print(f"    a4 roots in domain: {len(a4_domain)}")

                for a4_val in a4_domain:
                    # Solve for u from h1
                    h1_sub = h1.subs([(a4, Rational(a4_val).limit_denominator(10**12)),
                                       (b4, b4_rat)])
                    u_roots = nroots(h1_sub, n=15)
                    u_real_pos = [complex(r).real for r in u_roots
                                  if abs(complex(r).imag) < 1e-8 and complex(r).real > 1e-10]

                    for u_val in u_real_pos:
                        a3_val = u_val**0.5
                        # Check domain constraints
                        disc_p_val = 256*a4_val**3 - 128*a4_val**2 - 144*u_val*a4_val - 27*u_val**2 + 16*a4_val + 4*u_val
                        disc_q_val = 256*b4_val**3 - 128*b4_val**2 + 16*b4_val
                        f1_p_val = 1 + 12*a4_val
                        f1_q_val = 1 + 12*b4_val
                        f2_p_val = 9*u_val + 8*a4_val - 2
                        f2_q_val = 8*b4_val - 2

                        in_dom = (disc_p_val >= -1e-6 and disc_q_val >= -1e-6 and
                                  f1_p_val > -1e-6 and f1_q_val > -1e-6 and
                                  f2_p_val < 1e-6 and f2_q_val < 1e-6)

                        # Also verify gradient actually vanishes
                        grad_check = max(abs(float(h1.subs([(u, u_val), (a4, a4_val), (b4, b4_val)]))),
                                         abs(float(h2.subs([(u, u_val), (a4, a4_val), (b4, b4_val)]))),
                                         abs(float(h3.subs([(u, u_val), (a4, a4_val), (b4, b4_val)]))),
                                         abs(float(h4.subs([(u, u_val), (a4, a4_val), (b4, b4_val)]))))

                        if in_dom:
                            print(f"      *** IN DOMAIN: a3={a3_val:.8f}, a4={a4_val:.8f}, b4={b4_val:.8f}")
                            print(f"          u={u_val:.8f}, disc_p={disc_p_val:.6f}, disc_q={disc_q_val:.6f}")
                            print(f"          f1_p={f1_p_val:.6f}, f1_q={f1_q_val:.6f}")
                            print(f"          f2_p={f2_p_val:.6f}, f2_q={f2_q_val:.6f}")
                            print(f"          |grad| check: {grad_check:.2e}")
                        elif grad_check < 1e-4:
                            print(f"      outside domain: a3={a3_val:.6f}, a4={a4_val:.6f}, b4={b4_val:.6f}"
                                  f" (disc_p={disc_p_val:.4f}, f2_p={f2_p_val:.4f})")
            except Exception as e:
                print(f"    Failed: {e}")

    print(f"\n{'='*70}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
