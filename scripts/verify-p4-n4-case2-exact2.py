#!/usr/bin/env python3
"""Case 2 exact proof, take 2: handle common factor in resultants.

When resultant(R12, R13, a4) = 0, R12 and R13 share a common factor in a4.
We compute the GCD, factor it, and analyze roots of each component.
"""

import sympy as sp
from sympy import (Rational, expand, symbols, diff, resultant, factor,
                   Poly, nroots, gcd, cancel, together, fraction, S,
                   factor_list, degree, LC, pprint)
import time
import sys

sys.stdout.reconfigure(line_buffering=True)


def build_neg_N():
    a3, a4, b3, b4 = sp.symbols('a3 a4 b3 b4')
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


def sub_u(expr, a3_sym, u_sym):
    """Replace a3² → u in an expression that is even in a3."""
    p = Poly(expr, a3_sym)
    result = S.Zero
    for monom, coeff in p.as_dict().items():
        power = monom[0]
        assert power % 2 == 0, f"Odd power {power} in even polynomial"
        result += coeff * u_sym**(power // 2)
    return expand(result)


def main():
    print("=" * 70)
    print("CASE 2 (v2): Exact algebraic analysis for b₃=0, a₃≠0")
    print("=" * 70)
    t0 = time.time()

    print("\nBuilding -N, gradient, restricting to b₃=0...")
    neg_N, (a3, a4, b3, b4) = build_neg_N()
    u = sp.Symbol('u')
    vars_orig = (a3, a4, b3, b4)

    g = [expand(diff(neg_N, v).subs(b3, 0)) for v in vars_orig]
    # g[0], g[2] are odd in a3; g[1], g[3] are even in a3

    # Divide odd components by a3
    h1_a3 = expand(sp.div(g[0], a3, a3)[0])  # g1/a3
    h3_a3 = expand(sp.div(g[2], a3, a3)[0])  # g3/a3

    # Substitute u = a3²
    h1 = sub_u(h1_a3, a3, u)
    h2 = sub_u(g[1], a3, u)
    h3 = sub_u(h3_a3, a3, u)
    h4 = sub_u(g[3], a3, u)
    print(f"  h1: deg_u={Poly(h1, u).degree()}, {len(h1.as_ordered_terms())} terms")
    print(f"  h2: deg_u={Poly(h2, u).degree()}, {len(h2.as_ordered_terms())} terms")
    print(f"  h3: deg_u={Poly(h3, u).degree()}, {len(h3.as_ordered_terms())} terms")
    print(f"  h4: deg_u={Poly(h4, u).degree()}, {len(h4.as_ordered_terms())} terms")
    print(f"  ({time.time()-t0:.1f}s)")

    print("\nComputing all pairwise resultants w.r.t. u...")
    pairs = [(0,1,'h1,h2'), (0,2,'h1,h3'), (0,3,'h1,h4'), (1,2,'h2,h3'), (1,3,'h2,h4'), (2,3,'h3,h4')]
    hs = [h1, h2, h3, h4]
    R = {}
    for i, j, name in pairs:
        t1 = time.time()
        rij = expand(resultant(hs[i], hs[j], u))
        R[(i,j)] = rij
        nterms = len(rij.as_ordered_terms()) if rij != 0 else 0
        if rij == 0:
            print(f"  res({name}, u) = 0  ({time.time()-t1:.1f}s)")
        else:
            da = Poly(rij, a4).degree()
            db = Poly(rij, b4).degree()
            print(f"  res({name}, u): {nterms} terms, deg_a4={da}, deg_b4={db}  ({time.time()-t1:.1f}s)")

    # Find a nonzero pair
    nonzero_pairs = [(k, v) for k, v in R.items() if v != 0]
    print(f"\n  Nonzero resultants: {len(nonzero_pairs)}/{len(R)}")

    if len(nonzero_pairs) < 2:
        print("  ERROR: Need at least 2 nonzero resultants for elimination")
        print("  The system may have a positive-dimensional solution component")
        print("  Trying Gröbner basis instead...")

        print("\n  Computing Gröbner basis of {h1, h2, h3, h4} w.r.t. (u, a4, b4)...")
        t1 = time.time()
        try:
            gb = groebner([h1, h2, h3, h4], u, a4, b4, order='grevlex')
            print(f"    Gröbner basis size: {len(gb)}")
            for i, p in enumerate(gb):
                terms = len(expand(p).as_ordered_terms())
                print(f"    gb[{i}]: {terms} terms, degree {Poly(p, u, a4, b4).total_degree()}")
            print(f"    ({time.time()-t1:.1f}s)")

            # Convert to lex order for back-substitution
            print("\n  Converting to lex order...")
            t1 = time.time()
            gb_lex = groebner([h1, h2, h3, h4], u, a4, b4, order='lex')
            print(f"    Gröbner basis (lex) size: {len(gb_lex)}")
            for i, p in enumerate(gb_lex):
                terms = len(expand(p).as_ordered_terms())
                vars_in = [str(v) for v in [u, a4, b4] if p.has(v)]
                print(f"    gb_lex[{i}]: {terms} terms, variables: {vars_in}")
            print(f"    ({time.time()-t1:.1f}s)")
        except Exception as e:
            print(f"    Gröbner basis failed: {e}")
            print(f"    ({time.time()-t1:.1f}s)")

        # Also try: compute GCD of all 4 polynomials
        print("\n  Computing pairwise GCDs...")
        for i, j, name in pairs:
            t1 = time.time()
            g_ij = gcd(Poly(hs[i], u, a4, b4), Poly(hs[j], u, a4, b4))
            if g_ij.is_number:
                print(f"    gcd({name}): constant")
            else:
                terms = len(expand(g_ij.as_expr()).as_ordered_terms())
                td = g_ij.total_degree()
                print(f"    gcd({name}): {terms} terms, total degree {td}  ({time.time()-t1:.1f}s)")

        return

    # Pick two nonzero resultants
    (i1, j1), R1 = nonzero_pairs[0]
    (i2, j2), R2 = nonzero_pairs[1]
    print(f"\n  Using R_{i1}{j1} and R_{i2}{j2} for a4 elimination...")

    # Factor them first
    print(f"  Factoring R_{i1}{j1}...")
    t1 = time.time()
    R1_factors = factor_list(R1, a4, b4)
    print(f"    Content: {R1_factors[0]}")
    for fac, mult in R1_factors[1]:
        nterms = len(expand(fac).as_ordered_terms())
        da = Poly(fac, a4).degree() if fac.has(a4) else 0
        db = Poly(fac, b4).degree() if fac.has(b4) else 0
        print(f"    Factor^{mult}: {nterms} terms, deg_a4={da}, deg_b4={db}")
    print(f"    ({time.time()-t1:.1f}s)")

    print(f"  Factoring R_{i2}{j2}...")
    t1 = time.time()
    R2_factors = factor_list(R2, a4, b4)
    print(f"    Content: {R2_factors[0]}")
    for fac, mult in R2_factors[1]:
        nterms = len(expand(fac).as_ordered_terms())
        da = Poly(fac, a4).degree() if fac.has(a4) else 0
        db = Poly(fac, b4).degree() if fac.has(b4) else 0
        print(f"    Factor^{mult}: {nterms} terms, deg_a4={da}, deg_b4={db}")
    print(f"    ({time.time()-t1:.1f}s)")

    # GCD of R1 and R2
    print(f"\n  Computing GCD of R_{i1}{j1} and R_{i2}{j2} in (a4, b4)...")
    t1 = time.time()
    G = gcd(Poly(R1, a4, b4), Poly(R2, a4, b4))
    if G.is_number:
        print(f"    GCD is constant → R_{i1}{j1} and R_{i2}{j2} are coprime")
    else:
        G_expr = expand(G.as_expr())
        nterms = len(G_expr.as_ordered_terms())
        print(f"    GCD: {nterms} terms, total degree {G.total_degree()}")
    print(f"    ({time.time()-t1:.1f}s)")

    # If GCD is nontrivial, analyze its zeros; otherwise compute resultant
    if not G.is_number:
        # Common solutions lie on the GCD variety
        print("\n  Analyzing common factor G(a4, b4)...")
        G_factors = factor_list(G.as_expr(), a4, b4)
        print(f"    G factors: {len(G_factors[1])}")
        for fac, mult in G_factors[1]:
            nterms = len(expand(fac).as_ordered_terms())
            print(f"    Factor^{mult}: {nterms} terms")
            # If factor depends on only one variable, find roots directly
            if not fac.has(a4):
                print(f"      Pure b4 polynomial, finding roots...")
                roots = nroots(fac, n=15)
                real_r = [complex(r).real for r in roots if abs(complex(r).imag) < 1e-8]
                print(f"      Real roots: {[f'{r:.8f}' for r in sorted(real_r)]}")
            elif not fac.has(b4):
                print(f"      Pure a4 polynomial, finding roots...")
                roots = nroots(fac, n=15)
                real_r = [complex(r).real for r in roots if abs(complex(r).imag) < 1e-8]
                print(f"      Real roots: {[f'{r:.8f}' for r in sorted(real_r)]}")
    else:
        # Coprime: compute resultant
        print(f"\n  Computing resultant(R_{i1}{j1}, R_{i2}{j2}, a4)...")
        t1 = time.time()
        R_final = expand(resultant(R1, R2, a4))
        if R_final == 0:
            print(f"    ZERO (unexpected for coprime polynomials)")
        else:
            db = Poly(R_final, b4).degree()
            nterms = len(R_final.as_ordered_terms())
            print(f"    R_final: {nterms} terms, degree {db} in b4")

            # Factor and find roots
            print("    Factoring...")
            R_final_f = factor_list(R_final, b4)
            print(f"    Content: {R_final_f[0]}")
            for fac, mult in R_final_f[1]:
                print(f"    Factor^{mult}: degree {Poly(fac, b4).degree()}")

            print("    Finding real roots...")
            roots = nroots(R_final, n=15)
            real_r = sorted([complex(r).real for r in roots if abs(complex(r).imag) < 1e-8])
            domain_r = [r for r in real_r if -1/12 - 0.01 <= r <= 0.25 + 0.01]
            print(f"    Real roots: {len(real_r)}, in domain: {len(domain_r)}")
            for r in domain_r:
                print(f"      b4 = {r:.12f}")
        print(f"    ({time.time()-t1:.1f}s)")

    print(f"\n{'='*70}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
