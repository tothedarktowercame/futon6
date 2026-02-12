#!/usr/bin/env python3
"""Case 3c: General (off-diagonal) critical points with a₃≠0, b₃≠0, a₃≠±b₃.

Strategy: Use the parity + exchange symmetry to reduce the system.

The gradient system has 4 equations g₁..g₄ in 4 variables (a₃,a₄,b₃,b₄).
On the "off-diagonal" where a₃≠±b₃:
- g₁ and g₃ are odd under parity (a₃,b₃)→(-a₃,-b₃)
- g₂ and g₄ are even

We can write g₁ = a₃·P₁ + b₃·P₂ where P₁,P₂ are polynomials in (a₃²,a₃b₃,b₃²,a₄,b₄).
Similarly g₃ = a₃·P₃ + b₃·P₄.

For a₃≠0, b₃≠0 with g₁=g₃=0, this gives a 2×2 linear system in (a₃,b₃):
[P₁  P₂] [a₃]   [0]
[P₃  P₄] [b₃] = [0]

For nontrivial solutions: det(M) = P₁P₄ - P₂P₃ = 0.

This is a polynomial in (a₃²,a₃b₃,b₃²,a₄,b₄). Together with the even equations
g₂ = 0, g₄ = 0, we have 3 equations in the invariant variables.

Approach here: eliminate a₃ from the full 4D system using resultants.
"""

import sympy as sp
from sympy import (Rational, expand, symbols, diff, resultant,
                   Poly, gcd, together, fraction, S, factor_list, quo,
                   count_roots)
import numpy as np
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


def main():
    print("=" * 70)
    print("CASE 3c: Off-diagonal critical points (a₃≠0, b₃≠0, a₃≠±b₃)")
    print("=" * 70)
    t_start = time.time()

    a3, a4, b3, b4 = sp.symbols('a3 a4 b3 b4')

    print("\nBuilding -N and gradient...")
    neg_N, _ = build_neg_N()
    g1 = expand(diff(neg_N, a3))
    g2 = expand(diff(neg_N, a4))
    g3 = expand(diff(neg_N, b3))
    g4 = expand(diff(neg_N, b4))
    print(f"  g1: {len(g1.as_ordered_terms())} terms, deg a3={Poly(g1,a3).degree()}")
    print(f"  g2: {len(g2.as_ordered_terms())} terms, deg a3={Poly(g2,a3).degree()}")
    print(f"  g3: {len(g3.as_ordered_terms())} terms, deg a3={Poly(g3,a3).degree()}")
    print(f"  g4: {len(g4.as_ordered_terms())} terms, deg a3={Poly(g4,a3).degree()}")
    print(f"  ({time.time()-t_start:.1f}s)")

    # Step 1: Eliminate a₃ from pairs of equations
    # Use g1 (odd in a3) with g2 (even in a3)
    # g1 is odd: deg 9 in a3 with odd powers only → effectively deg 4 in a3² after division by a3
    # g2 is even: deg 8 in a3 with even powers only → effectively deg 4 in a3²
    # But the elimination variable is a3, not a3², so direct resultant has degree deg_g1 * deg_g2 in other vars

    # Strategy: since g1 is odd in (a3,b3), we know g1 = a3*h1e + b3*h1o
    # where h1e, h1o depend on a3², a3*b3, b3², a4, b4
    # Similarly g3 = a3*h3e + b3*h3o
    # The determinant condition: h1e*h3o - h1o*h3e = 0

    # But this substitution is complex. Let's try direct resultant first.
    # res(g1, g2, a3) gives a polynomial in (a4, b3, b4) of degree ~72
    # which is likely too high.

    # Better: use the structure. g1 is degree 5 in a3 (odd powers: 1,3,5,7,9 → max 9),
    # g2 is degree 8 in a3 (even powers: 0,2,4,6,8).
    # Actually let me check the actual degrees.

    print("\nStep 1: Analyze a₃-degree structure...")
    for name, g in [('g1', g1), ('g2', g2), ('g3', g3), ('g4', g4)]:
        p = Poly(g, a3)
        powers = sorted(p.as_dict().keys())
        max_pow = max(m[0] for m in powers)
        min_pow = min(m[0] for m in powers)
        even_odd = "odd" if all(m[0] % 2 == 1 for m in powers) else "even" if all(m[0] % 2 == 0 for m in powers) else "mixed"
        print(f"  {name}: a3-degree {max_pow}, {even_odd}, {len(powers)} distinct powers")

    # For the off-diagonal case, we want to find solutions where a3 ≠ 0, b3 ≠ 0,
    # and (a3,a4) ≠ (b3,b4) and (a3,a4) ≠ (-b3,b4).
    #
    # Approach: eliminate a3 from g1 and g2 using resultant.
    # Since g1 is odd in a3 (degree 9) and g2 is even (degree 8),
    # the resultant has degree 9*8 = 72 in (b3, a4, b4).
    # That's already borderline. Let's try.

    print("\nStep 2: Compute res(g1, g2, a3)...")
    t1 = time.time()
    R12 = expand(resultant(g1, g2, a3))
    print(f"  R12: {len(R12.as_ordered_terms())} terms ({time.time()-t1:.1f}s)")
    print(f"    deg b3={Poly(R12, b3).degree()}, deg a4={Poly(R12, a4).degree()}, deg b4={Poly(R12, b4).degree()}")

    print("\nStep 3: Compute res(g1, g4, a3)...")
    t1 = time.time()
    R14 = expand(resultant(g1, g4, a3))
    print(f"  R14: {len(R14.as_ordered_terms())} terms ({time.time()-t1:.1f}s)")
    print(f"    deg b3={Poly(R14, b3).degree()}, deg a4={Poly(R14, a4).degree()}, deg b4={Poly(R14, b4).degree()}")

    # Also try g3 with g2 — might give a simpler resultant
    print("\nStep 4: Compute res(g3, g2, a3)...")
    t1 = time.time()
    R32 = expand(resultant(g3, g2, a3))
    print(f"  R32: {len(R32.as_ordered_terms())} terms ({time.time()-t1:.1f}s)")
    print(f"    deg b3={Poly(R32, b3).degree()}, deg a4={Poly(R32, a4).degree()}, deg b4={Poly(R32, b4).degree()}")

    # Check if any pair shares a GCD that could simplify things
    print("\nStep 5: Check for common factors...")
    t1 = time.time()
    G = gcd(Poly(R12, b3, a4, b4), Poly(R14, b3, a4, b4))
    if G.is_number:
        print(f"  gcd(R12, R14) = constant ({time.time()-t1:.1f}s)")
    else:
        G_terms = len(expand(G.as_expr()).as_ordered_terms())
        print(f"  gcd(R12, R14): {G_terms} terms, total deg {G.total_degree()} ({time.time()-t1:.1f}s)")

    # Now eliminate b3 from the pair (R12, R14) or (R12, R32)
    # This will give a polynomial in (a4, b4).
    # R12 has degree ~72 in b3, R14 has degree ~72 in b3.
    # Resultant would give degree ~72^2 = 5184 in (a4, b4). Way too high!

    # Better approach: use the fact that R12 and R32 might have common factors
    # related to the diagonal/anti-diagonal solutions we already know about.

    # Factor out known solutions:
    # On diagonal (b3=a3): this was already analyzed
    # On anti-diagonal (b3=-a3): already analyzed
    # So R12 should be divisible by (b3 - a3_val)(b3 + a3_val) for each diagonal solution

    # Actually, since g1=g3=0 on the diagonal (and g1=-g3=0 on anti-diagonal),
    # the resultant R12 = res(g1,g2,a3) encodes all solutions of g1=g2=0 as a function of b3.
    # This includes spurious solutions where g3, g4 ≠ 0.

    # The true system is g1=g2=g3=g4=0. We've eliminated a3 from g1,g2 → R12=0
    # and from g1,g4 → R14=0. Now we need R12=R14=0 simultaneously.
    # R12 has degree ~72 in b3, and R14 has degree ~72.
    # Resultant res(R12, R14, b3) would be degree ~72^2 ≈ 5000 in (a4, b4).
    # That's too high for exact computation.

    print("\nDegrees are too high for direct elimination.")
    print("Trying reduced approach: fix b4, solve 3D system numerically...")

    # Alternative: use the exchange symmetry to reduce.
    # For the known CP at (0.0624, 0.1665, -0.2485, 0.0204):
    # Let's check: what are a4+b4, a4-b4, a3+b3, a3-b3?
    a3_cp, a4_cp, b3_cp, b4_cp = 0.062444085273, 0.166488887453, -0.248538785912, 0.020345739459
    print(f"\n  Known CP: ({a3_cp:.6f}, {a4_cp:.6f}, {b3_cp:.6f}, {b4_cp:.6f})")
    print(f"    a3+b3 = {a3_cp+b3_cp:.6f}")
    print(f"    a3-b3 = {a3_cp-b3_cp:.6f}")
    print(f"    a4+b4 = {a4_cp+b4_cp:.6f}")
    print(f"    a4-b4 = {a4_cp-b4_cp:.6f}")

    # Try: fix the sum S = a4+b4 and difference s = a3+b3
    # and parameterize with d = a3-b3, D = a4-b4
    # Then a3 = (s+d)/2, b3 = (s-d)/2, a4 = (S+D)/2, b4 = (S-D)/2

    print("\nStep 6: Try (s, d, S, D) coordinates...")
    s, d, S, D = sp.symbols('s d S D_')

    # Substitute
    subs_list = [(a3, (s+d)/2), (b3, (s-d)/2), (a4, (S+D)/2), (b4, (S-D)/2)]
    print("  Substituting into gradient...")
    t1 = time.time()
    g1_sd = expand(g1.subs(subs_list))
    g2_sd = expand(g2.subs(subs_list))
    g3_sd = expand(g3.subs(subs_list))
    g4_sd = expand(g4.subs(subs_list))
    print(f"    g1: {len(g1_sd.as_ordered_terms())} terms")
    print(f"    g2: {len(g2_sd.as_ordered_terms())} terms")
    print(f"    g3: {len(g3_sd.as_ordered_terms())} terms")
    print(f"    g4: {len(g4_sd.as_ordered_terms())} terms")

    # Form symmetric/antisymmetric combinations
    g_plus_1 = expand(g1_sd + g3_sd)   # symmetric in (a3,b3) exchange
    g_minus_1 = expand(g1_sd - g3_sd)  # antisymmetric
    g_plus_2 = expand(g2_sd + g4_sd)   # symmetric
    g_minus_2 = expand(g2_sd - g4_sd)  # antisymmetric

    print(f"    g1+g3: {len(g_plus_1.as_ordered_terms())} terms")
    print(f"    g1-g3: {len(g_minus_1.as_ordered_terms())} terms")
    print(f"    g2+g4: {len(g_plus_2.as_ordered_terms())} terms")
    print(f"    g2-g4: {len(g_minus_2.as_ordered_terms())} terms")

    # Check which are odd/even in d, D
    # Exchange symmetry: (s,d,S,D) → (s,-d,S,-D), so d→-d, D→-D
    # g1+g3 should be symmetric (even in d,D)
    # g1-g3 should be antisymmetric (odd in d,D)
    print(f"    g1+g3 even in (d,D): {expand(g_plus_1 - g_plus_1.subs([(d,-d),(D,-D)])) == 0}")
    print(f"    g1-g3 odd in (d,D): {expand(g_minus_1 + g_minus_1.subs([(d,-d),(D,-D)])) == 0}")
    print(f"    g2+g4 even in (d,D): {expand(g_plus_2 - g_plus_2.subs([(d,-d),(D,-D)])) == 0}")
    print(f"    g2-g4 odd in (d,D): {expand(g_minus_2 + g_minus_2.subs([(d,-d),(D,-D)])) == 0}")

    # Parity: (s,d,S,D) → (-s,-d,S,D)
    # g1+g3 should be odd in (s,d) jointly (since g1 is odd under parity)
    print(f"    g1+g3 odd in (s,d): {expand(g_plus_1 + g_plus_1.subs([(s,-s),(d,-d)])) == 0}")

    # For the off-diagonal (d≠0, D≠0):
    # g1-g3 is odd in (d,D): write g1-g3 = d·A + D·B
    # g2-g4 is odd in (d,D): write g2-g4 = d·C + D·E
    # For d≠0, D≠0: these give equations A + (D/d)B = 0, C + (D/d)E = 0
    # Setting t = D/d: A + t·B = 0, C + t·E = 0 → t = -A/B = -C/E

    # But A, B, C, E depend on (s², d², S, D²=t²d², sd, dD=td²) which is complex.

    # Let me try the direct numerical approach for Case 3c instead:
    # Show that all off-diagonal CPs have -N > 0 by interval arithmetic.
    print(f"\n  ({time.time()-t1:.1f}s)")

    print("\n" + "="*70)
    print("Direct elimination has degree ~5000 — infeasible.")
    print("The 4 off-diagonal CPs (all -N ≈ 1679) need a different approach.")
    print("="*70)

    print(f"\nTotal time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
