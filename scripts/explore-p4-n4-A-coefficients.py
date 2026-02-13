#!/usr/bin/env python3
"""Extract and analyze the coefficients of A(r,x,y,P,Q) where K = A + pq*B.

A has 13 (P,Q)-monomials. For each, extract the coefficient as a polynomial
in (r,x,y), try to factor it, and look for patterns.

The key structure:
  A = Σ C_{ij}(r,x,y) * P^i * Q^j

where the 13 non-zero (i,j) are:
  (0,0), (0,1), (0,2), (0,3),
  (1,0), (1,1), (1,2), (1,3),
  (2,0), (2,1), (2,2),
  (3,0), (3,1)

Goal: factor C_{ij}, look for a common factor, find the SOS/Gram structure.
"""

import sympy as sp
from sympy import Rational


def pr(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def disc_poly(e2, e3, e4):
    return (256*e4**3 - 128*e2**2*e4**2 + 144*e2*e3**2*e4
            + 16*e2**4*e4 - 27*e3**4 - 4*e2**3*e3**2)


def phi4_disc(e2, e3, e4):
    return (-8*e2**5 - 64*e2**3*e4 - 36*e2**2*e3**2
            + 384*e2*e4**2 - 432*e3**2*e4)


def build_A():
    pr("Building A(r,x,y,P,Q)...")
    s, t, u, v, a, b = sp.symbols('s t u v a b', positive=True)
    r, x, y, p, q = sp.symbols('r x y p q', real=True)
    P, Q = sp.symbols('P Q', positive=True)

    inv_phi = lambda e2, e3, e4: sp.cancel(disc_poly(e2, e3, e4) / phi4_disc(e2, e3, e4))
    inv_p = inv_phi(-s, u, a)
    inv_q = inv_phi(-t, v, b)
    inv_c = inv_phi(-(s+t), u+v, a+b+s*t/6)

    surplus = sp.together(inv_c - inv_p - inv_q)
    N, D = sp.fraction(surplus)
    N = sp.expand(N)
    pr(f"  N terms: {len(sp.Poly(N, s,t,u,v,a,b).as_dict())}")

    subs = {t: r*s, a: x*s**2/4, b: y*r**2*s**2/4,
            u: p*s**Rational(3,2), v: q*s**Rational(3,2)}
    K = sp.expand(sp.expand(N.subs(subs)) / s**16)
    pr(f"  K terms: {len(sp.Poly(K, r,x,y,p,q).as_dict())}")

    # Split: A collects even-even monomials, expressed in P=p², Q=q²
    poly_pq = sp.Poly(K, p, q)
    A_coeffs = {}  # (P-exp, Q-exp) -> coefficient in (r,x,y)
    B_coeffs = {}

    for monom, coeff in poly_pq.as_dict().items():
        i, j = monom
        if i % 2 == 0 and j % 2 == 0:
            A_coeffs[(i//2, j//2)] = sp.expand(coeff)
        elif i % 2 == 1 and j % 2 == 1:
            B_coeffs[((i-1)//2, (j-1)//2)] = sp.expand(coeff)

    return r, x, y, P, Q, A_coeffs, B_coeffs


def main():
    r, x, y, P, Q, A_coeffs, B_coeffs = build_A()

    pr(f"\n{'='*72}")
    pr("A COEFFICIENTS: C_{{ij}}(r,x,y) for P^i Q^j")
    pr('='*72)

    for (i, j) in sorted(A_coeffs.keys()):
        c = A_coeffs[(i,j)]
        f = sp.factor(c)
        nt = len(sp.Poly(c, r, x, y).as_dict())
        pr(f"\n  C_{{{i},{j}}} (P^{i}Q^{j}): {nt} terms")
        pr(f"    expanded: {c}")
        pr(f"    factored: {f}")

    pr(f"\n{'='*72}")
    pr("B COEFFICIENTS: D_{{ij}}(r,x,y) for pq * P^i Q^j")
    pr('='*72)

    for (i, j) in sorted(B_coeffs.keys()):
        c = B_coeffs[(i,j)]
        f = sp.factor(c)
        nt = len(sp.Poly(c, r, x, y).as_dict())
        pr(f"\n  D_{{{i},{j}}} (pq·P^{i}Q^{j}): {nt} terms")
        pr(f"    expanded: {c}")
        pr(f"    factored: {f}")

    # Key structural checks
    pr(f"\n{'='*72}")
    pr("STRUCTURAL ANALYSIS")
    pr('='*72)

    # Check: does (1+3x) or (1+3y) divide all coefficients?
    A1 = 1 + 3*x
    A2 = 1 + 3*y
    for name, coeffs in [("A", A_coeffs), ("B", B_coeffs)]:
        pr(f"\n  Divisibility by (1+3x) in {name}:")
        for (i, j) in sorted(coeffs.keys()):
            c = coeffs[(i,j)]
            q_div, r_div = sp.div(c, A1, r, x, y)
            r_str = str(sp.expand(r_div))
            pr(f"    {name}_{{{i},{j}}}: remainder = {r_str[:80]}{'...' if len(r_str) > 80 else ''}")

    # Check Pmax and Qmax substitution
    pr(f"\n{'='*72}")
    pr("BOUNDARY VALUE A(r,x,y,Pmax,Qmax)")
    pr('='*72)

    Pmax = 2*(1-x)/9
    Qmax = 2*r**3*(1-y)/9

    A_boundary = sp.Integer(0)
    for (i, j), c in A_coeffs.items():
        A_boundary = sp.expand(A_boundary + c * Pmax**i * Qmax**j)

    A_boundary = sp.factor(A_boundary)
    pr(f"  A(Pmax,Qmax) = {A_boundary}")

    # Check: A at P=0, Q=0 (should be K0)
    pr(f"\n  A(0,0) = C_{{0,0}} = K0")
    K0 = A_coeffs[(0,0)]
    K0_f = sp.factor(K0)
    pr(f"    factored: {K0_f}")

    # Check symmetry: C_{ij}(r,x,y) vs C_{ji}(1/r,y,x)
    pr(f"\n{'='*72}")
    pr("SYMMETRY: C_{{i,j}}(r,x,y) vs C_{{j,i}}(1/r,y,x)")
    pr('='*72)

    swap = {r: 1/r, x: y, y: x}
    for (i, j) in sorted(A_coeffs.keys()):
        c_ij = A_coeffs[(i,j)]
        if (j, i) in A_coeffs:
            c_ji_swapped = sp.expand(A_coeffs[(j,i)].subs(swap))
            # Need to account for the scaling: Pmax(x) -> Qmax(1/r,y)/r^3
            # Actually the swap (r,x,y) -> (1/r,y,x) sends
            # P=p² -> p² (same), Q=q² -> q² (same)... no wait
            # The normalization is different. P = p² = u²/s³ and Q = q² = v²/s³
            # Under the swap: p and q don't swap, but their bounds do.
            # Let's just check the simple ratio
            ratio = sp.simplify(c_ij / c_ji_swapped) if c_ji_swapped != 0 else "div by 0"
            pr(f"  C_{{{i},{j}}} / C_{{{j},{i}}}(1/r,y,x) = {ratio}")
        else:
            pr(f"  C_{{{i},{j}}}: no C_{{{j},{i}}} exists")

    # PART: check if A is a sum of squares in (P,Q) for some (r,x,y) ranges
    pr(f"\n{'='*72}")
    pr("GRAM MATRIX FOR A IN (P,Q)")
    pr('='*72)
    pr("v = [1, P, Q, P², PQ, Q², P³, P²Q, PQ², Q³, P³Q, P²Q², PQ³]")
    pr("Too many monomials for Gram. Try restricted: v = [1, P, Q, PQ]")

    # Actually, A has max total degree 4 in (P,Q).
    # Monomials in v for degree ≤ 2: {1, P, Q, P², PQ, Q²}
    # v^T G v covers degree ≤ 4.
    # Number of monomials: 6, so G is 6×6.
    # But A has 13 terms while v^T G v generates 21 products.
    # So there are 21 - 13 = 8 zero constraints, leaving 21 - 13 = 8... wait.
    # v^T G v generates monomials {P^a Q^b : a+b ≤ 4, a ≤ ..}
    # Actually, products of degree-≤-2 monomials give degree ≤ 4.
    # But P^2·Q^2 = P^2Q^2 (degree 4) ✓
    # P²·P² = P⁴ — but A has no P⁴ term!
    # So A has no P^4, P^3Q^2, P^2Q^3, P^3Q^3, Q^4, etc.
    # Some of these would need to be zero in the Gram representation.

    # Let me count which monomials v^T G v can generate:
    v_monoms = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2)]
    generated = {}
    for a in range(len(v_monoms)):
        for b in range(a, len(v_monoms)):
            pa, qa = v_monoms[a]
            pb, qb = v_monoms[b]
            m = (pa+pb, qa+qb)
            if m not in generated:
                generated[m] = []
            generated[m].append((a, b))

    pr(f"\n  Monomials from v⊗v:")
    A_monoms = set(A_coeffs.keys())
    for m in sorted(generated.keys()):
        in_A = "✓" if m in A_monoms else "=0"
        pr(f"    P^{m[0]}Q^{m[1]}: {len(generated[m])} entries [{in_A}]")

    missing = A_monoms - set(generated.keys())
    if missing:
        pr(f"\n  A has monomials NOT covered by degree-2 Gram: {missing}")
        pr(f"  Need degree-3 monomials in v to cover these!")

    # So we need v with degree-3 monomials too
    # v = [1, P, Q, P², PQ, Q², P³, P²Q, PQ², Q³]
    # But P³Q and PQ³ also appear in A (degrees 4 total)
    # P³Q needs P³·Q or P²Q·P or PQ²·P² etc — these are degree 3+1 or 3+... hmm
    # P³Q is degree 4 but from v⊗v with degree-2 monomials: (P²)·(PQ) = P³Q ✓
    # So (2,0)·(1,1) = P³Q — this IS covered!

    # Wait, let me recheck. I think all A monomials up to degree 4 are covered
    # by v = [1, P, Q, P², PQ, Q²] since all products have degree ≤ 4.
    # The only A monomials with individual P-degree 3 are (3,0) and (3,1).
    # (3,0) = P³: from (2,0)·(1,0) = P²·P = P³ ✓
    # (3,1) = P³Q: from (2,0)·(1,1) = P²·PQ = P³Q ✓
    # And (1,3) = PQ³: from (0,2)·(1,1) = Q²·PQ = PQ³ ✓

    pr(f"\n  All A monomials covered by degree-2 Gram? {len(missing) == 0}")

    # Count: how many v⊗v entries (free parameters)?
    n = len(v_monoms)
    n_free = n * (n + 1) // 2  # upper triangle of symmetric 6×6
    pr(f"  Gram matrix: {n}×{n} = {n_free} upper triangle entries")
    n_constraints = len(generated)  # number of distinct monomials
    n_zero_constraints = len([m for m in generated if m not in A_monoms])
    n_nonzero_constraints = len([m for m in generated if m in A_monoms])
    pr(f"  Monomial equations: {n_constraints} ({n_nonzero_constraints} non-zero + {n_zero_constraints} zero)")
    pr(f"  Free parameters: {n_free - n_constraints}")


if __name__ == '__main__':
    main()
