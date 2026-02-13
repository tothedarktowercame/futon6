#!/usr/bin/env python3
"""Prove nonnegativity of the 30-term symmetric n=4 Stam numerator.

Setting (centered, e3=0):
  p(x) = x^4 - s x^2 + a,   q(x) = x^4 - t x^2 + b
with feasible cone:
  s,t > 0,   0 < 4a < s^2,   0 < 4b < t^2.

For this case, the Stam surplus has denominator > 0 on the cone, and
numerator N(s,t,a,b) equals the 30-term polynomial from
CODEX-P4-N4-SOS-HANDOFF.md. We prove N >= 0 on the cone.

Proof strategy:
1. Normalize:
     r = t/s > 0,  x = 4a/s^2 in (0,1),  y = 4b/t^2 in (0,1).
   Then N = s^10 r^4 G(r,x,y), so sign(N) = sign(G).
2. Shift box to p,q variables:
     p = 3x - 1 in (-1,2),  q = 3y - 1 in (-1,2).
3. Show:
     G = A(p,q) r^2 + B(p,q) r + C(p,q),
   with A,B,C >= 0 on p,q in [-1,2].
4. Since r>0, G >= 0; hence N >= 0.
"""

import sympy as sp


def build_handcrafted_numerator():
    s, t, a, b = sp.symbols("s t a b", positive=True)
    N = (
        10368 * a**3 * b * t**2 + 864 * a**3 * t**4
        + 10368 * a**2 * b**2 * s**2 + 10368 * a**2 * b**2 * t**2
        + 864 * a**2 * b * s**2 * t**2 + 3456 * a**2 * b * s * t**3
        + 1728 * a**2 * b * t**4 + 288 * a**2 * s**2 * t**4
        + 576 * a**2 * s * t**5 + 72 * a**2 * t**6
        + 10368 * a * b**3 * s**2 + 1728 * a * b**2 * s**4
        + 3456 * a * b**2 * s**3 * t + 864 * a * b**2 * s**2 * t**2
        - 936 * a * b * s**4 * t**2 - 1728 * a * b * s**3 * t**3
        - 936 * a * b * s**2 * t**4 - 42 * a * s**4 * t**4
        - 24 * a * s**3 * t**5 + 18 * a * s**2 * t**6
        + 864 * b**3 * s**4 + 72 * b**2 * s**6
        + 576 * b**2 * s**5 * t + 288 * b**2 * s**4 * t**2
        + 18 * b * s**6 * t**2 - 24 * b * s**5 * t**3
        - 42 * b * s**4 * t**4 + 3 * s**6 * t**4
        + 4 * s**5 * t**5 + 3 * s**4 * t**6
    )
    return s, t, a, b, sp.expand(N)


def check_symbolic_reduction():
    s, t, a, b, N = build_handcrafted_numerator()

    r, x, y = sp.symbols("r x y", positive=True)
    subs_norm = {t: r * s, a: x * s**2 / 4, b: y * t**2 / 4}
    N_norm = sp.expand(N.subs(subs_norm))
    G = sp.expand(N_norm / (s**10 * r**4))
    assert sp.simplify(
        N - sp.expand((s**10) * (r**4) * G).subs({r: t / s, x: 4 * a / s**2, y: 4 * b / t**2})
    ) == 0

    p, q = sp.symbols("p q", real=True)
    Gpq = sp.expand(G.subs({x: (p + 1) / 3, y: (q + 1) / 3}))
    poly_r = sp.Poly(Gpq, r)
    A = sp.expand(poly_r.nth(2))
    B = sp.expand(poly_r.nth(1))
    C = sp.expand(poly_r.nth(0))

    A_expected = sp.expand(
        sp.Rational(1, 2) * p**2 * q**2 + 2 * p**2 * q + 2 * p**2
        + sp.Rational(1, 2) * p * q**3 + 3 * p * q**2 + q**3 + 6 * q**2
    )
    B_expected = sp.expand(2 * p**2 * q + 6 * p**2 + 2 * p * q**2 - 4 * p * q + 6 * q**2)
    C_expected = sp.expand(
        sp.Rational(1, 2) * p**3 * q + p**3
        + sp.Rational(1, 2) * p**2 * q**2 + 3 * p**2 * q + 6 * p**2
        + 2 * p * q**2 + 2 * q**2
    )

    assert sp.simplify(A - A_expected) == 0
    assert sp.simplify(B - B_expected) == 0
    assert sp.simplify(C - C_expected) == 0

    return {
        "N": N,
        "G": G,
        "A": A,
        "B": B,
        "C": C,
        "symbols": (s, t, a, b, r, x, y, p, q),
    }


def prove_nonnegativity_components():
    p, q = sp.symbols("p q", real=True)

    # A as quadratic in p.
    aA = sp.expand(sp.Rational(1, 2) * (q + 2) ** 2)
    bA = sp.expand(sp.Rational(1, 2) * q**2 * (q + 6))
    cA = sp.expand(q**2 * (q + 6))
    deltaA = sp.expand(bA**2 - 4 * aA * cA)
    deltaA_expected = sp.expand(sp.Rational(1, 4) * q**2 * (q + 6) * (q**3 - 2 * q**2 - 32 * q - 32))
    assert sp.simplify(deltaA - deltaA_expected) == 0

    # B as quadratic in p.
    aB = sp.expand(2 * (q + 3))
    bB = sp.expand(2 * q * (q - 2))
    cB = sp.expand(6 * q**2)
    deltaB = sp.expand(bB**2 - 4 * aB * cB)
    deltaB_expected = sp.expand(4 * q**2 * (q**2 - 16 * q - 32))
    assert sp.simplify(deltaB - deltaB_expected) == 0

    # C as quadratic in q.
    aC = sp.expand(sp.Rational(1, 2) * (p + 2) ** 2)
    bC = sp.expand(sp.Rational(1, 2) * p**2 * (p + 6))
    cC = sp.expand(p**2 * (p + 6))
    deltaC = sp.expand(bC**2 - 4 * aC * cC)
    deltaC_expected = sp.expand(sp.Rational(1, 4) * p**2 * (p + 6) * (p**3 - 2 * p**2 - 32 * p - 32))
    assert sp.simplify(deltaC - deltaC_expected) == 0

    # Interval-sign facts used in the proof.
    z = sp.symbols("z", real=True)
    h = z**3 - 2 * z**2 - 32 * z - 32
    hprime = sp.diff(h, z)
    # h' = 3z^2 - 4z - 32 has roots -8/3 and 4, so h' < 0 on [-1,2].
    assert sp.simplify(hprime.subs(z, -1)) < 0
    assert sp.simplify(hprime.subs(z, 2)) < 0
    # Therefore h is decreasing on [-1,2], and h <= h(-1) = -3 < 0.
    assert sp.simplify(h.subs(z, -1)) == -3
    assert sp.simplify(h.subs(z, 2)) == -96

    # g(z)=z^2-16z-32 has endpoint values -15 and -60 on [-1,2], so g<0 there.
    g = z**2 - 16 * z - 32
    assert sp.simplify(g.subs(z, -1)) == -15
    assert sp.simplify(g.subs(z, 2)) == -60

    return {
        "deltaA": deltaA,
        "deltaB": deltaB,
        "deltaC": deltaC,
        "h": h,
        "g": g,
    }


def numeric_spotcheck():
    """Strong random sanity check (not the proof)."""
    import numpy as np

    rng = np.random.default_rng(20260213)
    s_sym, t_sym, a_sym, b_sym, N_sym = build_handcrafted_numerator()
    N_fn = sp.lambdify((s_sym, t_sym, a_sym, b_sym), N_sym, "numpy")

    min_val = float("inf")
    worst = None

    n = 200000
    s = rng.uniform(0.3, 4.0, size=n)
    t = rng.uniform(0.3, 4.0, size=n)
    a = rng.uniform(0.0, 1.0, size=n) * (0.2499 * s * s - 1e-6) + 1e-6
    b = rng.uniform(0.0, 1.0, size=n) * (0.2499 * t * t - 1e-6) + 1e-6

    vals = N_fn(s, t, a, b)
    i = int(np.argmin(vals))
    min_val = float(vals[i])
    worst = (float(s[i]), float(t[i]), float(a[i]), float(b[i]), float(vals[i]))
    return min_val, worst


def main():
    red = check_symbolic_reduction()
    prove_nonnegativity_components()
    min_val, worst = numeric_spotcheck()

    print("=" * 78)
    print("P4 n=4 symmetric quartic: 30-term numerator certificate")
    print("=" * 78)
    print("[1] Symbolic reduction checks")
    print("  Verified N = s^10 r^4 G(r,x,y), with r=t/s, x=4a/s^2, y=4b/t^2.")
    print("  Verified G = A(p,q) r^2 + B(p,q) r + C(p,q), p=3x-1, q=3y-1.")
    print()
    print("[2] Component nonnegativity route")
    print("  A,B,C are quadratics with positive leading coefficient.")
    print("  Their discriminants are:")
    print("    DeltaA = q^2(q+6)(q^3-2q^2-32q-32)/4")
    print("    DeltaB = 4q^2(q^2-16q-32)")
    print("    DeltaC = p^2(p+6)(p^3-2p^2-32p-32)/4")
    print("  On p,q in [-1,2], each discriminant <= 0, hence A,B,C >= 0.")
    print("  Therefore G >= 0 for r>0, hence N >= 0 on the feasible cone.")
    print()
    print("[3] Numeric sanity")
    print(f"  min N over 200000 random feasible samples: {min_val:.6e}")
    print(f"  worst sample (s,t,a,b,N): {worst}")
    print()
    print("QED (symmetric quartic case e3=0): Stam surplus numerator is nonnegative.")


if __name__ == "__main__":
    main()
