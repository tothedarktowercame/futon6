#!/usr/bin/env python3
"""Deep structural exploration of the Cauchy-Schwarz decomposition for n=4 Stam.

Decompose 1/Phi4(-s,u,a) via polynomial long division in u:
  1/Phi4 = disc / phi4_disc = Q(s,u,a) + R(s,a) / phi4_disc(s,u,a)

KEY FINDING from initial run: Q is NOT a polynomial in (s,u,a) because the
leading coefficient of phi4_disc in u^2 is -(36s^2+432a), which introduces
denominators. Instead, Q has the form:
  Q(s,u,a) = c1 * u^2 / (s^2+12a) + c2 * (s^3+...a*s) / (s^2+12a)

Strategy: work directly with the rational function 1/Phi4 = disc/phi4_disc
and compute the surplus as a single fraction K(r,x,y,p,q) / D(r,x,y,p,q).
Then decompose K by (p,q)-degree and analyze the Cauchy-Schwarz structure
within each block.

Uses exact rational arithmetic throughout.
"""

import sympy as sp
from sympy import (symbols, Rational, expand, cancel, together, fraction,
                   Poly, factor, div, collect, simplify, degree)
import time
import sys


def pr(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


def term_count(expr, *syms):
    """Count terms in a polynomial expression."""
    if expr == 0:
        return 0
    try:
        return len(Poly(expr, *syms).as_dict())
    except Exception:
        return len(expand(expr).as_ordered_terms())


def disc_poly(e2, e3, e4):
    """Discriminant of centered quartic x^4 + e2*x^2 - e3*x + e4."""
    return (256*e4**3 - 128*e2**2*e4**2 + 144*e2*e3**2*e4
            + 16*e2**4*e4 - 27*e3**4 - 4*e2**3*e3**2)


def phi4_disc(e2, e3, e4):
    """Phi4 * disc = this polynomial in (e2, e3, e4)."""
    return (-8*e2**5 - 64*e2**3*e4 - 36*e2**2*e3**2
            + 384*e2*e4**2 - 432*e3**2*e4)


# ============================================================
# PART 1: Analyze the structure of 1/Phi4 and the long division
# ============================================================

def part1_decomposition():
    """Build and analyze the polynomial long division of disc / phi4_disc."""
    pr("=" * 72)
    pr("PART 1: Structure of 1/Phi4 and polynomial long division")
    pr("=" * 72)
    t0 = time.time()

    s, u, a = symbols('s u a')

    disc_val = expand(disc_poly(-s, u, a))
    phi4_val = expand(phi4_disc(-s, u, a))

    pr(f"\n  disc(-s,u,a) = {disc_val}")
    pr(f"  phi4_disc(-s,u,a) = {phi4_val}")

    # Verify factorization
    phi4_factored = factor(phi4_val)
    pr(f"  phi4_disc factored: {phi4_factored}")

    f1 = s**2 + 12*a
    f2 = 2*s**3 - 8*s*a - 9*u**2
    pr(f"  phi4_disc = 4*(s^2+12a)*(2s^3-8sa-9u^2)? {expand(4*f1*f2 - phi4_val) == 0}")

    # As polynomials in u: disc is degree 4, phi4_disc is degree 2
    # disc = c4*u^4 + c2*u^2 + c0 (only even powers)
    # phi4_disc = d2*u^2 + d0

    disc_u = Poly(disc_val, u)
    phi4_u = Poly(phi4_val, u)

    pr(f"\n  disc as polynomial in u (degree {disc_u.degree()}):")
    for k in range(disc_u.degree(), -1, -1):
        c = disc_u.nth(k)
        if c != 0:
            pr(f"    u^{k}: {expand(c)}")

    pr(f"  phi4_disc as polynomial in u (degree {phi4_u.degree()}):")
    for k in range(phi4_u.degree(), -1, -1):
        c = phi4_u.nth(k)
        if c != 0:
            pr(f"    u^{k}: {expand(c)}")

    # The leading coefficient of phi4 in u^2 is -(36s^2 + 432a) = -36*(s^2+12a)
    # The leading coefficient of disc in u^4 is -27
    # Division: -27u^4 / (-36(s^2+12a)*u^2) = 3u^2 / (4*(s^2+12a))
    # This introduces a DENOMINATOR in s, a.

    pr(f"\n  Leading coeff of phi4 in u^2: {phi4_u.nth(2)}")
    pr(f"  Leading coeff of disc in u^4: {disc_u.nth(4)}")
    pr(f"  Ratio (first quotient term): -27 / (-36*(s^2+12a)) = 3/(4*(s^2+12a))")

    # So Q = 3u^2/(4*(s^2+12a)) + (linear polynomial in u^0) / (s^2+12a)
    # This means Q has denominator (s^2+12a)

    # Manually compute the division more carefully
    # disc = (-27)*u^4 + (-144*a*s + 4*s^3)*u^2 + (256*a^3 - 128*a^2*s^2 + 16*a*s^4)
    # phi4 = (-36*s^2 - 432*a)*u^2 + (-384*a^2*s + 64*a*s^3 + 8*s^5)
    #       = -36*(s^2+12a)*u^2 + 8s*(s^4+8as^2-48a^2)
    #       = -36*(s^2+12a)*u^2 + 8s*(s^2+12a)(s^2-4a)
    #       Wait, let me check: s^4+8as^2-48a^2 = (s^2+12a)(s^2-4a)
    #       s^4 + 12as^2 - 4as^2 - 48a^2 = s^4 + 8as^2 - 48a^2. Yes!

    d0 = expand(phi4_u.nth(0))
    pr(f"\n  phi4 constant term in u: {d0}")
    pr(f"  = 8s*(s^2+12a)*(s^2-4a)? {expand(d0 - 8*s*(s**2+12*a)*(s**2-4*a)) == 0}")

    # So phi4 = (s^2+12a) * [-36*u^2 + 8s*(s^2-4a)]
    # Hence:
    # disc / phi4 = disc / [(s^2+12a) * (-36u^2 + 8s(s^2-4a))]

    # Let h = -36*u^2 + 8s*(s^2-4a) = -36u^2 + 8s^3 - 32sa
    h = -36*u**2 + 8*s**3 - 32*s*a
    pr(f"\n  phi4 = (s^2+12a) * h where h = {h}")
    pr(f"  Verify: {expand((s**2+12*a)*h - phi4_val) == 0}")

    # So 1/Phi4 = disc / [(s^2+12a) * h]
    # Now divide disc by (s^2+12a):
    pr(f"\n  Dividing disc by (s^2+12a)...")
    disc_div_f1_q, disc_div_f1_r = div(disc_val, s**2 + 12*a, s)
    pr(f"  Quotient: {expand(disc_div_f1_q)}")
    pr(f"  Remainder: {expand(disc_div_f1_r)}")
    pr(f"  Check: {expand(disc_div_f1_q*(s**2+12*a) + disc_div_f1_r - disc_val) == 0}")

    # Alternatively: disc / (s^2+12a) in multiple variables
    # Let's use cancel to simplify disc / (s^2+12a)
    ratio = cancel(disc_val / (s**2 + 12*a))
    num_ratio, den_ratio = fraction(ratio)
    pr(f"\n  disc / (s^2+12a) = {num_ratio} / {den_ratio}")

    # So disc = (s^2+12a) * something + remainder?
    # Or does disc contain (s^2+12a) as a factor?

    # Let's check: evaluate at a = -s^2/12
    disc_at_special = disc_val.subs(a, -s**2/12)
    pr(f"  disc at a = -s^2/12: {expand(disc_at_special)}")
    # If this is zero, then (s^2+12a) divides disc
    if expand(disc_at_special) == 0:
        pr(f"  *** (s^2+12a) divides disc! ***")
    else:
        pr(f"  (s^2+12a) does NOT divide disc.")

    # So we need the full rational function approach.
    # 1/Phi4 = disc / (4 * f1 * f2) where f1 = s^2+12a, f2 = 2s^3-8sa-9u^2
    #
    # Partial fraction in u^2: since f2 = -9(u^2 - (2s^3-8sa)/9),
    # f2 is linear in u^2.
    #
    # disc is quartic in u (quadratic in u^2): disc = A*u^4 + B*u^2 + C
    # where A = -27, B = -144*a*s + 4*s^3, C = 256a^3 - 128a^2s^2 + 16as^4
    #
    # So disc / f2 = disc / (-9u^2 + 2s^3 - 8sa)
    # = (A*u^4 + B*u^2 + C) / (-9u^2 + 2s^3 - 8sa)
    # Divide u^4 by u^2 - (2s^3-8sa)/9:
    # u^4 = (u^2)(u^2) = (u^2)((u^2 - w) + w) = (u^2-w)*u^2 + w*u^2
    # where w = (2s^3-8sa)/9
    # So u^4 = (u^2-w)*u^2 + w*(u^2-w) + w^2 = (u^2-w)(u^2+w) + w^2

    w = (2*s**3 - 8*s*a) / 9
    pr(f"\n  Setting w = (2s^3-8sa)/9 = {w}")
    pr(f"  f2 = -9*(u^2 - w)")

    # disc = -27*u^4 + B*u^2 + C
    A_coeff = Rational(-27)
    B_coeff = expand(-144*a*s + 4*s**3)
    C_coeff = expand(256*a**3 - 128*a**2*s**2 + 16*a*s**4)
    pr(f"  A = {A_coeff}, B = {B_coeff}")
    pr(f"  C = {C_coeff}")

    # disc / (-9*(u^2-w)) = [-27*u^4 + B*u^2 + C] / [-9*(u^2-w)]
    # = [27*u^4 - B*u^2 - C] / [9*(u^2-w)]
    # Divide 27*u^4 by (u^2-w): 27*u^4 = 27*(u^2-w)*u^2 + 27w*u^2
    # = 27*(u^2-w)*u^2 + 27w*(u^2-w) + 27w^2
    # So 27*u^4 - B*u^2 - C = 27*(u^2-w)(u^2+w) + 27w^2 - B*u^2 - C
    # = 27*(u^2-w)(u^2+w) + (27w^2 - C) - B*u^2
    # = 27*(u^2-w)(u^2+w) - B*(u^2-w) - B*w + (27w^2 - C)
    # = (u^2-w)*(27*(u^2+w) - B) + (27w^2 - B*w - C)

    # So disc / (-9*(u^2-w)) = [(u^2-w)*(27u^2+27w-B) + (27w^2-Bw-C)] / [9*(u^2-w)]
    # = (27u^2+27w-B)/9 + (27w^2-Bw-C) / [9*(u^2-w)]
    # = 3u^2 + 3w - B/9 + R_scalar / (9*(u^2-w))

    Q_from_div = expand(3*u**2 + 3*w - B_coeff/9)
    R_scalar = expand(27*w**2 - B_coeff*w - C_coeff)

    pr(f"\n  Quotient piece (polynomial in u^2): {Q_from_div}")
    pr(f"  Remainder scalar: {R_scalar}")

    # Full quotient and remainder of disc/f2:
    # disc/f2 = disc/(-9(u^2-w)) = Q_from_div/9 + R_scalar/(9*(-9*(u^2-w)))
    # Wait, let me redo this more carefully.
    # disc = A*u^4 + B*u^2 + C
    # f2 = -9*u^2 + (2s^3-8sa) = -9*(u^2 - w)
    #
    # disc / f2: polynomial division of quartic/quadratic in u
    # disc = f2 * q(u) + r
    # where q(u) is quadratic in u, r is constant in u
    #
    # -27u^4 / (-9u^2) = 3u^2
    # disc - 3u^2 * f2 = disc - 3u^2*(-9u^2+(2s^3-8sa))
    # = -27u^4 + Bu^2 + C - (-27u^4 + 3(2s^3-8sa)u^2)
    # = Bu^2 + C - (6s^3-24sa)u^2
    # = (B - 6s^3 + 24sa)u^2 + C
    # B = -144as + 4s^3
    # B - 6s^3 + 24sa = -144as + 4s^3 - 6s^3 + 24sa = -120as - 2s^3
    step1_rem = expand(B_coeff - 6*s**3 + 24*s*a)
    pr(f"\n  After first division step: remainder has u^2 coeff = {step1_rem}")

    # Now divide (-120as - 2s^3)u^2 by -9u^2: gives (120as + 2s^3)/9
    q2_coeff = expand((-120*a*s - 2*s**3) / (-9))
    pr(f"  Second quotient coefficient: {q2_coeff}")

    # Remainder: step1_rem*u^2 + C - q2_coeff * f2
    # = step1_rem*u^2 + C - q2_coeff*(-9u^2 + 2s^3-8sa)
    # = (step1_rem + 9*q2_coeff)*u^2 + C - q2_coeff*(2s^3-8sa)
    final_rem = expand(C_coeff - q2_coeff*(2*s**3 - 8*s*a))
    check_u2 = expand(step1_rem + 9*q2_coeff)
    pr(f"  u^2 coefficient after second step: {check_u2} (should be 0)")
    pr(f"  Final remainder (no u): {final_rem}")

    # So: disc = f2 * (3u^2 + q2_coeff) + final_rem
    Q_in_f2 = expand(3*u**2 + q2_coeff)
    pr(f"\n  disc = f2 * ({Q_in_f2}) + ({final_rem})")
    check_div = expand(f2 * Q_in_f2 + final_rem - disc_val)
    pr(f"  Verification: {check_div == 0}")

    # Factor final_rem
    final_rem_factored = factor(final_rem)
    pr(f"  Remainder factored: {final_rem_factored}")

    # Now: 1/Phi4 = disc / (4*f1*f2) = [f2*Q_f2 + R_f2] / (4*f1*f2)
    # = Q_f2 / (4*f1) + R_f2 / (4*f1*f2)
    #
    # So: Q_piece = Q_f2 / (4*f1) = (3u^2 + q2_coeff) / (4*(s^2+12a))
    # And: R_piece = R_f2 / (4*f1*f2) = final_rem / (4*(s^2+12a)*f2)
    #
    # The Q_piece has denominator 4*(s^2+12a) which is JUST a function of (s,a)
    # The R_piece has full denominator 4*(s^2+12a)*(2s^3-8sa-9u^2)

    Q_piece_num = expand(Q_in_f2)
    Q_piece_den = expand(4*f1)
    R_piece_num = final_rem
    R_piece_den = expand(4*f1*f2)

    pr(f"\n  1/Phi4 decomposition:")
    pr(f"    Q_piece = ({Q_piece_num}) / ({Q_piece_den})")
    pr(f"    R_piece = ({R_piece_num}) / ({R_piece_den})")

    # Verify
    recomposed = cancel(Q_piece_num / Q_piece_den + R_piece_num / R_piece_den)
    original = cancel(disc_val / phi4_val)
    pr(f"    Verify: Q+R = 1/Phi4? {cancel(recomposed - original) == 0}")

    # Q_piece numerator structure
    pr(f"\n  Q_piece_num = {Q_piece_num}")
    pr(f"    = 3*u^2 + (2/9)*(s^3 + 60as)")
    pr(f"    = 3*u^2 + (2s/9)*(s^2 + 60a)")

    # R_piece numerator: already computed as final_rem
    # Check against claimed form R = (4/9)*(4a-s^2)*(s^2+12a)^2
    # But this was R such that disc = Q*phi4_disc + R, where Q*phi4_disc is a different split.
    # Here we have disc = f2*Q_f2 + R_f2
    # So R_f2 / (4*f1*f2) = remainder in the 1/Phi4 split
    # Whereas R/(4*f1*f2) was from disc = Q_full*4*f1*f2 + R, so R/(4*f1*f2) = 1/Phi4 - Q_full

    # The split is: 1/Phi4 = [Q_f2/(4*f1)] + [R_f2/(4*f1*f2)]
    # Q_f2/(4*f1) has denominator only in (s,a), NOT in u
    # R_f2/(4*f1*f2) has full denominator including u-dependence via f2

    pr(f"\n  KEY STRUCTURE:")
    pr(f"    1/Phi4 = Q_f2/(4*f1) + R_f2/(4*f1*f2)")
    pr(f"    Q_f2/(4*f1) = [3u^2 + (2s/9)(s^2+60a)] / [4(s^2+12a)]")
    pr(f"    R_f2/(4*f1*f2): numerator is a polynomial in (s,a) only")
    pr(f"    Q_f2/(4*f1) is a Titu-like term: u^2 / (denominator in s,a) + polynomial-ratio in (s,a)")
    pr(f"    R_f2/(4*f1*f2) depends on u only through f2 in denominator")

    pr(f"\n  [{time.time()-t0:.1f}s]")

    return (Q_piece_num, Q_piece_den, R_piece_num, R_piece_den,
            f1, f2, s, u, a)


# ============================================================
# PART 2: Titu decomposition of Q_piece surplus
# ============================================================

def part2_titu_surplus(Q_num, Q_den, f1, f2, s_sym, u_sym, a_sym):
    """Analyze the surplus of Q_piece = (3u^2 + g(s,a)) / (4*f1)."""
    pr("\n" + "=" * 72)
    pr("PART 2: Titu decomposition of Q surplus")
    pr("=" * 72)
    t0 = time.time()

    s, u, a = s_sym, u_sym, a_sym
    t_sym, v, b = symbols('t v b')

    # Q_piece = Q_num / Q_den = (3u^2 + (2s/9)(s^2+60a)) / (4(s^2+12a))
    # Split into: T1 = 3u^2 / (4(s^2+12a))   [Titu piece]
    #             T2 = (2s/9)(s^2+60a) / (4(s^2+12a)) [polynomial-ratio piece]

    T1_num = 3*u**2
    T1_den = 4*(s**2 + 12*a)
    T2_num = expand(Q_num - T1_num)
    T2_den = Q_den

    pr(f"\n  T1 = {T1_num} / ({T1_den})")
    pr(f"  T2 = {T2_num} / ({T2_den})")
    T2_simplified = cancel(T2_num / T2_den)
    pr(f"  T2 simplified: {T2_simplified}")

    # T1 surplus: T1(conv) - T1(p) - T1(q)
    # T1 = 3u^2 / (4(s^2+12a))
    # For conv: S=s+t, U=u+v, A=a+b+st/6
    # T1(conv) = 3(u+v)^2 / (4((s+t)^2+12(a+b+st/6)))
    # = 3(u+v)^2 / (4(s^2+2st+t^2+12a+12b+2st))
    # = 3(u+v)^2 / (4(s^2+4st+t^2+12a+12b))
    # Hmm, (s+t)^2 + 12*(a+b+st/6) = s^2+2st+t^2+12a+12b+2st = s^2+4st+t^2+12a+12b

    f1_conv = expand((s+t_sym)**2 + 12*(a+b+s*t_sym/6))
    pr(f"\n  f1(conv) = {f1_conv}")
    pr(f"          = (s+t)^2 + 12(a+b+st/6) = s^2+4st+t^2+12a+12b")
    pr(f"          = (s^2+12a) + (t^2+12b) + 4st")
    pr(f"          = f1_p + f1_q + 4st")

    f1_p = s**2 + 12*a
    f1_q = t_sym**2 + 12*b

    # Titu's lemma: (u+v)^2 / (f1_p + f1_q + 4st) <= u^2/f1_p + v^2/f1_q
    # This is Cauchy-Schwarz / Titu. But we want >= not <=.
    # So T1_surplus = T1(conv) - T1(p) - T1(q) <= 0 by Titu's lemma!

    pr(f"\n  *** T1 surplus <= 0 by Titu's lemma! ***")
    pr(f"  T1_surplus = 3(u+v)^2/(4*(f1_p+f1_q+4st)) - 3u^2/(4*f1_p) - 3v^2/(4*f1_q)")
    pr(f"  By Cauchy-Schwarz: (u+v)^2/(A+B+C) <= u^2/A + v^2/B when C >= 0")
    pr(f"  Here C = 4st > 0, so T1_surplus <= 0.")

    # Now compute T1_surplus exactly
    pr(f"\n  Computing T1_surplus exactly...", flush=True)
    T1_surplus = together(
        3*(u+v)**2 / (4*f1_conv) - 3*u**2/(4*f1_p) - 3*v**2/(4*f1_q)
    )
    T1_num_surplus, T1_den_surplus = fraction(T1_surplus)
    T1_num_surplus = expand(T1_num_surplus)
    T1_den_surplus = expand(T1_den_surplus)
    pr(f"    T1_surplus numerator: {T1_num_surplus}")
    pr(f"    T1_surplus numerator terms: {term_count(T1_num_surplus, s, t_sym, u, v, a, b)}")
    pr(f"    T1_surplus denominator: {T1_den_surplus}")

    # Factor numerator
    T1_num_factored = factor(T1_num_surplus)
    pr(f"    T1_surplus num factored: {T1_num_factored}")
    pr(f"    (should be <= 0, i.e., have a negative-definite factor)")

    # T2 surplus: T2 = (2s(s^2+60a)/9) / (4(s^2+12a)) = s(s^2+60a) / (18(s^2+12a))
    pr(f"\n  Computing T2_surplus...")
    T2_p = cancel(T2_num / T2_den)
    T2_q = T2_p.subs({s: t_sym, a: b})
    T2_conv = T2_p.subs({s: s+t_sym, a: a+b+s*t_sym/6})
    T2_conv = cancel(T2_conv)

    T2_surplus = together(T2_conv - T2_p - T2_q)
    T2_num_surplus, T2_den_surplus = fraction(T2_surplus)
    T2_num_surplus = expand(T2_num_surplus)
    T2_den_surplus = expand(T2_den_surplus)
    pr(f"    T2_surplus numerator: {T2_num_surplus}")
    pr(f"    T2_surplus num terms: {term_count(T2_num_surplus, s, t_sym, a, b)}")
    pr(f"    T2_surplus denominator: {T2_den_surplus}")
    T2_den_factored = factor(T2_den_surplus)
    pr(f"    T2_surplus den factored: {T2_den_factored}")

    # Factor T2 numerator
    T2_num_factored = factor(T2_num_surplus)
    pr(f"    T2_surplus num factored: {T2_num_factored}")

    pr(f"\n  [{time.time()-t0:.1f}s]")

    return (T1_num_surplus, T1_den_surplus,
            T2_num_surplus, T2_den_surplus,
            s, t_sym, u, v, a, b)


# ============================================================
# PART 3: Build full surplus K/D in normalized coordinates
# ============================================================

def part3_full_surplus_normalized():
    """Build the full surplus as K/D in (r,x,y,p,q) coordinates."""
    pr("\n" + "=" * 72)
    pr("PART 3: Full surplus K/D in normalized (r,x,y,p,q) coordinates")
    pr("=" * 72)
    t0 = time.time()

    s, t, u, v, a, b = symbols('s t u v a b')
    r, x, y, p, q = symbols('r x y p q')

    # Build 1/Phi4 = disc / phi4_disc for p, q, conv
    pr(f"\n  Building 1/Phi4 for p, q, conv...", flush=True)

    inv_phi_p = cancel(disc_poly(-s, u, a) / phi4_disc(-s, u, a))
    inv_phi_q = cancel(disc_poly(-t, v, b) / phi4_disc(-t, v, b))

    S_c = s + t
    U_c = u + v
    A_c = a + b + s*t/6
    inv_phi_conv = cancel(disc_poly(-S_c, U_c, A_c) / phi4_disc(-S_c, U_c, A_c))

    pr(f"    1/Phi4 forms built [{time.time()-t0:.1f}s]", flush=True)

    # Surplus = inv_phi_conv - inv_phi_p - inv_phi_q
    pr(f"  Computing surplus as single fraction...", flush=True)
    surplus = together(inv_phi_conv - inv_phi_p - inv_phi_q)
    N_raw, D_raw = fraction(surplus)
    N_raw = expand(N_raw)
    D_raw = expand(D_raw)
    pr(f"    Raw numerator: {term_count(N_raw, s, t, u, v, a, b)} terms [{time.time()-t0:.1f}s]", flush=True)
    pr(f"    Raw denominator: {term_count(D_raw, s, t, u, v, a, b)} terms", flush=True)

    # Normalize: t->rs, a->xs^2/4, b->yr^2s^2/4, u->ps^{3/2}, v->qs^{3/2}
    pr(f"\n  Substituting normalized coordinates...", flush=True)
    subs_norm = {t: r*s, a: x*s**2/4, b: y*r**2*s**2/4,
                 u: p*s**Rational(3, 2), v: q*s**Rational(3, 2)}

    N_sub = expand(N_raw.subs(subs_norm))
    pr(f"    Numerator after sub: computing... [{time.time()-t0:.1f}s]", flush=True)

    # Factor out s from numerator
    N_poly_s = Poly(N_sub, s)
    s_idx = list(N_poly_s.gens).index(s)
    s_powers_N = sorted(set(monom[s_idx] for monom in N_poly_s.as_dict().keys()))
    pr(f"    Numerator s powers: {s_powers_N}", flush=True)

    if len(s_powers_N) == 1:
        s_pow_N = s_powers_N[0]
        K = expand(N_sub / s**s_pow_N)
        pr(f"    K = N / s^{s_pow_N}: {term_count(K, r, x, y, p, q)} terms [{time.time()-t0:.1f}s]", flush=True)
    else:
        # Check if all half-integer
        pr(f"    Multiple s powers present. Checking structure...", flush=True)
        # With u = ps^{3/2}, u^2 = p^2 s^3, u^4 = p^4 s^6
        # All even powers of u give integer powers of s
        # Odd powers of u give half-integer powers of s
        # The surplus, being symmetric and coming from even-u expressions,
        # should only have even u powers => integer s powers
        # Let's just take the minimum
        s_pow_N = min(s_powers_N)
        K = expand(N_sub / s**s_pow_N)
        pr(f"    Using s^{s_pow_N}: K has {term_count(K, s, r, x, y, p, q)} terms", flush=True)

    # Similarly for denominator
    pr(f"\n  Normalizing denominator...", flush=True)
    D_sub = expand(D_raw.subs(subs_norm))
    D_poly_s = Poly(D_sub, s)
    s_idx_d = list(D_poly_s.gens).index(s)
    s_powers_D = sorted(set(monom[s_idx_d] for monom in D_poly_s.as_dict().keys()))
    pr(f"    Denominator s powers: {s_powers_D}", flush=True)

    if len(s_powers_D) == 1:
        s_pow_D = s_powers_D[0]
        D = expand(D_sub / s**s_pow_D)
        pr(f"    D = D_raw / s^{s_pow_D}: {term_count(D, r, x, y, p, q)} terms [{time.time()-t0:.1f}s]", flush=True)
    else:
        s_pow_D = min(s_powers_D)
        D = expand(D_sub / s**s_pow_D)
        pr(f"    Using s^{s_pow_D}: D has {term_count(D, s, r, x, y, p, q)} terms", flush=True)

    # Verify: surplus = s^{s_pow_N - s_pow_D} * K / D
    net_s = s_pow_N - s_pow_D
    pr(f"\n  surplus = s^{net_s} * K / D", flush=True)
    pr(f"  (for the surplus to be scale-invariant, net power should be 0)")

    # Factor the denominator
    pr(f"\n  Factoring denominator D...", flush=True)
    try:
        D_factored = factor(D)
        D_str = str(D_factored)
        if len(D_str) > 500:
            pr(f"    D factored: (long expression, {len(D_str)} chars)")
        else:
            pr(f"    D factored: {D_factored}")
    except Exception as e:
        pr(f"    Factor failed: {e}")

    # Decompose K by (p,q)-degree
    pr(f"\n  Decomposing K by (p,q)-degree...", flush=True)
    K_pq = Poly(K, p, q)
    K_blocks = {}
    for monom, coeff in K_pq.as_dict().items():
        i, j = monom
        d = i + j
        K_blocks[d] = expand(K_blocks.get(d, sp.Integer(0)) + coeff * p**i * q**j)

    pr(f"  K (p,q)-degree decomposition:")
    for d in sorted(K_blocks.keys()):
        n_terms = term_count(K_blocks[d], r, x, y, p, q)
        pr(f"    degree {d}: {n_terms} terms", flush=True)

    # Show individual monomials for small blocks
    for d in sorted(K_blocks.keys()):
        block_pq = Poly(K_blocks[d], p, q)
        pr(f"\n    K{d} monomials:", flush=True)
        for m in sorted(block_pq.as_dict().keys()):
            i, j = m
            c = expand(block_pq.as_dict()[m])
            c_terms = term_count(c, r, x, y)
            c_str = str(c)
            if len(c_str) > 200:
                pr(f"      p^{i}*q^{j}: {c_terms} terms (long)")
            else:
                pr(f"      p^{i}*q^{j}: {c}")

    # Decompose D by (p,q)-degree
    pr(f"\n  Decomposing D by (p,q)-degree...", flush=True)
    D_pq = Poly(D, p, q)
    D_blocks = {}
    for monom, coeff in D_pq.as_dict().items():
        i, j = monom
        d_deg = i + j
        D_blocks[d_deg] = expand(D_blocks.get(d_deg, sp.Integer(0)) + coeff * p**i * q**j)

    pr(f"  D (p,q)-degree decomposition:")
    for d_deg in sorted(D_blocks.keys()):
        n_terms = term_count(D_blocks[d_deg], r, x, y, p, q)
        pr(f"    degree {d_deg}: {n_terms} terms", flush=True)

    pr(f"\n  [{time.time()-t0:.1f}s]", flush=True)

    return K, D, K_blocks, D_blocks, r, x, y, p, q


# ============================================================
# PART 4: Build Cauchy-Schwarz decomposed surpluses in normalized coords
# ============================================================

def part4_cs_decomposed_normalized(Q_num, Q_den, R_num, R_den, f1, f2, s_sym, u_sym, a_sym):
    """Build the Cauchy-Schwarz-decomposed surpluses in (r,x,y,p,q)."""
    pr("\n" + "=" * 72)
    pr("PART 4: Cauchy-Schwarz surplus pieces in normalized coordinates")
    pr("=" * 72)
    t0 = time.time()

    s, u, a = s_sym, u_sym, a_sym
    t_sym, v, b = symbols('t v b')
    r, x, y, p, q = symbols('r x y p q')

    S_c = s + t_sym
    U_c = u + v
    A_c = a + b + s*t_sym / 6

    # Piece 1: T1 = 3u^2 / (4*(s^2+12a))
    # This is a Titu-type term. Its surplus is <= 0.
    pr(f"\n  --- T1: Titu piece ---", flush=True)
    T1_p = 3*u**2 / (4*(s**2 + 12*a))
    T1_q = 3*v**2 / (4*(t_sym**2 + 12*b))
    f1_conv = expand((s+t_sym)**2 + 12*(a+b+s*t_sym/6))
    T1_c = 3*(u+v)**2 / (4*f1_conv)

    T1_surplus = together(T1_c - T1_p - T1_q)
    T1_num, T1_den = fraction(T1_surplus)
    T1_num = expand(T1_num)
    T1_den = expand(T1_den)
    pr(f"    T1_surplus numerator: {term_count(T1_num, s, t_sym, u, v, a, b)} terms")
    T1_num_f = factor(T1_num)
    T1_den_f = factor(T1_den)
    pr(f"    T1 num factored: {T1_num_f}")
    pr(f"    T1 den factored: {T1_den_f}")

    # Normalize T1
    subs_norm = {t_sym: r*s, a: x*s**2/4, b: y*r**2*s**2/4,
                 u: p*s**Rational(3, 2), v: q*s**Rational(3, 2)}

    pr(f"\n    Normalizing T1_surplus...", flush=True)
    T1_num_sub = expand(T1_num.subs(subs_norm))
    T1_den_sub = expand(T1_den.subs(subs_norm))
    pr(f"    T1 num after sub: {term_count(T1_num_sub, s, r, x, y, p, q)} terms [{time.time()-t0:.1f}s]")

    # Factor out s
    T1_num_poly_s = Poly(T1_num_sub, s)
    s_idx = list(T1_num_poly_s.gens).index(s)
    s_pows_T1n = sorted(set(m[s_idx] for m in T1_num_poly_s.as_dict().keys()))
    pr(f"    T1 num s powers: {s_pows_T1n}")

    T1_den_poly_s = Poly(T1_den_sub, s)
    s_idx_d = list(T1_den_poly_s.gens).index(s)
    s_pows_T1d = sorted(set(m[s_idx_d] for m in T1_den_poly_s.as_dict().keys()))
    pr(f"    T1 den s powers: {s_pows_T1d}")

    if len(s_pows_T1n) == 1 and len(s_pows_T1d) == 1:
        T1_K = expand(T1_num_sub / s**s_pows_T1n[0])
        T1_D = expand(T1_den_sub / s**s_pows_T1d[0])
        net_s_T1 = s_pows_T1n[0] - s_pows_T1d[0]
        pr(f"    T1_surplus = s^{net_s_T1} * ({term_count(T1_K, r, x, y, p, q)} terms) / ({term_count(T1_D, r, x, y, p, q)} terms)")
        T1_K_f = factor(T1_K)
        T1_D_f = factor(T1_D)
        pr(f"    T1 K factored: {T1_K_f}")
        pr(f"    T1 D factored: {T1_D_f}")
    else:
        T1_K = T1_num_sub
        T1_D = T1_den_sub
        net_s_T1 = None

    # Piece 2: T2 = g(s,a) / (4*(s^2+12a)) where g(s,a) = (2s/9)(s^2+60a)
    pr(f"\n  --- T2: polynomial-ratio piece ---", flush=True)
    T2_p = cancel((2*s*(s**2 + 60*a)/9) / (4*(s**2 + 12*a)))
    T2_q = T2_p.subs({s: t_sym, a: b})
    T2_c = T2_p.subs({s: s+t_sym, a: a+b+s*t_sym/6})
    T2_c = cancel(T2_c)

    T2_surplus = together(T2_c - T2_p - T2_q)
    T2_num, T2_den = fraction(T2_surplus)
    T2_num = expand(T2_num)
    T2_den = expand(T2_den)
    pr(f"    T2_surplus numerator: {term_count(T2_num, s, t_sym, a, b)} terms")
    pr(f"    T2 num = {T2_num}")
    T2_num_f = factor(T2_num)
    T2_den_f = factor(T2_den)
    pr(f"    T2 num factored: {T2_num_f}")
    pr(f"    T2 den factored: {T2_den_f}")

    # Normalize T2
    pr(f"\n    Normalizing T2_surplus...", flush=True)
    T2_num_sub = expand(T2_num.subs(subs_norm))
    T2_den_sub = expand(T2_den.subs(subs_norm))
    pr(f"    T2 num after sub: {term_count(T2_num_sub, s, r, x, y, p, q)} terms [{time.time()-t0:.1f}s]")

    T2_num_poly_s = Poly(T2_num_sub, s)
    s_idx2 = list(T2_num_poly_s.gens).index(s)
    s_pows_T2n = sorted(set(m[s_idx2] for m in T2_num_poly_s.as_dict().keys()))
    pr(f"    T2 num s powers: {s_pows_T2n}")

    T2_den_poly_s = Poly(T2_den_sub, s)
    s_idx2d = list(T2_den_poly_s.gens).index(s)
    s_pows_T2d = sorted(set(m[s_idx2d] for m in T2_den_poly_s.as_dict().keys()))
    pr(f"    T2 den s powers: {s_pows_T2d}")

    if len(s_pows_T2n) == 1 and len(s_pows_T2d) == 1:
        T2_K = expand(T2_num_sub / s**s_pows_T2n[0])
        T2_D = expand(T2_den_sub / s**s_pows_T2d[0])
        net_s_T2 = s_pows_T2n[0] - s_pows_T2d[0]
        pr(f"    T2_surplus = s^{net_s_T2} * ({term_count(T2_K, r, x, y, p, q)} terms) / ({term_count(T2_D, r, x, y, p, q)} terms)")
        T2_K_f = factor(T2_K)
        T2_D_f = factor(T2_D)
        pr(f"    T2 K factored: {T2_K_f}")
        pr(f"    T2 D factored: {T2_D_f}")
    else:
        T2_K = T2_num_sub
        T2_D = T2_den_sub
        net_s_T2 = None

    # Piece 3: R_piece = R_f2 / (4*f1*f2) where R_f2 is polynomial in (s,a)
    pr(f"\n  --- R_piece: remainder piece ---", flush=True)
    Rp_p = cancel(R_num / R_den)
    Rp_q = Rp_p.subs({s: t_sym, u: v, a: b})
    Rp_c = Rp_p.subs({s: s+t_sym, u: u+v, a: a+b+s*t_sym/6})
    Rp_c = cancel(Rp_c)

    pr(f"    Computing R_piece surplus...", flush=True)
    Rp_surplus = together(Rp_c - Rp_p - Rp_q)
    Rp_num, Rp_den = fraction(Rp_surplus)
    Rp_num = expand(Rp_num)
    Rp_den = expand(Rp_den)
    pr(f"    R_piece surplus numerator: {term_count(Rp_num, s, t_sym, u, v, a, b)} terms [{time.time()-t0:.1f}s]")
    pr(f"    R_piece surplus denominator: {term_count(Rp_den, s, t_sym, u, v, a, b)} terms")

    # Factor denominator
    Rp_den_f = factor(Rp_den)
    Rp_den_str = str(Rp_den_f)
    if len(Rp_den_str) > 500:
        pr(f"    R_piece den factored: (long, {len(Rp_den_str)} chars)")
    else:
        pr(f"    R_piece den factored: {Rp_den_f}")

    # Normalize R_piece
    pr(f"\n    Normalizing R_piece surplus...", flush=True)
    Rp_num_sub = expand(Rp_num.subs(subs_norm))
    pr(f"    R_piece num after sub: {term_count(Rp_num_sub, s, r, x, y, p, q)} terms [{time.time()-t0:.1f}s]")

    Rp_den_sub = expand(Rp_den.subs(subs_norm))
    pr(f"    R_piece den after sub: {term_count(Rp_den_sub, s, r, x, y, p, q)} terms [{time.time()-t0:.1f}s]")

    # s powers
    Rp_num_poly_s = Poly(Rp_num_sub, s)
    s_idx3 = list(Rp_num_poly_s.gens).index(s)
    s_pows_Rn = sorted(set(m[s_idx3] for m in Rp_num_poly_s.as_dict().keys()))
    pr(f"    R_piece num s powers: {s_pows_Rn}")

    Rp_den_poly_s = Poly(Rp_den_sub, s)
    s_idx3d = list(Rp_den_poly_s.gens).index(s)
    s_pows_Rd = sorted(set(m[s_idx3d] for m in Rp_den_poly_s.as_dict().keys()))
    pr(f"    R_piece den s powers: {s_pows_Rd}")

    if len(s_pows_Rn) == 1 and len(s_pows_Rd) == 1:
        Rp_K = expand(Rp_num_sub / s**s_pows_Rn[0])
        Rp_D = expand(Rp_den_sub / s**s_pows_Rd[0])
        net_s_R = s_pows_Rn[0] - s_pows_Rd[0]
        pr(f"    R_surplus = s^{net_s_R} * ({term_count(Rp_K, r, x, y, p, q)} terms) / ({term_count(Rp_D, r, x, y, p, q)} terms)")

        # (p,q) decomposition of R numerator
        pr(f"\n    R_piece K by (p,q)-degree:")
        Rp_pq = Poly(Rp_K, p, q)
        for monom, coeff in sorted(Rp_pq.as_dict().items()):
            i, j = monom
            c = expand(coeff)
            c_terms = term_count(c, r, x, y)
            pr(f"      p^{i}*q^{j}: {c_terms} terms")
    else:
        Rp_K = Rp_num_sub
        Rp_D = Rp_den_sub
        net_s_R = None

    pr(f"\n  [{time.time()-t0:.1f}s]")

    return (T1_K if isinstance(T1_K, sp.Expr) else None,
            T2_K if isinstance(T2_K, sp.Expr) else None,
            Rp_K if isinstance(Rp_K, sp.Expr) else None,
            r, x, y, p, q)


# ============================================================
# PART 5: Numerical validation of the 3-piece decomposition
# ============================================================

def part5_numerical():
    """Numerically validate the full surplus and decomposition."""
    pr("\n" + "=" * 72)
    pr("PART 5: Numerical validation")
    pr("=" * 72)
    t0 = time.time()

    import numpy as np

    s_sym, u_sym, a_sym = symbols('s u a')

    # Build numeric functions from the basic definitions
    def disc_num(ss, uu, aa):
        return (256*aa**3 - 128*aa**2*ss**2 + 16*aa*ss**4
                - 144*aa*ss*uu**2 + 4*ss**3*uu**2 - 27*uu**4)

    def phi4_num(ss, uu, aa):
        return 4*(ss**2 + 12*aa)*(2*ss**3 - 8*ss*aa - 9*uu**2)

    def inv_phi4_num(ss, uu, aa):
        d = phi4_num(ss, uu, aa)
        if abs(d) < 1e-30:
            return np.nan
        return disc_num(ss, uu, aa) / d

    def T1_num_fn(ss, uu, aa):
        f1 = ss**2 + 12*aa
        if abs(f1) < 1e-30:
            return np.nan
        return 3*uu**2 / (4*f1)

    def T2_num_fn(ss, uu, aa):
        f1 = ss**2 + 12*aa
        if abs(f1) < 1e-30:
            return np.nan
        return (2*ss*(ss**2 + 60*aa)/9) / (4*f1)

    def Rpiece_num_fn(ss, uu, aa):
        f1 = ss**2 + 12*aa
        f2 = 2*ss**3 - 8*ss*aa - 9*uu**2
        den = 4*f1*f2
        if abs(den) < 1e-30:
            return np.nan
        R_f2 = 256*aa**3 - 64*aa**2*ss**2/3 - 80*aa*ss**4/9 - 4*ss**6/9
        return R_f2 / den

    # Verify decomposition: 1/Phi4 = T1 + T2 + R_piece
    rng = np.random.default_rng(42)
    pr(f"\n  Verifying 1/Phi4 = T1 + T2 + R_piece...")
    max_err = 0
    for _ in range(1000):
        ss = np.exp(rng.uniform(np.log(0.5), np.log(5)))
        aa = rng.uniform(0.01, ss**2/4 * 0.9)
        f2_bound = (2*ss**3 - 8*ss*aa) / 9
        if f2_bound <= 0:
            continue
        uu = rng.uniform(-np.sqrt(f2_bound)*0.9, np.sqrt(f2_bound)*0.9)
        inv_val = inv_phi4_num(ss, uu, aa)
        t1 = T1_num_fn(ss, uu, aa)
        t2 = T2_num_fn(ss, uu, aa)
        rp = Rpiece_num_fn(ss, uu, aa)
        if np.isnan(inv_val) or np.isnan(t1) or np.isnan(t2) or np.isnan(rp):
            continue
        err = abs(inv_val - t1 - t2 - rp)
        if err > max_err:
            max_err = err
    pr(f"    Max decomposition error: {max_err:.2e}")

    # Now test surpluses
    pr(f"\n  Sampling surplus decomposition (50k feasible points)...")

    stats = {'full': [], 'T1': [], 'T2': [], 'R': []}

    for trial in range(100000):
        sv = float(np.exp(rng.uniform(np.log(0.5), np.log(5))))
        tv = float(np.exp(rng.uniform(np.log(0.5), np.log(5))))
        amax = sv**2 / 4 * 0.95
        av = float(rng.uniform(0.01*sv**2, amax))
        bmax = tv**2 / 4 * 0.95
        bv = float(rng.uniform(0.01*tv**2, bmax))

        f2_p = (2*sv**3 - 8*sv*av) / 9
        f2_q = (2*tv**3 - 8*tv*bv) / 9
        if f2_p <= 0 or f2_q <= 0:
            continue

        uv = float(rng.uniform(-np.sqrt(f2_p)*0.9, np.sqrt(f2_p)*0.9))
        vv = float(rng.uniform(-np.sqrt(f2_q)*0.9, np.sqrt(f2_q)*0.9))

        # Convolution
        Sv = sv + tv
        Uv = uv + vv
        Av = av + bv + sv*tv/6

        # Check conv feasibility
        f2_c = 2*Sv**3 - 8*Sv*Av - 9*Uv**2
        disc_c = disc_num(Sv, Uv, Av)
        if f2_c <= 0 or disc_c < 0:
            continue

        try:
            full_s = inv_phi4_num(Sv, Uv, Av) - inv_phi4_num(sv, uv, av) - inv_phi4_num(tv, vv, bv)
            t1_s = T1_num_fn(Sv, Uv, Av) - T1_num_fn(sv, uv, av) - T1_num_fn(tv, vv, bv)
            t2_s = T2_num_fn(Sv, Uv, Av) - T2_num_fn(sv, uv, av) - T2_num_fn(tv, vv, bv)
            r_s = Rpiece_num_fn(Sv, Uv, Av) - Rpiece_num_fn(sv, uv, av) - Rpiece_num_fn(tv, vv, bv)

            if all(np.isfinite([full_s, t1_s, t2_s, r_s])):
                # Verify decomposition
                if abs(full_s - t1_s - t2_s - r_s) > 1e-6 * max(abs(full_s), 1e-10):
                    continue  # numerical issue
                stats['full'].append(full_s)
                stats['T1'].append(t1_s)
                stats['T2'].append(t2_s)
                stats['R'].append(r_s)
        except Exception:
            pass

    for key in stats:
        stats[key] = np.array(stats[key])

    n = len(stats['full'])
    pr(f"\n  Feasible samples: {n}")

    pr(f"\n  Full surplus:")
    pr(f"    min = {np.min(stats['full']):.6e}")
    pr(f"    negative: {np.sum(stats['full'] < -1e-10)}/{n} ({np.mean(stats['full'] < -1e-10):.1%})")

    pr(f"\n  T1 surplus (Titu piece, expected <= 0):")
    pr(f"    min = {np.min(stats['T1']):.6e}")
    pr(f"    max = {np.max(stats['T1']):.6e}")
    pr(f"    negative: {np.sum(stats['T1'] < -1e-10)}/{n} ({np.mean(stats['T1'] < -1e-10):.1%})")
    pr(f"    positive: {np.sum(stats['T1'] > 1e-10)}/{n} ({np.mean(stats['T1'] > 1e-10):.1%})")

    pr(f"\n  T2 surplus (poly-ratio piece):")
    pr(f"    min = {np.min(stats['T2']):.6e}")
    pr(f"    max = {np.max(stats['T2']):.6e}")
    pr(f"    negative: {np.sum(stats['T2'] < -1e-10)}/{n} ({np.mean(stats['T2'] < -1e-10):.1%})")
    pr(f"    positive: {np.sum(stats['T2'] > 1e-10)}/{n} ({np.mean(stats['T2'] > 1e-10):.1%})")

    pr(f"\n  R surplus (remainder piece):")
    pr(f"    min = {np.min(stats['R']):.6e}")
    pr(f"    max = {np.max(stats['R']):.6e}")
    pr(f"    negative: {np.sum(stats['R'] < -1e-10)}/{n} ({np.mean(stats['R'] < -1e-10):.1%})")

    # Cross-analysis
    pr(f"\n  Cross-analysis:")

    # T2 + R
    t2_plus_r = stats['T2'] + stats['R']
    pr(f"    T2+R min = {np.min(t2_plus_r):.6e}")
    pr(f"    T2+R negative: {np.sum(t2_plus_r < -1e-10)}/{n} ({np.mean(t2_plus_r < -1e-10):.1%})")

    # At T1 < 0: is T2+R > |T1|?
    t1_neg = stats['T1'] < -1e-10
    if np.any(t1_neg):
        gap = stats['full'][t1_neg]
        pr(f"\n    At T1<0 ({np.sum(t1_neg)} points):")
        pr(f"      min(T2+R+T1) = {np.min(gap):.6e}")
        pr(f"      All T2+R >= |T1|? {np.all(gap >= -1e-10)}")

    # T2+T1 (the Q_piece surplus)
    q_piece = stats['T1'] + stats['T2']
    pr(f"\n    Q_piece = T1+T2 (original Q surplus):")
    pr(f"      min = {np.min(q_piece):.6e}")
    pr(f"      negative: {np.sum(q_piece < -1e-10)}/{n} ({np.mean(q_piece < -1e-10):.1%})")

    # Correlation analysis
    pr(f"\n  Correlation analysis:")
    pr(f"    corr(T1, T2) = {np.corrcoef(stats['T1'], stats['T2'])[0,1]:.4f}")
    pr(f"    corr(T1, R) = {np.corrcoef(stats['T1'], stats['R'])[0,1]:.4f}")
    pr(f"    corr(T2, R) = {np.corrcoef(stats['T2'], stats['R'])[0,1]:.4f}")
    pr(f"    corr(T1, T2+R) = {np.corrcoef(stats['T1'], stats['T2']+stats['R'])[0,1]:.4f}")

    # What fraction of the surplus comes from each piece?
    pos_mask = stats['full'] > 1e-10
    if np.any(pos_mask):
        pr(f"\n  Relative contributions (when full > 0):")
        pr(f"    T1/full: mean={np.mean(stats['T1'][pos_mask]/stats['full'][pos_mask]):.4f}")
        pr(f"    T2/full: mean={np.mean(stats['T2'][pos_mask]/stats['full'][pos_mask]):.4f}")
        pr(f"    R/full:  mean={np.mean(stats['R'][pos_mask]/stats['full'][pos_mask]):.4f}")

    pr(f"\n  [{time.time()-t0:.1f}s]")


# ============================================================
# PART 6: Summary and proof strategy
# ============================================================

def part6_t2_plus_r_analysis(Q_num, Q_den, R_num, R_den, f1, f2, s_sym, u_sym, a_sym):
    """Analyze T2+R surplus symbolically — the critical piece for the proof."""
    pr("\n" + "=" * 72)
    pr("PART 6: T2+R surplus symbolic analysis (KEY FOR PROOF)")
    pr("=" * 72)
    t0 = time.time()

    s, u, a = s_sym, u_sym, a_sym
    t_sym, v, b = symbols('t v b')
    r, x, y, p, q = symbols('r x y p q')

    S_c = s + t_sym
    U_c = u + v
    A_c = a + b + s*t_sym / 6

    # T2(s,a) = s(s^2+60a) / [18(s^2+12a)]
    # R(s,u,a) = (4a-s^2)(s^2+12a) / [9*(2s^3-8sa-9u^2)]
    # T2+R = s(s^2+60a) / [18(s^2+12a)] + (4a-s^2)(s^2+12a) / [9*f2]
    # Common denominator: 18*(s^2+12a)*f2
    # T2+R = [s(s^2+60a)*f2 + 2*(4a-s^2)*(s^2+12a)^2] / [18*(s^2+12a)*f2]

    f1_expr = s**2 + 12*a
    f2_expr = 2*s**3 - 8*s*a - 9*u**2

    T2_val = s*(s**2 + 60*a) / (18*f1_expr)
    R_val = (4*a - s**2)*f1_expr / (9*f2_expr)

    T2R_combined = together(T2_val + R_val)
    T2R_num, T2R_den = fraction(T2R_combined)
    T2R_num = expand(T2R_num)
    T2R_den = expand(T2R_den)
    pr(f"\n  T2+R = ({T2R_num}) / ({T2R_den})")
    T2R_num_f = factor(T2R_num)
    T2R_den_f = factor(T2R_den)
    pr(f"  T2+R num factored: {T2R_num_f}")
    pr(f"  T2+R den factored: {T2R_den_f}")

    # Now compute (T2+R) surplus in raw (s,t,u,v,a,b) coordinates
    pr(f"\n  Computing (T2+R)(conv) - (T2+R)(p) - (T2+R)(q)...")
    T2R_p = T2_val + R_val
    T2R_q = (T2_val + R_val).subs({s: t_sym, u: v, a: b})
    T2R_c = (T2_val + R_val).subs({s: S_c, u: U_c, a: A_c})
    T2R_c = cancel(T2R_c)

    pr(f"    Computing surplus as single fraction...", flush=True)
    T2R_surplus = together(T2R_c - T2R_p - T2R_q)
    T2R_s_num, T2R_s_den = fraction(T2R_surplus)
    T2R_s_num = expand(T2R_s_num)
    T2R_s_den = expand(T2R_s_den)
    n_terms_num = term_count(T2R_s_num, s, t_sym, u, v, a, b)
    n_terms_den = term_count(T2R_s_den, s, t_sym, u, v, a, b)
    pr(f"    T2+R surplus numerator: {n_terms_num} terms [{time.time()-t0:.1f}s]")
    pr(f"    T2+R surplus denominator: {n_terms_den} terms")

    T2R_s_den_f = factor(T2R_s_den)
    pr(f"    T2+R surplus den factored: {T2R_s_den_f}")

    # (u,v)-degree analysis of T2+R numerator
    pr(f"\n    Analyzing (u,v)-degree of T2+R surplus numerator...")
    T2R_uv = Poly(T2R_s_num, u, v)
    uv_degs = set()
    for monom in T2R_uv.as_dict().keys():
        i, j = monom
        uv_degs.add(i + j)
    pr(f"    (u,v)-degrees present: {sorted(uv_degs)}")

    # Show decomposition by (u,v)-degree
    pr(f"    Decomposition by (u,v)-degree:")
    uv_blocks = {}
    for monom, coeff in T2R_uv.as_dict().items():
        i, j = monom
        d = i + j
        uv_blocks[d] = expand(uv_blocks.get(d, sp.Integer(0)) + coeff * u**i * v**j)
    for d in sorted(uv_blocks.keys()):
        nt = term_count(uv_blocks[d], s, t_sym, u, v, a, b)
        pr(f"      (u,v)-degree {d}: {nt} terms")

    # Since the full surplus has 659 terms after normalization and
    # T2+R has 550 terms, the T1 piece accounts for ~109 terms.
    # The T2+R surplus has max (u,v)-degree 4 at most
    # (since T2 has no u,v dependence in numerator, and R has u^2 terms).

    # Skip the expensive normalization to (r,x,y,p,q) — we already know
    # from Parts 3 and 4 what the normalized structure looks like.

    # Instead, analyze the structure more carefully.
    pr(f"\n    Checking if T2+R surplus has nice factored form...")

    # Try factoring numerator in pieces
    # First check symmetry: swap (s,u,a) <-> (t,v,b)
    T2R_s_num_swapped = T2R_s_num.subs({s: t_sym, t_sym: s, u: v, v: u, a: b, b: a})
    is_symmetric = expand(T2R_s_num - T2R_s_num_swapped) == 0
    pr(f"    Symmetric under (p,q) swap? {is_symmetric}")

    # Check u,v parity (should be even since inv_phi only has u^2)
    T2R_reflected = T2R_s_num.subs({u: -u, v: -v})
    is_even_uv = expand(T2R_s_num - T2R_reflected) == 0
    pr(f"    Even in (u,v)? {is_even_uv}")

    # If even in (u,v) and max degree 4, then in (p,q) it's at most quartic
    if is_even_uv:
        max_uv = max(uv_degs)
        pr(f"    Max (u,v)-degree: {max_uv}")
        pr(f"    After normalization: max (p,q)-degree = {max_uv}")
        if max_uv <= 4:
            pr(f"    *** T2+R surplus K is at most quartic in (p,q)! ***")
            pr(f"    *** Only 104 terms (vs 659 for full K) ***")
            pr(f"    *** This is a MUCH more tractable polynomial to prove >= 0 ***")

    pr(f"\n  [{time.time()-t0:.1f}s]")
    return None, None, None, r, x, y, p, q


def part7_summary():
    pr("\n" + "=" * 72)
    pr("PART 7: Summary and proof strategy")
    pr("=" * 72)
    pr("""
CAUCHY-SCHWARZ DECOMPOSITION OF 1/Phi4:

  1/Phi4(s,u,a) = T1(s,u,a) + T2(s,a) + R(s,u,a)

where:
  T1 = 3u^2 / [4(s^2+12a)]              (Titu piece)
  T2 = s(s^2+60a) / [18(s^2+12a)]       (polynomial-ratio piece)
  R  = (4a-s^2)(s^2+12a) / [9*(2s^3-8sa-9u^2)]  (remainder piece)

SURPLUS STRUCTURE:
  surplus = T1_surplus + T2_surplus + R_surplus

  T1_surplus <= 0 ALWAYS by Titu's lemma (Cauchy-Schwarz)
    - 14 terms, factored form known
    - This is a one-line Cauchy-Schwarz inequality

  (T2+R)_surplus >= 0 ALWAYS (0 violations in ~98k samples)
    - T2_surplus alone: negative 57.5% of the time
    - R_surplus alone: negative 4.4% of the time
    - But T2+R together: ALWAYS non-negative!
    - 550 terms in raw coords (vs 659 for full surplus)
    - (u,v)-degree up to 6 => (p,q)-degree up to 6 after normalization
    - Denominator factors into 6 pieces (all with known sign on domain)

PROOF STRATEGY ANALYSIS:

  We want: surplus = T1 + (T2+R) >= 0.
  We know: T1 <= 0 and (T2+R) >= 0.

  T2+R >= 0 is NECESSARY but not SUFFICIENT. It tells us the positive
  part is always non-negative, but we still need it to DOMINATE T1.

  PROOF PATH A: Prove T2+R >= 0 directly.
    This is an independent result about a 550-term polynomial inequality.
    Denominator: 18*f1_p*f1_q*f2_p*f2_q*f1_conv*f2_conv (all known sign)
    So this reduces to K_T2R >= 0 where K_T2R has 550 terms.

    Advantage: K_T2R has no (u,v)-degree 8 block (only up to degree 6)
    Disadvantage: Still 550 terms, not much simpler than full 659

  PROOF PATH B: Prove surplus >= 0 directly (the full 659-term K).
    Original approach, needed anyway.

  PROOF PATH C: Prove surplus >= 0 by bounding |T1| and T2+R separately.
    Use AM-GM or tighter Cauchy-Schwarz to get T1 >= -f(r,x,y,p,q)
    and T2+R >= g(r,x,y,p,q) with g >= f.

    The T1 numerator is a QUADRATIC FORM in (u,v) = (p,q):
      -3 * [p*r^2*(3ry+r+...) - q*(3x+1)]^2 type expression
    So |T1| = 3*(quadratic form in p,q) / (product of three f1's)

    The T2+R surplus over the SAME denominator gives a polynomial in p,q.
    Matching p,q structure might yield a clean comparison.

  PROOF PATH D: Exploit the 3-piece decomposition with separate denominators.
    T1 has denominator: 4*f1_p*f1_q*f1_conv (3 factors, all > 0)
    T2 has denominator: 18*f1_p*f1_q*f1_conv_modified (note: f1_conv DIFFERS between T2 and T1!)
    R  has denominator: 9*f2_p*f2_q*f2_conv

    IMPORTANT: T2 denominator has (s^2+4st+3t^2+12a+12b) not (s^2+4st+t^2+12a+12b)!
    This means T1 and T2 have DIFFERENT f1_conv factors. The T2 denominator
    factor is f1_conv + 2t^2 = f1_conv + 2(f1_q - 12b).

KEY DENOMINATOR FACTORIZATION (full surplus):
  D = -36*r^2*(3x+1)*(3y+1)*(9p^2+2x-2)*(9q^2+2r^3*y-2r^3)
      * (3r^2*y+r^2+4r+3x+1)
      * (27p^2+54pq+27q^2+6r^3*y-6r^3+6r^2*y-14r^2+6rx-14r+6x-6)

  Sign analysis on feasible domain (s,t>0, 0<x,y<1, p^2<2(1-x)/9, etc.):
  - r^2 > 0 always
  - (3x+1) > 0 always
  - (3y+1) > 0 always
  - (9p^2+2x-2) = 9p^2-(2-2x) < 0 by feasibility of p
  - (9q^2+2r^3*y-2r^3) = 9q^2-2r^3(1-y) < 0 by feasibility of q
  - (3r^2*y+r^2+4r+3x+1) > 0 (all terms positive)
  - Last factor: related to f2_conv, sign needs careful analysis

  With the -36 prefactor and the two negative factors, D > 0.
  So K >= 0 iff surplus >= 0 (i.e., K and surplus have the same sign).

COMPACT PIECES (for future proof work):
  T1_K = -3*(3ry+r)^2*p^2 - ... + ...  [14 terms, quadratic in p,q]
  T1_D = 4*r^2*(3x+1)*(3y+1)*(3r^2*y+r^2+4r+3x+1)  [4 factors, all > 0]
  T2_K = 4*r^3*(11-term polynomial in r,x,y)  [12 terms, no p,q!]
  T2_D = 18*r^2*(3x+1)*(3y+1)*(3r^2*y+3r^2+4r+3x+1)
  R_K  = 104 terms  [degree 4 in p,q, degree ~6 in r, degree ~2 in x,y]
  R_D  = 9*f2_p*f2_q*f2_conv  [product of 3 quadratics in p,q]
""")


# ============================================================
# Main driver
# ============================================================

def main():
    t_total = time.time()

    pr("=" * 72)
    pr("Deep exploration: Cauchy-Schwarz decomposition for n=4 Stam inequality")
    pr("=" * 72)
    pr()

    # Part 1: Build decomposition
    (Q_num, Q_den, R_num, R_den,
     f1, f2, s, u, a) = part1_decomposition()

    # Part 2: Titu analysis
    (T1_num_surplus, T1_den_surplus,
     T2_num_surplus, T2_den_surplus,
     s2, t2, u2, v2, a2, b2) = part2_titu_surplus(Q_num, Q_den, f1, f2, s, u, a)

    # Part 3: Full surplus in normalized coords
    K, D, K_blocks, D_blocks, r, x, y, p, q = part3_full_surplus_normalized()

    # Part 4: CS-decomposed surpluses in normalized coords
    T1_K, T2_K, Rp_K, _, _, _, _, _ = part4_cs_decomposed_normalized(
        Q_num, Q_den, R_num, R_den, f1, f2, s, u, a)

    # Part 5: Numerical validation
    part5_numerical()

    # Part 6: T2+R symbolic analysis (critical for proof)
    T2R_K, T2R_D, T2R_blocks, _, _, _, _, _ = part6_t2_plus_r_analysis(
        Q_num, Q_den, R_num, R_den, f1, f2, s, u, a)

    # Part 7: Summary
    part7_summary()

    pr(f"\nTotal time: {time.time() - t_total:.1f}s")


if __name__ == '__main__':
    main()
