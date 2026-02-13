#!/usr/bin/env python3
"""Approach C: Cauchy-Schwarz decomposition of the n=4 surplus.

Decompose 1/Φ₄(s,u,a) = T₁(s,u,a) + T₂(s,u,a) where:
  T₁ = 3u²/[4(s²+12a)]  (the "Titu-like" piece)
  T₂ = remainder          (the "symmetric-like" piece)

Then surplus = Δ₁ + Δ₂ where Δ₁ ≤ 0 (anti-Titu) and Δ₂ = ?

Key question: Is Δ₂ ≥ |Δ₁| everywhere on the feasible domain?
If so, we get surplus ≥ 0 from Δ₂ ≥ Δ₁.

Also: explore alternative decompositions that might give both pieces non-negative.
"""

import numpy as np
import sympy as sp
from sympy import Rational, cancel, expand, together, fraction, Poly


def disc_poly(e2, e3, e4):
    return (256*e4**3 - 128*e2**2*e4**2 + 144*e2*e3**2*e4
            + 16*e2**4*e4 - 27*e3**4 - 4*e2**3*e3**2)


def phi4_disc(e2, e3, e4):
    return (-8*e2**5 - 64*e2**3*e4 - 36*e2**2*e3**2
            + 384*e2*e4**2 - 432*e3**2*e4)


def main():
    print("Building 1/Φ₄ decomposition...")
    e2, e3, e4 = sp.symbols('e2 e3 e4')
    s, t = sp.symbols('s t', positive=True)
    u, v = sp.symbols('u v', real=True)
    a, b = sp.symbols('a b', positive=True)

    # 1/Φ₄ = disc / phi4_disc
    inv_phi = cancel(disc_poly(e2, e3, e4) / phi4_disc(e2, e3, e4))

    # For polynomial p: e2=-s, e3=u, e4=a
    inv_p = inv_phi.subs({e2: -s, e3: u, e4: a})
    inv_p = cancel(inv_p)
    print(f"  1/Φ₄(p) = {inv_p}")

    # Decompose as partial fractions in u²
    # 1/Φ₄ = N(s,u²,a) / [4·f1·f2]
    # where f1 = s²+12a, f2 = 2s³-8sa+9u²
    # Wait, need to check the sign: phi4_disc = -4·f1·f2 or +4·f1·f2?

    # Let me compute f1, f2 from the formula
    # phi4_disc(-s, u, a) = -8(-s)^5 - 64(-s)^3·a - 36(-s)^2·u² + 384(-s)·a² - 432·u²·a
    # = 8s^5 + 64s^3·a - 36s^2·u^2 - 384s·a^2 - 432u^2·a
    phi4 = expand(phi4_disc(-s, u, a))
    print(f"\n  Φ₄·disc(-s,u,a) = {phi4}")

    # Factor:
    f1 = s**2 + 12*a
    f2 = 2*s**3 - 8*s*a + 9*u**2  # NEED TO CHECK SIGN
    print(f"  f1 = {f1}")
    print(f"  f2_candidate = {f2}")
    check = expand(-4*f1*f2)
    print(f"  -4*f1*f2 = {check}")
    print(f"  phi4_disc(-s,u,a) = {phi4}")
    print(f"  Match? {expand(check - phi4) == 0}")

    # Try different sign
    f2b = 2*s**3 - 8*s*a - 9*u**2
    check2 = expand(-4*f1*f2b)
    print(f"  -4*f1*(2s³-8sa-9u²) = {check2}")
    print(f"  Match? {expand(check2 - phi4) == 0}")

    # Hmm, let me just factor phi4 directly
    print(f"\n  Trying to factor phi4_disc(-s,u,a) in s,u,a...")
    phi4_factored = sp.factor(phi4)
    print(f"  Factored: {phi4_factored}")

    # OK let me just compute the disc and form 1/Phi4 numerically to verify
    # disc(-s, u, a) = 256a³ - 128s²a² + 144su²a + 16s⁴a - 27u⁴ - 4s³u²
    disc_val = expand(disc_poly(-s, u, a))
    print(f"\n  disc(-s,u,a) = {disc_val}")

    # So 1/Φ₄ = disc / phi4_disc
    # Let me verify the factored form
    # phi4_disc = 4s(s²+12a)(s²-4a) - 36(s²+12a)u² - ???
    # Actually let me just work with the expanded form

    # Numerator as polynomial in u:
    disc_in_u = Poly(disc_val, u)
    phi4_in_u = Poly(phi4, u)
    print(f"\n  disc as poly in u: degree {disc_in_u.degree()}, coeffs:")
    for k in range(disc_in_u.degree(), -1, -1):
        c = disc_in_u.nth(k)
        if c != 0:
            print(f"    u^{k}: {expand(c)}")

    print(f"\n  phi4 as poly in u: degree {phi4_in_u.degree()}, coeffs:")
    for k in range(phi4_in_u.degree(), -1, -1):
        c = phi4_in_u.nth(k)
        if c != 0:
            print(f"    u^{k}: {expand(c)}")

    # The denominator is degree 2 in u (linear in u²).
    # So 1/Φ₄ = (quartic in u) / (quadratic in u) = quotient + remainder/(quadratic)
    # Polynomial division in u:
    print(f"\n  Polynomial long division of disc by phi4 in u...")
    quot, rem = sp.div(disc_val, phi4, u)
    print(f"  Quotient: {expand(quot)}")
    print(f"  Remainder: {expand(rem)}")
    print(f"  Check: disc = quot*phi4 + rem? {expand(disc_val - quot*phi4 - rem) == 0}")

    # So 1/Φ₄ = quot + rem/phi4
    # The quotient is a polynomial in u (degree = deg(disc)-deg(phi4) in u = 4-2 = 2)
    # The remainder is degree < 2 in u (so degree 0 or 1)
    quot_in_u = Poly(expand(quot), u)
    rem_in_u = Poly(expand(rem), u)
    print(f"\n  Quotient degree in u: {quot_in_u.degree()}")
    print(f"  Remainder degree in u: {rem_in_u.degree()}")

    # Now the surplus of the quotient part:
    # Since 1/Φ₄ = Q(s,u,a) + R(s,u,a)/phi4(s,u,a)
    # surplus = Q_surplus + R_surplus
    # where Q_surplus = Q(S,U,A) - Q(s,u,a) - Q(t,v,b)
    # and R_surplus = R(S,U,A)/phi4(S,U,A) - R(s,u,a)/phi4(s,u,a) - R(t,v,b)/phi4(t,v,b)

    # The Q part is a polynomial — its surplus is "convexity-like"
    # The R part involves denominators — its surplus is "concavity-like"

    # Let me evaluate both parts numerically
    print(f"\n{'='*72}")
    print("Numerical evaluation of decomposition")
    print('='*72)

    Q = expand(quot)
    R = expand(rem)

    Q_fn = sp.lambdify((s, u, a), Q, 'numpy')
    R_fn = sp.lambdify((s, u, a), R, 'numpy')
    phi4_fn = sp.lambdify((s, u, a), phi4, 'numpy')

    inv_phi_fn = sp.lambdify((s, u, a), cancel(disc_val / phi4), 'numpy')

    def full_surplus(sv, uv, av, tv, vv, bv):
        S = sv + tv
        U = uv + vv
        A = av + bv + sv*tv/6
        return inv_phi_fn(S, U, A) - inv_phi_fn(sv, uv, av) - inv_phi_fn(tv, vv, bv)

    def Q_surplus(sv, uv, av, tv, vv, bv):
        S = sv + tv
        U = uv + vv
        A = av + bv + sv*tv/6
        return Q_fn(S, U, A) - Q_fn(sv, uv, av) - Q_fn(tv, vv, bv)

    def R_phi_surplus(sv, uv, av, tv, vv, bv):
        S = sv + tv
        U = uv + vv
        A = av + bv + sv*tv/6
        return (R_fn(S, U, A)/phi4_fn(S, U, A)
                - R_fn(sv, uv, av)/phi4_fn(sv, uv, av)
                - R_fn(tv, vv, bv)/phi4_fn(tv, vv, bv))

    # Sample feasible points
    rng = np.random.default_rng(42)
    results = {'full': [], 'Q': [], 'R_phi': []}

    for _ in range(50000):
        sv = float(np.exp(rng.uniform(np.log(0.5), np.log(5.0))))
        tv = float(np.exp(rng.uniform(np.log(0.5), np.log(5.0))))
        # a must satisfy disc(p) >= 0 and a > 0 and s²-4a > 0
        amax = sv**2 / 4 * 0.95
        av = float(rng.uniform(0.01 * sv**2, amax))
        bmax = tv**2 / 4 * 0.95
        bv = float(rng.uniform(0.01 * tv**2, bmax))
        # u must satisfy f2 > 0: 2s³-8sa-9u² > 0
        f2_bound_p = (2*sv**3 - 8*sv*av) / 9
        f2_bound_q = (2*tv**3 - 8*tv*bv) / 9
        if f2_bound_p <= 0 or f2_bound_q <= 0:
            continue
        umax = np.sqrt(f2_bound_p) * 0.9
        vmax = np.sqrt(f2_bound_q) * 0.9
        uv = float(rng.uniform(-umax, umax))
        vv = float(rng.uniform(-vmax, vmax))

        # Check convolution constraints
        S = sv + tv
        U = uv + vv
        A = av + bv + sv*tv/6
        disc_c = disc_poly(-S, U, A)
        disc_c_val = float(disc_c) if isinstance(disc_c, (int, float)) else float(disc_c.evalf())
        # Simpler: just check numerically
        disc_c_num = (256*A**3 - 128*S**2*A**2 + 144*S*U**2*A
                      + 16*S**4*A - 27*U**4 - 4*S**3*U**2)
        f2c = 2*S**3 - 8*S*A - 9*U**2
        if disc_c_num < 0 or f2c < 0:
            continue

        try:
            fs = full_surplus(sv, uv, av, tv, vv, bv)
            qs = Q_surplus(sv, uv, av, tv, vv, bv)
            rs = R_phi_surplus(sv, uv, av, tv, vv, bv)
            if np.isfinite(fs) and np.isfinite(qs) and np.isfinite(rs):
                results['full'].append(fs)
                results['Q'].append(qs)
                results['R_phi'].append(rs)
        except Exception:
            pass

    for key in results:
        results[key] = np.array(results[key])

    n = len(results['full'])
    print(f"  Feasible samples: {n}")

    print(f"\n  Full surplus: min={np.min(results['full']):.6e}, neg={np.sum(results['full']<-1e-10)}")
    print(f"  Q surplus:    min={np.min(results['Q']):.6e}, neg={np.sum(results['Q']<-1e-10)}")
    print(f"  R/Φ surplus:  min={np.min(results['R_phi']):.6e}, neg={np.sum(results['R_phi']<-1e-10)}")

    # Check ratios
    if np.all(results['R_phi'] > 0):
        ratios = -results['Q'] / results['R_phi']
        print(f"\n  |Q_surplus| / R_surplus: max={np.max(ratios):.4f}")
        if np.max(ratios) < 1:
            print(f"  *** R/Φ surplus DOMINATES Q surplus everywhere! ***")
            print(f"  *** Sufficient to prove R/Φ surplus ≥ |Q surplus|. ***")
    elif np.all(results['Q'] > 0):
        print(f"\n  *** Q surplus is actually POSITIVE! ***")
    else:
        # Both can be negative
        q_neg = np.sum(results['Q'] < -1e-10)
        r_neg = np.sum(results['R_phi'] < -1e-10)
        print(f"\n  Q negative: {q_neg}/{n} ({q_neg/n:.1%})")
        print(f"  R/Φ negative: {r_neg}/{n} ({r_neg/n:.1%})")

        # At points where Q < 0, is R_phi > |Q|?
        q_neg_mask = results['Q'] < -1e-10
        if np.any(q_neg_mask):
            gap = results['R_phi'][q_neg_mask] + results['Q'][q_neg_mask]
            print(f"  At Q<0: min(R_phi + Q) = {np.min(gap):.6e} (should be >=0)")
            print(f"  At Q<0: fraction with R_phi+Q>=0: {np.mean(gap>=0):.1%}")


if __name__ == '__main__':
    main()
