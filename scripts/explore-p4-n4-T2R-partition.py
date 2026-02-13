#!/usr/bin/env python3
"""Explore the T2/R surplus partition structure.

Key finding: T2_surplus < 0 and R_surplus < 0 appear to be MUTUALLY EXCLUSIVE.
If proved, this gives: T2+R >= 0 via partition argument.

Also: explore R concavity in u² and the factored d6 structure.
"""

import numpy as np
import sympy as sp
from sympy import Rational, symbols, expand, Poly, factor, cancel, together, fraction
import time


def pr(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


def main():
    t0 = time.time()
    r, x, y, p, q = symbols('r x y p q')

    pr("="*72)
    pr("T2/R SURPLUS PARTITION ANALYSIS")
    pr("="*72)

    # ================================================================
    # SECTION 1: Symbolic T2_surplus
    # ================================================================
    pr("\n--- T2_surplus (function of r,x,y only) ---")

    # T2(s,a) = s(s^2+60a)/(18(s^2+12a))
    # T2_p: s=1, a=x/4 → (1+15x)/(18(1+3x))
    # T2_q: s=r, a=yr^2/4 → r(1+15y)/(18(1+3y))
    # T2_conv: S=1+r, A=(3x+3yr^2+2r)/12
    #   S^2+60A = (1+r)^2+5(3x+3yr^2+2r) = 1+12r+r^2+15x+15yr^2
    #   S^2+12A = (1+r)^2+3x+3yr^2+2r = 1+4r+r^2+3x+3yr^2

    N2c = (1+r)**2 + 15*x + 15*y*r**2 + 10*r  # S^2+60A
    F1c = (1+r)**2 + 3*x + 3*y*r**2 + 2*r  # S^2+12A
    T2c = (1+r)*N2c / (18*F1c)

    N2p = 1 + 15*x; F1p = 1 + 3*x
    T2p = N2p / (18*F1p)

    N2q = 1 + 15*y; F1q = 1 + 3*y
    T2q = r * N2q / (18*F1q)

    # T2_surplus numerator (after clearing positive denominators)
    T2_surp_num = expand((1+r)*N2c*F1p*F1q - N2p*F1c*F1q - r*N2q*F1c*F1p)
    n_terms = len(Poly(T2_surp_num, r, x, y).as_dict())
    pr(f"  T2_surplus_num: {n_terms} terms")

    T2_surp_factored = factor(T2_surp_num)
    pr(f"  T2_surplus_num factored: {T2_surp_factored}")

    # ================================================================
    # SECTION 2: Symbolic R_surplus
    # ================================================================
    pr("\n--- R_surplus analysis ---")

    # R(s,u,a) = (4a-s^2)(s^2+12a)/(9*f2)
    # f2 = 2s^3-8sa-9u^2
    # In normalized:
    # R_p: C_p = x-1, f1p = 1+3x, f2p = 2(1-x)-9p^2
    # R_p = (x-1)(1+3x)/(9*(2(1-x)-9p^2))
    # R_q: C_q = r^2(y-1), f1q = r^2(1+3y), f2q = 2r^3(1-y)-9q^2
    # R_q = r^4(y-1)(1+3y)/(9*(2r^3(1-y)-9q^2))

    # For partition: when is R_surplus < 0?
    # R_surplus = R_conv - R_p - R_q
    # Since R_p < 0, R_q < 0, R_conv < 0 on feasible:
    # R_surplus = (neg) - (neg) - (neg) = (neg) + |R_p| + |R_q|
    # R_surplus < 0 ⟺ |R_conv| > |R_p| + |R_q|
    # R_surplus > 0 ⟺ |R_conv| < |R_p| + |R_q| (sub-additivity of |R|)

    # |R| = (1-x)(1+3x)/(9*(2(1-x)-9p^2)) for p-polynomial
    # The larger p^2 (closer to boundary), the larger |R_p|

    # What controls whether |R_conv| < |R_p| + |R_q|?
    # Key: the "extra" a-term A_extra = st/6 in the convolution's A parameter
    # A_conv = a + b + st/6 = (actual a + b) + st/6
    # This makes |C_conv| = |4A_conv - S^2| = |4(a+b+st/6) - (s+t)^2|
    # = |4a+4b+2st/3 - s^2-2st-t^2| = |4a-s^2 + 4b-t^2 - 4st/3|
    # = |C_p + C_q - 4st/3|
    # Since C_p < 0, C_q < 0, and -4st/3 < 0: |C_conv| = |C_p| + |C_q| + 4st/3

    pr("  Key identity: 4A_conv - S^2 = (4a-s^2) + (4b-t^2) - 4st/3")
    pr("  So: C_conv = C_p + C_q - 4st/3")
    pr("  |C_conv| = |C_p| + |C_q| + 4st/3 (all terms < 0, same sign)")

    # In normalized: C_conv = (x-1) + r^2(y-1) - 4r/3
    C_conv_normalized = (x-1) + r**2*(y-1) - 4*r/3

    # Similarly, for f1:
    # f1_conv = S^2+12A = (s+t)^2+12(a+b+st/6) = s^2+12a + t^2+12b + 4st
    # = f1p + f1q + 4st
    pr("  f1_conv = f1p + f1q + 4st")
    pr("  In normalized: f1c = (1+3x) + r^2(1+3y) + 4r")

    # For f2:
    # f2 = 2s^3-8sa-9u^2
    # f2_conv = 2S^3-8SA-9U^2

    # ================================================================
    # SECTION 3: Check mutual exclusivity numerically
    # ================================================================
    pr("\n--- Numerical partition check ---")

    rng = np.random.default_rng(42)

    def T2_surplus_fn(rv, xv, yv):
        f1p = 1 + 3*xv; f1q = 1 + 3*yv
        f1c = (1+rv)**2 + 3*xv + 3*yv*rv**2 + 2*rv
        n2p = 1 + 15*xv; n2q = 1 + 15*yv
        n2c = (1+rv)**2 + 15*xv + 15*yv*rv**2 + 10*rv
        return ((1+rv)*n2c*f1p*f1q - n2p*f1c*f1q - rv*n2q*f1c*f1p) / (18*f1c*f1p*f1q)

    def R_surplus_fn(rv, xv, yv, pv, qv):
        # R_p
        f2p = 2*(1-xv) - 9*pv**2
        Rp = (xv-1)*(1+3*xv)/(9*f2p)
        # R_q
        f2q = 2*rv**3*(1-yv) - 9*qv**2
        Rq = rv**4*(yv-1)*(1+3*yv)/(9*f2q)
        # R_conv
        Sv = 1+rv; Av = (3*xv + 3*yv*rv**2 + 2*rv)/12
        Cc = 4*Av - Sv**2
        f1c = Sv**2 + 12*Av
        f2c = 2*Sv**3 - 8*Sv*Av - 9*(pv+qv)**2
        Rc = Cc*f1c/(9*f2c)
        return Rc - Rp - Rq

    samples = []
    tries = 0
    while len(samples) < 200000 and tries < 6000000:
        tries += 1
        rv = float(np.exp(rng.uniform(np.log(0.05), np.log(20.0))))
        xv = float(rng.uniform(1e-5, 1-1e-5))
        yv = float(rng.uniform(1e-5, 1-1e-5))
        pmax2 = 2*(1-xv)/9
        qmax2 = 2*(rv**3)*(1-yv)/9
        pv = float(rng.uniform(-0.99*np.sqrt(pmax2), 0.99*np.sqrt(pmax2)))
        qv = float(rng.uniform(-0.99*np.sqrt(qmax2), 0.99*np.sqrt(qmax2)))
        Sv = 1 + rv; Uv = pv + qv
        Av = xv/4 + yv*rv**2/4 + rv/6
        f2c = 2*Sv**3 - 8*Sv*Av - 9*Uv**2
        if f2c <= 1e-8:
            continue
        samples.append((rv, xv, yv, pv, qv))
    samples = np.array(samples)
    n = len(samples)
    pr(f"  Got {n} samples (wider range: r∈[0.05,20])")

    rv_a = samples[:,0]; xv_a = samples[:,1]; yv_a = samples[:,2]
    pv_a = samples[:,3]; qv_a = samples[:,4]

    T2_vals = T2_surplus_fn(rv_a, xv_a, yv_a)
    R_vals = R_surplus_fn(rv_a, xv_a, yv_a, pv_a, qv_a)
    total = T2_vals + R_vals

    T2_neg = T2_vals < -1e-15
    R_neg = R_vals < -1e-15
    both_neg = T2_neg & R_neg

    pr(f"  T2 < 0: {np.sum(T2_neg)}/{n} ({100*np.sum(T2_neg)/n:.1f}%)")
    pr(f"  R < 0: {np.sum(R_neg)}/{n} ({100*np.sum(R_neg)/n:.1f}%)")
    pr(f"  BOTH < 0: {np.sum(both_neg)}/{n}")
    pr(f"  T2+R < 0: {np.sum(total < -1e-12)}/{n}")
    pr(f"  T2+R min: {np.min(total):.4e}")

    if np.sum(both_neg) == 0:
        pr("  *** T2_surplus < 0 and R_surplus < 0 are MUTUALLY EXCLUSIVE! ***")

    # Where T2 < 0
    if np.any(T2_neg):
        R_at_T2neg = R_vals[T2_neg]
        pr(f"\n  Where T2 < 0 ({np.sum(T2_neg)}):")
        pr(f"    R_surplus min: {np.min(R_at_T2neg):.4e}")
        pr(f"    R_surplus always > 0? {np.all(R_at_T2neg > 0)}")

    # Where R < 0
    if np.any(R_neg):
        T2_at_Rneg = T2_vals[R_neg]
        pr(f"\n  Where R < 0 ({np.sum(R_neg)}):")
        pr(f"    T2_surplus min: {np.min(T2_at_Rneg):.4e}")
        pr(f"    T2_surplus always > 0? {np.all(T2_at_Rneg > 0)}")
        pr(f"    T2/|R| ratio min: {np.min(T2_at_Rneg/np.abs(R_vals[R_neg])):.4f}")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("T2_SURPLUS POLYNOMIAL ANALYSIS")
    pr('='*72)

    pr(f"\n  T2_surplus_num factored: {T2_surp_factored}")

    # Check: is T2_surplus_num a polynomial with nice roots?
    # This determines the boundary between T2 > 0 and T2 < 0 regions

    # ================================================================
    pr(f"\n{'='*72}")
    pr("FACTOR L ANALYSIS (common d6 factor)")
    pr('='*72)

    L = -27*r*x*y + 3*r*x + 9*r*y**2 - 3*r*y + 2*r + 9*x**2 - 27*x*y - 3*x + 3*y + 2
    pr(f"  L = {L}")

    # Value at equality r=1, x=y=1/3
    L_eq = float(L.subs({r: 1, x: sp.Rational(1,3), y: sp.Rational(1,3)}))
    pr(f"  L(1, 1/3, 1/3) = {L_eq}")

    # Factor L as polynomial in (r,x,y)
    L_factored = factor(L)
    pr(f"  L factored: {L_factored}")

    # L as polynomial in r:
    pr(f"  L as quadratic in nothing... it's linear in r:")
    L_r0 = L.subs(r, 0)
    L_r1 = expand(L - L_r0)
    pr(f"    L(r=0) = {L_r0} = {factor(L_r0)}")
    pr(f"    L - L(r=0) = {L_r1} = r*({factor(expand(L_r1/r))})")

    # L = (9x^2-27xy-3x+3y+2) + r*(-27xy+3x+9y^2-3y+2)
    L_const = expand(9*x**2 - 27*x*y - 3*x + 3*y + 2)
    L_coeff_r = expand(-27*x*y + 3*x + 9*y**2 - 3*y + 2)
    pr(f"  L = {factor(L_const)} + r*({factor(L_coeff_r)})")

    # Sign of L on feasible domain
    L_fn = sp.lambdify((r, x, y), L, 'numpy')
    L_vals = L_fn(rv_a, xv_a, yv_a)
    pr(f"\n  L numerical: min={np.min(L_vals):.4f}, max={np.max(L_vals):.4f}")
    pr(f"  L > 0: {np.sum(L_vals > 0)}/{n} ({100*np.sum(L_vals > 0)/n:.1f}%)")
    pr(f"  L < 0: {np.sum(L_vals < 0)}/{n}")

    # Relationship between L sign and T2_surplus sign
    L_pos = L_vals > 0
    L_neg = L_vals < 0
    pr(f"\n  L>0 ∩ T2<0: {np.sum(L_pos & T2_neg)}/{n}")
    pr(f"  L>0 ∩ T2>0: {np.sum(L_pos & ~T2_neg)}/{n}")
    pr(f"  L<0 ∩ T2<0: {np.sum(L_neg & T2_neg)}/{n}")
    pr(f"  L<0 ∩ T2>0: {np.sum(L_neg & ~T2_neg)}/{n}")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("R CONCAVITY AND SUPER-ADDITIVITY IN u²")
    pr('='*72)

    # R(s,u,a) = C·f1/(9·(F-9u²)) where C=4a-s², f1=s²+12a, F=2s³-8sa
    # dR/du² = C·f1·9/(9·(F-9u²)²) = C·f1/(F-9u²)²
    # Since C<0 and f1>0: dR/du² < 0 (R decreasing in u²)
    # d²R/d(u²)² = 2·9·C·f1/(F-9u²)³ = 18C·f1/(F-9u²)³ < 0 (concave in u²)

    pr("  R is concave and decreasing in u² (for fixed s,a)")
    pr("  This means for fixed (s,a):")
    pr("    R(s,u₁+u₂,a) <= R(s,u₁,a) + R(s,u₂,a) - R(s,0,a)")
    pr("  But convolution changes (s,a) too, so we need multivariate analysis.")

    # What is R(s,0,a)? = C·f1/(9·F) = (4a-s²)(s²+12a)/(9(2s³-8sa))
    # = (4a-s²)(s²+12a)/(18s(s²-4a))
    # = -(s²-4a)(s²+12a)/(18s(s²-4a))   [since 4a<s² ⟹ C=4a-s²=-(s²-4a)]
    # = -(s²+12a)/(18s)  when u=0
    # Interesting: R(s,0,a) = -(s²+12a)/(18s) = -f₁/(18s)

    pr("\n  R(s,0,a) = -f1/(18s) = -(s²+12a)/(18s)")
    pr("  In normalized: R_p(0) = -(1+3x)/18")
    pr("  R_q(0) = -r(1+3y)/18")
    pr("  R_conv(0) = -(1+r)f1c_norm/(18(1+r)) = ... needs computation")

    # R_conv at U=0 (p=q=0): A_conv at U=0
    # R_conv(S,0,A) = (4A-S²)(S²+12A)/(9(2S³-8SA))
    # = (4A-S²)f1c/(18S(S²-4A))
    # Hmm, same simplification:
    # = -(S²-4A)(S²+12A)/(18S(S²-4A)) = -(S²+12A)/(18S)
    # Wait, this simplification requires S²-4A ≠ 0, which is true since C<0.
    # Actually:
    # R(S,0,A) = (4A-S²)·(S²+12A)/(9·(2S³-8SA-0))
    # = (4A-S²)·(S²+12A)/(18S(S²-4A))
    # = -(S²-4A)·(S²+12A)/(18S(S²-4A))
    # = -(S²+12A)/(18S)

    pr("  R_conv(U=0) = -(S²+12A)/(18S) = -f1c/(18(1+r))")

    # So at p=q=0:
    # R_surplus(p=q=0) = R_conv(0) - R_p(0) - R_q(0)
    # = -f1c/(18(1+r)) + (1+3x)/18 + r(1+3y)/18
    # = [-f1c/(1+r) + (1+3x) + r(1+3y)] / 18
    # f1c = 1+4r+r²+3x+3yr²
    # (1+3x)(1+r) + r(1+3y)(1+r)... hmm let me compute
    # (1+3x) + r(1+3y) = 1+3x+r+3ry
    # f1c/(1+r) = (1+4r+r²+3x+3yr²)/(1+r)
    # = ((1+r)²+2r+3x+3yr²)/(1+r) = (1+r) + (2r+3x+3yr²)/(1+r)

    # Numerator of R_surplus(0): (1+3x)(1+r) + r(1+3y)(1+r) - f1c
    R0_num = expand((1+3*x)*(1+r) + r*(1+3*y)*(1+r) - (1+4*r+r**2+3*x+3*y*r**2+2*r))
    R0_num = expand(R0_num)
    pr(f"\n  R_surplus(p=q=0) numerator (×18(1+r)):")
    pr(f"    = {R0_num} = {factor(R0_num)}")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("R_SURPLUS AS FUNCTION OF p,q WITH FIXED (r,x,y)")
    pr('='*72)

    # R_surplus depends on p through f2p and q through f2q, and p+q through f2c
    # Let's view R_surplus as a function of w1=p², w2=q² and their interaction through (p+q)²

    # Key question: is R_surplus monotone in p² or q²?

    # For fixed (r,x,y), R_p = (x-1)(1+3x)/(9*(2(1-x)-9p²))
    # |R_p| = (1-x)(1+3x)/(9*(2(1-x)-9p²))
    # |R_p| is INCREASING in p² (closer to boundary → larger |R_p|)
    # Similarly |R_q| is increasing in q²

    # R_conv depends on (p+q)² through f2c = F_c - 9(p+q)²
    # |R_conv| = |C_conv|·f1c/(9*(F_c-9(p+q)²))
    # |R_conv| is INCREASING in (p+q)²

    # R_surplus = R_conv + |R_p| + |R_q| (all R values are negative, surplus = conv - p - q)
    # = -|R_conv| + |R_p| + |R_q|

    # So R_surplus increases with p² (through |R_p|), increases with q² (through |R_q|),
    # but DECREASES with (p+q)² (through |R_conv|)

    # R_surplus is LARGEST when |p-q| is large and (p+q) is small
    # R_surplus is SMALLEST when p and q have the SAME SIGN (maximizing (p+q)²)

    pr("  R_surplus = |R_p| + |R_q| - |R_conv|")
    pr("  |R_p| increases with p² (always)")
    pr("  |R_q| increases with q² (always)")
    pr("  |R_conv| increases with (p+q)² (always)")
    pr("")
    pr("  So R_surplus is smallest when p,q have same sign")
    pr("  And d6 = -1296r·L·p²q²(p+q)² is most negative when p,q same sign")
    pr("  The SAME regime makes both R_surplus smaller and d6 more negative!")
    pr("  This is why the compensation works — they cancel each other.")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("BOUNDARY ANALYSIS: R_surplus AT p=pmax, q=qmax (worst case)")
    pr('='*72)

    # Worst case for R_surplus: p and q at max with same sign
    worst_vals = []
    for _ in range(50000):
        rv = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
        xv = float(rng.uniform(0.01, 0.99))
        yv = float(rng.uniform(0.01, 0.99))
        pmax = np.sqrt(2*(1-xv)/9)
        qmax = np.sqrt(2*rv**3*(1-yv)/9)

        for frac in [0.99, 0.95, 0.9, 0.5]:
            pv = frac*pmax; qv = frac*qmax  # same sign
            Sv = 1 + rv; Uv = pv + qv
            Av = xv/4 + yv*rv**2/4 + rv/6
            f2c = 2*Sv**3 - 8*Sv*Av - 9*Uv**2
            if f2c <= 1e-8:
                continue

            T2v = T2_surplus_fn(rv, xv, yv)
            Rv = R_surplus_fn(rv, xv, yv, pv, qv)
            worst_vals.append((rv, xv, yv, pv, qv, T2v, Rv, T2v+Rv))

    worst_vals = np.array(worst_vals)
    pr(f"  Tested {len(worst_vals)} worst-case points (p,q same sign)")
    T2_worst = worst_vals[:, 5]
    R_worst = worst_vals[:, 6]
    total_worst = worst_vals[:, 7]

    pr(f"  T2_surplus: min={np.min(T2_worst):.4e}")
    pr(f"  R_surplus: min={np.min(R_worst):.4e}")
    pr(f"  T2+R surplus: min={np.min(total_worst):.4e}")
    pr(f"  Both < 0: {np.sum((T2_worst < 0) & (R_worst < 0))}")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("APPROACH: PROVE T2+R >= 0 VIA JOINT ANALYSIS OF R")
    pr('='*72)

    # R_surplus = -|R_conv| + |R_p| + |R_q|
    # where |R(s,u,a)| = (s²-4a)(s²+12a)/(9(2s³-8sa-9u²))
    # = (1-x)(1+3x)/(9(2(1-x)-9p²)) for the p-polynomial

    # The question: does |R_p| + |R_q| >= |R_conv| always on feasible?
    # From the data: R_surplus is negative at ~4.6% of samples
    # So |R_conv| > |R_p| + |R_q| can happen!
    # In those cases, T2_surplus must compensate.

    # BUT: When does R_surplus < 0? When p and q have the same sign and are large.
    # In that region, T2_surplus (which doesn't depend on p,q) must be positive.

    # The condition for T2_surplus >= 0 is a polynomial condition on (r,x,y).
    # The condition for R_surplus < 0 is a condition on (r,x,y,p,q).
    # We need: T2_surplus(r,x,y) >= 0 wherever R_surplus can be negative.

    # Check: what (r,x,y) values have R_surplus < 0 for SOME (p,q)?
    pr("\n  Finding (r,x,y) where R_surplus can be negative...")
    rxy_with_Rneg = {}  # key: (r,x,y) approx, value: min R_surplus
    for idx in range(len(worst_vals)):
        rv, xv, yv, pv, qv, T2v, Rv, totalv = worst_vals[idx]
        if Rv < -1e-10:
            key = (round(rv,2), round(xv,2), round(yv,2))
            if key not in rxy_with_Rneg or Rv < rxy_with_Rneg[key][0]:
                rxy_with_Rneg[key] = (Rv, T2v, pv, qv)

    pr(f"  {len(rxy_with_Rneg)} distinct (r,x,y) have R_surplus < 0 for some (p,q)")
    if rxy_with_Rneg:
        T2_at_Rneg_points = [v[1] for v in rxy_with_Rneg.values()]
        pr(f"  T2_surplus at those points: min={min(T2_at_Rneg_points):.4e}, all>0? {all(t > 0 for t in T2_at_Rneg_points)}")

    # ================================================================
    pr(f"\n{'='*72}")
    pr("VERIFY: d6 = -1296*r*L*p^2*q^2*(p+q)^2")
    pr('='*72)

    # Build K_T2R/r^2 to verify
    pr("  Building K_T2R/r^2 for d6 verification...")
    s, t_s, u, v_s, a, b = symbols('s t u v a b')
    def T2R_num(ss, uu, aa):
        return 8*aa*(ss**2 - 4*aa)**2 - ss*uu**2*(ss**2 + 60*aa)
    def T2R_den(ss, uu, aa):
        return 2*(ss**2 + 12*aa)*(2*ss**3 - 8*ss*aa - 9*uu**2)

    S_s = s + t_s; U_s = u + v_s; A_s = a + b + s*t_s/6
    surplus_num = expand(
        T2R_num(S_s,U_s,A_s)*T2R_den(s,u,a)*T2R_den(t_s,v_s,b)
        - T2R_num(s,u,a)*T2R_den(S_s,U_s,A_s)*T2R_den(t_s,v_s,b)
        - T2R_num(t_s,v_s,b)*T2R_den(S_s,U_s,A_s)*T2R_den(s,u,a))

    subs_norm = {t_s: r*s, a: x*s**2/4, b: y*r**2*s**2/4,
                 u: p*s**Rational(3,2), v_s: q*s**Rational(3,2)}
    K_T2R = expand(expand(surplus_num.subs(subs_norm)) / s**16)
    K_red = expand(sp.cancel(K_T2R / r**2))

    # Extract d6
    poly_pq = Poly(K_red, p, q)
    d6 = sp.Integer(0)
    for monom, coeff in poly_pq.as_dict().items():
        i, j = monom
        if i + j == 6:
            d6 = expand(d6 + coeff * p**i * q**j)

    pr(f"  d6 computed [{time.time()-t0:.1f}s]")

    # Verify d6 = -1296*r*L*p^2*q^2*(p+q)^2
    L_sym = -27*r*x*y + 3*r*x + 9*r*y**2 - 3*r*y + 2*r + 9*x**2 - 27*x*y - 3*x + 3*y + 2
    d6_predicted = -1296*r*L_sym*p**2*q**2*(p+q)**2
    check = expand(d6 - d6_predicted)
    pr(f"  d6 = -1296*r*L*p²q²(p+q)²? {check == 0}")

    if check != 0:
        pr(f"  Difference: {check}")

    pr(f"\n  Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
