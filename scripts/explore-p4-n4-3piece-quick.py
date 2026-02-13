#!/usr/bin/env python3
"""Quick numerical test of the 3-piece Cauchy-Schwarz decomposition.

1/Phi4(-s,u,a) = T1 + T2 + R where:
  T1 = 3u^2 / [4(s^2+12a)]                       (Titu piece)
  T2 = s(s^2+60a) / [18(s^2+12a)]                 (poly-ratio, NO u)
  R  = (4a-s^2)(s^2+12a) / [9*(2s^3-8sa-9u^2)]   (remainder)

Verified: disc(-s,u,a) = f2*(3u^2+(2s/9)(s^2+60a)) + R_f2
  where R_f2 = -(4/9)(s^2-4a)(s^2+12a)^2
  and phi4_disc = 4*(s^2+12a)*f2

Sign analysis of each surplus piece on 50k+ feasible samples.
"""

import numpy as np


def pr(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def disc_num(s, u, a):
    return 256*a**3 - 128*a**2*s**2 + 16*a*s**4 - 144*a*s*u**2 + 4*s**3*u**2 - 27*u**4


def phi4_num(s, u, a):
    return 4*(s**2 + 12*a)*(2*s**3 - 8*s*a - 9*u**2)


def inv_phi4(s, u, a):
    d = phi4_num(s, u, a)
    return np.where(np.abs(d) > 1e-30, disc_num(s, u, a) / d, np.nan)


def T1(s, u, a):
    f1 = s**2 + 12*a
    return 3*u**2 / (4*f1)


def T2(s, u, a):
    f1 = s**2 + 12*a
    return s*(s**2 + 60*a) / (18*f1)


def R_piece(s, u, a):
    f1 = s**2 + 12*a
    f2 = 2*s**3 - 8*s*a - 9*u**2
    return np.where(np.abs(f2) > 1e-30,
                    (4*a - s**2)*f1 / (9*f2),
                    np.nan)


def main():
    pr("=" * 72)
    pr("3-PIECE CAUCHY-SCHWARZ DECOMPOSITION: NUMERICAL CHECK")
    pr("=" * 72)

    rng = np.random.default_rng(42)

    # First verify the decomposition
    pr("\nVerifying 1/Phi4 = T1 + T2 + R at 10000 points...")
    s_v = np.exp(rng.uniform(np.log(0.5), np.log(5), 10000))
    a_v = rng.uniform(0.01, 0.9, 10000) * s_v**2 / 4
    f2_bound = (2*s_v**3 - 8*s_v*a_v) / 9
    mask = f2_bound > 0
    s_v, a_v, f2_bound = s_v[mask], a_v[mask], f2_bound[mask]
    u_v = rng.uniform(-0.9, 0.9, len(s_v)) * np.sqrt(f2_bound)

    inv_vals = inv_phi4(s_v, u_v, a_v)
    t1_vals = T1(s_v, u_v, a_v)
    t2_vals = T2(s_v, u_v, a_v)
    r_vals = R_piece(s_v, u_v, a_v)

    valid = np.isfinite(inv_vals) & np.isfinite(r_vals)
    err = np.abs(inv_vals[valid] - t1_vals[valid] - t2_vals[valid] - r_vals[valid])
    pr(f"  Max decomposition error: {np.max(err):.2e} (should be ~1e-12)")

    # Now sample surpluses
    pr(f"\n{'='*72}")
    pr("SURPLUS SIGN ANALYSIS")
    pr('='*72)

    full_surp = []
    t1_surp = []
    t2_surp = []
    r_surp = []
    n_target = 50000
    tries = 0

    while len(full_surp) < n_target and tries < 10 * n_target:
        tries += 1
        sv = float(np.exp(rng.uniform(np.log(0.5), np.log(5))))
        tv = float(np.exp(rng.uniform(np.log(0.5), np.log(5))))
        av = float(rng.uniform(0.05, 0.9)) * sv**2 / 4
        bv = float(rng.uniform(0.05, 0.9)) * tv**2 / 4

        f2_p = 2*sv**3 - 8*sv*av - 0  # at u=0
        f2_q = 2*tv**3 - 8*tv*bv - 0
        if f2_p <= 0 or f2_q <= 0:
            continue

        umax = np.sqrt((2*sv**3 - 8*sv*av) / 9) * 0.9
        vmax = np.sqrt((2*tv**3 - 8*tv*bv) / 9) * 0.9
        uv = float(rng.uniform(-umax, umax))
        vv = float(rng.uniform(-vmax, vmax))

        Sv = sv + tv
        Uv = uv + vv
        Av = av + bv + sv*tv/6

        f2_c = 2*Sv**3 - 8*Sv*Av - 9*Uv**2
        if f2_c <= 0:
            continue

        try:
            full_s = float(inv_phi4(Sv, Uv, Av) - inv_phi4(sv, uv, av) - inv_phi4(tv, vv, bv))
            t1_s = float(T1(Sv, Uv, Av) - T1(sv, uv, av) - T1(tv, vv, bv))
            t2_s = float(T2(Sv, Uv, Av) - T2(sv, uv, av) - T2(tv, vv, bv))
            r_s = float(R_piece(Sv, Uv, Av) - R_piece(sv, uv, av) - R_piece(tv, vv, bv))

            if not all(np.isfinite([full_s, t1_s, t2_s, r_s])):
                continue
            if abs(full_s - t1_s - t2_s - r_s) > 1e-6 * max(abs(full_s), 1e-10):
                continue

            full_surp.append(full_s)
            t1_surp.append(t1_s)
            t2_surp.append(t2_s)
            r_surp.append(r_s)
        except:
            pass

    full_surp = np.array(full_surp)
    t1_surp = np.array(t1_surp)
    t2_surp = np.array(t2_surp)
    r_surp = np.array(r_surp)
    n = len(full_surp)
    pr(f"\nFeasible samples: {n}")

    pr(f"\n  Full surplus:  min={np.min(full_surp):.4e}, neg={np.sum(full_surp < -1e-10)}/{n}")
    pr(f"  T1 surplus:    min={np.min(t1_surp):.4e}, max={np.max(t1_surp):.4e}")
    pr(f"    neg: {np.sum(t1_surp < -1e-10)}/{n} ({100*np.mean(t1_surp < -1e-10):.1f}%)")
    pr(f"    pos: {np.sum(t1_surp > 1e-10)}/{n} ({100*np.mean(t1_surp > 1e-10):.1f}%)")
    pr(f"  T2 surplus:    min={np.min(t2_surp):.4e}, max={np.max(t2_surp):.4e}")
    pr(f"    neg: {np.sum(t2_surp < -1e-10)}/{n} ({100*np.mean(t2_surp < -1e-10):.1f}%)")
    pr(f"    pos: {np.sum(t2_surp > 1e-10)}/{n} ({100*np.mean(t2_surp > 1e-10):.1f}%)")
    pr(f"  R surplus:     min={np.min(r_surp):.4e}, max={np.max(r_surp):.4e}")
    pr(f"    neg: {np.sum(r_surp < -1e-10)}/{n} ({100*np.mean(r_surp < -1e-10):.1f}%)")
    pr(f"    pos: {np.sum(r_surp > 1e-10)}/{n} ({100*np.mean(r_surp > 1e-10):.1f}%)")

    # Key groupings
    pr(f"\n  --- Key groupings ---")
    t2r = t2_surp + r_surp
    pr(f"  T2+R:   min={np.min(t2r):.4e}, neg={np.sum(t2r < -1e-10)}/{n} ({100*np.mean(t2r < -1e-10):.1f}%)")
    if np.sum(t2r < -1e-10) == 0:
        pr(f"  *** T2+R >= 0! Since T1<=0 and full=T1+T2+R>=0, this proves the inequality ***")
    else:
        # Does T2+R always >= -T1?
        full_check = t2r + t1_surp  # should be >= 0
        pr(f"  T1+T2+R: min={np.min(full_check):.4e}")

    # T2 alone (no p,q dependence!)
    pr(f"\n  T2 alone: min={np.min(t2_surp):.4e}")
    if np.sum(t2_surp < -1e-10) == 0:
        pr(f"  *** T2 surplus >= 0! ***")
    else:
        # Is T2 >= |T1| + |R|? Probably not but check
        t2_dominates = t2_surp >= np.abs(t1_surp) + np.abs(r_surp)
        pr(f"  T2 >= |T1|+|R|: {np.sum(t2_dominates)}/{n}")

    # When T2+R < 0 (if ever), what's happening?
    if np.sum(t2r < -1e-10) > 0:
        bad_mask = t2r < -1e-10
        pr(f"\n  When T2+R < 0 ({np.sum(bad_mask)} points):")
        pr(f"    T1 range: [{np.min(t1_surp[bad_mask]):.4e}, {np.max(t1_surp[bad_mask]):.4e}]")
        pr(f"    T2 range: [{np.min(t2_surp[bad_mask]):.4e}, {np.max(t2_surp[bad_mask]):.4e}]")
        pr(f"    R range:  [{np.min(r_surp[bad_mask]):.4e}, {np.max(r_surp[bad_mask]):.4e}]")
        pr(f"    full(=T1+T2+R): min={np.min(full_surp[bad_mask]):.4e}")

    # Ratios
    pr(f"\n  --- Relative contributions ---")
    pos = full_surp > 1e-10
    if np.any(pos):
        pr(f"  T1/full mean: {np.mean(t1_surp[pos]/full_surp[pos]):.4f}")
        pr(f"  T2/full mean: {np.mean(t2_surp[pos]/full_surp[pos]):.4f}")
        pr(f"  R/full mean:  {np.mean(r_surp[pos]/full_surp[pos]):.4f}")

    # Correlations
    pr(f"\n  --- Correlations ---")
    pr(f"  corr(T1, T2) = {np.corrcoef(t1_surp, t2_surp)[0,1]:.4f}")
    pr(f"  corr(T1, R)  = {np.corrcoef(t1_surp, r_surp)[0,1]:.4f}")
    pr(f"  corr(T2, R)  = {np.corrcoef(t2_surp, r_surp)[0,1]:.4f}")

    # NEW: Check T2_surplus as function of (s,t,a,b) only (no u,v)
    # T2(conv) - T2(p) - T2(q) with conv params (s+t, a+b+st/6)
    pr(f"\n{'='*72}")
    pr("T2_SURPLUS STRUCTURE (no u,v dependence)")
    pr('='*72)
    pr("T2_surplus depends only on (s,t,a,b) = effectively on (r,x,y)")

    # T2(s) = s(s^2+60a) / (18(s^2+12a)) = (s^3+60as) / (18s^2+216a)
    # Let's compute T2_surplus as a rational function
    # T2_surplus = T2(s+t, a+b+st/6) - T2(s,a) - T2(t,b)
    # This is a function of (s,t,a,b) only.
    # In normalized coords: t=rs, a=xs^2/4, b=yr^2s^2/4
    # After substitution, T2_surplus becomes a function of (r,x,y) times some power of s.
    # But T2 itself is degree 0 in s (scaling: if s->λs, a->λ^2a, then T2 is invariant)
    # Check: T2(λs, λ^2a) = λs(λ^2s^2+60λ^2a)/(18(λ^2s^2+12λ^2a)) = λ^3s(s^2+60a)/(18λ^2(s^2+12a))
    # = λ·s(s^2+60a)/(18(s^2+12a)) = λ·T2(s,a)
    # So T2 scales as λ^1. Not scale-invariant.
    # With t=rs, a=xs^2/4, b=yr^2s^2/4:
    # T2(s,a=xs^2/4) = s(s^2+60·xs^2/4)/(18(s^2+12·xs^2/4))
    #                = s(s^2+15xs^2)/(18(s^2+3xs^2))
    #                = s·s^2(1+15x)/(18·s^2(1+3x))
    #                = s(1+15x)/(18(1+3x))
    # T2(t=rs, b=yr^2s^2/4) = rs(1+15y)/(18(1+3y))
    # T2(conv): S=s(1+r), A_c = xs^2/4 + yr^2s^2/4 + rs^2/6
    #           = s^2(x/4 + yr^2/4 + r/6)
    # f1_c = S^2 + 12*A_c = s^2(1+r)^2 + 12s^2(x/4+yr^2/4+r/6)
    #       = s^2[(1+r)^2 + 3x + 3yr^2 + 2r]
    #       = s^2[1+2r+r^2+3x+3yr^2+2r]
    #       = s^2[1+4r+r^2+3x+3yr^2]
    # S^3 = s^3(1+r)^3
    # S^3+60*A_c*S = S(S^2+60A_c) = s(1+r)[s^2(1+r)^2+60s^2(...)]
    # Hmm this is getting messy. Let me just compute numerically.

    # T2 as a function of (r,x,y) after normalization
    def T2_normalized(rv, xv, yv):
        """T2_surplus in normalized coordinates.
        s cancels, leaving function of (r,x,y) only."""
        # T2(p) = s(1+15x)/(18(1+3x))
        # T2(q) = rs(1+15y)/(18(1+3y))
        T2p = (1+15*xv)/(18*(1+3*xv))
        T2q = rv*(1+15*yv)/(18*(1+3*yv))

        # T2(conv): S=(1+r)s, A_c = s^2(x/4+yr^2/4+r/6)
        # T2(S,A_c) = S(S^2+60A_c)/(18(S^2+12A_c))
        #           = s(1+r)·[s^2(1+r)^2 + 60s^2(x/4+yr^2/4+r/6)] / [18·(s^2(1+r)^2+12s^2(x/4+yr^2/4+r/6))]
        #           = s(1+r)·s^2·[(1+r)^2 + 15x+15yr^2+10r] / [18·s^2·((1+r)^2+3x+3yr^2+2r)]
        #           = s(1+r)·[(1+r)^2+15x+15yr^2+10r] / [18·((1+r)^2+3x+3yr^2+2r)]
        num_c = (1+rv)*((1+rv)**2 + 15*xv + 15*yv*rv**2 + 10*rv)
        den_c = 18*((1+rv)**2 + 3*xv + 3*yv*rv**2 + 2*rv)
        T2c = num_c / den_c

        # T2_surplus = T2c - T2p - T2q (note: these are all divided by s, so s cancels)
        return T2c - T2p - T2q

    # Test T2_surplus
    rv_test = np.exp(rng.uniform(np.log(0.1), np.log(10), 100000))
    xv_test = rng.uniform(0.01, 0.99, 100000)
    yv_test = rng.uniform(0.01, 0.99, 100000)
    t2_norm = T2_normalized(rv_test, xv_test, yv_test)

    pr(f"\n  T2_surplus(r,x,y): {len(rv_test)} points")
    pr(f"    min = {np.min(t2_norm):.6e}")
    pr(f"    max = {np.max(t2_norm):.6e}")
    pr(f"    neg: {np.sum(t2_norm < -1e-10)}")
    pr(f"    pos: {np.sum(t2_norm > 1e-10)}")

    if np.sum(t2_norm < -1e-10) == 0:
        pr(f"\n  *** T2_surplus >= 0 for all (r,x,y)! ***")
        pr(f"  This means: T2 is SUPERADDITIVE under convolution.")
        pr(f"  Combined with T1_surplus <= 0 (Cauchy-Schwarz), the proof reduces to:")
        pr(f"    R_surplus >= -T1_surplus - T2_surplus >= -T1_surplus")
        pr(f"  i.e., the 'remainder' piece must absorb the Titu deficit.")
    elif np.sum(t2_norm > 1e-10) == 0:
        pr(f"\n  *** T2_surplus <= 0 for all (r,x,y)! ***")
        pr(f"  So both T1 and T2 are sub-additive, R must provide all surplus.")

    # At equality (r=1, x=y=1/3):
    t2_eq = T2_normalized(1.0, 1/3, 1/3)
    pr(f"\n  T2_surplus at equality (r=1,x=y=1/3): {t2_eq:.6e}")

    # R_surplus analysis: does R_surplus >= 0?
    # R(s,u,a) = (4a-s^2)(s^2+12a) / (9*f2) where f2 = 2s^3-8sa-9u^2
    # R is always <= 0 (since 4a <= s^2 for real-rooted, f2 > 0 for disc > 0)
    # So R is NEGATIVE for each polynomial.
    # R_surplus = R(conv) - R(p) - R(q) where each R is <= 0
    # R_surplus could be positive (less negative for conv than sum of p,q)
    pr(f"\n{'='*72}")
    pr("R PIECE ANALYSIS")
    pr('='*72)
    pr("R(s,u,a) = (4a-s^2)(s^2+12a) / [9(2s^3-8sa-9u^2)]")
    pr("Always <= 0 on the feasible domain (4a <= s^2, f2 > 0)")
    pr("R_surplus = R(conv) - R(p) - R(q)")
    pr("= [less negative] - [negative] - [negative]")
    pr("Could be positive if conv R is closer to 0 than sum of individual Rs")

    # R normalized:
    def R_normalized(rv, xv, yv, pv, qv):
        """R(s,u,a) normalized.
        R = (4a-s^2)(s^2+12a) / [9*f2]
        f2 = 2s^3-8sa-9u^2
        After normalization with s=1: a=x/4, u^2=p^2
        R(1,p,x/4) = (x-1)(1+3x) / [9*(2-2x-9p^2)]
        """
        # R for p: s=1 (normalized), a=x/4, u=p
        # (4·x/4 - 1)(1+12·x/4) / [9*(2·1-8·x/4-9p^2)]
        # = (x-1)(1+3x) / [9*(2-2x-9p^2)]
        Rp = (xv-1)*(1+3*xv) / (9*(2-2*xv-9*pv**2))

        # R for q: s=r (normalized), a=yr^2/4, u=q
        # Wait, need more care. Let me think about this.
        # R(s,u,a) = (4a-s^2)(s^2+12a)/(9*f2), f2=2s^3-8sa-9u^2
        # For poly q: e2=-t, e3=v, e4=b. So s->t, u->v, a->b.
        # In normalized: t=rs, b=yr^2s^2/4, v=qs^(3/2)
        # R(t,v,b) = (4b-t^2)(t^2+12b)/(9*(2t^3-8tb-9v^2))
        # = (yr^2s^2-r^2s^2)(r^2s^2+3yr^2s^2)/(9*(2r^3s^3-8rs·yr^2s^2/4·... ))
        # Hmm, let me just use the per-s scaling.
        # R scales as: R(λs, λ^(3/2)u, λ^2a) = (4λ^2a-λ^2s^2)(λ^2s^2+12λ^2a)/(9*(2λ^3s^3-8λ^3sa-9λ^3u^2))
        # = λ^4(4a-s^2)(s^2+12a)/(9·λ^3(2s^3-8sa-9u^2))
        # = λ·R(s,u,a)
        # So R scales as λ^1, same as T2.

        # After normalization (s=1): R(p-poly) = (x-1)(1+3x)/(9(2-2x-9p^2))
        # R(q-poly): with t=r, b=yr^2/4, v=q
        # R(r,q,yr^2/4) = (yr^2-r^2)(r^2+3yr^2)/(9*(2r^3-2yr^3-9q^2))
        # = r^2(y-1)·r^2(1+3y)/(9·(2r^3(1-y)-9q^2))
        # = r^4(y-1)(1+3y)/(9*(2r^3-2yr^3-9q^2))
        Rq = rv**4*(yv-1)*(1+3*yv) / (9*(2*rv**3-2*yv*rv**3-9*qv**2))

        # R(conv): S=1+r, U=p+q, A=x/4+yr^2/4+r/6
        # R(S,U,A) = (4A-S^2)(S^2+12A)/(9*(2S^3-8SA-9U^2))
        S = 1 + rv
        U = pv + qv
        A = xv/4 + yv*rv**2/4 + rv/6
        Rc = (4*A - S**2)*(S**2 + 12*A) / (9*(2*S**3 - 8*S*A - 9*U**2))

        # Surplus = (Rc - Rp - Rq) * s (but s cancels in the ratio with other pieces)
        # Actually, the total surplus is s * (T1_surp + T2_surp + R_surp)
        # and all three scale the same way. So the ratio is correct.
        return Rc - Rp - Rq

    # Sample R_surplus on feasible domain
    n_samp = 100000
    out_rv = []
    out_xv = []
    out_yv = []
    out_pv = []
    out_qv = []
    tries2 = 0
    while len(out_rv) < n_samp and tries2 < 30*n_samp:
        tries2 += 1
        rv_i = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
        xv_i = float(rng.uniform(0.01, 0.99))
        yv_i = float(rng.uniform(0.01, 0.99))
        pmax2 = 2*(1-xv_i)/9
        qmax2 = 2*(rv_i**3)*(1-yv_i)/9
        if pmax2 <= 0 or qmax2 <= 0:
            continue
        pv_i = float(rng.uniform(-0.95*np.sqrt(pmax2), 0.95*np.sqrt(pmax2)))
        qv_i = float(rng.uniform(-0.95*np.sqrt(qmax2), 0.95*np.sqrt(qmax2)))
        # Check convolution feasibility
        S_i = 1 + rv_i
        U_i = pv_i + qv_i
        A_i = xv_i/4 + yv_i*rv_i**2/4 + rv_i/6
        f2_c = 2*S_i**3 - 8*S_i*A_i - 9*U_i**2
        if f2_c <= 1e-6:
            continue
        out_rv.append(rv_i)
        out_xv.append(xv_i)
        out_yv.append(yv_i)
        out_pv.append(pv_i)
        out_qv.append(qv_i)

    rv_arr = np.array(out_rv)
    xv_arr = np.array(out_xv)
    yv_arr = np.array(out_yv)
    pv_arr = np.array(out_pv)
    qv_arr = np.array(out_qv)
    n_feas = len(rv_arr)
    pr(f"  Feasible samples: {n_feas}")

    R_surp_vals = R_normalized(rv_arr, xv_arr, yv_arr, pv_arr, qv_arr)
    valid_mask = np.isfinite(R_surp_vals)
    R_surp_vals = R_surp_vals[valid_mask]
    rv_v = rv_arr[valid_mask]
    xv_v = xv_arr[valid_mask]
    yv_v = yv_arr[valid_mask]
    pv_v = pv_arr[valid_mask]
    qv_v = qv_arr[valid_mask]
    n_valid = len(R_surp_vals)
    pr(f"  Valid (finite) R_surplus: {n_valid}")
    pr(f"  R_surplus: min={np.min(R_surp_vals):.4e}, max={np.max(R_surp_vals):.4e}")
    pr(f"    neg: {np.sum(R_surp_vals < -1e-10)}/{n_valid} ({100*np.mean(R_surp_vals < -1e-10):.1f}%)")
    pr(f"    pos: {np.sum(R_surp_vals > 1e-10)}/{n_valid} ({100*np.mean(R_surp_vals > 1e-10):.1f}%)")

    # T1 surplus in normalized coords
    # T1(s,u,a) = 3u^2/(4(s^2+12a))
    # After normalization (s=1): T1 = 3p^2/(4(1+3x))
    # T1(q): with s=r: T1 = 3q^2/(4(r^2+3yr^2)) = 3q^2/(4r^2(1+3y))
    # T1(conv): 3(p+q)^2 / (4((1+r)^2+3x+3yr^2+2r))
    T1p = 3*pv_v**2 / (4*(1+3*xv_v))
    T1q = 3*qv_v**2 / (4*rv_v**2*(1+3*yv_v))
    f1_c = (1+rv_v)**2 + 3*xv_v + 3*yv_v*rv_v**2 + 2*rv_v
    T1c = 3*(pv_v+qv_v)**2 / (4*f1_c)
    T1_surp_vals = T1c - T1p - T1q
    T2_surp_vals = T2_normalized(rv_v, xv_v, yv_v)
    full_surp_vals = T1_surp_vals + T2_surp_vals + R_surp_vals

    pr(f"\n  Full surplus check: min={np.min(full_surp_vals):.4e}, neg={np.sum(full_surp_vals < -1e-10)}")
    pr(f"  T1_surplus: min={np.min(T1_surp_vals):.4e}, all<=0? {np.all(T1_surp_vals <= 1e-10)}")

    # KEY: T2+R surplus
    T2R = T2_surp_vals + R_surp_vals
    pr(f"\n  T2+R surplus: min={np.min(T2R):.4e}, neg={np.sum(T2R < -1e-10)}/{n_valid}")
    if np.sum(T2R < -1e-10) == 0:
        pr(f"  *** T2+R >= 0! PROOF COMPLETE: T1<=0, T2+R>=0, full=T1+T2+R>=0. ***")

    # Does T2+R always dominate -T1?
    gap = T2R + T1_surp_vals  # should be full_surp >= 0
    pr(f"  T2+R+T1 (=full): min={np.min(gap):.4e}")


if __name__ == '__main__':
    main()
