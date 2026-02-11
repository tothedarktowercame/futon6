#!/usr/bin/env python3
"""Problem 4: Study how the minimum surplus scales with n.

Key questions:
1. Does the adversarial minimum ratio grow with n, or is inf=1 for all n?
2. What do the tight cases look like structurally?
3. Is there a free-probability limiting argument?
"""
import numpy as np
from scipy.optimize import minimize
import sys, os, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
_mod = __import__("verify-p4-inequality")
phi_n = _mod.phi_n
finite_free_conv = _mod.finite_free_conv
coeffs_to_roots = _mod.coeffs_to_roots


def roots_to_coeffs(roots):
    c = np.polynomial.polynomial.polyfromroots(roots)[::-1]
    return c / c[0]


def surplus_ratio(roots_p, roots_q):
    n = len(roots_p)
    cp = roots_to_coeffs(roots_p)
    cq = roots_to_coeffs(roots_q)
    cc = finite_free_conv(cp, cq)
    rc = coeffs_to_roots(cc)
    if not np.allclose(rc.imag, 0, atol=1e-6):
        return float('nan')
    roots_p = np.sort(roots_p.real)
    roots_q = np.sort(roots_q.real)
    roots_c = np.sort(rc.real)
    for r in [roots_p, roots_q, roots_c]:
        if np.min(np.diff(r)) < 1e-10:
            return float('nan')
    pp = phi_n(roots_p)
    pq = phi_n(roots_q)
    pc = phi_n(roots_c)
    if any(v in (0, float('inf')) or v != v for v in [pp, pq, pc]):
        return float('nan')
    rhs = 1/pp + 1/pq
    if rhs < 1e-15:
        return float('nan')
    return (1/pc) / rhs


def find_adversarial_min(n, n_starts=60):
    """Find the minimum surplus ratio at degree n via optimization."""
    best_ratio = float('inf')
    best_p = None
    best_q = None

    def objective(params):
        rp = np.sort(params[:n])
        rq = np.sort(params[n:])
        for i in range(1, n):
            if rp[i] - rp[i-1] < 0.005:
                return 100.0
            if rq[i] - rq[i-1] < 0.005:
                return 100.0
        ratio = surplus_ratio(rp, rq)
        if np.isnan(ratio):
            return 100.0
        return ratio

    for _ in range(n_starts):
        x0 = np.random.randn(2 * n) * 3
        x0[:n] = np.sort(x0[:n])
        x0[n:] = np.sort(x0[n:])
        for i in range(1, n):
            if x0[i] - x0[i-1] < 0.1:
                x0[i] = x0[i-1] + 0.1
            if x0[n+i] - x0[n+i-1] < 0.1:
                x0[n+i] = x0[n+i-1] + 0.1

        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_ratio and res.fun < 90:
                best_ratio = res.fun
                best_p = np.sort(res.x[:n])
                best_q = np.sort(res.x[n:])
        except Exception:
            pass

    return best_ratio, best_p, best_q


def analyze_tight_case(roots_p, roots_q):
    """Analyze what makes a tight case tight."""
    n = len(roots_p)
    rp = np.sort(roots_p)
    rq = np.sort(roots_q)

    # Root statistics
    spread_p = rp[-1] - rp[0]
    spread_q = rq[-1] - rq[0]
    gaps_p = np.diff(rp)
    gaps_q = np.diff(rq)
    gap_cv_p = np.std(gaps_p) / np.mean(gaps_p)  # coefficient of variation
    gap_cv_q = np.std(gaps_q) / np.mean(gaps_q)

    # Scale ratio
    scale_ratio = max(spread_p, spread_q) / min(spread_p, spread_q)

    # Center of mass
    mean_p = np.mean(rp)
    mean_q = np.mean(rq)

    return {
        'spread_p': spread_p,
        'spread_q': spread_q,
        'scale_ratio': scale_ratio,
        'gap_cv_p': gap_cv_p,
        'gap_cv_q': gap_cv_q,
        'mean_p': mean_p,
        'mean_q': mean_q,
        'min_gap_p': np.min(gaps_p),
        'min_gap_q': np.min(gaps_q),
    }


def test_scale_separation(n, n_trials=2000):
    """Test whether scale separation produces tight cases."""
    min_ratio = float('inf')
    best_scale = None

    for _ in range(n_trials):
        # One polynomial at scale s, other at scale 1/s
        log_s = np.random.uniform(0, 4)
        s = 10 ** log_s

        rp = np.sort(np.random.randn(n)) * s
        rq = np.sort(np.random.randn(n)) / s

        for i in range(1, n):
            if rp[i] - rp[i-1] < 0.005 * s:
                rp[i] = rp[i-1] + 0.005 * s
            if rq[i] - rq[i-1] < 0.005 / s:
                rq[i] = rq[i-1] + 0.005 / s

        ratio = surplus_ratio(rp, rq)
        if np.isnan(ratio):
            continue
        if ratio < min_ratio:
            min_ratio = ratio
            best_scale = s

    return min_ratio, best_scale


if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("Problem 4: Scaling analysis of minimum surplus")
    print("=" * 70)

    # Part 1: Adversarial minimum vs n
    print("\n--- Adversarial minimum vs n ---")
    print(f"{'n':>4s} {'min_ratio':>16s} {'surplus':>16s} {'spread_p':>10s} "
          f"{'spread_q':>10s} {'scale_ratio':>12s} {'gap_cv_p':>10s} {'gap_cv_q':>10s}")

    adv_results = {}
    for n in [3, 4, 5, 6, 7, 8, 10]:
        starts = 80 if n <= 6 else 40
        ratio, rp, rq = find_adversarial_min(n, n_starts=starts)
        if rp is not None:
            stats = analyze_tight_case(rp, rq)
            surplus = ratio - 1.0
            print(f"{n:4d} {ratio:16.12f} {surplus:16.2e} "
                  f"{stats['spread_p']:10.4f} {stats['spread_q']:10.4f} "
                  f"{stats['scale_ratio']:12.4f} "
                  f"{stats['gap_cv_p']:10.4f} {stats['gap_cv_q']:10.4f}")
            adv_results[n] = {
                'ratio': ratio, 'surplus': surplus,
                'roots_p': rp.tolist(), 'roots_q': rq.tolist(),
                **stats
            }
        else:
            print(f"{n:4d}  (failed)")

    # Part 2: Scale separation analysis
    print("\n--- Scale separation: how tight can we get? ---")
    print(f"{'n':>4s} {'min_ratio':>16s} {'best_scale':>12s}")
    for n in [4, 5, 6, 7, 8]:
        mr, bs = test_scale_separation(n, n_trials=3000)
        print(f"{n:4d} {mr:16.12f} {bs:12.4f}")

    # Part 3: Random percentiles vs n (systematic)
    print("\n--- Random sampling percentiles vs n ---")
    print(f"{'n':>4s} {'p0 (min)':>14s} {'p1':>10s} {'p5':>10s} "
          f"{'p25':>10s} {'p50':>10s} {'valid':>6s}")

    for n in [3, 4, 5, 6, 7, 8, 10, 12, 15]:
        trials = 5000 if n <= 8 else 2000
        ratios = []
        for _ in range(trials):
            rp = np.sort(np.random.randn(n) * 2)
            rq = np.sort(np.random.randn(n) * 2)
            for i in range(1, n):
                if rp[i] - rp[i-1] < 0.02:
                    rp[i] = rp[i-1] + 0.02
                if rq[i] - rq[i-1] < 0.02:
                    rq[i] = rq[i-1] + 0.02
            r = surplus_ratio(rp, rq)
            if not np.isnan(r):
                ratios.append(r)
        ratios = np.array(ratios)
        if len(ratios) > 10:
            print(f"{n:4d} {np.min(ratios):14.8f} {np.percentile(ratios,1):10.4f} "
                  f"{np.percentile(ratios,5):10.4f} {np.percentile(ratios,25):10.4f} "
                  f"{np.median(ratios):10.4f} {len(ratios):6d}")

    # Part 4: Structure of tight cases
    print("\n--- Structure of adversarial tight cases ---")
    for n, data in sorted(adv_results.items()):
        rp = np.array(data['roots_p'])
        rq = np.array(data['roots_q'])
        print(f"\n  n={n}: ratio = {data['ratio']:.12f}")
        print(f"    roots_p: {rp}")
        print(f"    roots_q: {rq}")
        print(f"    spread_p={data['spread_p']:.4f}  spread_q={data['spread_q']:.4f}  "
              f"scale_ratio={data['scale_ratio']:.4f}")
        print(f"    gap_cv_p={data['gap_cv_p']:.4f}  gap_cv_q={data['gap_cv_q']:.4f}")

        # Check if roots are approximately arithmetic progressions
        gaps_p = np.diff(rp)
        gaps_q = np.diff(rq)
        print(f"    gaps_p: {gaps_p}")
        print(f"    gaps_q: {gaps_q}")
        print(f"    gaps_p uniform? cv={np.std(gaps_p)/np.mean(gaps_p):.4f}")
        print(f"    gaps_q uniform? cv={np.std(gaps_q)/np.mean(gaps_q):.4f}")
