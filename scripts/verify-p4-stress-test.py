#!/usr/bin/env python3
"""Problem 4: Stress test the superadditivity inequality for n>=4.

Three strategies to try to BREAK the inequality:
1. Adversarial optimization (scipy.minimize on -surplus)
2. Near-degenerate cases (roots almost coincident)
3. Large random samples at higher n (6, 7, 8, 10)
4. Boundary hunting: what polynomial pairs minimize the ratio?
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
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


def inv_phi_safe(roots):
    """Compute 1/Phi_n, returning nan on failure."""
    roots = np.sort(roots.real)
    gaps = np.diff(roots)
    if np.min(gaps) < 1e-10:
        return float('nan')
    phi = phi_n(roots)
    if phi == float('inf') or phi == 0 or phi != phi:
        return float('nan')
    return 1.0 / phi


def surplus_ratio(roots_p, roots_q):
    """Compute LHS/RHS = [1/Phi(conv)] / [1/Phi(p) + 1/Phi(q)].
    Returns nan on failure, ratio >= 1 means inequality holds."""
    n = len(roots_p)
    cp = roots_to_coeffs(roots_p)
    cq = roots_to_coeffs(roots_q)
    cc = finite_free_conv(cp, cq)

    rc = coeffs_to_roots(cc)
    if not np.allclose(rc.imag, 0, atol=1e-6):
        return float('nan')

    ip = inv_phi_safe(roots_p)
    iq = inv_phi_safe(roots_q)
    ic = inv_phi_safe(np.sort(rc.real))

    if np.isnan(ip) or np.isnan(iq) or np.isnan(ic):
        return float('nan')
    if ip + iq < 1e-15:
        return float('nan')
    return ic / (ip + iq)


# ============================================================
# Strategy 1: Adversarial optimization
# ============================================================
def adversarial_test(n, n_starts=50):
    """Use optimization to try to minimize the surplus ratio at degree n."""
    print(f"\n  n={n}: {n_starts} optimization starts...")

    best_ratio = float('inf')
    best_roots = None

    def objective(params):
        """Minimize the surplus ratio. params = [roots_p..., roots_q...]."""
        rp = np.sort(params[:n])
        rq = np.sort(params[n:])
        # Enforce distinct roots
        for i in range(1, n):
            if rp[i] - rp[i-1] < 0.01:
                return 100.0  # penalty
            if rq[i] - rq[i-1] < 0.01:
                return 100.0
        ratio = surplus_ratio(rp, rq)
        if np.isnan(ratio):
            return 100.0
        return ratio  # minimize this — looking for values < 1

    for start in range(n_starts):
        x0 = np.random.randn(2 * n) * 3
        # Sort each half
        x0[:n] = np.sort(x0[:n])
        x0[n:] = np.sort(x0[n:])
        # Enforce gaps
        for i in range(1, n):
            if x0[i] - x0[i-1] < 0.1:
                x0[i] = x0[i-1] + 0.1
            if x0[n+i] - x0[n+i-1] < 0.1:
                x0[n+i] = x0[n+i-1] + 0.1

        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_ratio and res.fun < 90:
                best_ratio = res.fun
                best_roots = res.x.copy()
        except Exception:
            pass

    if best_roots is not None:
        rp = np.sort(best_roots[:n])
        rq = np.sort(best_roots[n:])
        print(f"    best ratio = {best_ratio:.10f}")
        print(f"    roots_p = {rp}")
        print(f"    roots_q = {rq}")
        if best_ratio < 1.0:
            print(f"    *** VIOLATION FOUND! ratio = {best_ratio} < 1 ***")
    else:
        print(f"    (all optimization runs failed)")
    return best_ratio


# ============================================================
# Strategy 2: Differential evolution (global optimizer)
# ============================================================
def global_adversarial_test(n):
    """Use differential evolution for more thorough global search."""
    print(f"\n  n={n}: Differential evolution (global search)...")

    def objective(params):
        rp = np.sort(params[:n])
        rq = np.sort(params[n:])
        for i in range(1, n):
            if rp[i] - rp[i-1] < 0.01:
                return 100.0
            if rq[i] - rq[i-1] < 0.01:
                return 100.0
        ratio = surplus_ratio(rp, rq)
        if np.isnan(ratio):
            return 100.0
        return ratio

    bounds = [(-5, 5)] * (2 * n)
    try:
        res = differential_evolution(objective, bounds, maxiter=500,
                                     seed=42, tol=1e-10, popsize=30,
                                     mutation=(0.5, 1.5), recombination=0.9)
        print(f"    best ratio = {res.fun:.10f}")
        if res.fun < 1.0 and res.fun < 90:
            rp = np.sort(res.x[:n])
            rq = np.sort(res.x[n:])
            print(f"    *** VIOLATION FOUND! ***")
            print(f"    roots_p = {rp}")
            print(f"    roots_q = {rq}")
        return res.fun
    except Exception as e:
        print(f"    failed: {e}")
        return float('inf')


# ============================================================
# Strategy 3: Near-degenerate cases
# ============================================================
def near_degenerate_test(n, n_trials=2000):
    """Test with roots that are nearly coincident."""
    print(f"\n  n={n}: {n_trials} near-degenerate trials...")

    violations = 0
    min_ratio = float('inf')
    total = 0

    for trial in range(n_trials):
        # Strategy: base roots + small perturbation
        base_p = np.sort(np.random.randn(n) * 2)
        base_q = np.sort(np.random.randn(n) * 2)

        # Make some roots very close
        eps = 10 ** np.random.uniform(-6, -2)
        k = np.random.randint(1, n)
        base_p[k] = base_p[k-1] + eps
        k = np.random.randint(1, n)
        base_q[k] = base_q[k-1] + eps

        ratio = surplus_ratio(base_p, base_q)
        if np.isnan(ratio):
            continue
        total += 1
        if ratio < min_ratio:
            min_ratio = ratio
        if ratio < 1.0 - 1e-10:
            violations += 1
            print(f"    VIOLATION at trial {trial}: ratio={ratio:.10f}")
            print(f"      roots_p = {base_p}")
            print(f"      roots_q = {base_q}")

    print(f"    {violations}/{total} violations, min ratio = {min_ratio:.10f}")
    return violations, min_ratio


# ============================================================
# Strategy 4: Large random sample at higher n
# ============================================================
def large_random_test(n, n_trials=5000):
    """Large random sample test."""
    print(f"\n  n={n}: {n_trials} random trials...")

    violations = 0
    total = 0
    min_ratio = float('inf')
    ratios = []

    for _ in range(n_trials):
        rp = np.sort(np.random.randn(n) * 2)
        rq = np.sort(np.random.randn(n) * 2)
        for i in range(1, n):
            if rp[i] - rp[i-1] < 0.02:
                rp[i] = rp[i-1] + 0.02
            if rq[i] - rq[i-1] < 0.02:
                rq[i] = rq[i-1] + 0.02

        ratio = surplus_ratio(rp, rq)
        if np.isnan(ratio):
            continue
        total += 1
        ratios.append(ratio)
        if ratio < min_ratio:
            min_ratio = ratio
        if ratio < 1.0 - 1e-10:
            violations += 1

    ratios = np.array(ratios)
    print(f"    {violations}/{total} violations")
    print(f"    min={min_ratio:.10f}  p5={np.percentile(ratios,5):.6f}  "
          f"p1={np.percentile(ratios,1):.6f}  median={np.median(ratios):.6f}")
    return violations, min_ratio


# ============================================================
# Strategy 5: Extreme coefficient ratios
# ============================================================
def extreme_coefficient_test(n, n_trials=2000):
    """Test with very skewed root configurations."""
    print(f"\n  n={n}: {n_trials} extreme-ratio trials...")

    violations = 0
    total = 0
    min_ratio = float('inf')

    for _ in range(n_trials):
        # One polynomial with tightly clustered roots, one with spread out
        scale_p = 10 ** np.random.uniform(-2, 2)
        scale_q = 10 ** np.random.uniform(-2, 2)

        rp = np.sort(np.random.randn(n) * scale_p)
        rq = np.sort(np.random.randn(n) * scale_q)
        for i in range(1, n):
            if rp[i] - rp[i-1] < 0.01:
                rp[i] = rp[i-1] + 0.01
            if rq[i] - rq[i-1] < 0.01:
                rq[i] = rq[i-1] + 0.01

        ratio = surplus_ratio(rp, rq)
        if np.isnan(ratio):
            continue
        total += 1
        if ratio < min_ratio:
            min_ratio = ratio
        if ratio < 1.0 - 1e-10:
            violations += 1

    print(f"    {violations}/{total} violations, min ratio = {min_ratio:.10f}")
    return violations, min_ratio


if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("Problem 4 Stress Test: Can we break the inequality for n >= 4?")
    print("=" * 70)

    results = {}

    # --- Large random samples ---
    print("\n" + "=" * 70)
    print("STRATEGY 1: Large random samples")
    print("=" * 70)
    for n in [4, 5, 6, 7, 8]:
        trials = 5000 if n <= 6 else 2000
        v, mr = large_random_test(n, n_trials=trials)
        results[f'random-n{n}'] = {'violations': v, 'min_ratio': mr}

    # --- Near-degenerate ---
    print("\n" + "=" * 70)
    print("STRATEGY 2: Near-degenerate roots")
    print("=" * 70)
    for n in [4, 5, 6]:
        v, mr = near_degenerate_test(n, n_trials=2000)
        results[f'degenerate-n{n}'] = {'violations': v, 'min_ratio': mr}

    # --- Extreme coefficients ---
    print("\n" + "=" * 70)
    print("STRATEGY 3: Extreme coefficient ratios")
    print("=" * 70)
    for n in [4, 5, 6]:
        v, mr = extreme_coefficient_test(n, n_trials=2000)
        results[f'extreme-n{n}'] = {'violations': v, 'min_ratio': mr}

    # --- Adversarial optimization ---
    print("\n" + "=" * 70)
    print("STRATEGY 4: Adversarial optimization (Nelder-Mead)")
    print("=" * 70)
    for n in [4, 5, 6]:
        mr = adversarial_test(n, n_starts=40)
        results[f'adversarial-n{n}'] = {'min_ratio': mr}

    # --- Global optimization ---
    print("\n" + "=" * 70)
    print("STRATEGY 5: Global optimization (differential evolution)")
    print("=" * 70)
    for n in [4, 5]:
        mr = global_adversarial_test(n)
        results[f'global-n{n}'] = {'min_ratio': mr}

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    any_violation = False
    print(f"\n  {'test':30s} {'violations':>12s} {'min_ratio':>12s}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    for key, val in sorted(results.items()):
        v = val.get('violations', '-')
        mr = val.get('min_ratio', float('inf'))
        mr_str = f"{mr:.10f}" if mr < 90 else "N/A"
        print(f"  {key:30s} {str(v):>12s} {mr_str:>12s}")
        if isinstance(v, int) and v > 0:
            any_violation = True
        if isinstance(mr, float) and mr < 1.0:
            any_violation = True

    print()
    if any_violation:
        print("  *** VIOLATIONS FOUND — inequality may be FALSE for some n>=4 ***")
    else:
        print("  NO VIOLATIONS FOUND across all strategies.")
        print("  The inequality appears to hold for n=4,...,8 with high confidence.")
        min_all = min(v.get('min_ratio', float('inf')) for v in results.values())
        if min_all < 90:
            print(f"  Tightest case: ratio = {min_all:.10f} (surplus margin)")
