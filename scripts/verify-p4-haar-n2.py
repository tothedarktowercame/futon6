#!/usr/bin/env python3
"""Quick check: n=2 case and ratio scaling with n."""
import numpy as np
from scipy.stats import unitary_group
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
_mod = __import__("verify-p4-inequality")
phi_n = _mod.phi_n
finite_free_conv = _mod.finite_free_conv
roots_to_coeffs = _mod.roots_to_coeffs
coeffs_to_roots = _mod.coeffs_to_roots


def make_distinct(roots, min_gap=0.02):
    roots = np.sort(roots.real)
    for i in range(1, len(roots)):
        if roots[i] - roots[i-1] < min_gap:
            roots[i] = roots[i-1] + min_gap
    return roots


def explore(roots_p, roots_q, n_samples=3000):
    n = len(roots_p)
    A = np.diag(roots_p.astype(float))
    B = np.diag(roots_q.astype(float))

    coeffs_p = roots_to_coeffs(roots_p)
    coeffs_q = roots_to_coeffs(roots_q)
    coeffs_conv = finite_free_conv(coeffs_p, coeffs_q)
    roots_conv = coeffs_to_roots(coeffs_conv)
    roots_conv = make_distinct(roots_conv)

    phi_conv = phi_n(roots_conv)
    phi_p = phi_n(roots_p)
    phi_q = phi_n(roots_q)

    if phi_conv in (0, float('inf')) or phi_p in (0, float('inf')) or phi_q in (0, float('inf')):
        return None

    inv_phis = []
    for _ in range(n_samples):
        Q = unitary_group.rvs(n)
        M = A + Q @ B @ Q.conj().T
        eigs = make_distinct(np.linalg.eigvalsh(M))
        v = phi_n(eigs)
        if v not in (0, float('inf')):
            inv_phis.append(1.0 / v)

    if len(inv_phis) < 100:
        return None

    return {
        'inv_phi_conv': 1.0 / phi_conv,
        'E_inv_phi': np.mean(inv_phis),
        'ratio': (1.0 / phi_conv) / np.mean(inv_phis),
        'inv_phi_p': 1.0 / phi_p,
        'inv_phi_q': 1.0 / phi_q,
    }


if __name__ == "__main__":
    np.random.seed(42)

    # n=2 detailed
    print("=== n=2 case (should show equality) ===\n")
    for trial in range(5):
        rp = make_distinct(np.random.randn(2) * 2)
        rq = make_distinct(np.random.randn(2) * 2)
        r = explore(rp, rq, n_samples=3000)
        if r:
            print(f"  trial {trial}: 1/Phi(conv)={r['inv_phi_conv']:.6f}  "
                  f"E[1/Phi]={r['E_inv_phi']:.6f}  "
                  f"ratio={r['ratio']:.6f}  "
                  f"sum={r['inv_phi_p']+r['inv_phi_q']:.6f}")

    # Ratio scaling with n
    print("\n=== Ratio 1/Phi(conv) / E[1/Phi(M)] vs n ===\n")
    for n in [2, 3, 4, 5]:
        ratios = []
        for _ in range(100):
            rp = make_distinct(np.random.randn(n) * 2)
            rq = make_distinct(np.random.randn(n) * 2)
            r = explore(rp, rq, n_samples=500)
            if r:
                ratios.append(r['ratio'])
        if ratios:
            ratios = np.array(ratios)
            print(f"  n={n}: mean ratio = {ratios.mean():.4f} ± {ratios.std():.4f}  "
                  f"(min={ratios.min():.4f}, max={ratios.max():.4f})")

    # Key test: does 1/Phi(conv) = E[1/Phi(M)] + correction?
    # For n=2 the ratio should be 1.0 (equality)
    # For n>2 the ratio grows — what's the correction?
    print("\n=== Decomposition test: 1/Phi(conv) - E[1/Phi(M)] ===\n")
    for n in [2, 3, 4]:
        diffs = []
        sum_invs = []
        for _ in range(100):
            rp = make_distinct(np.random.randn(n) * 2)
            rq = make_distinct(np.random.randn(n) * 2)
            r = explore(rp, rq, n_samples=500)
            if r:
                diffs.append(r['inv_phi_conv'] - r['E_inv_phi'])
                sum_invs.append(r['inv_phi_p'] + r['inv_phi_q'])
        if diffs:
            diffs = np.array(diffs)
            sum_invs = np.array(sum_invs)
            # Is the correction proportional to sum(1/Phi)?
            if np.std(sum_invs) > 0:
                corr = np.corrcoef(diffs, sum_invs)[0, 1]
            else:
                corr = float('nan')
            print(f"  n={n}: mean diff = {diffs.mean():.6f} ± {diffs.std():.6f}  "
                  f"corr(diff, sum(1/Phi)) = {corr:.4f}")
