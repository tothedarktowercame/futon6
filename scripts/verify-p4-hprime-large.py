#!/usr/bin/env python3
"""Large-scale test of h'(0) ≤ 0 for n=4,5,6,7.

h'(0) = 4r(p⊞q) - 2(w_p·r_p + w_q·r_q)  where r = Ψ/Φ
= 4·Ψ_c/Φ_c - 2·(r_p/Φ_p + r_q/Φ_q)/(1/Φ_p + 1/Φ_q)

The question: for n ≥ 4, does h'(0) ≤ 0 ALWAYS hold?
If not, how do the violating pairs look?
"""

import numpy as np
from math import factorial
import sys

np.random.seed(2026)
sys.stdout.reconfigure(line_buffering=True)


def mss_convolve(a_coeffs, b_coeffs, n):
    c = np.zeros(n)
    for k in range(1, n + 1):
        s = 0.0
        for i in range(k + 1):
            j = k - i
            if i > n or j > n:
                continue
            ai = 1.0 if i == 0 else a_coeffs[i - 1]
            bj = 1.0 if j == 0 else b_coeffs[j - 1]
            w = (factorial(n - i) * factorial(n - j)) / (factorial(n) * factorial(n - k))
            s += w * ai * bj
        c[k - 1] = s
    return c


def score_field(roots):
    n = len(roots)
    S = np.zeros(n)
    for k in range(n):
        S[k] = sum(1.0 / (roots[k] - roots[j]) for j in range(n) if j != k)
    return S


def phi_n(roots):
    S = score_field(roots)
    return np.sum(S ** 2)


def psi_n(roots):
    S = score_field(roots)
    n = len(roots)
    total = 0.0
    for k in range(n):
        for j in range(k + 1, n):
            total += (S[k] - S[j]) ** 2 / (roots[k] - roots[j]) ** 2
    return total


def r_ratio(roots):
    return psi_n(roots) / phi_n(roots)


def coeffs_to_roots(coeffs):
    return np.sort(np.roots(np.concatenate([[1.0], coeffs])).real)


def is_real_rooted(coeffs, tol=1e-6):
    return np.max(np.abs(np.roots(np.concatenate([[1.0], coeffs])).imag)) < tol


def hermite_roots(n):
    from numpy.polynomial.hermite_e import hermeroots
    return np.sort(hermeroots([0] * n + [1]))


def hermite_convolve(p_roots, t, n):
    he_r = hermite_roots(n) * np.sqrt(t)
    he_coeffs = np.poly(he_r)[1:]
    p_coeffs = np.poly(p_roots)[1:]
    conv = mss_convolve(p_coeffs, he_coeffs, n)
    if not is_real_rooted(conv):
        return None
    return coeffs_to_roots(conv)


print("=" * 70)
print("LARGE-SCALE TEST: h'(0) ≤ 0?")
print("4000 random pairs per n, with varying scales")
print("=" * 70)

for n in [3, 4, 5, 6, 7]:
    violations = 0
    total = 0
    worst_ratio = 0  # max of 2r_c/wavg
    worst_pair = None

    for trial in range(4000):
        spread = np.random.uniform(0.3, 5.0)
        p = np.sort(np.random.randn(n) * spread)
        q = np.sort(np.random.randn(n) * spread)

        pq_coeffs = mss_convolve(np.poly(p)[1:], np.poly(q)[1:], n)
        if not is_real_rooted(pq_coeffs):
            continue
        pq_roots = coeffs_to_roots(pq_coeffs)

        rc = r_ratio(pq_roots)
        rp = r_ratio(p)
        rq = r_ratio(q)

        alpha = 1.0 / phi_n(p)
        beta = 1.0 / phi_n(q)
        wavg = (rp * alpha + rq * beta) / (alpha + beta)

        ratio = 2 * rc / wavg if wavg > 1e-15 else 0

        total += 1
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_pair = (p.copy(), q.copy(), ratio, 2*rc, wavg)

        if 2 * rc > wavg + 1e-10:
            violations += 1

    print(f"\n  n={n}: h'(0)>0 violations = {violations}/{total}"
          f" ({100*violations/total:.2f}%)")
    print(f"    worst 2r_c/wavg = {worst_ratio:.6f}")

    if worst_pair is not None and worst_ratio > 0.99:
        p, q, ratio, trc, wv = worst_pair
        print(f"    worst p = {p}")
        print(f"    worst q = {q}")
        print(f"    2r_c = {trc:.6f}, wavg = {wv:.6f}")

        # For the worst violator, check at small t > 0
        # Maybe h'(t) becomes ≤ 0 quickly?
        if violations > 0:
            pq_coeffs = mss_convolve(np.poly(p)[1:], np.poly(q)[1:], n)
            pq_roots = coeffs_to_roots(pq_coeffs)
            print(f"    h'(t) at small t for worst pair:")
            for t in [0.001, 0.01, 0.05, 0.1, 0.5]:
                r_pq_t = hermite_convolve(pq_roots, 2 * t, n)
                r_p_t = hermite_convolve(p, t, n)
                r_q_t = hermite_convolve(q, t, n)
                if r_pq_t is None or r_p_t is None or r_q_t is None:
                    continue
                rc_t = r_ratio(r_pq_t)
                rp_t = r_ratio(r_p_t)
                rq_t = r_ratio(r_q_t)
                a_t = 1.0/phi_n(r_p_t)
                b_t = 1.0/phi_n(r_q_t)
                wavg_t = (rp_t*a_t + rq_t*b_t)/(a_t + b_t)
                hprime_t = 4*rc_t - 2*wavg_t
                print(f"      t={t:.3f}: h'={hprime_t:.8f}")


# Also test: does h'(t) ≤ 0 for ALL t ≥ ε for some small ε?
print(f"\n{'='*70}")
print("h'(t) ≤ 0 for t ≥ ε?")
print("=" * 70)

for n in [3, 4, 5]:
    for t_min in [0.01, 0.1, 0.5]:
        t_grid = np.linspace(t_min, 20.0, 50)
        violations = 0
        total = 0

        for trial in range(200):
            p = np.sort(np.random.randn(n) * np.random.uniform(0.5, 3.0))
            q = np.sort(np.random.randn(n) * np.random.uniform(0.5, 3.0))

            pq_coeffs = mss_convolve(np.poly(p)[1:], np.poly(q)[1:], n)
            if not is_real_rooted(pq_coeffs):
                continue

            pq_roots = coeffs_to_roots(pq_coeffs)

            for t in t_grid:
                r_pq_t = hermite_convolve(pq_roots, 2 * t, n)
                r_p_t = hermite_convolve(p, t, n)
                r_q_t = hermite_convolve(q, t, n)
                if r_pq_t is None or r_p_t is None or r_q_t is None:
                    continue

                rc = r_ratio(r_pq_t)
                rp = r_ratio(r_p_t)
                rq = r_ratio(r_q_t)
                a = 1.0/phi_n(r_p_t)
                b = 1.0/phi_n(r_q_t)
                wavg = (rp*a + rq*b)/(a + b)

                total += 1
                if 4*rc - 2*wavg > 1e-10:
                    violations += 1

        if total > 0:
            print(f"  n={n}, t≥{t_min}: h'(t)>0 violations = {violations}/{total}"
                  f" ({100*violations/total:.2f}%)")


print(f"\n{'='*70}")
print("DONE")
print("=" * 70)
