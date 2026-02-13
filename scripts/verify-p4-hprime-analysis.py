#!/usr/bin/env python3
"""Analysis of h'(t) for the log-Stam monotonicity proof.

h(t) = log(1/Φ_c(2t)) - log(1/Φ_p(t) + 1/Φ_q(t))
     = -log Φ_c(2t) - log(1/Φ_p(t) + 1/Φ_q(t))

where Φ_c = Φ of (p⊞q)_{2t}, Φ_p = Φ of p_t, Φ_q = Φ of q_t.

Key formula: d/dt log(1/Φ(f_t)) = 2r(f_t) where r = Ψ/Φ and
  Ψ(f) = Σ_{k<j} (S_k - S_j)²/(γ_k - γ_j)²   (the "SOS rate")
  Φ(f) = Σ_k S_k²                                (root-force energy)

So: h'(t) = 4r_c(2t) - 2(r_p·α + r_q·β)/(α + β)
where α = 1/Φ_p, β = 1/Φ_q

The condition h'(t) ≤ 0 is:  2r_c ≤ (r_p·α + r_q·β)/(α + β)
i.e., 2 × (relative SOS rate of convolution at 2t) ≤ weighted average of
(relative SOS rates of parts at t), with weights proportional to 1/Φ.
"""

import numpy as np
from math import factorial
import sys

np.random.seed(42)
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
    """Ψ(f) = Σ_{k<j} (S_k - S_j)²/(γ_k - γ_j)²"""
    S = score_field(roots)
    n = len(roots)
    total = 0.0
    for k in range(n):
        for j in range(k + 1, n):
            total += (S[k] - S[j]) ** 2 / (roots[k] - roots[j]) ** 2
    return total


def r_ratio(roots):
    """r = Ψ/Φ — the relative SOS decay rate."""
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


# ═══════════════════════════════════════════════════════════════════
# TEST 1: Verify h'(t) formula and check sign
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("h'(t) = 4r_c(2t) - 2(r_p·α + r_q·β)/(α+β)")
print("where r = Ψ/Φ, α = 1/Φ_p, β = 1/Φ_q")
print("Need: h'(t) ≤ 0, equivalently 2r_c ≤ weighted_avg(r_p, r_q)")
print("=" * 70)

t_values = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])

for n in [3, 4, 5]:
    print(f"\n  n={n}:")
    hprime_neg_count = np.zeros(len(t_values), dtype=int)
    total_count = np.zeros(len(t_values), dtype=int)

    two_r_c_vals = {t: [] for t in t_values}
    wavg_r_vals = {t: [] for t in t_values}

    for trial in range(100):
        p = np.sort(np.random.randn(n) * np.random.uniform(0.5, 3.0))
        q = np.sort(np.random.randn(n) * np.random.uniform(0.5, 3.0))

        pq_coeffs = mss_convolve(np.poly(p)[1:], np.poly(q)[1:], n)
        if not is_real_rooted(pq_coeffs):
            continue
        pq_roots = coeffs_to_roots(pq_coeffs)

        for ti, t in enumerate(t_values):
            r_pq = hermite_convolve(pq_roots, 2 * t, n)
            r_p = hermite_convolve(p, t, n)
            r_q = hermite_convolve(q, t, n)
            if r_pq is None or r_p is None or r_q is None:
                continue

            rc = r_ratio(r_pq)
            rp = r_ratio(r_p)
            rq = r_ratio(r_q)

            phi_p = phi_n(r_p)
            phi_q = phi_n(r_q)
            alpha = 1.0 / phi_p
            beta = 1.0 / phi_q

            two_rc = 2 * rc
            wavg = (rp * alpha + rq * beta) / (alpha + beta)

            hprime = 4 * rc - 2 * wavg  # = 2*(2rc - wavg)

            total_count[ti] += 1
            if hprime <= 1e-10:
                hprime_neg_count[ti] += 1

            two_r_c_vals[t].append(two_rc)
            wavg_r_vals[t].append(wavg)

    print(f"    {'t':>6s} | h'≤0  | 2r_c mean  | wavg mean  | gap mean")
    print(f"    -------+-------+------------+------------+---------")
    for ti, t in enumerate(t_values):
        if total_count[ti] == 0:
            continue
        frac = hprime_neg_count[ti] / total_count[ti]
        m_2rc = np.mean(two_r_c_vals[t])
        m_wavg = np.mean(wavg_r_vals[t])
        gap = m_wavg - m_2rc
        print(f"    {t:6.2f} | {frac:5.1%} | {m_2rc:10.6f} | {m_wavg:10.6f} | {gap:+.6f}")


# ═══════════════════════════════════════════════════════════════════
# TEST 2: Verify h'(t) formula against finite differences
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("TEST 2: Verify h'(t) formula vs finite differences")
print("=" * 70)

dt = 0.001

for n in [3, 4]:
    print(f"\n  n={n}:")
    for trial in range(5):
        p = np.sort(np.random.randn(n) * 2)
        q = np.sort(np.random.randn(n) * 2)

        pq_coeffs = mss_convolve(np.poly(p)[1:], np.poly(q)[1:], n)
        if not is_real_rooted(pq_coeffs):
            continue
        pq_roots = coeffs_to_roots(pq_coeffs)

        t = 1.0

        # Finite difference h'(t)
        def h_at(t_val):
            r_pq = hermite_convolve(pq_roots, 2 * t_val, n)
            r_p = hermite_convolve(p, t_val, n)
            r_q = hermite_convolve(q, t_val, n)
            if r_pq is None or r_p is None or r_q is None:
                return None
            F = 1.0 / phi_n(r_pq)
            G = 1.0 / phi_n(r_p) + 1.0 / phi_n(r_q)
            return np.log(F) - np.log(G)

        h_plus = h_at(t + dt)
        h_minus = h_at(t - dt)
        if h_plus is None or h_minus is None:
            continue
        hprime_fd = (h_plus - h_minus) / (2 * dt)

        # Analytic formula
        r_pq_t = hermite_convolve(pq_roots, 2 * t, n)
        r_p_t = hermite_convolve(p, t, n)
        r_q_t = hermite_convolve(q, t, n)

        rc = r_ratio(r_pq_t)
        rp = r_ratio(r_p_t)
        rq = r_ratio(r_q_t)
        alpha = 1.0 / phi_n(r_p_t)
        beta = 1.0 / phi_n(r_q_t)
        wavg = (rp * alpha + rq * beta) / (alpha + beta)
        hprime_analytic = 4 * rc - 2 * wavg

        print(f"    trial {trial}: h'(fd)={hprime_fd:.8f}, h'(formula)={hprime_analytic:.8f},"
              f" ratio={hprime_fd/hprime_analytic:.6f}" if abs(hprime_analytic) > 1e-15 else
              f"    trial {trial}: h'(fd)={hprime_fd:.8f}, h'(formula)={hprime_analytic:.8f}")


# ═══════════════════════════════════════════════════════════════════
# TEST 3: Profile of r = Ψ/Φ along the Coulomb flow
# Is r monotone? Does r·t → constant (as suggested by Hermite scaling)?
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("TEST 3: r(t) = Ψ(f_t)/Φ(f_t) profile along Coulomb flow")
print("Hermite scaling predicts r ~ const/t for large t")
print("=" * 70)

t_prof = np.array([0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0])

for n in [3, 4, 5]:
    print(f"\n  n={n}:")
    for trial in range(3):
        p = np.sort(np.random.randn(n) * 2)

        r_vals = []
        r_times_t = []
        valid = True

        for t in t_prof:
            pt = hermite_convolve(p, t, n)
            if pt is None:
                valid = False
                break
            r = r_ratio(pt)
            r_vals.append(r)
            r_times_t.append(r * t)

        if not valid:
            continue

        print(f"    trial {trial}:")
        print(f"      r(t): [{', '.join(f'{v:.4f}' for v in r_vals)}]")
        print(f"      r·t:  [{', '.join(f'{v:.4f}' for v in r_times_t)}]")


# ═══════════════════════════════════════════════════════════════════
# TEST 4: Does the condition 2r_c ≤ wavg simplify for n=3?
# For n=3, try to understand what 2r_c ≤ wavg says algebraically.
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("TEST 4: Structure of Ψ/Φ for small n")
print("For n=3 centered with roots -a, 0, a: what is Ψ/Φ?")
print("=" * 70)

# n=3 centered: roots -a, 0, a
# S_1 = 1/(-a-0) + 1/(-a-a) = -1/a - 1/(2a) = -3/(2a)
# S_2 = 1/(0+a) + 1/(0-a) = 1/a - 1/a = 0
# S_3 = 1/(a-0) + 1/(a+a) = 1/a + 1/(2a) = 3/(2a)
# Φ = 9/(4a²) + 0 + 9/(4a²) = 9/(2a²)
# (S_1-S_2)/(γ_1-γ_2) = (-3/(2a))/(- a) = 3/(2a²)
# (S_1-S_3)/(γ_1-γ_3) = (-3/a)/(-2a) = 3/(2a²)
# (S_2-S_3)/(γ_2-γ_3) = (-3/(2a))/(-a) = 3/(2a²)
# Ψ = 3*(3/(2a²))² = 3*9/(4a⁴) = 27/(4a⁴)
# r = Ψ/Φ = (27/(4a⁴))/(9/(2a²)) = (27*2)/(4*9*a²) = 3/(2a²)

print("  For n=3, roots = {-a, 0, a}:")
print("    Φ = 9/(2a²)")
print("    Ψ = 27/(4a⁴)")
print("    r = Ψ/Φ = 3/(2a²)")
print("    So r·a² = 3/2 = constant!")
print()

for a in [0.5, 1.0, 2.0, 5.0]:
    roots = np.array([-a, 0.0, a])
    r = r_ratio(roots)
    print(f"    a={a}: r={r:.6f}, r·a²={r*a*a:.6f} (expect 1.5)")

print()

# For Hermite roots at scale √t:
# He_3 zeros are approximately -√3, 0, √3 (exact: -√3, 0, √3)
he3 = hermite_roots(3)
print(f"  He_3 roots: {he3}")
print(f"  r(He_3) = {r_ratio(he3):.6f}")
for t in [1.0, 2.0, 5.0, 10.0]:
    rt = he3 * np.sqrt(t)
    print(f"  r(He_3 × √{t}) = {r_ratio(rt):.6f}, r·t = {r_ratio(rt)*t:.6f}")


# ═══════════════════════════════════════════════════════════════════
# TEST 5: Large-scale test of the core inequality at t=0
# 2r_c ≤ (r_p·α + r_q·β)/(α+β) at t=0 (no flow needed)
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("TEST 5: Core inequality at t=0 (no flow)")
print("2·r(p⊞q) ≤ w_p·r(p) + w_q·r(q)")
print("where w_p = (1/Φ_p)/(1/Φ_p + 1/Φ_q)")
print("=" * 70)

for n in [3, 4, 5, 6]:
    violations = 0
    total = 0
    ratios = []  # 2r_c / wavg, should be ≤ 1

    for trial in range(500):
        p = np.sort(np.random.randn(n) * np.random.uniform(0.5, 3.0))
        q = np.sort(np.random.randn(n) * np.random.uniform(0.5, 3.0))

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

        ratio = 2 * rc / wavg if wavg > 1e-15 else float('nan')
        ratios.append(ratio)

        total += 1
        if 2 * rc > wavg + 1e-10:
            violations += 1

    r = np.array(ratios)
    print(f"  n={n}: violations={violations}/{total} ({100*violations/total:.1f}%),"
          f" max(2r_c/wavg)={r.max():.6f}, mean={r.mean():.6f}")
    if violations == 0:
        print(f"    *** INEQUALITY HOLDS AT t=0 ***")


# ═══════════════════════════════════════════════════════════════════
# TEST 6: What is h'(t) at t=0 specifically?
# h'(0) = 4r(p⊞q) - 2(r_p/Φ_p + r_q/Φ_q)/(1/Φ_p + 1/Φ_q)
# Does h'(0) ≤ 0 always?
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("TEST 6: Sign of h'(0) = 4r_c - 2·wavg at t=0")
print("=" * 70)

for n in [3, 4, 5, 6]:
    neg = 0
    pos = 0
    total = 0

    for trial in range(500):
        p = np.sort(np.random.randn(n) * np.random.uniform(0.5, 3.0))
        q = np.sort(np.random.randn(n) * np.random.uniform(0.5, 3.0))

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

        hprime0 = 4 * rc - 2 * wavg

        total += 1
        if hprime0 <= 1e-10:
            neg += 1
        else:
            pos += 1

    print(f"  n={n}: h'(0)≤0 in {neg}/{total} ({100*neg/total:.1f}%),"
          f" h'(0)>0 in {pos}/{total} ({100*pos/total:.1f}%)")


print(f"\n{'='*70}")
print("ALL TESTS COMPLETE")
print("=" * 70)
