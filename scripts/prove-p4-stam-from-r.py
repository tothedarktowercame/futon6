#!/usr/bin/env python3
"""Investigate whether "Stam for r" implies actual Stam for Φ.

KEY RESULTS SO FAR:
  - r(p⊞q) ≤ r(p)·r(q)/(r(p)+r(q))  [harmonic mean, "Stam for r"]
  - r(p⊞q) ≤ min(r(p), r(q))  [weaker but cleaner]
  - Both hold at 100% across n=3..10, 60000+ tests

THE CHALLENGE:
  h'(t) = 4r_c(2t) - 2·wavg  is NOT always ≤ 0
  The factor 4 (from semigroup p_t⊞q_t = (p⊞q)_{2t}) creates a gap.
  "Stam for r" gives 4r_c ≤ 4·harm(r_p,r_q), need 4r_c ≤ 2·wavg.

THIS SCRIPT INVESTIGATES:
  1. Does g(t) = 1/Φ_c(2t) - 1/Φ_p(t) - 1/Φ_q(t) have g'(t) ≤ 0?
     (actual Stam surplus, not log)
  2. Is there a STRONGER r bound that closes the factor-2 gap?
     e.g., r(p⊞q) ≤ r(p)·r(q)/(2·(r(p)+r(q)))?
  3. Alternative: direct proof via Cauchy-Schwarz + "Stam for r"?
  4. Test whether Ψ_c/Φ_c² ≤ (Ψ_p/Φ_p² + Ψ_q/Φ_q²)/2
     (this is the g'(t) ≤ 0 condition)
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


def W_func(roots):
    """W = Ψ/Φ² = r/Φ — controls g'(t) via g'=4W_c - 2W_p - 2W_q."""
    Phi = phi_n(roots)
    Psi = psi_n(roots)
    return Psi / Phi ** 2


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


def random_pair(n, scale_range=(0.3, 5.0)):
    s1 = np.random.uniform(*scale_range)
    s2 = np.random.uniform(*scale_range)
    p = np.sort(np.random.randn(n) * s1)
    q = np.sort(np.random.randn(n) * s2)
    return p, q


def convolve_pair(p, q, n):
    pq_coeffs = mss_convolve(np.poly(p)[1:], np.poly(q)[1:], n)
    if not is_real_rooted(pq_coeffs):
        return None
    return coeffs_to_roots(pq_coeffs)


# ═══════════════════════════════════════════════════════════════════
# TEST 1: Does g'(t) ≤ 0? (actual Stam surplus, not log)
# g(t) = 1/Φ_c(2t) - 1/Φ_p(t) - 1/Φ_q(t)
# g'(t) = 4W_c(2t) - 2W_p(t) - 2W_q(t) where W = Ψ/Φ²
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("TEST 1: g'(0) = 4W_c - 2W_p - 2W_q ≤ 0?  (W = Ψ/Φ²)")
print("This is the condition for the ACTUAL Stam surplus to decrease.")
print("=" * 70)

for n in [3, 4, 5, 6, 7]:
    violations = 0
    total = 0
    worst_ratio = 0

    for trial in range(5000):
        p, q = random_pair(n, (0.1, 5.0))
        pq = convolve_pair(p, q, n)
        if pq is None:
            continue

        Wc = W_func(pq)
        Wp = W_func(p)
        Wq = W_func(q)

        total += 1
        gprime = 4 * Wc - 2 * Wp - 2 * Wq
        ratio = 2 * Wc / (Wp + Wq) if (Wp + Wq) > 1e-30 else 0

        if ratio > worst_ratio:
            worst_ratio = ratio
        if gprime > 1e-10:
            violations += 1

    pct = 100 * violations / total if total > 0 else 0
    print(f"  n={n}: g'(0)>0 violations = {violations}/{total} ({pct:.2f}%)"
          f"  worst 2W_c/(W_p+W_q) = {worst_ratio:.6f}")


# ═══════════════════════════════════════════════════════════════════
# TEST 2: Is W = Ψ/Φ² subadditive under ⊞_n?
# i.e., W(p⊞q) ≤ W(p) + W(q)?
# This would give g'(t) ≤ 0 if 4W_c ≤ 2(W_p + W_q), i.e., 2W_c ≤ W_p + W_q.
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("TEST 2: W(p⊞q) vs W(p) + W(q)   and   2W(p⊞q) vs W(p) + W(q)")
print("=" * 70)

for n in [3, 4, 5, 6, 7]:
    sub_viol = 0  # W_c > W_p + W_q
    half_sub_viol = 0  # 2W_c > W_p + W_q
    total = 0

    for trial in range(5000):
        p, q = random_pair(n, (0.1, 5.0))
        pq = convolve_pair(p, q, n)
        if pq is None:
            continue

        Wc = W_func(pq)
        Wp = W_func(p)
        Wq = W_func(q)
        total += 1

        if Wc > Wp + Wq + 1e-10:
            sub_viol += 1
        if 2 * Wc > Wp + Wq + 1e-10:
            half_sub_viol += 1

    print(f"  n={n}: W_c > W_p+W_q: {sub_viol}/{total}"
          f"  |  2W_c > W_p+W_q: {half_sub_viol}/{total}")


# ═══════════════════════════════════════════════════════════════════
# TEST 3: Stam for W? i.e., 1/W(p⊞q) ≥ 1/W(p) + 1/W(q)?
# Also: W(p⊞q) ≤ min(W(p), W(q))?
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("TEST 3: Stam for W = Ψ/Φ²?  i.e., 1/W(p⊞q) ≥ 1/W(p) + 1/W(q)")
print("Also: W(p⊞q) ≤ min(W(p), W(q))?")
print("=" * 70)

for n in [3, 4, 5, 6, 7]:
    min_viol = 0
    stam_viol = 0
    total = 0

    for trial in range(5000):
        p, q = random_pair(n, (0.1, 5.0))
        pq = convolve_pair(p, q, n)
        if pq is None:
            continue

        Wc = W_func(pq)
        Wp = W_func(p)
        Wq = W_func(q)
        total += 1

        if Wc > min(Wp, Wq) + 1e-10:
            min_viol += 1
        # "Stam for W": 1/Wc ≥ 1/Wp + 1/Wq, i.e., Wc ≤ Wp·Wq/(Wp+Wq)
        harm = Wp * Wq / (Wp + Wq) if (Wp + Wq) > 1e-30 else 0
        if Wc > harm + 1e-10:
            stam_viol += 1

    print(f"  n={n}: W_c > min(W_p,W_q): {min_viol}/{total}"
          f"  |  Stam for W: {stam_viol}/{total}")


# ═══════════════════════════════════════════════════════════════════
# TEST 4: The "nuclear option" — test g'(t) along the full flow
# For the ACTUAL surplus g(t) = 1/Φ_c(2t) - 1/Φ_p(t) - 1/Φ_q(t),
# check if g is monotone decreasing for ALL t.
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("TEST 4: g(t) monotone decreasing along full flow?")
print("g(t) = 1/Φ_c(2t) - 1/Φ_p(t) - 1/Φ_q(t)")
print("=" * 70)

for n in [3, 4, 5, 6]:
    monotone_count = 0
    non_monotone_count = 0
    total = 0

    for trial in range(500):
        p, q = random_pair(n, (0.3, 3.0))
        pq = convolve_pair(p, q, n)
        if pq is None:
            continue

        t_vals = np.concatenate([[0], np.logspace(-2, 1, 30)])
        g_vals = []
        ok = True

        for t in t_vals:
            if t == 0:
                pt, qt, ct = p, q, pq
            else:
                pt = hermite_convolve(p, t, n)
                qt = hermite_convolve(q, t, n)
                ct = hermite_convolve(pq, 2 * t, n)
                if pt is None or qt is None or ct is None:
                    ok = False
                    break

            gval = 1.0 / phi_n(ct) - 1.0 / phi_n(pt) - 1.0 / phi_n(qt)
            g_vals.append(gval)

        if not ok or len(g_vals) < 10:
            continue

        total += 1
        is_mono = all(g_vals[i] >= g_vals[i + 1] - 1e-10 for i in range(len(g_vals) - 1))
        if is_mono:
            monotone_count += 1
        else:
            non_monotone_count += 1

    if total > 0:
        print(f"  n={n}: monotone ↓: {monotone_count}/{total}"
              f" ({100*monotone_count/total:.1f}%),"
              f" non-monotone: {non_monotone_count}/{total}")


# ═══════════════════════════════════════════════════════════════════
# TEST 5: Is there a STRONGER r inequality that gives the factor 2?
# Test: r(p⊞q) ≤ r(p)·r(q)/(2(r(p)+r(q))) = harm/2?
# This would give h'(t) ≤ 0 directly.
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("TEST 5: Stronger r bounds to close the factor-2 gap")
print("=" * 70)

for n in [3, 4, 5, 6, 7]:
    harm_half_viol = 0  # r_c > harm(r_p,r_q)/2
    total = 0
    worst_excess = 0

    for trial in range(5000):
        p, q = random_pair(n, (0.1, 5.0))
        pq = convolve_pair(p, q, n)
        if pq is None:
            continue

        rc = r_ratio(pq)
        rp = r_ratio(p)
        rq = r_ratio(q)
        total += 1

        harm = rp * rq / (rp + rq)
        excess = rc / (harm / 2) - 1 if harm > 1e-30 else 0
        if excess > worst_excess:
            worst_excess = excess
        if rc > harm / 2 + 1e-10:
            harm_half_viol += 1

    print(f"  n={n}: r_c > harm/2: {harm_half_viol}/{total}"
          f"  worst r_c/(harm/2) = {1+worst_excess:.6f}")


# ═══════════════════════════════════════════════════════════════════
# TEST 6: Stam DIRECT — does it hold? And what's the margin?
# 1/Φ(p⊞q) ≥ 1/Φ(p) + 1/Φ(q)
# Also test: the ratio (1/Φ_c - 1/Φ_p - 1/Φ_q) / (1/Φ_p + 1/Φ_q)
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("TEST 6: Direct Stam inequality check (for reference)")
print("1/Φ(p⊞q) ≥ 1/Φ(p) + 1/Φ(q)")
print("=" * 70)

for n in [3, 4, 5, 6, 7, 8]:
    violations = 0
    total = 0
    min_margin = float('inf')
    max_margin = 0

    for trial in range(5000):
        p, q = random_pair(n, (0.1, 5.0))
        pq = convolve_pair(p, q, n)
        if pq is None:
            continue

        lhs = 1.0 / phi_n(pq)
        rhs = 1.0 / phi_n(p) + 1.0 / phi_n(q)
        margin = (lhs - rhs) / rhs  # relative margin
        total += 1

        if margin < min_margin:
            min_margin = margin
        if margin > max_margin:
            max_margin = margin
        if lhs < rhs - 1e-12:
            violations += 1

    print(f"  n={n}: Stam violations: {violations}/{total},"
          f"  margin range: [{min_margin:.6f}, {max_margin:.6f}]")


# ═══════════════════════════════════════════════════════════════════
# TEST 7: Can we use "Stam for r" + Φ relationship to bound Stam?
# Key: Φ(p⊞q) vs Φ(p), Φ(q). What's the Φ relationship?
# If Φ_c ≤ Φ_p + Φ_q (or something similar), combined with r
# bounds, might give Stam.
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("TEST 7: Φ relationships under ⊞_n")
print("=" * 70)

for n in [3, 4, 5, 6, 7]:
    # Various Φ inequalities
    sub_add = 0  # Φ_c ≤ Φ_p + Φ_q
    harm_phi = 0  # 1/Φ_c ≥ 1/Φ_p + 1/Φ_q (actual Stam)
    harm_phi2 = 0  # 1/sqrt(Φ_c) ≥ 1/sqrt(Φ_p) + 1/sqrt(Φ_q)?
    min_phi = 0  # Φ_c ≤ min(Φ_p, Φ_q)
    hmean_phi = 0  # Φ_c ≤ harm(Φ_p, Φ_q)
    total = 0

    for trial in range(3000):
        p, q = random_pair(n, (0.1, 5.0))
        pq = convolve_pair(p, q, n)
        if pq is None:
            continue

        Fc = phi_n(pq)
        Fp = phi_n(p)
        Fq = phi_n(q)
        total += 1

        if Fc <= Fp + Fq + 1e-8:
            sub_add += 1
        if 1 / Fc >= 1 / Fp + 1 / Fq - 1e-10:
            harm_phi += 1
        if 1 / np.sqrt(Fc) >= 1 / np.sqrt(Fp) + 1 / np.sqrt(Fq) - 1e-10:
            harm_phi2 += 1
        if Fc <= min(Fp, Fq) + 1e-8:
            min_phi += 1
        harm = 2 * Fp * Fq / (Fp + Fq)
        if Fc <= harm + 1e-8:
            hmean_phi += 1

    print(f"  n={n} ({total} tests):")
    print(f"    Φ_c ≤ Φ_p+Φ_q:      {sub_add}/{total}")
    print(f"    Stam (1/Φ):          {harm_phi}/{total}")
    print(f"    1/√Φ superadditive:  {harm_phi2}/{total}")
    print(f"    Φ_c ≤ min(Φ_p,Φ_q): {min_phi}/{total}")
    print(f"    Φ_c ≤ harm(Φ_p,Φ_q):{hmean_phi}/{total}")


# ═══════════════════════════════════════════════════════════════════
# TEST 8: Is there a combined (Φ, r) inequality?
# Like: r_c · Φ_c ≤ r_p · Φ_p + r_q · Φ_q (Ψ subadditive)
# Or: r_c / Φ_c ≤ r_p / Φ_p + r_q / Φ_q  (W subadditive)
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("TEST 8: Combined (Φ, r) inequalities")
print("=" * 70)

for n in [3, 4, 5, 6, 7]:
    psi_sub = 0  # Ψ_c ≤ Ψ_p + Ψ_q
    W_sub = 0  # W_c ≤ W_p + W_q
    psi_harm = 0  # Ψ_c ≤ harm(Ψ_p, Ψ_q)
    psi_min = 0  # Ψ_c ≤ min(Ψ_p, Ψ_q)
    total = 0

    for trial in range(3000):
        p, q = random_pair(n, (0.1, 5.0))
        pq = convolve_pair(p, q, n)
        if pq is None:
            continue

        Psic = psi_n(pq)
        Psip = psi_n(p)
        Psiq = psi_n(q)
        Wc = W_func(pq)
        Wp = W_func(p)
        Wq = W_func(q)
        total += 1

        if Psic <= Psip + Psiq + 1e-8:
            psi_sub += 1
        if Wc <= Wp + Wq + 1e-10:
            W_sub += 1
        harm = 2 * Psip * Psiq / (Psip + Psiq) if Psip + Psiq > 0 else 0
        if Psic <= harm + 1e-8:
            psi_harm += 1
        if Psic <= min(Psip, Psiq) + 1e-8:
            psi_min += 1

    print(f"  n={n} ({total} tests):")
    print(f"    Ψ_c ≤ Ψ_p+Ψ_q:        {psi_sub}/{total}")
    print(f"    W_c ≤ W_p+W_q:          {W_sub}/{total}")
    print(f"    Ψ_c ≤ harm(Ψ_p,Ψ_q):   {psi_harm}/{total}")
    print(f"    Ψ_c ≤ min(Ψ_p,Ψ_q):    {psi_min}/{total}")


print(f"\n{'='*70}")
print("DONE")
print("=" * 70)
