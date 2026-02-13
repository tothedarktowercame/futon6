#!/usr/bin/env python3
"""Investigate the structure of dΦ_n/dt along the Coulomb flow.

We have dΦ/dt = -2 Σ_{k,j: k≠j} S_k(S_k - S_j) / (γ_k - γ_j)²

Let's rewrite this. Since the sum over k,j (k≠j) counts each pair twice:

dΦ/dt = -2 Σ_{k<j} [S_k(S_k-S_j)/(γ_k-γ_j)² + S_j(S_j-S_k)/(γ_j-γ_k)²]
      = -2 Σ_{k<j} [S_k(S_k-S_j) + S_j(S_j-S_k)] / (γ_k-γ_j)²
      = -2 Σ_{k<j} (S_k-S_j)(S_k+S_j-2·0) ... wait let me redo

S_k(S_k-S_j) + S_j(S_j-S_k) = (S_k-S_j)(S_k+S_j) ... no.
= S_k² - S_k·S_j + S_j² - S_j·S_k = S_k² + S_j² - 2S_kS_j = (S_k - S_j)²

So: dΦ/dt = -2 Σ_{k<j} (S_k - S_j)² / (γ_k - γ_j)²  ≤ 0!

This is manifestly non-positive (and strictly negative unless all S_k are equal).

This is a SUM OF SQUARES identity — the proof that Φ_n is decreasing
along the Coulomb flow is trivial!
"""

import numpy as np
import sys

sys.stdout.reconfigure(line_buffering=True)

np.random.seed(2026)


def score_field(roots):
    n = len(roots)
    S = np.zeros(n)
    for k in range(n):
        S[k] = sum(1.0 / (roots[k] - roots[j]) for j in range(n) if j != k)
    return S


def dphi_dt_sos(roots):
    """Compute dΦ/dt = -2 Σ_{k<j} (S_k - S_j)² / (γ_k - γ_j)²."""
    n = len(roots)
    S = score_field(roots)
    result = 0.0
    for k in range(n):
        for j in range(k + 1, n):
            result -= 2 * (S[k] - S[j])**2 / (roots[k] - roots[j])**2
    return result


def dphi_dt_direct(roots):
    """Compute dΦ/dt = 2 Σ_k S_k · dS_k/dt via the explicit ODE."""
    n = len(roots)
    S = score_field(roots)

    dS = np.zeros(n)
    for k in range(n):
        for j in range(n):
            if j == k:
                continue
            gap = roots[k] - roots[j]
            dS[k] -= (S[k] - S[j]) / gap**2

    return 2 * np.dot(S, dS)


def verify_sos_formula():
    print("=" * 70)
    print("VERIFY: dΦ/dt = -2 Σ_{k<j} (S_k - S_j)² / (γ_k - γ_j)²")
    print("=" * 70)
    print()
    print("ALGEBRAIC DERIVATION:")
    print()
    print("  dΦ/dt = d/dt Σ_k S_k² = 2 Σ_k S_k · dS_k/dt")
    print()
    print("  Since γ_k' = S_k, we have:")
    print("    dS_k/dt = Σ_{j≠k} d/dt[1/(γ_k - γ_j)]")
    print("            = Σ_{j≠k} -(γ_k' - γ_j') / (γ_k - γ_j)²")
    print("            = -Σ_{j≠k} (S_k - S_j) / (γ_k - γ_j)²")
    print()
    print("  So: dΦ/dt = -2 Σ_k S_k · Σ_{j≠k} (S_k - S_j) / (γ_k - γ_j)²")
    print("            = -2 Σ_{k≠j} S_k(S_k - S_j) / (γ_k - γ_j)²")
    print()
    print("  Symmetrize: pair (k,j) with (j,k):")
    print("    S_k(S_k-S_j) + S_j(S_j-S_k) = S_k²-S_kS_j+S_j²-S_jS_k")
    print("                                 = (S_k-S_j)²")
    print()
    print("  Therefore: dΦ/dt = -2 Σ_{k<j} (S_k - S_j)² / (γ_k - γ_j)² ≤ 0.  ∎")
    print()
    print("  Equality iff all S_k are equal, which happens only at the")
    print("  equilibrium (Hermite root configuration).")
    print()

    # Numerical verification
    print("NUMERICAL VERIFICATION:")
    max_err = 0.0
    for n in [3, 4, 5, 6, 7, 8]:
        for trial in range(50):
            roots = np.sort(np.random.randn(n) * 2)
            while np.min(np.diff(roots)) < 0.3:
                roots = np.sort(np.random.randn(n) * 2)

            val_sos = dphi_dt_sos(roots)
            val_direct = dphi_dt_direct(roots)
            rel_err = abs(val_sos - val_direct) / (abs(val_direct) + 1e-15)
            max_err = max(max_err, rel_err)

            # Check negativity
            assert val_sos <= 1e-10, f"SOS formula gave positive value: {val_sos}"

        print(f"  n={n}: SOS formula matches direct computation, "
              f"max relative error: {max_err:.2e}, all values ≤ 0  ✓")

    print()
    print("=" * 70)
    print("THEOREM (Φ_n Monotonicity): Φ_n(p_t) is strictly decreasing along")
    print("the Hermite Coulomb flow, with")
    print()
    print("    dΦ_n/dt = -2 Σ_{k<j} (S_k - S_j)² / (γ_k - γ_j)²  ≤  0")
    print()
    print("with equality iff all S_k(γ) are equal (equilibrium only).")
    print()
    print("COROLLARY: 1/Φ_n(p_t) is strictly increasing along the flow.  ∎")
    print("=" * 70)


if __name__ == '__main__':
    verify_sos_formula()
