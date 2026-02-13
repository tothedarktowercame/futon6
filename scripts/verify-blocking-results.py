#!/usr/bin/env python3
"""Verify the four blocking results for Problem 6 GPL-H.

BR1: Operator inequality M+F<=Pi does NOT imply rho_1 <= 1/2.
BR2: Per-edge alpha_uv -> 1/2 for K_n (no universal per-edge bound).
BR3: Interlacing failure witness from Codex C4 data.
BR4: Schur-convexity failure: concentrated eigenvalue exceeds uniform.
"""

import json
import numpy as np

print("=" * 80)
print("BLOCKING RESULT 1: Operator inequality insufficient")
print("=" * 80)

# Counterexample in R^3, Pi = I - (1/3)J (rank 2)
# Work in range(Pi). Take orthonormal basis e1, e2 for range(Pi).
# M = a * |e1><e1|, F = (1-a)|e1><e1| + b|e2><e2|
# M + F = |e1><e1| + b|e2><e2| <= Pi (need b <= 1)
# rho_1 = (1-a)/(1-a+b)

print("\nFamily: M=a|e1><e1|, F=(1-a)|e1><e1|+b|e2><e2|, M+F<=Pi")
print(f"{'a':>6} {'b':>6} {'rho_1':>8} {'>1/2?':>6} {'alpha':>8}")
print("-" * 40)
for a in [0.3, 0.5, 0.7, 0.9]:
    for b in [0.01, 0.1, 0.3, 0.5, 1.0]:
        rho_1 = (1 - a) / (1 - a + b)
        alpha = (1 - a) / (1 - a + b)  # same as rho_1 when M has one eigenvalue
        exceeds = "YES" if rho_1 > 0.5 else "no"
        if rho_1 > 0.5:
            print(f"{a:>6.2f} {b:>6.2f} {rho_1:>8.4f} {exceeds:>6} {alpha:>8.4f}")

# Most extreme: b=0, F entirely in col(M)
print("\nExtreme case: a=0.5, b=0 (F entirely in col(M))")
print("M = 0.5*|e1><e1|, F = 0.5*|e1><e1|, M+F = |e1><e1| <= Pi")
print("rho_1 = 0.5/0.5 = 1.0")
print("alpha = tr(P_M F)/tr(F) = 0.5/0.5 = 1.0")

# Verify with actual matrices
n = 3
Pi = np.eye(n) - np.ones((n, n)) / n
# Eigenvectors of Pi: any ONB of range(Pi)
eigvals, eigvecs = np.linalg.eigh(Pi)
e1 = eigvecs[:, 1]  # first nonzero eigenvector
e2 = eigvecs[:, 2]  # second nonzero eigenvector

a_val = 0.5
b_val = 0.1
M = a_val * np.outer(e1, e1)
F = (1 - a_val) * np.outer(e1, e1) + b_val * np.outer(e2, e2)

# Check M + F <= Pi
gap = Pi - M - F
print(f"\nNumerical verification (a={a_val}, b={b_val}):")
print(f"  eigenvalues of Pi - M - F: {np.linalg.eigvalsh(gap)}")
print(f"  all >= 0: {np.all(np.linalg.eigvalsh(gap) >= -1e-10)}")
norm_M = np.linalg.norm(M, ord=2)
tr_MF = np.trace(M @ F)
tr_F = np.trace(F)
rho1 = tr_MF / (norm_M * tr_F)
print(f"  rho_1 = {rho1:.6f} (expected {(1-a_val)/(1-a_val+b_val):.6f})")
print(f"  rho_1 > 1/2: {rho1 > 0.5}")

print("\n" + "=" * 80)
print("BLOCKING RESULT 2: Per-edge alpha_uv -> 1/2 for K_n")
print("=" * 80)

print("\nFor K_n at step t: every cross-edge has alpha_uv = (t-1)/(2t)")
print(f"{'n':>6} {'eps':>6} {'T':>6} {'alpha_uv':>10} {'margin':>10}")
print("-" * 44)
for n in [40, 80, 200, 500, 1000, 10000]:
    for eps in [0.3, 0.5]:
        T = int(eps * n / 3)
        if T < 2:
            continue
        alpha_uv = (T - 1) / (2 * T)
        margin = 0.5 - alpha_uv
        print(f"{n:>6d} {eps:>6.1f} {T:>6d} {alpha_uv:>10.6f} {margin:>10.6f}")

print("\nAs n -> inf: alpha_uv -> 1/2. No universal c < 1/2 works as per-edge bound.")

print("\n" + "=" * 80)
print("BLOCKING RESULT 3: Interlacing failure for vertex selection")
print("=" * 80)

# Load Codex C4 data for interlacing failures
try:
    with open("/home/joe/code/futon6/data/first-proof/problem6-codex-cycle4-results.json") as f:
        c4 = json.load(f)

    summary = c4["summary"]
    t4 = summary["task4_interlacing"]
    print(f"\nInterlacing probe results (Codex C4):")
    print(f"  Total trials: {t4['interlace_trials_total']}")
    print(f"  Passed: {t4['interlace_trials_pass']}")
    print(f"  Pass rate: {t4['interlace_pass_rate']:.4f}")
    print(f"  Q real-rooted failures: {t4['Q_real_rooted_failures']}")

    # Show specific failure witnesses
    failures = t4.get("interlace_failures", [])
    print(f"\n  Worst interlacing failure witnesses ({len(failures)} total):")
    for f in failures[:6]:
        print(f"    {f['graph']} eps={f['eps']} t={f['t']}: "
              f"{f['interlace_pass']}/{f['trials']} pass")

    # Q real-rooted failures
    qfails = t4.get("Q_real_rooted_failure_examples", [])
    if qfails:
        print(f"\n  Q not real-rooted examples ({len(qfails)} shown):")
        for q in qfails[:5]:
            print(f"    {q['graph']} eps={q['eps']} t={q['t']}: max_imag={q['max_imag']:.2e}")
except Exception as e:
    print(f"  Could not load C4 data: {e}")
    print("  Structural argument: Y_t(v) share edges (non-independent atoms)")

print("\n  Structural reason: Y_t(v) = B^{1/2} C_t(v) B^{1/2} where")
print("  C_t(v) = sum_{u in S, u~v} X_{uv}. If u is adjacent to both v1, v2,")
print("  then C_t(v1) and C_t(v2) share the X_{u,v1} resp X_{u,v2} component")
print("  through the common u-vertex. This correlation breaks the independence")
print("  required for interlacing families (MSS 2015, Theorem 4.4).")

print("\n" + "=" * 80)
print("BLOCKING RESULT 4: Schur-convexity failure at M != 0")
print("=" * 80)

print("\nAt M=0: dbar = tr(F)/(eps*r). This depends on leverage sums, which")
print("are controlled by Foster's theorem. K_n (uniform leverage) maximizes.")
print("\nAt M!=0: dbar = tr(B*F)/r where B = (eps*I - M)^{-1}.")
print("The amplification 1/(eps - mu_i) is CONVEX in mu_i.")
print("Convex + Schur would say: concentrated > uniform (opposite of what we want).")
print("\nExplicit computation:")
print("  Consider r eigenvalues of M, with F_ii = 1-mu_i (tight).")
print("  Contribution to dbar from col(M): sum (1-mu_i)/(eps-mu_i)")
print("  Compare uniform (all mu_i = tau/r) vs concentrated (one mu_i = tau, rest 0):")

eps = 0.5
for r, tau in [(5, 0.3), (10, 0.5), (12, 0.6)]:
    # Uniform
    mu_unif = tau / r
    dbar_unif = r * (1 - mu_unif) / (eps - mu_unif)

    # Concentrated: one eigenvalue = tau, rest = 0
    # But need tau < eps for barrier. If tau >= eps, concentrated is infeasible.
    if tau < eps:
        dbar_conc = (1 - tau) / (eps - tau) + (r - 1) * 1 / eps
    else:
        dbar_conc = float('inf')

    ratio = dbar_conc / dbar_unif if dbar_unif > 0 else float('nan')
    print(f"  r={r:>2d}, tau={tau:.1f}, eps={eps}: "
          f"uniform={dbar_unif:.4f}, concentrated={dbar_conc:.4f}, "
          f"ratio={ratio:.4f} {'CONCENTRATED WINS' if ratio > 1 else 'uniform wins'}")

print("\nConcentrated eigenvalue gives HIGHER dbar than uniform.")
print("Schur-convexity (which would say uniform is worst) is FALSE for dbar at M!=0.")
print("Therefore: cannot prove dbar_G <= dbar_Kn via Schur-convexity of leverage structure.")
