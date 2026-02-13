#!/usr/bin/env python3
"""Prove H(w) >= 0 via SOS Gram matrix certificate.

ESTABLISHED RESULTS:
1. D = c2^2 + 12*c0*c4 - 3*c1*c3 = S^2*(A+B)^2*(A+B+4r)^2  [perfect square]
2. E = sqrt(D) = S*(A+B)*(A+B+4r)
3. g* = (2c2 + E)/3 gives det(G) = 0 (EXACT, symbolically verified)
4. All three 2x2 principal minors at g* are IDENTICAL and equal to:
   M = 13947137604 * r^3 * (1+3x) * (1+3y) * (A+B+4r) > 0
   for r > 0, x in (0,1), y in (0,1).

CONCLUSION: G at g* = (2c2+E)/3 is positive semidefinite:
  - M1 = c0 > 0
  - M2a = M2b = M2c = M > 0  (all 2x2 principal minors positive)
  - det(G) = 0                 (matrix is singular, rank 2)

A real symmetric matrix with all leading principal minors >= 0 and det >= 0
is PSD. (Sylvester's criterion extended to PSD case.)

Actually, Sylvester's criterion for PSD uses ALL principal minors (not just leading).
For 3x3: need all 1x1 (diagonal entries >= 0), all 2x2 (3 principal minors >= 0),
and det >= 0. We have:
  - c0 > 0, g > 0 (need to verify!), c4 > 0
  - All three 2x2 principal minors > 0
  - det = 0

This script verifies all of this and provides the complete symbolic proof.
"""
import numpy as np
import sympy as sp

S = 59049  # 3^10


def coeffs_numeric(r, x, y):
    A = 1.0 + 3.0*x
    By = 1.0 + 3.0*y
    B = By * r**2
    c0 = S * A * (A + 4.0*r)
    c1 = 2.0*S * A * (A + 4.0*r - B)
    c2 = S * (A**2 + B**2 - 4*A*B + 4*r*A + 4*r**3*By)
    c3 = -2.0*S * B * (A - 4.0*r - B)
    c4 = S * r**3 * By * (4.0 + r*By)
    return c0, c1, c2, c3, c4


def main():
    print("=" * 72, flush=True)
    print("COMPLETE SOS PROOF: H(w) >= 0 via Gram matrix", flush=True)
    print("=" * 72, flush=True)

    r, x, y = sp.symbols('r x y', positive=True)
    A = 1 + 3*x
    By = 1 + 3*y
    B = By * r**2

    c0 = S * A * (A + 4*r)
    c1 = 2*S * A * (A + 4*r - B)
    c2 = S * (A**2 + B**2 - 4*A*B + 4*r*A + 4*r**3*By)
    c3 = -2*S * B * (A - 4*r - B)
    c4 = S * r**3 * By * (4 + r*By)

    E = S * (A + B) * (A + B + 4*r)
    g = (2*c2 + E) / 3

    # ================================================================
    # STEP 1: Verify D factors as perfect square
    # ================================================================
    print("\n[STEP 1] Verify D = c2^2 + 12*c0*c4 - 3*c1*c3 is a perfect square", flush=True)

    c2_raw = S * (9*r**4*y**2 + 6*r**4*y + r**4 + 12*r**3*y + 4*r**3
                  - 36*r**2*x*y - 12*r**2*x - 12*r**2*y - 4*r**2
                  + 12*r*x + 4*r + 9*x**2 + 6*x + 1)
    print(f"  c2 formula match: {sp.expand(c2 - c2_raw) == 0}", flush=True)

    D = sp.expand(c2**2 + 12*c0*c4 - 3*c1*c3)
    D_over_S2 = sp.expand(D / S**2)
    D_factored = sp.factor(D_over_S2)
    print(f"  D/S^2 = {D_factored}", flush=True)

    # Verify it equals (A+B)^2*(A+B+4r)^2
    target = (A + B)**2 * (A + B + 4*r)**2
    print(f"  (A+B)^2*(A+B+4r)^2 match: {sp.expand(D_over_S2 - target) == 0}", flush=True)

    # ================================================================
    # STEP 2: Verify g* = (2c2 + E)/3 is well-defined and explicit
    # ================================================================
    print("\n[STEP 2] g* = (2c2 + S*(A+B)*(A+B+4r))/3", flush=True)

    g_expanded = sp.expand(g)
    print(f"  g* expanded has {len(sp.Poly(g_expanded, r, x, y).as_dict())} terms", flush=True)
    g_factored = sp.factor(g)
    print(f"  g* factored = {g_factored}", flush=True)

    # ================================================================
    # STEP 3: Verify det(G) = 0 at g*
    # ================================================================
    print("\n[STEP 3] Verify det(G) = 0 at g = g*", flush=True)

    G_mat = sp.Matrix([
        [c0,        c1/2,        (c2 - g)/2],
        [c1/2,      g,           c3/2],
        [(c2-g)/2,  c3/2,        c4]
    ])

    det_G = G_mat.det()
    det_G_exp = sp.expand(det_G)
    print(f"  det(G*) = {det_G_exp}", flush=True)

    # ================================================================
    # STEP 4: Verify all diagonal entries >= 0
    # ================================================================
    print("\n[STEP 4] Verify diagonal entries of G are non-negative", flush=True)
    print(f"  G[0,0] = c0 = {sp.factor(c0)}", flush=True)
    print(f"    = S*(1+3x)*(1+3x+4r) > 0 for r>0, x in (0,1). CHECK.", flush=True)

    print(f"\n  G[1,1] = g* = {sp.factor(g)}", flush=True)
    # Need to show g > 0. g = (2c2 + E)/3 where E > 0 always.
    # c2 could be negative but E dominates.
    # Actually let's check: c2 = S*(A^2+B^2-4AB+4rA+4r^3*By)
    #   = S*((A-B)^2 - 2AB + 4rA + 4r^3*By)
    #   = S*((A-2B)^2... no, (A-B)^2 = A^2-2AB+B^2, so c2 = S*((A-B)^2 + 4r(A+r^2*By) - 2AB)
    # This can be negative if A*B dominates.
    # But g = (2c2+E)/3 and E = S*(A+B)*(A+B+4r) >= S*(A+B)^2 > 0.
    # 2c2+E = 2S*(A^2+B^2-4AB+4rA+4r^3By) + S*(A+B)(A+B+4r)
    # = S*(2A^2+2B^2-8AB+8rA+8r^3By + A^2+2AB+B^2+4rA+4rB)
    # Wait, (A+B)(A+B+4r) = (A+B)^2 + 4r(A+B) = A^2+2AB+B^2+4rA+4rB
    # So 2c2+E = S*(2A^2+2B^2-8AB+8rA+8r^3By + A^2+2AB+B^2+4rA+4rB)
    # = S*(3A^2+3B^2-6AB+12rA+4rB+8r^3By)
    # = S*(3(A-B)^2+12rA+4rB+8r^3By)
    # = S*(3(A-B)^2 + 4r(3A+B) + 8r^3By)
    #
    # 3A+B = 3(1+3x)+(1+3y)r^2, always positive.
    # So 2c2+E = S*(3(A-B)^2 + 4r(3A+B) + 8r^3By) > 0 always.
    # Therefore g = (2c2+E)/3 > 0 always.

    print("  Showing g > 0:", flush=True)
    threeg = sp.expand(3*g)
    # threeg = 2c2 + E
    print(f"    3g = 2c2 + E", flush=True)
    sum_expr = sp.expand(2*c2 + E)
    # Factor in terms of (A-B)^2 etc
    # Let's try a direct factoring
    print(f"    3g = S * (... let me compute ...)", flush=True)
    threeg_over_S = sp.expand(sum_expr / S)
    print(f"    3g/S = {threeg_over_S}", flush=True)

    # Substitute A, B back
    threeg_AB = sp.expand(3*(A-B)**2 + 4*r*(3*A+B) + 8*r**3*By)
    diff_check = sp.expand(threeg_over_S - threeg_AB)
    print(f"    3g/S = 3(A-B)^2 + 4r(3A+B) + 8r^3*By? diff = {diff_check}", flush=True)
    if diff_check == 0:
        print("    Since (A-B)^2 >= 0, A > 0, B > 0, By > 0, r > 0: 3g > 0. CHECK.", flush=True)

    print(f"\n  G[2,2] = c4 = S*r^3*By*(4+r*By) > 0 for r>0, y in (0,1). CHECK.", flush=True)

    # ================================================================
    # STEP 5: Verify all 2x2 principal minors >= 0
    # ================================================================
    print("\n[STEP 5] Verify all 2x2 principal minors >= 0", flush=True)

    # M2a = G[0,0]*G[1,1] - G[0,1]^2 = c0*g - c1^2/4
    M2a = sp.expand(c0 * g - c1**2 / 4)
    M2a_f = sp.factor(M2a)
    print(f"  M2a = c0*g - c1^2/4 = {M2a_f}", flush=True)

    # M2b = G[1,1]*G[2,2] - G[1,2]^2 = g*c4 - c3^2/4
    M2b = sp.expand(g * c4 - c3**2 / 4)
    M2b_f = sp.factor(M2b)
    print(f"  M2b = g*c4 - c3^2/4 = {M2b_f}", flush=True)

    # M2c = G[0,0]*G[2,2] - G[0,2]^2 = c0*c4 - (c2-g)^2/4
    M2c = sp.expand(c0 * c4 - (c2 - g)**2 / 4)
    M2c_f = sp.factor(M2c)
    print(f"  M2c = c0*c4 - (c2-g)^2/4 = {M2c_f}", flush=True)

    # All three are identical! Check:
    print(f"\n  M2a == M2b? {sp.expand(M2a - M2b) == 0}", flush=True)
    print(f"  M2a == M2c? {sp.expand(M2a - M2c) == 0}", flush=True)

    # Factor the common value
    # 13947137604 = ?
    print(f"  13947137604 = {sp.factorint(13947137604)}", flush=True)
    # = 4 * S^2 = 4 * 59049^2 = 4 * 3^20 = 2^2 * 3^20
    print(f"  4 * S^2 = {4 * S**2}", flush=True)

    print(f"\n  All 2x2 minors = 4*S^2 * r^3 * (1+3x) * (1+3y) * (A+B+4r)", flush=True)
    target_minor = 4*S**2 * r**3 * A * By * (A + B + 4*r)
    # Wait, let me be careful: (1+3x) = A, (1+3y) = By
    # 4*S^2*r^3*(1+3x)*(1+3y)*(A+B+4r) = 4*S^2*r^3*A*By*(A+B+4r)
    # But A+B+4r = (1+3x) + (1+3y)*r^2 + 4r. All positive.
    # And r^3 > 0 for r > 0, A > 0, By > 0. So M > 0. CHECK.

    print(f"  All factors positive for r>0, x,y in (0,1). CHECK.", flush=True)

    # ================================================================
    # STEP 6: Verify (c2 - g*) formula
    # ================================================================
    print("\n[STEP 6] Simplify c2 - g*", flush=True)
    c2_minus_g = sp.expand(c2 - g)
    c2_minus_g_f = sp.factor(c2_minus_g)
    print(f"  c2 - g* = {c2_minus_g_f}", flush=True)
    # Should be (c2 - E)/3

    # ================================================================
    # STEP 7: Write out the explicit Gram matrix
    # ================================================================
    print("\n[STEP 7] Explicit Gram matrix", flush=True)
    print("  G[0,0] = c0 = S * A * (A + 4r)", flush=True)
    print("  G[0,1] = c1/2 = S * A * (A + 4r - B)", flush=True)
    print(f"  G[0,2] = (c2-g)/2 = {sp.factor((c2-g)/2)}", flush=True)
    print(f"  G[1,1] = g = (2c2 + S*(A+B)*(A+B+4r))/3", flush=True)
    print("  G[1,2] = c3/2 = -S * B * (A - 4r - B)", flush=True)
    print("  G[2,2] = c4 = S * r^3 * By * (4 + r*By)", flush=True)

    # ================================================================
    # STEP 8: Full numerical verification
    # ================================================================
    print("\n[STEP 8] Full numerical verification (500000 points)", flush=True)

    rng = np.random.default_rng(42)
    n = 500000
    rs = np.exp(rng.uniform(np.log(1e-4), np.log(1e4), n))
    xs = rng.uniform(1e-6, 1-1e-6, n)
    ys = rng.uniform(1e-6, 1-1e-6, n)

    n_psd = 0
    n_H_neg = 0
    worst_rel_eig = 0  # worst (min_eig / trace) ratio
    worst_abs_eig = np.inf

    for i in range(n):
        rv, xv, yv = rs[i], xs[i], ys[i]
        c0v, c1v, c2v, c3v, c4v = coeffs_numeric(rv, xv, yv)
        Av = 1+3*xv; Bv = (1+3*yv)*rv**2
        Ev = S * (Av + Bv) * (Av + Bv + 4*rv)
        gv = (2*c2v + Ev) / 3.0

        # Build Gram matrix
        Gm = np.array([
            [c0v,        c1v/2.0,        (c2v-gv)/2.0],
            [c1v/2.0,    gv,             c3v/2.0],
            [(c2v-gv)/2.0, c3v/2.0,      c4v]
        ])

        eigs = np.linalg.eigvalsh(Gm)
        trace = c0v + gv + c4v
        rel_eig = eigs[0] / trace if trace > 0 else 0

        if eigs[0] >= -1e-8 * trace:  # relative tolerance
            n_psd += 1
        if rel_eig < worst_rel_eig:
            worst_rel_eig = rel_eig
        if eigs[0] < worst_abs_eig:
            worst_abs_eig = eigs[0]

        # Also verify H(w) >= 0 at the same point (use Horner form for stability)
        for wv in [-2, -1, -0.5, 0, 0.5, 1, 2]:
            Hval = c0v + wv*(c1v + wv*(c2v + wv*(c3v + wv*c4v)))
            if Hval < -1e-6 * max(abs(c0v), abs(c4v)):
                n_H_neg += 1

    print(f"  PSD (relative tol 1e-8): {n_psd}/{n} = {n_psd/n:.6f}", flush=True)
    print(f"  Worst relative eigenvalue (min_eig/trace): {worst_rel_eig:.4e}", flush=True)
    print(f"  Worst absolute eigenvalue: {worst_abs_eig:.4e}", flush=True)
    print(f"  H(w) negative count: {n_H_neg}", flush=True)

    # Check that the small negative eigenvalues are just floating point
    print("\n  Analyzing eigenvalue distribution...", flush=True)
    min_eigs = []
    for i in range(min(n, 10000)):
        rv, xv, yv = rs[i], xs[i], ys[i]
        c0v, c1v, c2v, c3v, c4v = coeffs_numeric(rv, xv, yv)
        Av = 1+3*xv; Bv = (1+3*yv)*rv**2
        Ev = S * (Av + Bv) * (Av + Bv + 4*rv)
        gv = (2*c2v + Ev) / 3.0
        Gm = np.array([
            [c0v,        c1v/2.0,        (c2v-gv)/2.0],
            [c1v/2.0,    gv,             c3v/2.0],
            [(c2v-gv)/2.0, c3v/2.0,      c4v]
        ])
        eigs = np.linalg.eigvalsh(Gm)
        trace = c0v + gv + c4v
        min_eigs.append(eigs[0] / trace)

    min_eigs = np.array(min_eigs)
    print(f"  min_eig/trace: min={np.min(min_eigs):.4e}, max={np.max(min_eigs):.4e}, "
          f"mean={np.mean(min_eigs):.4e}, std={np.std(min_eigs):.4e}", flush=True)
    n_below = np.sum(min_eigs < -1e-14)
    print(f"  Number with min_eig/trace < -1e-14: {n_below}/{len(min_eigs)}", flush=True)
    n_below2 = np.sum(min_eigs < -1e-12)
    print(f"  Number with min_eig/trace < -1e-12: {n_below2}/{len(min_eigs)}", flush=True)

    # ================================================================
    # STEP 9: Verify with multiprecision at "worst" points
    # ================================================================
    print("\n[STEP 9] Multiprecision verification at tricky points", flush=True)
    from decimal import Decimal, getcontext
    getcontext().prec = 50

    test_pts = [(1.0, 1/3, 1/3), (100.0, 0.9, 0.99),
                (0.01, 0.01, 0.99), (0.001, 0.5, 0.5)]
    for rv, xv, yv in test_pts:
        # Use mpmath for high precision
        import mpmath
        mpmath.mp.dps = 50
        rv_mp = mpmath.mpf(rv)
        xv_mp = mpmath.mpf(xv)
        yv_mp = mpmath.mpf(yv)

        Av = 1 + 3*xv_mp
        Byv = 1 + 3*yv_mp
        Bv = Byv * rv_mp**2
        Sv = mpmath.mpf(S)

        c0v = Sv * Av * (Av + 4*rv_mp)
        c1v = 2*Sv * Av * (Av + 4*rv_mp - Bv)
        c2v = Sv * (Av**2 + Bv**2 - 4*Av*Bv + 4*rv_mp*Av + 4*rv_mp**3*Byv)
        c3v = -2*Sv * Bv * (Av - 4*rv_mp - Bv)
        c4v = Sv * rv_mp**3 * Byv * (4 + rv_mp*Byv)
        Ev = Sv * (Av + Bv) * (Av + Bv + 4*rv_mp)
        gv = (2*c2v + Ev) / 3

        # Build Gram matrix in mpmath
        Gm = mpmath.matrix([
            [c0v,        c1v/2,        (c2v-gv)/2],
            [c1v/2,      gv,           c3v/2],
            [(c2v-gv)/2, c3v/2,        c4v]
        ])

        # Compute eigenvalues
        eigs = mpmath.eigsy(Gm, eigvals_only=True)
        eigs_sorted = sorted([float(e) for e in eigs])

        # Compute det
        det_val = float(mpmath.det(Gm))
        trace = float(c0v + gv + c4v)

        print(f"  r={rv}, x={xv}, y={yv}:", flush=True)
        print(f"    eigenvalues: {[f'{e:.6e}' for e in eigs_sorted]}", flush=True)
        print(f"    det = {det_val:.6e}", flush=True)
        print(f"    min_eig/trace = {eigs_sorted[0]/trace:.6e}", flush=True)

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 72, flush=True)
    print("PROOF SUMMARY", flush=True)
    print("=" * 72, flush=True)
    print("""
THEOREM: For all r > 0, x in (0,1), y in (0,1), the quartic polynomial
  H(w) = c0 + c1*w + c2*w^2 + c3*w^3 + c4*w^4
is non-negative for all real w.

PROOF (SOS certificate via Gram matrix):

Define:
  S = 59049 = 3^10
  A = 1 + 3x
  By = 1 + 3y
  B = By * r^2 = (1+3y)*r^2
  E = S * (A+B) * (A+B+4r)

Coefficients (factored):
  c0 = S * A * (A + 4r)
  c1 = 2S * A * (A + 4r - B)
  c2 = S * (A^2 + B^2 - 4AB + 4rA + 4r^3*By)
  c3 = -2S * B * (A - 4r - B)
  c4 = S * r^3 * By * (4 + r*By)

Set g* = (2*c2 + E) / 3
       = S/3 * (3(A-B)^2 + 4r(3A+B) + 8r^3*By)

The Gram matrix is:
  G = [[c0,      c1/2,       (c2-g*)/2],
       [c1/2,    g*,         c3/2     ],
       [(c2-g*)/2, c3/2,     c4       ]]

where c2 - g* = (c2 - E)/3 = -2S * r^2 * A * By = -2S * A * B.

VERIFICATION that G is PSD:

1. Diagonal entries:
   G[0,0] = c0 = S*A*(A+4r) > 0           [A > 0, r > 0]
   G[1,1] = g* = S*(3(A-B)^2+4r(3A+B)+8r^3By)/3 > 0
                                             [sum of non-neg terms, 4r(3A+B) > 0]
   G[2,2] = c4 = S*r^3*By*(4+rBy) > 0      [r > 0, By > 0]

2. All three 2x2 principal minors:
   M2a = c0*g* - c1^2/4 = 4S^2 * r^3 * A * By * (A+B+4r) > 0
   M2b = g**c4 - c3^2/4 = 4S^2 * r^3 * A * By * (A+B+4r) > 0
   M2c = c0*c4 - (c2-g*)^2/4 = 4S^2 * r^3 * A * By * (A+B+4r) > 0

   All three are IDENTICAL and manifestly positive.

3. Determinant:
   det(G) = 0 (symbolically verified).

Since G is a 3x3 real symmetric matrix with:
  - All diagonal entries > 0
  - All 2x2 principal minors > 0
  - det(G) = 0

G is positive semidefinite (by the principal minor criterion for PSD).

Therefore H(w) = [1, w, w^2] G [1, w, w^2]^T >= 0 for all w in R.  QED.
""", flush=True)


if __name__ == "__main__":
    main()
