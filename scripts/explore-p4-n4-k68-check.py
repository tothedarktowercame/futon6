#!/usr/bin/env python3
"""Check: is K6+K8 >= 0 on the feasible domain?

If so, then K = (K0+K2+K4) + (K6+K8) >= 0 since both groups are non-negative.
This splits the problem into two independent proofs:
  1. K0+K2+K4 >= 0 (503 terms, already 0/50k negative)
  2. K6+K8 >= 0 (156 terms, to be checked)

Also: check K8 >= 0 globally (it might be SOS via quartic structure).
And: extract K8 as p²q² * quartic and check quartic non-negativity.
"""

import numpy as np
import sympy as sp
from sympy import Rational


def disc_poly(e2, e3, e4):
    return (256*e4**3 - 128*e2**2*e4**2 + 144*e2*e3**2*e4
            + 16*e2**4*e4 - 27*e3**4 - 4*e2**3*e3**2)


def phi4_disc(e2, e3, e4):
    return (-8*e2**5 - 64*e2**3*e4 - 36*e2**2*e3**2
            + 384*e2*e4**2 - 432*e3**2*e4)


def build_blocks():
    print("Building K and decomposing...", flush=True)
    s, t, u, v, a, b = sp.symbols('s t u v a b', positive=True)
    r, x, y, p, q = sp.symbols('r x y p q', real=True)

    inv_phi = lambda e2, e3, e4: sp.cancel(disc_poly(e2, e3, e4) / phi4_disc(e2, e3, e4))
    inv_p = inv_phi(-s, u, a)
    inv_q = inv_phi(-t, v, b)
    inv_c = inv_phi(-(s+t), u+v, a+b+s*t/6)

    surplus = sp.together(inv_c - inv_p - inv_q)
    N, D = sp.fraction(surplus)
    N = sp.expand(N)

    subs = {t: r*s, a: x*s**2/4, b: y*r**2*s**2/4,
            u: p*s**Rational(3,2), v: q*s**Rational(3,2)}
    K = sp.expand(sp.expand(N.subs(subs)) / s**16)

    poly_pq = sp.Poly(K, p, q)
    blocks = {}
    for monom, coeff in poly_pq.as_dict().items():
        i, j = monom
        d = i + j
        blocks[d] = sp.expand(blocks.get(d, sp.Integer(0)) + coeff * p**i * q**j)

    return (r, x, y, p, q), blocks


def sample_feasible(rng, n_samples):
    out = []
    tries = 0
    while len(out) < n_samples and tries < 30 * n_samples:
        tries += 1
        rv = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
        xv = float(rng.uniform(1e-4, 1 - 1e-4))
        yv = float(rng.uniform(1e-4, 1 - 1e-4))
        pmax2 = 2*(1-xv)/9
        qmax2 = 2*(rv**3)*(1-yv)/9
        if pmax2 <= 0 or qmax2 <= 0:
            continue
        pv = float(rng.uniform(-0.95*np.sqrt(pmax2), 0.95*np.sqrt(pmax2)))
        qv = float(rng.uniform(-0.95*np.sqrt(qmax2), 0.95*np.sqrt(qmax2)))
        g3 = (6 + 14*rv + 14*rv**2 + 6*rv**3
              - 6*xv - 6*xv*rv - 6*yv*rv**2 - 6*yv*rv**3
              - 27*(pv+qv)**2)
        if g3 <= 1e-6:
            continue
        out.append((rv, xv, yv, pv, qv))
    return np.array(out)


def main():
    syms, blocks = build_blocks()
    r, x, y, p, q = syms

    block_fns = {d: sp.lambdify((r, x, y, p, q), blocks[d], 'numpy')
                 for d in sorted(blocks.keys())}

    print("\n" + "="*72, flush=True)
    print("CHECKING K6+K8 NON-NEGATIVITY", flush=True)
    print("="*72, flush=True)

    rng = np.random.default_rng(20260213)
    samples = sample_feasible(rng, 100000)
    print(f"Feasible samples: {len(samples)}", flush=True)

    rv = samples[:, 0]
    xv = samples[:, 1]
    yv = samples[:, 2]
    pv = samples[:, 3]
    qv = samples[:, 4]

    K6_vals = block_fns[6](rv, xv, yv, pv, qv)
    K8_vals = block_fns[8](rv, xv, yv, pv, qv)
    K68_vals = K6_vals + K8_vals
    K024_vals = block_fns[0](rv, xv, yv, pv, qv) + block_fns[2](rv, xv, yv, pv, qv) + block_fns[4](rv, xv, yv, pv, qv)

    n = len(samples)
    print(f"\n  K6:     min={np.min(K6_vals):.4e}, neg={np.sum(K6_vals < -1e-12)}/{n}", flush=True)
    print(f"  K8:     min={np.min(K8_vals):.4e}, neg={np.sum(K8_vals < -1e-12)}/{n}", flush=True)
    print(f"  K6+K8:  min={np.min(K68_vals):.4e}, neg={np.sum(K68_vals < -1e-12)}/{n}", flush=True)
    print(f"  K0+K2+K4: min={np.min(K024_vals):.4e}, neg={np.sum(K024_vals < -1e-12)}/{n}", flush=True)

    if np.sum(K68_vals < -1e-12) == 0:
        print(f"\n*** K6+K8 >= 0 on all {n} feasible samples! ***", flush=True)
        print(f"*** Proof reduces to: (1) K0+K2+K4 >= 0 and (2) K6+K8 >= 0 ***", flush=True)
    else:
        print(f"\n  K6+K8 can be negative. Need different grouping.", flush=True)
        # Check where K6+K8 is negative
        neg_mask = K68_vals < -1e-12
        print(f"  At K6+K8 < 0 points:", flush=True)
        print(f"    K0+K2+K4 min: {np.min(K024_vals[neg_mask]):.4e}", flush=True)
        print(f"    K0+K2+K4 > |K6+K8|? {np.all(K024_vals[neg_mask] > -K68_vals[neg_mask])}", flush=True)

    # PART 2: K8 quartic structure
    print(f"\n{'='*72}", flush=True)
    print("K8 QUARTIC ANALYSIS", flush=True)
    print("="*72, flush=True)

    # K8 = p²q² * H(r,x,y,p/q) where H is a quartic in w=p/q
    # K8 = c26*p²q⁶ + c35*p³q⁵ + c44*p⁴q⁴ + c53*p⁵q³ + c62*p⁶q²
    # = p²q² * (c26*q⁴ + c35*pq³ + c44*p²q² + c53*p³q + c62*p⁴)
    # = p²q² * q⁴ * (c26 + c35*w + c44*w² + c53*w³ + c62*w⁴)

    poly_K8 = sp.Poly(blocks[8], p, q)
    K8_coeffs = {}
    for monom, coeff in poly_K8.as_dict().items():
        K8_coeffs[monom] = sp.expand(coeff)

    # Extract coefficients of the quartic in w=p/q
    # p^a * q^b with a+b=8, minimum a=2, b=6
    c = {}
    for (a, b), coeff in K8_coeffs.items():
        # w-power is a-2 (since K8 = p²q²q⁴ * quartic(w))
        w_pow = a - 2
        c[w_pow] = coeff

    print("  Quartic H(w) = c0 + c1*w + c2*w² + c3*w³ + c4*w⁴", flush=True)
    for k in sorted(c.keys()):
        ck = sp.expand(c[k])
        n_terms = len(sp.Poly(ck, r, x, y).as_dict())
        print(f"  c{k}: {n_terms} terms = {ck}", flush=True)

    # Lambdify quartic coefficients
    c_fns = {k: sp.lambdify((r, x, y), c[k], 'numpy') for k in c}

    # Check if H(w) >= 0 for all w at sampled (r,x,y) points
    print(f"\n  Testing H(w) >= 0 for all real w at 10000 (r,x,y) points:", flush=True)
    rxy_samples = np.column_stack([
        np.exp(rng.uniform(np.log(0.1), np.log(10.0), 10000)),
        rng.uniform(0.01, 0.99, 10000),
        rng.uniform(0.01, 0.99, 10000)
    ])

    psd_count = 0
    not_psd_count = 0
    w_test = np.linspace(-100, 100, 5000)

    for idx in range(len(rxy_samples)):
        rv_i, xv_i, yv_i = rxy_samples[idx]
        c_vals = [float(c_fns[k](rv_i, xv_i, yv_i)) for k in range(5)]
        # Evaluate quartic at many w
        H_vals = c_vals[0] + c_vals[1]*w_test + c_vals[2]*w_test**2 + c_vals[3]*w_test**3 + c_vals[4]*w_test**4
        if np.min(H_vals) >= -1e-8:
            psd_count += 1
        else:
            not_psd_count += 1
            if not_psd_count <= 3:
                min_w = w_test[np.argmin(H_vals)]
                print(f"    NOT PSD: r={rv_i:.3f} x={xv_i:.3f} y={yv_i:.3f} "
                      f"min_H={np.min(H_vals):.4e} at w={min_w:.2f}", flush=True)
                print(f"      c = {c_vals}", flush=True)

    print(f"  H(w)>=0: {psd_count}/{len(rxy_samples)}", flush=True)
    print(f"  H(w)<0: {not_psd_count}/{len(rxy_samples)}", flush=True)

    if psd_count == len(rxy_samples):
        print(f"\n*** K8 = p²q²q⁴ H(p/q) with H >= 0 for all p/q ***", flush=True)
        print(f"*** K8 >= 0 GLOBALLY (not just feasible) ***", flush=True)

        # Can we prove H >= 0 via SOS?
        # H = c4*w⁴ + c3*w³ + c2*w² + c1*w + c0
        # For H SOS: H = (a*w² + b*w + c)² + (d*w + e)²
        # This gives: c4 = a² + d², c3 = 2ab + 2de, c2 = 2ac + b² + e²,
        #             c1 = 2bc + 2de... wait that doesn't work easily.
        # Gram matrix for degree-4 univariate:
        # v = [1, w, w²], G 3×3 PSD
        # H = v^T G v = G00 + 2G01*w + (2G02+G11)*w² + 2G12*w³ + G22*w⁴
        # So: c0 = G00, c1 = 2G01, c2 = 2G02+G11, c3 = 2G12, c4 = G22
        # One free param: G11 (with G02 = (c2-G11)/2)
        # Check if PSD: G00 ≥ 0, det(G[0:2,0:2]) ≥ 0, det(G) ≥ 0

        print(f"\n  3×3 Gram matrix for H (1 free param g11):", flush=True)
        g11 = sp.Symbol('g11')
        G = sp.Matrix([
            [c[0], c[1]/2, (c[2] - g11)/2],
            [c[1]/2, g11, c[3]/2],
            [(c[2] - g11)/2, c[3]/2, c[4]]
        ])

        # Test at sampled points: find optimal g11
        psd_gram = 0
        for idx in range(min(500, len(rxy_samples))):
            rv_i, xv_i, yv_i = rxy_samples[idx]
            c_vals = {k: float(c_fns[k](rv_i, xv_i, yv_i)) for k in range(5)}

            def min_eig_of_G(g11_val):
                G_num = np.array([
                    [c_vals[0], c_vals[1]/2, (c_vals[2]-g11_val)/2],
                    [c_vals[1]/2, g11_val, c_vals[3]/2],
                    [(c_vals[2]-g11_val)/2, c_vals[3]/2, c_vals[4]]
                ])
                return np.linalg.eigvalsh(G_num)[0]

            # Search g11
            best_g11 = 0
            best_eig = -np.inf
            for g11_try in np.linspace(0, 2*max(c_vals[0], c_vals[4]), 200):
                eig = min_eig_of_G(g11_try)
                if eig > best_eig:
                    best_eig = eig
                    best_g11 = g11_try
            # Refine
            lo, hi = best_g11 - c_vals[4]/50, best_g11 + c_vals[4]/50
            for _ in range(50):
                m1, m2 = lo + (hi-lo)*0.382, lo + (hi-lo)*0.618
                if min_eig_of_G(m1) < min_eig_of_G(m2):
                    lo = m1
                else:
                    hi = m2
            best_eig = min_eig_of_G((lo+hi)/2)

            if best_eig >= -1e-8:
                psd_gram += 1

        print(f"  3×3 Gram PSD: {psd_gram}/{min(500, len(rxy_samples))}", flush=True)
        if psd_gram == min(500, len(rxy_samples)):
            print(f"  *** K8 is SOS in w=p/q! ***", flush=True)


if __name__ == '__main__':
    main()
