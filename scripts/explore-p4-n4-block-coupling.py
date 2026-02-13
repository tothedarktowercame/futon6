#!/usr/bin/env python3
"""Explore coupling structure between (p,q)-degree blocks of the full n=4 surplus.

Building on analyze-p4-n4-full-normalized.py, which showed:
- K0 (102 terms): the proved symmetric base layer
- K2 (200 terms): INDEFINITE — blockwise proof fails here
- K4 (201 terms), K6 (114 terms), K8 (42 terms): unknown signs

Key questions:
1. Are K4, K6, K8 individually non-negative?
2. Is K0 + K2 >= 0 (does the base layer absorb the negative K2)?
3. What is the minimum partial sum K0 + K2 + ... + K_{2j} that stays non-negative?
4. Can we write K as (K0 + K2) + (K4 + K6 + K8) where each group is non-negative?
"""

import numpy as np
import sympy as sp


def disc_poly(e2, e3, e4):
    return (256*e4**3 - 128*e2**2*e4**2 + 144*e2*e3**2*e4
            + 16*e2**4*e4 - 27*e3**4 - 4*e2**3*e3**2)


def phi4_disc(e2, e3, e4):
    return (-8*e2**5 - 64*e2**3*e4 - 36*e2**2*e3**2
            + 384*e2*e4**2 - 432*e3**2*e4)


def build_K():
    """Build the normalized surplus K(r,x,y,p,q) and decompose by (p,q)-degree."""
    print("Building full surplus (this takes ~2 min)...")
    s, t, u, v, a, b = sp.symbols('s t u v a b', positive=True)

    inv_phi = lambda e2, e3, e4: sp.cancel(disc_poly(e2, e3, e4) / phi4_disc(e2, e3, e4))

    inv_p = inv_phi(-s, u, a)
    inv_q = inv_phi(-t, v, b)
    inv_c = inv_phi(-(s+t), u+v, a+b+s*t/6)

    surplus = sp.together(inv_c - inv_p - inv_q)
    N, D = sp.fraction(surplus)
    N = sp.expand(N)
    print(f"  N: {len(sp.Poly(N, s,t,u,v,a,b).as_dict())} terms")

    # Normalize: r=t/s, x=4a/s^2, y=4b/t^2, p=u/s^(3/2), q=v/s^(3/2)
    r, x, y, p, q = sp.symbols('r x y p q', real=True)
    subs = {t: r*s, a: x*s**2/4, b: y*r**2*s**2/4,
            u: p*s**sp.Rational(3,2), v: q*s**sp.Rational(3,2)}
    Nn = sp.expand(N.subs(subs))
    K = sp.expand(Nn / s**16)
    print(f"  K: {len(sp.Poly(K, r,x,y,p,q).as_dict())} terms")

    # Decompose by total (p,q)-degree
    poly = sp.Poly(K, p, q)
    blocks = {}
    for monom, coeff in poly.as_dict().items():
        i, j = monom  # p^i * q^j
        d = i + j
        blocks[d] = sp.expand(blocks.get(d, sp.Integer(0)) + coeff * p**i * q**j)

    print(f"  Blocks: {sorted(blocks.keys())}")
    for d in sorted(blocks.keys()):
        terms = len(sp.Poly(blocks[d], r, x, y, p, q).as_dict())
        print(f"    K{d}: {terms} terms")

    return (r, x, y, p, q), blocks


def sample_feasible(rng, n_samples):
    """Sample from the normalized feasible cone."""
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
        # Check convolution constraint
        g3 = (6 + 14*rv + 14*rv**2 + 6*rv**3
              - 6*xv - 6*xv*rv - 6*yv*rv**2 - 6*yv*rv**3
              - 27*(pv+qv)**2)
        if g3 <= 1e-6:
            continue
        out.append((rv, xv, yv, pv, qv))
    return np.array(out)


def main():
    (r, x, y, p, q), blocks = build_K()

    # Lambdify each block
    block_fns = {}
    for d in sorted(blocks.keys()):
        block_fns[d] = sp.lambdify((r, x, y, p, q), blocks[d], 'numpy')

    # Also make partial sum functions
    partial_sums = {}
    cumulative = sp.Integer(0)
    for d in sorted(blocks.keys()):
        cumulative = sp.expand(cumulative + blocks[d])
        partial_sums[d] = sp.lambdify((r, x, y, p, q), cumulative, 'numpy')

    print(f"\n{'='*72}")
    print("Block Coupling Analysis")
    print('='*72)

    rng = np.random.default_rng(20260213)
    samples = sample_feasible(rng, 50000)
    print(f"Feasible samples: {len(samples)}")

    rv = samples[:, 0]
    xv = samples[:, 1]
    yv = samples[:, 2]
    pv = samples[:, 3]
    qv = samples[:, 4]

    # 1. Individual block signs
    print(f"\n[1] Individual block signs:")
    for d in sorted(blocks.keys()):
        vals = block_fns[d](rv, xv, yv, pv, qv)
        neg_count = np.sum(vals < -1e-12)
        neg_frac = neg_count / len(vals)
        print(f"  K{d}: min={np.min(vals):.4e}, max={np.max(vals):.4e}, "
              f"neg={neg_count}/{len(vals)} ({neg_frac:.1%})")

    # 2. Partial sums
    print(f"\n[2] Partial sums (cumulative from K0):")
    for d in sorted(blocks.keys()):
        vals = partial_sums[d](rv, xv, yv, pv, qv)
        neg_count = np.sum(vals < -1e-12)
        min_val = np.min(vals)
        print(f"  K0+...+K{d}: min={min_val:.4e}, neg={neg_count}/{len(vals)}")

    # 3. Pairwise combinations
    print(f"\n[3] Key combinations:")
    ds = sorted(blocks.keys())
    combos = [
        ("K0+K2", [0, 2]),
        ("K0+K4", [0, 4]),
        ("K2+K4", [2, 4]),
        ("K0+K2+K4", [0, 2, 4]),
        ("K4+K6+K8", [4, 6, 8]),
        ("K2+K4+K6", [2, 4, 6]),
    ]
    for name, ds_list in combos:
        vals = sum(block_fns[d](rv, xv, yv, pv, qv) for d in ds_list)
        neg_count = np.sum(vals < -1e-12)
        min_val = np.min(vals)
        print(f"  {name}: min={min_val:.4e}, neg={neg_count}/{len(vals)}")

    # 4. Ratio analysis: how much of K is each block?
    print(f"\n[4] Block contribution ratios at points where K2 is most negative:")
    k2_vals = block_fns[2](rv, xv, yv, pv, qv)
    worst_k2_idx = np.argsort(k2_vals)[:10]
    for idx in worst_k2_idx:
        pt = samples[idx]
        block_vals = {d: float(block_fns[d](*pt)) for d in sorted(blocks.keys())}
        total = sum(block_vals.values())
        print(f"  r={pt[0]:.3f} x={pt[1]:.3f} y={pt[2]:.3f} "
              f"p={pt[3]:.4f} q={pt[4]:.4f}")
        print(f"    " + "  ".join(f"K{d}={v:.3f}" for d, v in block_vals.items())
              + f"  total={total:.4f}")

    # 5. Check if K2 discriminant (as quadratic in p,q) has structure
    print(f"\n[5] K2 quadratic form analysis:")
    A2 = sp.expand(sp.Poly(blocks[2], p, q).coeff_monomial(p**2))
    B2 = sp.expand(sp.Poly(blocks[2], p, q).coeff_monomial(p*q))
    C2 = sp.expand(sp.Poly(blocks[2], p, q).coeff_monomial(q**2))
    print(f"  A2 terms: {len(sp.Poly(A2, r, x, y).as_dict())}")
    print(f"  B2 terms: {len(sp.Poly(B2, r, x, y).as_dict())}")
    print(f"  C2 terms: {len(sp.Poly(C2, r, x, y).as_dict())}")

    fA2 = sp.lambdify((r, x, y), A2, 'numpy')
    fB2 = sp.lambdify((r, x, y), B2, 'numpy')
    fC2 = sp.lambdify((r, x, y), C2, 'numpy')

    A2v = fA2(rv, xv, yv)
    B2v = fB2(rv, xv, yv)
    C2v = fC2(rv, xv, yv)
    disc2v = B2v**2 - 4*A2v*C2v

    # At points where A2 < 0, what are p,q doing?
    neg_A2_mask = A2v < -1e-10
    if np.any(neg_A2_mask):
        print(f"  A2 < 0 at {np.sum(neg_A2_mask)} points")
        # At these points, what's the maximum p^2 allowed?
        p2_bound = 2*(1 - xv[neg_A2_mask])/9
        actual_p2 = pv[neg_A2_mask]**2
        ratio = actual_p2 / p2_bound
        print(f"  At A2<0 points: mean(p^2/p^2_max) = {np.mean(ratio):.3f}")
        print(f"  At A2<0 points: mean(x) = {np.mean(xv[neg_A2_mask]):.3f}")
        print(f"  At A2<0 points: mean(r) = {np.mean(rv[neg_A2_mask]):.3f}")

    # 6. The key question: can we bound K2 by K0 on the feasible domain?
    print(f"\n[6] Ratio K2/K0 on feasible domain (where K0 > 0):")
    k0_vals = block_fns[0](rv, xv, yv, pv, qv)
    pos_k0_mask = k0_vals > 1e-12
    ratios = k2_vals[pos_k0_mask] / k0_vals[pos_k0_mask]
    print(f"  min(K2/K0) = {np.min(ratios):.4f}")
    print(f"  max(K2/K0) = {np.max(ratios):.4f}")
    print(f"  => K2/K0 > -1 everywhere?" +
          f" {'YES' if np.min(ratios) > -1 else 'NO'}")
    if np.min(ratios) > -1:
        print(f"  ==> K0 + K2 >= 0 on feasible domain!")
        print(f"  ==> The symmetric certificate K0 ABSORBS the negative K2.")


if __name__ == '__main__':
    main()
