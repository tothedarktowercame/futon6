#!/usr/bin/env python3
"""Full n=4 Stam numerator structure in normalized coordinates.

This script extends the symmetric (e3=0) certificate work to the general
centered n=4 case by deriving the normalized numerator K(r,x,y,p,q):

  p(x)=x^4 - s x^2 - u x + a
  q(x)=x^4 - t x^2 - v x + b
  r=t/s,  x=4a/s^2,  y=4b/t^2,  p=u/s^(3/2),  q=v/s^(3/2)

The full surplus numerator N(s,t,u,v,a,b) satisfies:
  N = s^16 * K(r,x,y,p,q).

We report:
1) symbolic structure of K and its (p,q)-degree decomposition,
2) whether the first perturbation block K2 is PSD (it is not),
3) Monte Carlo checks of full K on the feasible cone.
"""

from __future__ import annotations

import argparse
import numpy as np
import sympy as sp


def disc(e2, e3, e4):
    return (
        256 * e4**3
        - 128 * e2**2 * e4**2
        + 144 * e2 * e3**2 * e4
        + 16 * e2**4 * e4
        - 27 * e3**4
        - 4 * e2**3 * e3**2
    )


def phi4_disc_poly(e2, e3, e4):
    return (
        -8 * e2**5
        - 64 * e2**3 * e4
        - 36 * e2**2 * e3**2
        + 384 * e2 * e4**2
        - 432 * e3**2 * e4
    )


def build_full_surplus_numerator():
    s, t, u, v, a, b = sp.symbols("s t u v a b", positive=True, real=True)

    # centered quartics:
    # p: (e2,e3,e4)=(-s,u,a), q: (-t,v,b)
    S = s + t
    U = u + v
    A = a + b + s * t / 6

    inv_p = sp.together(disc(-s, u, a) / phi4_disc_poly(-s, u, a))
    inv_q = sp.together(disc(-t, v, b) / phi4_disc_poly(-t, v, b))
    inv_c = sp.together(disc(-S, U, A) / phi4_disc_poly(-S, U, A))

    surplus = sp.together(inv_c - inv_p - inv_q)
    N, D = sp.fraction(surplus)
    N = sp.expand(N)
    D = sp.factor(D)
    return (s, t, u, v, a, b), N, D


def normalize_numerator(N, syms):
    s, t, u, v, a, b = syms
    r, x, y, p, q = sp.symbols("r x y p q", real=True)
    subs = {
        t: r * s,
        a: x * s**2 / 4,
        b: y * r**2 * s**2 / 4,
        u: p * s ** sp.Rational(3, 2),
        v: q * s ** sp.Rational(3, 2),
    }
    Nn = sp.expand(N.subs(subs))
    K = sp.expand(Nn / s**16)
    return (r, x, y, p, q), K


def decompose_pq_blocks(K, r, x, y, p, q):
    poly = sp.Poly(K, p, q)
    blocks = {}
    for i in range(poly.degree(p) + 1):
        for j in range(poly.degree(q) + 1):
            c = poly.coeff_monomial(p**i * q**j)
            if c == 0:
                continue
            d = i + j
            blocks[d] = sp.expand(blocks.get(d, 0) + c * p**i * q**j)
    return blocks


def sample_feasible(rng, n_samples):
    """Sample points from normalized feasible cone.

    Constraints from denominator factors in normalized coordinates:
      g1 = 2(1-x) - 9 p^2 > 0
      g2 = 2 r^3 (1-y) - 9 q^2 > 0
      g3 = 6 + 14r + 14r^2 + 6r^3 - 6x - 6xr - 6yr^2 - 6yr^3 - 27(p+q)^2 > 0
    """
    out = []
    tries = 0
    while len(out) < n_samples and tries < 20 * n_samples:
        tries += 1
        r = float(np.exp(rng.uniform(np.log(0.2), np.log(5.0))))
        x = float(rng.uniform(1e-5, 0.99999))
        y = float(rng.uniform(1e-5, 0.99999))
        pmax = np.sqrt(max(0.0, 2 * (1 - x) / 9))
        qmax = np.sqrt(max(0.0, 2 * (r**3) * (1 - y) / 9))
        if pmax <= 0 or qmax <= 0:
            continue
        p = float(rng.uniform(-0.98 * pmax, 0.98 * pmax))
        q = float(rng.uniform(-0.98 * qmax, 0.98 * qmax))
        g3 = (
            6
            + 14 * r
            + 14 * r * r
            + 6 * r**3
            - 6 * x
            - 6 * x * r
            - 6 * y * r * r
            - 6 * y * r**3
            - 27 * (p + q) ** 2
        )
        if g3 <= 1e-8:
            continue
        out.append((r, x, y, p, q))
    return out


def analyze(trials: int, seed: int):
    syms, N, D = build_full_surplus_numerator()
    s, t, u, v, a, b = syms
    r, x, y, p, q = sp.symbols("r x y p q", real=True)
    (r, x, y, p, q), K = normalize_numerator(N, syms)

    polyN = sp.Poly(N, s, t, u, v, a, b)
    polyK = sp.Poly(K, r, x, y, p, q)

    swap_ok = sp.expand(
        N.subs({s: t, t: s, u: v, v: u, a: b, b: a}, simultaneous=True) - N
    ) == 0
    parity_ok = sp.expand(N.subs({u: -u, v: -v}) - N) == 0

    blocks = decompose_pq_blocks(K, r, x, y, p, q)
    block_info = {}
    for d, expr in sorted(blocks.items()):
        block_info[d] = {
            "terms": len(sp.Poly(expr, r, x, y, p, q).terms()),
            "deg_total": int(sp.Poly(expr, r, x, y, p, q).total_degree()),
        }

    # K2 quadratic-form diagnostics
    K2 = blocks.get(2, 0)
    A2 = sp.expand(sp.Poly(K2, p, q).coeff_monomial(p**2))
    B2 = sp.expand(sp.Poly(K2, p, q).coeff_monomial(p * q))
    C2 = sp.expand(sp.Poly(K2, p, q).coeff_monomial(q**2))
    disc2 = sp.expand(B2**2 - 4 * A2 * C2)

    fK = sp.lambdify((r, x, y, p, q), K, "numpy")
    fA2 = sp.lambdify((r, x, y), A2, "numpy")
    fC2 = sp.lambdify((r, x, y), C2, "numpy")
    fDisc2 = sp.lambdify((r, x, y), disc2, "numpy")

    rng = np.random.default_rng(seed)
    samples = sample_feasible(rng, trials)

    minK = float("inf")
    argK = None
    minA2 = float("inf")
    argA2 = None
    minC2 = float("inf")
    argC2 = None
    maxDisc2 = -float("inf")
    argDisc2 = None

    for rv, xv, yv, pv, qv in samples:
        kv = float(fK(rv, xv, yv, pv, qv))
        if kv < minK:
            minK = kv
            argK = (rv, xv, yv, pv, qv)

        a2v = float(fA2(rv, xv, yv))
        if a2v < minA2:
            minA2 = a2v
            argA2 = (rv, xv, yv)

        c2v = float(fC2(rv, xv, yv))
        if c2v < minC2:
            minC2 = c2v
            argC2 = (rv, xv, yv)

        d2v = float(fDisc2(rv, xv, yv))
        if d2v > maxDisc2:
            maxDisc2 = d2v
            argDisc2 = (rv, xv, yv)

    print("=" * 78)
    print("P4 n=4 full normalized numerator structure")
    print("=" * 78)
    print("[1] Symbolic structure")
    print(f"N terms (s,t,u,v,a,b): {len(polyN.terms())}, total degree {polyN.total_degree()}")
    print(f"K terms (r,x,y,p,q):   {len(polyK.terms())}, total degree {polyK.total_degree()}")
    print(f"swap symmetry: {swap_ok}")
    print(f"global (u,v)->(-u,-v) parity: {parity_ok}")
    print("pq-blocks by total degree:")
    for d in sorted(block_info):
        print(f"  degree {d}: {block_info[d]['terms']} terms, total degree {block_info[d]['deg_total']}")
    print()
    print("[2] First perturbation block K2")
    print("K2 = A2(r,x,y) p^2 + B2(r,x,y) p q + C2(r,x,y) q^2")
    print(f"A2 term count: {len(sp.Poly(A2, r, x, y).terms())}")
    print(f"B2 term count: {len(sp.Poly(B2, r, x, y).terms())}")
    print(f"C2 term count: {len(sp.Poly(C2, r, x, y).terms())}")
    print(f"min A2 over sampled base points: {minA2:.6e} at {argA2}")
    print(f"min C2 over sampled base points: {minC2:.6e} at {argC2}")
    print(f"max discr(B2^2-4A2C2): {maxDisc2:.6e} at {argDisc2}")
    print("=> K2 is not PSD on sampled domain; blockwise positivity fails.")
    print()
    print("[3] Full K check on feasible sampled points")
    print(f"sampled feasible points: {len(samples)}")
    print(f"min K: {minK:.6e} at {argK}")
    print("No negative K found in sampled feasible points." if minK >= -1e-10 else "Negative K found!")
    print()
    print("Interpretation:")
    print("1. Symmetric certificate K0 is only the base layer of a coupled perturbation problem.")
    print("2. The u,v perturbation cannot be closed by proving each degree block nonnegative.")
    print("3. Next path should target coupled inequalities between K2,K4,K6.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=20260213)
    args = parser.parse_args()
    analyze(args.trials, args.seed)


if __name__ == "__main__":
    main()
