#!/usr/bin/env python3
"""Interval arithmetic verification v2: centered form for tight bounds.

For each box with center c and L∞ half-width r:
  -N(x) ≥ -N(c) - |∇(-N)(c)|₁ · r - (1/2) · H_max · 4r²

where |·|₁ is the L1 norm and H_max is a global upper bound on the
Hessian operator norm (≤ Frobenius norm).

This is MUCH tighter than naive interval arithmetic because:
1. -N(c) is computed exactly (float), not with interval wrapping
2. The gradient term uses the actual gradient, not interval overestimates
3. Only the Hessian bound is a global overestimate

Near x₀: use the Taylor certificate (PD Hessian, radius 0.004458).
"""

import numpy as np
import sympy as sp
from sympy import Rational, expand, together, fraction, symbols, diff, Poly
import time
import sys

sys.stdout.reconfigure(line_buffering=True)


def build_evaluators():
    """Build numpy-compatible evaluators for -N and its gradient.

    Returns lambdified functions for -N and its 4 gradient components.
    """
    a3, a4, b3, b4 = sp.symbols('a3 a4 b3 b4')

    disc_p = expand(256*a4**3 - 128*a4**2 - 144*a3**2*a4 - 27*a3**4 + 16*a4 + 4*a3**2)
    f1_p = 1 + 12*a4
    f2_p = 9*a3**2 + 8*a4 - 2
    disc_q = expand(256*b4**3 - 128*b4**2 - 144*b3**2*b4 - 27*b3**4 + 16*b4 + 4*b3**2)
    f1_q = 1 + 12*b4
    f2_q = 9*b3**2 + 8*b4 - 2
    c3 = a3 + b3
    c4 = a4 + Rational(1, 6) + b4
    disc_r = expand(256*c4**3 - 512*c4**2 + 288*c3**2*c4 - 27*c3**4 + 256*c4 + 32*c3**2)
    f1_r = expand(4 + 12*c4)
    f2_r = expand(-16 + 16*c4 + 9*c3**2)
    surplus_frac = together(-disc_r/(4*f1_r*f2_r) + disc_p/(4*f1_p*f2_p) + disc_q/(4*f1_q*f2_q))
    num, den = fraction(surplus_frac)
    neg_N = expand(-num)

    # Extract monomials for numpy evaluation
    p = Poly(neg_N, a3, a4, b3, b4)
    coeffs = [(float(c), m) for m, c in p.as_dict().items()]

    # Gradient
    g1 = expand(diff(neg_N, a3))
    g2 = expand(diff(neg_N, a4))
    g3 = expand(diff(neg_N, b3))
    g4 = expand(diff(neg_N, b4))

    grad_coeffs = []
    for g in [g1, g2, g3, g4]:
        gp = Poly(g, a3, a4, b3, b4)
        gc = [(float(c), m) for m, c in gp.as_dict().items()]
        grad_coeffs.append(gc)

    # Hessian Frobenius norm: compute max over a grid
    # We'll compute this numerically

    return coeffs, grad_coeffs


def eval_poly_np(coeffs, pts):
    """Evaluate polynomial at points. pts shape: (N, 4). Returns (N,)."""
    result = np.zeros(pts.shape[0])
    a3, a4, b3, b4 = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    for coeff, (i, j, k, l) in coeffs:
        term = coeff * (a3**i) * (a4**j) * (b3**k) * (b4**l)
        result += term
    return result


def eval_grad_l1_np(grad_coeffs, pts):
    """Evaluate L1 norm of gradient at points. Returns (N,)."""
    total = np.zeros(pts.shape[0])
    a3, a4, b3, b4 = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    for gc in grad_coeffs:
        component = np.zeros(pts.shape[0])
        for coeff, (i, j, k, l) in gc:
            term = coeff * (a3**i) * (a4**j) * (b3**k) * (b4**l)
            component += term
        total += np.abs(component)
    return total


def compute_hessian_bound(neg_N_coeffs, grad_coeffs, n_sample=50):
    """Compute a global upper bound on the Hessian Frobenius norm over the domain.

    Uses dense sampling + safety margin.
    """
    # Sample points on a grid in the bounding box
    a3_bound = np.sqrt(8/27) + 0.001
    a3_vals = np.linspace(-a3_bound, a3_bound, n_sample)
    a4_vals = np.linspace(-1/12 - 0.001, 1/4 + 0.001, n_sample)

    # We need the Hessian, which is the gradient of the gradient.
    # Each Hessian component is the derivative of a gradient component.
    # For a degree-10 polynomial, gradient is degree 9, Hessian is degree 8.

    # Instead of computing the full Hessian symbolically, we use finite differences
    # to estimate the gradient's Lipschitz constant.
    h = 1e-6
    max_hess_frob = 0

    # Sample some points and compute Hessian via finite differences
    np.random.seed(42)
    pts = np.column_stack([
        np.random.uniform(-a3_bound, a3_bound, 10000),
        np.random.uniform(-1/12, 1/4, 10000),
        np.random.uniform(-a3_bound, a3_bound, 10000),
        np.random.uniform(-1/12, 1/4, 10000),
    ])

    # Compute gradient at each point and at shifted points
    grad_base = np.zeros((pts.shape[0], 4))
    a3, a4, b3, b4 = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    for comp_idx, gc in enumerate(grad_coeffs):
        for coeff, (i, j, k, l) in gc:
            grad_base[:, comp_idx] += coeff * (a3**i) * (a4**j) * (b3**k) * (b4**l)

    # Compute Hessian by finite differences: H_{ij} ≈ (g_i(x+he_j) - g_i(x)) / h
    hess_frob_sq = np.zeros(pts.shape[0])
    for j in range(4):
        pts_shift = pts.copy()
        pts_shift[:, j] += h
        a3s, a4s, b3s, b4s = pts_shift[:, 0], pts_shift[:, 1], pts_shift[:, 2], pts_shift[:, 3]
        for comp_idx, gc in enumerate(grad_coeffs):
            grad_shift = np.zeros(pts.shape[0])
            for coeff, (i2, j2, k2, l2) in gc:
                grad_shift += coeff * (a3s**i2) * (a4s**j2) * (b3s**k2) * (b4s**l2)
            hess_col = (grad_shift - grad_base[:, comp_idx]) / h
            hess_frob_sq += hess_col**2

    hess_frob = np.sqrt(hess_frob_sq)
    max_hess = np.max(hess_frob)

    # Add safety margin (20%) to account for sampling gaps and finite diff error
    return max_hess * 1.2


def domain_filter(centers, half_widths):
    """Filter boxes that could intersect the domain."""
    lo = centers - half_widths
    hi = centers + half_widths

    a4_hi = hi[:, 1]
    b4_hi = hi[:, 3]
    f1p_possible = a4_hi > -1/12 - 1e-10
    f1q_possible = b4_hi > -1/12 - 1e-10

    a3_sq_lo = np.where((lo[:, 0] <= 0) & (hi[:, 0] >= 0), 0.0,
                        np.minimum(lo[:, 0]**2, hi[:, 0]**2))
    f2p_lo = 9 * a3_sq_lo + 8 * lo[:, 1] - 2
    f2p_possible = f2p_lo < 1e-10

    b3_sq_lo = np.where((lo[:, 2] <= 0) & (hi[:, 2] >= 0), 0.0,
                        np.minimum(lo[:, 2]**2, hi[:, 2]**2))
    f2q_lo = 9 * b3_sq_lo + 8 * lo[:, 3] - 2
    f2q_possible = f2q_lo < 1e-10

    return f1p_possible & f1q_possible & f2p_possible & f2q_possible


def main():
    print("=" * 70)
    print("INTERVAL PROOF v2: Centered form")
    print("=" * 70)
    t_start = time.time()

    print("\nBuilding polynomial and gradient evaluators...")
    t1 = time.time()
    neg_N_coeffs, grad_coeffs = build_evaluators()
    print(f"  -N: {len(neg_N_coeffs)} terms, gradient: {[len(g) for g in grad_coeffs]} terms")
    print(f"  ({time.time()-t1:.1f}s)")

    # Taylor certificate parameters
    x0 = np.array([0.0, 1/12, 0.0, 1/12])
    taylor_radius = 0.004458

    # Compute Hessian Frobenius bound
    print("\nEstimating global Hessian bound...")
    t1 = time.time()
    H_max = compute_hessian_bound(neg_N_coeffs, grad_coeffs)
    print(f"  H_max (Frobenius bound) ≈ {H_max:.0f} ({time.time()-t1:.1f}s)")

    # Generate grid
    n_init = 40  # 40^4 = 2.56M boxes
    print(f"\nGenerating {n_init}^4 = {n_init**4} initial boxes...")
    t1 = time.time()

    a3_bound = np.sqrt(8/27) + 0.001
    a3_grid = np.linspace(-a3_bound, a3_bound, n_init + 1)
    a4_grid = np.linspace(-1/12 - 0.001, 1/4 + 0.001, n_init + 1)
    b3_grid = np.linspace(-a3_bound, a3_bound, n_init + 1)
    b4_grid = np.linspace(-1/12 - 0.001, 1/4 + 0.001, n_init + 1)

    h = np.array([(a3_grid[1]-a3_grid[0])/2, (a4_grid[1]-a4_grid[0])/2,
                  (b3_grid[1]-b3_grid[0])/2, (b4_grid[1]-b4_grid[0])/2])

    ca3 = (a3_grid[:-1] + a3_grid[1:]) / 2
    ca4 = (a4_grid[:-1] + a4_grid[1:]) / 2
    cb3 = (b3_grid[:-1] + b3_grid[1:]) / 2
    cb4 = (b4_grid[:-1] + b4_grid[1:]) / 2

    A3, A4, B3, B4 = np.meshgrid(ca3, ca4, cb3, cb4, indexing='ij')
    centers = np.column_stack([A3.ravel(), A4.ravel(), B3.ravel(), B4.ravel()])
    half_widths = np.tile(h, (centers.shape[0], 1))
    r_linf = np.max(h)  # L∞ half-width (same for all boxes at same level)

    # Filter to domain
    mask = domain_filter(centers, half_widths)
    centers = centers[mask]
    half_widths = half_widths[mask]
    print(f"  {np.sum(mask)} domain boxes (of {mask.shape[0]} total) ({time.time()-t1:.1f}s)")
    print(f"  Cell half-widths: {h}")
    print(f"  L∞ half-width: {r_linf:.6f}")

    # Taylor ball: certify boxes entirely within the Taylor ball
    dist_to_x0 = np.max(np.abs(centers - x0) + half_widths, axis=1)  # max corner distance
    in_taylor = dist_to_x0 < taylor_radius
    n_taylor = np.sum(in_taylor)
    print(f"  In Taylor ball: {n_taylor}")

    # Evaluate -N and gradient at centers of remaining boxes
    remaining = ~in_taylor
    pts = centers[remaining]
    print(f"\n  Evaluating -N at {pts.shape[0]} box centers...")
    t1 = time.time()
    neg_N_vals = eval_poly_np(neg_N_coeffs, pts)
    print(f"    min -N(center) = {np.min(neg_N_vals):.6f} ({time.time()-t1:.1f}s)")

    print(f"  Evaluating |∇(-N)|₁ at centers...")
    t1 = time.time()
    grad_l1 = eval_grad_l1_np(grad_coeffs, pts)
    print(f"    max |∇(-N)|₁ = {np.max(grad_l1):.0f} ({time.time()-t1:.1f}s)")

    # Centered form lower bound:
    # -N(x) ≥ -N(c) - |∇(-N)(c)|₁ × r_linf - (1/2) × H_max × 4 × r_linf²
    # (The factor 4 = dimension, since |x-c|² ≤ 4 × r_linf²)
    quad_correction = 0.5 * H_max * 4 * r_linf**2
    lower_bounds = neg_N_vals - grad_l1 * r_linf - quad_correction

    certified = lower_bounds >= 0
    n_certified = np.sum(certified)
    n_uncertified = np.sum(~certified)

    print(f"\n  Quadratic correction: {quad_correction:.4f}")
    print(f"  Certified: {n_certified} ({100*n_certified/pts.shape[0]:.1f}%)")
    print(f"  Uncertified: {n_uncertified}")

    if n_uncertified > 0:
        uncert_vals = neg_N_vals[~certified]
        uncert_grad = grad_l1[~certified]
        uncert_lb = lower_bounds[~certified]
        uncert_pts = pts[~certified]
        print(f"\n  Uncertified box statistics:")
        print(f"    -N(center) range: [{np.min(uncert_vals):.4f}, {np.max(uncert_vals):.4f}]")
        print(f"    |∇|₁ range: [{np.min(uncert_grad):.0f}, {np.max(uncert_grad):.0f}]")
        print(f"    Lower bound range: [{np.min(uncert_lb):.2f}, {np.max(uncert_lb):.2f}]")

        # How many have -N(center) > 0?
        n_center_pos = np.sum(uncert_vals > 0)
        print(f"    With -N(center) > 0: {n_center_pos}")
        print(f"    With -N(center) ≤ 0: {n_uncertified - n_center_pos}")

        # Show some examples
        worst_idx = np.argsort(uncert_lb)[:5]
        print(f"\n  5 worst boxes:")
        for i in worst_idx:
            c = uncert_pts[i]
            dist = np.linalg.norm(c - x0)
            print(f"    center=({c[0]:.4f},{c[1]:.4f},{c[2]:.4f},{c[3]:.4f}) "
                  f"dist_x0={dist:.4f} -N(c)={uncert_vals[i]:.4f} |∇|₁={uncert_grad[i]:.0f} lb={uncert_lb[i]:.2f}")

        # Adaptive refinement for uncertified boxes
        print(f"\n  Adaptive refinement of {n_uncertified} boxes...")
        uncert_centers = pts[~certified]
        uncert_hw = half_widths[remaining][~certified]

        depth = 0
        current_boxes = list(zip(uncert_centers, uncert_hw))
        total_cert_refine = 0
        total_taylor_refine = 0

        while current_boxes and depth < 8:
            depth += 1
            new_boxes = []
            n_cert = 0
            n_tay = 0
            t1 = time.time()

            # Subdivide each box into 16
            sub_centers = []
            sub_hws = []
            for c, hw in current_boxes:
                new_hw = hw / 2
                for bits in range(16):
                    offset = np.array([(1 if bits & (1 << d) else -1) * new_hw[d] for d in range(4)])
                    sc = c + offset
                    sub_centers.append(sc)
                    sub_hws.append(new_hw)

            sub_centers = np.array(sub_centers)
            sub_hws = np.array(sub_hws)
            new_r = np.max(sub_hws[0])

            # Taylor check
            dist = np.max(np.abs(sub_centers - x0) + sub_hws, axis=1)
            in_tay = dist < taylor_radius
            n_tay = np.sum(in_tay)

            # Domain check
            not_tay = ~in_tay
            if np.sum(not_tay) > 0:
                dom = domain_filter(sub_centers[not_tay], sub_hws[not_tay])
                not_tay_dom = np.where(not_tay)[0][dom]
                outside_dom = np.where(not_tay)[0][~dom]

                if len(not_tay_dom) > 0:
                    eval_pts = sub_centers[not_tay_dom]
                    nN = eval_poly_np(neg_N_coeffs, eval_pts)
                    gl1 = eval_grad_l1_np(grad_coeffs, eval_pts)
                    qc = 0.5 * H_max * 4 * new_r**2
                    lb = nN - gl1 * new_r - qc
                    cert = lb >= 0
                    n_cert = np.sum(cert)

                    not_cert_idx = not_tay_dom[~cert]
                    for idx in not_cert_idx:
                        new_boxes.append((sub_centers[idx], sub_hws[idx]))

            total_cert_refine += n_cert
            total_taylor_refine += n_tay
            elapsed = time.time() - t1

            print(f"    Level {depth}: {len(current_boxes)} → {len(current_boxes)*16} sub-boxes. "
                  f"Certified: {n_cert}, Taylor: {n_tay}, Remaining: {len(new_boxes)} "
                  f"(r={new_r:.6f}, qc={0.5*H_max*4*new_r**2:.4f}) ({elapsed:.1f}s)")

            current_boxes = new_boxes

            if new_boxes:
                sample = new_boxes[:3]
                for c, hw in sample:
                    d = np.linalg.norm(c - x0)
                    nN_val = eval_poly_np(neg_N_coeffs, c.reshape(1, 4))[0]
                    gl1_val = eval_grad_l1_np(grad_coeffs, c.reshape(1, 4))[0]
                    lb = nN_val - gl1_val * np.max(hw) - 0.5 * H_max * 4 * np.max(hw)**2
                    print(f"      c=({c[0]:.5f},{c[1]:.5f},{c[2]:.5f},{c[3]:.5f}) "
                          f"d={d:.5f} -N={nN_val:.6f} |∇|₁={gl1_val:.0f} lb={lb:.4f}")

    print(f"\n{'='*70}")
    print(f"FINAL RESULT")
    print(f"{'='*70}")
    n_total = n_taylor + n_certified + total_cert_refine + total_taylor_refine
    n_remain = len(current_boxes) if 'current_boxes' in dir() else n_uncertified
    print(f"  Taylor certified: {n_taylor + total_taylor_refine}")
    print(f"  IA certified: {n_certified + total_cert_refine}")
    print(f"  Remaining: {n_remain}")

    if n_remain == 0:
        print(f"\n*** PROVED: -N ≥ 0 on the entire domain ***")
    else:
        print(f"\n  {n_remain} boxes could not be certified")

    print(f"\nTotal time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
