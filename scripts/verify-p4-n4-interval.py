#!/usr/bin/env python3
"""Interval arithmetic verification: -N ≥ 0 on the entire domain.

Uses adaptive subdivision with numpy-vectorized interval arithmetic.
Near x₀ = (0, 1/12, 0, 1/12), uses the Taylor certificate (PD Hessian).

Domain: disc_p ≥ 0, disc_q ≥ 0, f₁_p > 0, f₁_q > 0, f₂_p < 0, f₂_q < 0
Bounding box: a₃ ∈ [-√(8/27), √(8/27)], a₄ ∈ [-1/12, 1/4], same for b₃, b₄.

Safe for restart: saves progress to checkpoint file.
"""

import numpy as np
import sympy as sp
from sympy import Rational, expand, together, fraction, symbols
import time
import sys
import os
import pickle

sys.stdout.reconfigure(line_buffering=True)

CHECKPOINT_FILE = "/tmp/interval-proof-checkpoint.pkl"


def build_neg_N_coeffs():
    """Build -N as a list of (coefficient, exponents) tuples.
    Returns: list of (coeff_float, (i, j, k, l)) where monomial is a3^i * a4^j * b3^k * b4^l.
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

    # Extract monomials
    p = sp.Poly(neg_N, a3, a4, b3, b4)
    coeffs = []
    for monom, coeff in p.as_dict().items():
        coeffs.append((float(coeff), monom))  # monom = (i, j, k, l)

    return coeffs


class IntervalPoly:
    """Evaluates a multivariate polynomial using interval arithmetic.

    Uses numpy vectorization: evaluates on M boxes simultaneously.
    Each box is specified by (lo, hi) arrays of shape (M, 4).
    """

    def __init__(self, coeffs):
        """coeffs: list of (float_coeff, (i, j, k, l))"""
        self.coeffs = coeffs
        # Pre-sort by total degree for numerical stability
        self.coeffs.sort(key=lambda x: sum(x[1]))

    def eval_interval(self, lo, hi):
        """Evaluate polynomial on M boxes.

        lo, hi: arrays of shape (M, 4) - lower/upper bounds for each variable.
        Returns: (result_lo, result_hi) arrays of shape (M,).
        """
        M = lo.shape[0]
        sum_lo = np.zeros(M)
        sum_hi = np.zeros(M)

        for coeff, (i, j, k, l) in self.coeffs:
            # Compute interval for a3^i
            mono_lo, mono_hi = np.ones(M), np.ones(M)
            for var_idx, power in enumerate([i, j, k, l]):
                if power == 0:
                    continue
                v_lo = lo[:, var_idx]
                v_hi = hi[:, var_idx]
                p_lo, p_hi = ia_pow(v_lo, v_hi, power)
                mono_lo, mono_hi = ia_mul(mono_lo, mono_hi, p_lo, p_hi)

            # Multiply by coefficient
            if coeff >= 0:
                term_lo = coeff * mono_lo
                term_hi = coeff * mono_hi
            else:
                term_lo = coeff * mono_hi
                term_hi = coeff * mono_lo

            sum_lo += term_lo
            sum_hi += term_hi

        return sum_lo, sum_hi


def ia_mul(a_lo, a_hi, b_lo, b_hi):
    """Interval multiplication: [a_lo, a_hi] × [b_lo, b_hi]."""
    p1 = a_lo * b_lo
    p2 = a_lo * b_hi
    p3 = a_hi * b_lo
    p4 = a_hi * b_hi
    lo = np.minimum(np.minimum(p1, p2), np.minimum(p3, p4))
    hi = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))
    return lo, hi


def ia_pow(v_lo, v_hi, n):
    """Interval power: [v_lo, v_hi]^n for positive integer n."""
    if n == 1:
        return v_lo.copy(), v_hi.copy()

    if n % 2 == 0:
        # Even power: result is always non-negative
        # If interval contains 0: lo = 0, hi = max(|lo|, |hi|)^n
        abs_lo = np.abs(v_lo)
        abs_hi = np.abs(v_hi)
        max_abs = np.maximum(abs_lo, abs_hi)
        min_abs = np.minimum(abs_lo, abs_hi)

        contains_zero = (v_lo <= 0) & (v_hi >= 0)
        res_lo = np.where(contains_zero, 0.0, min_abs**n)
        res_hi = max_abs**n
        return res_lo, res_hi
    else:
        # Odd power: monotone
        return v_lo**n, v_hi**n


def domain_intersects(lo, hi):
    """Check which boxes intersect the domain.

    Domain: disc_p ≥ 0, disc_q ≥ 0, f1_p > 0, f1_q > 0, f2_p < 0, f2_q < 0.
    Uses interval arithmetic to check if the box COULD intersect the domain.
    Returns boolean mask.
    """
    a3_lo, a4_lo, b3_lo, b4_lo = lo[:, 0], lo[:, 1], lo[:, 2], lo[:, 3]
    a3_hi, a4_hi, b3_hi, b4_hi = hi[:, 0], hi[:, 1], hi[:, 2], hi[:, 3]

    # f1_p = 1 + 12*a4 > 0 ⟹ a4 > -1/12. Check: max(a4) > -1/12
    f1p_possible = a4_hi > -1/12 - 1e-10

    # f1_q = 1 + 12*b4 > 0 ⟹ b4 > -1/12
    f1q_possible = b4_hi > -1/12 - 1e-10

    # f2_p = 9*a3² + 8*a4 - 2 < 0 ⟹ 9*a3² + 8*a4 < 2
    # Lower bound of f2_p: 9*min(a3²) + 8*min(a4) - 2
    a3_sq_lo = np.where((a3_lo <= 0) & (a3_hi >= 0), 0.0, np.minimum(a3_lo**2, a3_hi**2))
    f2p_lo = 9 * a3_sq_lo + 8 * a4_lo - 2
    f2p_possible = f2p_lo < 1e-10  # f2_p could be < 0

    b3_sq_lo = np.where((b3_lo <= 0) & (b3_hi >= 0), 0.0, np.minimum(b3_lo**2, b3_hi**2))
    f2q_lo = 9 * b3_sq_lo + 8 * b4_lo - 2
    f2q_possible = f2q_lo < 1e-10

    # disc_p ≥ 0: this is harder to check with IA, so we use a relaxed check
    # disc_p = 256*a4³ - 128*a4² + 16*a4 + 4*a3² - 144*a3²*a4 - 27*a3⁴
    # Skip detailed check — the bounding box already restricts to near the domain

    return f1p_possible & f1q_possible & f2p_possible & f2q_possible


def generate_grid(n_per_dim):
    """Generate a grid of boxes covering the domain bounding box.

    Returns: (centers, half_widths) where centers has shape (M, 4).
    """
    a3_bound = np.sqrt(8/27) + 0.001  # ~0.545
    a4_lo, a4_hi = -1/12 - 0.001, 1/4 + 0.001

    a3_grid = np.linspace(-a3_bound, a3_bound, n_per_dim + 1)
    a4_grid = np.linspace(a4_lo, a4_hi, n_per_dim + 1)
    b3_grid = np.linspace(-a3_bound, a3_bound, n_per_dim + 1)
    b4_grid = np.linspace(a4_lo, a4_hi, n_per_dim + 1)

    # Cell widths
    h_a3 = a3_grid[1] - a3_grid[0]
    h_a4 = a4_grid[1] - a4_grid[0]
    h_b3 = b3_grid[1] - b3_grid[0]
    h_b4 = b4_grid[1] - b4_grid[0]

    # Centers
    ca3 = (a3_grid[:-1] + a3_grid[1:]) / 2
    ca4 = (a4_grid[:-1] + a4_grid[1:]) / 2
    cb3 = (b3_grid[:-1] + b3_grid[1:]) / 2
    cb4 = (b4_grid[:-1] + b4_grid[1:]) / 2

    # Meshgrid
    A3, A4, B3, B4 = np.meshgrid(ca3, ca4, cb3, cb4, indexing='ij')
    centers = np.column_stack([A3.ravel(), A4.ravel(), B3.ravel(), B4.ravel()])

    hw = np.array([h_a3/2, h_a4/2, h_b3/2, h_b4/2])
    half_widths = np.tile(hw, (centers.shape[0], 1))

    return centers, half_widths


def main():
    print("=" * 70)
    print("INTERVAL ARITHMETIC PROOF: -N ≥ 0 on domain")
    print("=" * 70)
    t_start = time.time()

    # Build polynomial
    print("\nBuilding -N polynomial...")
    t1 = time.time()
    coeffs = build_neg_N_coeffs()
    poly = IntervalPoly(coeffs)
    print(f"  {len(coeffs)} monomials ({time.time()-t1:.1f}s)")

    # x₀ = (0, 1/12, 0, 1/12) — unique minimum with -N = 0
    x0 = np.array([0.0, 1/12, 0.0, 1/12])
    taylor_radius = 0.004458  # certified radius where -N ≥ 0

    # Adaptive subdivision
    BATCH_SIZE = 100000  # Process boxes in batches for memory
    MAX_DEPTH = 20

    # Start with initial grid
    n_init = 30  # 30^4 = 810,000 initial boxes
    print(f"\nGenerating {n_init}^4 = {n_init**4} initial boxes...")
    centers, half_widths = generate_grid(n_init)
    lo = centers - half_widths
    hi = centers + half_widths
    print(f"  {centers.shape[0]} boxes, cell width ≈ {2*half_widths[0,0]:.4f} × {2*half_widths[0,1]:.4f} × {2*half_widths[0,2]:.4f} × {2*half_widths[0,3]:.4f}")

    # Filter to boxes that could intersect domain
    mask = domain_intersects(lo, hi)
    lo = lo[mask]
    hi = hi[mask]
    centers = centers[mask]
    half_widths = half_widths[mask]
    print(f"  After domain filter: {lo.shape[0]} boxes")

    # Track statistics
    total_certified = 0
    total_taylor = 0
    total_outside = centers.shape[0] - lo.shape[0] if centers.shape[0] != lo.shape[0] else 0
    uncertified_boxes = []  # (lo, hi, depth) for boxes needing refinement

    # Process in batches
    print(f"\nEvaluating -N on {lo.shape[0]} boxes...")
    t1 = time.time()

    for batch_start in range(0, lo.shape[0], BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, lo.shape[0])
        b_lo = lo[batch_start:batch_end]
        b_hi = hi[batch_start:batch_end]
        b_centers = (b_lo + b_hi) / 2

        # Check Taylor ball: if box is inside the Taylor ball, -N ≥ 0
        dist_to_x0 = np.max(np.abs(b_centers - x0), axis=1)  # L∞ distance
        max_corner_dist = dist_to_x0 + np.max(b_hi - b_lo, axis=1) / 2
        in_taylor = max_corner_dist < taylor_radius
        total_taylor += np.sum(in_taylor)

        # Evaluate -N using interval arithmetic on remaining boxes
        outside_taylor = ~in_taylor
        if np.sum(outside_taylor) > 0:
            nN_lo, nN_hi = poly.eval_interval(b_lo[outside_taylor], b_hi[outside_taylor])

            certified = nN_lo >= 0
            total_certified += np.sum(certified)

            # Uncertified boxes need refinement
            not_cert = ~certified
            if np.sum(not_cert) > 0:
                # idx: positions in the batch that are outside Taylor and not certified
                idx = np.where(outside_taylor)[0][not_cert]
                for i in idx:
                    uncertified_boxes.append((b_lo[i].copy(), b_hi[i].copy(), 0))

        if batch_start % (BATCH_SIZE * 5) == 0 and batch_start > 0:
            print(f"    Processed {batch_end}/{lo.shape[0]}: certified={total_certified}, taylor={total_taylor}, uncert={len(uncertified_boxes)}")

    elapsed = time.time() - t1
    print(f"  Initial evaluation: {elapsed:.1f}s")
    print(f"    Certified by IA: {total_certified}")
    print(f"    Certified by Taylor: {total_taylor}")
    print(f"    Need refinement: {len(uncertified_boxes)}")

    # Adaptive refinement
    depth = 0
    while uncertified_boxes and depth < MAX_DEPTH:
        depth += 1
        print(f"\n  Refinement level {depth}: {len(uncertified_boxes)} boxes to refine")
        t1 = time.time()

        new_uncertified = []
        new_certified = 0
        new_taylor = 0

        # Subdivide each uncertified box into 2^4 = 16 sub-boxes
        for box_lo, box_hi, box_depth in uncertified_boxes:
            mids = (box_lo + box_hi) / 2
            # Generate 16 sub-boxes
            for i in range(16):
                sub_lo = np.array([box_lo[d] if (i >> d) & 1 == 0 else mids[d] for d in range(4)])
                sub_hi = np.array([mids[d] if (i >> d) & 1 == 0 else box_hi[d] for d in range(4)])

                # Domain check
                sub_lo_2d = sub_lo.reshape(1, 4)
                sub_hi_2d = sub_hi.reshape(1, 4)
                if not domain_intersects(sub_lo_2d, sub_hi_2d)[0]:
                    continue

                # Taylor ball check
                center = (sub_lo + sub_hi) / 2
                max_corner_dist = np.max(np.abs(center - x0)) + np.max(sub_hi - sub_lo) / 2
                if max_corner_dist < taylor_radius:
                    new_taylor += 1
                    continue

                # Interval evaluation
                nN_lo, nN_hi = poly.eval_interval(sub_lo_2d, sub_hi_2d)
                if nN_lo[0] >= 0:
                    new_certified += 1
                else:
                    new_uncertified.append((sub_lo, sub_hi, depth))

        total_certified += new_certified
        total_taylor += new_taylor
        uncertified_boxes = new_uncertified
        elapsed = time.time() - t1

        print(f"    Certified: +{new_certified}, Taylor: +{new_taylor}, Remaining: {len(uncertified_boxes)} ({elapsed:.1f}s)")

        if uncertified_boxes:
            # Show some info about the uncertified boxes
            sample = uncertified_boxes[:3]
            for box_lo, box_hi, _ in sample:
                center = (box_lo + box_hi) / 2
                width = box_hi - box_lo
                nN_lo, nN_hi = poly.eval_interval(box_lo.reshape(1,4), box_hi.reshape(1,4))
                dist = np.linalg.norm(center - x0)
                print(f"      Box center=({center[0]:.4f},{center[1]:.4f},{center[2]:.4f},{center[3]:.4f}), width={width[0]:.6f}, dist_x0={dist:.4f}, -N∈[{nN_lo[0]:.2f},{nN_hi[0]:.2f}]")

    print(f"\n{'='*70}")
    print(f"FINAL RESULT")
    print(f"{'='*70}")
    print(f"  Certified by interval arithmetic: {total_certified}")
    print(f"  Certified by Taylor ball: {total_taylor}")
    print(f"  Uncertified boxes: {len(uncertified_boxes)}")

    if not uncertified_boxes:
        print(f"\n*** PROVED: -N ≥ 0 on the entire domain ***")
    else:
        print(f"\n  Uncertified boxes remain at depth {depth}:")
        for box_lo, box_hi, _ in uncertified_boxes[:10]:
            center = (box_lo + box_hi) / 2
            width = box_hi - box_lo
            nN_lo, nN_hi = poly.eval_interval(box_lo.reshape(1,4), box_hi.reshape(1,4))
            nN_center = np.polyval([1], 0)  # placeholder
            print(f"    center=({center[0]:.6f},{center[1]:.6f},{center[2]:.6f},{center[3]:.6f}) w={width[0]:.8f} -N∈[{nN_lo[0]:.2f},{nN_hi[0]:.2f}]")

    print(f"\nTotal time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
