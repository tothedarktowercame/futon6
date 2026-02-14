#!/usr/bin/env python3
"""P4 Path2: Structure analysis of B(P,Q) - the odd-odd part.

K_red = A(P,Q) + pq·B(P,Q) where P=p², Q=q².
We know B ≤ 0 numerically. Prove it algebraically.
B(P,Q) has degree 2 in (P,Q): B = b10·P + b01·Q + b11·PQ (or similar).
"""
import json
import time
import numpy as np
import sympy as sp
from pathlib import Path

def pr(*args):
    print(*args, flush=True)

def build_exact_k_red():
    s, t, u, v, a, b = sp.symbols("s t u v a b")
    r, x, y, p, q = sp.symbols("r x y p q")
    def T2R_num(ss, uu, aa):
        return 8*aa*(ss**2 - 4*aa)**2 - ss*uu**2*(ss**2 + 60*aa)
    def T2R_den(ss, uu, aa):
        return 2*(ss**2 + 12*aa)*(2*ss**3 - 8*ss*aa - 9*uu**2)
    S = s+t; U = u+v; A_conv = a+b+s*t/6
    surplus_num = sp.expand(
        T2R_num(S,U,A_conv)*T2R_den(s,u,a)*T2R_den(t,v,b)
        - T2R_num(s,u,a)*T2R_den(S,U,A_conv)*T2R_den(t,v,b)
        - T2R_num(t,v,b)*T2R_den(S,U,A_conv)*T2R_den(s,u,a))
    subs_norm = {t: r*s, a: x*s**2/4, b: y*r**2*s**2/4,
                 u: p*s**sp.Rational(3,2), v: q*s**sp.Rational(3,2)}
    K_exact = sp.expand(surplus_num.subs(subs_norm) / s**16)
    return sp.expand(sp.cancel(K_exact / r**2))

def main():
    t0 = time.time()
    r, x, y, p, q = sp.symbols("r x y p q")
    P, Q = sp.symbols("P Q")

    pr("Building K_red...")
    K_red = build_exact_k_red()
    
    pr("Decomposing A, B...")
    poly = sp.Poly(K_red, p, q)
    B_expr = sp.Integer(0)
    for (i, j), c in poly.as_dict().items():
        if i % 2 == 1 and j % 2 == 1:
            # Term is c * p^i * q^j = c * p * q * p^(i-1) * q^(j-1) = pq * c * P^((i-1)/2) Q^((j-1)/2)
            B_expr += c * p**(i-1) * q**(j-1)
    
    B_PQ = sp.expand(B_expr.subs({p**2: P, q**2: Q}))
    pr(f"B(P,Q) has degree structure:")
    B_poly = sp.Poly(B_PQ, P, Q)
    B_dict = B_poly.as_dict()
    
    coeffs = {}
    for (i,j), c in sorted(B_dict.items()):
        label = f"b{i}{j}"
        c_factored = sp.factor(c)
        pr(f"\n  {label} (P^{i}Q^{j} coeff):")
        pr(f"    = {c_factored}")
        coeffs[label] = {"raw": str(sp.expand(c)), "factored": str(c_factored)}
    
    # B at corners of box
    Pmax = sp.Rational(2,9)*(1-x)
    Qmax = sp.Rational(2,9)*r**3*(1-y)
    
    pr("\n=== B at boundary corners ===")
    corners = {
        "(0,0)": {P: 0, Q: 0},
        "(Pmax,0)": {P: Pmax, Q: 0},
        "(0,Qmax)": {P: 0, Q: Qmax},
        "(Pmax,Qmax)": {P: Pmax, Q: Qmax},
    }
    for name, subs in corners.items():
        val = sp.factor(sp.expand(B_PQ.subs(subs)))
        pr(f"  B{name} = {val}")
    
    # B on edges
    pr("\n=== B on edges ===")
    B_P0 = sp.factor(sp.expand(B_PQ.subs(Q, 0)))
    B_0Q = sp.factor(sp.expand(B_PQ.subs(P, 0)))
    B_PmQ = sp.factor(sp.expand(B_PQ.subs(P, Pmax)))
    B_PQm = sp.factor(sp.expand(B_PQ.subs(Q, Qmax)))
    
    pr(f"  B(P,0) = {B_P0}")
    pr(f"  B(0,Q) = {B_0Q}")
    pr(f"  B(Pmax,Q) = {B_PmQ}")
    pr(f"  B(P,Qmax) = {B_PQm}")
    
    # Numeric sign check
    pr("\n=== Numeric B sign check ===")
    B_fn = sp.lambdify((r, x, y, P, Q), B_PQ, "numpy")
    
    rng = np.random.default_rng(42)
    n_test = 500000
    n_pos = 0
    n_neg = 0
    max_B = float("-inf")
    
    for _ in range(n_test):
        rv = float(np.exp(rng.uniform(np.log(1e-3), np.log(1e3))))
        xv = float(rng.uniform(1e-6, 1 - 1e-6))
        yv = float(rng.uniform(1e-6, 1 - 1e-6))
        Pm = 2*(1-xv)/9
        Qm = 2*rv**3*(1-yv)/9
        Pv = rng.uniform(0, Pm)
        Qv = rng.uniform(0, Qm)
        
        bv = float(B_fn(rv, xv, yv, Pv, Qv))
        if bv > 1e-10:
            n_pos += 1
        elif bv < -1e-10:
            n_neg += 1
        if bv > max_B:
            max_B = bv
    
    pr(f"B > 0: {n_pos}/{n_test}")
    pr(f"B < 0: {n_neg}/{n_test}")
    pr(f"max B: {max_B:.6e}")
    
    # ---------------------------------------------------------------
    # Can we prove B <= 0 on the box?
    # B is linear in P for fixed Q (or vice versa), so max on edges.
    # B = b10*P + b01*Q + b11*PQ = P*(b10 + b11*Q) + Q*b01
    # On [0,Pmax]x[0,Qmax], max at P=Pmax if (b10+b11*Q)>0, else at P=0.
    # Since b10+b11*Q is linear in Q, its sign depends on Q.
    # ---------------------------------------------------------------
    pr("\n=== B structure: linearity analysis ===")
    # B = P·(b10 + b11·Q) + Q·b01
    # = Q·(b01 + b11·P) + P·b10
    b10_v = B_dict.get((1,0), 0)
    b01_v = B_dict.get((0,1), 0)
    b11_v = B_dict.get((1,1), 0)
    b00_v = B_dict.get((0,0), 0)
    
    pr(f"B has constant term: {b00_v != 0}")
    
    # For B <= 0 on box, sufficient to check 4 corners (since bilinear in P,Q)
    # IF B is exactly bilinear: B = b00 + b10P + b01Q + b11PQ
    # Then max is at a corner.
    
    # Check if B is indeed bilinear (no higher terms)
    max_deg = max(i+j for (i,j) in B_dict.keys())
    pr(f"Max total degree of B: {max_deg}")
    
    if max_deg <= 2:
        pr("B has quadratic terms, not purely bilinear.")
        # But it might still be manageable
        
    # Check all terms
    for (i,j), c in sorted(B_dict.items()):
        pr(f"  B has P^{i}Q^{j} term: {i+j > 0}")
    
    # If there's a P² or Q² term, B is NOT bilinear
    has_P2 = (2,0) in B_dict
    has_Q2 = (0,2) in B_dict
    has_PQ = (1,1) in B_dict
    pr(f"  Has P² term: {has_P2}")
    pr(f"  Has Q² term: {has_Q2}")
    pr(f"  Has PQ term: {has_PQ}")
    
    # B on edges should be ≤ 0. We already factored them above.
    # If B is degree 2 and ≤ 0 on all four edges, is it ≤ 0 inside?
    # For a quadratic on a rectangle, if ≤ 0 on boundary, then ≤ 0 inside
    # (since max of a quadratic on a convex set is on the boundary).
    # WAIT: that's only for convex functions. A general quadratic on a rectangle
    # need not have max on boundary.
    # But: B is degree 2 in (P,Q). Its Hessian is constant (w.r.t. P,Q).
    # If Hessian is NSD (negative semi-definite), B is concave → max on boundary.
    
    pr("\n=== Hessian of B w.r.t. (P,Q) ===")
    # B_PP = 2*b20 (if exists)
    # B_QQ = 2*b02 (if exists)
    # B_PQ = b11
    b20_v = B_dict.get((2,0), sp.Integer(0))
    b02_v = B_dict.get((0,2), sp.Integer(0))
    
    pr(f"  B_PP = 2*b20 = {sp.factor(2*b20_v)}")
    pr(f"  B_QQ = 2*b02 = {sp.factor(2*b02_v)}")
    pr(f"  B_PQ = b11 = {sp.factor(b11_v)}")
    
    # Hessian det = 4*b20*b02 - b11²
    hess_det = sp.expand(4*b20_v*b02_v - b11_v**2)
    pr(f"  Hessian det: {sp.factor(hess_det)}")
    
    # If B_PP <= 0, B_QQ <= 0, and det >= 0: B is concave → max on boundary
    # Alternatively, check numerically
    
    b20_fn = sp.lambdify((r, x, y), b20_v, "numpy") if b20_v != 0 else lambda r,x,y: 0
    b02_fn = sp.lambdify((r, x, y), b02_v, "numpy") if b02_v != 0 else lambda r,x,y: 0
    
    rng2 = np.random.default_rng(99)
    n_concave = 0
    for _ in range(100000):
        rv = float(np.exp(rng2.uniform(np.log(1e-3), np.log(1e3))))
        xv = float(rng2.uniform(1e-6, 1-1e-6))
        yv = float(rng2.uniform(1e-6, 1-1e-6))
        
        bpp = float(2*b20_fn(rv, xv, yv))
        bqq = float(2*b02_fn(rv, xv, yv))
        
        if bpp <= 1e-10 and bqq <= 1e-10:
            n_concave += 1
    
    pr(f"  B concave (B_PP, B_QQ <= 0): {n_concave}/100000")
    
    elapsed = time.time() - t0
    pr(f"\nRuntime: {elapsed:.1f}s")
    
    out = {
        "B_coefficients": coeffs,
        "B_sign": {"pos": n_pos, "neg": n_neg, "max_B": max_B},
    }
    Path("data/first-proof/p4-path2-B-structure.json").write_text(json.dumps(out, indent=2, default=str))
    pr("Wrote results")

if __name__ == "__main__":
    main()
