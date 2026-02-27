#!/usr/bin/env python3
"""P4 Path2: Enhanced discriminant via u >= u²/Pmax conversion.

Key idea: on the box [0,Pmax]×[0,Qmax], u = Pmax-P satisfies u >= u²/Pmax.
So ā₁₀·u >= (ā₁₀/Pmax)·u², and similarly ā₀₁·v >= (ā₀₁/Qmax)·v².

This converts linear terms into quadratic diagonal terms:
  Ā >= (ā₂₀+ā₁₀/Pmax)u² + ā₁₁uv + (ā₀₂+ā₀₁/Qmax)v² + cubic terms

If 4(ā₂₀+ā₁₀/Pmax)(ā₀₂+ā₀₁/Qmax) >= ā₁₁², the enhanced quadratic is PSD → Ā >= 0.
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

def decompose_coeffs(K_red):
    r, x, y, p, q = sp.symbols("r x y p q")
    P, Q = sp.symbols("P Q")
    poly = sp.Poly(K_red, p, q)
    A_expr = sp.Integer(0)
    for (i, j), c in poly.as_dict().items():
        if i % 2 == 0 and j % 2 == 0:
            A_expr += c * p**i * q**j
    A_PQ = sp.expand(A_expr.subs({p**2: P, q**2: Q}))
    Ad = sp.Poly(A_PQ, P, Q).as_dict()
    return {k: sp.expand(v) for k, v in {
        "a00": Ad[(0,0)], "a01": Ad[(0,1)], "a02": Ad[(0,2)],
        "a10": Ad[(1,0)], "a11": Ad[(1,1)], "a12": Ad[(1,2)],
        "a20": Ad[(2,0)],
    }.items()}

def main():
    t0 = time.time()
    r, x, y = sp.symbols("r x y")

    pr("Building K_red & decomposing...")
    K_red = build_exact_k_red()
    c = decompose_coeffs(K_red)
    a00=c["a00"]; a01=c["a01"]; a02=c["a02"]; a10=c["a10"]
    a11=c["a11"]; a12=c["a12"]; a20=c["a20"]

    Pmax = sp.Rational(2,9)*(1-x)
    Qmax = sp.Rational(2,9)*r**3*(1-y)

    # Taylor coefficients
    a10_bar = sp.expand(-(a10 + 2*a20*Pmax + a11*Qmax + 2*a12*Pmax*Qmax + a12*Qmax**2))
    a01_bar = sp.expand(-(a01 + a11*Pmax + 2*a02*Qmax + a12*Pmax**2 + 2*a12*Pmax*Qmax))
    a20_bar = sp.expand(a20 + a12*Qmax)
    a02_bar = sp.expand(a02 + a12*Pmax)
    a11_bar = sp.expand(a11 + 2*a12*(Pmax + Qmax))

    # Enhanced quadratic coefficients
    pr("Computing enhanced coefficients...")
    # ā₁₀/Pmax = ā₁₀ * 9/(2(1-x))
    # ā₀₁/Qmax = ā₀₁ * 9/(2r³(1-y))
    A20_enh = sp.expand(a20_bar + a10_bar * 9 / (2*(1-x)))
    A02_enh = sp.expand(a02_bar + a01_bar * 9 / (2*r**3*(1-y)))
    
    pr("Computing enhanced discriminant...")
    disc_enh = sp.expand(4*A20_enh*A02_enh - a11_bar**2)
    
    # Factor out known pieces
    # A20_enh and A02_enh should simplify
    pr("Simplifying A20_enh...")
    A20_enh_simplified = sp.cancel(A20_enh)
    pr(f"A20_enh = {sp.factor(A20_enh_simplified)}")
    
    pr("Simplifying A02_enh...")
    A02_enh_simplified = sp.cancel(A02_enh)
    pr(f"A02_enh = {sp.factor(A02_enh_simplified)}")
    
    # Numeric check
    pr("\n=== Numeric check: enhanced discriminant ===")
    A20_fn = sp.lambdify((r, x, y), A20_enh, "numpy")
    A02_fn = sp.lambdify((r, x, y), A02_enh, "numpy")
    a11b_fn = sp.lambdify((r, x, y), a11_bar, "numpy")
    
    rng = np.random.default_rng(42)
    n_test = 500000
    n_disc_neg = 0
    n_disc_pos = 0
    min_disc = float("inf")
    min_disc_params = None
    
    for _ in range(n_test):
        rv = float(np.exp(rng.uniform(np.log(1e-4), np.log(1e4))))
        xv = float(rng.uniform(1e-8, 1 - 1e-8))
        yv = float(rng.uniform(1e-8, 1 - 1e-8))
        
        a20v = float(A20_fn(rv, xv, yv))
        a02v = float(A02_fn(rv, xv, yv))
        a11v = float(a11b_fn(rv, xv, yv))
        
        dv = 4*a20v*a02v - a11v**2
        
        if dv < -1e-6 * a11v**2:
            n_disc_neg += 1
        if dv > 1e-6 * a11v**2:
            n_disc_pos += 1
        
        # Ratio: 4*A20*A02 / ā₁₁²
        if a11v**2 > 0:
            ratio = 4*a20v*a02v / a11v**2
            if ratio < min_disc:
                min_disc = ratio
                min_disc_params = (rv, xv, yv)
    
    pr(f"Enhanced disc >= 0: {n_disc_pos}/{n_test}")
    pr(f"Enhanced disc < 0: {n_disc_neg}/{n_test}")
    pr(f"Min ratio 4A20A02/ā₁₁²: {min_disc:.6f}")
    if min_disc_params:
        rv, xv, yv = min_disc_params
        pr(f"  at r={rv:.6f}, x={xv:.6f}, y={yv:.6f}")
    
    # ---------------------------------------------------------------
    # If enhanced disc >= 0, try to prove it symbolically
    # ---------------------------------------------------------------
    if n_disc_neg == 0:
        pr("\n=== Enhanced discriminant is ALWAYS >= 0! Attempting proof... ===")
        
        # disc_enh = 4*A20_enh*A02_enh - a11_bar²
        # Let's compute it in cleared-denominator form
        # A20_enh = a20_bar + 9*a10_bar/(2(1-x))
        # A02_enh = a02_bar + 9*a01_bar/(2r³(1-y))
        # So 4*A20*A02 = 4*(a20_bar + 9a10_bar/(2(1-x)))*(a02_bar + 9a01_bar/(2r³(1-y)))
        # Clear: 4(1-x)*r³(1-y) * A20 * A02 = (2(1-x)*a20_bar + 9*a10_bar)(2r³(1-y)*a02_bar + 9*a01_bar)
        
        # Denominators: 4(1-x)*r³(1-y)
        lhs_cleared = sp.expand(
            (2*(1-x)*a20_bar + 9*a10_bar) * (2*r**3*(1-y)*a02_bar + 9*a01_bar)
        )
        rhs_cleared = sp.expand(a11_bar**2 * (1-x) * r**3 * (1-y))
        delta_cleared = sp.expand(lhs_cleared - rhs_cleared)
        
        n_terms = len(sp.Add.make_args(delta_cleared))
        pr(f"Cleared delta has {n_terms} terms")
        
        # Try factoring pieces
        pr("Factoring LHS factor 1...")
        f1 = sp.factor(sp.expand(2*(1-x)*a20_bar + 9*a10_bar))
        pr(f"  2(1-x)*ā₂₀ + 9*ā₁₀ = {f1}")
        
        pr("Factoring LHS factor 2...")
        f2 = sp.factor(sp.expand(2*r**3*(1-y)*a02_bar + 9*a01_bar))
        pr(f"  2r³(1-y)*ā₀₂ + 9*ā₀₁ = {f2}")
        
        # Try to extract common factors from delta
        pr("\nLooking for factors in delta...")
        # Try pos_core first
        pos_core = sp.expand(3*r**2*y + r**2 + 4*r + 3*x + 1)
        d_by_pc = sp.cancel(delta_cleared / pos_core)
        try:
            d_test = sp.Poly(sp.expand(d_by_pc), r, x, y)
            pr("delta / pos_core is polynomial")
            d_by_pc2 = sp.cancel(d_by_pc / pos_core)
            try:
                d_test2 = sp.Poly(sp.expand(d_by_pc2), r, x, y)
                pr("delta / pos_core² is polynomial")
                n2 = len(sp.Add.make_args(sp.expand(d_by_pc2)))
                pr(f"  with {n2} terms")
            except:
                pr("delta / pos_core² is NOT polynomial")
        except:
            pr("delta / pos_core is NOT polynomial")
    
    else:
        pr(f"\nEnhanced discriminant fails for {n_disc_neg} samples.")
        pr("Exploring compensation from cubic terms...")
        
        # When disc_enh < 0, we still have cubic terms 1296rL(u²v+uv²) to help
        # For L >= 0, these are positive and contribute.
        # Maybe a weighted version: use fraction α of linear for quadratic enhancement,
        # and keep (1-α) for cubic compensation
        
        # Or try different enhancement: instead of u >= u²/Pmax (which is tight at u=0,Pmax),
        # use a parameterized bound.
        
        # Actually, try the approach: for each v, the quadratic in u is:
        # (ā₂₀ + 1296rLv)u² + (ā₁₀ + ā₁₁v + 1296rLv²)u + (ā₀₁v + ā₀₂v²)
        # Minimum over u >= 0. If the linear coeff is >= 0, min at u=0, Ā = ā₀₁v+ā₀₂v² >= 0.
        # If linear coeff < 0, min at u* where derivative = 0, and we need Ā(u*,v) >= 0.
        # Ā(u*,v) = (ā₀₁v + ā₀₂v²) - (ā₁₀ + ā₁₁v + 1296rLv²)² / (4(ā₂₀ + 1296rLv))
        
        # This is a function of v alone. Need this >= 0 for v in [0, Qmax].
        pass

    elapsed = time.time() - t0
    pr(f"\nRuntime: {elapsed:.1f}s")

    out = {
        "enhanced_disc": {
            "pos_count": n_disc_pos,
            "neg_count": n_disc_neg,
            "min_ratio": min_disc,
        }
    }
    Path("data/first-proof/p4-path2-A-enhanced-disc.json").write_text(json.dumps(out, indent=2))
    pr("Wrote results")

if __name__ == "__main__":
    main()
