#!/usr/bin/env python3
"""P4 Path2: Quadratic-in-u reduction for A >= 0.

Key idea: Ā(u,v) is quadratic in u for fixed v:
  Ā = A_u·u² + D_u·u + G(v)
where A_u(v) = ā₂₀ + 1296rLv, D_u(v) = ā₁₀ + ā₁₁v + 1296rLv², G(v) = ā₀₁v + ā₀₂v²

Cases:
1. A_u ≤ 0: concave, min at boundary, both boundaries ≥ 0 ✓
2. A_u > 0 and u* ≤ 0 (D_u ≥ 0): min at u=0, Ā = G(v) ≥ 0 ✓  
3. A_u > 0 and u* ≥ Pmax: min at u=Pmax, Ā = A(0, Qmax-v) ≥ 0 (edge) ✓
4. A_u > 0 and 0 < u* < Pmax: need Ā(u*,v) = G(v) - D_u²/(4A_u) ≥ 0
   i.e., Φ(v) := 4A_u·G - D_u² ≥ 0

So the ENTIRE proof of A ≥ 0 reduces to: Φ(v) ≥ 0 for v in the relevant range.
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
    v_sym = sp.Symbol("v", nonnegative=True)

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
    L = sp.expand(9*x**2 - 27*x*y*(1+r) + 3*x*(r-1) + 9*r*y**2 - 3*r*y + 2*r + 3*y + 2)

    # Quadratic-in-u components
    A_u = sp.expand(a20_bar + 1296*r*L*v_sym)  # leading coeff of quadratic in u
    D_u = sp.expand(a10_bar + a11_bar*v_sym + 1296*r*L*v_sym**2)  # linear coeff
    G_v = sp.expand(a01_bar*v_sym + a02_bar*v_sym**2)  # constant (in u) part

    # Discriminant polynomial Φ(v) = 4·A_u·G_v - D_u²
    pr("Computing Φ(v) = 4·A_u·G - D_u²...")
    Phi = sp.expand(4*A_u*G_v - D_u**2)
    
    Phi_poly = sp.Poly(Phi, v_sym)
    pr(f"Φ(v) degree in v: {Phi_poly.degree()}")
    pr(f"Φ(v) terms: {len(sp.Add.make_args(Phi))}")
    
    # Extract Phi coefficients in v
    Phi_dict = Phi_poly.as_dict()
    for (deg,), coeff in sorted(Phi_dict.items()):
        cf = sp.factor(coeff)
        cf_str = str(cf)[:150]
        pr(f"  v^{deg}: {cf_str}{'...' if len(str(cf))>150 else ''}")

    # Key check: Φ(0) = -ā₁₀² ≤ 0 (expected, since at v=0 the critical point u*<0)
    Phi_0 = sp.expand(Phi.subs(v_sym, 0))
    pr(f"\nΦ(0) = {sp.factor(Phi_0)}")
    
    # Φ(Qmax): at the corner
    Phi_Qmax = sp.expand(Phi.subs(v_sym, Qmax))
    pr(f"Φ(Qmax) = {sp.factor(Phi_Qmax)}")
    
    # A_u at Qmax: should be a20 (since ā₂₀ + 1296rL·Qmax = a20 + a12·Qmax + 1296rL·Qmax = a20 since a12 = -1296rL)
    A_u_Qmax = sp.expand(A_u.subs(v_sym, Qmax))
    pr(f"A_u(Qmax) = a20? {sp.expand(A_u_Qmax - a20) == 0}")

    # v* where D_u(v*) = 0: quadratic in v
    # D_u = ā₁₀ + ā₁₁·v + 1296rL·v²
    # v* = (-ā₁₁ ± √(ā₁₁² - 4·1296rL·ā₁₀)) / (2·1296rL)
    
    # ---------------------------------------------------------------
    # Numeric verification
    # ---------------------------------------------------------------
    pr("\n=== Numeric check: Φ(v) ≥ 0 where needed ===")
    Phi_fn = sp.lambdify((r, x, y, v_sym), Phi, "numpy")
    A_u_fn = sp.lambdify((r, x, y, v_sym), A_u, "numpy")
    D_u_fn = sp.lambdify((r, x, y, v_sym), D_u, "numpy")
    G_fn = sp.lambdify((r, x, y, v_sym), G_v, "numpy")
    
    rng = np.random.default_rng(42)
    n_test = 500000
    n_phi_neg = 0  # Φ < 0 where actually needed
    n_phi_neg_boundary = 0  # Φ < 0 but u* not in (0, Pmax)
    n_A_neg = 0  # Ā < 0 (the actual thing we care about)
    min_phi_needed = float("inf")
    
    for _ in range(n_test):
        rv = float(np.exp(rng.uniform(np.log(1e-3), np.log(1e3))))
        xv = float(rng.uniform(1e-6, 1 - 1e-6))
        yv = float(rng.uniform(1e-6, 1 - 1e-6))
        
        Pm = 2*(1-xv)/9
        Qm = 2*rv**3*(1-yv)/9
        
        # Sample v from [0, Qmax]
        vv = rng.uniform(0, Qm)
        
        au = float(A_u_fn(rv, xv, yv, vv))
        du = float(D_u_fn(rv, xv, yv, vv))
        gv = float(G_fn(rv, xv, yv, vv))
        phi_v = float(Phi_fn(rv, xv, yv, vv))
        
        # Check if interior critical point exists
        if au > 1e-15:  # convex
            u_star = -du / (2*au)
            if u_star > 1e-15 and u_star < Pm - 1e-15:
                # Interior critical point
                A_val = gv - du**2 / (4*au)  # = phi_v / (4*au)
                if A_val < -1e-10:
                    n_A_neg += 1
                if phi_v < -1e-6 * abs(gv + du**2/(4*au) + 1):
                    n_phi_neg += 1
                if phi_v < min_phi_needed:
                    min_phi_needed = phi_v
    
    pr(f"Φ < 0 at interior critical points: {n_phi_neg}/{n_test}")
    pr(f"Ā < 0 at interior critical points: {n_A_neg}/{n_test}")
    pr(f"min Φ (where needed): {min_phi_needed:.6e}")
    
    # ---------------------------------------------------------------
    # Try to factor Φ out of v
    # ---------------------------------------------------------------
    pr("\n=== Factoring Φ(v) ===")
    # Φ(0) = -ā₁₀² ≤ 0. So v=0 is NOT a root of Φ in general.
    # But: we only need Φ ≥ 0 for v > v*, and at v=v*, D_u=0 → Φ = 4A_u·G - 0 = 4A_u·G ≥ 0.
    # So Φ(v*) ≥ 0 ✓.
    
    # Can we write Φ(v) = (stuff)·D_u(v)² + (remainder)?
    # Or better: substitute v → w where D_u(v*) = 0. 
    # Since we only need Φ ≥ 0 where D_u < 0 (i.e., v > v*):
    # and Φ(v*) = 4·A_u(v*)·G(v*) ≥ 0...
    
    # Actually, let's check: is Φ(v)/v² a clean polynomial?
    # Φ = v^0·(-ā₁₀²) + v^1·(...) + v²·(...) + v³·(...) + v⁴·(...)
    # Not divisible by v.
    
    # But: we want Φ ≥ 0 only when D_u ≤ 0. Note D_u = ā₁₀ + ā₁₁v + 1296rLv².
    # When D_u ≤ 0: ā₁₀ + ā₁₁v + 1296rLv² ≤ 0.
    # So -D_u ≥ 0, and D_u² ≥ 0.
    # Φ = 4A_u·G - D_u² ≥ 0 ⟺ 4A_u·G ≥ D_u²
    
    # Can we show: G/D_u² is bounded below when D_u is small?
    # At the transition D_u=0: Φ = 4A_u·G ≥ 0 ✓ (since A_u > 0 and G ≥ 0)
    # For v slightly beyond v*: D_u = O(v-v*), D_u² = O((v-v*)²)
    # G ≥ G(v*) > 0 (assuming v* > 0)
    # So Φ ≈ 4A_u·G - O((v-v*)²) > 0 for small |v-v*|.
    
    # The question is whether Φ stays ≥ 0 all the way to v=Qmax.
    
    # Try: Φ + D_u² · something ≥ 0 where "something" is chosen to make it obvious
    # Φ + α·D_u² = 4A_u·G - (1-α)D_u²
    
    # Or: try dividing Φ by D_u to see if the quotient is informative
    # Φ = 4A_u·G - D_u² = q(v)·D_u + remainder(v)
    pr("Polynomial division Φ / D_u...")
    Phi_poly_v = sp.Poly(Phi, v_sym)
    D_u_poly_v = sp.Poly(D_u, v_sym)
    
    quot, rem = sp.div(Phi_poly_v.as_expr(), D_u_poly_v.as_expr(), v_sym)
    pr(f"Φ = ({sp.factor(quot)})·D_u + ({sp.factor(rem)})")
    
    # Check sign of remainder
    rem_fn = sp.lambdify((r, x, y), rem, "numpy")
    rng2 = np.random.default_rng(99)
    n_rem_neg = 0
    for _ in range(100000):
        rv = float(np.exp(rng2.uniform(np.log(1e-3), np.log(1e3))))
        xv = float(rng2.uniform(1e-6, 1-1e-6))
        yv = float(rng2.uniform(1e-6, 1-1e-6))
        val = float(rem_fn(rv, xv, yv))
        if val < -1e-10:
            n_rem_neg += 1
    pr(f"Remainder < 0: {n_rem_neg}/100000")
    
    elapsed = time.time() - t0
    pr(f"\nRuntime: {elapsed:.1f}s")
    
    out = {
        "phi_degree": Phi_poly.degree(),
        "phi_terms": len(sp.Add.make_args(Phi)),
        "phi_neg_needed": n_phi_neg,
        "A_neg_at_critical": n_A_neg,
        "min_phi_needed": min_phi_needed,
    }
    Path("data/first-proof/p4-path2-A-quadu.json").write_text(json.dumps(out, indent=2))
    pr("Wrote results")

if __name__ == "__main__":
    main()
