#!/usr/bin/env python3
"""P4 Path2: Alternative proof strategies for A >= 0.

Strategy 1: Weighted grouping.
  Ā = ā₁₀u + ā₀₁v + ā₂₀u² + ā₁₁uv + ā₀₂v² + 1296rL(u²v + uv²)

Split linear terms: use fraction α of ā₁₀u for AM-GM with uv, rest for positivity.
  α·ā₁₀·u + |ā₁₁|·uv ≥ 0 when αā₁₀ ≥ |ā₁₁|v = |ā₁₁|·Qmax (worst case v=Qmax)
  So need: α ≥ |ā₁₁|·Qmax/ā₁₀

Similarly from ā₀₁ side: β ≥ |ā₁₁|·Pmax/ā₀₁

Strategy 2: Complete the square in the quadratic form.
  ā₂₀u² + ā₁₁uv + ā₀₂v² = ā₂₀(u + ā₁₁v/(2ā₂₀))² + (ā₀₂ - ā₁₁²/(4ā₂₀))v²
  The first term ≥ 0, the second has negative coefficient (disc < 0).
  Deficit = (ā₁₁² - 4ā₂₀ā₀₂)/(4ā₂₀) · v²
  Need linear term ā₀₁v ≥ deficit v² = deficit·v·v
  i.e. ā₀₁ ≥ deficit·v for all v ∈ [0,Qmax]
  i.e. ā₀₁ ≥ deficit·Qmax

Strategy 3: Direct quadratic-in-u approach.
  For fixed v, Ā is quadratic in u:
  Ā = (ā₂₀ + 1296rLv)u² + (ā₁₀ + ā₁₁v + 1296rLv²)u + (ā₀₁v + ā₀₂v²)
  If ā₂₀ + 1296rLv ≥ 0 (true when L≥0), this is convex in u.
  Minimum at u* = -(ā₁₀ + ā₁₁v + 1296rLv²)/(2(ā₂₀ + 1296rLv))
  But u* must be ≥ 0 (feasible). Since ā₁₀ ≥ 0 and ā₁₁ < 0:
    Numerator = -(ā₁₀ + ā₁₁v + 1296rLv²) ≥ 0 when v is small enough
  At u=0: Ā = ā₀₁v + ā₀₂v² ≥ 0 ✓ (both terms ≥ 0)
  So we need: Ā(u*,v) ≥ 0 when u* ∈ [0, Pmax].

Strategy 4: Global minimum analysis.
  Just find min_{u,v ∈ box} Ā(u,v) numerically with high precision.
"""
import time
import numpy as np
import sympy as sp

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

    # Taylor coefficients (from Taylor expansion)
    a10_bar = sp.expand(-(a10 + 2*a20*Pmax + a11*Qmax + 2*a12*Pmax*Qmax + a12*Qmax**2))
    a01_bar = sp.expand(-(a01 + a11*Pmax + 2*a02*Qmax + a12*Pmax**2 + 2*a12*Pmax*Qmax))
    a20_bar = sp.expand(a20 + a12*Qmax)
    a02_bar = sp.expand(a02 + a12*Pmax)
    a11_bar = sp.expand(a11 + 2*a12*(Pmax + Qmax))
    L = sp.expand(9*x**2 - 27*x*y*(1+r) + 3*x*(r-1) + 9*r*y**2 - 3*r*y + 2*r + 3*y + 2)

    # ---------------------------------------------------------------
    # Strategy 2: Complete the square + linear compensation
    # ---------------------------------------------------------------
    pr("\n=== Strategy 2: Complete square + linear ===")
    # Deficit coefficient: (ā₁₁² - 4ā₂₀ā₀₂)/(4ā₂₀) (this is > 0 since disc < 0)
    # Need: ā₀₁ ≥ deficit·Qmax, i.e., ā₀₁·4ā₂₀ ≥ (ā₁₁²-4ā₂₀ā₀₂)·Qmax
    # i.e., 4ā₂₀ā₀₁ + 4ā₂₀ā₀₂·Qmax ≥ ā₁₁²·Qmax

    check2 = sp.expand(4*a20_bar*a01_bar + 4*a20_bar*a02_bar*Qmax - a11_bar**2*Qmax)

    check2_fn = sp.lambdify((r, x, y), check2, "numpy")
    rng = np.random.default_rng(42)
    n_test = 300000
    n_fail2 = 0
    min_r2 = float("inf")

    for _ in range(n_test):
        rv = float(np.exp(rng.uniform(np.log(1e-4), np.log(1e4))))
        xv = float(rng.uniform(1e-8, 1 - 1e-8))
        yv = float(rng.uniform(1e-8, 1 - 1e-8))
        val = float(check2_fn(rv, xv, yv))
        if val < -1e-6:
            n_fail2 += 1
        if val < min_r2:
            min_r2 = val

    pr(f"Strategy 2 fails: {n_fail2}/{n_test}")
    pr(f"min value: {min_r2:.6e}")

    # ---------------------------------------------------------------
    # Strategy 3: Quadratic-in-u, check discriminant
    # ---------------------------------------------------------------
    pr("\n=== Strategy 3: Quadratic in u, check discriminant ===")
    # Ā = (ā₂₀ + 1296rLv)u² + (ā₁₀ + ā₁₁v + 1296rLv²)u + (ā₀₁v + ā₀₂v²)
    # For Ā ≥ 0 for all u ≥ 0:
    # If leading coeff > 0 (convex) and vertex u* < 0: automatic.
    # If u* ≥ 0: need discriminant ≤ 0 or Ā(u*) ≥ 0.
    # u* = -(ā₁₀ + ā₁₁v + 1296rLv²) / (2(ā₂₀ + 1296rLv))
    # Note: ā₁₀ ≥ 0, ā₁₁ < 0. So numerator = -(positive + negative·v + mixed·v²)
    # u* ≥ 0 iff ā₁₀ + ā₁₁v + 1296rLv² ≤ 0, i.e. |ā₁₁|v > ā₁₀ + 1296rLv²
    
    # Discriminant of the quadratic in u:
    # Δ_u(v) = (ā₁₀ + ā₁₁v + 1296rLv²)² - 4(ā₂₀ + 1296rLv)(ā₀₁v + ā₀₂v²)
    # If Δ_u(v) ≤ 0 for all v ∈ [0,Qmax], then Ā ≥ 0. 
    
    # But Δ_u(0) = ā₁₀² ≥ 0, so the discriminant is nonneg at v=0.
    # That means u*=0 at v=0 with value 0 (degenerate).
    
    # Actually at v=0: Ā = ā₂₀u² + ā₁₀u ≥ 0 since ā₂₀,ā₁₀ ≥ 0. ✓
    
    # The issue is for v > 0 where ā₁₁v makes the linear-in-u term negative.
    # Minimum of Ā over u ≥ 0, for fixed v:
    # If u* ≤ 0: min at u=0, Ā = ā₀₁v + ā₀₂v² ≥ 0 ✓
    # If u* > 0: Ā(u*) = ā₀₁v + ā₀₂v² - (ā₁₀+ā₁₁v+1296rLv²)²/(4(ā₂₀+1296rLv))
    # Also need u* ≤ Pmax.
    
    # Let's just check numerically: min over the box
    pr("Numerical min-over-box via grid search...")
    
    a10b_fn = sp.lambdify((r, x, y), a10_bar, "numpy")
    a01b_fn = sp.lambdify((r, x, y), a01_bar, "numpy")
    a20b_fn = sp.lambdify((r, x, y), a20_bar, "numpy")
    a02b_fn = sp.lambdify((r, x, y), a02_bar, "numpy")
    a11b_fn = sp.lambdify((r, x, y), a11_bar, "numpy")
    L_fn = sp.lambdify((r, x, y), L, "numpy")
    
    rng3 = np.random.default_rng(999)
    n_param = 50000
    n_grid = 20  # grid points per dimension for u,v
    min_Abar = float("inf")
    min_Abar_params = None
    n_neg = 0
    
    for _ in range(n_param):
        rv = float(np.exp(rng3.uniform(np.log(1e-4), np.log(1e4))))
        xv = float(rng3.uniform(1e-8, 1 - 1e-8))
        yv = float(rng3.uniform(1e-8, 1 - 1e-8))
        
        Pm = 2*(1-xv)/9
        Qm = 2*rv**3*(1-yv)/9
        
        c10 = float(a10b_fn(rv, xv, yv))
        c01 = float(a01b_fn(rv, xv, yv))
        c20 = float(a20b_fn(rv, xv, yv))
        c02 = float(a02b_fn(rv, xv, yv))
        c11 = float(a11b_fn(rv, xv, yv))
        Lv = float(L_fn(rv, xv, yv))
        cubic = 1296*rv*Lv
        
        us = np.linspace(0, Pm, n_grid)
        vs = np.linspace(0, Qm, n_grid)
        
        for ui in us:
            for vi in vs:
                val = c10*ui + c01*vi + c20*ui**2 + c11*ui*vi + c02*vi**2 + cubic*(ui**2*vi + ui*vi**2)
                if val < min_Abar:
                    min_Abar = val
                    min_Abar_params = (rv, xv, yv, ui, vi)
                if val < -1e-10:
                    n_neg += 1
    
    pr(f"Min Ā over box: {min_Abar:.6e}")
    pr(f"Negative count: {n_neg}/{n_param * n_grid**2}")
    if min_Abar_params:
        rv, xv, yv, ui, vi = min_Abar_params
        pr(f"  at r={rv:.4f}, x={xv:.4f}, y={yv:.4f}, u={ui:.4e}, v={vi:.4e}")
        Pm = 2*(1-xv)/9
        Qm = 2*rv**3*(1-yv)/9
        pr(f"  Pmax={Pm:.4e}, Qmax={Qm:.4e}, u/Pmax={ui/Pm:.4f}, v/Qmax={vi/Qm:.4f}")

    # ---------------------------------------------------------------
    # Strategy 5: Look for the structure min_Abar / (Pmax*Qmax) ratio
    # This tells us how tight the inequality is
    # ---------------------------------------------------------------
    pr("\n=== Ratio analysis: min Ā / (a20_bar*Pmax² + ...) ===")
    # Check ratio Ā_min / max_term at the minimizer
    if min_Abar_params:
        rv, xv, yv, ui, vi = min_Abar_params
        c10 = float(a10b_fn(rv, xv, yv))
        c01 = float(a01b_fn(rv, xv, yv))
        c20 = float(a20b_fn(rv, xv, yv))
        c02 = float(a02b_fn(rv, xv, yv))
        c11 = float(a11b_fn(rv, xv, yv))
        Lv = float(L_fn(rv, xv, yv))
        cubic = 1296*rv*Lv
        
        terms = {
            "ā10·u": c10*ui,
            "ā01·v": c01*vi,
            "ā20·u²": c20*ui**2,
            "ā11·uv": c11*ui*vi,
            "ā02·v²": c02*vi**2,
            "cubic·u²v": cubic*ui**2*vi,
            "cubic·uv²": cubic*ui*vi**2,
        }
        pr("Terms at minimum:")
        for name, val in terms.items():
            pr(f"  {name} = {val:.6e}")
        total = sum(terms.values())
        pr(f"  total = {total:.6e}")

    # ---------------------------------------------------------------
    # Strategy 6: Maybe A ≥ 0 follows from a different decomposition
    # Try: A = (boundary terms) + cross correction
    # A(P,Q) = A(P,0) + A(0,Q) - A(0,0) + cross(P,Q) with A(P,0)≥0, A(0,Q)≥0
    # ---------------------------------------------------------------
    pr("\n=== Strategy 6: Boundary decomposition A = A(P,0)+A(0,Q)-a00+cross ===")
    P_sym, Q_sym = sp.symbols("P Q")
    A_PQ = sp.expand(a00 + a10*P_sym + a01*Q_sym + a20*P_sym**2 + a11*P_sym*Q_sym
                     + a02*Q_sym**2 + a12*P_sym**2*Q_sym + a12*P_sym*Q_sym**2)
    A_P0 = sp.expand(a00 + a10*P_sym + a20*P_sym**2)
    A_0Q = sp.expand(a00 + a01*Q_sym + a02*Q_sym**2)
    cross = sp.expand(A_PQ - A_P0 - A_0Q + a00)
    cross_simplified = sp.factor(cross)
    pr(f"Cross term: {cross_simplified}")
    # cross = a11*P*Q + a12*P²Q + a12*PQ² = PQ(a11 + a12*P + a12*Q)
    # = PQ·(a11 + a12(P+Q))
    
    # At (Pmax, Qmax): cross = Pmax·Qmax·(a11 + a12*(Pmax+Qmax))
    cross_corner = sp.expand(cross.subs({P_sym: Pmax, Q_sym: Qmax}))
    cross_corner_f = sp.factor(cross_corner)
    pr(f"Cross at corner: {cross_corner_f}")
    
    # a11 + a12*(Pmax+Qmax) = a11_bar - 2a12(Pmax+Qmax) + a12(Pmax+Qmax)
    # Wait, let me just compute directly
    inner = sp.expand(a11 + a12*(Pmax + Qmax))
    inner_f = sp.factor(inner)
    pr(f"a11 + a12(Pmax+Qmax) = {inner_f}")
    
    # So A = A(P,0) + A(0,Q) - a00 + PQ(a11 + a12P + a12Q)
    # We know A(P,0) ≥ 0 and A(0,Q) ≥ 0.
    # a00 ≥ 0 (it's the value at origin).
    # The cross term PQ(a11 + a12(P+Q)) can be negative since a11 < 0.
    # Need: A(P,0) + A(0,Q) ≥ a00 + |cross|
    # This is just rephrasing the problem...
    
    # ---------------------------------------------------------------
    # Strategy 7: Use the factorization at boundaries
    # A(Pmax, Q) and A(P, Qmax) are both proven ≥ 0 (quadratics with pos coeffs)
    # A is bilinear in a sense... multilinear?
    # ---------------------------------------------------------------
    pr("\n=== Strategy 7: Bilinear interpolation bound ===")
    # A(P,Q) = A(Pmax,Q) + (Pmax-P)·[-∂A/∂P|_{Pmax}] 
    #          + (Pmax-P)²·[A_PP/2] + higher
    # This is essentially the Taylor expansion we already did.
    # A different interpolation: write P = λ·Pmax, Q = μ·Qmax with λ,μ ∈ [0,1]
    # A(λPmax, μQmax) as a function of λ,μ.
    
    # Actually, let me try a KEY observation:
    # The cross term = PQ·(a11 + a12P + a12Q)
    # Factor a11 + a12P + a12Q. Since a12 = -1296rL, and a11 is complicated.
    # At (0,0): just a11 (negative)
    # At (Pmax, Qmax): a11 + a12(Pmax+Qmax) = a11_bar/something? No.
    # Let's compute a11 + a12*(Pmax+Qmax) directly
    sum_PQ = sp.expand(Pmax + Qmax)
    val_at_corner = sp.expand(a11 + a12*sum_PQ)
    val_f = sp.factor(val_at_corner)
    pr(f"a11 + a12*(Pmax+Qmax) factored: {val_f}")
    
    # Check if a11 + a12*P + a12*Q changes sign on the box [0,Pmax]×[0,Qmax]
    # It's linear in P,Q, so extremes at corners:
    # (0,0): a11 (neg)
    # (Pmax,0): a11+a12*Pmax
    # (0,Qmax): a11+a12*Qmax
    # (Pmax,Qmax): a11+a12*(Pmax+Qmax)
    
    c1_fn = sp.lambdify((r, x, y), sp.expand(a11 + a12*Pmax), "numpy")
    c2_fn = sp.lambdify((r, x, y), sp.expand(a11 + a12*Qmax), "numpy")
    c3_fn = sp.lambdify((r, x, y), val_at_corner, "numpy")
    a11_fn = sp.lambdify((r, x, y), a11, "numpy")
    
    rng4 = np.random.default_rng(42)
    signs = {"a11": [0,0], "a11+a12Pm": [0,0], "a11+a12Qm": [0,0], "a11+a12(Pm+Qm)": [0,0]}
    
    for _ in range(100000):
        rv = float(np.exp(rng4.uniform(np.log(1e-4), np.log(1e4))))
        xv = float(rng4.uniform(1e-8, 1-1e-8))
        yv = float(rng4.uniform(1e-8, 1-1e-8))
        
        v0 = float(a11_fn(rv, xv, yv))
        v1 = float(c1_fn(rv, xv, yv))
        v2 = float(c2_fn(rv, xv, yv))
        v3 = float(c3_fn(rv, xv, yv))
        
        for name, val in [("a11", v0), ("a11+a12Pm", v1), ("a11+a12Qm", v2), ("a11+a12(Pm+Qm)", v3)]:
            if val >= 0:
                signs[name][0] += 1
            else:
                signs[name][1] += 1
    
    for name, (pos, neg) in signs.items():
        pr(f"  {name}: pos={pos}, neg={neg}")

    elapsed = time.time() - t0
    pr(f"\nRuntime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
