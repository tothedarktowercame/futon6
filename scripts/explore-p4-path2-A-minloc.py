#!/usr/bin/env python3
"""P4 Path2: Where does K_red achieve its minimum for same-sign p,q >= 0?

If the min always occurs at the corner (sqrt(Pmax), sqrt(Qmax)) where K=0,
then we need to prove that this is a global minimum. Check gradient conditions.
"""
import time
import numpy as np
import sympy as sp
from scipy.optimize import minimize

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

    pr("Building K_red...")
    K_red = build_exact_k_red()
    K_fn = sp.lambdify((r, x, y, p, q), K_red, "numpy")
    
    # Gradient
    dKdp = sp.diff(K_red, p)
    dKdq = sp.diff(K_red, q)
    dKdp_fn = sp.lambdify((r, x, y, p, q), dKdp, "numpy")
    dKdq_fn = sp.lambdify((r, x, y, p, q), dKdq, "numpy")

    pr("Searching for min location...")
    rng = np.random.default_rng(42)
    n_test = 100000
    
    # Track where minimum occurs
    n_corner = 0  # at (sqrt(Pmax), sqrt(Qmax))
    n_edge_P = 0  # at P=Pmax edge (not corner)
    n_edge_Q = 0  # at Q=Qmax edge
    n_edge_p0 = 0 # at p=0
    n_edge_q0 = 0 # at q=0
    n_interior = 0
    n_other = 0
    min_K_all = float("inf")
    
    for trial in range(n_test):
        rv = float(np.exp(rng.uniform(np.log(0.01), np.log(100))))
        xv = float(rng.uniform(0.01, 0.99))
        yv = float(rng.uniform(0.01, 0.99))
        
        Pm = float(np.sqrt(2*(1-xv)/9))  # sqrt(Pmax) = max p
        Qm = float(np.sqrt(2*rv**3*(1-yv)/9))  # sqrt(Qmax) = max q
        
        # Minimize K_red over [0, pm] x [0, qm]
        def neg_K(pq):
            return float(K_fn(rv, xv, yv, pq[0], pq[1]))
        
        best_val = float("inf")
        best_loc = None
        
        # Try several starting points
        starts = [
            [Pm*0.5, Qm*0.5],
            [Pm*0.9, Qm*0.9],
            [Pm*0.99, Qm*0.99],
            [Pm, Qm],
            [0.01*Pm, 0.01*Qm],
        ]
        
        for s0 in starts:
            try:
                res = minimize(neg_K, s0, bounds=[(0, Pm), (0, Qm)], method='L-BFGS-B')
                if res.fun < best_val:
                    best_val = res.fun
                    best_loc = res.x
            except:
                pass
        
        if best_val < min_K_all:
            min_K_all = best_val
        
        if best_loc is not None:
            pv, qv = best_loc
            # Classify location
            tol_p = 1e-4 * Pm
            tol_q = 1e-4 * Qm
            
            at_p0 = pv < tol_p
            at_q0 = qv < tol_q
            at_Pm = abs(pv - Pm) < tol_p
            at_Qm = abs(qv - Qm) < tol_q
            
            if at_Pm and at_Qm:
                n_corner += 1
            elif at_Pm and not at_Qm:
                n_edge_P += 1
            elif at_Qm and not at_Pm:
                n_edge_Q += 1
            elif at_p0:
                n_edge_p0 += 1
            elif at_q0:
                n_edge_q0 += 1
            else:
                n_interior += 1
                if trial < 5 or best_val < 1e-6:
                    pr(f"  Interior min: r={rv:.3f}, x={xv:.3f}, y={yv:.3f}, p/Pm={pv/Pm:.4f}, q/Qm={qv/Qm:.4f}, K={best_val:.6e}")
        
        if (trial+1) % 20000 == 0:
            pr(f"  ... {trial+1}/{n_test} done")
    
    pr(f"\nMinimum location classification ({n_test} tests):")
    pr(f"  Corner (Pm,Qm): {n_corner}")
    pr(f"  Edge P=Pmax: {n_edge_P}")  
    pr(f"  Edge Q=Qmax: {n_edge_Q}")
    pr(f"  Edge p=0: {n_edge_p0}")
    pr(f"  Edge q=0: {n_edge_q0}")
    pr(f"  Interior: {n_interior}")
    pr(f"  Other: {n_other}")
    pr(f"  Global min K_red: {min_K_all:.6e}")
    
    # ---------------------------------------------------------------
    # Check gradient at corner (sqrt(Pmax), sqrt(Qmax))
    # ---------------------------------------------------------------
    pr("\n=== Gradient at corner ===")
    # If grad K points INTO the box at the corner, then corner is a local min.
    # Into the box means ∂K/∂p < 0 (decrease p from Pm) and ∂K/∂q < 0
    # i.e. ∂K/∂p ≥ 0 at (Pm,Qm) — because we're at the upper boundary.
    # Wait: for the min to be at the corner, we need the gradient to point
    # outward (into the constraint), meaning ∂K/∂p ≥ 0 and ∂K/∂q ≥ 0.
    
    p_sym, q_sym = sp.symbols("p q")
    Pm_sym = sp.sqrt(sp.Rational(2,9)*(1-x))
    Qm_sym = sp.sqrt(sp.Rational(2,9)*r**3*(1-y))
    
    # Actually, let me compute ∂K/∂p and ∂K/∂q at (Pm, Qm) symbolically
    pr("Computing symbolic gradient at corner...")
    dKdp_corner = sp.expand(dKdp.subs({p: Pm_sym, q: Qm_sym}))
    dKdq_corner = sp.expand(dKdq.subs({p: Pm_sym, q: Qm_sym}))
    
    # These involve sqrt, simplify
    dKdp_corner_sq = sp.expand(dKdp_corner**2)  # check sign by evaluating
    
    # Numeric gradient check
    pr("Numeric gradient at corner...")
    n_dp_pos = 0
    n_dq_pos = 0
    rng2 = np.random.default_rng(123)
    for _ in range(50000):
        rv = float(np.exp(rng2.uniform(np.log(0.01), np.log(100))))
        xv = float(rng2.uniform(0.01, 0.99))
        yv = float(rng2.uniform(0.01, 0.99))
        
        Pm = float(np.sqrt(2*(1-xv)/9))
        Qm = float(np.sqrt(2*rv**3*(1-yv)/9))
        
        dp = float(dKdp_fn(rv, xv, yv, Pm, Qm))
        dq = float(dKdq_fn(rv, xv, yv, Pm, Qm))
        
        if dp >= -1e-10:
            n_dp_pos += 1
        if dq >= -1e-10:
            n_dq_pos += 1
    
    pr(f"∂K/∂p >= 0 at corner: {n_dp_pos}/50000")
    pr(f"∂K/∂q >= 0 at corner: {n_dq_pos}/50000")

    # ---------------------------------------------------------------
    # Hessian at corner
    # ---------------------------------------------------------------
    pr("\n=== Hessian at corner ===")
    d2Kpp = sp.diff(K_red, p, p)
    d2Kpq = sp.diff(K_red, p, q)
    d2Kqq = sp.diff(K_red, q, q)
    
    d2Kpp_fn = sp.lambdify((r, x, y, p, q), d2Kpp, "numpy")
    d2Kpq_fn = sp.lambdify((r, x, y, p, q), d2Kpq, "numpy")
    d2Kqq_fn = sp.lambdify((r, x, y, p, q), d2Kqq, "numpy")
    
    n_psd = 0
    n_not_psd = 0
    rng3 = np.random.default_rng(456)
    for _ in range(50000):
        rv = float(np.exp(rng3.uniform(np.log(0.01), np.log(100))))
        xv = float(rng3.uniform(0.01, 0.99))
        yv = float(rng3.uniform(0.01, 0.99))
        
        Pm = float(np.sqrt(2*(1-xv)/9))
        Qm = float(np.sqrt(2*rv**3*(1-yv)/9))
        
        hpp = float(d2Kpp_fn(rv, xv, yv, Pm, Qm))
        hpq = float(d2Kpq_fn(rv, xv, yv, Pm, Qm))
        hqq = float(d2Kqq_fn(rv, xv, yv, Pm, Qm))
        
        # PSD iff hpp >= 0 and hpp*hqq >= hpq²
        if hpp >= -1e-10 and hpp*hqq - hpq**2 >= -1e-10*(hpp*hqq + hpq**2 + 1):
            n_psd += 1
        else:
            n_not_psd += 1
    
    pr(f"Hessian PSD at corner: {n_psd}/50000")
    pr(f"Hessian not PSD: {n_not_psd}/50000")

    elapsed = time.time() - t0
    pr(f"\nRuntime: {elapsed:.1f}s")

if __name__ == "__main__":
    main()
