#!/usr/bin/env python3
"""Find interior critical points of K_red for same-sign p,q > 0.
If K_red has no interior minimum with negative value, we're done."""

import numpy as np
from scipy.optimize import minimize
import time

def build_kred_numeric():
    """Build K_red as a numerical function."""
    import sympy as sp
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
    K_red = sp.expand(sp.cancel(K_exact / r**2))
    return sp.lambdify((r, x, y, p, q), K_red)

print("Building K_red...", flush=True)
K_func = build_kred_numeric()
print("K_red built successfully.", flush=True)

t0 = time.time()
rng = np.random.default_rng(123)

# For each (r,x,y), minimize K_red over (p,q) in [0,sqrt(Pmax)] x [0,sqrt(Qmax)]
N = 50000
r_vals = np.exp(rng.uniform(-6, 6, N))
x_vals = rng.uniform(0.001, 0.999, N)
y_vals = rng.uniform(0.001, 0.999, N)

interior_mins = 0
boundary_mins = 0
min_K = 1e10
worst = None
skipped = 0

for i in range(N):
    rv, xv, yv = r_vals[i], x_vals[i], y_vals[i]
    Pm = np.sqrt(2*(1-xv)/9)
    Qm = np.sqrt(2*rv**3*(1-yv)/9)
    
    if Pm < 1e-10 or Qm < 1e-10:
        skipped += 1
        continue
    
    # Try multiple starting points
    best_val = 1e10
    best_pq = None
    for p0, q0 in [(Pm/2, Qm/2), (Pm/3, Qm/3), (Pm*0.8, Qm*0.8),
                    (Pm*0.1, Qm*0.9), (Pm*0.9, Qm*0.1)]:
        try:
            res = minimize(lambda z: float(K_func(rv, xv, yv, z[0], z[1])),
                           [p0, q0], bounds=[(0, Pm), (0, Qm)],
                           method='L-BFGS-B')
            if res.fun < best_val:
                best_val = res.fun
                best_pq = res.x.copy()
        except Exception:
            pass
    
    if best_pq is None:
        skipped += 1
        continue
    
    if best_val < min_K:
        min_K = best_val
        worst = (rv, xv, yv, best_pq[0], best_pq[1])
    
    # Check if minimizer is on boundary
    eps = 1e-8
    on_boundary = (best_pq[0] < eps or best_pq[1] < eps or 
                   abs(best_pq[0] - Pm) < eps*Pm or 
                   abs(best_pq[1] - Qm) < eps*Qm)
    if on_boundary:
        boundary_mins += 1
    else:
        interior_mins += 1
        if best_val < 1e-6:
            print(f"  Interior min at ({rv:.4f}, {xv:.4f}, {yv:.4f}): "
                  f"p={best_pq[0]:.6f}, q={best_pq[1]:.6f}, K={best_val:.2e}", flush=True)
    
    if (i+1) % 10000 == 0:
        print(f"  ... {i+1}/{N} done, interior_mins={interior_mins}, "
              f"min_K={min_K:.2e}", flush=True)

print(f"\nResults:", flush=True)
print(f"  Total: {N}", flush=True)
print(f"  Skipped: {skipped}", flush=True)
print(f"  Interior mins: {interior_mins}", flush=True)
print(f"  Boundary mins: {boundary_mins}", flush=True)
print(f"  Min K_red: {min_K:.6e}", flush=True)
if worst:
    print(f"  Worst case: r={worst[0]:.6f}, x={worst[1]:.6f}, y={worst[2]:.6f}, "
          f"p={worst[3]:.6f}, q={worst[4]:.6f}", flush=True)
print(f"  Runtime: {time.time()-t0:.1f}s", flush=True)
