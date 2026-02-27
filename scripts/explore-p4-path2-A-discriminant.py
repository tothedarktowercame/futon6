#!/usr/bin/env python3
"""P4 Path2: Check if the quadratic form in the Taylor expansion has non-negative discriminant.

If 4*ā20*ā02 >= ā11^2, then the quadratic form ā20*u^2 + ā11*uv + ā02*v^2 >= 0,
and combined with ā10*u >= 0, ā01*v >= 0, and the cubic terms, we get A >= 0.
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
    u_var, v_var = sp.symbols("u v", nonnegative=True)
    P_sym, Q_sym = sp.symbols("P Q")

    pr("Building K_red & decomposing A...")
    K_red = build_exact_k_red()
    c = decompose_coeffs(K_red)

    a00=c["a00"]; a01=c["a01"]; a02=c["a02"]; a10=c["a10"]
    a11=c["a11"]; a12=c["a12"]; a20=c["a20"]

    A_poly = sp.expand(a00 + a10*P_sym + a01*Q_sym + a20*P_sym**2 + a11*P_sym*Q_sym
                       + a02*Q_sym**2 + a12*P_sym**2*Q_sym + a12*P_sym*Q_sym**2)

    Pmax = sp.Rational(2,9)*(1-x)
    Qmax = sp.Rational(2,9)*r**3*(1-y)

    # Compute expansion coefficients directly
    pr("Computing Taylor coefficients...")
    a20_bar = sp.expand(a20 + a12*Qmax)
    a02_bar = sp.expand(a02 + a12*Pmax)

    # ā11 = a11 + 2*a12*(Pmax + Qmax) -- this is the cross-derivative
    a11_bar = sp.expand(a11 + 2*a12*(Pmax + Qmax))

    # Discriminant: 4*ā20*ā02 - ā11²
    pr("Computing discriminant...")
    disc = sp.expand(4*a20_bar*a02_bar - a11_bar**2)

    # Try to factor
    pr("Factoring discriminant...")
    disc_factor = sp.factor(disc)
    pr("disc = 4*ā20*ā02 - ā11² factored:", disc_factor)

    # Numeric check
    pr("\n=== Numeric discriminant check ===")
    disc_fn = sp.lambdify((r, x, y), disc, "numpy")
    a11_bar_fn = sp.lambdify((r, x, y), a11_bar, "numpy")
    a20_bar_fn = sp.lambdify((r, x, y), a20_bar, "numpy")
    a02_bar_fn = sp.lambdify((r, x, y), a02_bar, "numpy")

    rng = np.random.default_rng(42)
    n_test = 300000
    n_disc_neg = 0
    n_disc_pos = 0
    min_disc = float("inf")
    max_disc = float("-inf")

    for _ in range(n_test):
        rv = float(np.exp(rng.uniform(np.log(1e-4), np.log(1e4))))
        xv = float(rng.uniform(1e-8, 1 - 1e-8))
        yv = float(rng.uniform(1e-8, 1 - 1e-8))

        dv = float(disc_fn(rv, xv, yv))
        if dv < -1e-10:
            n_disc_neg += 1
        if dv > 1e-10:
            n_disc_pos += 1
        if dv < min_disc:
            min_disc = dv
        if dv > max_disc:
            max_disc = dv

    pr(f"Discriminant tests: {n_test}")
    pr(f"disc ≥ 0: {n_disc_pos}")
    pr(f"disc < 0: {n_disc_neg}")
    pr(f"min disc: {min_disc:.6e}")
    pr(f"max disc: {max_disc:.6e}")

    # If disc < 0 sometimes, check if the linear terms compensate
    # For the quadratic form to give Ā >= 0 when disc < 0:
    # Need: ā10*u + ā01*v >= max(0, |ā11|*uv - ā20*u² - ā02*v²)
    # The worst case is when the quadratic form is most negative.
    # Minimum of quadratic form: -(ā11² - 4ā20ā02)/(4ā20) * v² (completing the square)
    # For fixed v, min over u: at u = -ā11*v/(2ā20), value = -(ā11²-4ā20ā02)/(4ā20)*v²

    # If disc < 0: the quadratic form can go as negative as:
    # -|disc|/(4ā20) * v²
    # And we need ā10*u + ā01*v >= |disc|/(4ā20) * v²
    # At the optimal u for the quadratic: u = -ā11*v/(2ā20) (which should be ≥ 0 since ā11 < 0, ā20 > 0)
    # Then ā10*u = ā10*|ā11|*v/(2ā20), and ā10*u + ā01*v = (ā10|ā11|/(2ā20) + ā01)*v
    # Need: (ā10|ā11|/(2ā20) + ā01)*v >= |disc|/(4ā20)*v²
    # i.e.: v <= (ā10|ā11|/(2ā20) + ā01)/(|disc|/(4ā20)) = (2ā10|ā11| + 4ā01ā20)/|disc|
    # But v can be up to Qmax. So we'd need this upper bound >= Qmax.

    # Also check if the cubic terms help when disc < 0:
    # ā21*u²v + ā12*uv² (= 1296rL each)

    elapsed = time.time() - t0
    pr(f"\nRuntime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
