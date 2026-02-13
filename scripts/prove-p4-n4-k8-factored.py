#!/usr/bin/env python3
"""Factor K8 coefficients and verify the SOS decomposition.

K8 = p²q² · (c0*q⁴ + c1*p*q³ + c2*p²*q² + c3*p³*q + c4*p⁴)

Known: c0 = 59049(1+3x)(1+4r+3x)
       c4 = 59049r³(1+3y)(4+r+3ry)
       H = c0 + c1*w + c2*w² + c3*w³ + c4*w⁴ >= 0 for all real w

Factor c1, c2, c3 and find the SOS certificate.
"""
import sympy as sp
import numpy as np

r, x, y = sp.symbols('r x y', positive=True)

# From the output
c0_raw = 708588*r*x + 236196*r + 531441*x**2 + 354294*x + 59049
c1_raw = (-1062882*r**2*x*y - 354294*r**2*x - 354294*r**2*y - 118098*r**2
           + 1417176*r*x + 472392*r + 1062882*x**2 + 708588*x + 118098)
c2_raw = (531441*r**4*y**2 + 354294*r**4*y + 59049*r**4 + 708588*r**3*y + 236196*r**3
          - 2125764*r**2*x*y - 708588*r**2*x - 708588*r**2*y - 236196*r**2
          + 708588*r*x + 236196*r + 531441*x**2 + 354294*x + 59049)
c3_raw = (1062882*r**4*y**2 + 708588*r**4*y + 118098*r**4 + 1417176*r**3*y + 472392*r**3
          - 1062882*r**2*x*y - 354294*r**2*x - 354294*r**2*y - 118098*r**2)
c4_raw = 531441*r**4*y**2 + 354294*r**4*y + 59049*r**4 + 708588*r**3*y + 236196*r**3

print("="*60)
print("FACTORING K8 QUARTIC COEFFICIENTS")
print("="*60)

# Factor each
for name, expr in [("c0", c0_raw), ("c1", c1_raw), ("c2", c2_raw),
                   ("c3", c3_raw), ("c4", c4_raw)]:
    f = sp.factor(expr)
    print(f"\n{name} = {expr}")
    print(f"   = {f}")

# Verify c0, c4 manually
print("\n\nVerifying c0 = 59049(1+3x)(1+4r+3x):")
print(sp.expand(59049*(1+3*x)*(1+4*r+3*x)) - c0_raw == 0)

print("Verifying c4 = 59049r³(1+3y)(4+r+3ry):")
print(sp.expand(59049*r**3*(1+3*y)*(4+r+3*r*y)) - c4_raw == 0)

# Set A = (1+3x), B = (1+3y)*r^2
# Check: c1 = 118098*(1+3x)*[(1+3x) + 4r - (1+3y)*r²]
A = 1 + 3*x
B = (1 + 3*y)*r**2
L = A + 4*r - B  # = (1+3x) + 4r - (1+3y)r²

print("\n\nFactoring relative to A=(1+3x), B=(1+3y)r²:")
print(f"L = A + 4r - B = {sp.expand(L)}")

c1_test = 118098*A*L
print(f"\n118098*A*L = {sp.expand(c1_test)}")
print(f"c1 matches? {sp.expand(c1_test - c1_raw) == 0}")

c3_test = 118098*B*(-L)  # Note: c3 has the opposite L sign
print(f"\n-118098*B*L = {sp.expand(-118098*B*L)}")
print(f"c3 = {sp.expand(c3_raw)}")
c3_alt = -118098*B*L
print(f"c3 matches -118098*B*L? {sp.expand(c3_alt - c3_raw) == 0}")

# Now c2
print(f"\n\nc2 = {sp.expand(c2_raw)}")
c2_factored = sp.factor(c2_raw)
print(f"c2 factored = {c2_factored}")

# Try c2 in terms of A, B:
# c2 should relate to c0, c1, c3, c4 via the quartic structure
# For a non-neg quartic c0+c1w+c2w²+c3w³+c4w⁴, the key is
# 4c0c4 - c1c3 (related to the discriminant of the quadratic factor)
print("\n\nKey invariants:")
p_c0c4 = sp.expand(4*c0_raw*c4_raw)
p_c1c3 = sp.expand(c1_raw*c3_raw)
diff = sp.expand(p_c0c4 - p_c1c3)
print(f"4c0c4 - c1c3 = {sp.factor(diff)}")

# Also check: c2² - 3c1c3 (another discriminant-related quantity)
d2 = sp.expand(c2_raw**2 - 3*c1_raw*c3_raw)
print(f"\nc2² - 3c1c3 = {sp.factor(d2)}")

# Check if c0*c4 is a perfect square-ish
print(f"\nc0*c4 = {sp.factor(sp.expand(c0_raw*c4_raw))}")

# The SOS structure: H(w) = c4(w-α)(w-β)(w-γ)(w-δ) with all roots complex
# Since H>=0 for all w and c4>0, all roots are complex: two conjugate pairs
# H = c4[(w-a)²+b²][(w-c)²+d²]
# The question is whether we can find a nice symbolic form

# Try: can we write H as a sum of two squares times c4?
# H = c4(w²+pw+q)² + c4·λ²(w+μ)² (doesn't work exactly but conceptually)

# Alternative: Gram matrix approach
# H = [1,w,w²] G [1,w,w²]^T where G is 3x3 PSD
# G[0,0]=c0, G[0,1]=c1/2, G[0,2]=(c2-G[1,1])/2, G[1,2]=c3/2, G[2,2]=c4
# Free param: g = G[1,1]

# For PSD: need all principal minors ≥ 0
# M1 = c0 ≥ 0 ✓
# M2 = c0*g - c1²/4 ≥ 0
# M3 = det(G) ≥ 0

# With the factored forms:
# c0 = 59049*A*(A+4r), c4 = 59049*r³*(1+3y)*(4+r+3ry)
# c1 = 118098*A*L, c3 = -118098*B*L = 118098*B*(-L)

# c1² = 118098²*A²*L²
# c1²/(4c0) = 118098²*A²*L²/(4*59049*A*(A+4r)) = 118098²*A*L²/(4*59049*(A+4r))
# = 4*59049*A*L²/(A+4r)  [since 118098 = 2*59049]

print("\n\nc1²/(4c0):")
g_min1 = sp.expand(c1_raw**2 / (4*c0_raw))
g_min1_simple = 4*59049*A*L**2/(A+4*r)
print(f"  = {sp.simplify(g_min1)}")
print(f"  = {sp.simplify(g_min1_simple)}")
print(f"  match? {sp.simplify(g_min1 - g_min1_simple) == 0}")

print("\nc3²/(4c4):")
g_min2 = sp.expand(c3_raw**2 / (4*c4_raw))
g_min2_simple = 4*59049*r*B*L**2/(r*B/r + 4)  # B = (1+3y)r², so r*B/r = (1+3y)r
# Actually: c3² = 118098²*B²*L², 4c4 = 4*59049*r³*(1+3y)*(4+r+3ry)
# c3²/(4c4) = 118098²*B²*L²/(4*59049*r³*(1+3y)*(4+r+3ry))
# B = (1+3y)r², B² = (1+3y)²*r⁴
# = 4*59049*(1+3y)²*r⁴*L²/(r³*(1+3y)*(4+r(1+3y)))
# = 4*59049*r*(1+3y)*L²/(4+r(1+3y))
g_min2_simple = 4*59049*r*(1+3*y)*L**2/(4+r*(1+3*y))
print(f"  = {sp.simplify(g_min2_simple)}")

# Try g = c1²/(4c0) + c3²/(4c4) (sum of the two lower bounds)
g_sum = sp.simplify(g_min1_simple + g_min2_simple)
print(f"\ng = c1²/(4c0) + c3²/(4c4) = {g_sum}")

# At r=1, x=y=1/3:
vals = {r: 1, x: sp.Rational(1,3), y: sp.Rational(1,3)}
print(f"\nAt equality point (r=1, x=y=1/3):")
print(f"  c0 = {c0_raw.subs(vals)}")
print(f"  c1 = {c1_raw.subs(vals)}")
print(f"  c2 = {c2_raw.subs(vals)}")
print(f"  c3 = {c3_raw.subs(vals)}")
print(f"  c4 = {c4_raw.subs(vals)}")
print(f"  L = {L.subs(vals)}")
print(f"  A = {A.subs(vals)}, B = {B.subs(vals)}")

# Check H(w) at equality: all c's
c_eq = [c0_raw.subs(vals), c1_raw.subs(vals), c2_raw.subs(vals),
        c3_raw.subs(vals), c4_raw.subs(vals)]
print(f"\n  H(w) = {c_eq[4]}w⁴ + {c_eq[3]}w³ + {c_eq[2]}w² + {c_eq[1]}w + {c_eq[0]}")
# At equality with r=1,x=y: H should be a perfect square-ish
# c1=c3 and c0=c4 by symmetry?
print(f"  c0==c4? {c_eq[0] == c_eq[4]}")
print(f"  c1==c3? {c_eq[1] == c_eq[3]}")
if c_eq[0] == c_eq[4] and c_eq[1] == c_eq[3]:
    print(f"  H is palindromic at equality! H(w) = w⁴H(1/w)")
    print(f"  Setting u = w + 1/w: H = w²(c4u² + c3u + (c2-2c4))")
    u_coeffs = [c_eq[4], c_eq[3], c_eq[2] - 2*c_eq[4]]
    print(f"  Quadratic in u: {u_coeffs[0]}u² + {u_coeffs[1]}u + {u_coeffs[2]}")
    disc_u = u_coeffs[1]**2 - 4*u_coeffs[0]*u_coeffs[2]
    print(f"  Discriminant: {disc_u}")
    if disc_u < 0:
        print(f"  Disc < 0: quadratic always > 0 → H always > 0")

# Check symmetry more generally: does c0(r,x,y) = c4(1/r,y,x) / r^something?
print(f"\n  Symmetry check: c0(r,x,y) vs c4(1/r,y,x):")
c4_swapped = c4_raw.subs({r: 1/r, x: y, y: x})
print(f"  c4(1/r,y,x) = {sp.expand(c4_swapped)}")
print(f"  c0/c4_swapped = {sp.simplify(c0_raw / c4_swapped)}")
