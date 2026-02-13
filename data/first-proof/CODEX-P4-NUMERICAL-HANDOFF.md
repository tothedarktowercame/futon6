# Codex Handoff: Problem 4 Numerical Verification of Conditional Theorem

**Date:** 2026-02-13
**From:** Claude (monograph author)
**Priority:** HIGH — determines which proof approach is viable

---

## Context

We have formalized a **Conditional Finite Stam Theorem** with two approaches
(see `problem4-conditional-stam.md`). Before attempting proofs, we need to
NUMERICALLY VERIFY that the conditions actually hold. If a condition fails
numerically, the corresponding approach is dead.

All infrastructure exists in `scripts/verify-p4-*.py`. Core functions
(`phi_n`, `inv_phi_n`, `mss_convolve`, random real-rooted polynomial
generation) can be imported from `verify-p4-strategy-a.py`.

---

## Test 1: Jensen Condition (A1)

**Statement:** Φ_n(p ⊞_n q) ≤ E_U[Φ_n(char(A + UBU*))]

**Algorithm:**
```python
for each (n, p, q) test case:
    A = diag(roots of p)
    B = diag(roots of q)
    phi_conv = phi_n(roots of p ⊞_n q)     # LHS

    samples = []
    for trial in range(N_samples):          # N_samples = 1000+
        U = random_haar_unitary(n)
        C = A + U @ B @ U.conj().T
        eigenvalues = sorted(np.linalg.eigvalsh(C))
        samples.append(phi_n(eigenvalues))

    E_phi = np.mean(samples)                # RHS estimate
    ratio = phi_conv / E_phi                # should be <= 1
    print(f"n={n}, ratio={ratio:.6f}")
```

**Test cases:**
- n = 3, 4, 5
- Random real-rooted pairs: 100 pairs per n
- Also test adversarial cases: scale-separated pairs (roots of p at
  scale s, roots of q at scale 1/s, for s = 10, 100, 1000)
- N_samples = 1000 per pair (sufficient for ~3% relative error on mean)

**Expected result:** ratio ≤ 1 for all cases. Record the distribution
of ratios and any violations.

**Generating random Haar unitaries:**
```python
from scipy.stats import unitary_group
U = unitary_group.rvs(n)
```

**Generating random real-rooted polynomials:**
Use `random_real_rooted(n)` from verify-p4-strategy-a.py, or generate
n random roots from N(0,1) and compute coefficients.

---

## Test 2: Sample-Level Stam (A2)

**Statement:** E_U[Φ_n(char(A + UBU*))] ≤ Φ_n(p)·Φ_n(q)/(Φ_n(p)+Φ_n(q))

**Algorithm:** Same sampling as Test 1, but compare E_phi with the
harmonic mean Φ_n(p)·Φ_n(q)/(Φ_n(p)+Φ_n(q)).

```python
harmonic_mean = phi_n(roots_p) * phi_n(roots_q) / (phi_n(roots_p) + phi_n(roots_q))
ratio = E_phi / harmonic_mean    # should be <= 1
```

**If (A2) fails:** The projection approach (Approach A) is blocked.
Record the worst violation ratio and the specific (p, q) pair.

---

## Test 3: Finite De Bruijn (B1)

**Statement:** d/dt H_n(p_t)|_{t=0+} = -c · Φ_n(p) where
H_n = (2/n²) Σ_{i<j} log|λ_i - λ_j| and p_t = p ⊞_n h_t.

**Algorithm:**
```python
def log_discriminant(roots):
    n = len(roots)
    H = 0.0
    for i in range(n):
        for j in range(i+1, n):
            H += np.log(abs(roots[i] - roots[j]))
    return 2.0 * H / (n * n)

# Heat kernel: h_t has roots = sqrt(t) * (equally spaced around 0)
# or Hermite roots scaled by sqrt(t)
def heat_kernel_roots(n, t):
    # Hermite roots (zeros of H_n)
    from numpy.polynomial.hermite_e import hermeroots
    base_roots = sorted(hermeroots([0]*n + [1]))
    return np.array(base_roots) * np.sqrt(t)

# Compute p_t = p ⊞_n h_t via MSS
for t in [0.001, 0.01, 0.1, 0.5, 1.0]:
    conv_roots = roots_of_mss_convolve(p_coeffs, h_t_coeffs, n)
    H_t = log_discriminant(conv_roots)
    Phi_t = phi_n(conv_roots)

    # Numerical derivative
    dH_dt = (H_{t+dt} - H_{t-dt}) / (2*dt)
    ratio = -dH_dt / Phi_t           # should be constant = c
```

**Test cases:** n = 3, 4, 5. Multiple starting polynomials p. Multiple t values.

**Expected result:** The ratio -dH/dt / Φ_n should be approximately
constant (= c) across different p and t values. If c varies, the
de Bruijn identity doesn't hold as stated.

---

## Test 4: Log-Discriminant Superadditivity (B2)

**Statement:** H_n(p ⊞_n q) ≥ H_n(p) + H_n(q)

**Algorithm:**
```python
H_p = log_discriminant(roots_p)
H_q = log_discriminant(roots_q)
H_conv = log_discriminant(roots_of_mss_convolve(p, q, n))
surplus = H_conv - H_p - H_q    # should be >= 0
```

**Test cases:** Same as Tests 1-2. 100 random pairs per n, n = 3, 4, 5.

**Expected result:** surplus ≥ 0 in all cases. This is the finite analog
of Voiculescu's free entropy power inequality.

**Note:** This is the FREE ENTROPY superadditivity, not the Fisher
information superadditivity. Both are needed but they're different.

---

## Test 5: Hessian Signature on C_n Tangent Space

**Statement:** The Hessian of F = 1/Φ_n, restricted to the tangent space
of the real-rooted cone C_n at a given point, has controlled signature.

**Algorithm:**
```python
# At a point κ = (κ_2, ..., κ_n) in cumulant space on C_n:
# 1. Compute Hessian of F numerically (finite differences)
# 2. Compute tangent space of C_n at κ
#    (directions δκ such that p(κ + ε·δκ) remains real-rooted for small ε)
# 3. Project Hessian onto tangent space
# 4. Count eigenvalue signs

# For the tangent space: the boundary of C_n is where disc(p) = 0.
# The tangent space at an interior point is all of R^{n-1}.
# At the boundary, the normal direction is ∇ disc(p).
# So this test is really about: what is the Hessian signature of F
# at points NEAR the boundary of C_n?

for trial in range(100):
    κ = random_cumulant_on_Cn(n)
    H = numerical_hessian(F, κ, dx=1e-6)
    eigenvalues = np.linalg.eigvalsh(H)
    n_pos = np.sum(eigenvalues > 1e-10)
    n_neg = np.sum(eigenvalues < -1e-10)
    n_zero = len(eigenvalues) - n_pos - n_neg
    print(f"Signature: ({n_pos}, {n_neg}, {n_zero})")
```

**Test cases:** n = 3, 4, 5. Points near boundary and interior of C_n.

**Expected result:** If the Lorentzian direction is right, the Hessian
on the tangent space should have signature (1, n-2) or (0, n-1)
(i.e., at most one positive eigenvalue). This is the Lorentzian condition.

---

## Test 6: n=4 Surplus in Cumulant Coordinates

**Statement:** The surplus Δ_4 = F(κ+λ) - F(κ) - F(λ) should be
simpler in cumulant coordinates than in coefficient coordinates.

**Algorithm:**
```python
# Cumulant transformation for n=4 centered:
# κ_2 = a_2, κ_3 = a_3, κ_4 = a_4 + (1/12)·a_2²
# Inverse: a_4 = κ_4 - (1/12)·κ_2²

# In cumulant coords, ⊞_4 = plain addition:
# κ(conv) = κ(p) + κ(q)

# Compute surplus symbolically using SymPy
from sympy import symbols, Rational, expand, together, fraction

k2, k3, k4, l2, l3, l4 = symbols('k2 k3 k4 l2 l3 l4')

# Convert to coefficients
a2, a3, a4 = k2, k3, k4 - Rational(1,12)*k2**2
b2, b3, b4 = l2, l3, l4 - Rational(1,12)*l2**2

# Convolution coefficients (plain addition in cumulant coords)
c2 = k2 + l2
c3 = k3 + l3
c4_cum = k4 + l4
c4_coeff = c4_cum - Rational(1,12)*(k2+l2)**2  # back to coefficient

# Verify: c4_coeff should equal a4 + (1/6)*a2*b2 + b4
# a4 + (1/6)*a2*b2 + b4 = (k4 - k2²/12) + k2*l2/6 + (l4 - l2²/12)
#                        = k4 + l4 - k2²/12 + k2*l2/6 - l2²/12
#                        = k4 + l4 - (k2+l2)²/12 + k2*l2/6 + k2*l2/12 - k2*l2/12
# Hmm, let me just verify numerically.

# Then compute 1/Phi_4 symbolically in cumulant coords and extract surplus.
```

**Goal:** Get the symbolic expression for the surplus in cumulant coords.
If it's simpler than in coefficient coords, this validates the cumulant
approach. The coefficient-space surplus has been computed in
`verify-p4-n4-algebraic.py` — compare complexity.

---

## Output Format

For each test, return:
1. **Pass/Fail** (any violations?)
2. **Statistics:** min, max, mean, std of the test ratio
3. **Worst case:** the specific (p, q) pair giving the closest-to-violation result
4. **Visualization suggestion:** histograms of ratios, scatter plots of
   ratio vs. n or vs. condition number of roots

Save results to `data/first-proof/problem4-conditional-tests.jsonl`
(one JSON object per test).

---

## Dependencies

- numpy, scipy (for Haar sampling, eigenvalue computation)
- sympy (for Test 6 symbolic computation)
- Existing scripts: `scripts/verify-p4-strategy-a.py` (core functions)

Install if needed:
```bash
pip install numpy scipy sympy
```
