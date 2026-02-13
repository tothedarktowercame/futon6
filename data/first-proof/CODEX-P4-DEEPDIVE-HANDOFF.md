# Codex Handoff: Problem 4 Deep Dive — Remaining Proof Directions

**Date:** 2026-02-13
**From:** Claude (monograph author)
**Priority:** HIGH — this is the main open problem in the monograph
**Prerequisite reading:** `problem4-session-summary.md`, `problem4-debruijn-discovery.md`

---

## Executive Summary

We have PROVED:
- Backward heat equation: p_t = p ⊞_n He_t satisfies ∂p_t/∂t = -(1/2)∂²p_t/∂x²
- Finite Coulomb flow: dγ_k/dt = S_k(γ) = Σ_{j≠k} 1/(γ_k - γ_j)
- Finite de Bruijn identity: d/dt H'_n(p_t) = Φ_n(p_t)
- Φ_n monotonicity: dΦ/dt = -2 Σ_{k<j} (S_k-S_j)²/(γ_k-γ_j)² ≤ 0

We have KILLED:
- Log-discriminant superadditivity (scale-dependent, fails for all n at large spread)
- Finite entropy power inequality (85%+ violations)
- Haar projection / sample-level Stam (A2 fails 47%)

We still NEED to prove for n ≥ 4:
**1/Φ_n(p ⊞_n q) ≥ 1/Φ_n(p) + 1/Φ_n(q)**

This handoff defines four independent deep-dive tasks, each exploring a different
remaining proof direction. Each task can be worked on in parallel.

---

## DEAD ENDS — DO NOT RE-EXPLORE

These approaches have been conclusively killed. Codex should NOT spend time on them:

1. **Log-discriminant superadditivity (B2):** H'_n(p⊞q) ≥ H'_n(p) + H'_n(q) is
   FALSE. The surplus shifts by -n(n-1)/2 · log(c) under scaling by c. At
   spread=2.0, violation rate is 30-43% for ALL n. Script: `verify-p4-logdisc-superadd.py`.

2. **Finite EPI:** N_n(p⊞q) ≥ N_n(p) + N_n(q) where N_n = exp(2H'/(n(n-1)))
   fails 85%+ of the time. Script: `verify-p4-scale-invariant.py`.

3. **Haar averaging route (Approach A):** E_U[Φ_n(A+UBU*)] ≤ harmonic mean(Φ(p),Φ(q))
   fails ~47%. The Jensen + sample Stam decomposition is dead.

4. **De Bruijn + entropy superadditivity → Stam:** This classical/free proof
   pattern requires a superadditive entropy functional. No such functional has
   been found. The de Bruijn identity is proved but cannot be chained into Stam
   without an additional ingredient.

---

## Direction A: Score Projection / Conjugate Variable Cauchy-Schwarz

### Motivation

Voiculescu's ACTUAL proof of the free Stam inequality (1998, Invent. Math. 132)
does NOT use heat flow. It uses a **score projection argument**:

1. Define the conjugate variable J(X:X,Y) = projection of the free score of X
   onto the von Neumann algebra W*(X+Y)
2. Show ‖J(X:X,Y)‖² ≤ Φ*(X) by L²-contractivity
3. Show J(X:X,Y) + J(Y:X,Y) = J(X+Y) by linearity
4. Show ⟨J(X:X,Y), J(Y:X,Y)⟩ = 0 from freeness
5. Pythagorean theorem: Φ*(X+Y) = ‖J(X:X,Y)‖² + ‖J(Y:X,Y)‖² ≤ Φ*(X) + Φ*(Y)
6. Wait... that's the wrong direction. The ACTUAL step uses Cauchy-Schwarz on
   the inverse: 1/Φ*(X+Y) = 1/‖J‖² ≥ ... via the harmonic decomposition.

The key insight: Voiculescu decomposes the score of X+Y into "X-part" and "Y-part"
that are orthogonal (from freeness). This orthogonality + Cauchy-Schwarz gives Stam.

### Task A1: Literature Deep Dive on Voiculescu's Proof

**Objective:** Extract the EXACT mechanism of Voiculescu (1998) and determine
which steps have finite analogs.

**Papers to read in full:**
- Voiculescu (1998), "The analogues of entropy and Fisher's information measure
  in free probability theory V," Invent. Math. 132, 189-227.
- Voiculescu (1993), "The analogues of entropy... I," CMP 155, 71-92.
- Hiai, Petz (2006), "The Semicircle Law, Free Random Variables, and Entropy,"
  AMS Mathematical Surveys and Monographs 77 — Chapter on free Fisher information.

**Extract specifically:**
1. The PRECISE definition of J(X) as ∂/∂X * 1 (the non-commutative derivative
   applied to 1 in L²(W*(X))).
2. How the conditional expectation E_{W*(X+Y)} decomposes J(X).
3. The exact Cauchy-Schwarz / projection step that gives 1/Φ* ≥ 1/Φ*(X) + 1/Φ*(Y).
4. Whether Voiculescu makes any remarks about finite-dimensional analogs.
5. The role of FREENESS specifically — which step breaks without it?

**The finite analog question:**
In the finite setting, the "score" of polynomial p at root γ_k is:
    S_k(γ) = Σ_{j≠k} 1/(γ_k - γ_j)

For p ⊞_n q with roots γ (coming from eigenvalues of A + UBU*), we want to
decompose S_k(γ) into an "A-part" and a "B-part" that are approximately orthogonal
under Haar averaging.

**Specific questions:**
- Is there a finite analog of J(X:X,Y)? Perhaps the partial derivative of the
  characteristic polynomial with respect to the A-eigenvalues?
- Does Haar averaging give an approximate orthogonality result for finite n?
- What correction terms appear at finite n, and do they have the right sign?

### Task A2: Numerical Experiment — Score Decomposition

**Objective:** Numerically test whether the score of p⊞q decomposes into
approximately orthogonal "p-part" and "q-part".

```python
import numpy as np
from scipy.stats import unitary_group

def score_field(roots):
    """S_k = sum_{j!=k} 1/(roots[k] - roots[j])"""
    n = len(roots)
    S = np.zeros(n)
    for k in range(n):
        S[k] = sum(1.0/(roots[k]-roots[j]) for j in range(n) if j!=k)
    return S

def experiment_score_decomposition(n, n_trials=200, n_samples=500):
    """Test orthogonality of score decomposition under Haar averaging."""
    for trial in range(n_trials):
        # Random real-rooted polynomials
        lam = np.sort(np.random.randn(n))  # roots of p
        mu = np.sort(np.random.randn(n))   # roots of q

        A = np.diag(lam)
        B = np.diag(mu)

        # For each sample U, compute eigenvalues γ of A+UBU*
        # and decompose S(γ) = S_A(γ) + S_B(γ) where:
        #   S_A_k(γ) = sum_j v_{jk}² / (γ_k - λ_j)   [if U*e_k = Σ v_{jk} f_j]
        #   S_B_k(γ) = sum_j w_{jk}² / (γ_k - μ_j)
        # (This is just a GUESS at the decomposition — the correct one needs
        # to come from the literature.)

        inner_products = []
        for _ in range(n_samples):
            U = unitary_group.rvs(n)
            C = A + U @ B @ U.conj().T
            gamma = np.sort(np.linalg.eigvalsh(C))

            S = score_field(gamma)

            # Attempt decomposition via resolvent:
            # The eigenvalues of A+UBU* relate to A-eigenvalues by resolvent
            # S_A_k = partial/partial(gamma_k) of log det(gamma_k I - A)
            #       = sum_j 1/(gamma_k - lambda_j)
            S_A = np.array([sum(1.0/(gamma[k]-lam[j]) for j in range(n))
                           for k in range(n)])
            S_B = np.array([sum(1.0/(gamma[k]-mu[j]) for j in range(n))
                           for k in range(n)])

            # Check: does S_A + S_B = S + correction?
            # And: is <S_A, S_B> small?
            inner = np.dot(S_A, S_B)
            inner_products.append(inner)

        mean_ip = np.mean(inner_products)
        # If mean_ip ≈ 0 under Haar, orthogonality holds!
        print(f"n={n}, trial {trial}: <S_A, S_B>_Haar = {mean_ip:.6f}")
```

**Key outputs:**
1. Does S_A + S_B = S (the score of the convolution)? Or is there a correction?
2. Is E_U[⟨S_A, S_B⟩] = 0? Or approximately 0?
3. What is the distribution of the inner product across U?
4. Does E_U[‖S_A‖²] relate to Φ_n(p)?

**Success criterion:** If E_U[⟨S_A, S_B⟩] ≤ 0 and S_A + S_B ≈ S, then
Cauchy-Schwarz on ‖S‖² = ‖S_A + S_B‖² = ‖S_A‖² + ‖S_B‖² + 2⟨S_A,S_B⟩
would give Φ(p⊞q) ≤ something, leading toward Stam.

### Task A3: Search for "Finite Conjugate Variable" Literature

**Search terms:**
- "conjugate variable" "finite dimensional" OR "matrix" OR "finite free"
- "free score" "finite" projection OR "conditional expectation"
- "non-commutative derivative" "finite" eigenvalue OR "characteristic polynomial"
- "finite free" Cauchy-Schwarz OR "Fisher information" decomposition
- Voiculescu "finite dimensional" OR "matrix" "Stam" OR "Fisher"
- "Stieltjes transform" "score" "finite" decomposition

**What we're looking for:**
- Any finite-dimensional analog of the free conjugate variable J(X)
- Decomposition of spectral functionals under random matrix addition
- Any paper that proves a Stam-type inequality for finite matrices

---

## Direction B: Direct Algebraic SOS for n=4

### Motivation

For n=4, after centering (κ_1=0) and using cumulant coordinates, the Stam
inequality becomes superadditivity of F(κ) = 1/Φ_4(p(κ)) on the real-rooted
cone C_4 ⊂ R³. With the additional normalization a2 = b2 = -1 (which is
WLOG by scaling), we have just TWO free parameters per polynomial: (a3, a4)
and (b3, b4). The surplus F(κ+λ) - F(κ) - F(λ) is a rational function of
(a3, a4, b3, b4) that we need to show is ≥ 0 on the semialgebraic set
defined by the discriminant constraints disc(p) > 0, disc(q) > 0, disc(p⊞q) > 0.

### What's Been Done

Extensive prior work exists in `scripts/verify-p4-n4-*.py` (30+ scripts).
Key results from earlier sessions:

- `verify-p4-n4-algebraic.py`: Verified Φ_4 · disc = -4(a2²+12a4)(2a2³-8a2a4+9a3²)
  symbolically. Built the surplus numerator in 4 variables.
- `verify-p4-n4-sos-reduced.py`: Tried SOS certificate search with reflection
  symmetry (N is even in a3, b3). Reduced basis from 126 to ~66 entries.
- `verify-p4-n4-sos-d12.py`, `verify-p4-n4-sos-d12-scs.py`: SOS certificate
  search using SCS solver.
- `verify-p4-n4-case2-*.py`, `verify-p4-n4-case3*.py`: Case analysis approach,
  splitting by critical points.

**Status:** No SOS certificate has been found yet. The numerator is degree ~12
in 4 variables, which is at the edge of computational feasibility.

### Task B1: Cumulant-Space SOS Reformulation

**Objective:** Reformulate the n=4 surplus entirely in cumulant coordinates.

In coefficient coordinates:
- ⊞_4 has cross-terms: c_4 = a_4 + (1/6)a_2 b_2 + b_4 etc.
- The surplus numerator is complicated because the convolution formula is nonlinear.

In cumulant coordinates:
- ⊞_4 = plain addition: κ(conv) = κ(p) + κ(q)
- The surplus is F(κ+λ) - F(κ) - F(λ) where F depends on κ through the
  coefficient-cumulant transformation.

**Specific computation:**
```python
from sympy import symbols, Rational, expand, together, fraction

# Cumulant variables (centered, so kappa_1 = 0)
k2, k3, k4, l2, l3, l4 = symbols('k2 k3 k4 l2 l3 l4')

# Cumulant-to-coefficient for n=4 centered:
# a2 = k2, a3 = k3, a4 = k4 - (1/12)*k2^2
# (The (1/12) comes from the finite free cumulant-moment relation)

# Convolution in cumulant space = plain addition
# c_k2 = k2+l2, c_k3 = k3+l3, c_k4 = k4+l4

# Convert all three polynomials to coefficients, compute F = 1/Phi_4 for each,
# compute surplus F(c) - F(k) - F(l), simplify.

# After a2=b2=-1 normalization (k2=l2=-1):
# a3=k3, a4=k4-1/12, b3=l3, b4=l4-1/12
# c2=-2, c3=k3+l3, c4=k4+l4-4/12=k4+l4-1/3
# c_a2=-2, c_a3=k3+l3, c_a4=(k4+l4-1/3) - (-2)^2/12 = k4+l4-1/3-1/3 = k4+l4-2/3

# The point: express the surplus as a function of (k3,k4,l3,l4) with
# the real-rootedness constraints expressed as discriminant > 0 in these coords.
# This may have simpler algebraic structure than the coefficient version.
```

**Deliverable:** The symbolic expression for the surplus in cumulant coordinates
and a comparison of its degree/complexity with the coefficient version. If simpler,
attempt SOS decomposition.

### Task B2: Interval Arithmetic Certificate for n=4

**Objective:** If symbolic SOS fails, try a computer-verified proof using
interval arithmetic.

**Approach:**
1. Parameterize the domain (a3,a4,b3,b4) with disc(p)>0, disc(q)>0, disc(p⊞q)>0.
2. Use subdivision: split the domain into boxes.
3. On each box, bound the surplus from below using interval arithmetic.
4. If every box has surplus > 0, we have a certified proof.

**Tools:** `mpmath` for interval arithmetic, or a specialized library like
`pyinterval`. For certification, the DSOS/SDSOS approach (Ahmadi-Majumdar 2019)
provides LP-based certificates that are easier to verify than SDP-based SOS.

**Key challenge:** The domain is unbounded (a3 can be any real number as long
as disc > 0). Need to either:
- Prove surplus → +∞ as |(a3,a4,b3,b4)| → ∞, reducing to a compact domain
- Use a substitution that compactifies the domain
- Handle the asymptotics analytically and the finite region numerically

**Search terms for tools/techniques:**
- "interval arithmetic" "polynomial inequality" certificate
- DSOS SDSOS "polynomial optimization" semialgebraic
- "Positivstellensatz" computational "4 variables"
- "cylindrical algebraic decomposition" "4 variables" quartic

### Task B3: Boundary Analysis

**Objective:** Analyze the surplus at the boundary of the real-rooted cone
(where discriminant = 0, i.e., two roots coincide).

When disc(p) → 0, some roots merge. The surplus should remain ≥ 0 at the
boundary (continuity). But the RATE at which the surplus vanishes encodes
the difficulty of the proof.

**Computation:**
```python
# At the boundary, parameterize by repeated root:
# For n=4 with a double root at r: p(x) = (x-r)^2(x-s)(x-t) where s,t are distinct
# This gives a 3-parameter family (r,s,t). Express surplus as function of
# (r,s,t) for p and (r',s',t') for q.
#
# Does the surplus vanish to order 1 or 2 at the boundary?
# If order 2, a quadratic form argument might work.
```

---

## Direction C: Lorentzian Polynomial / Cone-Restricted Convexity

### Motivation

The Stam functional 1/Φ_n is superadditive on the real-rooted cone WITHOUT
being globally convex (Hessian is indefinite in 100% of tested interior points).
This pattern — cone-restricted superadditivity — is reminiscent of:

1. **Lorentzian polynomials** (Brändén-Huh 2019): polynomials whose Hessian
   has signature (1, n-1) on the positive orthant. These have strong
   log-concavity properties.

2. **Hyperbolic polynomials** (Gårding 1959): polynomials with all real roots
   in every direction of a cone. The MSS convolution preserves hyperbolicity.

3. **Complete log-concavity** (ALOGV 2018): operators on the space of polynomials
   that preserve log-concavity. The MSS differentiation identity might fit.

### Task C1: Hessian Signature Analysis

**Objective:** Compute the Hessian of 1/Φ_n in cumulant coordinates, restricted
to tangent directions of the real-rooted cone, and characterize its signature.

```python
import numpy as np
from scipy.optimize import approx_fprime

def numerical_hessian(f, x, dx=1e-6):
    """Compute Hessian of f at x by finite differences."""
    n = len(x)
    H = np.zeros((n, n))
    fx = f(x)
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i] = dx; ej[j] = dx
            H[i,j] = (f(x+ei+ej) - f(x+ei-ej) - f(x-ei+ej) + f(x-ei-ej)) / (4*dx*dx)
            H[j,i] = H[i,j]
    return H

def inv_phi_cumulant(kappa, n):
    """Compute 1/Phi_n given cumulant vector kappa = (k2, k3, ..., k_n)."""
    # Convert cumulants to coefficients
    # ... (need cumulant-to-coefficient transform for degree n)
    # Then coefficients to roots
    # Then compute 1/phi_n(roots)
    pass

# For n=4 specifically (after centering):
# kappa = (k2, k3, k4) with a2=k2, a3=k3, a4=k4-(1/12)*k2^2
# At many random real-rooted points, compute Hessian of F = 1/Phi_4
# and record eigenvalue signature.

# KEY QUESTION: On the tangent space of C_n at the boundary, does the
# Hessian have signature (1, d-1)? This would be the Lorentzian condition.
```

**What to record:**
1. Hessian eigenvalues at 100+ interior points of C_4
2. Hessian eigenvalues at points near the boundary (disc ≈ 0)
3. How the signature changes from interior to boundary
4. Whether the negative eigenspace is always of dimension ≥ n-2

**Success criterion:** If the Hessian restricted to cone-tangent directions has
signature (1, d-1) or better (i.e., at most one positive eigenvalue), this
matches the Lorentzian condition and suggests a proof via Brändén-Huh theory.

### Task C2: Literature Search — Cone-Restricted Superadditivity

**Search terms:**
- "Lorentzian polynomial" superadditive OR superadditivity
- "real-rooted" cone convexity OR superadditivity "restricted"
- "hyperbolic polynomial" "Fisher information" OR "root energy"
- Brändén Huh "completely log-concave" operator
- "stable polynomial" cone "superadditive"
- "log-concave" "cone" "superadditive" OR "submodular"
- "Hodge-Riemann" "real-rooted" function OR functional
- "mixed discriminant" superadditive OR log-concave
- Gurvits "capacity" "stable polynomial" superadditive

**What we're looking for:**
- Any theorem that says: "A function f on the real-rooted cone satisfying
  [Hessian condition] is superadditive under [addition operation]."
- Connection between Lorentzian conditions and superadditivity under
  coordinate-wise addition (since ⊞_n = addition in cumulant coords).
- Any result by Brändén, Huh, Anari, Gharan, or collaborators on
  energy-type functionals on polynomial spaces.
- Results from matroid theory / ALOGV that might apply to 1/Φ_n.

### Task C3: MSS + Lorentzian Connection

**Objective:** Determine whether the MSS finite free convolution has been
studied in the context of Lorentzian/stable polynomial theory.

Marcus-Spielman-Srivastava (2015) proved the Kadison-Singer conjecture using
interlacing families and the method of mixed characteristic polynomials. Their
⊞_n operation preserves real-rootedness. Brändén-Huh (2019) developed Lorentzian
polynomial theory partly inspired by the MSS work.

**Specific questions:**
- Does the MSS convolution preserve Lorentzian-ness?
- Is there a "Lorentzian lift" of the real-rooted cone that makes ⊞_n natural?
- Has anyone studied 1/Φ_n or similar energy functionals as Lorentzian functions?

---

## Direction D: Prove -log Φ_n Superadditivity First

### Motivation

We discovered numerically that **Φ(p⊞q) ≤ Φ(p)·Φ(q)** (0 violations in 500+
tests per n, for n=3..5). Equivalently, -log Φ_n is superadditive. This is
STRICTLY WEAKER than the Stam inequality:

    Stam:  1/Φ(pq) ≥ 1/Φ(p) + 1/Φ(q)
    ⟹ log(1/Φ(pq)) ≥ log(1/Φ(p) + 1/Φ(q)) ≥ log(max(1/Φ(p), 1/Φ(q)))

    Weaker: Φ(pq) ≤ Φ(p)·Φ(q)
    ⟺ log Φ(pq) ≤ log Φ(p) + log Φ(q)

The Stam inequality implies log-subadditivity of Φ_n (via AM-HM), but
log-subadditivity does NOT imply Stam. However:

1. Log-subadditivity might be EASIER to prove.
2. It could serve as a stepping stone: prove Φ(p⊞q) ≤ Φ(p)·Φ(q), then
   strengthen to the full Stam.
3. In the classical case, this is the statement I(X+Y) ≤ I(X)·I(Y)/(I(X)+I(Y))...
   wait, no. The classical analog is I(X+Y) ≤ I(X) (data processing), which is
   weaker. The log-submultiplicativity Φ(p⊞q) ≤ Φ(p)Φ(q) does not have a
   direct classical analog.

### Task D1: Extended Numerical Verification

**Objective:** Strengthen the numerical evidence for Φ(p⊞q) ≤ Φ(p)·Φ(q).

```python
import numpy as np
from math import factorial

def mss_convolve(a_coeffs, b_coeffs, n):
    c = np.zeros(n)
    for k in range(1, n + 1):
        s = 0.0
        for i in range(k + 1):
            j = k - i
            if i > n or j > n: continue
            ai = 1.0 if i == 0 else a_coeffs[i - 1]
            bj = 1.0 if j == 0 else b_coeffs[j - 1]
            w = (factorial(n - i) * factorial(n - j)) / (factorial(n) * factorial(n - k))
            s += w * ai * bj
        c[k - 1] = s
    return c

def phi_n(roots):
    n = len(roots)
    return sum(sum(1.0/(roots[i]-roots[j]) for j in range(n) if j!=i)**2 for i in range(n))

# Test at larger n (6, 7, 8) and with adversarial inputs:
# - Scale-separated: roots of p at scale 1, roots of q at scale 100
# - Near-degenerate: two roots nearly coinciding
# - Hermite × random: p = Hermite, q = random
# Run 10,000+ tests per n

for n in [3, 4, 5, 6, 7, 8]:
    violations = 0
    total = 0
    min_ratio = float('inf')  # Phi(pq) / (Phi(p)*Phi(q)), should be <= 1

    for trial in range(10000):
        p = np.sort(np.random.randn(n) * np.random.uniform(0.1, 10.0))
        q = np.sort(np.random.randn(n) * np.random.uniform(0.1, 10.0))

        conv = mss_convolve(np.poly(p)[1:], np.poly(q)[1:], n)
        # check real-rootedness, compute phi, etc.
        # ...

    print(f"n={n}: {violations}/{total} violations, min ratio = {min_ratio:.10f}")
```

**Also test:** self-convolution p ⊞_n p. Is Φ(p⊞p) ≤ Φ(p)²? This would be
a simpler special case.

### Task D2: Relate to Proved Results

**Objective:** Determine whether Φ(p⊞q) ≤ Φ(p)·Φ(q) can be derived from
the proved Coulomb flow machinery.

**Specific question:** Along the joint flow (p_t, q_t, (p⊞q)_t), we have:
- dΦ(p_t)/dt ≤ 0 (proved, SOS identity)
- dΦ(q_t)/dt ≤ 0 (proved)
- dΦ((p⊞q)_t)/dt ≤ 0 (proved)

Can we show that the RATIO Φ(p_t ⊞ q_t) / (Φ(p_t) · Φ(q_t)) is monotone
along the flow? If so, and if it has the right limit as t → ∞, we'd get
the inequality.

**Computation:**
```python
# Track ratio R(t) = Phi((p⊞q)_t) / (Phi(p_t) * Phi(q_t)) along Hermite flow
# and test: dR/dt ≤ 0?

# Also: as t → ∞, p_t → He (the Hermite polynomial). So the limit is
# Phi(He⊞He) / Phi(He)^2. Since He⊞He = He_{2t} (semigroup), this gives
# a concrete ratio to check.
```

### Task D3: Prove for n=3 Algebraically

**Objective:** Prove Φ_3(p⊞q) ≤ Φ_3(p)·Φ_3(q) for n=3 as a warm-up.

For n=3 centered with a2=b2=-1, we have:
    Φ_3(p) = 18·a2² / disc(p)·(-1) = ...

Actually, use the identity Φ_3·disc = 18·a2² (or whatever the exact form is
for n=3) to write:

    Φ_3(p) = 18 / disc_norm(p)

where disc_norm is the discriminant divided by the appropriate power of a2.

Then Φ(p)·Φ(q)/Φ(p⊞q) = disc_norm(p⊞q) / (disc_norm(p)·disc_norm(q)) ≥ 1.

This would follow from the MULTIPLICATIVITY or SUPERmultiplicativity of the
normalized discriminant under ⊞_3. Check whether this simplifies.

### Task D4: Connection to Log-Submodularity

**Search terms:**
- "Fisher information" "log-subadditivity" OR "submultiplicative"
- "free Fisher" "multiplicative" OR "submultiplicative"
- "log-submodular" "eigenvalue" OR "root" OR "spectral"
- "Φ" "convolution" submultiplicative OR "product inequality"

---

## General Infrastructure Notes

### Core functions (reusable across all tasks)

All scripts should import or re-implement these core functions:

```python
from math import factorial
import numpy as np

def mss_convolve(a_coeffs, b_coeffs, n):
    """Finite free additive convolution via MSS weights."""
    c = np.zeros(n)
    for k in range(1, n + 1):
        s = 0.0
        for i in range(k + 1):
            j = k - i
            if i > n or j > n: continue
            ai = 1.0 if i == 0 else a_coeffs[i - 1]
            bj = 1.0 if j == 0 else b_coeffs[j - 1]
            w = (factorial(n - i) * factorial(n - j)) / (factorial(n) * factorial(n - k))
            s += w * ai * bj
        c[k - 1] = s
    return c

def phi_n(roots):
    """Root-force energy: Σ_i (Σ_{j≠i} 1/(γ_i - γ_j))²."""
    n = len(roots)
    return sum(sum(1.0/(roots[i]-roots[j]) for j in range(n) if j!=i)**2
               for i in range(n))

def inv_phi_n(roots):
    return 1.0 / phi_n(roots)

def log_disc(roots):
    """H'_n = Σ_{i<j} log|γ_i - γ_j|."""
    n = len(roots)
    return sum(np.log(abs(roots[i]-roots[j]))
               for i in range(n) for j in range(i+1,n))

def coeffs_to_roots(coeffs):
    return np.sort(np.roots(np.concatenate([[1.0], coeffs])).real)

def is_real_rooted(coeffs, tol=1e-6):
    return np.max(np.abs(np.roots(np.concatenate([[1.0], coeffs])).imag)) < tol

def hermite_heat_convolve(p_roots, t, n):
    """Compute p ⊞_n He_t where He_t has roots sqrt(t) * (Hermite zeros)."""
    from numpy.polynomial.hermite_e import hermeroots
    he_roots = np.sort(hermeroots([0]*n + [1])) * np.sqrt(t)
    he_coeffs = np.poly(he_roots)[1:]
    p_coeffs = np.poly(p_roots)[1:]
    conv = mss_convolve(p_coeffs, he_coeffs, n)
    if not is_real_rooted(conv):
        return None
    return coeffs_to_roots(conv)
```

### Dependencies

```bash
pip install numpy scipy sympy
# For SOS/SDP: pip install cvxpy scs
# For interval arithmetic: pip install mpmath
```

### Existing scripts to reference

- `scripts/verify-p4-n4-algebraic.py` — n=4 symbolic surplus computation
- `scripts/verify-p4-n4-sos-reduced.py` — SOS with reflection symmetry
- `scripts/verify-p4-n4-sos-d12-scs.py` — SOS via SCS solver
- `scripts/prove-p4-coulomb-flow.py` — Complete symbolic proof of Coulomb flow
- `scripts/verify-p4-scale-invariant.py` — Functional superadditivity tests
- `scripts/verify-p4-dphi-structure.py` — SOS formula for dΦ/dt

---

## Output Format

For each task, return:

1. **Result type:** theorem / conjecture / counterexample / negative result
2. **Key finding** (1-3 sentences)
3. **Evidence:** numerical statistics, symbolic expressions, or literature citations
4. **Implication for Stam proof:** does this open a viable path?
5. **Suggested next step** if the result is promising
6. **Code and data** saved to `scripts/` and `data/first-proof/`

Save all numerical results to `data/first-proof/problem4-deepdive-results.jsonl`
(one JSON object per experiment).

---

## Priority Order

1. **Direction D** (log-subadditivity) — lowest-hanging fruit, weakest statement,
   most likely to yield a proof. Tasks D1-D3 can be done quickly.

2. **Direction A** (score projection) — highest potential payoff if it works,
   since it would directly prove Stam. But requires understanding Voiculescu's
   exact mechanism, which is the hardest literature task.

3. **Direction B** (n=4 SOS) — well-defined computational task with clear
   success/failure. The cumulant reformulation (B1) might simplify enough
   to make the SOS feasible.

4. **Direction C** (Lorentzian) — highest-risk, highest-novelty. Could open
   a completely new proof technique but requires significant theoretical
   development. Start with the numerical Hessian analysis (C1) to determine
   if the Lorentzian condition even holds.
