# Problem 4 Session Summary — 2026-02-13

## What We Proved

### Theorem 1: Backward Heat Equation for MSS Convolution
**Statement:** p_t = p ⊞_n He_t satisfies ∂p_t/∂t = -(1/2) ∂²p_t/∂x².

**Proof:** Via the MSS Weight Identity:
w(n,i,j)·(n-j+2)(n-j+1) = (n-i-j+2)(n-i-j+1)·w(n,i,j-2),
which is a one-line factorial cancellation: both sides = (n-i)!(n-j+2)!/(n!(n-i-j)!).

This lifts the Hermite backward heat equation (Lemma 1: dHe_t/dt = -(1/2)He_t'')
to the full MSS convolution.

**Verification:** Symbolic (SymPy) for n=2..6 with generic coefficients.
Script: `scripts/prove-p4-coulomb-flow.py`

### Theorem 2: Finite Coulomb Flow
**Statement:** Roots of p_t satisfy dγ_k/dt = S_k(γ) = Σ_{j≠k} 1/(γ_k - γ_j).

**Proof:** Implicit differentiation + backward heat equation + standard root identity
p''(γ_k) = 2p'(γ_k)S_k. Three lines from Theorem 1.

**Verification:** Numerical, c = 1.00000006 ± 2×10⁻⁸ for n=3,4,5.
Script: `scripts/verify-p4-root-velocity.py`

### Corollary: Finite De Bruijn Identity
**Statement:** d/dt H'_n(p_t) = Φ_n(p_t) where H'_n = Σ_{i<j} log|gaps|.

**Proof:** Chain rule + rearrangement + Theorem 2. Four lines.

### Theorem 3: Φ_n Monotonicity (SOS Identity)
**Statement:** dΦ_n/dt = -2 Σ_{k<j} (S_k - S_j)²/(γ_k - γ_j)² ≤ 0.

**Proof:** Differentiate Φ_n = Σ S_k², use dS_k/dt from Coulomb flow, symmetrize.
Sum-of-squares identity — three lines.

**Corollary:** 1/Φ_n(p_t) is strictly increasing along the Hermite heat flow.
Verified: 0 violations in 3480 consecutive time pairs for n=3..6.
Script: `scripts/verify-p4-dphi-structure.py`

### Additional Results
- **Hermite semigroup:** He_s ⊞_n He_t = He_{s+t} (verified numerically)
- **Universal de Bruijn constant:** c'/(dVar/dt) = 1/(n-1) for any kernel
  (Hermite c'=1, equally-spaced c'=(n+1)/12, etc.)

---

## What We Discovered (Numerically Verified, Unproved)

1. **-log Φ_n is superadditive:** Φ(p⊞q) ≤ Φ(p)·Φ(q) (0 violations in 500 tests)
   Weaker than Stam but potentially easier to prove.

2. **Stam inequality itself:** 1/Φ(p⊞q) ≥ 1/Φ(p) + 1/Φ(q) (0 violations in 35K+ tests)
   The target inequality. Still unproved for n ≥ 4.

---

## What We Killed

1. **Approach A (Haar projection):** Sample-level Stam (A2) fails 47% of the time.
   The Haar-averaging Jensen step doesn't work.

2. **Log-discriminant superadditivity:** SCALE-DEPENDENT, fails at all n for large
   root spread. The surplus shifts by -n(n-1)/2 · log(c) under scaling.
   The "de Bruijn + B2 → Stam" route is NOT viable.

3. **Finite entropy power inequality:** N_n(p⊞q) ≥ N_n(p) + N_n(q) fails 85%+.
   The classical EPI route is completely dead in the finite case.

---

## What Remains Open

The Stam inequality proof for n ≥ 4 requires an ingredient beyond our proved results.
The most promising directions:

a) **Score projection argument:** Finite analog of Voiculescu's conjugate-variable
   Cauchy-Schwarz. This is what Voiculescu actually uses (NOT heat flow).

b) **Direct algebraic SOS:** For n=4 specifically, after centering, there are only
   2 free parameters per polynomial. A computer algebra certificate might be feasible.

c) **Lorentzian polynomial approach:** The cone of real-rooted polynomials may have
   special properties under the MSS convolution that restrict Φ_n behavior.

d) **Prove -log Φ_n superadditivity first:** This is weaker than Stam but might be
   more tractable (log-subadditivity of Φ_n). Could then strengthen to Stam.

---

## Files Created/Modified This Session

### New scripts (7 files)
- `scripts/prove-p4-coulomb-flow.py` — Complete symbolic proof verification
- `scripts/verify-p4-conditional-stam.py` — Tests for conditions A1, A2, B1, B2
- `scripts/verify-p4-debruijn-proof.py` — Kernel-dependence tests
- `scripts/verify-p4-root-velocity.py` — Root velocity = S_k test
- `scripts/verify-p4-phi-monotonicity.py` — Φ_n monotonicity along flow
- `scripts/verify-p4-dphi-structure.py` — SOS formula for dΦ/dt
- `scripts/verify-p4-stam-chain.py` — Proof chain investigation
- `scripts/verify-p4-logdisc-superadd.py` — Log-disc superadditivity deep dive
- `scripts/verify-p4-scale-invariant.py` — Scale-invariant functional search

### New data/documentation (5 files)
- `data/first-proof/problem4-debruijn-discovery.md` — Full discovery report (PROVED)
- `data/first-proof/problem4-conditional-stam.md` — Conditional theorem formalization
- `data/first-proof/problem4-conditional-tests.jsonl` — Test results
- `data/first-proof/CODEX-P4-LITERATURE-HANDOFF.md` — Literature search for Codex
- `data/first-proof/CODEX-P4-NUMERICAL-HANDOFF.md` — Numerical verification for Codex

### Modified LaTeX (1 file)
- `data/first-proof/latex/full/problem4-solution-full.tex` — Added Sections 5c-5d
  with all proved results (backward heat eq, Coulomb flow, de Bruijn, Φ monotonicity)

---

## Commits This Session
1. `710e168` — P4: discover Finite Coulomb Flow Theorem and De Bruijn Identity
2. `df67f9a` — P4: prove Coulomb Flow Theorem + Φ_n monotonicity, investigate Stam chain
3. `cc167f0` — P4 LaTeX: add Sections 5c-5d with proved results
4. `01425a7` — P4: correct log-disc superadditivity analysis
5. `98f46a3` — P4: search for scale-invariant superadditive functionals
