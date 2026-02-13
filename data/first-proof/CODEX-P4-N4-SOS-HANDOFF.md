# Codex Handoff: Problem 4 — n=4 Stam Proof via Φ₄·disc Identity

**Date:** 2026-02-13
**From:** Claude (explore cycle)
**Priority:** HIGH — extends the n=3 proof to n=4, the first open case
**Prerequisite:** Task D is done (n=3 proved via Titu/SOS)

---

## What We Found

### Identity 1: Φ₄·disc is a polynomial

For centered degree-4 polynomials (e₁ = 0):

```
Φ₄ · disc = -8e₂⁵ - 64e₂³e₄ - 36e₂²e₃² + 384e₂e₄² - 432e₃²e₄
```

This is EXACT (verified numerically to machine precision over 200 random tests,
and confirmed symbolically at 3 specific root sets). Script: `explore-p4-n4-inv-phi.py`.

This generalizes the n=3 identity Φ₃·disc = 18e₂².

Therefore:

```
1/Φ₄ = disc(e₂,e₃,e₄) / (-8e₂⁵ - 64e₂³e₄ - 36e₂²e₃² + 384e₂e₄² - 432e₃²e₄)
```

where disc = 256e₄³ - 128e₂²e₄² + 144e₂e₃²e₄ + 16e₂⁴e₄ - 27e₃⁴ - 4e₂³e₃².

### Identity 2: Centered ⊞₄ coefficient formulas

For centered polynomials (e₁(p) = e₁(q) = 0):

```
E₂(p⊞q) = e₂(p) + e₂(q)              [additive]
E₃(p⊞q) = e₃(p) + e₃(q)              [additive]
E₄(p⊞q) = e₄(p) + e₄(q) + (1/6)e₂(p)·e₂(q)   [CROSS TERM]
```

The cross term in E₄ is the key difference from n=3 (where ALL coefficients add).

### Notation

Write p as x⁴ + se₂x² - e₃x + e₄ (centered quartic). Use variables:
- s, t for -e₂(p), -e₂(q) (so s, t > 0 for real-rooted; we negate because e₂ < 0)
- u, v for e₃(p), e₃(q)
- a, b for e₄(p), e₄(q)

Then:
- S = s + t  (= -E₂)
- U = u + v  (= E₃)
- A = a + b + st/6  (= E₄, absorbing the cross term)

And Stam becomes:

```
1/Φ₄(S, U, A) ≥ 1/Φ₄(s, u, a) + 1/Φ₄(t, v, b)
```

where 1/Φ₄ = disc / P with P = -8e₂⁵ - 64e₂³e₄ - ...

---

## Task: Prove Stam for n=4

### Step 1: Verify the identities symbolically

Use sympy to:
1. Confirm Φ₄·disc = -8e₂⁵ - 64e₂³e₄ - 36e₂²e₃² + 384e₂e₄² - 432e₃²e₄
   from the root-level definition
2. Confirm the centered ⊞₄ formula with the (1/6)e₂(p)e₂(q) cross term

### Step 2: Express the Stam surplus

Compute:
```
surplus = 1/Φ₄(S, U, A) - 1/Φ₄(s, u, a) - 1/Φ₄(t, v, b)
```

This is a rational function in 6 variables (s, t, u, v, a, b). Clear the
common denominator to get surplus = N/D where N, D are polynomials.

### Step 3: Attempt SOS on the numerator

The numerator N should be non-negative when:
1. s, t > 0
2. Both cubics are real-rooted (discriminant constraints on (s,u,a) and (t,v,b))
3. The convolution is real-rooted (discriminant constraint on (S,U,A))

Try:
- Direct SOS on N (ignoring constraints — may fail)
- SOS with Positivstellensatz (adding multiples of the constraints)
- Reduction by symmetry (e.g., if surplus is symmetric in (p,q), use
  Schur decomposition)

### Step 4: If full SOS is intractable, try special cases

1. **e₃ = 0 case** (both cubics have no cubic term — "even" quartics):
   Reduces to 4 variables (s, t, a, b). Much more tractable.

2. **Equal shape** (s=t, u=v, a=b): Reduces to checking self-convolution,
   which should give equality in the Pythagorean sense.

3. **Small e₃ perturbation**: Expand surplus around e₃ = 0 and check the
   leading-order terms.

### Complication: the e₄ cross term

For n=3, the proof used Titu's lemma on e₃²/e₂² terms, which worked because
MSS adds e₂ and e₃ independently. For n=4, E₄ = e₄(p) + e₄(q) + (1/6)e₂(p)e₂(q),
so the cross term couples e₂ and e₄. This means the surplus polynomial involves
MIXED terms that Titu alone won't handle.

However: the cross term (1/6)e₂(p)e₂(q) = (1/6)·s·t (using our sign convention)
is always POSITIVE (since s, t > 0). This means E₄ > e₄(p) + e₄(q), i.e., the
convolution INCREASES e₄ relative to pure addition. Depending on the sign structure
of 1/Φ₄, this could help or hurt.

### Key structural question

Factor the denominator P = -8e₂⁵ - 64e₂³e₄ - 36e₂²e₃² + 384e₂e₄² - 432e₃²e₄:
```
P = -4e₂(2e₂⁴ + 16e₂²e₄ + 9e₂e₃² - 96e₄²) - 432e₃²e₄
```
Hmm, not clean. Try other factorizations. If P factors nicely, the surplus may
decompose into individually non-negative terms.

---

---

## NEW: Symmetric Quartic Case (e₃ = 0) — Already Reduced

Script: `explore-p4-n4-symmetric-quartic.py`

For e₃ = 0 (symmetric quartics with roots ±α, ±β), everything factors:

```
disc(s, 0, e₄) = 16·e₄·(s² - 4e₄)²
P(s, 0, e₄) = -8·(-s)·(s² + 12e₄)·(s² - 4e₄)
1/Φ₄(s, 0, e₄) = 2·e₄·(s² - 4e₄) / [s·(s² + 12e₄)]
```

The Stam surplus numerator has **30 terms** (degree 10 in s,t,a,b) compared to
233 monomials in root coordinates (Codex Task 4). The denominator is fully factored
and positive: `9st(12a+s²)(12b+t²)(s+t)(12a+12b+s²+4st+t²)`.

Self-convolution (s=t, a=b) factors **manifestly non-negative**:
```
2s²(s² - 12a)²(12a + s²)(12a + 5s²)
```

**Priority approach:** Prove the 30-term numerator ≥ 0 first (symmetric quartics),
then tackle the full 6-variable case.

Convolution is automatically real-rooted when both inputs are (no extra constraint).

Feasibility region: s, t > 0, 0 < 4a < s², 0 < 4b < t².

### Concrete numerator to certify

```
N = 10368·a³bt² + 864·a³t⁴ + 10368·a²b²s² + 10368·a²b²t²
  + 864·a²bs²t² + 3456·a²bst³ + 1728·a²bt⁴ + 288·a²s²t⁴
  + 576·a²st⁵ + 72·a²t⁶ + 10368·ab³s² + 1728·ab²s⁴
  + 3456·ab²s³t + 864·ab²s²t² - 936·abs⁴t² - 1728·abs³t³
  - 936·abs²t⁴ - 42·as⁴t⁴ - 24·as³t⁵ + 18·as²t⁶
  + 864·b³s⁴ + 72·b²s⁶ + 576·b²s⁵t + 288·b²s⁴t²
  + 18·bs⁶t² - 24·bs⁵t³ - 42·bs⁴t⁴ + 3·s⁶t⁴ + 4·s⁵t⁵ + 3·s⁴t⁶
```

Some coefficients are negative (the ab·s⁴t² etc. terms). Need SOS or
Positivstellensatz on the cone {s,t > 0, 4a < s², 4b < t²}.

---

## Deliverable

A sympy script that:
1. Verifies the Φ₄·disc identity symbolically (from roots)
2. For the **e₃ = 0 case**: produces SOS certificate for the 30-term numerator
3. For the **general case**: computes surplus numerator and reports structure
4. If SDP tooling is available (cvxpy/scs), attempts automated SOS
5. If not, tries Titu/AM-GM/Schur decomposition by hand

---

## Context for This Task

| n | Status | Identity | MSS structure |
|---|--------|----------|---------------|
| 2 | PROVED (equality) | Φ₂·disc trivial | All coefficients add |
| 3 | PROVED (SOS) | Φ₃·disc = 18e₂² | All coefficients add |
| 4 | **THIS TASK** | Φ₄·disc = polynomial | e₂,e₃ add; e₄ has cross term |
| ≥5 | Open | Expect Φ_n·disc = polynomial | Higher e_k have cross terms |

The pattern suggests: Φ_n·disc is always a polynomial in the elementary
symmetric functions. If true, this gives a GENERAL algebraic framework
for proving Stam degree by degree.
