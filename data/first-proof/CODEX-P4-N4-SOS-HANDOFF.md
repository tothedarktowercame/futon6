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
- A = a + b - st/6  (= E₄, absorbing the cross term)

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

## Deliverable

A sympy script that:
1. Verifies both identities symbolically
2. Computes the surplus polynomial (numerator after clearing denominators)
3. Reports the degree and number of terms
4. Attempts SOS decomposition (or identifies the obstruction)
5. Handles the e₃ = 0 special case if full SOS is too large

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
