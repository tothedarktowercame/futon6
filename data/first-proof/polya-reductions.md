# Polya Reductions for Problems 3 & 5

Applying `math-informal/try-a-simpler-case` and `math-informal/reduce-to-known-result`
with the CT lens: find problems with the **same wiring diagram shape** that we can
verify, then compose into the full answer.

---

## Problem 3: ASEP Markov Chain

### The proof shape (as a wiring diagram)

```
exchange-relations(T_i, F*_mu) --> detailed-balance(rate, pi) --> stationary(chain, pi)
```

Three nodes, two edges. The first arrow is the hard step (do exchange relations
give valid rates?), the second is mechanical (detailed balance implies stationarity).

### Reduction 1: n=2 (composable lemma)

**This is a genuine lemma, not just a sanity check.**

State space: S_2(lambda) = {(a,b), (b,a)} where a = lambda_1 > b = lambda_2.
Single swap. The chain is a two-state Markov chain:

    (a,b) --[r]--> (b,a) --[r']--> (a,b)

Stationary distribution: pi(a,b) = r'/(r+r'), pi(b,a) = r/(r+r').

We need: r'/(r+r') = F*_{(a,b)}(x;1,t) / (F*_{(a,b)} + F*_{(b,a)}).

This gives us the **rate ratio**:

    r/r' = F*_{(b,a)}(x;1,t) / F*_{(a,b)}(x;1,t)

Now from the exchange relation: T_1 F*_{(a,b)} = c F*_{(a,b)} + d F*_{(b,a)}.
At q=1, the Hecke generator T_1 satisfies T_1^2 = (t-1)T_1 + t.

**This is completely checkable.** We need to:
1. Look up the explicit formula for F*_{(a,b)}(x_1, x_2; 1, t)
2. Compute the exchange coefficient ratio c/d at q=1
3. Verify it matches the rate ratio r/r' = F*_{(b,a)}/F*_{(a,b)}

If this works, the n=2 lemma **composes** into the full chain because:
- The full chain is a product of pairwise swap operators M_i
- Each M_i acts on positions (i, i+1) only
- Stationarity of the composed chain follows from detailed balance of
  EACH pairwise swap separately (this is the standard argument for
  reversible Markov chains built from transpositions)

### Reduction 2: q=t case (reduce to known result)

At q=t (not q=1), the interpolation Macdonald polynomials become
interpolation Jack polynomials. The associated Markov chain should be the
**Heckman-Opdam process** or a discrete analogue. This is better understood.

If we can verify that the q=t chain exists and then take the limit as
q -> 1 (with t fixed), the q=1 chain follows by continuity of the
stationary distribution.

### Reduction 3: t=0 (degenerate case)

At t=0, the Hecke relation becomes T_i^2 = -T_i (nilpotent), so
T_i is a projection. The rates become:
- Forward: rate 1 (always swap to sorted order)
- Backward: rate 0 (never swap against sorted order)

Stationary distribution: concentrated on the sorted composition
(lambda_1, ..., lambda_n). This is trivially correct — the chain
converges to the identity permutation.

**Check:** Does F*_mu(x; 1, 0) / P*_lambda(x; 1, 0) equal delta_{mu, lambda}?

---

## Problem 5: O-Slice Connectivity

### The proof shape (as a wiring diagram)

```
O-slice-cells(H, rho_H^O) --> detection(Phi^H, tom-Dieck) --> connectivity(X, T_O)
```

Same three-node shape! Input: cell structure. Middle: detection theorem.
Output: connectivity criterion. The first arrow requires knowing the right
cells, the second is the categorical machinery.

### Reduction 1: G = C_2 (composable lemma)

G = C_2 has exactly two subgroups: {e} and C_2.
Transfer systems: trivial {e} or complete {e, C_2}. Only two cases.

**Complete case (known):** O-slice = regular slice (HHR).
- Slice cells: C_2+ wedge S^{n sigma} (sigma = sign representation)
  and S^{n(1+sigma)} = S^{n rho}
- Connectivity: Phi^e X is (2n-1)-connected AND Phi^{C_2} X is (n-1)-connected

**Trivial case:** O-slice = Postnikov.
- Only slice cell: S^n (trivial representation, only H = {e})
- Connectivity: underlying spectrum X^e is (n-1)-connected
- Phi^{C_2} is unconstrained

**Check:** These two cases should be easy to verify directly against the
HHR paper (Theorem 4.42 or equivalent). If our formula gives the wrong
connectivity bounds for C_2, the whole approach is wrong.

### Reduction 2: G = C_{p^2}, intermediate transfer system

G = C_{p^2} has three subgroups: {e}, C_p, C_{p^2}.
Transfer systems (interesting ones):
- T = {{e}, C_p} — transfers from e to C_p only, not to C_{p^2}

This gives an intermediate O-slice filtration. The O-slice cells are:
- G_+ wedge_{e} S^n = G_+ smash S^n (free cells)
- G_+ wedge_{C_p} S^{n rho_{C_p}^O}

Now what is rho_{C_p}^O? If the transfer system only includes the transfer
from e to C_p, then rho_{C_p}^O should be... the regular representation of
C_p? Or something smaller?

**This is where the uncertainty lives.** The definition of rho_H^O (the
O-regular representation) is exactly what we're not sure about. In the
C_{p^2} case with this particular T, we should be able to work it out
from first principles.

### Reduction 3: same shape, different category (reduce to known result)

The proof shape (cells --> detection --> connectivity) also appears in:

**Chromatic homotopy theory:** the chromatic filtration on spectra.
E(n)-local spectra are detected by K(n)-homology. The "cells" are
type-n complexes, the "detection" is the thick subcategory theorem.

**Motivic homotopy theory:** the effective slice filtration (Voevodsky).
Effective cells are Sigma^n_T HZ. Detection via motivic cohomology.

These are the same wiring diagram shape applied in different categories.
The HHR proof already generalized the motivic case to equivariant. Our
Problem 5 generalizes HHR to incomplete transfer systems. The question
is whether the detection step (tom Dieck splitting) still works when you
restrict to a subset of subgroups.

**Key lemma to verify:** Does the tom Dieck splitting for Phi^H still
compute [G_+ wedge_H S^V, X]_G correctly when H is in T_O? Yes — the
tom Dieck splitting is a statement about the geometric fixed point functor
and doesn't depend on the transfer system. It's a property of the
equivariant stable homotopy category, not of the N-infinity operad.

So the detection step is unconditional. The only thing that changes is
WHICH cells you use (determined by T_O) and hence which subgroups you
need to check.

---

## Composition Structure

Both problems have the same three-node shape:

    structure --> detection --> criterion

For Problem 3: exchange relations --> detailed balance --> stationary chain
For Problem 5: O-cells --> tom Dieck --> connectivity

In both cases:
- The **detection arrow** is the known/reliable part
- The **structure arrow** is where our uncertainty lives

The Polya reductions target the structure arrow:
- Problem 3: verify exchange relations at n=2 (a finite computation)
- Problem 5: verify O-cell definition at G=C_2 (two cases, both known)

If the structure arrow checks out on the simple cases, the detection arrow
composes it into the full answer.

---

## What Codex Should Verify

### For Problem 3 (computational)
1. Compute F*_{(a,b)}(x_1, x_2; q=1, t) explicitly
2. Compute the exchange relation T_1 F*_{(a,b)} at q=1
3. Check that the coefficient ratio gives a valid rate for detailed balance
4. Verify the t=0 degenerate case: F*_mu(x; 1, 0) = delta_{mu,lambda} · P*_lambda?

### For Problem 5 (structural)
1. Confirm: for G=C_2 complete case, does our formula match HHR Theorem 4.42?
2. Confirm: the tom Dieck splitting Phi^H(G_+ wedge_H S^V) = S^{dim V}
   holds independently of any transfer system
3. Clarify: what is rho_H^O for a specific incomplete T? (use C_{p^2} example)

---

## Codex Verification Notes (2026-02-11)

### Problem 3 (computational)

Status: **Partially verified with corrections**.

1. **Exchange/Hecke relations at q=1 are verified in primary source form.**  
   In the qKZ-family definition, the ASEP polynomials satisfy Hecke-generator relations
   (including the q=1 specialization), and the companion relation for the opposite inequality:
   - Definition 3.6, equations (3.5)-(3.7)
   - Remark 3.8, equation (3.8)
   Source: Ayyer-Martin-Williams (2024), arXiv:2403.10485.

2. **n=2 composable check can be done explicitly for restricted partition length 2.**  
   For a restricted partition of length 2, we necessarily have lambda=(a,0), so the two states are
   (a,0) and (0,a). From the t-PushTASEP definition, jump rates are 1/x_1 and 1/x_2 respectively,
   hence:
   - pi(a,0)=x_1/(x_1+x_2)
   - pi(0,a)=x_2/(x_1+x_2)
   Therefore pi(0,a)/pi(a,0)=x_2/x_1.  
   By Theorem 1.1 (same paper), this equals F_(0,a)(x;1,t)/F_(a,0)(x;1,t).

3. **Correction to Reduction 3 (`t=0` degeneration).**  
   The claim "stationary distribution concentrated on the sorted composition" is false for the
   ring PushTASEP model used here. Proposition 2.4 gives a non-degenerate stationary law
   already at t=0 in the single-species sector (weighted by products of x_i / elementary symmetric
   polynomial), not a delta mass at a single sorted state.

4. **Unresolved item.**  
   A closed-form general F*_(a,b)(x_1,x_2;1,t) formula and direct c/d extraction were not found
   in this pass; the verified reduction above gives a finite two-state rate ratio check in the
   restricted n=2 case actually induced by the benchmark constraints.

### Problem 5 (structural)

Status: **Core structural claims verified; one definition corrected**.

1. **C_2 complete-case slope check is consistent with modern slice criterion.**  
   Hill-Yarnall give:
   "X is slice n-connective iff for all H, Phi^H(X) is (n/|H|-1)-connective"
   (equivalently with ceilings in integer indexing).  
   For G=C_2 this gives the expected two-slope behavior (underlying vs C_2-fixed), matching the
   intended sanity check up to indexing normalization.
   Source: Hill-Yarnall (2017), arXiv:1703.10526.

2. **Detection via geometric fixed points is independent of transfer systems.**  
   Geometric fixed points are defined in the genuine equivariant stable category itself and used to
   produce equivalences (e.g. rational decomposition by geometric fixed points), with no dependence
   on an N_infinity transfer system parameter.
   Source: Wimmer (2019), arXiv:1905.12420 (Theorem 3.10 and surrounding setup).

3. **Correction: `rho_H^O` is not a standard object in N_infinity/transfer-system literature.**  
   In the Blumberg-Hill framework, transfer systems/indexing systems control **which admissible
   H-sets/transfers/norms exist**, not a modified "O-regular representation" replacing rho_H in the
   usual slice-cell templates.
   Source: Blumberg-Hill (2013), arXiv:1309.1750.

4. **Working clarification for the C_{p^2} intermediate transfer system.**  
   For T allowing e->C_p but excluding transfers to C_{p^2}, the robust interpretation is:
   - keep standard representation inputs for slice-cell constructions (regular-representation based),
   - restrict the subgroup-indexed cells checked/allowed by the transfer system.
   I.e. the uncertainty is primarily in admissible subgroup indexing, not in inventing a new
   representation rho_H^O.

### Sources

- https://arxiv.org/abs/2403.10485
- https://arxiv.org/abs/1703.10526
- https://arxiv.org/abs/1905.12420
- https://arxiv.org/abs/1309.1750
