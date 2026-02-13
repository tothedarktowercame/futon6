# Codex Review Handoff: Problem 6 — Epsilon-Light Subsets

## What This Is

An attempt at Problem 6 of the **First Proof** benchmark
(arxiv.org/abs/2602.05192 — Spielman's problem). Given a weighted graph
G=(V,E,w) with Laplacian L, find S ⊆ V with |S| ≥ cεn such that the
induced Laplacian satisfies L_S ≤ εL (Loewner order).

We have a **90-95% complete proof** with one quantitative sub-gap remaining.

## Files

| File | What it is |
|------|-----------|
| `problem6-proof-draft.md` | **THE MAIN DOCUMENT.** Near-final proof, 6 sections, theorem-proof format. |
| `problem6-solution.md` | Earlier working notes (messier, more history) |
| `problem6-gpl-h-attack-paths.md` | Three attack paths for the gap (pre-partial-averages) |
| `../scripts/verify-p6-dbar-bound.py` | Verification: 440/440 steps pass d̄ < 1 |
| `../scripts/verify-p6-gpl-h.py` | Graph construction + leverage computation module |

## The Proof Structure

```
Foster's theorem (avg leverage < 2)
    + Partial averages inequality (sum of T smallest ≤ T × avg)
        → Σ_{k=1}^T ℓ_{(k)} < 2T
            → d̄ < (2/3)/(1-ε/3) < 1  [at M_t = 0]
                → ∃v: ||Y_t(v)|| ≤ tr(Y_t(v)) ≤ d̄ < 1  [PSD + pigeonhole]
                    → barrier greedy continues for T = εm/3 steps
                        → |S| ≥ ε²n/9, L_S ≤ εL  ∎
```

## What's Proved (Codex: please verify these)

### 1. Reformulation (§1)
- X_e = L^{+/2}(w_e b_e b_e^T)L^{+/2}, τ_e = tr(X_e) = w_e R_eff(e)
- L_S ≤ εL ⟺ ||Σ_{e∈E(S)} X_e|| ≤ ε
- Foster: Σ τ_e = n-1

**Risk: LOW.** Standard spectral graph theory.

### 2. Heavy/light + Turán (§2a-2b)
- Heavy graph has ≤ (n-1)/ε edges (from Foster)
- Turán: α(G_heavy) ≥ n²/(2(n-1)/ε + n) ≥ εn/3
- So ∃ I_0 with |I_0| ≥ εn/3, all internal edges ε-light

**Risk: LOW.** Standard combinatorics.

### 3. Leverage degrees + Foster bound on I_0 (§2c)
- ℓ_v = Σ_{u ∈ I_0, u~v} τ_{uv} (leverage degree within I_0)
- Claim: Σ_v ℓ_v = 2·Σ_{e internal} τ_e ≤ 2(|I_0|-1)
- Uses: Rayleigh monotonicity (τ_e in G ≤ τ_e in induced subgraph)
  then Foster on induced subgraph

**Risk: MEDIUM.** The Rayleigh monotonicity step needs checking.
Specifically: for e internal to I_0, is the effective resistance of e
in G always ≤ the effective resistance in the induced subgraph G[I_0]?
This should follow from Rayleigh's monotonicity principle (adding edges
can only decrease resistance), since G has more edges than G[I_0].

### 4. Partial averages bound (§2d + §4c) — THE KEY NEW RESULT
- Lemma 4.1: avg of T smallest values ≤ overall average (trivial)
- Combined with Foster (avg ℓ < 2): Σ_{k=1}^T ℓ_{(k)} < 2T(m-1)/m
- Theorem 4.2: d̄_t ≤ (2/3)/(1-ε/3) < 1 at M_t = 0

**Risk: LOW for the inequality itself, MEDIUM for the d̄ formula.**
The formula d̄_t = Σ_{u∈S_t} ℓ_u / (ε·r_t) at M_t = 0 needs checking.
Specifically: when H_t = εI, Y_t(v) = (1/ε)C_t(v), so
tr(Y_t(v)) = tr(C_t(v))/ε = (Σ_{u∈S_t} τ_{uv})/ε. Then
d̄_t = (1/r_t)Σ_{v∈R_t} tr(Y_t(v)) = (1/(εr_t))Σ_{v∈R_t}Σ_{u∈S_t} τ_{uv}
     = (1/(εr_t))Σ_{u∈S_t} ℓ_u^R  where ℓ_u^R ≤ ℓ_u.

Check: does the bound use ℓ_u or ℓ_u^R? The proof uses ℓ_u (upper bound),
which is valid but conservative. The actual d̄ uses ℓ_u^R (edges to R_t only).

### 5. Pigeonhole + PSD trace bound (§2f)
- ||Y|| ≤ tr(Y) for PSD Y
- min_v tr(Y_t(v)) ≤ d̄_t
- If d̄_t < 1: ∃v with ||Y_t(v)|| < 1

**Risk: VERY LOW.** Three lines, each elementary.

### 6. K_n complete proof (§3)
- τ_e = 2/n, ℓ_v = 2(n-1)/n, d̄_t = 2t/(nε)
- At T = εn/3: d̄ = 2/3 < 1
- Including M_t ≠ 0: exact formula d̄ → 5/6 < 1

**Risk: LOW.** Verified numerically to 6 decimal places.

## The One Remaining Gap (Codex: please focus here)

### Sub-gap: M_t ≠ 0 amplification (§4e)

**The problem:** Theorem 4.2 proves d̄ < 1 when M_t = 0 (no accumulated
barrier matrix). After adding vertices with mutual edges, M_t ≻ 0 and
the barrier headroom H_t = εI - M_t has H_t^{-1} ≻ (1/ε)I, which
amplifies traces:

    tr(Y_t(v)) = tr(H_t^{-1} C_t(v)) ≥ tr(C_t(v))/ε = (M_t=0 value)

The crude bound gives amplification ≤ ε/(ε - ||M_t||), which is too
loose to close the gap (would need ||M_T|| < 0.078 for ε=0.3, but
K_100 has ||M_T|| = 0.09).

**What we know empirically:**
- Max amplification ratio: 1.30 (Barbell_60)
- Max d̄ with amplification: 0.714 (K_100)
- Critical threshold: amplification must be < 1/0.741 ≈ 1.35
- 440/440 barrier greedy steps pass across all tested graph families

**Graph families where M_t = 0 (gap already closed):**
- K_{a,b} with a ≠ b (min-ℓ greedy picks from larger independent side)
- Cycles, grids, sparse graphs (I_0 vertices too far apart for edges)
- Any graph where min-ℓ vertices form an independent set

**What would close it (ideas for Codex to evaluate):**

**(A) Tighter trace inequality.** Show tr(H_t^{-1} A_t) < ||H_t^{-1}|| · tr(A_t)
by exploiting directional mismatch between M_t and A_t. The cross-edge
matrix A_t and internal-edge matrix M_t live in "different directions" because
they involve disjoint edge sets.

**(B) BSS potential function.** Track φ_t = tr(H_t^{-1}) or log det(H_t).
BSS-style barrier arguments bound the potential change per step. If the
min-ℓ greedy keeps ||Y_t(v)|| bounded, the potential argument may directly
give ||M_T|| < ε without needing d̄ < 1 at intermediate steps.

**(C) Reduced horizon.** Run greedy for T = εm/4 instead of εm/3.
Then d̄^{M=0} ≤ 0.54 (for ε=0.3), leaving room for amplification up to
1.85 (vs empirical max 1.30). Cost: |S| shrinks by 25%.
Could combine with a provable amplification bound.

**(D) Random sampling bypass.** Sample vertices with probability p = ε/3.
Use matrix concentration (Tropp/Oliveira) to bound ||M_S||. Avoids the
greedy amplification issue entirely. E[M_S] = p²·(internal edge sum) ≤ p²Π,
so ||E[M_S]|| ≤ p² = ε²/9 ≪ ε. Need to handle dependencies (both
endpoints must be sampled).

**(E) Foster on the rescaled system.** At step t, define rescaled leverage
scores τ̃_e = tr(H_t^{-1} X_e). Does a "Foster-like" identity hold for
these? If Σ τ̃_e is bounded, the d̄ bound at M_t ≠ 0 follows.

## Specific Verification Requests

1. **Check §2c (Foster on I_0).** Is Rayleigh monotonicity applied correctly?
   Specifically: for an edge e = {u,v} with u,v ∈ I_0, is
   R_eff^G(e) ≤ R_eff^{G[I_0]}(e)? (G has MORE edges than G[I_0],
   so resistance should be LESS in G. This is the right direction.)

2. **Check Theorem 4.2 bound.** The chain:
   Σ ℓ_{(k)} < 2T(m-1)/m → d̄ < 2T(m-1)/(mεr_t) → d̄ < (2/3)/(1-ε/3) < 1.
   Is each step correct? Is the bound tight (approaching 1 as ε → 1)?

3. **Evaluate approaches (A)-(E) above.** Which is most promising for
   closing the M_t ≠ 0 gap? Any approach we're missing?

4. **Check the K_n exact formula.** d̄(K_k,t) = (t-1)/(kε-t) + (t+1)/(kε).
   Is this derivable from the structure of K_n? Does it → 5/6 as k → ∞?

5. **Is the ε² bottleneck real?** The proof gives |S| ≥ ε²n/9. Can we
   do better? The problem asks for |S| ≥ cεn. For fixed ε this is fine
   (c = ε/9), but the quadratic dependence on ε is worth flagging.

## Key Identities (Quick Reference)

```
L_S ≤ εL  ⟺  ||M_S|| ≤ ε           (spectral equivalence)
τ_e = w_e · R_eff(e)                  (leverage = weight × resistance)
Σ_e τ_e = n - 1                       (Foster's theorem)
||Y|| ≤ tr(Y)  for PSD Y              (spectral ≤ trace)
min f(v) ≤ avg f(v)                    (pigeonhole)
avg_{v∈I_0} ℓ_v < 2                   (Foster on induced subgraph)
d̄ ≤ (2/3)/(1-ε/3) < 1  at M_t=0     (partial averages + Foster)
```
