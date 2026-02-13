# Codex Review: Problem 6 Proof — Epsilon-Light Subsets

## What This Is

An attempt at Problem 6 of the **First Proof** benchmark (Spielman's problem):
for any graph G=(V,E,w) and eps in (0,1), find S with |S| >= c*eps*n and
L_S <= eps*L (Loewner order on Laplacians).

The proof is ~90-95% complete. We need Codex to:
1. **Verify** the proved components (especially the partial averages argument)
2. **Identify** any errors in the chain of reasoning
3. **Attempt to close** Sub-gap 2 (M_t != 0 amplification bound)

## File Map

| File | What it contains |
|------|-----------------|
| `problem6-proof-draft.md` | **THE MAIN DOCUMENT.** Near-final proof, 6 sections. Read this first. |
| `problem6-solution.md` | Earlier working notes (messier, more detail on dead ends) |
| `problem6-gpl-h-attack-paths.md` | Three attack paths tried for the gap (historical) |
| `../scripts/verify-p6-dbar-bound.py` | Verification: 440/440 steps pass dbar < 1 |
| `../scripts/verify-p6-gpl-h.py` | Base module for graph construction + leverage computation |

## The Proof Architecture

```
Foster's theorem (avg leverage degree < 2)
    |
    v
Partial averages inequality (sum of T smallest < 2T)
    |
    v
dbar < (2/3)/(1 - eps/3) < 1  [at M_t = 0]    ... PROVED
    |
    v
Pigeonhole: min trace <= dbar < 1
    |
    v
PSD trace bound: ||Y|| <= tr(Y)
    |
    v
exists v with ||Y_t(v)|| < 1 => barrier greedy continues
    |
    v
|S| = eps*m/3 >= eps^2*n/9 => Problem 6 solved
```

## What's Proved (verify these)

### 1. Reformulation (Section 1)
- L_S <= eps*L iff ||M_S|| <= eps where M_S = sum X_e (normalized edge matrices)
- Standard, should be correct.

### 2. Heavy/light + Turan (Section 2a-2b)
- At most (n-1)/eps heavy edges (from Foster)
- Turan gives I_0 with |I_0| >= eps*n/3, independent in heavy graph
- Standard. The Turan bound uses alpha >= n^2/(2m+n).

### 3. Foster bound on I_0 (Section 2c) — KEY LEMMA
- Claim: sum of leverage degrees within I_0 <= 2(|I_0| - 1)
- Proof uses: (a) Rayleigh monotonicity (R_eff in G <= R_eff in induced subgraph),
  (b) Foster on the induced subgraph
- **VERIFY THIS CAREFULLY.** The Rayleigh monotonicity direction matters.
  Removing edges INCREASES effective resistance. So tau_e in G <= tau_e in
  the induced subgraph? The induced subgraph has FEWER edges, so R_eff is
  LARGER, so tau_e^{induced} = w_e * R_eff^{induced}(e) >= w_e * R_eff^G(e) = tau_e.
  This seems right but please double-check.

### 4. Partial averages bound (Section 4c) — THE BREAKTHROUGH
**Theorem 4.2:** For the min-ell greedy at M_t = 0:
  dbar_t <= (2/3)/(1 - eps/3) < 1.

The proof chain:
1. Foster: avg ell < 2(m-1)/m < 2
2. Partial averages: sum_{k=1}^t ell_{(k)} <= t * avg(ell) < 2t(m-1)/m
3. dbar = (1/(eps*r_t)) * sum ell_u <= 2t(m-1)/(eps*m*r_t)
4. At t=T=eps*m/3, r_t >= m(1-eps/3): dbar < (2/3)/(1-eps/3)
5. For eps < 1: (2/3)/(1-eps/3) < 1 (check: at eps=1, equals 1; strict ineq for eps<1)

**VERIFY:** Step 3 uses ell_u (leverage degree toward ALL of I_0). But dbar
actually uses ell_u^R (leverage toward R_t = remaining candidates). Since
R_t subset I_0, we have ell_u^R <= ell_u, so the bound is valid (but conservative).
Is this correct?

### 5. K_n exact formula (Section 3)
  dbar = 2t/(n*eps)   [at M_t = 0]
  dbar(K_k, t) = (t-1)/(k*eps-t) + (t+1)/(k*eps)  [with M_t != 0]
  Limit as k->inf at T=eps*k/3: dbar -> 5/6 < 1

## The One Remaining Gap (try to close this)

### Sub-gap 2: M_t != 0 amplification

**The problem:** After step 1, M_t = sum_{e internal to S_t} X_e may be nonzero.
The barrier matrix H_t = eps*I - M_t has H_t^{-1} >= (1/eps)*I, which amplifies
traces: tr(H_t^{-1} C_t(v)) >= tr(C_t(v))/eps.

**The crude bound:**
  dbar_t <= (eps/(eps - ||M_t||)) * dbar_t^{M=0}

This requires ||M_t|| < eps*(1 - dbar^{M=0}). At worst case dbar^{M=0} = 0.741:
need ||M_t|| < 0.078. But K_100 has ||M_t|| = 0.09 at the last step.
So the crude bound FAILS.

**Why it's still true (empirically):**
The bound tr(H^{-1}A) <= ||H^{-1}|| * tr(A) is conservative because it assumes
A is aligned with H's minimum eigenvector. In practice, the cross-edge matrix
A_t and M_t point in different directions, so the amplification is milder.

**Empirical data:**
- Max amplification ratio (dbar/dbar_m0): 1.30 (Barbell_60)
- Max dbar with M_t != 0: 0.714 (K_100)
- ALL tested (30+ graph families): dbar < 1

**Ideas for closure:**

(a) **Neumann series approach:** Write H_t^{-1} = (1/eps) sum_k (M_t/eps)^k.
    Then dbar = dbar^{M=0} + (1/(eps^2 r_t)) sum_v tr(M_t C_t(v)) + ...
    Bound the correction terms using structure of M_t and C_t(v).

(b) **Self-compensation:** The cross-leverage sum decreases as internal
    leverage grows: sum ell_u^R = sum ell_u - 2*tr(M_t). So dbar^{M=0}
    itself drops as M_t grows, partially offsetting the amplification.
    Can this cancellation be made precise?

(c) **Reduce T:** Run greedy for T = eps*m/4 instead of eps*m/3. Then
    dbar^{M=0} <= (2/4)/(1-eps/4) = 0.54 (at eps=0.3), leaving room for
    amplification up to 1.85x. Cost: |S| = eps*m/4 instead of eps*m/3.
    The amplification factor is empirically <= 1.3, so 0.54 * 1.3 = 0.70 < 1.
    But can we PROVE amplification <= 1.85?

(d) **Independence structure:** For many graph families (K_{a,b}, cycles,
    grids, sparse graphs), the min-ell vertices form an independent set
    in the original graph, so M_t = 0 exactly. Can we prove this holds
    for ALL graphs? (Probably not universally, but maybe for "most" of the
    greedy steps.)

(e) **BSS potential function:** Track phi_t = log det(eps*I - M_t) or
    phi_t = sum 1/(eps - lambda_i(M_t)). The BSS barrier method bounds
    potential change per step. If each step's potential increase is bounded,
    the barrier survives T steps even without bounding dbar directly.

(f) **Random sampling:** Skip the greedy. Sample vertices of I_0 with
    probability eps/3. Matrix Chernoff/Tropp for E[M_S] = p^2 * sum X_e.
    ||E[M_S]|| <= (eps/3)^2. Need concentration to show ||M_S|| < eps
    with positive probability.

## Verification Checklist

Please verify each item and mark CORRECT / ERROR / GAP:

- [ ] Section 1: L_S <= eps*L iff ||M_S|| <= eps
- [ ] Lemma 2.1: At most (n-1)/eps heavy edges
- [ ] Lemma 2.2: Turan gives |I_0| >= eps*n/3
- [ ] Lemma 2.3: Foster bound sum ell_v <= 2(|I_0|-1) — check Rayleigh direction
- [ ] Corollary 2.4: avg ell < 2
- [ ] Lemma 2.5: Partial averages inequality
- [ ] Claim 2.6 + proof: dbar < 1 implies greedy continues (pigeonhole + PSD)
- [ ] Theorem 3.1: K_n proof with c = 1/3
- [ ] Theorem 4.2: dbar < (2/3)/(1-eps/3) at M_t=0 — check ell vs ell^R issue
- [ ] Sub-gap 2: Any idea for a formal amplification bound?

## Key Identities to Check

1. tau_e = w_e * R_eff(e), sum tau_e = n-1 (Foster)
2. ||Y|| <= tr(Y) for PSD Y (used in pigeonhole step)
3. Rayleigh monotonicity: removing edges increases effective resistance
4. sum_{e internal to J} tau_e <= |J|-1 (Foster on induced subgraph)
5. (2/3)/(1-eps/3) < 1 for eps in (0,1) — check boundary: at eps=1, = 1 exactly
