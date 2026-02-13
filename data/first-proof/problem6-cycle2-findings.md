# P6 Cycle 2: K_n Exact Formula and Near-Extremality

Date: 2026-02-13
Agent: Claude (Cycle 2), building on Codex Cycle 1

## Key Discovery: Exact K_n d̄ Formula

Derived from eigenstructure analysis of M_t and F_t for K_n:

    d̄_Kn(t) = (t-1)/(m₀ε - t) + (t+1)/(m₀ε)

**Derivation:**
- M_t = (1/n)L_{K_t} has eigenvalue t/n (mult t-1) and 0 (mult n-t+1)
- B_t = (εI-M_t)⁻¹ has eigenvalue 1/(ε-t/n) on S_t internal, 1/ε on rest
- F_t = (1/n)L_{K_{t,n-t}}: cross-edge Laplacian
- tr(P_S · F_t) = (t-1)(n-t)/n (projection onto M_t's nonzero eigenspace)
- tr(P_rest · F_t) = (t+1)(n-t)/n
- tr(B_t F_t) = (t-1)(n-t)/(n(ε-t/n)) + (t+1)(n-t)/(nε)
- d̄ = tr(B_t F_t)/(n-t) = (t-1)/(nε-t) + (t+1)/(nε)

**Verification:** Matches observed d̄ for K_n exactly to machine precision.
Example: K_80, ε=0.5, t=12: formula gives 14/28 + 16/40 = 0.7179. Observed: 0.7179. ✓

**At horizon T = εm₀/3:**
```
d̄_Kn(T) = (εm₀/3 - 1)/(2εm₀/3) + (εm₀/3 + 1)/(εm₀)
         → (1/3)/(2/3) + (1/3)/1 = 1/2 + 1/3 = 5/6   as m₀ → ∞
```

**5/6 ≈ 0.833 < 1.** This confirms the known K_n result (Section 5e of solution.md).

## Attack D Result: K_n Is Nearly Extremal

Tested d̄ / d̄_Kn(t,m₀,ε) across all graphs:

| Family   | n_steps | mean  | max    | ≤1? |
|----------|---------|-------|--------|-----|
| K        | 91      | 1.000 | 1.000  | YES |
| Barbell  | 47      | 0.966 | 0.988  | YES |
| ER       | 80      | 0.921 | 1.005  | NO  |

**Global max ratio: 1.0047** (ER_60_p0.5, eps=0.5, t=6, d̄=0.4437)

K_n is extremal within 0.5%. The single ER overshoot is likely a
finite-size effect (non-uniform leverage in the independent set).

**Implication:** If d̄ ≤ (1+δ)·d̄_Kn with δ < 1/5, then at the horizon:
d̄ ≤ (1+δ)·(5/6) < 1. The observed δ = 0.005 gives d̄ ≤ 0.838.

## Attack A Result: Log-Det Potential

Φ(t) = log det(εI - M_t) tracks barrier consumption.

- Per-step decrease: ΔΦ ≈ -step_lev/(ε-‖M_t‖) (ratio 0.5–0.8)
- Total budget usage: ≤ 7.2% across all instances
- The greedy uses a tiny fraction of the barrier budget

This means ‖M_t‖ stays far from ε throughout. The log-det potential could
provide a formal ‖M_t‖ bound if needed, but the K_n extremality path
(Attack D) is more direct.

## Proof Strategy for Closure

### Path 1: Prove K_n extremality (strongest)
Show d̄_G(t) ≤ d̄_Kn(t,m₀,ε) for all graphs G. This would give:
- At horizon: d̄ ≤ 5/6 < 1
- |S| = εm₀/3 ≥ ε²n/9
- Universal c = ε/9 (or c = ε/3 matching K_n if I₀ = V)

**Approach:** The K_n formula has the "most uniform" leverage structure
(τ_e = 2/n for all edges). For non-uniform leverage, the amplification
is bounded by a convexity argument: the function f(λ) = 1/(ε-λ) is convex,
so Σ f(λ_i)w_i ≤ f(Σ λ_i w_i) when weights are uniform... Actually this
goes the wrong way. Need a different argument.

### Path 2: Tighten the horizon (easiest)
Reduce T from εm₀/3 to εm₀/4. Then:
- d̄_Kn(T) → 3/8 + 3/8 = 3/4 at large m₀. Wait: (1/4)/(3/4) + (1/4)/1 = 1/3 + 1/4 = 7/12 ≈ 0.583.
- With 0.5% overshoot: 0.586 < 1. Huge margin.
- Trade-off: |S| = εm₀/4 instead of εm₀/3 (worse constant but still ε²n).

### Path 3: Combined scalar + K_n bound
Show d̄ ≤ d̄_Kn + O(1/(m₀ε)). The O(1/(m₀ε)) correction vanishes for large
instances and can be absorbed by the 17% margin at the K_n horizon.

## Next Steps for Codex (Cycle 2)

1. **Verify the K_n exact formula derivation** — check the eigenstructure
   computation for M_t and F_t in K_n.

2. **Test K_n extremality at larger n** — run n=200, 500 to see if the
   0.5% overshoot persists or shrinks.

3. **Investigate the ER outlier** — is the 1.005 ratio due to non-uniform
   leverage in I₀, or a genuine structural effect?

4. **Test the convexity conjecture** — is there a Schur-convexity argument
   that the K_n leverage structure maximizes d̄?

## Files

| File | Purpose |
|------|---------|
| `scripts/verify-p6-cycle2-logdet-dbar.py` | Cycle 2 diagnostic script |
| `data/first-proof/problem6-cycle2-results.json` | Machine-readable results |
| `data/first-proof/problem6-dbar-mechanism.md` | Cycle 1 mechanism discovery |
| `data/first-proof/problem6-codex-cycle1-verification.md` | Codex Cycle 1 results |
