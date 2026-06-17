# CAS-0 worked example #4 — apm-a96J04 (AC monotone maps null sets to null sets)

*Fourth data point. Chosen as an **ε-δ / measure-theory** proof — a region #1–#3 never
touched. It added the **third new pattern** and surfaced a structural observation about the
pool. CPU/semi-manual. Source: `futon3c/data/apm-informal-proofs/apm-a96J04.md`.
claude-1, 2026-06-17.*

## The proof

`f:[a,b]→ℝ` absolutely continuous + monotone increasing; `E⊆[a,b]`, `m(E)=0` ⇒ `m(f(E))=0`.
1. **Unfold AC:** ∀ε>0 ∃δ>0 s.t. disjoint intervals with `Σ(bₖ−aₖ)<δ` ⇒ `Σ|f(bₖ)−f(aₖ)|<ε`.
2. **Monotone:** `|f(bₖ)−f(aₖ)| = f(bₖ)−f(aₖ)`; `(aₖ,bₖ)` maps into an interval of that length.
3. **Unfold `m(E)=0`:** cover `E` by disjoint open intervals with `Σ(bₖ−aₖ)<δ`.
4. `f((aₖ,bₖ)) ⊆ [f(aₖ),f(bₖ)]` ⇒ `f(E) ⊆ ⋃ₖ[f(aₖ),f(bₖ)]`.
5. **Bound:** `m*(f(E)) ≤ Σ(f(bₖ)−f(aₖ)) < ε`.
6. **ε arbitrary** ⇒ `m(f(E)) = 0`.

## Step → pattern (2 existing + **1 new**)

| step | pattern | status |
|---|---|---|
| 1,2,3 unfold AC / monotone / null-set | **unfold-the-definition** (×3) | existing ✓ |
| 5 `m*(f(E)) ≤ Σ lengths` (countable subadditivity) | **estimate-by-bounding** | existing ✓ |
| 6 `≤ ε ∀ε ⇒ = 0` | **epsilon-of-room** | **NEW — written** |

**The gap.** The closing move — prove `m*(f(E)) ≤ ε` for *every* ε, then let ε→0 — is the
single most characteristic manoeuvre of analysis, and the 36 had no pattern for it. I checked
the four nearest: `estimate-by-bounding` *produces* a bound but doesn't model the
∀ε→limit collapse; `optimise-a-free-parameter` *tunes* ε for the sharpest bound (opposite —
here ε is sent to 0, not optimised); `show-both-inequalities` is the ≤/≥ wrapper (it's how
`=0` is finished, but not the ε-collapse itself); `exhaustion-as-theorem` is an obstruction
meta-pattern. So I wrote **`math-informal/epsilon-of-room`** (`[🤏/微]`, "Give Yourself an
Epsilon of Room", registered): weaken a sharp goal (`X=0`, `X≤Y`, a limit) to the ε-slack
version reachable from an ε–δ budget (continuity, AC, measure zero, convergence), prove that
for arbitrary ε, then let ε→0.

## Wiring + sorry (mechanism holds a fourth time)

```
unfold-def(AC) ─▷ ε–δ budget        unfold-def(m(E)=0) ─▷ cover, Σlen<δ
unfold-def(monotone) ─▷ f((aₖ,bₖ)) ⊆ [f(aₖ),f(bₖ)], length f(bₖ)−f(aₖ)
            ▽
estimate-by-bounding(countable subadditivity) ─▷ m*(f(E)) ≤ Σ(f(bₖ)−f(aₖ)) < ε
            ▽
epsilon-of-room(ε arbitrary) ─▷ m(f(E)) = 0 ∎
```

**Sorry (= undischarged HOWEVERs — 4th confirmation):**
| sorry | from pattern | obligation |
|---|---|---|
| S1 the δ from AC fits the cover (δ depends on ε only, not on `f(E)`) | **epsilon-of-room**'s HOWEVER ("nothing chosen depends on the bounded quantity") | AC gives δ(ε); the cover is then chosen with `Σ<δ` |
| S2 `m*(⋃) ≤ Σ m*` | **estimate-by-bounding**'s HOWEVER | countable subadditivity of outer measure |
| S3 which characterisation of AC / null set | **unfold-the-definition**'s HOWEVER ("choose the right unfolding") | AC's ε–δ form (not ∫f′); null = open-cover form |

## What #4 adds — two findings

**Finding A — analysis was the under-covered region; the discovery rate hasn't saturated.**
Tally now **+0, +1, +1, +1**. #1 (complex analysis) was 0-new *because it reduced to a named
theorem* (Liouville) — the original 36 cover "which big theorem to cite" well. But the
*working idioms* of analysis — make the pieces independent (#2), count over a decomposition
(#3), give yourself ε of room (#4) — were all missing. So the gap isn't random: the original
36 skew toward **strategy** ("which approach": reduce-to-known, induct, contradict, construct)
and were thin on **execution idioms** ("how to run the approach"). Each new analysis shape is
still adding ~1 pattern — the pool is **not yet saturated** at the analysis frontier.

**Finding B — but the *mechanism* is saturated (4/4), so CAS-SEL-1 is specifiable now.**
Across all 4 shapes (reduce-to-known / construction / induction+counting / ε-δ-measure), the
procedure is identical and stable: segment → match each step to a pattern (family + `:cites`
slot) → wiring = patterns' conclusions, sorry = patterns' undischarged HOWEVERs. The spec does
**not** depend on pool completeness — the pool grows demand-driven during RUN (exactly the
prior-art seeding loop). So: write CAS-SEL-1 on these 4, and let proofs keep minting patterns.

## Running corpus (4 proofs)
| # | proof | shape | new patterns |
|---|---|---|---|
| 1 | a93J05 | reduce-to-known (Liouville) | — |
| 2 | a96J01 | construction (disjoint bumps) | separate-into-independent-pieces |
| 3 | b97J01 | induction + counting (p-groups) | count-over-a-decomposition |
| 4 | a96J04 | ε-δ / measure (AC null sets) | epsilon-of-room |

Pool **36 → 39**. Mechanism 4/4. **Recommend: write CAS-SEL-1's spec spike on this corpus.**
