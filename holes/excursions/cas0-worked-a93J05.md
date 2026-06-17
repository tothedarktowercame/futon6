# CAS-0 worked example #1 — apm-a93J05 (doubly-periodic entire ⇒ constant)

*The first empirical data point for CAS-0 (seed the cascade pattern pool via a worked
proof). Semi-manual / CPU, no GPU. Source proof:
`futon3c/data/apm-informal-proofs/apm-a93J05.md`. claude-1, 2026-06-17.*

## The proof, decomposed into reasoning steps

1. Let `P` = the fundamental parallelogram `{sω₁+tω₂ : s,t∈[0,1]}`.
2. `P` is compact; `f` continuous ⇒ `|f|` attains a max `M` on `P`.
3. Any `z ∈ ℂ` is `z = z₀ + mω₁ + nω₂` with `z₀∈P`, `m,n∈ℤ`.
4. `f(z)=f(z₀)` (periodicity), so `|f(z)|≤M` for all `z` ⇒ `f` bounded.
5. `f` bounded + entire ⇒ `f` constant (Liouville). ∎

## Each step → a `math-informal` pattern (all matched; none missing for this proof)

| step | pattern (math-informal) | role |
|---|---|---|
| 1 | **construct-auxiliary-object** | introduce `P`, the fundamental domain |
| 2 | **reduce-to-known-result** (EVT) + **estimate-by-bounding** | continuous-on-compact attains max → `|f|≤M` on `P` |
| 3 | **quotient-by-irrelevance** | `z` mod the period lattice — translations are "irrelevant" to `f` |
| 4 | **local-to-global** | the bound on `P` (local) extends to all of `ℂ` (global) via periodicity |
| 5 | **reduce-to-known-result** (Liouville) | bounded + entire → constant |

**Coverage result:** 5/5 steps map to the existing 36 `math-informal` patterns — **no new
pattern needed for this proof**. (One proof; the gaps will surface on harder ones.)

## The cascade induces a WIRING and a SORRY (the Rank-D goal, on a real proof)

**Wiring** (the patterns' *conclusions* chain into the argument DAG):
```
construct-aux(P)  ──▷  reduce-to-known(EVT)[P compact, f cont]  ──▷  |f|≤M on P
                                                                       │
                       quotient-by-irrelevance(lattice)[z ≡ z₀∈P]  ───┤
                                                                       ▽
                       local-to-global[bound on P + periodicity]  ──▷  f bounded on ℂ
                                                                       ▽
                       reduce-to-known(Liouville)[bounded, entire]  ─▷  f constant ∎
```

**Sorry** (— the key finding): **the residual sorries are exactly the matched patterns'
`HOWEVER` clauses left undischarged.** Each pattern names its own proof obligation, and
the informal proof *asserts* the conclusions without discharging them:

| sorry | comes from pattern | the obligation |
|---|---|---|
| S1 `P` is compact | construct-auxiliary-object | `P` = continuous image of `[0,1]²` |
| S2 `ℂ = ⊔(z₀ + Λ)` | quotient-by-irrelevance's HOWEVER ("verify well-defined on equiv classes") | `ω₁,ω₂` ℝ-independent ⇒ they tile `ℂ` |
| S3 `f(z)=f(z₀)` | local-to-global's HOWEVER ("verify the pieces patch") | iterate both periods, `m,n∈ℤ` |

So pattern-matching does **both** Rank-D jobs at once: the patterns' `THEN`/conclusions
build the **wiring**, and their `HOWEVER` clauses generate the **sorry list**. That's the
mechanism — "induce a sorry + a wiring that can then be checked" — demonstrated on a real proof.

## What this teaches (empirically, for the open questions)

- **Q3 (topology vocabulary):** a proof's "shape" *is* its sequence of matched patterns
  (here: aux-object → reduce-to-known → quotient → local-to-global → reduce-to-known).
  `select` = which patterns match a proof's steps. The vocabulary is the **pattern set
  itself**, not a separate hand-authored taxonomy — confirming the empirical/by-not-fiat call.
- **Q5 (deterministic vs judge):** matching a step to a pattern + reading off its HOWEVER
  is largely **deterministic** once we can segment the proof into steps; the *judgement* is
  "does this step really instantiate this pattern?" (an LLM-as-judge / rung-3 spot, but
  bounded — verify a proposed match, like the adversarial-verify pattern).
- **Candidate refinement (not a new pattern):** `reduce-to-known-result` fired **twice**
  with different named-theorem parameters (EVT, Liouville) — the cited theorem is a **slot**
  of the pattern, not a new pattern. Suggests patterns carry a `:cites` parameter.
- **Pool signal:** 1 proof, 0 new patterns needed — encouraging, but we need the *next*
  few proofs to find where the 36 run out (a construction-heavy proof like a96J01, an
  induction, a diagram chase) before we know the pool is "enough to form cascades."

## Next
- Work proof #2 (a different shape — e.g. the construction `a96J01`, or an induction) to
  find the first pattern the 36 **don't** cover → the first genuinely-new `.flexiarg`.
- After ~3–4 proofs: the recurring step→pattern→(wiring,sorry) mechanism is the spec for
  CAS-SEL-1; the patterns we wrote seed the pool.
