# CAS-0 worked example #3 — apm-b97J01 (finite p-groups: nontrivial center & nilpotent)

*Third data point. Chosen as an **induction** probe (does the induction schema need a finer
hole than the one coarse pattern?) over a multi-part, genuinely hard proof. Two findings:
the induction pattern **covered cleanly — no finer hole needed** — and part (c) yielded the
**second new pattern** (a counting argument). CPU/semi-manual. Source:
`futon3c/data/apm-informal-proofs/apm-b97J01.md`. claude-1, 2026-06-17.*

## The proof (parts c+d are the real math)

- **(b) examples** separating abelian ⊊ nilpotent ⊊ solvable ⊊ all: `A₅` (simple ⇒ derived
  series constant ⇒ not solvable), `S₃` (solvable, trivial center ⇒ not nilpotent).
- **(c) `|G|=pⁿ ⇒ Z(G) ≠ {e}`.** Class equation `pⁿ = |Z(G)| + Σᵢ[G:C_G(xᵢ)]`; each
  non-central class size `[G:C_G(xᵢ)]` is a divisor of `pⁿ` that is `>1`, so `p | ` it; and
  `p | pⁿ`; therefore `p | |Z(G)|`, forcing `|Z(G)| ≥ p > 1`.
- **(d) `|G|=pⁿ ⇒ G` nilpotent.** Upper central series `Z₀ ⊊ Z₁ ⊊ …`, `Z_{i+1}/Z_i =
  Z(G/Z_i)`. While `Z_i ≠ G`, `G/Z_i` is a smaller nontrivial p-group, so by (c) its center is
  nontrivial ⇒ `Z_i ⊊ Z_{i+1}` strictly. A strictly increasing chain in finite `G` terminates,
  and only at `G` ⇒ `G` nilpotent. (Footnote does it as plain strong induction on `n`.)

## Step → pattern (6/7 existing; **1 new** — and the induction pattern sufficed)

| step | pattern | status |
|---|---|---|
| (b) separating examples | **construct-an-explicit-witness** | existing ✓ |
| (b) `A₅` commutator is `{e}` or `A₅` | **split-into-cases** (minor) | existing ✓ |
| (c) class equation / orbit-stabilizer | **reduce-to-known-result** (`:cites`) | existing ✓ |
| (c) **class-equation divisibility** (the "Key Insight") | **count-over-a-decomposition** | **NEW — written** |
| (c) `\|Z(G)\| ≥ p` | **estimate-by-bounding** (minor) | existing ✓ |
| (d) build the upper central series | **construct-auxiliary-object** | existing ✓ |
| (d) chain terminates / strong induction on order | **induction-and-well-ordering** | existing ✓ |

**Finding A — the induction schema needs no finer hole (the question we set out to test).**
`induction-and-well-ordering` covered (d) cleanly: induct on `n = ` order (the footnote's
"by induction on n"), with the strictly-increasing-chain-in-a-finite-group = the
well-founded-termination instance. The "base case *is* the inductive engine" framing (apply
(c) to each quotient) is a nice *instance* of the pattern, **not** a missing schema. So: a
real, non-trivial induction did not demand an induction-schema sub-pattern. One data point,
but it argues against pre-building one speculatively.

**Finding B — the second new pattern: counting.** Part (c)'s load-bearing move — the proof's
own labelled "Key Insight" — is a **counting/divisibility argument**, and the 36 have **no
counting pattern** (verified: no double-counting / pigeonhole / orbit-counting). The move:
decompose `|G|` over conjugacy classes; every *non-central* class size is `≡0 (mod p)`;
`|G| ≡ 0`; so the residual term `|Z(G)| ≡ 0`, forcing it nontrivial. I wrote
**`math-informal/count-over-a-decomposition`** (`[🧮/数]`, registered): *split a quantity over
a decomposition, control all-but-one part with a shared congruence/bound/vanishing, read off
the residual.* The proof's Remark confirms the generality — "the same counting principle
behind Cauchy's theorem and the Sylow theorems"; add Burnside, inclusion–exclusion, pigeonhole.

## Wiring + sorry (mechanism holds a third time)

**Wiring (c+d):**
```
reduce-to-known(class equation, orbit-stabilizer) ─▷ pⁿ=|Z(G)|+Σ[G:C_G(xᵢ)]
            │
count-over-a-decomposition(non-central classes ≡0 mod p) ─▷ p | |Z(G)|
            │  estimate-by-bounding ─▷ |Z(G)| ≥ p > 1   ── (c) Z(G)≠{e}
            ▽
construct-aux(upper central series Z₀⊊Z₁⊊…)
            │  Z_i≠G ⇒ apply (c) to G/Z_i ⇒ Z_i⊊Z_{i+1}   (reduce-to-known: own lemma (c))
            ▽
induction-and-well-ordering(chain in finite G terminates, only at G) ─▷ (d) G nilpotent ∎
```

**Sorry (= undischarged HOWEVERs of the matched patterns — third confirmation):**
| sorry | from pattern | obligation |
|---|---|---|
| S1 every non-central class size is `>1` and divides `pⁿ` ⇒ `≡0 mod p` | **count-over-a-decomposition**'s HOWEVER ("the shared constraint holds on *every* other part; decomposition exhaustive+disjoint") | orbit-stabilizer gives `[G:C_G(xᵢ)] \| pⁿ`; `>1` since non-central |
| S2 `G/Z_i` is a smaller p-group, `Z_i ◁ G` so `Z_{i+1}` well-defined | **induction-and-well-ordering**'s HOWEVER (right variable / the step) | order `pⁿ/\|Z_i\|`; preimage of center is normal |
| S3 the class equation itself (central ⟺ singleton class) | **reduce-to-known-result**'s HOWEVER (reduction is natural) | conjugation action, singleton orbit ⟺ central |

## What #3 adds (running tally over 3 proofs)

- **Mechanism is now 3/3 shape-independent** (reduce-to-known #1, construction #2, multi-part
  induction+counting #3) — wiring = patterns' conclusions, sorry = patterns' undischarged
  HOWEVERs, every time.
- **Pool growth: +0, +1, +1** (now **38** math-informal: `separate-into-independent-pieces`,
  `count-over-a-decomposition`). Demand-driven, ~1 new pattern per non-trivial new *shape* —
  and the two new ones are both top-tier general techniques (independence, counting), not niche.
- **Negative result is signal too:** a hard induction did *not* need an induction sub-schema.
  Confirms not to pre-build the menu by fiat — let the proofs demand patterns.
- **`select` matches families + `:cites`/own-lemma slots:** (d) applies the proof's *own* part
  (c) as the "known result" — `reduce-to-known-result` with the cited result being a
  *just-proved lemma*, not an external theorem. The `:cites` slot spans external + internal.

## Next
- The mechanism is well-attested (3 shapes, 2 new patterns, stable sorry-generator). A 4th
  proof (a **diagram-chase** or an **analysis/ε-δ** shape) would test two more regions, but we
  now have enough to write **CAS-SEL-1's spec spike *on* these worked examples** — the
  step→pattern→(wiring,sorry) procedure is concrete and repeatable. Recommend: one more shape,
  then spec.
