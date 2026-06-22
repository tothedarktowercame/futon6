# How CLean relates to Lean — round-trip + correspondence

Two questions Joe asked: (1) does CLean round-trip to Lean, and (2) for proofs
that already have Lean, how do the two relate? Both answered here.

## 1. The round-trip (CLean → DarkTower Lean, 0 sorry)

`scripts/clean_to_lean.py` deterministically renders each `holes/clean/*.clean.edn`
to the DarkTower Lean types, exactly as `Examples.lean §MissionExample` does for a
mission. Each proof becomes a namespace with a `BV` method spine, a `TypedHole`
(positions = steps; a step's direction is `Obligation` if it has an open hole,
else `Empty`), `SatietyGrade`s, `DischargeKind`, the reduce-to-known `discharges`,
the comb `wires`, and the informal∥formal `copar` — plus self-verifying `example`s.

**Verified:** the standalone render (`--mode standalone`, a minimal no-Mathlib
shim of the exact DarkTower signatures) compiles **0 sorry** under core Lean
4.31 — 7 proofs, **85 examples**, ~1.8 s. The repo file
(`mathlib4/DarkTower/CLeanProofs.lean`, `--mode real`) is construct-for-construct
identical and imports the Mathlib-backed modules; its compile is gated only on a
Mathlib olean build (the cache has no files for this DarkTower fork commit, so a
full build is hours — not run here). Build it with `lake build DarkTower.CLeanProofs`
once mathlib is built.

**What "0 sorry" means here (important):** the DarkTower render makes the proof's
*shape* a first-class typed value that type-checks — it does **not** claim the
mathematics is proved. The open holes are represented as data (an `Obligation`
direction / `DischargeKind.sorryProof`), not as a Lean tactic `sorry`. This is the
E-clean tagline literally: *clean carries the structure without the proof-term
weight.* So there are two distinct senses of "sorry":

| | a real Lean proof | CLean / DarkTower |
|---|---|---|
| an unproved step | tactic `sorry` (the proof is incomplete) | a `TypedHole` with `Obligation` (the shape is complete; the obligation is recorded) |
| a finished step | a tactic block / lemma call | a step with `Empty` direction + a `discharges` target |

CLean is the *map of the obligations*, type-checked; a real Lean proof is the
*discharge of them*.

## 2. CLean ↔ an existing Lean proof — the a01A01 (Pringsheim) correspondence

The APM proofs carry "Lean 4 statement" blocks: a main theorem plus a lemma
decomposition, every leaf `:= by sorry`. These line up with the CLean almost 1:1
— independently produced, same skeleton. For `a01A01`:

| CLean box (`a01A01.clean.edn`) | method | hole? | existing Lean lemma (apm-a01A01.md) |
|---|---|---|---|
| s1 assume analytic at 1 | argue-by-contradiction | — (setup) | the `¬ ∃ f, AnalyticAt …` contradiction frame of `pringsheim_singular_at_one` |
| s2 nonneg Taylor coeffs at 1 | local-to-global | 🔴 payoff | `taylor_at_one_nonneg` |
| s3 converges past 1 | local-to-global | 🔴 canon | `summable_beyond_one_of_analytic_at_one` |
| s4 root-test radius clash | reduce-to-known | 🟢 `root-test-radius` | `radius_ge_of_summable_nonneg` → closes `pringsheim_singular_at_one` |

The correspondence is exact:

- **theorem statement ↔ the proof's goal** = the conclusion box's `:produces`
  (`:contradiction` / the main `pringsheim_singular_at_one`).
- **each `sorry` lemma ↔ a CLean `:hole`** (a `sorryProof`/`Obligation`). The
  Claude-written Lean's three `sorry` lemmas are exactly the three load-bearing
  CLean holes (s2, s3, and the root-test leaf).
- **a CLean `:discharges {:to thm}` ↔ a named mathlib lemma application** — where
  the proof bottoms out in a library result rather than an open obligation
  (`root-test-radius` ↔ `FormalMultilinearSeries.radius` monotonicity).
- **the method spine ↔ the proof skeleton** — the order of `have`/`apply` steps.

So CLean is the **structural skeleton both the informal proof and the Lean
decomposition share**. Going from CLean to a *complete* Lean proof = discharging
each `Obligation` hole with a tactic block and wiring the `discharges` to the
named mathlib lemmas. CLean tells Rob's pipeline *which* lemmas are load-bearing
and *where* the gaps are — the structure to index — without needing the proof
terms.

## Practical upshot for the LLaMA superpod pass

Because the render is deterministic and the round-trip type-checks, the superpod
pipeline can: LLaMA types the CLean boxes (method + consumes/produces) → gate
with `clean_argcheck.bb` → `clean_to_lean.py` renders DarkTower Lean → `lake build`
is the *machine* correctness gate (well-formed CLean ⟺ 0-sorry build) → Rob's
neo4j+pgvector ingests the graph + structure embedding. No Claude/Codex in the
loop — just LLaMA for the typing and deterministic tools for everything else.

*Cross-refs:* `E-clean.md`, `NEO4J-PGVECTOR-MAPPING.md`, `scripts/clean_to_lean.py`,
`mathlib4/DarkTower/{Examples,CLeanProofs}.lean`.
