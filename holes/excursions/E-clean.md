# E-clean — CLean: EDN-as-Lean-sketch for the CT treatment of a proof

*Author: claude-1, 2026-06-18. Spawned from the EXP-3/EXP-3b arc
(`E-bge-retrieval-cas-sel-3b.md`) and Joe's de-scope: we don't have to **be** the
structure-aware retriever — Rob's downstream already does Lean → neo4j + pgvector with
graph+embedding indexing. Our job is to **produce structure-bearing artifacts his pipeline
can ingest.** CLean is that artifact format.*

## HEAD (one line)

**CLean = the EDN image of the CT treatment of a proof** — a comb of typed Poly
interfaces with typed holes — that renders to the DarkTower Lean types Rob already
consumes. "Clojure/EDN as Lean-sketch"; *clean* because it carries the structure without
the proof-term weight.

## Why this exists (the de-scope)

The EXP-3/EXP-3b finding: text retrieval (any model size, with or without context) cannot
reach structural proof-similarity — the load-bearing signal is the **compositional shape**
(combs + typed holes), not the prose. We were auditioning to build the retriever
(R-GCN/HyperGCN/comb-encoder). **Dropped.** If Rob's pipeline indexes structure, the
encoder choice is *his* side of the fence. Ours collapses to: emit the structure in a form
he ingests. EXP-3 becomes a one-line note to him — *"index the structure, not the prose;
text embeddings provably plateau here"* — not a project.

## This is not greenfield — it's the proof-side of M-typed-holes

**`futon3c/holes/missions/M-typed-holes`** already did exactly this move for **missions**:
semi-formalise an informal artifact as a BV-typed wiring diagram with typed holes, emit a
machine-readable EDN sketch, and map it field-for-field to DarkTower Lean
(`mathlib4/DarkTower/Examples.lean §MissionExample`, 0 sorry). CLean applies the *same
datatype* to **pattern-tagged proofs**. Consequence (Joe): because it's the same typed-hole
vocabulary, Rob's pipeline ingests **both** — missions (already) and APM proofs (via CLean)
— by one path.

## The dual structure (the interesting part)

We tag proof steps with **patterns** (the iching/CT method concepts), so each proof has
**two structures at once**:
- **Informal** — the pattern-tagged method spine (`construct-auxiliary-object → … →
  reduce-to-known`). The iching/CT concept reading.
- **Formal** — the comb of typed Poly interfaces (consumes/produces) with typed holes.

These are the two readings of one object — M-typed-holes' **`copar`** — and well-formedness
is their *coherence* (every method tag is exactly one box, in order). CLean carries both in
`:clean/copar`. Pure Lean has only the formal; pure prose has only the informal; CLean is
the join.

## The schema (mirrors the DarkTower types)

| CLean (EDN) | DarkTower Lean | meaning |
|---|---|---|
| `:clean/seq` (method-tag chain) | `BV.seq` over `inductive Method` | the non-commutative proof spine |
| `:clean/boxes[]` (`:method` + `:consumes`/`:produces`) | `TypedHole` = `PFunctor` position + directions | a step as a typed interface |
| `:hole {:satiety …}` | `TypedHole.SatietyGrade` (`parse/payoff/canon/bundling/role`) | what *kind* of obligation the hole is |
| `:hole {:discharge …}` | `Discharge.DischargeKind` (`sorryProof/queryAnswer/ungroundedBinder`) | how the hole closes |
| `:clean/wires[]` (construct→consume) | `Comb.comp` (dependent-lens) / `Fill` (`PFunctor.comp`) | the comb wiring / substitution |
| `:clean/copar` (informal ∥ formal) | `BV.copar` over `inductive Reading` | the two-readings coherence |

**The `DischargeKind` ↔ our grains identity** (the session's three threads, unified):
- `sorryProof` = a `:missing-warrant` hole closed by a proof (the IATC holes).
- `queryAnswer` = a rung-3-3 ArSE question closed by an answer (typed bells `:query`/`:answer`).
- `ungroundedBinder` = an ungrounded symbol closed by a binder (sfc symbol-grounding).

So the symbol-grounding and rung-3 work aren't side-grains — they're **discharges of typed
holes**, the same datatype as the warrant holes.

## Worked example — `holes/clean/a93J05.clean.edn`

`a93J05` ("an elliptic entire function is constant") — the cleanest comb shape we have:
**construct a fundamental domain → exploit the lattice symmetry (quotient) → extend
locally-to-globally → discharge to Liouville.** Holes sit on the three creative moves
(s1 construct, s3 quotient, s4 local-to-global); the two `reduce-to-known` steps discharge
to named theorems (extreme-value, Liouville) and carry no hole. The aux object fans out
(s1 → s2 *and* s3). See the file for the full CLean.

## How Rob consumes it

- **Now (zero change for Rob): `CLean → render → Lean → his pipeline.`** A small emitter
  maps the table above to `Comb`/`TypedHole`/`BV` constructors — exactly as
  `Examples.lean §MissionExample` instantiates the mission sketch. The render **doubles as a
  correctness gate**: CLean is well-formed iff it produces type-correct DarkTower Lean.
- **Later (if his ingestion eats graph/EDN directly):** `CLean → neo4j schema` — drop the
  Lean render. The boxes are nodes, wires are edges, holes are typed properties.

## Next steps
1. **Verify the round-trip on `a93J05`:** add a `ProofExample` to `DarkTower/Examples.lean`
   rendered from `a93J05.clean.edn`, and confirm it builds 0-sorry against the landed types
   (mirrors `MissionExample`). One worked round-trip de-risks the whole direction.
2. **Write the `CLean → Lean` emitter** (deterministic; the table is the spec).
3. **Automate the producer:** IATC argument-graph → typed boxes (method from the pattern
   tag; interface from premise/conclusion) → CLean. The 70B does the box-typing; the rest is
   mechanical. Re-type the IATC edges construct→consume as part of this.

## Honest caveats 🔒
- **N=1 worked example** (`a93J05`, hand-lifted). The schema is proven *for missions*
  (M-typed-holes); the proof-side round-trip is asserted-not-yet-built until step 1 compiles.
- **Box interfaces are hand-named here.** Automating the producer needs the
  consumes/produces extracted from the IATC graph reliably — design work.
- **Rob's exact ingestion contract** (Lean source? a specific neo4j schema?) is still the
  one external unknown; the `CLean → Lean` path is the safe default that needs no answer from
  him, but a graph-direct path would.

*Cross-refs:* `E-bge-retrieval-cas-sel-3b.md` (why structure, not text);
`futon3c/holes/missions/M-typed-holes*` (the mission-side sibling + the EDN↔Lean method);
`mathlib4/DarkTower/{Comb,TypedHole,Fill,Discharge,BV,Examples}.lean` (the target types);
`futon3/library/iching/` (the 64-concept method vocabulary); `holes/clean/a93J05.clean.edn`.
