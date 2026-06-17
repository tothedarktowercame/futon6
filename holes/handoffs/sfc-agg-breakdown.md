# SFC-AGG breakdown — per-concept map-reduce aggregator

*Breakdown of the `SFC-AGG` "needs breakdown" card in `holes/proofcheck-readiness.html`
(Phase A · structure-first concepts). Owner excursion: **E-structure-first-concepts** (claude-1).
Drafted by claude-loop, 2026-06-17 — DRAFT for review, not dispatched. SFC-AGG is **design-blocked**
("reduce semantics + schema unspecified"), so the breakdown leads with a spec spike, then builds.*

## IDENTIFY — the gap

`build_concept_encyclopedia.py` grounds each concept from **one** chosen defining paper: a single
`gloss {paper, text}` and `_components` (genus + differentiae) extracted from that *one* passage.
`defined_in` lists `n_papers` + a sample, but the **structure is from a single paper**. What's missing is
the **reduce**: fold the concept's *many* per-paper grounded instances into one canonical entry that
captures (a) the **shared genus**, (b) the **variant axes** — parameters that differ across papers (base
field ℝ/ℂ, ambient category, finite/infinite) kept as a *family* not averaged away ("fixed in post"), and
(c) **polysemy** — genuinely distinct senses. The card's words: *"reduce([per-paper grounded instances]) →
{genus, variant-axes}, a **monoid** (incremental, polysemy-tolerant). Reduce semantics + schema unspecified."*
Downstream (R2d, the cascade's concept layer) needs the concept's *corpus shape*, not a one-paper gloss.

## MAP — what exists (the per-paper instances are already on disk)

```
build_concept_encyclopedia.py   per-concept entry; gloss = ONE paper; _components = genus + differentiae
                                from that one gloss; defined_in{n_papers, sample}; :structure = a hole.
data/warp/def-snippets.json     concept -> [real definition passages] across papers (972 concepts)  ← THE INSTANCES
data/warp/defined-index.json    concept -> defining papers
data/warp/concept-usage.json    {paper: [concepts]} usage instances
mark3_thread_tapestry           per-concept phylogeny (typed activations) — variation along citations/time
```

So the **per-paper grounded instances already exist** (def-snippets carries multiple passages per concept).
SFC-AGG is the missing **fold** over them — not new mining.

Current entry schema (`concept-encyclopedia-ct.json`): `{concept, msc, kind, df, pagerank, used_papers,
depends_on, gloss{paper,text}, components{genus, differentiae[{clause, refs}]}, defined_in{n_papers, sample},
provenance, holes[{kind:formalise, wanted:"render differentiae as typed ∀/∃ conditions"}]}`.

## DERIVE — the aggregator

`reduce(concept, [instance…]) → aggregate`, where each `instance = {paper, gloss-passage,
components{genus, differentiae}, (later) symbol-bindings}`. The reduce:

- **Genus consensus** — cluster the per-paper genera; the shared genus is the concept's genus; agreement
  score reported. (e.g. "adjoint functor" → genus *functor* across nearly all papers.)
- **Variant axes** — where differentiae differ across papers along a *parameter*, record a named axis with
  per-paper values: `{axis:"base-field", values:[{paper, "ℝ"}, {paper, "ℂ"}]}`. The concept is "the same up
  to this axis" — the ℝ/ℂ "fixed in post". This is the family-preservation the card asks for.
- **Polysemy split** — if genera cluster into genuinely *distinct* groups (not a parameter), emit multiple
  **senses** (`concept@sense-1`, `concept@sense-2`) rather than collapsing them.
- **Monoid** — reduce is associative with an identity (empty), so a *new* paper's instance folds into the
  running aggregate incrementally (no corpus re-read). This is what makes the encyclopedia an *incremental
  substrate* over the corpus, not a batch rebuild.

Schema add (extends the entry, doesn't replace it): `:aggregate {genus, agreement, n-instances,
variant-axes:[{axis, values:[{paper, value}]}], senses:[…]}`.

## ARGUE

> **IF** R2d and the cascade need each concept's *corpus shape* (shared genus, the axes it varies along, its
> senses),
> **HOWEVER** the encyclopedia grounds each concept from one paper and the reduce semantics + schema are
> unspecified,
> **THEN** first *spec the reduce against real worked concepts*, then implement it as a monoid fold over the
> already-on-disk per-paper instances (`def-snippets`),
> **BECAUSE** the instances exist (this is a fold, not new mining), and a monoid makes the substrate
> incremental + polysemy-tolerant — but the reduce is a genuine design decision (what is a variant-axis vs a
> polysemy split?) that must be grounded in examples before code, or we'll hard-code the wrong collapse.

## VERIFY — acceptance for the whole breakdown

1. The reduce is **specified against ≥4 worked concepts** (a clean one, a variant-axis one e.g. ℝ/ℂ, a
   polysemous one, a single-instance one) — hand-traced expected aggregates before coding.
2. The implemented fold reproduces those expected aggregates; the **monoid laws hold** (associativity +
   identity, tested) and an *incremental* fold of one new paper equals the batch result.
3. Variant-axes surface on the ℝ/ℂ-style example; a polysemous concept splits into senses; a clean concept
   has agreement ≈ 1.0 and no spurious axes.
4. Deterministic; CPU; extends the encyclopedia entry without breaking existing consumers (SFC1).

## INSTANTIATE — sub-handoffs (SFC-AGG-1 is a SPEC spike, do it first)

### SFC-AGG-1 · Spec the reduce against worked concepts · CPU · design spike
**Goal:** resolve "reduce semantics + schema unspecified" *before* code. Pick ≥4 concepts from
`def-snippets.json` — one clean (e.g. *adjoint functor*), one with a real **variant axis** (a concept defined
over ℝ in one paper and ℂ in another, or over a varying ambient category), one **polysemous**, one
single-instance. For each, hand-trace from its actual per-paper passages what `:aggregate` *should* be
(genus + agreement, variant-axes with per-paper values, senses). **Deliverable:** a short spec note
(`holes/excursions/sfc-agg-spec.md`) fixing (a) the `:aggregate` schema, (b) the genus-consensus rule, (c)
the variant-axis-vs-polysemy discriminator, (d) the monoid identity + merge. Acceptance: the 4 worked
aggregates + the schema, reviewed by the excursion owner before SFC-AGG-2.

### SFC-AGG-2 · Implement the monoid reduce · CPU · PY
**Depends on** SFC-AGG-1. **Goal:** implement `reduce` per the spec over `def-snippets.json`'s per-paper
instances, extending `build_concept_encyclopedia.py` (reuse `_components`; do not fork). Emit `:aggregate`
on each entry. **Acceptance:** reproduces SFC-AGG-1's 4 worked aggregates; a `pytest` asserts the **monoid
laws** (assoc + identity) and `incremental(fold-one-more) == batch`; variant-axis + polysemy examples pass;
deterministic. **Gates:** PY + report the agreement/axis numbers on a sample.

### SFC-AGG-3 · Wire into the encyclopedia build + manifest · CPU · PY
**Depends on** SFC-AGG-2 (+ WARP-ORCH-2's runner). **Goal:** the aggregate is produced by the encyclopedia
stage inside `warp_run` (incremental: re-running after one new paper folds it in, not a full rebuild), and
recorded in `warp-manifest.json`. **Acceptance:** `warp_run` emits aggregated entries; SFC1 still reproduces;
an incremental re-run after adding one paper updates only the touched concepts.

**Gates (all):** PY (`pytest`) + report numbers. Coordinate with the E-structure-first-concepts owner —
SFC-AGG extends the same encyclopedia/`data/warp/` artifacts the live SFC work touches.

## Note — relation to neighbours
- **`:structure` lift (Phase B / SFC2a):** SFC-AGG aggregates *glosses/components*; the `:structure` typed
  ∀/∃ form (the `holes[{kind:formalise}]`) is the deeper per-concept fill — SFC-AGG's variant-axes are the
  natural place those typed conditions later vary. Keep the hole; SFC-AGG populates the family around it.
- **SFC2b symbol grounding:** once symbols are bound per-paper, each instance gains symbol-bindings — a
  richer reduce (the variant-axis "base field" is literally a symbol binding that differs across papers).
  So SFC-AGG's instance schema should leave room for `symbol-bindings`, even before SFC2b lands.
