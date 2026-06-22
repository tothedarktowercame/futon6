# CLean → neo4j + pgvector — the ingestion mapping for Rob

This is the contract between **CLean** (our structure-bearing proof artifact) and
**Rob's existing workflow** (Lean → neo4j graph index + pgvector embedding index).
The de-scope (E-clean): we don't build the retriever; we emit the structure his
pipeline already knows how to index. This doc + the generated `ingest/` artifacts
ARE the handoff.

## The one-line claim to put in front of Rob

> Index the proof's **structure**, not its prose. A 33-dim hand-built structural
> embedding clusters proofs by *method* across unrelated topics (a p-group proof
> and a torus-rotation proof come out as nearest neighbors, cosine 0.95); the
> MiniLM **text** embedding of the same proofs scatters them (0.24, and it picks
> the wrong twin). Text embeddings provably plateau on structural similarity —
> EXP-3.

## What gets produced (run locally, no GPU)

```
holes/clean/*.clean.edn                          the CLean artifacts (one per proof)
scripts/clean_argcheck.bb                         well-formedness gate (7/7 PASS)
scripts/clean_structure_embed.py                  structure (33-d) + text (384-d) embeddings
scripts/clean_graph_export.py                     -> ingest/{clean-graph.json, load.cypher, pgvector.sql}
```

## The field mapping

| CLean (EDN)                       | neo4j                                       | pgvector                |
|-----------------------------------|---------------------------------------------|-------------------------|
| `:clean/proof` / `:clean/title`   | `(:Proof {id, title, macro})`               | `clean_proof.id`        |
| `:clean/shape :macro`             | `Proof.macro`                               | `clean_proof.macro`     |
| `:clean/boxes[]`                  | `(:Step {id, method, text, has_hole, …})`   | —                       |
| `box :method`                     | `Step.method`                               | (a feature dim)         |
| `box :hole {:satiety :discharge}` | `Step.satiety`, `Step.discharge`            | (feature dims)          |
| `:clean/wires[]`                  | `(:Step)-[:WIRES {carries}]->(:Step)`       | —                       |
| `box :discharges {:to}`           | `(:Step)-[:DISCHARGES_TO]->(:Theorem)`      | —                       |
| structure embedding               | (optional `Proof.embedding`)                | `clean_proof.structure vector(33)` |
| text embedding (baseline)         | —                                           | `clean_proof.text_emb vector(384)` |

The `:clean/copar` coherence (informal method-spine ∥ formal comb) is enforced by
`clean_argcheck.bb` *before* export, so anything that reaches neo4j/pgvector is
well-formed by construction — the same gate-before-ingest discipline as IATC.

## Two ways for Rob to consume (his choice, no work needed on our side)

1. **Graph-direct (recommended):** load `ingest/load.cypher` into neo4j and
   `ingest/pgvector.sql` into postgres. The boxes are nodes, wires are edges,
   holes/discharges are typed properties/edges, and the structure vectors land
   in pgvector. The two sample queries at the bottom of `pgvector.sql` reproduce
   the structure-vs-text contrast directly in SQL (`<=>` cosine distance).

2. **Via Lean (zero schema change for Rob):** the CLean schema maps field-for-field
   to the DarkTower Lean types (`Comb`/`TypedHole`/`BV`/`Discharge`) — see the
   E-clean table. A deterministic `CLean → Lean` emitter (E-clean next-step 2,
   not built yet) would render each proof to a `ProofExample` exactly as
   `Examples.lean §MissionExample` does, and Rob's existing Lean→neo4j+pgvector
   path ingests it unchanged. This is the safe default that needs no answer from
   him; the graph-direct path is faster if his ingestion eats EDN/graph directly.

## Scaling note (the Linode round)

This demo lifts 7 APM proofs to CLean by hand (the "Claude-written round of
processing"). The full corpus is 462 informal proofs at
`futon3c/data/apm-informal-proofs/`. At scale the box-typing is an LLM batch pass
(the 70B does the `:method` + `:consumes`/`:produces` typing; the comb wiring and
the embedding are mechanical), constrained to `clean-method-vocab.edn` so the
embedding space stays shared. Provisioning already exists (`linode-4gpu-*.sh`).

## What's real vs. asserted (honesty boundary)

- **Real:** the 7 CLean files, the 7/7 well-formedness gate, the 33-d structural
  embedding, the MiniLM text baseline, the cross-topic clustering result, and the
  generated cypher/SQL artifacts.
- **Stand-in:** we emit the cypher/SQL rather than standing up neo4j+postgres
  here — the artifacts are the contract, loadable into Rob's stack as-is.
- **Not yet built:** the `CLean → Lean` emitter + the a93J05 round-trip compile
  (E-clean next-steps 1–2); the automated IATC-graph → CLean producer.
