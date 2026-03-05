# Known Limitations

Last updated: 2026-03-04

This document tracks known production-readiness gaps in the superpod pipeline
outputs (`storage/math-processed-gpu/` and `storage/mo-processed-gpu/`) from
the Feb 27, 2026 run.

## Critical

### Stage 6: Reverse Morphogenesis — <10% Parse Success

JSON parse success rate is catastrophically low:
- Math.SE: 77,391 / 805,200 (9.61%)
- MathOverflow: 6,391 / 95,321 (6.70%)

Root cause: local Llama-3-8B produces unclosed/malformed JSON without
schema constraints. Schema-constrained decoding (Gemini `response_schema`,
Codex `--output-schema`) achieves 100% parse on a 50-row sample but has
**not been deployed at scale**.

Impact: `reverse-morphogenesis.json` covers <10% of threads. Stage 7
thread wiring partially depends on this data.

Fix: Full-corpus rerun with schema-constrained decoding via API (B1 in
improvement checklist).

### Stage 7: Categorical Consistency — 3-5%

Category-theoretic structural assertions have very low confidence:
- Categorical consistency: Math 3.31%, MO 3.87%
- Port compatibility: Math 4.67%, MO 5.93%

IATC (performative) alignment is usable: Math 71.83%, MO 66.06%.

Impact: Categorical edges in `thread-wiring-ct.json` cannot be trusted for
decision-making. Only IATC performatives (clarify, assert, query, etc.) are
reliable.

Recommendation: Treat as two confidence tiers. Tier 1 (IATC) = usable.
Tier 2 (categorical) = experimental only.

## Moderate

### Stage 4: Clustering Not Run

HDBSCAN clustering was skipped in shard mode. No global cluster assignments
exist. The `embeddings.npy` files (805K + 95K vectors) are available for
post-merge clustering but it hasn't been done yet.

### Stage 9b: Graph Embedding Validation May Be Too Easy

Contrastive learning achieved 99.8% Acc@1 (Math, epoch 33) and 99.85%
(MO, epoch 30). This is suspiciously high.

Audit results (2026-03-04) show embeddings are **not collapsed**:
- All 128 dimensions active, healthy per-dim variance
- Random-pair cosine similarity centered near 0 (median ≈ -0.001)
- Top-k neighbor similarities (0.62–0.79) well above random baseline

The high accuracy likely reflects easy contrastive negatives from graph
augmentation (dropout creates near-identical positive views). The embeddings
are discriminative in absolute terms but the training metric overstates
quality. External validation (human-judged retrieval P@k) is needed.

### Retrieval Quality Unvalidated

No end-to-end human-judged retrieval evaluation has been completed:
- Math review-50: 9/50 pairs judged (5 yes, 3 no, 1 unsure)
- MO review-50: 50/50 judged (strict precision 42%, MAP 0.295)

Additionally, review-50 thread IDs partially mismatch the embedding index:
only ~31% of MO review thread IDs exist in the FAISS index (the review
pairs were sampled from a broader set than what produced hypergraphs).
This limits the ability to validate embeddings against human judgments.

A new review-100 set was generated (2026-03-04) using pairs sampled
from the embedding index with tag/title heuristic judgments:

| Tier | Math.SE Strict P | MO Strict P |
|------|-----------------|-------------|
| High (top-5 neighbor) | 36.4% | 27.3% |
| Medium (rank 50-150) | 18.2% | 15.2% |
| Low (random) | 0.0% | 2.9% |

Tier separation confirms embeddings discriminate, but absolute precision
is modest and the heuristic produced 45-63% "unsure". An LLM judgment
pass reading full thread bodies would be more informative.

### Frontier Retrieval: Structural Neighbors Show ~50% Zero Overlap

FM-002 and FM-003 frontier trials show ~50% of structural neighbors have
zero tag/term overlap with the query. Ungated structural candidates
produce generic/off-topic matches. Current gating (`--struct-min-overlap 1`)
may collapse structural novelty into lexical redundancy.

## Low Priority

### Stage 3: Pattern Tag Precision Unaudited

Coverage is high (98% non-empty, 3.26 tags/entity) but no human-judged
precision audit has been done on a representative sample. Tag reliability
is assumed but not measured.

### Stage 8: ~1% LaTeX Parse Failures (Analyzed 2026-03-04)

99.0% (Math, 332K/33.4M) / 99.1% (MO, 44K/5.1M) of LaTeX fragments
parse successfully. Failure analysis shows:
- 85% of failures are <20 characters (tiny expressions)
- 62-68% match no complex LaTeX construct — they're simple tokens
- Dominant failure: `*` (adjoint/dual) and `#` as postfix operators
- Second: `\rm` + spacing commands (`\rm\, R\,` patterns)
- Third: unrecognized multi-letter commands (34% of failures)

Fix would be: add `*`/`#` as postfix operators and handle `\rm` in
the parser. Would eliminate ~70% of failures. Low urgency.

### Stage 5: Scope Detection at ~80%

NER coverage is near-complete (99.6%+). Scope detection (quantifiers,
binders, constraints) covers 78% (Math) / 82% (MO). The remaining 20%
of entities lack scope annotations. No precision audit on detected scopes.

## Proof Quality

All 10 frontier proof drafts have gaps (per REVIEWER.md, 2026-02-11):
- **Critical** (P2, P7, P8): Deep surgical gaps affecting correctness
- **Major** (P1, P3, P4, P5, P6, P10): Need theorem citations or
  conditional labeling
- P4 should be recast as evidence-backed conjecture (0/18K violations
  but no formal proof)

None are publication-ready without significant rework.

## Test Infrastructure

Resolved 2026-03-04: venv created, package installed in editable mode,
`test_graph_embed.py` unpack fixed, `conftest.py` added for skip markers.
Current status: **265 passed, 24 skipped** (skips = missing PlanetMath
`category-theory.edn` fixture data).
