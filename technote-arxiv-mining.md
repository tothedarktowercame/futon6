# Technote: arXiv Mining — Theoretical Background for M-superpod-mark2

*Why the Mark 1 embeddings collapsed, what the literature says about
fixing it, and how the pipeline's design choices connect to the
research landscape.*

**Date:** 2026-04-12
**Companion to:** `holes/missions/M-superpod-mark2.md`

---

## The embedding collapse

The R-GCN graph embeddings (stage 9b) trained with self-supervised
contrastive learning (InfoNCE / NT-Xent loss). Graph augmentation
produced two views of each paper's hypergraph via random node dropout
(10%) and edge dropout (20%). The model learned to match the two views
of the same graph against views of other graphs in the batch.

The problem: with random negatives, the task was trivially solvable.
Validation accuracy reached 99.8% — but pairwise cosine similarities
clustered at ~1.0. The model found a shortcut (likely collapsing to a
constant or near-constant embedding) rather than learning structural
features. The FAISS index built from these embeddings could not
distinguish between papers.

Switching to BGE text embeddings (stage 2, 1024-dim) resolved
retrieval immediately: mean cosine similarity 0.71, std 0.054, range
[0.44, 0.92]. This was a one-line change to the FAISS index source
with zero additional compute. See `apm-lean/report/apm-canaries.tex`
Appendix A for the comparison table.

## Contrastive learning with hard negatives

The standard finding in contrastive learning (Chen et al., SimCLR;
Robinson et al., 2021 "hard negative mixing") is that random negatives
produce trivially solvable tasks. The model can separate positives
from negatives using surface features without learning the intended
representation.

Hard negatives — examples that are *similar* on an easy-to-learn
dimension but *different* on the dimension you want the model to
capture — force the model past the shortcut. In our case:

- **Easy dimension:** text similarity (what BGE already captures)
- **Target dimension:** proof structure (hypergraph topology)
- **Hard negatives:** paper pairs with BGE cosine 0.7–0.8 (textually
  confusable) but different hypergraph structure

This is the Mark 2 fix for stage 9b. The R-GCN architecture
(Schlichtkrull et al.) is not the problem — it handles typed
multi-relational graphs, which is exactly what our hypergraphs are
(6 edge types: iatc, mention, discourse, scope, surface, categorical;
4 node types: post, term, expression, scope). The training signal was
too weak for the architecture to learn anything.

## Hybrid retrieval

Dense retrieval works best when combining complementary signals
(Karpukhin et al., DPR; Izacard & Grave, Atlas). Our pipeline
produces two:

- **BGE (stage 2):** text similarity, 1024-dim, working
- **R-GCN (stage 9b):** structural similarity, 128-dim, to be fixed

The conservative hybrid: concatenate frozen BGE and R-GCN embeddings,
let the FAISS index handle the combined space. This preserves BGE
quality (which we know works) while adding structural signal. Joint
training is richer but risks degrading BGE — not worth it until we
know R-GCN carries useful signal on its own.

## Technique-level NER

The learn-to-swim canaries revealed a granularity gap in the NER
kernel. The current 19,236 terms from PlanetMath are concept-level:
"functor", "equivalence", "adjunction". Retrieval helped for canary
C7 because the technique — Pettis integral as monad algebra map — is
a specific construction, not a concept.

The distinction (related: `futon3/library/math-informal/find-the-right-abstraction`):
- **Concept:** "adjunction" — appears everywhere, low discriminative power
- **Technique:** "Borel completion adjunction" — appears in specific
  proof contexts, high discriminative power

Scope-aware extraction (terms that appear inside theorem statements or
proof environments, with their binding context) is a principled way to
capture technique-level terms. This improves stage 5 NER and
consequently the hypergraphs built in stage 9a.

## Pattern library cross-references

- `enrichment/rational-reconstruction` — build evaluation sets
  incrementally from working retrieval results, not as one-shot dumps
- `enrichment/layered-ingestion` — the pipeline already runs in
  stages; improvements target stages 5, 9b, 10 without reprocessing
  earlier stages
- `math-informal/find-the-right-abstraction` — the technique NER
  problem: what level of abstraction captures proof architecture?

## Evidence from the learn-to-swim canaries

From `apm-lean/report/apm-canaries.tex`:

| Canary | Technique | Retrieval useful? | Why |
|--------|-----------|-------------------|-----|
| C5 (ultraproducts) | Standard | No | Define map, check bijectivity — no external guidance needed |
| C6 (Borel spectra) | Library gap | No | Right domain found (sim 0.86) but technique already known |
| C7 (monadic integration) | Non-obvious | **Yes** | Companion paper with identical construction (sim 0.82) |

The signal: retrieval helps when the technique is non-obvious and
domain-specific. Text similarity (BGE) was sufficient to find C7's
companion paper. The open question is whether structural similarity
would find technique-relevant papers that BGE misses — papers with
the same proof architecture but different vocabulary.
