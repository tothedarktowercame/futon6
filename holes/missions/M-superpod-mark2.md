# Mission: Superpod Mark 2 — Structural Retrieval that Discriminates

**Date:** 2026-03-29
**Status:** IDENTIFY (mission proposal)
**Origin:** Learn-to-swim canary findings (M-apm-solutions, 2026-03-29).
Embedding collapse in Mark 1 diagnosed during retrieval-augmented
theorem proving.
**Owner:** futon6
**Cross-ref:** M-apm-solutions (futon3c), technote-learn-to-swim (futon6)

## Motivation

The superpod pipeline processes mathematical corpora (arXiv, Stack
Exchange) into typed hypergraphs and produces embeddings for
structural similarity retrieval. The Mark 1 run on 9,916 math.CT
papers demonstrated the pipeline is operationally complete — all ten
stages run, all outputs are produced.

However, the pipeline's distinctive contribution — structural
embeddings from typed hypergraphs — does not yet work. The R-GCN
graph embeddings (stage 9b) collapsed during training: all pairwise
cosine similarities ~1.0, zero discriminative power. The contrastive
objective trained with random negatives solved a task too easy to
learn anything useful. We fell back to BGE text embeddings (stage 2),
which work well for text similarity but cannot see proof structure.

The gap: the superpod builds rich structural representations (typed
hypergraphs with scope bindings, term co-occurrence, discourse
edges) but then fails to embed them usefully. The Mark 1 hypergraphs
are good; the Mark 1 training of embeddings from those hypergraphs
is not. This mission fixes that.

**Why this matters beyond the pipeline itself:** retrieval-augmented
theorem proving (the learn-to-swim protocol) showed that retrieval
helps when the proof technique is non-obvious. Text similarity finds
papers that use the same words. Structural similarity would find
papers that have the same *proof architecture* — the same pattern of
scope bindings, the same dependency structure between claims. That is
the retrieval signal the learn-to-swim protocol needs and currently
lacks.

## The gap (specific)

1. **Stage 9b training signal.** The contrastive objective needs hard
   negatives: papers that are textually similar but structurally
   different. The current random sampling produces a trivially
   solvable task.

2. **Evaluation.** The Mark 1 run reported 99.8% Acc@1 as a quality
   metric. This is misleading (easy negatives). The pipeline needs an
   evaluation protocol that measures whether structural embeddings
   capture proof-architecture similarity, not just topic similarity.
   The learn-to-swim canaries provide a concrete evaluation: does
   structural retrieval find technique-relevant papers that text
   retrieval misses?

3. **Technique-level NER.** The NER kernel (19,214 terms) is
   concept-level ("functor", "equivalence", "adjunction"). The
   learn-to-swim finding was that retrieval helps when it surfaces
   technique-level terms ("Borel completion adjunction", "Pettis
   integral as algebra map"). The pipeline needs scope-aware
   multi-word term extraction to capture these.

4. **Hybrid embedding architecture.** Text and structure capture
   different signals. The pipeline should produce a hybrid embedding
   that uses both, rather than forcing a choice between BGE (text-only,
   works) and R-GCN (structure-only, broken).

## Scope

### In scope

- Improve stage 9b training: hard negative mining, better contrastive
  objective, validation against a human-judged evaluation set
- Add technique-level NER to stage 5 (scope-aware multi-word extraction)
- Design hybrid embedding architecture (text + structure)
- Evaluation protocol using learn-to-swim canaries as ground truth
- Changes to `scripts/superpod-job.py` and related training scripts

### Out of scope

- Reprocessing stages 1–8 (they work; improvements are to 5, 9b, 10)
- Scaling to new corpora beyond math.CT (future mission, after Mark 2
  is validated on the existing data)
- Integration with downstream consumers (M-apm-solutions,
  M-artificial-stack-exchange handle their own wiring)

## Completion criteria

1. Stage 9b embeddings on the existing math.CT hypergraphs produce
   pairwise cosine similarity with std > 0.10 (vs current ~0.45 with
   bimodal collapse). Validation accuracy < 90% (indicating a
   non-trivial task).

2. On a 20-paper human-judged evaluation set, structural retrieval
   (R-GCN or hybrid) finds at least 3 technique-relevant papers that
   BGE-only retrieval ranks below position 10.

3. Technique-level NER extracts at least 200 multi-word technique
   terms from the math.CT corpus, validated by spot-checking 50
   against the source text.

4. The improved pipeline is documented and reproducible: a fresh
   superpod run on math.CT with the Mark 2 training produces
   discriminating embeddings without manual intervention.

## Relationship to other missions

| Mission | Relationship |
|---------|-------------|
| M-apm-solutions (futon3c) | Consumer: learn-to-swim uses retrieval |
| technote-learn-to-swim (futon6) | Defines the validation protocol |
| M-artificial-stack-exchange (futon6) | Future consumer of structural retrieval |
| M-distributed-frontiermath (futon3c) | Future consumer for FM proof campaigns |

## Open questions

1. **Hard negative mining strategy.** Use BGE similarity (confusable
   pairs at 0.7–0.8) or mine from hypergraph structure (same terms,
   different scope topology)? Or both?

2. **Hybrid architecture.** Concatenate frozen embeddings (simple,
   preserves BGE quality) or train jointly (richer, riskier)? The
   frozen approach is the conservative first step.

3. **Technique NER boundary.** "Adjunction" is a concept;
   "Borel completion adjunction" is a technique. Where's the line?
   Scope-aware extraction (terms that appear inside let-bindings or
   theorem statements) might be a principled criterion.
