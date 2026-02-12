# Futon5 Proof Bridge: Tensor Search over Argument Diagrams

**Date:** 2026-02-12
**Status:** Design note. Pre-spring-break scoping.

## The Idea

Mathematical argument structures — proof steps, challenges, reformulations,
resolutions — can be represented as typed wiring diagrams. These diagrams can
be embedded as tensors. Tensor similarity search over a corpus of embedded
argument diagrams is a fundamentally new way to find proof strategies:
searching by argument shape rather than by keyword.

The core insight: **a proof gap is a wiring diagram with an open node. Embedding
it alongside resolved diagrams is equivalent to asking a question on Stack
Exchange. The nearest neighbors with that node closed are answers.**

## Pipeline

```
SE threads  →  wiring diagrams  →  tensor embeddings  →  structural search
  Stage 7       IATC performatives    futon5 + JAX        query by shape
```

### Stage 1: SE threads → wiring diagrams (futon6, Stage 7)

Each StackExchange thread becomes a typed directed graph:

- **Nodes**: question, answer, comment (with post body, score, date)
- **Edges**: typed by IATC performative (assert, challenge, reform, clarify,
  query, exemplify, reference, agree, retract)

Structural edges (answer→question, comment→parent) are always present.
Classical detection overrides edge types via regex on hotwords. LLM
classification (moist-run) refines further.

**Current state**: Stage 7 is designed (plan file), superpod job handles
Stages 1-6, Comments.xml parser is specified. Not yet implemented.

**Pre-spring-break target**: Run Stage 7 probe on 50 physics.SE threads
to validate the SE → diagram step.

### Stage 2: Wiring diagrams → tensor embeddings (futon5)

futon5 already has the core machinery:

| Module | What it does | Status |
|--------|-------------|--------|
| `wiring/embedding.clj` | Graph → path signatures → Jaccard similarity | Working |
| `wiring/embedding.clj` | Landmark-based embedding (5D coordinate vector) | Working |
| `wiring/features.clj` | 20+ structural features from graph topology | Working |
| `hexagram/lift.clj` | Matrix → eigendecomposition (Apache Commons / pod-eigs) | Working |
| `hexagram/metrics.clj` | Spectral properties (gap, effective rank, alpha) | Working |

**Adaptation needed**: The current pipeline expects CA rule wirings (256 sigils,
kernel functions). The proof wiring diagrams use a different vocabulary (IATC
performatives, Q/D/M/C/O/B method nodes). But the path signature extraction is
vocabulary-agnostic — it extracts all directed paths through a graph and
compares path sets structurally. The adaptation is a vocabulary extension, not
a rewrite.

**What JAX adds** (spring break):

1. **Batched eigendecomposition**: Scale from one-at-a-time to 1000s of diagram
   embeddings in parallel.
2. **Differentiable embeddings**: Replace Jaccard (discrete, fixed) with a
   learned manifold where structural similarity is a smooth function.
3. **Gradient-based search**: Given a gap diagram, compute the gradient of
   "resolution likelihood" with respect to the embedding, and follow it.

### Stage 3: Structural search (futon5 + JAX + futon3b)

Two modes:

**Retrieval**: Given a gap diagram, find nearest neighbors in the embedded
corpus. Return threads whose argument structure most closely matches the
gap's shape, regardless of mathematical content vocabulary.

**Generation**: Given the gap's shape and the distribution of resolution
patterns in the training data, propose candidate resolution diagrams ranked
by structural frequency.

## Artificial Stack Exchange

The key conceptual move: **inducing** new wiring diagrams (not just mining
existing threads) and observing where they sit relative to known diagrams.

### Asking a question

Construct a wiring diagram from the proof gap:

```
[assert: proved result A] → [assert: proved result B] → [query: open step]
     ↑                                                        |
     └── [challenge: technique X fails because Y] ────────────┘
```

Embed this diagram in the same space as all SE-mined diagrams. Its position
relative to resolved diagrams is the "question" — structurally, what kind of
gap is this, and what kinds of resolutions have worked for similar gaps?

### Finding answers

The nearest neighbors with the `query` node replaced by a resolution
(e.g., `assert → reform → assert`) are candidate answers. The resolution
pattern — the typed subgraph that closes the open node — transfers to the
proof gap as a candidate strategy.

Example for P6's GPL-H gap:

```
Gap diagram:
  [assert: leverage threshold] → [assert: Turán bound]
    → [assert: Cases 1,2a] → [query: operator norm beyond trace]
    → [challenge: 6 trace-based techniques hit q² vs q wall]

Search result (hypothetical MO thread on random matrices):
  [assert: concentration bound] → [challenge: trace loses rank factor]
    → [reform: use matrix Freedman with anisotropic variance proxy]
    → [assert: operator norm bound without trace conversion]
```

The resolution pattern `challenge:trace-loses-rank → reform:anisotropic-
variance-proxy → assert:direct-norm-bound` transfers as a candidate
strategy for GPL-H, even though the original thread is about random
matrices and P6 is about graph Laplacians.

### Rating answers

Resolution patterns that appear frequently near structurally similar gaps
rank higher. "Gaps shaped like yours (trace bound insufficient, need
operator norm control) were most often resolved by:
1. Barrier argument with anisotropic control (23 threads)
2. Probabilistic existence via second moment method (14 threads)
3. Algebraic identity that bypasses the norm entirely (7 threads)"

The ranking is a function of tensor similarity, not keyword overlap.

## Cross-Repo Architecture

| Repo | Role in the pipeline |
|------|---------------------|
| **futon6** | SE ingestion (Stages 1-6), thread → wiring diagram (Stage 7), proof gap diagrams (first-proof/) |
| **futon5** | Tensor embedding (path signatures → learned embeddings), similarity search, eigendecomposition |
| **futon3** | Pattern library = vocabulary of resolution types. Flexiarg patterns (assert, challenge, reform, ...) are the edge types. Pattern selection records (PSR/PUR) become training signal. |
| **futon3b** | Query layer = federated search interface. Accepts a gap diagram, routes to futon5 for embedding, returns ranked resolution patterns from futon6's corpus. |

## What Exists vs. What's Needed

### Exists now

- SE processing pipeline: 114K physics QA pairs, superpod job for math.SE
  (futon6 Stages 1-6)
- Proof wiring diagrams in JSON: P6 method library (10 diagrams with bridge
  status), P7 hypothetical architectures (5 diagrams with node status),
  P4 proof architecture (27 nodes, 39 edges)
- Path signature extraction + Jaccard similarity (futon5 `wiring/embedding.clj`)
- Landmark-based embedding coordinates (futon5, 5D)
- Structural feature extraction, 20+ features (futon5 `wiring/features.clj`)
- Eigendecomposition with caching (futon5 `hexagram/lift.clj`)
- IATC performative vocabulary: 9 edge types with hotword banks (Stage 7 plan)
- futon3 pattern library: 850+ patterns with typed IF/HOWEVER/THEN/BECAUSE
  structure and hotword lists

### Needed: pre-spring-break

1. **Stage 7 probe**: Implement thread parsing + classical performative
   detection. Run on 50 physics.SE threads. Validate that structural edges
   are correct, classical detection fires on 30%+ of comments, and at
   least 4 of 9 performative types are detected. (Specified in plan file.)

2. **Vocabulary bridge**: Map IATC performative types to futon5 wiring node/edge
   types. This is a configuration file, not code — a mapping from
   `{assert, challenge, reform, clarify, query, exemplify, reference, agree,
   retract}` to futon5's wiring schema.

3. **Proof diagram conversion**: Convert the existing JSON wiring diagrams
   (P4, P6, P7) into the format expected by `wiring/embedding.clj`. Test
   that path signature extraction and Jaccard similarity produce sensible
   results on proof diagrams.

### Needed: spring break (JAX week)

4. **JAX eigendecomposition**: Replace Apache Commons / pod-eigs with
   `jax.numpy.linalg.eigh`. Differentiable, batched, GPU-ready.

5. **Learned embedding**: Train a small model (Flax) that maps proof diagram
   features → latent vector, supervised by bridge-status labels from P6's
   method library and structural similarity judgements.

6. **Nearest-neighbor search**: Build an index (FAISS or ScaNN) over the
   embedded math.SE thread corpus. Accept a gap diagram as query, return
   top-k structurally similar threads with resolution patterns.

### Needed: post-spring-break

7. **math.SE full run**: Process math.SE through Stages 1-7 on the superpod.
   Embed all thread diagrams. Build the searchable tensor library.

8. **Generative mode**: Given a gap embedding, sample resolution patterns from
   the conditional distribution P(resolution | gap shape). Rank by frequency
   and structural coherence.

9. **Feedback loop**: When a resolution pattern successfully closes a proof
   gap (as with P4 Case 3c → PHCpack), record the gap→resolution pair as
   training signal. The system learns which argument shapes lead to which
   resolutions.

## Validation Plan

### Test 1: Jaccard on proof diagrams (pre-JAX)

Convert P6's 10 method diagrams (D1-D10) to futon5 wiring format. Compute
pairwise Jaccard similarity. Check: do diagrams with the same bridge status
(partial vs. none) cluster together? If yes, structural similarity predicts
usefulness even without learned embeddings.

### Test 2: Landmark embedding with proved cases

Use proved proof steps as landmarks instead of CA rules:
- L1: P4 resultant elimination (symmetry-stratified, computational)
- L2: P6 Cases 1/2a (leverage threshold, structural)
- L3: P7 E2 discharge (Fowler criterion, reference-based)
- L4: P4 Case 3c (homotopy continuation, computational handoff)
- L5: P6 star domination critique (counterexample-based refutation)

Each proof method diagram gets a 5D coordinate (distance to each landmark).
Check: does the coordinate predict proof strategy type?

### Test 3: Gap retrieval on math.SE (post-JAX)

Embed the P6 GPL-H gap diagram. Search the math.SE corpus. Check: do the
top-10 results contain threads about operator norm bounds, spectral
concentration, or paving-type results? If yes, the search is working.
If no, the embedding needs refinement.

## Connection to Corneli 2017

The IATC argument diagram framework (Corneli 2017, "Modelling the way
mathematics is actually done") provides the theoretical foundation for
treating SE threads as wiring diagrams. The key insight from that work:
mathematical discourse has typed performative structure (assert, challenge,
reform) that can be represented as a directed graph. We extend this from
description (modelling how math is done) to prescription (using the models
to find proof strategies).

The futon5 tensor embedding adds a computational layer that Corneli 2017
doesn't have: instead of human inspection of argument diagrams, we compute
structural similarity in embedding space. The "artificial stack exchange"
concept — inducing gap diagrams and searching for resolutions — is a
natural extension of treating mathematical argumentation as a typed graph.

## Design Patterns Involved

| Pattern | Role |
|---------|------|
| `math-informal/technique-landscape-map` | The method wiring library (P6 D1-D10) is the prototype for the corpus |
| `agent/hypothetical-proof-architecture` | Gap diagrams with node status are the queries |
| `agent/reduction-to-kernel` | The gap's explicit hypotheses become the query diagram's typed nodes |
| `math-informal/numerical-scout` | Probe runs (50 threads) validate the pipeline before scaling |
| `agent/typed-literature-mining` | SE thread mining with bridge-status labels produces training data |
