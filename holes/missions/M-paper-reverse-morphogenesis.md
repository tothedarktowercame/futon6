# Mission: Paper Reverse Morphogenesis

**Date:** 2026-04-15
**Status:** DERIVE (design spec for batch-002; implementation follows)
**Parent mission:** M-superpod-mark2 (Checkpoint — Learn As We Go, 2026-04-15)
**Consumer mission:** M-apm-solutions (forward solver for theorem proving)
**Owner:** Joe (design, pipeline code), Rob (superpod execution)
**Repos:** futon6 (pipeline), futon3 (pattern library seed), apm-lean (downstream evaluation)

## Motivation

Mark 1 processed 9,916 arXiv math.CT papers and showed that BGE retrieval
surfaces papers whose techniques name the problem (canary C7: Pettis integral
as monad algebra map). That finding was a happy accident. Mark 2 makes it the
default case — and, more importantly, makes the mining pipeline produce a
training corpus for a **forward problem-solving model** rather than a static
retrieval index.

The thesis: backward reconstruction at arXiv scale (~570K papers) yields
grounded (problem, techniques, terminology, patterns, result) tuples. A
forward model trained on that corpus learns to propose (techniques, patterns,
sketched construction) given a novel problem. Retrieval quality is a side
effect; the real output is the forward solver.

## Principles

### The paper is evidence, not testimony

Authors cannot be trusted to state what problem their paper really solves.
Some papers articulate it well; many don't; and even when they do, the real
problem often differs from the marketed one. The reconstruction reads the
paper as a trace and reasons backwards from techniques, terminology, and
argumentative skeleton to the problem-class the paper actually addresses.

The author's stated framing is **one input among several**, not the
extraction target. A significant stated-vs-reconstructed gap is diagnostic
signal, not noise.

### Terminology constrains problem-class

Technique vocabulary carves out the space of problems solvable with that
vocabulary. Ruler-and-compass is the clean base case: Wantzel gives you
exactly the quadratic closure of ℚ — the terminology *is* the solvability
envelope. For fuzzier techniques the envelope is softer, but the move is the
same: let the named techniques tell you what problem-class is in reach,
compare against what the author claimed, treat the gap as signal.

This means stage 5c (technique-level NER) is load-bearing — not an auxiliary
index, but the atomic vocabulary from which problem-class is reconstructed.

### Each paper is its own gold standard

No external labels. Quality is measured against the paper itself: does the
sketch generated from (reconstructed problem + techniques + patterns) match
the paper's actual structure and findings? If yes, the reconstruction
captured something real; if no, it missed the heart.

The criterion is **sense of inevitability** — given the reconstructed
ingredients, the paper should feel like the natural construction, not one
of many. That's what mathematical understanding feels like when it clicks,
and it's a measurable per-paper metric.

### The mining approach is ML

The pipeline is a self-improving system with four update channels (pattern
library, technique vocabulary, hypergraph features, prompts). Per-paper
reconstruction gaps drive per-batch pipeline updates. 570K papers × modest
per-batch sharpening = a pipeline very different by the time the corpus is
exhausted. The corpus quality co-improves with the pipeline; not static
accumulation.

## Pipeline architecture

```
parse (existing stage 1)
  → embeddings (existing stage 2)
  → concept-NER (existing stage 5) ─┐
  → technique-NER (new 5c) ─────────┼──→ paper-hypergraph (new 5d)
  → paper-patterns (extended 3) ────┘         │
                                              ▼
                                  paper-reverse-morphogenesis (rewritten 6)
                                              │
                                              ▼
                                        sketch-and-score (new 11)
                                              │
                                              ▼
                                   forward-model training record
```

Stages 4 (clustering), 7 (thread wiring), 8 (expression surfaces), 9a/9b
(thread hypergraph, R-GCN), 10 (FAISS) remain as in Mark 1; arXiv pipeline
skips 7. The new chain runs in parallel with the existing backbone.

### Stage 5c — Technique-level NER

**Goal:** Extract scope-aware, multi-word technique-level terms.
Concept-level NER (existing stage 5) captures atomic concepts ("functor",
"adjunction"). Stage 5c captures technique phrases ("Borel completion
adjunction", "Pettis integral as monad algebra map", "ruler and compass
construction") — the scale at which terminology constrains problem-class.

**Input:** Parsed paper (title, abstract, body sections, references).
**Output:** JSON with one record per extracted technique term.

```json
{
  "paper_id": "arxiv-2401.12345",
  "techniques": [
    {
      "term": "Borel completion adjunction",
      "canonical": "borel-completion-adjunction",
      "loci": [
        {"section": "3", "paragraph": 2, "char_span": [412, 439]},
        {"section": "5", "paragraph": 7, "char_span": [1203, 1230]}
      ],
      "first_defined_at": {"section": "3", "paragraph": 1},
      "extraction_source": "classical"  // or "llm" or "both"
    },
    ...
  ]
}
```

**Extraction strategies (both, kept distinct for the natural experiment):**

- **Classical**: multi-word noun-phrase extraction keyed on the existing
  concept-level kernel. Start with bigram/trigram patterns around known
  concept terms (e.g., `<ADJ>* <CONCEPT> <PREP> <CONCEPT>*`), prune by
  corpus frequency and cross-paper consistency.
- **LLM**: single-prompt extraction with few-shot examples of technique
  terms from hand-labeled math.CT papers (Mark 1 output provides a seed
  set for calibration).

Extraction provenance (`classical`/`llm`/`both`) is recorded per term.
Intersection (both) is highest-confidence; disjoint terms go into the
analysis ledger for prompt and classical-pattern iteration.

### Stage 5d — Paper hypergraph

**Goal:** Lift the paper into a structure-and-terminology-first semantic
object: the argumentative skeleton plus its terminological anchors.

**Input:** Parsed paper + stage 5 (concepts) + stage 5c (techniques).
**Output:** A hypergraph (nodes + hyperedges + sectional features).

**Node types:**

- `concept` — from stage 5
- `technique` — from stage 5c
- `definition` — a definition block, with the terms it introduces
- `theorem` / `lemma` / `proposition` / `corollary` — a claim block, with its statement
- `proof` — a proof block, with its target claim
- `result` — the paper's declared main result(s)
- `equation` — numbered or named equations
- `citation` — external references the paper relies on

**Hyperedge types** (the signal — these are the argumentative skeleton):

- `derivation`: `(theorem T, definitions D₁..Dₖ, techniques T₁..Tₘ, result R)` —
  theorem T depends on definitions D₁..Dₖ, is proved using techniques T₁..Tₘ,
  yields result R.
- `definition-use`: term U defined at locus X, used at loci Y₁..Yₙ.
- `structural-cooccurrence`: terms appearing in the same proof/definition/
  theorem block.
- `citation-grounding`: paper P uses technique T via citation [N].
- `motivation-link`: the paper's intro connects problem/motivation to a
  result or technique elsewhere in the paper.

**Sectional features:** where intro ends, where methods begin, where proofs
sit, presence/absence of related-work section, conclusion structure.

**Two extraction arms (kept distinct, natural experiment for batch-002):**

- **Classical (5d-classical)**: regex + structural parsing on the paper's
  LaTeX source or parsed prose. Fast, auditable, misses implicit structure.
- **LLM-augmented (5d-llm)**: classical output feeds into an LLM pass that
  adds implicit hyperedges ("theorem T uses lemma L's construction without
  citing it"). Slower but richer.

Per-paper provenance records which arm produced which hyperedges.

### Stage 3 (extended) — Paper-level pattern tagging

**Goal:** Tag paper-level compositional patterns — the generative grammar
that links (problem, techniques) to argumentative structure.

**Library seeds (natural experiment for batch-002):**

- **flexiarg-only**: reuse futon3's existing flexiarg patterns as-is.
- **flexiarg + mined**: flexiarg as seed, plus paper-structure patterns
  mined from prior batches (initially empty; grows each batch).

**Pattern types for papers (examples, not exhaustive):**

- `motivation-contribution`: introduction frames a problem, states what
  this paper contributes.
- `definition-theorem-proof`: new object defined, claim made about it,
  proof given.
- `problem-technique-result`: problem stated, technique deployed, result
  obtained.
- `construction-verification`: a construction presented, then verified
  (roundtrip).
- `generalization`: prior result recast in broader setting.
- `reduction`: problem P reduced to problem P' whose solution is known.

**Mining protocol:** when stage 11 scores a paper low on structural match
despite high technique/problem coherence, stage 3 is a prime update
candidate. The mining step looks at the paper's structure, proposes a
candidate pattern that would explain it, validates the pattern on a small
set of other low-match papers, and adds it to the library if it fires
non-trivially. Mined patterns carry provenance (batch of origin, validation
sample size).

**Output:** JSON per paper listing pattern tags with loci and confidence.

```json
{
  "paper_id": "arxiv-2401.12345",
  "pattern_tags": [
    {"pattern": "problem-technique-result", "locus": "sec 1-3", "confidence": 0.9},
    {"pattern": "reduction", "locus": "sec 2.2", "confidence": 0.7}
  ],
  "library_version": "flexiarg@v1 + mined@batch-002"
}
```

### Stage 6 (rewritten) — Paper reverse morphogenesis

**Goal:** Reconstruct the problem the paper really solves, grounded in
evidence from prose + hypergraph + terminology.

**Input (per paper):**

- Prose: abstract + introduction + conclusions (+ methods/results sections
  if present and parseable).
- Stage 5d hypergraph (both arms if run, as separate reconstructions).
- Stage 5c technique list with loci.
- Stage 5 concept list (as context).

**Output (four-layer reconstruction):**

```json
{
  "paper_id": "arxiv-2401.12345",
  "pass_index": 1,  // 1 for single-pass arm, 1..N for multi-pass
  "arm": {
    "hypergraph": "classical",  // or "llm" or "both"
    "patterns": "flexiarg-only"  // or "flexiarg+mined"
  },

  "stated_problem": {
    "text": "...",
    "evidence_loci": ["abstract", "intro para 2"],
    "confidence": 0.8,
    "thinness": "explicit|implicit|absent"
  },

  "techniques_deployed": [
    {
      "term": "Borel completion adjunction",
      "role": "primary|supporting|auxiliary",
      "locus": {"section": "3", "paragraph": 2}
    },
    ...
  ],

  "reconstructed_problem_class": {
    "problem_statement": "...",   // what the paper actually solves
    "problem_class": "...",       // the envelope of problems these techniques address
    "derivation": "...",          // why these techniques imply this class
    "evidence_loci": [
      {"type": "theorem", "id": "3.1"},
      {"type": "technique", "term": "Borel completion adjunction", "locus": "..."}
    ],
    "confidence": 0.7
  },

  "stated_vs_reconstructed": {
    "gap_description": "...",
    "severity": "none|minor|significant|reframe",
    "diagnostic_note": "..."  // e.g., "paper's real contribution is narrower than claimed"
  }
}
```

**Prompt skeleton** (draft — to be tuned against first-batch output):

```
You are a mathematics research analyst. You are reading a paper and
your task is to reconstruct what problem it really solves, not what the
author claims. Authors often misrepresent, understate, or leave implicit
the real problem. Your reconstruction must be grounded in evidence:
the paper's techniques, its argumentative skeleton, and the terminology
it deploys.

TECHNIQUES DEPLOYED (from automatic extraction):
  {technique_list_with_loci}

ARGUMENTATIVE SKELETON (derivation hyperedges):
  {hypergraph_summary}

PROSE (abstract + intro + conclusions):
  {paper_prose}

Return a JSON object with these fields (schema above):
  - stated_problem: what the author says, with evidence loci
  - techniques_deployed: primary/supporting/auxiliary roles
  - reconstructed_problem_class: what the paper actually solves, with
    a derivation showing why the techniques imply this class
  - stated_vs_reconstructed: the gap and its severity

Reasoning discipline: a technique vocabulary constrains the problem-class
it can address. "Ruler and compass" names exactly the problems solvable in
the quadratic closure of ℚ. If this paper's techniques are T₁..Tₘ, what is
the envelope of problems solvable with T₁..Tₘ, and does the paper's actual
result sit inside that envelope? Does the author's stated framing match,
or does the paper quietly do something different?
```

**Multi-pass refinement:** if enabled, passes 2..N are keyed on the weak
fields from pass N-1 (e.g., if `reconstructed_problem_class.confidence < 0.6`,
pass 2 runs with additional context emphasizing the argumentative skeleton).
Fixed N = 3 for batch-002.

### Stage 11 — Sketch-and-score

**Goal:** Generate a structural-and-findings sketch from the reconstructed
output, score it against the actual paper, produce the inevitability metric.

**Sketch schema** (starting granularity — tighten or loosen after batch-002):

```json
{
  "paper_id": "arxiv-2401.12345",
  "sketch": {
    "main_result": "...",
    "derivation_steps": [
      {"step": "...", "techniques_used": ["..."], "pattern": "..."},
      ...  // 3-5 steps
    ]
  }
}
```

**Scoring (three components):**

1. **Term coverage** — fraction of the paper's key terms (top-N concept+
   technique by locus-count) that appear in the sketch. Direct measure of
   whether the reconstruction's vocabulary overlaps the paper's vocabulary.
2. **Structural match** — do the derivation steps, in order, correspond to
   the paper's actual argumentative skeleton (hyperedges from 5d)? Measured
   via graph-edit distance or a simpler ordered-overlap metric.
3. **Finding coherence** — does the sketch's `main_result` align with the
   paper's declared main result (from 5d's `result` nodes)? LLM-scored
   or embedding-similarity-scored.

**Inevitability score** = weighted combination (weights tuned per batch).

The score distribution across a batch is the metric we track. Weak-score
papers become the iteration signal for the per-batch update channels.

## Experimental design

### Dimensions for batch-002

| Dimension | Arms |
|---|---|
| Stage 5d extractor | classical, llm, both |
| Stage 6 refinement passes | 1, 2, 3 (fixed N=3) |
| Stage 3 pattern library | flexiarg-only, flexiarg+mined |

All three dimensions cross-product to 3×3×2 = 18 configurations, but the
multi-pass arm's pass-1 result **is** the single-pass arm (so pass count
is free in evidence cost — only extra LLM calls for passes 2–3). Net new
LLM cost is bounded and predictable.

### Per-paper provenance

Every reconstruction output carries an `arm` field identifying which arms
produced it. A batch-level `experiment_meta.json` records the full
configuration matrix and which papers received which configuration.

```json
// experiment_meta.json (batch-002)
{
  "batch_id": "batch-002",
  "pipeline_version": "futon6@<sha>",
  "library_versions": {
    "flexiarg": "v1",
    "mined_patterns": []  // empty at batch-002 start
  },
  "arms": {
    "5d_extractor": ["classical", "llm"],
    "6_max_passes": 3,
    "3_library_seed": ["flexiarg-only", "flexiarg+mined"]
  },
  "per_paper_assignment_rule": "all arms run on all papers where cost permits; "
                               "fallback assignments logged per paper"
}
```

### Per-batch learning loop

After each batch:

1. **Aggregate score distribution** across all papers × all arms.
2. **Identify gaps per update channel:**
   - Pattern library: papers with high technique/problem coherence but
     low structural match → candidate for mined pattern.
   - Technique vocabulary: terms invented in stage 6's reconstruction
     that don't appear in stage 5c's extraction → vocabulary hole.
   - Hypergraph features: papers whose sketch fails because a derivation
     hyperedge was missing → extractor update.
   - Prompts: systematic weakness in one of the four output layers →
     prompt-shape update.
3. **Apply updates** to `futon6/src/futon6/` and the pattern library.
   Version-tag each update with the batch it came from.
4. **Re-run a 100-paper evaluation sample** from the prior batch using
   the updated pipeline. Compare to that sample's prior scores. If
   improved, proceed to next batch. If not, diagnose before advancing.

## Forward model interface

The mining pipeline produces, per paper, a training record usable by the
forward model downstream:

```json
// forward-model-training-record.json (per paper)
{
  "paper_id": "arxiv-2401.12345",
  "input": {
    "problem_statement": "...",      // from reconstructed_problem_class
    "problem_class": "..."
  },
  "target": {
    "techniques": ["..."],            // ordered, with roles
    "pattern_sequence": ["..."],      // the compositional grammar
    "sketch": {...}                   // main result + derivation steps
  },
  "grounding": {
    "evidence_loci": [...],           // points back into the paper
    "hypergraph_nodes": [...]         // for graph-level grounding
  },
  "quality": {
    "inevitability_score": 0.82,
    "reconstruction_quality_tag": "high|medium|low",
    "arm_provenance": {...}
  }
}
```

Forward model training uses `input → target` pairs, with quality tag as a
sample weight or curriculum signal. Consumer (M-apm-solutions) receives
these records and trains the forward solver.

## Evaluation — inside the pipeline, not after

The inevitability score distribution is the primary metric, computed every
batch. Secondary checks:

- **Learn-to-swim canaries** (apm-canaries.tex): for each canary, papers
  whose reconstructed problem-class should cover the canary's problem
  should be retrievable. Run after every few batches.
- **Technique vocabulary growth curve**: tracked across batches. A healthy
  curve grows sublinearly (new terms per paper drops as the vocabulary
  saturates). Linear or accelerating growth suggests 5c is missing
  stable structure.
- **Gap severity distribution**: fraction of papers with "significant" or
  "reframe" stated-vs-reconstructed gaps. Informative about corpus
  character — high gap rate in a sub-discipline may be a signal rather
  than a bug.
- **Pattern library growth**: new patterns per batch, validation fire
  counts. Healthy curve is front-loaded (many new patterns early, fewer
  as the grammar saturates).

## Open questions

1. **Paper body availability.** Rob's batch-002 tarballs include eprint
   sources. Need to confirm: are they parseable LaTeX in every case, or
   will we fall back to PDF extraction for a nontrivial fraction? Affects
   5d hypergraph coverage.
2. **Stage 6 prompt length.** Abstract + intro + conclusions + hypergraph
   summary + technique list may run 4–8k tokens per paper. Confirm the
   chosen LLM's context window handles it without truncation; if not,
   decide which inputs to summarize first.
3. **Sketch scoring LLM.** Stage 11's finding-coherence component uses an
   LLM scorer. Should that be the same model as stage 6, or a different
   one for independence? (Different = less correlated error; same = cheaper.)
4. **Pattern mining throughput.** Mining new patterns from a batch is
   itself an LLM-heavy step. Budget and schedule to be set — probably
   not every batch, maybe every 2–3.
5. **Stage 9a/9b in the arXiv pipeline.** 5d is the paper-level hypergraph
   for stage-6 consumption. Does 9a still run on 5d output for FAISS
   indexing (stage 10), or does 9a remain thread-only and 10 indexes on
   stage 2 BGE embeddings alone? Revisit when stage 9b training signal
   is proven (Mark 2 parent mission's open item).

## Implementation order for batch-002

1. Stage 5c technique-NER extractor (classical and LLM, kept distinct).
2. Stage 5d paper hypergraph (classical arm first, then LLM-augmented arm).
3. Stage 6 rewritten with four-layer schema, single-pass first, multi-pass
   gated behind `--stage6-max-passes N`.
4. Stage 3 paper-level pattern tagging, flexiarg seed, mining protocol
   stubbed (no mining on batch-002's first run; mining enabled from
   batch-002 → batch-003 transition).
5. Stage 11 sketch-and-score with the three scoring components.
6. Experiment-meta ledger + per-paper provenance wiring.
7. Per-batch learning-loop script (aggregate scores, surface gaps, write
   update proposals for human review before applying).

Steps 1–6 are the batch-002 pipeline. Step 7 is the machinery that makes
batches 003+ sharper than 002.

## Related missions

| Mission | Relationship |
|---|---|
| M-superpod-mark2 | Parent: arXiv-scale execution, batch coordination, GPU utilization |
| M-apm-solutions | Consumer: forward solver trained on this corpus |
| M-artificial-stack-exchange | Future consumer: agents use forward solver |
| M-diagramprover | Separate proof-search workflow; may consume forward solver output |
