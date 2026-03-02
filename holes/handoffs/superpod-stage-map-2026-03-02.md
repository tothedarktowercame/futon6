# Superpod Stage Map (Math + MO)

Date: 2026-03-02
Artifacts:
- /home/joe/code/storage/math-processed-gpu
- /home/joe/code/storage/mo-processed-gpu

Goal of this note: for each stage, capture expected outcome, observed reality, signal value, and next-run improvements.

## Executive view

- Strong signal now: Stages 1, 2, 3, 5, 8, 9a, 9b (with caveats), 10 infrastructure.
- Weak/blocked signal now: Stage 4 (skipped), Stage 5b (not run), Stage 6 (output contract failures), Stage 7 categorical correctness.
- Immediate usefulness today: corpus-scale structured feature extraction and retrieval infra are working; categorical/causal interpretability layers are not yet robust enough for decision use.

## Stage-by-stage map

### Stage 0: Data ingestion / sharding orchestration
- Expectation: shardable full run with deterministic merge.
- Reality: 8-shard runs completed and merged for both datasets (`manifest.json` has `merged: true`).
- Signal: good operational reliability for large runs.
- Improve next run: write explicit `skip_reason` fields into manifests (for auto-skipped stages) to avoid ambiguity.

### Stage 1: Parse XML to QA entities
- Expectation: produce clean QA corpus with metadata.
- Reality:
  - math: 805,200 QA pairs
  - mo: 95,321 QA pairs
  - with LaTeX: math 783,957 (97.36%), mo 88,772 (93.13%)
- Signal: strong; sufficient data mass and math density.
- Improve next run: add explicit malformed/empty-body counters and duplicate detection in manifest stats.

### Stage 2: Dense embeddings (BGE-large)
- Expectation: one dense vector per entity.
- Reality:
  - math embeddings shape: (805200, 1024)
  - mo embeddings shape: (95321, 1024)
- Signal: strong; embeddings are complete and aligned with entity counts.
- Improve next run: add embedding quality checks (nearest-neighbor sanity sample, cosine distribution guardrails).

### Stage 3: LLM pattern tagging
- Expectation: assign tactical pattern tags from fixed palette.
- Reality:
  - math: 805,200 records; non-empty tags 789,375 (98.03%); avg 3.26 tags/entity
  - mo: 95,321 records; non-empty tags 88,777 (93.13%); avg 3.20 tags/entity
- Signal: high coverage and plausible density; useful as weak supervision feature.
- Improve next run: add calibrated precision audit (human labels on 200 sampled rows) and per-pattern confusion stats.

### Stage 4: Clustering
- Expectation: global motif clusters over embeddings.
- Reality: skipped in shard mode by design (`--shard-index/--num-shards` auto-sets `skip_clustering=true`), and no post-merge clustering artifact present.
- Signal: none from this run.
- Improve next run: run Stage 4 once post-merge on full embeddings and emit cluster-level metrics.

### Stage 5: NER + scope detection
- Expectation: recover terms and logical scope structure.
- Reality:
  - NER coverage: math 99.58%, mo 99.98%
  - Scope coverage: math 78.06%, mo 81.89%
  - Open-ner candidate writes: math 1,102; mo 63
- Signal: strong extraction backbone; scope density is substantial.
- Improve next run: precision audit for scope typing (quantifier/binder/constraint) and false-positive controls for open NER.

### Stage 5b: Distinctor/MIT labeling (optional)
- Expectation: disambiguation and stronger identity/contradiction signal.
- Reality: not run (`stage5b_stats: null`).
- Signal: none in this run.
- Improve next run: enable only when downstream tasks need identity/consistency adjudication; otherwise keep optional.

### Stage 6: Reverse morphogenesis (S <- Q <- A)
- Expectation: strict JSON output for each entity.
- Reality:
  - parse success: math 77,391/805,200 (9.61%), mo 6,391/95,321 (6.70%)
  - dominant failures: unclosed JSON and invalid JSON
- Signal: content quality in parsed subset is good, but availability is too low for broad use.
- Improve next run:
  - enforce schema-constrained decoding (primary fix)
  - keep parser hardening as secondary safety net
  - parser-only simulation shows uplift but not enough (math ~47.6%, mo ~39.1%; still far from 90%).

### Stage 7: Thread wiring + CT verification
- Expectation: discourse wiring with category-theoretic coherence.
- Reality:
  - threads processed: math 1,068,196; mo 138,933
  - CT verification aggregate (full files):
    - avg overall score: math 0.0777, mo 0.0746
    - categorical_consistent_rate: math 3.31%, mo 3.87%
    - port_compatible_rate: math 4.67%, mo 5.93%
    - iatc_aligned_rate: math 71.83%, mo 66.06%
- Signal: IATC/discourse labeling has signal; categorical structural correctness is currently weak.
- Improve next run:
  - split outputs into two confidence tiers (discourse vs categorical)
  - add strict typing/compatibility gates before accepting categorical edges
  - fail fast on low categorical consistency in preflight mode.

### Stage 8: Expression surface parsing (LaTeX -> s-exp)
- Expectation: high parse coverage on math expressions.
- Reality:
  - math parse rate: 99.00% (33,053,654 / 33,386,230)
  - mo parse rate: 99.14% (5,023,935 / 5,067,765)
- Signal: very strong; this is production-viable at extraction level.
- Improve next run: stratify fallback errors by construct family and surface top missing grammar cases.

### Stage 9a: Hypergraph assembly
- Expectation: one non-trivial typed hypergraph per thread.
- Reality:
  - hypergraphs produced: math 1,068,196; mo 138,933 (100% of processed threads)
  - avg size: math 57.43 nodes / 69.97 edges; mo 71.20 nodes / 89.42 edges
- Signal: strong assembly throughput and non-trivial graph structure.
- Improve next run: add invariants on role integrity/connectivity and report violation rates.

### Stage 9b: Graph embedding
- Expectation: train thread-level structural embeddings for retrieval.
- Reality:
  - embedded threads: math 1,068,183; mo 138,929; dim=128
  - very high validation metrics (acc@1 near 1.0)
- Signal: training converges and embeddings are numerically coherent.
- Caveat: current validation may be too easy (possible leakage/easy negatives).
- Improve next run: harder split strategy and external retrieval benchmark (human-judged P@k / nDCG).

### Stage 10: FAISS structural index
- Expectation: fast nearest-neighbor retrieval over graph embeddings.
- Reality:
  - index built at full vector count for both datasets
  - sample query sims look plausible (roughly 0.65 to 0.78)
  - review files exist but are mostly/fully unjudged
- Signal: infrastructure complete; utility still unproven without relevance judgments.
- Improve next run: complete judgment pass for review sets and compute retrieval metrics.

## What is already useful vs not yet

Useful now:
- Corpus-scale extraction and graph construction (Stages 1,2,3,5,8,9a)
- Embedding + index infrastructure (Stages 9b,10) as an experimental retrieval backend

Not yet trustworthy for decision layers:
- Stage 6 as broad feature source (format reliability)
- Stage 7 categorical assertions (low structural consistency)
- Stage 10 relevance quality (until judged evaluation exists)

## Suggested next session order (before deep fixes)

1. Finish evaluation map artifacts (not code changes):
   - judge existing `review-50.json` files
   - add 100-row precision audits for Stage 3 and Stage 5 scope typing
2. Run one schema-constrained Stage 6 moist sample (100 rows, strict pass gate)
3. Decide priority branch:
   - branch A: retrieval usefulness (Stages 9b/10 evaluation)
   - branch B: semantic reliability (Stages 6/7 hardening)

