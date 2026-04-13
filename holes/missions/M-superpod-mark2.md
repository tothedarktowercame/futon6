# Mission: Superpod Mark 2 — All of arXiv

**Date:** 2026-03-29 (started), 2026-04-12 (rewritten)
**Status:** MAP
**Owner:** Rob (superpod runs), Joe (pipeline code + evaluation)
**Repos:** futon6 (pipeline), futon3c (downstream retrieval), apm-lean (evaluation)

## What Mark 1 showed

The pipeline (`scripts/superpod-job.py`, 4800 lines, 10 stages) ran on
9,916 arXiv math.CT papers in 614 seconds on GPU. Everything works
end-to-end: parse → embed → NER → hypergraphs → FAISS index.

We used the output for retrieval-augmented theorem proving (the
learn-to-swim canaries, documented in `apm-lean/report/apm-canaries.tex`).
Result: BGE text retrieval found a paper using the identical construction
for canary C7 (Pettis integral as monad algebra map, similarity 0.82).
For standard techniques (C5, C6), retrieval added nothing. **Retrieval
helps when the technique is non-obvious and domain-specific.**

The R-GCN graph embeddings (stage 9b) collapsed — all cosine similarities
~1.0. Switching to BGE text embeddings (stage 2) was a one-line fix,
zero additional compute. BGE is the working retrieval backend now.

## What Mark 2 is

**Process all of arXiv** (not just the 9,916-paper math.CT slice),
with pipeline improvements based on what Mark 1 taught us.

This is a scale-up with targeted fixes, not a research project. The
pipeline works. The fixes are:

1. **Better training signal for stage 9b.** Hard negatives instead of
   random negatives. Use BGE similarity bands (0.7–0.8) to find papers
   that are textually confusable but structurally different. The R-GCN
   architecture is fine; the training data was too easy.

2. **Technique-level NER (stage 5).** The current NER kernel has
   19,236 concept-level terms ("functor", "adjunction"). The
   learn-to-swim finding: retrieval helps for technique-level terms
   ("Borel completion adjunction", "Pettis integral as algebra map").
   Scope-aware multi-word extraction would capture these.

3. **Hybrid embeddings (stage 10).** BGE captures text similarity.
   R-GCN (once fixed) captures structural similarity. Combine them
   in the FAISS index rather than forcing a choice. Conservative
   approach: concatenate frozen embeddings.

## What exists already

| What | Where | State |
|------|-------|-------|
| 10-stage pipeline | `futon6/scripts/superpod-job.py` | Working, tested on math.CT |
| ArXiv input adapter | `--arxiv-jsonl` flag | Working (added Feb 2026) |
| Laptop mode | `--laptop` flag | Working (CPU-friendly dev iteration) |
| arXiv manifest | `storage/arxiv-manifest/arxiv_manifest.sqlite` | 570,209 math papers, all pending |
| R-GCN module | `futon6/src/futon6/graph_embed.py` | Working code, bad training signal |
| Embedding audit | `futon6/scripts/audit-graph-embeddings.py` | Validates collapse/quality |
| Review pair generator | `futon6/scripts/generate-review-pairs.py` | Tier-stratified sampling |
| NER kernel | `futon6/data/ner-kernel/terms.tsv` | 19,236 terms from PlanetMath |
| BGE retrieval bridge | `futon3c/scripts/corpus_ws_bridge.py` | Live, serves downstream consumers |
| Canaries evaluation | `apm-lean/report/apm-canaries.tex` | 7 problems, 3 with retrieval assessment |
| LeanDojo pilot-20 | `futon3c/data/leandojo-pilot-20/` | 20 APM problems with Mathlib cross-refs |
| Mark 1 outputs | on superpod storage | 9,916 papers, all stages, hypergraphs + embeddings |

## Data staging

The arXiv manifest (`storage/arxiv-manifest/arxiv_manifest.sqlite`)
contains 570,209 math papers across all categories (math.AP: 56K,
math.CO: 52K, math.PR: 43K, ... down to smaller categories).

The `mark2` coordinator (`scripts/mark2`) runs on Joe's Chicago
Linode server and manages the three-party relay:

```
manifest (570K papers)
  → mark2 builds batch N (5,000 papers + eprint sources, tarball)
  → Rob scps batch from Chicago, runs pipeline on superpod
  → Rob scps results back to Chicago
  → Joe scps results from Chicago to local storage
  → repeat
```

The coordinator is storage-aware (configurable inbox budget, default
2GB) and self-advancing: when Rob marks a batch as pulled, the next
batch builds automatically. Joe seeds the first batch; after that
the machine runs itself.

### Batch lifecycle

```
build → inbox → pulled (auto-builds next) → returned → collected → done
```

### Rob's workflow

```bash
# See what's ready
ssh chicago mark2 next

# Pull batch
scp chicago:~/mark2/inbox/batch-003.tar.gz .
ssh chicago mark2 pulled 3        # triggers batch-004 build

# Unpack and run pipeline
tar xf batch-003.tar.gz
cd batch-003
python ~/futon6/scripts/superpod-job.py \
  --arxiv-jsonl batch-003.jsonl \
  --site arxiv.math \
  --output-dir ./output/

# Upload results
tar czf results-003.tar.gz output/
scp results-003.tar.gz chicago:~/mark2/outbox/
ssh chicago mark2 returned 3
```

The pipeline handles checkpointing — if interrupted, re-run with
the same `--output-dir` to resume from the last completed stage.

### Joe's workflow

```bash
ssh chicago mark2 status          # overview
scp chicago:~/mark2/outbox/results-003.tar.gz .
ssh chicago mark2 collected 3     # cleans up outbox
```

### Iterating on stage 9b

Once a batch has been processed, re-run just stage 9b with different
training parameters (stages 1–8 outputs are reused):

```bash
python scripts/superpod-job.py \
  --arxiv-jsonl batch-003.jsonl \
  --site arxiv.math \
  --output-dir ./output/ \
  --skip-stages 1,2,3,4,5,6,7,8,9a,10
```

### LeanDojo (separate workload)

M-diagramprover, not Mark 2. The pilot-20 problems are at
`futon3c/data/leandojo-pilot-20/`. Shares the superpod but runs
independently of the mining pipeline.

## Open items

| Item | Owner | Status |
|------|-------|--------|
| Deploy mark2 + manifest to Chicago server | Joe | Next |
| Add `--skip-stages` flag if not present | Joe | Check |
| Pipeline improvements: hard negatives (9b), technique NER (5), hybrid (10) | Joe | In progress (math.CT pilot was the round-trip) |
| First batch test run on superpod | Rob | After mark2 deployed |

## How we'll know it worked

1. Run the improved pipeline on a corpus significantly larger than
   math.CT (target: all arXiv math, ~200K+ papers).

2. Stage 9b embeddings no longer collapse: pairwise cosine similarity
   std > 0.10, validation accuracy < 90% (non-trivial task).

3. On the learn-to-swim canaries (or LeanDojo pilot-20), structural
   or hybrid retrieval surfaces technique-relevant papers that BGE-only
   retrieval misses.

4. The FAISS index serves downstream consumers (`corpus_ws_bridge.py`)
   with measurably better retrieval precision than BGE-only Mark 1.

## Related missions

| Mission | Relationship |
|---------|-------------|
| M-apm-solutions | Consumer: uses retrieval for theorem proving |
| M-diagramprover | Shares superpod, separate workflow (proof search not mining) |
| M-artificial-stack-exchange | Future consumer of improved retrieval |
| M-distributed-frontiermath | Future consumer for FrontierMath campaigns |
