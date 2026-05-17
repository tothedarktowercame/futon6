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
2GB) and self-advancing. As of 2026-04-23 it builds 5,000-paper batches
by default (`MARK2_PAGE_SIZE=5000`) while preserving the conservative
one-request-per-three-seconds eprint fetch interval. A full batch therefore
takes about 4h10m to assemble before compression overhead. The manifest keeps
canonical arXiv URLs, but fetches are routed through `export.arxiv.org` by
default (`MARK2_EPRINT_HOST=export.arxiv.org`) in line with arXiv's
programmatic harvesting guidance.

When Rob marks a batch as pulled, the coordinator starts a background
`fill --if-room` job rather than blocking Rob's SSH command for the whole
fetch window. The same ready-target repair is also triggered when results are
registered or collected, so the lane recovers even if an operator forgets the
`pulled` step and the inbox tarball only becomes obviously stale once results
come back. The fill target is two ready inbox batches by default
(`MARK2_READY_TARGET=2`), guarded by a build lock so manual fills, cron
fills, and lifecycle-triggered fills cannot overlap. Joe seeds or repairs
state when needed; under normal operation the machine keeps itself slightly
ahead of Rob's superpod processing pace.

### Batch lifecycle

```
build → inbox → pulled/returned/collected (restore ready target) → done
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
ssh chicago mark2 fill --if-room  # ensure the configured ready-batch target
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
  --graph-embed-epochs 200 \
  --graph-embed-eval-every 5 \
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

### Open items — future mark2

Design questions the current round-robin arXiv slicing doesn't answer, to
be addressed once the extraction/reconstruction pipeline is validated at
scale.

| Item | Why it matters | Triggering condition |
|---|---|---|
| **Topic-targeted batch cutting** | mark2 currently slices arXiv in round-robin order. Batch-N's content is uncorrelated with any specific downstream evaluation target (APM prelim topics, Mathlib/LeanDojo coverage, FrontierMath problem areas). Eventually, evaluating the forward-model claim (PREREG Claim 4) requires a batch whose content is topic-aligned to the evaluation target — otherwise the forward model has no training signal in the relevant direction. Open design questions: where do the topic filters come from (human-curated vs learned from broad-batch reconstructions)? how do we avoid overfitting the forward model to the filter's biases? does the topic filter itself need a preregistration protocol? | After ≥ 3 broad batches land cleanly and stage 11 scores are in the usable range, cut a first topic-targeted batch. |
| **Cross-batch vocabulary consolidation** | Each batch's 5c output grows the technique vocabulary, but without a canonicalization pass the same technique appears under multiple surface forms across batches ("Borel completion adjunction" vs "Borel completion" vs "completion adjunction"). Long-term corpus quality depends on canonicalization. | When the cumulative technique vocabulary across all batches exceeds ~10K terms. |
| **Batch-aware reproducibility** | Each batch's pipeline state (library version, prompts, code SHA) is recorded per-run, but a clean `batch-N-pipeline-snapshot.tar.gz` that freezes the state of all four learning-loop channels would make post-hoc replay possible. | Whenever a batch's results are surprising enough that we want to re-run it with the exact original pipeline. |

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

## Burnin Checkpoint — batch-001 (2026-04-15)

Rob ran batch-001 on the superpod (8× A100 80GB, one node, 16 CPU procs).
Observed problems and the changes made before batch-002:

**Observed on batch-001:**

1. Stage 2 (BGE embeddings) pinned to a single GPU — `SentenceTransformer.encode()` with `device="cuda"` is single-device despite the docstring claim of auto-distribution. 11 min wall time where 8 GPUs were sitting idle.
2. Stage 2 re-ran from scratch on job restart even though `embeddings.npy` already existed. `stage_status` is an in-memory dict, re-initialized every invocation; there was no on-disk resume check.
3. Stage 7 (thread-performative LLM classification) emitted `"You seem to be using the pipelines sequentially on GPU"` — offender was `classify_thread_performatives_llm_batch` passing a `list` of prompts per outer batch rather than a `Dataset`. Interleaved with stage 6 output because both share `_ensure_llm_pipeline`.
4. Python 3.12 `SyntaxWarning: invalid escape sequence '\i' / '\s' / '\m'` at `<unknown>:N` — transformers' jinja2 chat-template compilation under 3.12, cosmetic.

**Changes landed (futon6 @ 2026-04-15):**

- `src/futon6/stackexchange.py` — `compute_qa_embeddings` gained `num_workers: int = 1`. When >1 and device is cuda, uses `SentenceTransformer.start_multi_process_pool(target_devices=["cuda:0",...,"cuda:{N-1}"])` + `encode_multi_process`. Single-device path untouched for laptop mode.
- `scripts/superpod-job.py` — new `--embed-workers N` flag (default 1). For batch-002 Rob should pass `--embed-workers 8`. Expected: stage 2 drops from ~11 min to ~90s on 8× A100.
- `scripts/superpod-job.py` — stage 2 now checks `outdir/embeddings.npy` before running; if present and row-count matches `len(pairs)`, it's loaded and the stage manifest records `resumed: true`. Row-count mismatch (batch changed) triggers recompute with an explanatory log line. Force-rerun = `rm embeddings.npy`.
- `scripts/superpod-job.py` — `classify_thread_performatives_llm_batch` refactored to a `TorchDataset` pass, matching stages 3 and 6. HF warning gone; throughput improves because the pipeline streams instead of restarting per outer batch.
- `scripts/superpod-job.py` — `warnings.filterwarnings("ignore", category=SyntaxWarning, message=r"invalid escape sequence.*")` at import. Targeted: doesn't mask real syntax issues in pipeline code.

**Batch-002 invocation:**

```bash
python ~/futon6/scripts/superpod-job.py \
  --arxiv-jsonl batch-002.jsonl --site arxiv.math \
  --output-dir ./output/ \
  --embed-workers 8 \
  --llm-gpu-workers 8 \
  --llm-loader-workers 16
```

Rob's Slurm `short` queue allocation exposes 16 CPU cores to the job via
cpuset affinity. Do not size feeders from the physical node; use the job
affinity (`SLURM_CPUS_PER_TASK` / `os.sched_getaffinity(0)`). GPU data
parallelism does not remove the need to feed batches with enough Python
workers, and pure-CPU stages should use multiprocessing where their work is
embarrassingly parallel.

**Still open (deferred from this checkpoint):**

- Stage 9b R-GCN training is still single-GPU. Mark 2's stated goal (hard negatives, non-collapsed embeddings) doesn't require DDP yet at batch sizes we've tested; defer until training signal is proven on batch-002/003 outputs.

## Follow-up Checkpoint — Rob's throughput feedback (2026-04-16)

Rob's second burn-in pass caught two concrete throughput issues:

1. Eight GPUs were visible, but the runner did not also raise the CPU-side
   feeder worker count. On the Slurm `short` queue the usable CPU budget is
   16 cores, even though the node has more physical CPUs; code must honor
   affinity rather than `os.cpu_count()` alone.
2. Stage 5c/5d still invoked transformers pipelines once per paper, producing
   the HF warning: "You seem to be using the pipelines sequentially on GPU."
   The correct shape is Dataset-backed pipeline calls, as already used by
   stages 3, 6, and 7.

**Changes landed (futon6 @ 2026-04-16):**

- `scripts/superpod-job.py` — `--embed-workers` now defaults to auto: all
  visible GPUs for CUDA embeddings, one worker otherwise. Explicit
  `--embed-workers 8` remains fine and documents intent.
- `scripts/superpod-job.py` — new `--llm-gpu-workers N`; default is auto
  all visible GPUs for local LLM stages (3, 5c, 5d, 6, and legacy 7).
  Correction: this is replica fanout, not DeepSpeed or PyTorch DDP. It can
  make `nvidia-smi` show one Python model worker per GPU, but it must not be
  represented as satisfying Rob's DDP/DeepSpeed requirement.
- `scripts/superpod-job.py` — new `--llm-loader-workers N`; default is
  `min(16, Slurm/cpuset CPU affinity)`, with `LLM_LOADER_WORKERS` as an env
  override and `0` for inline loading.
- `src/futon6/technique_ner.py` and `src/futon6/paper_hypergraph.py` —
  added Dataset-backed batch helpers for Stage 5c and 5d LLM arms.
- `scripts/superpod-job.py` — Stage 5c and 5d now call the LLM in batched
  Dataset chunks instead of one pipeline invocation per paper.
- `scripts/superpod-job.py` — stages 3, 6, and 7 now pass the same explicit
  loader worker count into their Dataset-backed pipeline calls.
- `scripts/superpod-shard.py` — sharded mode splits the 16-core/job affinity
  budget across shard processes by default, so 8 shards use 2 feeder workers
  each unless overridden.

**Correction after Rob's DDP objection (2026-04-16):**

The `--llm-gpu-workers` path above is not the DDP/DeepSpeed fix Rob asked for.
It is only independent per-GPU model replication with parent-side task fanout.
That may improve throughput for independent inference items, but it is not a
substitute for a real `torchrun`/DeepSpeed distributed LLM path. Treat true
DDP/DeepSpeed LLM execution as still open.

**Correction after Rob's Dataset warning persisted (2026-04-16):**

Rob was also right that the HF warning still mattered. The previous change gave
each pipeline call a Dataset, but transformers emits the CUDA warning after
repeated `Pipeline.__call__` invocations regardless. `scripts/superpod-job.py`
now bypasses the transformers Pipeline API for local LLM inference: it loads
`AutoModelForCausalLM`/`AutoModelForSeq2SeqLM` directly and runs a Dataset /
DataLoader feeder into `model.generate`. Chunked resumability remains, but the
sequential Pipeline call path is gone.

**Correction after Rob's Stage 9b loss feedback (2026-04-17):**

Rob observed that the R-GCN training completed quickly at 50 epochs and the
loss did not look minimized. Stage 9b now defaults to 200 epochs in production,
while laptop mode still drops to 10 epochs for local iteration. Validation
retrieval is evaluated every 5 epochs by default via
`--graph-embed-eval-every N` (2 in laptop mode), and the runner prints
initial/final/best loss plus the last-10-epoch loss delta so a run log shows
whether training is still materially improving at the end.

## Checkpoint — Learn As We Go (2026-04-15)

Before lifting a finger on stage 6 multi-GPU, we reframed what stage 6 *is* in the arXiv setting. The conclusion: stage 6 isn't a retrieval enrichment — it's the **synthetic training corpus generator for a forward problem-solving model**. Backward reconstruction at arXiv scale (570K papers) produces grounded (problem, techniques, terminology, patterns, result) tuples; a forward model trained on that corpus is the actual research payoff. Retrieval quality is a side effect of doing reconstruction well.

### Why the old stage 6 doesn't fit papers

The arXiv adapter (`load_arxiv_pairs` in `src/futon6/stackexchange.py:683`) is a kludge: each paper becomes a fake Q/A where `question = title + abstract` and `answer = abstract` duplicated. Stage 6 then feeds the LLM the abstract twice and asks for the S←Q←A/xiang/salience/arrow reconstruction — a pedagogical move that doesn't apply. Papers have no questioner, no epistemic asymmetry between question and answer. The frame is incoherent.

### The reframed stage 6 principle

We don't trust authors. The paper is **evidence**, not testimony. Even a well-written paper that says "this solves problem X" is one input among several; the *real* problem a paper solves is reconstructed from what the paper actually does — its techniques, its terminology, its argumentative skeleton — and compared against what the author claimed.

**Why terminology is load-bearing**: technique vocabulary constrains the problem-class. Ruler-and-compass is the clean base case (Wantzel ⇒ quadratic closure of ℚ — the terminology *is* the solvability envelope). For fuzzier techniques the envelope is softer but the move is the same: let the techniques tell you what's in the solvability class, compare against the author's stated framing, and treat the gap as diagnostic.

### Four-layer output schema (per paper)

1. **Stated problem** — what the author says, with evidence loci. Often thin or absent.
2. **Techniques deployed** — technique-level terms (from stage 5c) with loci in the paper.
3. **Reconstructed problem-class** — from the technique vocabulary and argumentative skeleton back to the problems those techniques address, with a derivation showing why.
4. **Stated-vs-reconstructed gap** — diagnostic signal. Papers where these diverge are often the ones where the real contribution differs from the marketed one, and retrieval on technique beats retrieval on stated framing.

All four layers carry evidence pointers (section/paragraph/hypergraph-node refs). Not just for auditability — as grounding for the forward model's training signal.

### Pipeline shape for batch-002

| Stage | What it does | Status |
|---|---|---|
| 5 (existing) | Concept-level NER — unchanged | Keep |
| **5c (new)** | Technique-level NER — scope-aware multi-word term extraction | Draft |
| **5d (new)** | Paper hypergraph — argumentative skeleton + terminology lift. **Two arms kept distinct**: classical extractor, LLM-augmented extractor. | Draft |
| **6 (rewritten)** | Paper reverse morphogenesis. Consumes abstract + intro + conclusions prose + 5d hypergraph + 5c technique list. Multi-pass capable. | Draft |
| **3 (extended)** | Paper-level pattern tagging with a **growing library**: seed from futon3's flexiarg, mine paper-structure patterns as we go. | Draft |
| **11 (new)** | Sketch-and-score. Generates a structural-and-findings sketch from (reconstructed problem + techniques + patterns) and scores against the paper. | Draft |

Stage 7 (thread wiring) stays skipped for arXiv. Stage 9a hypergraph was thread-based; 5d is its paper-appropriate replacement.

### Evaluation — each paper is its own gold standard

No external labels. The primary quality criterion is **sense of inevitability**: given the reconstructed (problem + techniques + patterns), does the generated sketch match the paper's actual structure and findings? If yes, the reconstruction captured something real. If no, it missed the heart.

This operationalizes a criterion that mathematical understanding has always had — "of course, given these ingredients, you'd get this" — as a measurable per-paper metric (sketch-vs-paper: term coverage, structural match, finding coherence). Per-batch score distribution is what we track.

### The mining approach *is* ML

The pipeline is a self-improving system with four update channels, each mapped to a component:

- **Pattern library** grows when a paper has strong technique/problem coherence but weak pattern fit — that gap is a "mine a new pattern" signal.
- **Technique vocabulary** sharpens when the C-extraction misses terms the reconstruction later had to invent.
- **Hypergraph structural features** grow when argumentative hyperedges are missing — a paper whose sketch fails because the hypergraph didn't capture "theorem T depends on lemma L" tells us to add a new hyperedge type or sharpen the classical extractor.
- **Prompts tune** against the gap distribution — reconstructions systematically weak on one of the four output layers are a prompt-shape signal, not a model signal.

Two temporal scales:

- **Per-paper** (tight loop): stage 6 → stage 11 → optional refinement pass keyed on the specific weak field. Effectively a short self-play cycle.
- **Per-batch** (learning step): aggregate gaps across ~5K papers, update library/vocabulary/hypergraph-extractor/prompts. Next batch runs with a sharper pipeline.

**Corollary — training corpus curation**: the forward-model training corpus gets *score-weighted*, not just accumulated. High-inevitability reconstructions are clean training signal. Low-inevitability papers get a `reconstruction_quality` tag, used either for harder training tasks or held for later passes when the pipeline is stronger. Corpus quality tracks pipeline quality; co-improvement, not static accumulation.

### Experimental arms for batch-002

At 1% of arXiv per batch, experimental parallelism is cheap. Three dimensions running as natural experiments, kept distinct so we can read the tape afterward:

| Dimension | Arms |
|---|---|
| Stage 5d extractor | classical, LLM, both |
| Stage 6 reconstruction passes | 1, 2, 3 (fixed N = 3 for batch-002; adaptive in batch-003 once plateau shape is known) |
| Stage 3 pattern library seeds | flexiarg-only, flexiarg + mined |

Multi-pass is cheaper than it looks: the **first pass of the multi-pass arm IS the single-pass arm's output**. Paired comparisons per paper come free (minus the extra LLM cost at passes 2–3). Single-pass arm = "stop after pass-1 score."

Per-paper provenance metadata (which arm produced which output, which library version was live, which pass produced which score) is recorded in a batch-level `experiment_meta.json` so downstream analysis can disentangle the effects.

### Forward model — why this is the real payoff

- **Training signal**: (reconstructed problem-class, technique vocabulary, pattern tags) → sketched argumentative structure
- **Ground truth**: the actual paper
- **At inference**: novel problem → proposed techniques + patterns → sketched construction

570K self-validating training examples, with the evaluation metric built into the pipeline itself. Consumer mission: **M-apm-solutions** (theorem proving). The forward model turns reconstructed mining output into a candidate technique-and-structure proposer for novel problems.

### Next action

**Spec drafted:** `futon6/holes/missions/M-paper-reverse-morphogenesis.md`
(2026-04-15) covers all six components — 5c technique-NER, 5d paper
hypergraph (classical + LLM arms), stage-6 four-layer rewrite with
multi-pass, stage-3 paper-level patterns with mining protocol, stage-11
sketch-and-score, and the batch-level experiment-meta ledger.

Implementation order documented in that spec; pipeline improvements land
between batches with tight per-batch evaluation loops driving the updates.

## Post-run health and quality checks

Run through this checklist the moment a batch returns from the superpod,
*before* declaring the batch good and *before* applying per-batch
learning-loop updates. Most items read directly from the run's stdout/
stderr log and the `stage_status` manifest; a few want a 10-paper
eyeball pass.

### 1. Did the GPU utilization fix actually work?

- **Stage 2 wall time**: should be < 2 min for 5K papers on 8× A100.
  - ≈ 90 s = eight GPUs (success). ≈ 11 min = one GPU (the batch-001
    failure mode — Rob forgot `--embed-workers 8` or the flag isn't
    being honored).
  - > 3 min: check that `--embed-workers 8` was passed; check stage 2
    log line for the active worker count.
- **Stage 2 resume on restart**: if Rob ctrl-C'd mid-batch and re-ran,
  the log should say `Reusing existing embeddings.npy (shape (N, D))`
  and `stage_status.embeddings.resumed: true`. If it recomputed, the
  checkpoint logic regressed — file a bug.
- **HF "sequentially on GPU" warning absent from stderr.** If it appears
  during stage 6 or 7, Fix 2 from the batch-001 checkpoint regressed.
- **SyntaxWarnings absent.** If `<unknown>:N: SyntaxWarning: invalid
  escape sequence` floods stderr, Fix 4 regressed.

### 2. Did eprints actually load (or are we still on abstracts)?

- **Stage 5c / 5d log line**: `Text source: eprint=N, abstract-fallback=M`.
  - Healthy: `M < 100` on a 5K-paper batch (< 2%).
  - `M > 500`: check `--paper-eprint-dir` path, check `eprints/`
    integrity, check for corrupt tarballs. Loader's per-file status
    codes (`missing`, `tar-read-error`, `unusable`) are visible if
    you rerun with a smaller sample.
- **Stage 5d `with_claim_blocks`**: % of papers whose hypergraph
  contains at least one numbered theorem/lemma/proposition.
  - With eprints loaded for pure-math batches: expect 70–95%.
  - With eprints loaded but `with_claim_blocks` low: papers may use
    non-standard environments (`\newtheorem`, custom names). Look at
    a weak sample to find out.
  - Near-zero: eprints didn't load. Cross-check against §1.
- **`stage_status.paper_hypergraph.text_source_counts`** records this
  machine-readably — prefer that over grepping stdout for automation.

### 3. Is the natural experiment actually live?

- **Stage 5c arm balance** (`classical=A, llm=B, both=C`):
  - Healthy: `C / (A+B+C) > 0.2`. Means both arms are finding overlapping
    terms — validates that they're extracting from the same signal.
  - `B = 0`: LLM arm silently failing. Check for pipeline errors at
    stage 5c; check `_ensure_llm_pipeline` actually loaded a model.
  - `A / B > 5`: classical is dominant — expected early, but if it
    persists across batches, LLM prompt needs work.
  - `A = 0` with nonzero `B`: classical is too strict on this corpus
    (expected for non-eponymous technique-heavy sub-disciplines).
    That's iteration fuel — log it for the per-batch update.
- **Stage 5d edge provenance** (`classical=X, llm=Y, both=Z`):
  - `Z > 0`: LLM confirms classical structural edges. Good.
  - `Y > 0`: LLM is adding implicit edges the classical parser missed
    (motivation-link, implicit derivation). This is the main reason
    the LLM arm exists — if `Y = 0` across the batch, the prompt
    isn't eliciting net-new edges.
  - `Y / X` ratio is the cleanest "how much is the LLM contributing
    uniquely?" summary. Track across batches.

### 4. Sanity spot-check (10 papers, ~15 minutes)

Pick 10 random papers from the batch. For each:

- **`techniques.json`**: read the extracted techniques. Are they real,
  specific mathematical moves, or generic/noisy phrases?
  - False positives worth logging: sentence fragments, prepositional
    phrases, things that aren't techniques.
  - False negatives worth logging: techniques the paper clearly uses
    that didn't get extracted. These drive classical-pattern and LLM
    few-shot improvements.
- **`paper-hypergraphs.json`**: pick 3 papers with claim blocks. For each,
  read the hypergraph and check:
  - Does every numbered theorem have a matching derivation edge to a
    proof? (Miss rate = parsing failure.)
  - Do `definition-use` edges point from definitions to later mentions?
  - Do citation-grounding edges connect proofs to their cited
    references via the technique nodes actually used?
- Record observations in a small batch-N-spotcheck.md — these become
  inputs to the per-batch learning loop.

### 5. Regression-style distribution checks

- **Techniques per paper**: distribution across the batch.
  - Pure-math papers (10–30 pages): typically 5–50 techniques.
  - < 3 per paper on average: extraction too strict.
  - > 200 per paper: extraction too noisy.
- **Nodes / edges per paper** in `paper-hypergraphs.json`: should roughly
  track paper length. Zero-edge papers are likely eprint-fallback or
  non-standard environments.
- **Per-section term density**: techniques should cluster in
  definitions, theorem statements, and proofs — not in references or
  acknowledgements. A high-density references section suggests the
  LaTeX `thebibliography` environment leaked into the body.

### 6. Failure forensics

Look for these in `stage_status` and logs when something's off:

- Any stage marked `skipped` that shouldn't be. Expected skips:
  thread stages (`--skip-threads` auto-set for arXiv), optional 5b.
  Unexpected skips (NER kernel not found, embeddings skipped under
  `--moist-run` without intent) are batch-invalidating.
- `stage_status.<stage>.resumed: true` when you didn't expect a
  restart — means a previous run left partial outputs; verify they're
  current-batch outputs, not leftovers from another run.
- `n_llm_new_edges = 0` uniformly across all papers: LLM arm
  structurally not firing (pipeline error swallowed, prompt producing
  empty arrays, node-ID hallucination filter rejecting everything).

### 7. Before advancing the learning loop

After the batch passes §§1–6, and before you apply per-channel updates:

1. Per-paper scores available? (Once stage 11 lands; for batch-002's
   first run manual spot-check substitutes.)
2. Score distribution agrees with spot-check intuition — papers you
   flagged as weak in §4 have low scores, not high ones.
3. Any systematic failure modes — e.g., all papers in a sub-discipline
   cluster at the low end of the score distribution. If yes, that's a
   per-channel signal (pattern library gap, technique vocabulary gap,
   hypergraph-feature gap, or prompt gap per §"Learn As We Go").
4. Per-channel update proposals written down *before* applying, so
   there's a rollback target if the next batch regresses.
5. Proposals applied to code, tagged with the batch number in commit
   messages / library-version records, so we can attribute later
   effects.

### Automation

Most of §§1–3 and §5 are mechanical enough to automate. A
`scripts/post-run-health-check.py` that reads `stage_status` + the
stdout log + a sample of output JSONs and emits a pass/warn/fail
report would pay for itself in a few batches. Stub it as a next
action after stage 11.
