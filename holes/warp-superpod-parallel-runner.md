# Fork: superpod-parallel substrate runner (the production warp_run)

*Author: claude-1, 2026-06-17. A fork of claude-2's `warp_run` (WARP-ORCH-2, the laptop
make-like serial runner). Motivation (Joe): the substrate-build cascade is **not a one-off** —
it must run **per arXiv classification batch**, so Rob needs a runner for it too; and the
superpod has many CPUs, so the embarrassingly-parallel stages should be sharded to cut
wall-clock. NOT yet dispatched — design for review. Coordinate with claude-2 (owns `warp_run`
+ the stage scripts); do not disturb the in-flight laptop rebuild.*

## 0. Why fork (laptop runner ≠ production runner)

claude-2's `warp_run` is **dev/correctness-first**: serial, make-like, shells out to the stage
scripts in dependency order, with a content-hash drift gate. It is the right tool for *one*
supervised rebuild on the laptop (ETA 1–2h; concordance alone >30 min — which is why the 30-min
Codex limit can't run it). But the same cascade is a **per-batch production operation**: every
new arXiv classification batch needs its substrate (concordance → … → encyclopedia/phylogeny)
built or extended. On the superpod (A100×160 but, more relevantly here, **many CPU cores +
SLURM**) the parallelizable stages should be **sharded**, turning the >30-min serial bottleneck
into ~minutes. This fork is the **scale/production** sibling — same stage scripts, parallel scheduler.

## 0b. The orchestrator ALREADY EXISTS — reuse `superpod-shard.py` (don't build a scheduler)

**Found by scouting (Joe's prompt: "unless the answers are present in earlier superpod scripts").
They are.** `scripts/superpod-shard.py` is Rob's generic sharded-pipeline orchestrator — it already
does everything §2's "build a SLURM scheduler" proposed, and encodes his conventions:
- `run`: **partition input → launch N parallel device-pinned shard jobs → typed merge → post-merge
  stages**. The `shard i: CUDA_VISIBLE_DEVICES=X → <out>-shard-i` console line is literally this script.
- **Device policy = deferred to Rob's authority.** `cmd_run` respects `CUDA_VISIBLE_DEVICES` if a
  launcher set it; `mfuton-superpod-gpu-policy.sh` is that launcher (exports the SLURM-allocated GPU
  IDs; authority = `mfuton/agent_skills/development/superpod/current-job-gpus.sh`; single-GPU on
  dev/short). **So we do NOT hardcode devices — we run under the gpu-policy gate.**
- **Merge framework exists** — `merge_json_array_files` / `merge_json_lists` / `merge_jsonl_files` /
  `merge_npy_files` / `merge_stats`, dispatched by filename in `cmd_merge`. Add a new mergeable output
  = register its merger.
- **Knobs exist:** `--num-shards`, `--output-dir`, `--embed-batch-size`, `--graph-embed-*`,
  `--llm-loader-workers`, `--input-dir`, `--skip-post-merge`, `extra_args` pass-through, and
  `_default_loader_workers_per_shard` (auto-splits the cpuset across shard procs).

**Consequence — the fork is now SMALL and is "register warp into the existing runner", not a rewrite:**
1. Wire the warp substrate stages (concordance/def_snippets/concept_usage) as `superpod-shard.py`'s
   per-shard job (partition = eprints corpus; each shard runs the stage scripts on its slice, via the
   `--eprints/--out` hooks claude-2 is adding).
2. **Register the 3 warp-specific mergers** in the merge registry: concordance = **sum counts**,
   concept_usage = **dict-union** (disjoint paper sets), def_snippets = **list union + stable-sort by
   (paper-id, offset)** — the §5b trap; the existing `merge_json_lists` may need the stable-sort variant.
3. Device/GPU policy + shard-count = **Rob sets them at invocation** via the existing flags + the
   mfuton gpu-policy. Nothing for us to default-guess.

This answers §5 directly: **the sharding details Joe couldn't recall are already codified by Rob** —
`--num-shards` is the knob, device assignment is mfuton/SLURM (single-GPU on dev/short), merge is the
existing typed registry. §6b/§6c's flags largely map onto flags `superpod-shard.py` *already has*.

## 1. The DAG (from `warp_run.py` `SPINE_STAGES`) + where the parallelism is

```
S1a concordance.py        → concordance.json          ⟵ THE BOTTLENECK (>30 min). PER-DOCUMENT map.
S1b/S1c, S2 (defined-index / supporting)
S3  hitlist.py            (concordance, defined-index) → hitlist.json
   ├─ S4a def_snippets.py (hitlist, eprints)          → def-snippets.json     ⟵ per-document map
   └─ S4b concept_usage.py(hitlist, eprints)          → concept-usage.json    ⟵ per-document map  [the drift one]
S5  concept_graph.py      (hitlist, def-snippets)     → concept-graph.json
S4c embeddings            (hitlist, def-snippets, concept-graph)
S6t build_term_prior.py   → term-prior-ct.json
S6b build_concept_encyclopedia.py (term-prior, def-snippets, concept-graph) → encyclopedia
S6a phylogeny             (cite-resolution)            → concept-phylogeny.json  ⟵ INDEPENDENT branch
```

**Three kinds of parallelism, in priority order:**
1. **Per-document map stages → shard the corpus.** `concordance` (S1a), `def_snippets` (S4a),
   `concept_usage` (S4b) all scan the eprints corpus document-by-document. Split the corpus into
   N shards, run N workers, **reduce/merge** the partials. Concordance is the >30-min item →
   biggest win (≈ T/N + merge). **This is the single highest-leverage change.**
2. **Independent DAG branches → run concurrently.** `phylogeny` (S6a) depends only on
   cite-resolution, not on the concordance spine → runs in parallel with S1a..S6b from the start.
   `def_snippets` (S4a) ∥ `concept_usage` (S4b) are siblings off hitlist.
3. **(non-goal for v1)** intra-stage threading inside a single script — leave the scripts alone;
   parallelize *across* shards/branches in the scheduler.

## 2. Sharding strategy (superpod = SLURM + many CPUs)

- **Map:** a SLURM **CPU array job** (cf. the existing `superpod-job.py` GPU pattern, but CPU
  partition). Array index = corpus shard; each task runs the stage script over its shard →
  `<stage>.partial.<shard>.json`. Sized to the partition's core count (the superpod has lots).
- **Reduce:** a deterministic **merge** step per map-stage (`concordance.json` =
  merge(partials)). Merge must be **order-independent** (sort keys, deterministic tie-break) so
  the sharded output is **byte-identical to the serial output** — this is the acceptance bar
  (preserves claude-2's drift-gate semantics; same content-hash whether serial or sharded).
- **Schedule:** a thin DAG driver submits the SLURM arrays, waits on the barrier per map-stage,
  fires the reduce, then the next stage; independent branches (phylogeny) submitted up front.

## 3. Rob-facing contract (per arXiv batch)

`warp_run_superpod.py --corpus <eprints-batch> --out <substrate-dir> [--shards N] [--resume]`:
- **Reproducible + deterministic** (sharded output == serial output, content-hashed).
- **Resumable** (a shard/stage that completed is skipped — the make-like `runnable()` check
  claude-2 already has, extended to partials).
- **Per-batch** — points at one arXiv classification batch's eprints; extends the substrate.
- Emits the same artifacts the checker spine + R2d + CAS-CERT consume, so **nothing downstream
  changes** — it's a faster, scalable *producer* of the same substrate.

## 4. Boundary with claude-2's `warp_run` (no duplication)

- **Reuse the stage scripts verbatim** (`warp_concordance.py`, `warp_def_snippets.py`, …) — the
  fork is a *scheduler*, not a reimplementation. Each script must accept a `--shard`/`--corpus-
  subset` arg (small additions to the scripts, claude-2's call) OR the scheduler feeds it a
  pre-sharded corpus path.
- **Keep claude-2's drift gate** — the reduce step's content-hash IS the drift signal.
- **Dev vs prod:** `warp_run` stays the laptop correctness oracle (run it on a small corpus to
  validate the sharded runner reproduces it); `warp_run_superpod` is the batch/scale tool.

## 5. Open questions (for Joe / claude-2)
- **Shard granularity** — fixed N, or per-core auto? And the merge cost: does
  concordance-merge stay cheap as N grows (it's a dict union; should be ~linear)?
- **Determinism under sharding** — confirm each map-stage's merge is genuinely order-independent
  (concordance + concept-usage are dict-keyed; def-snippets is a list — needs a stable sort).
  This is the load-bearing correctness property (it's what lets the drift gate still work).
- **Do the stage scripts need a `--shard` arg, or do we pre-split the corpus on disk?** (The
  latter needs no script changes — pre-shard eprints into N dirs, run the unmodified script per
  dir. Probably the cleaner v1.)
- **Incremental vs full** — for a *new* arXiv batch, do we rebuild the whole substrate or extend
  it (only-new-documents map + merge into existing)? Incremental is the real production win but
  needs the merge to be additive. v1 = full per batch; incremental = follow-on.

## 5b. Stage-script audit + merge semantics (claude-2, 2026-06-17) — RESOLVES §5

claude-2 checked the 3 MAP scripts; the pre-shard-zero-change lean only holds for one, and the
merge is **not uniform** — one stage can silently break the byte-identical / drift-hash equivalence.

**CLI state (sharding hook):**
| stage | shardable as-is? | fix |
|---|---|---|
| `warp_concordance.py` | **yes** — has `--eprints <dir> --out <path>` | point at shard-dir + shard-out, zero changes |
| `warp_def_snippets.py` | **no** — only `--cap`; hard-codes `dp.EPRINTS` glob | add `--eprints/--out` (~5 lines, mirror concordance) |
| `warp_concept_usage.py` | **no** — no argparse; hard-codes `dp.EPRINTS` glob | add `--eprints/--out` (~5 lines) |

Decision: **add `--eprints/--out` to the two MAP scripts** (mirrors concordance's existing
pattern) — *not* env-overriding `dp.EPRINTS` (1 line but in a shared module everything imports →
risky blast radius). Pure pre-shard-on-disk can't redirect def_snippets/concept_usage because
they read the module constant, not a path.

**Merge semantics per MAP stage (the load-bearing correctness property):**
| stage | output shape | merge | byte-identical? |
|---|---|---|---|
| concordance | term → count | **SUM** counts across shards, serialize sorted-by-key | ✓ |
| concept_usage | paper-id → concepts | **dict union** (shards = disjoint paper sets), sorted-by-key | ✓ |
| def_snippets | concept → snippet **LIST** | union the lists — **but order currently follows paper-iteration**, so a naive shard-merge **REORDERS** | ✗ unless **sort each concept's merged list by a stable key (paper-id, offset)** |

**`def_snippets` is the trap:** it's the one stage whose naive merge silently diverges from the
serial output → breaks the drift-hash equivalence. The scheduler's reduce MUST stable-sort the
merged snippet lists. Flag verified.

**phylogeny** (`mark3_thread_tapestry`): confirmed independent concurrent branch — reads
golden+encyclopedia+cite-resolution, **not** the eprints corpus → no sharding needed.

**Division of labor:** claude-2 (owns the stage scripts) adds `--eprints/--out` to the two MAP
scripts once its -4 rebuild lands; claude-1 builds the scheduler + the reduce/merge (incl. the
`def_snippets` stable-sort). claude-2's `warp_run` stays the serial oracle the fork validates against.

## 6b. GPU stages + Rob's superpod conventions (Joe relayed Rob, 2026-06-17)

The CPU substrate stages above (concordance/def_snippets/concept_usage) are the *map* win. But the
pipeline also has **GPU stages** — `S4c` embeddings (warp spine), and, in the wider superpod run,
the **IATC 70B producer** + the checker's LLM stages (cas_select verify, rung-3-3, SFC2b). Rob has
standing opinions on how to run these; the runner should **set defaults but give him override flags**.

**Rob's sharding convention (observed):** *device-pinned*, not (only) SLURM arrays —
```
shard 0: CUDA_VISIBLE_DEVICES=0 → <out>-shard-0
shard 1: CUDA_VISIBLE_DEVICES=1 → <out>-shard-1   ... (1 shard per GPU, output-dir-per-shard)
```
So: **default 1 shard/GPU, pinned by `CUDA_VISIBLE_DEVICES`, each writing `<out>-shard-N`** — mirror
this exactly (it matches what Rob monitors via `top`/`nvidia-smi`/output dirs).

**Rob's throughput insight (load-bearing, and correct):** an A100-80GB at **100% util but only
11/82 GB used** is *not* maxed — useful work ≈ *memory-used × utilization*, so ~13% memory means
~7× headroom. The fix is **more model replicas per GPU** (or larger batch / more KV-cache) to fill
memory and saturate throughput — potentially ~7× on the LLM/embedding stages. Two corollaries he
flags: (a) a **bigger model** in the same GPU isn't necessarily slower if it's resident (quality for
near-free when memory allows); (b) **multiple LLM copies** likewise.
- Caveat to state honestly: LLM inference is often *bandwidth*-bound, so the gain is "fill the memory
  headroom with more concurrency/replicas," not a guaranteed 7×; the runner should make
  copies-per-GPU a **measured, tunable knob**, defaulting to memory-sized auto, Rob-overridable.

## 6c. Defaults + override flags (Joe: "set defaults, give Rob flags")

The runner ships sensible defaults; **every superpod-shape decision is a flag** Rob can override:
| flag | default | overrides |
|---|---|---|
| `--shards N` | = #GPUs (GPU stages) / #cores (CPU stages) | shard count |
| `--devices 0,1,..` / `CUDA_VISIBLE_DEVICES` | all visible GPUs, 1 shard each | device→shard pinning |
| `--copies-per-gpu K` | memory-sized auto (fill HBM headroom) | model replicas per GPU |
| `--model <name>` | the current 70B-AWQ | swap to a bigger/different LLM |
| `--out-shard-pattern <p>-shard-{n}` | matches Rob's convention | output layout |
| `--partition`, `--resume` | batch / on | SLURM partition; resume completed shards |

CPU substrate stages key on `--shards`/cores; GPU stages key on `--devices`/`--copies-per-gpu`.

## 6. Recommendation
v1 = **pre-shard the corpus on disk + SLURM CPU array per map-stage + deterministic merge**, no
stage-script changes. Validate it reproduces `warp_run`'s serial output byte-for-byte on a small
corpus (the drift-gate hash is the oracle). That alone makes the substrate build a routine
per-batch superpod job for Rob. Incremental rebuild + intra-stage threading are follow-ons.
