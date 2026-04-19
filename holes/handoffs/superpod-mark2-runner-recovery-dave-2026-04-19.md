# Superpod Mark 2 Runner Recovery Handoff for Dave (2026-04-19)

## Situation

Rob/Joe's batch-001 Mark 2 job ran out of runtime. On restart, the same command
failed through `conda run`:

```bash
conda run python3 scripts/superpod-job.py \
  --arxiv-jsonl batch-001.jsonl \
  --site arxiv.math \
  --input-dir /home/rjmeyers/gh/scratch/mfuton/world/data-sources/arxiv/batch-001 \
  --output-dir /home/rjmeyers/gh/scratch/mfuton/world/data-sources/arxiv/batch-001/output \
  --paper-eprint-dir ./eprints \
  --embed-workers 8 \
  --llm-gpu-workers 8 \
  --llm-loader-workers 16 \
  --llm-stage5d-batch-size 4 \
  --graph-embed-epochs 200 \
  --graph-embed-eval-every 5
```

The actual Python traceback above the conda wrapper error is the first thing to
recover. The conda wrapper line alone only says the process exited non-zero.

## Desired outcome

Get the runner resumable for the interrupted batch without weakening the Mark 2
quality invariants:

- Do not silently fall back to abstracts when `--paper-eprint-dir ./eprints` is
  supplied.
- Preserve already-computed expensive stages when their artifacts are complete
  and coherent.
- Recompute only the stage whose artifact is missing, partial, or lacks enough
  provenance to be trusted.
- Return a batch whose manifest proves that paper stages used full eprints.

The acceptance target for the rerun is not just "job exits 0". The returned
`output/manifest.json` should show:

- `paper_eprint.preflight.candidate_matches > 0`.
- `stage_status.technique_ner.text_source_counts.eprint > 0`.
- `stage_status.paper_hypergraph.text_source_counts.eprint > 0`.
- `stage_status.paper_hypergraph.with_claim_blocks > 0`.
- Stage 9b completed at 200 epochs and wrote `hypergraph-embeddings.npy`.

## Repository context

Primary files:

- `scripts/superpod-job.py`
- `src/futon6/technique_ner.py`
- `src/futon6/paper_hypergraph.py`
- `src/futon6/graph_embed.py`
- `holes/missions/M-superpod-mark2.md`
- `storage/mark2/qc/QC-FIRST-PASS-2026-04-17.md`

Relevant current behavior in `scripts/superpod-job.py`:

- Stage 2 resumes from `output/embeddings.npy` when row count matches.
- Stage 5c resumes from `output/techniques.json`.
- Stage 5d resumes from `output/paper-hypergraphs.json`.
- Stage 5c/5d eprint safety is strict: if `--paper-eprint-dir` is set, a
  paper-stage output with zero eprint usage must fail instead of silently using
  abstracts.
- `--paper-eprint-dir` also sets `--paper-hg-eprint-dir`; Rob's command is the
  right shape for paper stages.
- There is no general `--skip-stages` flag in the current live parser. Use
  the specific skip flags only when deliberately isolating a later stage.

## Likely restart failure

The most likely failure mode after a walltime kill is a resume/provenance edge
case around Stage 5c or Stage 5d:

1. A stage artifact such as `techniques.json` or `paper-hypergraphs.json` exists.
2. The process was killed before final `manifest.json` was written, or before
   `stage_status` for that stage was recorded.
3. On restart, the code sees the artifact and tries to reuse it.
4. Because the previous manifest lacks `text_source_counts`, the eprint guard
   treats the resumed stage as having zero eprint usage and raises:

```text
Stage 5c/Stage 5d was requested with --paper-eprint-dir=..., but loaded zero
eprints; refusing to write abstract-only paper-stage output
```

That guard is correct as an invariant. The bug, if this is the traceback, is
that a completed artifact cannot prove eprint usage when the manifest is missing
or stale. The fix is not to weaken the guard; it is to make resume provenance
recoverable from the artifact, or to force a clean recompute of just that stage.

## First commands on the superpod

Run these in the batch directory before changing files:

```bash
cd /home/rjmeyers/gh/scratch/mfuton/world/data-sources/arxiv/batch-001

find output -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM %9s %p\n' | sort

python3 - <<'PY'
import json
from pathlib import Path
p = Path("output/manifest.json")
print("manifest exists:", p.exists())
if p.exists():
    m = json.load(open(p))
    print("paper_eprint:", json.dumps(m.get("paper_eprint"), indent=2)[:4000])
    print("stage_status keys:", sorted((m.get("stage_status") or {}).keys()))
    for k in ["technique_ner", "paper_hypergraph", "graph_embedding"]:
        print(k, json.dumps((m.get("stage_status") or {}).get(k), indent=2)[:2000])
PY

python3 - <<'PY'
import json
from pathlib import Path
for name in ["entities.json", "techniques.json", "paper-hypergraphs.json",
             "reverse-morphogenesis.json", "hypergraphs.json"]:
    p = Path("output") / name
    if not p.exists():
        print(name, "MISSING")
        continue
    try:
        data = json.load(open(p))
        print(name, "rows", len(data) if hasattr(data, "__len__") else type(data))
    except Exception as e:
        print(name, "BAD_JSON", repr(e))
PY

python3 - <<'PY'
from pathlib import Path
print("eprint files:", sum(1 for p in Path("eprints").rglob("*") if p.is_file()))
print("sample:", [str(p) for p in list(Path("eprints").rglob("*"))[:20]])
PY
```

Capture the full traceback from the failed restart. If it is not the Stage
5c/5d eprint guard described above, debug the actual failing stage first.

## Recovery paths

### A. If the traceback is the Stage 5c/5d eprint resume guard

Treat the existing artifact as untrusted unless you can prove from the artifact
itself that it used eprints. The simple recovery is to move that stage's output
aside and rerun with the same command:

```bash
stash="output/interrupted-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$stash"

# If Stage 5c is the failing resume:
mv output/techniques.json "$stash"/ 2>/dev/null || true

# If Stage 5d is the failing resume:
mv output/paper-hypergraphs.json "$stash"/ 2>/dev/null || true

# Then rerun the same command.
```

Do not remove `embeddings.npy` unless Stage 2 row counts are wrong. Do not
remove Stage 3/6 chunk directories unless JSON validation says they are corrupt.

The code-level hardening Dave should consider: when `techniques.json` or
`paper-hypergraphs.json` exists but manifest provenance is missing, either:

- validate and reconstruct `text_source_counts` from per-record metadata if the
  artifact contains enough metadata, then write a repaired manifest entry; or
- raise an error that names the artifact and instructs the operator to move it
  aside for recompute.

Do not change `_require_paper_eprint_usage` to allow zero eprints.

### B. If eprint preflight reports zero matches

This is not a resume bug. It means the runner cannot map entity IDs to files in
`./eprints`. Fix the eprint naming/path issue before rerunning. The prior QC
showed that abstract-only batches are mechanically coherent but invalid for
Mark 2 structural validation.

Things to inspect:

- Entity IDs in `output/entities.json`, e.g. `arxiv-2401.01234`.
- Actual names under `./eprints`.
- `_paper_eprint_file_preflight` and `_load_eprint_text_for_entity` in
  `scripts/superpod-job.py`.

### C. If the failure is Stage 9b runtime/timeout

Stage 9b currently trains R-GCN for 200 epochs and is single-process GPU
training. The immediate target is correctness and non-collapse, not DDP. Resume
earlier stages, then rerun with the same 200-epoch flags if walltime allows.
If walltime is the only blocker, use a longer queue allocation or split the
pipeline operationally so Stage 9b runs after Stage 1-9a outputs are present.

Do not claim `--llm-gpu-workers 8` satisfies DDP/DeepSpeed. It is only
independent per-GPU model replication for local LLM inference.

## Suggested code hardening

1. Add artifact-level resume validation for Stage 5c and Stage 5d:
   - JSON parses.
   - Row count equals entity count.
   - IDs align with `entities.json`.
   - If `--paper-eprint-dir` is set, provenance proves eprint usage or the
     runner emits a precise recompute instruction.

2. Make final manifest writes more crash-tolerant:
   - Write per-stage sidecar status as soon as a stage artifact is committed,
     e.g. `output/stage-status/paper_hypergraph.json`.
   - At final manifest time, merge sidecars into `manifest.json`.
   - This avoids losing resume provenance when walltime kills the process after
     a stage file is written.

3. Add a preflight/QC command for Mark 2 batches:
   - Check eprint match rate before loading models.
   - Check full-text usage after Stage 5c/5d.
   - Fail loudly if paper stages used abstracts while `--paper-eprint-dir` was
     supplied.

4. Add a small local regression test:
   - Simulate an interrupted run by writing `paper-hypergraphs.json` without a
     final manifest.
   - Restart with `--paper-eprint-dir`.
   - Expected behavior: either resume with proven eprint counts or fail with a
     targeted "move this artifact aside/recompute" message, not a misleading
     abstract-only diagnosis.

## Do not do

- Do not remove the eprint guard.
- Do not set `--skip-paper-hypergraph` to get a green run; that bypasses the
  core Mark 2 structural output.
- Do not accept abstract-only `text_source_counts` for a production Mark 2
  batch.
- Do not delete the whole `output/` directory unless artifact-level validation
  shows the previous outputs are corrupt or mismatched.
