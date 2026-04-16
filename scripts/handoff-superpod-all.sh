# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#scripts/handoff-superpod-all.sh>>[init]
#!/usr/bin/env bash
# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#bash-strict>>[init]
set -euo pipefail
# ~/~ end

echo() {
  local ts
  ts="$(date '+%H:%M:%S')"
  if [[ "${1-}" == "-n" ]]; then
    shift
    builtin echo -n "[$ts] $*"
  else
    builtin echo "[$ts] $*"
  fi
}

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#all-header-comment>>[init]
# Single-command Superpod handoff runner.
# Runs 11-stage pipeline (stages 1-7 + LWGM stages 8-10).
#
# Default behavior (NUM_SHARDS=1 or unset):
#   1) bootstrap inputs
#   2) sanity tests
#   3) smoke run + verification
#   4) full CPU runs + verification (stages 1/5/7/8/9a)
#   5) required GPU backfill + verification (stages 1-10 incl. LWGM)
#   6) package outputs
#
# Sharded / Block 1 (NUM_SHARDS>1):
#   1) bootstrap inputs
#   2) sanity tests
#   3) smoke run + verification
#   4) sharded GPU pipeline for both corpora
#   5) package outputs
#
# Block 2 (BLOCK=2, requires Block 1 output):
#   1) verify Block 1 output exists + GPU/LLM prereqs
#   2) sharded LLM on thread sample (both corpora)
#   3) compose LLM files into Block 1 output
#
# Invocations:
#   Block 1:  NUM_SHARDS=8 EXTRA_SHARD_ARGS="--skip-llm" bash scripts/handoff-superpod-all.sh
#   Block 2:  BLOCK=2 NUM_SHARDS=8 bash scripts/handoff-superpod-all.sh
#
# Options:
#   --smoke-only       stop after smoke run + verification
#   --skip-bootstrap   do not run bootstrap script
#   --skip-tests       do not run pytest sanity checks
#
# Environment:
#   NUM_SHARDS          number of parallel shards (default: 1 = unsharded)
#   EXTRA_SHARD_ARGS    extra args passed to each shard job (e.g. "--skip-llm")
#   BLOCK               1 (default) or 2; Block 2 adds LLM to Block 1 output
#   LLM_THREAD_LIMIT    threads per shard for Block 2 LLM (default: 5000)
#   LLM_BATCH_SIZE         baseline LLM batch size (Stage 7, default: 24)
#   LLM_STAGE3_BATCH_SIZE  Stage 3 LLM batch size (default: 80)
#   LLM_STAGE3_CHUNKS_PER_SHARD  Stage 3 resumable chunks per shard (default: 10)
#   LLM_STAGE6_BATCH_SIZE  Stage 6 LLM batch size (default: 48)
#   LLM_STAGE6_CHUNKS_PER_SHARD  Stage 6 resumable chunks per shard (default: 10)
#   LLM_GPU_WORKERS       Process-level Stage 5c LLM GPU workers
#                          (default: 0 = auto all visible GPUs)
#   LLM_LOADER_WORKERS     Python workers feeding Dataset-backed LLM pipelines.
#                          Unsharded superpod-job defaults to min(16,
#                          Slurm/cpuset CPU affinity). Sharded superpod-shard
#                          splits that CPU budget across shard processes unless
#                          explicitly set.
#   GRAPH_EMBED_DIM         Stage 9b embedding dimension (default: 128)
#   GRAPH_EMBED_EPOCHS      Stage 9b epochs (default: 50)
#   GRAPH_EMBED_BATCH_SIZE  Stage 9b batch size (default: 1024)
#   GRAPH_EMBED_WORKERS     Stage 9b CPU workers for batch prep
#                           (default: SLURM_CPUS_PER_TASK if set, else 16)
#   RESUME_MATH_STAGE9      auto|0|1 (default: auto)
#                           auto: if math-processed-gpu has hypergraphs but no
#                           Stage 9b/10 artifacts, resume from post-merge 9b+10.
# ~/~ end

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#all-root-and-args>>[init]
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SMOKE_ONLY=0
SKIP_BOOTSTRAP=0
SKIP_TESTS=0
NUM_SHARDS="${NUM_SHARDS:-1}"
EXTRA_SHARD_ARGS="${EXTRA_SHARD_ARGS:-}"
BLOCK="${BLOCK:-1}"
LLM_THREAD_LIMIT="${LLM_THREAD_LIMIT:-5000}"
LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-24}"
LLM_STAGE3_BATCH_SIZE="${LLM_STAGE3_BATCH_SIZE:-80}"
LLM_STAGE3_CHUNKS_PER_SHARD="${LLM_STAGE3_CHUNKS_PER_SHARD:-10}"
LLM_STAGE6_BATCH_SIZE="${LLM_STAGE6_BATCH_SIZE:-48}"
LLM_STAGE6_CHUNKS_PER_SHARD="${LLM_STAGE6_CHUNKS_PER_SHARD:-10}"
LLM_GPU_WORKERS="${LLM_GPU_WORKERS:-0}"
GRAPH_EMBED_DIM="${GRAPH_EMBED_DIM:-128}"
GRAPH_EMBED_EPOCHS="${GRAPH_EMBED_EPOCHS:-50}"
GRAPH_EMBED_BATCH_SIZE="${GRAPH_EMBED_BATCH_SIZE:-1024}"
if [[ -z "${GRAPH_EMBED_WORKERS:-}" ]]; then
  if [[ "${SLURM_CPUS_PER_TASK:-}" =~ ^[0-9]+$ ]] && (( SLURM_CPUS_PER_TASK > 0 )); then
    GRAPH_EMBED_WORKERS="$SLURM_CPUS_PER_TASK"
  else
    GRAPH_EMBED_WORKERS=16
  fi
fi
RESUME_MATH_STAGE9="${RESUME_MATH_STAGE9:-auto}"

for arg in "$@"; do
  case "$arg" in
    --smoke-only) SMOKE_ONLY=1 ;;
    --skip-bootstrap) SKIP_BOOTSTRAP=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    *)
      echo "Unknown option: $arg"
      echo "Usage: $0 [--smoke-only] [--skip-bootstrap] [--skip-tests]"
      exit 1
      ;;
  esac
done
# ~/~ end

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#all-utility-functions>>[init]
STEP=0
step() {
  STEP=$((STEP + 1))
  echo
  echo "==> [all][step ${STEP}] $*"
}

fail() {
  echo "[all] ERROR: $*" >&2
  exit 1
}

assert_file() {
  local path="$1"
  [[ -f "$path" ]] || fail "missing file: $path"
}

resume_post_merge_stage9() {
  local outdir="$1"
  local site_label="$2"

  assert_file "$outdir/hypergraphs.json"
  echo "[all] resuming post-merge stages for $site_label in $outdir"
  python3 scripts/superpod-shard.py post-merge \
    --output-dir "$outdir" \
    --graph-embed-dim "$GRAPH_EMBED_DIM" \
    --graph-embed-epochs "$GRAPH_EMBED_EPOCHS" \
    --graph-embed-batch-size "$GRAPH_EMBED_BATCH_SIZE" \
    --graph-embed-workers "$GRAPH_EMBED_WORKERS"

  python3 scripts/ct-verifier.py verify \
    --wiring "$outdir/thread-wiring-ct.json" \
    --reference data/nlab-ct-reference.json \
    --output "$outdir/thread-wiring-ct-verification.json"
}
# ~/~ end

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#all-verify-run-dir>>[init]
verify_run_dir() {
  local dir="$1"
  assert_file "$dir/manifest.json"
  assert_file "$dir/thread-wiring-ct.json"
  assert_file "$dir/thread-wiring-ct-verification.json"

  python3 - "$dir" <<'PY'
import json
import pathlib
import sys

d = pathlib.Path(sys.argv[1])
manifest = json.loads((d / "manifest.json").read_text())
s7 = manifest.get("stage7_stats") or {}

# In sharded merge outputs, stage7_stats is kept per-shard under shard_manifests.
if not s7 and manifest.get("merged") and isinstance(manifest.get("shard_manifests"), list):
    shard_stats = []
    for shard_m in manifest.get("shard_manifests", []):
        if isinstance(shard_m, dict):
            shard_s7 = shard_m.get("stage7_stats") or {}
            if isinstance(shard_s7, dict):
                shard_stats.append(shard_s7)
    if shard_stats:
        s7 = {
            "ct_backed": all(bool(ss.get("ct_backed", False)) for ss in shard_stats),
            "threads_processed": sum(int(ss.get("threads_processed", 0)) for ss in shard_stats),
        }

if not s7.get("ct_backed", False):
    raise SystemExit(f"{d}: stage7_stats.ct_backed is false")
if int(s7.get("threads_processed", 0)) <= 0:
    raise SystemExit(f"{d}: stage7_stats.threads_processed <= 0")

ver = json.loads((d / "thread-wiring-ct-verification.json").read_text())
if isinstance(ver, dict):
    edges = int((ver.get("summary") or {}).get("edges_checked", 0))
elif isinstance(ver, list):
    edges = 0
    for item in ver:
        if isinstance(item, dict):
            edges += int((item.get("summary") or {}).get("edges_checked", 0))
else:
    edges = 0

if edges <= 0:
    raise SystemExit(f"{d}: verifier edges_checked <= 0")

print(f"{d}: ok (threads={s7.get('threads_processed')}, edges_checked={edges})")
PY
}
# ~/~ end

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#all-run-smoke>>[init]
run_smoke() {
  local out_smoke="$ROOT_DIR/tmp/superpod-rob-smoke-$(date +%s)"
  echo "[all] smoke output dir: $out_smoke"

  python3 scripts/superpod-job.py \
    tests/fixtures/se-mini/Posts.xml \
    --comments-xml tests/fixtures/se-mini/Comments.xml \
    --site math.stackexchange \
    --output-dir "$out_smoke" \
    --min-score 0 \
    --thread-limit 4 \
    --skip-embeddings \
    --skip-llm \
    --skip-clustering

  python3 scripts/ct-verifier.py verify \
    --wiring "$out_smoke/thread-wiring-ct.json" \
    --reference data/nlab-ct-reference.json \
    --output "$out_smoke/thread-wiring-ct-verification.json"

  verify_run_dir "$out_smoke"
}
# ~/~ end

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#all-package-outputs>>[init]
package_outputs() {
  if [[ "$NUM_SHARDS" -le 1 ]]; then
    tar czf superpod-math-processed.tar.gz \
      math-processed/entities.json \
      math-processed/relations.json \
      math-processed/tags.json \
      math-processed/stats.json \
      math-processed/ner-terms.json \
      math-processed/scopes.json \
      math-processed/thread-wiring-ct.json \
      math-processed/thread-wiring-ct-verification.json \
      math-processed/expression-surfaces.json \
      math-processed/hypergraphs.json \
      math-processed/manifest.json

    tar czf superpod-mo-processed.tar.gz \
      mo-processed/entities.json \
      mo-processed/relations.json \
      mo-processed/tags.json \
      mo-processed/stats.json \
      mo-processed/ner-terms.json \
      mo-processed/scopes.json \
      mo-processed/thread-wiring-ct.json \
      mo-processed/thread-wiring-ct-verification.json \
      mo-processed/expression-surfaces.json \
      mo-processed/hypergraphs.json \
      mo-processed/manifest.json
  fi

  # GPU tarballs include the whole directory: per-thread output AND
  # reusable model artifacts (graph-gnn-model.pt, structural-similarity-index.faiss,
  # hypergraph-embeddings.npy, etc.) for downstream hot re-embedding on CPU.
  tar czf superpod-math-processed-gpu.tar.gz math-processed-gpu
  tar czf superpod-mo-processed-gpu.tar.gz mo-processed-gpu
}
# ~/~ end

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#all-orchestration>>[init]
echo "[all] repo: $ROOT_DIR"
echo "[all] block=$BLOCK smoke_only=$SMOKE_ONLY skip_bootstrap=$SKIP_BOOTSTRAP skip_tests=$SKIP_TESTS num_shards=$NUM_SHARDS"
echo "[all] llm: stage3_batch=$LLM_STAGE3_BATCH_SIZE chunks=$LLM_STAGE3_CHUNKS_PER_SHARD stage6_batch=$LLM_STAGE6_BATCH_SIZE stage6_chunks=$LLM_STAGE6_CHUNKS_PER_SHARD base_batch=$LLM_BATCH_SIZE gpu_workers=$LLM_GPU_WORKERS loader_workers=${LLM_LOADER_WORKERS:-auto}"
echo "[all] graph-embed: dim=$GRAPH_EMBED_DIM epochs=$GRAPH_EMBED_EPOCHS batch=$GRAPH_EMBED_BATCH_SIZE workers=$GRAPH_EMBED_WORKERS"
echo "[all] resume_math_stage9=$RESUME_MATH_STAGE9"

# ---- Block 2: LLM enrichment (compose onto Block 1 output) ----
if [[ "$BLOCK" == "2" ]]; then
  for d in ./math-processed-gpu ./mo-processed-gpu; do
    [[ -d "$d" ]] || fail "Block 1 output not found: $d (run Block 1 first)"
  done

  command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found"
  python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null \
    || fail "PyTorch cannot see CUDA"

  LLM_MODEL="${LLM_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"

  step "Block 2: LLM on ${LLM_THREAD_LIMIT}/shard sample (math.stackexchange)"
  python3 scripts/superpod-shard.py run \
    --posts-xml ./se-data/math.stackexchange.com/Posts.xml \
    --comments-xml ./se-data/math.stackexchange.com/Comments.xml \
    --site math.stackexchange \
    --num-shards "$NUM_SHARDS" \
    --output-dir ./math-processed-gpu-llm \
    --skip-post-merge \
    -- --thread-limit "$LLM_THREAD_LIMIT" --skip-embeddings \
    --llm-model "$LLM_MODEL" --embed-device cuda \
    --llm-batch-size "$LLM_BATCH_SIZE" \
    --llm-stage3-batch-size "$LLM_STAGE3_BATCH_SIZE" \
    --llm-stage3-chunks-per-shard "$LLM_STAGE3_CHUNKS_PER_SHARD" \
    --llm-stage6-batch-size "$LLM_STAGE6_BATCH_SIZE" \
    --llm-stage6-chunks-per-shard "$LLM_STAGE6_CHUNKS_PER_SHARD" \
    --llm-gpu-workers "$LLM_GPU_WORKERS"

  step "Block 2: LLM on ${LLM_THREAD_LIMIT}/shard sample (mathoverflow)"
  python3 scripts/superpod-shard.py run \
    --posts-xml ./se-data/mathoverflow.net/Posts.xml \
    --comments-xml ./se-data/mathoverflow.net/Comments.xml \
    --site mathoverflow.net \
    --num-shards "$NUM_SHARDS" \
    --output-dir ./mo-processed-gpu-llm \
    --skip-post-merge \
    -- --thread-limit "$LLM_THREAD_LIMIT" --skip-embeddings \
    --llm-model "$LLM_MODEL" --embed-device cuda \
    --llm-batch-size "$LLM_BATCH_SIZE" \
    --llm-stage3-batch-size "$LLM_STAGE3_BATCH_SIZE" \
    --llm-stage3-chunks-per-shard "$LLM_STAGE3_CHUNKS_PER_SHARD" \
    --llm-stage6-batch-size "$LLM_STAGE6_BATCH_SIZE" \
    --llm-stage6-chunks-per-shard "$LLM_STAGE6_CHUNKS_PER_SHARD" \
    --llm-gpu-workers "$LLM_GPU_WORKERS"

  step "compose LLM files into Block 1 output"
  for f in pattern-tags.json reverse-morphogenesis.json; do
    if [[ -f ./math-processed-gpu-llm/"$f" ]]; then
      cp ./math-processed-gpu-llm/"$f" ./math-processed-gpu/
      echo "  math-processed-gpu/$f"
    fi
    if [[ -f ./mo-processed-gpu-llm/"$f" ]]; then
      cp ./mo-processed-gpu-llm/"$f" ./mo-processed-gpu/
      echo "  mo-processed-gpu/$f"
    fi
  done

  echo
  echo "[all] Block 2 complete. LLM files composed into Block 1 output."
  exit 0
fi

# ---- Block 1 / default ----
if (( ! SKIP_BOOTSTRAP )); then
  step "bootstrap inputs"
  bash scripts/handoff-superpod-bootstrap.sh
else
  step "bootstrap skipped by flag"
fi

if (( ! SKIP_TESTS )); then
  step "sanity tests"
  PYTHONPATH=src pytest -q tests/test_superpod_job_smoke.py tests/test_ct_verifier.py
else
  step "sanity tests skipped by flag"
fi

step "smoke run + verification"
run_smoke

if (( SMOKE_ONLY )); then
  echo
  echo "[all] done (smoke-only)."
  exit 0
fi

if [[ "$NUM_SHARDS" -gt 1 ]]; then
  # Block 1: sharded pipeline supersedes separate CPU baseline + GPU backfill.
  # NUM_SHARDS and EXTRA_SHARD_ARGS are exported so gpu-backfill.sh picks them up.
  export NUM_SHARDS EXTRA_SHARD_ARGS
  export LLM_BATCH_SIZE LLM_STAGE3_BATCH_SIZE LLM_STAGE3_CHUNKS_PER_SHARD
  export LLM_STAGE6_BATCH_SIZE LLM_STAGE6_CHUNKS_PER_SHARD LLM_GPU_WORKERS
  export GRAPH_EMBED_DIM GRAPH_EMBED_EPOCHS GRAPH_EMBED_BATCH_SIZE GRAPH_EMBED_WORKERS

  do_resume_math_stage9=0
  math_stage9_complete=0
  if [[ "$RESUME_MATH_STAGE9" == "1" ]]; then
    do_resume_math_stage9=1
  elif [[ "$RESUME_MATH_STAGE9" == "auto" ]]; then
    if [[ -f ./math-processed-gpu/hypergraphs.json ]] && \
       [[ ! -f ./math-processed-gpu/hypergraph-embeddings.npy || ! -f ./math-processed-gpu/structural-similarity-index.faiss ]]; then
      do_resume_math_stage9=1
    fi
    if [[ -f ./math-processed-gpu/hypergraph-embeddings.npy && -f ./math-processed-gpu/structural-similarity-index.faiss ]]; then
      math_stage9_complete=1
    fi
  fi

  if (( do_resume_math_stage9 )); then
    step "resume math.stackexchange from post-merge stages (9b + 10)"
    resume_post_merge_stage9 "./math-processed-gpu" "math.stackexchange"
    verify_run_dir "./math-processed-gpu"

    step "sharded pipeline ($NUM_SHARDS shards) + verification (mathoverflow)"
    bash scripts/handoff-superpod-gpu-backfill.sh mathoverflow
    verify_run_dir "./mo-processed-gpu"
  elif (( math_stage9_complete )); then
    step "math.stackexchange already complete; sharded pipeline ($NUM_SHARDS shards) + verification (mathoverflow)"
    verify_run_dir "./math-processed-gpu"
    bash scripts/handoff-superpod-gpu-backfill.sh mathoverflow
    verify_run_dir "./mo-processed-gpu"
  else
    step "sharded pipeline ($NUM_SHARDS shards) + verification"
    bash scripts/handoff-superpod-gpu-backfill.sh both
    verify_run_dir "./math-processed-gpu"
    verify_run_dir "./mo-processed-gpu"
  fi

  step "package outputs"
  package_outputs

  assert_file "superpod-math-processed-gpu.tar.gz"
  assert_file "superpod-mo-processed-gpu.tar.gz"

  echo
  echo "[all] complete (sharded / Block 1). Deliver:"
  echo "  superpod-math-processed-gpu.tar.gz"
  echo "  superpod-mo-processed-gpu.tar.gz"
else
  step "CPU baseline run + verification (math.stackexchange)"
  python3 scripts/superpod-job.py \
    ./se-data/math.stackexchange.com/Posts.xml \
    --comments-xml ./se-data/math.stackexchange.com/Comments.xml \
    --site math.stackexchange \
    --output-dir ./math-processed \
    --skip-embeddings \
    --skip-llm \
    --skip-clustering
  python3 scripts/ct-verifier.py verify \
    --wiring ./math-processed/thread-wiring-ct.json \
    --reference data/nlab-ct-reference.json \
    --output ./math-processed/thread-wiring-ct-verification.json
  verify_run_dir "./math-processed"

  step "CPU baseline run + verification (mathoverflow)"
  python3 scripts/superpod-job.py \
    ./se-data/mathoverflow.net/Posts.xml \
    --comments-xml ./se-data/mathoverflow.net/Comments.xml \
    --site mathoverflow.net \
    --output-dir ./mo-processed \
    --skip-embeddings \
    --skip-llm \
    --skip-clustering
  python3 scripts/ct-verifier.py verify \
    --wiring ./mo-processed/thread-wiring-ct.json \
    --reference data/nlab-ct-reference.json \
    --output ./mo-processed/thread-wiring-ct-verification.json
  verify_run_dir "./mo-processed"

  step "required GPU backfill + verification"
  bash scripts/handoff-superpod-gpu-backfill.sh both
  verify_run_dir "./math-processed-gpu"
  verify_run_dir "./mo-processed-gpu"

  step "package outputs"
  package_outputs

  assert_file "superpod-math-processed.tar.gz"
  assert_file "superpod-mo-processed.tar.gz"
  assert_file "superpod-math-processed-gpu.tar.gz"
  assert_file "superpod-mo-processed-gpu.tar.gz"

  echo
  echo "[all] complete. Deliver:"
  echo "  superpod-math-processed.tar.gz"
  echo "  superpod-mo-processed.tar.gz"
  echo "  superpod-math-processed-gpu.tar.gz"
  echo "  superpod-mo-processed-gpu.tar.gz"
fi
# ~/~ end
# ~/~ end
