# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#scripts/handoff-superpod-gpu-backfill.sh>>[init]
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

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#gpu-preamble>>[init]
# GPU backfill: full 11-stage pipeline including LWGM (stages 9b+10).
# This is a required handoff stage on Superpod.
# Usage:
#   bash scripts/handoff-superpod-gpu-backfill.sh [math|mathoverflow|both]
#
# Optional env knobs for Stage 3 + 9b:
#   LLM_BATCH_SIZE         baseline LLM batch size (Stage 7, default: 24)
#   LLM_STAGE3_BATCH_SIZE  Stage 3 LLM batch size (default: 80)
#   LLM_STAGE3_CHUNKS_PER_SHARD  Stage 3 resumable chunks per shard (default: 10)
#   LLM_STAGE6_BATCH_SIZE  Stage 6 LLM batch size (default: 48)
#   LLM_STAGE6_CHUNKS_PER_SHARD  Stage 6 resumable chunks per shard (default: 10)
#   LLM_GPU_WORKERS       Process-level Stage 5c LLM GPU workers
#                          (default: 0 = auto all visible GPUs)
#   LLM_LOADER_WORKERS     Python workers feeding Dataset-backed LLM pipelines.
#                          For unsharded runs, superpod-job defaults to
#                          min(16, Slurm/cpuset CPU affinity). For sharded
#                          runs, superpod-shard splits that CPU budget across
#                          concurrent shard processes unless explicitly set.
#   GRAPH_EMBED_DIM         embedding dimension (default: 128)
#   GRAPH_EMBED_EPOCHS      training epochs (default: 50)
#   GRAPH_EMBED_BATCH_SIZE  training batch size (default: 1024)
#   GRAPH_EMBED_WORKERS     CPU workers for batch prep
#                           (default: SLURM_CPUS_PER_TASK if set, else 16)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TARGET="${1:-both}"
case "$TARGET" in
  math|mathoverflow|both) ;;
  *)
    echo "Usage: $0 [math|mathoverflow|both]"
    exit 1
    ;;
esac
# ~/~ end

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#gpu-env>>[init]
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[gpu] FATAL: nvidia-smi not found. Install NVIDIA drivers first."
  echo "[gpu]   apt-get install -y ubuntu-drivers-common && ubuntu-drivers autoinstall && reboot"
  exit 1
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo

if ! python3 -c "import torch; assert torch.cuda.is_available(), 'no CUDA'" 2>/dev/null; then
  echo "[gpu] FATAL: PyTorch cannot see CUDA. Check driver/torch compatibility."
  echo "[gpu]   python3 -c \"import torch; print(torch.cuda.is_available())\""
  exit 1
fi

echo "[gpu] GPU OK: $(python3 -c "import torch; print(torch.cuda.get_device_name(0))")"

if [[ -z "${HF_TOKEN:-}" ]] && [[ "${LLM_MODEL:-}" == *"meta-llama"* ]]; then
  echo "[gpu] WARNING: HF_TOKEN not set but LLM_MODEL is Llama (gated)."
  echo "[gpu]   export HF_TOKEN=hf_your_token_here"
  echo "[gpu]   Or use the default (Mistral-7B, ungated)."
fi

LLM_MODEL="${LLM_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-24}"
LLM_STAGE3_BATCH_SIZE="${LLM_STAGE3_BATCH_SIZE:-80}"
LLM_STAGE3_CHUNKS_PER_SHARD="${LLM_STAGE3_CHUNKS_PER_SHARD:-10}"
LLM_STAGE6_BATCH_SIZE="${LLM_STAGE6_BATCH_SIZE:-48}"
LLM_STAGE6_CHUNKS_PER_SHARD="${LLM_STAGE6_CHUNKS_PER_SHARD:-10}"
LLM_GPU_WORKERS="${LLM_GPU_WORKERS:-0}"
EMBED_MODEL="${EMBED_MODEL:-BAAI/bge-large-en-v1.5}"
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

echo "[gpu] llm config: stage3_batch=${LLM_STAGE3_BATCH_SIZE} chunks=${LLM_STAGE3_CHUNKS_PER_SHARD} stage6_batch=${LLM_STAGE6_BATCH_SIZE} stage6_chunks=${LLM_STAGE6_CHUNKS_PER_SHARD} base_batch=${LLM_BATCH_SIZE} gpu_workers=${LLM_GPU_WORKERS} loader_workers=${LLM_LOADER_WORKERS:-auto}"
echo "[gpu] graph-embed config: dim=${GRAPH_EMBED_DIM} epochs=${GRAPH_EMBED_EPOCHS} batch=${GRAPH_EMBED_BATCH_SIZE} workers=${GRAPH_EMBED_WORKERS}"
# ~/~ end

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#gpu-run-site>>[init]
run_site() {
  local site="$1"
  local posts="$2"
  local comments="$3"
  local outdir="$4"

  echo "[gpu] running full 11-stage pipeline for $site ..."
  python3 scripts/superpod-job.py \
    "$posts" \
    --comments-xml "$comments" \
    --site "$site" \
    --output-dir "$outdir" \
    --embed-device cuda \
    --embed-model "$EMBED_MODEL" \
    --llm-model "$LLM_MODEL" \
    --llm-batch-size "$LLM_BATCH_SIZE" \
    --llm-stage3-batch-size "$LLM_STAGE3_BATCH_SIZE" \
    --llm-stage3-chunks-per-shard "$LLM_STAGE3_CHUNKS_PER_SHARD" \
    --llm-stage6-batch-size "$LLM_STAGE6_BATCH_SIZE" \
    --llm-stage6-chunks-per-shard "$LLM_STAGE6_CHUNKS_PER_SHARD" \
    --llm-gpu-workers "$LLM_GPU_WORKERS" \
    --graph-embed-dim "$GRAPH_EMBED_DIM" \
    --graph-embed-epochs "$GRAPH_EMBED_EPOCHS" \
    --graph-embed-batch-size "$GRAPH_EMBED_BATCH_SIZE" \
    --graph-embed-workers "$GRAPH_EMBED_WORKERS"

  python3 scripts/ct-verifier.py verify \
    --wiring "$outdir/thread-wiring-ct.json" \
    --reference data/nlab-ct-reference.json \
    --output "$outdir/thread-wiring-ct-verification.json"
}

run_site_sharded() {
  local site="$1"
  local posts="$2"
  local comments="$3"
  local outdir="$4"

  echo "[gpu] running sharded pipeline ($NUM_SHARDS shards) for $site ..."
  python3 scripts/superpod-shard.py run \
    --posts-xml "$posts" \
    --comments-xml "$comments" \
    --site "$site" \
    --num-shards "$NUM_SHARDS" \
    --output-dir "$outdir" \
    --graph-embed-dim "$GRAPH_EMBED_DIM" \
    --graph-embed-epochs "$GRAPH_EMBED_EPOCHS" \
    --graph-embed-batch-size "$GRAPH_EMBED_BATCH_SIZE" \
    --graph-embed-workers "$GRAPH_EMBED_WORKERS" \
    -- \
    --embed-device cuda \
    --embed-model "$EMBED_MODEL" \
    --llm-model "$LLM_MODEL" \
    --llm-batch-size "$LLM_BATCH_SIZE" \
    --llm-stage3-batch-size "$LLM_STAGE3_BATCH_SIZE" \
    --llm-stage3-chunks-per-shard "$LLM_STAGE3_CHUNKS_PER_SHARD" \
    --llm-stage6-batch-size "$LLM_STAGE6_BATCH_SIZE" \
    --llm-stage6-chunks-per-shard "$LLM_STAGE6_CHUNKS_PER_SHARD" \
    --llm-gpu-workers "$LLM_GPU_WORKERS" \
    $EXTRA_SHARD_ARGS

  python3 scripts/ct-verifier.py verify \
    --wiring "$outdir/thread-wiring-ct.json" \
    --reference data/nlab-ct-reference.json \
    --output "$outdir/thread-wiring-ct-verification.json"
}
# ~/~ end

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#gpu-dispatch>>[init]
NUM_SHARDS="${NUM_SHARDS:-1}"
EXTRA_SHARD_ARGS="${EXTRA_SHARD_ARGS:-}"

if [[ "$NUM_SHARDS" -gt 1 ]]; then
  run_fn=run_site_sharded
  echo "[gpu] sharded mode: $NUM_SHARDS shards"
else
  run_fn=run_site
fi

if [[ "$TARGET" == "math" || "$TARGET" == "both" ]]; then
  $run_fn \
    "math.stackexchange" \
    "./se-data/math.stackexchange.com/Posts.xml" \
    "./se-data/math.stackexchange.com/Comments.xml" \
    "./math-processed-gpu"
fi

if [[ "$TARGET" == "mathoverflow" || "$TARGET" == "both" ]]; then
  $run_fn \
    "mathoverflow.net" \
    "./se-data/mathoverflow.net/Posts.xml" \
    "./se-data/mathoverflow.net/Comments.xml" \
    "./mo-processed-gpu"
fi

echo "[gpu] done."
# ~/~ end
# ~/~ end
