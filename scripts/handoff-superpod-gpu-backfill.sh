# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#scripts/handoff-superpod-gpu-backfill.sh>>[init]
#!/usr/bin/env bash
# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#bash-strict>>[init]
set -euo pipefail
# ~/~ end

# ~/~ begin <<data/first-proof/superpod-handoff-rob.lit.md#gpu-preamble>>[init]
# GPU backfill: full 11-stage pipeline including LWGM (stages 9b+10).
# This is a required handoff stage on Superpod.
# Usage:
#   bash scripts/handoff-superpod-gpu-backfill.sh [math|mathoverflow|both]

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
EMBED_MODEL="${EMBED_MODEL:-BAAI/bge-large-en-v1.5}"
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
    --llm-model "$LLM_MODEL"

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
    -- \
    --embed-device cuda \
    --embed-model "$EMBED_MODEL" \
    --llm-model "$LLM_MODEL" \
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
