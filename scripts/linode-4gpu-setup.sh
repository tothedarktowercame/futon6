#!/usr/bin/env bash
# mark4 — 4-GPU Linode setup + vLLM serve.
# Run ON the freshly-provisioned Ubuntu 24.04 box (4x RTX 4000 Ada, 80GB aggregate).
# Proven last run: 70B-AWQ comes up in ~70s, TP=4 (~18.8GB/card), validated faithful.
#
# GOAL PATH:  CUDA toolkit (nvcc) on the image  -> flashinfer + torch.compile, full perf.
# FALLBACK:   driver-only                       -> --enforce-eager + flashinfer sampler off.
set -euo pipefail

MODEL="${MODEL:-hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4}"   # ungated AWQ-INT4
PORT="${PORT:-8000}"
TP="${TP:-4}"
VENV="${VENV:-$HOME/mark4-venv}"
LOG="${LOG:-$HOME/vllm-serve.log}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"       # preregistered go-live path; set 0 to allow CUDA graphs.
INSTALL_LINODE_CLI="${INSTALL_LINODE_CLI:-1}"

if [ "$(id -u)" -eq 0 ]; then
  SUDO=""
else
  SUDO="sudo"
fi

echo "== Ubuntu base packages =="
$SUDO apt-get update
$SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y \
  curl git python3 python3-venv python3-pip pipx

if [ "$INSTALL_LINODE_CLI" = "1" ]; then
  echo "== Linode CLI via pipx =="
  # Ubuntu 24.04's system Python is externally managed; Akamai/Linode TechDocs
  # recommend pipx for linode-cli rather than sudo pip.
  pipx ensurepath >/dev/null || true
  if command -v linode-cli >/dev/null 2>&1; then
    pipx upgrade linode-cli || true
  else
    pipx install linode-cli
  fi
  if [ -n "${LINODE_CLI_TOKEN:-}" ]; then
    echo "LINODE_CLI_TOKEN is set; linode-cli can run non-interactively with env auth."
  else
    echo "No LINODE_CLI_TOKEN set. Configure later with 'linode-cli configure' or export LINODE_CLI_TOKEN."
  fi
fi

echo "== GPU / driver =="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || { echo "FATAL: no nvidia-smi"; exit 1; }
NGPU=$(nvidia-smi -L | wc -l)
echo "GPUs visible: $NGPU (need TP=$TP)"
[ "$NGPU" -ge "$TP" ] || { echo "FATAL: fewer than $TP GPUs"; exit 1; }

echo "== CUDA toolkit (nvcc) detection =="
EAGER_FLAGS=""
if command -v nvcc >/dev/null 2>&1; then
  echo "nvcc present: $(nvcc --version | grep -i release) -> full perf (flashinfer + compile)"
else
  echo "WARNING: no nvcc (driver-only). Using fallback so the JIT-needing bits don't crash:"
  echo "  - --enforce-eager  (skips torch.compile + CUDA-graph capture; both need nvcc)"
  echo "  - VLLM_USE_FLASHINFER_SAMPLER=0  (native PyTorch sampling; flashinfer JITs kernels)"
  echo "  (verify the exact sampler env on this vLLM build if it still probes flashinfer)"
  EAGER_FLAGS="--enforce-eager"
  export VLLM_USE_FLASHINFER_SAMPLER=0
fi
if [ "$ENFORCE_EAGER" = "1" ] && [[ "$EAGER_FLAGS" != *"--enforce-eager"* ]]; then
  echo "ENFORCE_EAGER=1 -> serving with --enforce-eager as preregistered."
  EAGER_FLAGS="$EAGER_FLAGS --enforce-eager"
fi

echo "== venv + vLLM 0.23.0 (pulls torch cu130) =="
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q "vllm==0.23.0"

echo "== serve $MODEL on :$PORT (TP=$TP), detached -> $LOG =="
nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --tensor-parallel-size "$TP" \
  --served-model-name mark4-70b \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  $EAGER_FLAGS \
  > "$LOG" 2>&1 &
SERVE_PID=$!
echo "vLLM serving (pid $SERVE_PID). Tail: tail -f $LOG"

echo "== waiting for readiness (TP=4 shard ~70s) =="
for i in $(seq 1 60); do
  if curl -sf "localhost:$PORT/v1/models" >/dev/null 2>&1; then
    echo "READY after $((i*5))s:"; curl -s "localhost:$PORT/v1/models"; echo
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    exit 0
  fi
  sleep 5
done
echo "NOT READY after 300s — check $LOG"; exit 1
