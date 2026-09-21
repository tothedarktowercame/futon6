#!/usr/bin/env bash
# mark4 — multi-GPU Linode setup + vLLM serve.
# Run ON a freshly-provisioned Ubuntu 24.04 GPU box.
# Proven on 4x RTX 4000 Ada (80GB aggregate): 70B-AWQ up in ~70s at TP=4
# (~18.8GB/card), validated faithful. The GPU count is NOT assumed: TP defaults
# to what nvidia-smi reports, and any host with enough aggregate memory works.
#
# GOAL PATH:  CUDA toolkit (nvcc) on the image  -> flashinfer + torch.compile, full perf.
# FALLBACK:   driver-only                       -> --enforce-eager + flashinfer sampler off.
set -euo pipefail

MODEL="${MODEL:-hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4}"   # ungated AWQ-INT4
PORT="${PORT:-8000}"
# ONE GPU unless asked otherwise. This script may run on a SHARED host where the
# other cards belong to other people's jobs; taking them because they were merely
# visible is how you get evicted. TP=<n> to widen deliberately, TP=all to take
# everything this job has been allocated. There is deliberately no "auto": taking
# every GPU is a choice somebody makes, not a default the script arrives at.
TP="${TP:-1}"
ATTENTION_HEADS="${ATTENTION_HEADS:-64}"  # Llama-3.1-70B; TP must divide this
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
$SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y \
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
# nvidia-smi lists every PHYSICAL card on the host and ignores CUDA_VISIBLE_DEVICES,
# so on a Slurm node its count is the machine's, not the job's. Where a launcher has
# set CUDA_VISIBLE_DEVICES (mfuton-superpod-gpu-policy.sh does), that is the
# allocation and it is the only number we are entitled to.
NPHYS=$(nvidia-smi -L | wc -l)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  NGPU=$(tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | grep -c .)
  echo "GPUs allocated to this job: $NGPU (host has $NPHYS; CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
else
  NGPU="$NPHYS"
  echo "GPUs visible: $NGPU (no CUDA_VISIBLE_DEVICES set)"
fi
[ "$NGPU" -ge 1 ] || { echo "FATAL: no GPUs available"; exit 1; }

# vLLM shards attention heads across the tensor-parallel group, so TP must DIVIDE
# the head count -- a 6-GPU allocation cannot run TP=6 against 64 heads and fails
# deep inside model load with an opaque shape error.
if [ "$TP" = "all" ]; then
  TP="$NGPU"
  while [ "$TP" -gt 1 ] && [ $(( ATTENTION_HEADS % TP )) -ne 0 ]; do
    TP=$(( TP - 1 ))
  done
  echo "TP=all -> $TP of $NGPU allocated GPU(s)$([ "$TP" -ne "$NGPU" ] && echo ", $(( NGPU - TP )) idle ($ATTENTION_HEADS heads do not divide $NGPU)")"
fi

case "$TP" in
  *[!0-9]*|"")
    echo "FATAL: TP=\"$TP\" is not a GPU count."
    echo "       Use TP=<n> for a specific width, or TP=all for the whole allocation."
    echo "       (There is no TP=auto: taking every GPU is a choice, not a default.)"
    exit 1 ;;
  0) echo "FATAL: TP=0 is not a GPU count."; exit 1 ;;
esac

echo "serving with TP=$TP"
[ "$NGPU" -ge "$TP" ] || { echo "FATAL: TP=$TP requested but only $NGPU GPU(s) visible"; exit 1; }
[ $(( ATTENTION_HEADS % TP )) -eq 0 ] || {
  echo "FATAL: TP=$TP does not divide $ATTENTION_HEADS attention heads; vLLM will fail at load."
  echo "       Set ATTENTION_HEADS for your model, or choose a TP that divides it."; exit 1; }

# WEIGHTS only. 70B at INT4 is ~35GB; KV cache is then sized to fill whatever
# GPU_MEMORY_UTILIZATION leaves, so observed usage says nothing about the floor.
# The 18.8GB/card measured on 4x 20GB RTX 4000 Ada was 94% of each card -- vLLM
# taking what it was given, not what the model required. Deriving a 75GB
# requirement from it would refuse a single 80GB card, which loads this model
# without trouble.
CARD_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
WEIGHTS_MB="${WEIGHTS_MB:-35000}"
HAVE_MB=$(( CARD_MB * TP ))
if [ "$HAVE_MB" -lt "$WEIGHTS_MB" ]; then
  # Advisory only: vLLM decides. This arithmetic has been wrong before and the
  # cost of a bad refusal (blocking a run that works) exceeds the cost of a bad
  # warning (a line of text before an error vLLM would have given anyway).
  echo "NOTE: TP=$TP gives ${HAVE_MB}MB across $TP card(s) of ${CARD_MB}MB."
  echo "      $MODEL is roughly ${WEIGHTS_MB}MB of weights before any KV cache,"
  echo "      so this may OOM during load. You have $NGPU GPU(s) allocated;"
  echo "      TP=$NGPU or TP=all would give $(( CARD_MB * NGPU ))MB."
  echo "      Proceeding anyway -- set WEIGHTS_MB if this estimate is wrong for your model."
fi

echo "== CUDA toolkit (nvcc) detection =="
EAGER_FLAGS=""
# LIVE FINDING (2026-06-18, first real run on the StackScript box): the flashinfer
# *sampler* JIT-compiles its CUDA kernels at engine startup, and that compile FAILS on
# this toolchain — flashinfer 0.6.12's bundled cub header errors with
#   class "cub::...BlockAdjacentDifference..." has no member "FlagHeads"
# under the StackScript's CUDA 12.0 nvcc, so `ninja` fails and engine-core init dies.
# nvcc being PRESENT does NOT mean the flashinfer sampler builds. So disable it
# UNCONDITIONALLY and use native PyTorch sampling (proven faithful on the same prereg
# run: temp-0 generations come out correct, full Stage-A pipeline runs through it).
# Re-enable only after confirming a flashinfer/cub combo that compiles on the box image.
export VLLM_USE_FLASHINFER_SAMPLER=0
if command -v nvcc >/dev/null 2>&1; then
  echo "nvcc present: $(nvcc --version | grep -i release)"
  echo "  -> torch.compile / CUDA-graphs available (only used if ENFORCE_EAGER=0)."
  echo "  -> flashinfer sampler kept OFF regardless (see FlagHeads note above)."
else
  echo "WARNING: no nvcc (driver-only) -> --enforce-eager (skips torch.compile + CUDA-graph capture, both need nvcc)."
  EAGER_FLAGS="--enforce-eager"
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

echo "== waiting for readiness (TP=$TP shard, ~70s at TP=4) =="
for i in $(seq 1 60); do
  if curl -sf "localhost:$PORT/v1/models" >/dev/null 2>&1; then
    echo "READY after $((i*5))s:"; curl -s "localhost:$PORT/v1/models"; echo
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    exit 0
  fi
  sleep 5
done
echo "NOT READY after 300s — check $LOG"; exit 1
