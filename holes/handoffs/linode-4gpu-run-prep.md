# Linode 4-GPU run — prep / readiness (mined from claude-4 session, 2026-06-16)

Status: the 4-GPU box was **provisioned and mid-launch** when the laptop/JVM crashed
(~16:50). Linode now powered down. This consolidates claude-4's scattered transcript
notes (session `34299d44`, ~/.claude/projects/-home-joe/) into one checklist so the
run can be resumed cleanly. Supersedes the 1-GPU `linode-iatc-model-loop-2026-06-16.md`.

## Why a second (bigger) test
The 1-GPU run served **Llama-3.1-8B** and produced 10 gated IATC graphs — but the
hardened substance gate then **auto-failed 6/10** (8B too weak). So the second test
steps up to **70B**.

## Hardware (as actually provisioned)
- **4× RTX 4000 Ada — 80 GB aggregate**, driver loaded, **vLLM 0.23.0 sees 4 CUDA
  devices** (confirmed live before the crash).
- Driver **550+** (580 worked), Ubuntu 22.04/24.04. Disk **≥200 GB** (70B weights + HF
  cache), RAM **≥64 GB**.

## Model + serving
- **Llama-3.1-70B-AWQ (4-bit / INT4, ~40 GB)** — 80 GB aggregate fits 70B at **4-bit,
  not fp16** (140 GB). Candidate ungated checkpoint: `hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4`.
- Serve via **vLLM OpenAI-compatible server** with **`--tensor-parallel-size 4`**
  (shard across all 4 cards; NCCL intra-node, fast).
- Stack: vLLM 0.23.0 + torch 2.11/cu130.

## CUDA features to ENABLE (the key lesson — this is what Joe was asking about)
The 1-GPU box was **driver-only (no CUDA toolkit / `nvcc`)**, which broke two things:
1. **flashinfer** (vLLM's sampler) JIT-compiles CUDA kernels needing `nvcc` → "Could
   not find nvcc".
2. **torch.compile / `VLLM_COMPILE` / CUDA-graph capture** need `nvcc` / `CUDA_HOME`.

**→ For full perf on the 4-GPU run: install the CUDA toolkit (`nvcc`) on the image.**
Then flashinfer + torch.compile + cudagraph all work.

**Fallback if the image is driver-only again** (claude-4's flags already handle it):
- `--enforce-eager` (skips torch.compile + CUDA-graph capture)
- disable the **flashinfer sampler** (native PyTorch sampling)
- FLASH_ATTN attention is prebuilt — fine either way.

(BGE embedding lane, if run: `scripts/mark3_embed.py` already does CUDA multi-GPU
fanout — `EMBED_WORKERS=0` auto over visible devices, `start_multi_process_pool`,
parent model on CPU; `handoff-superpod-mark3-embed.sh` is the runner. Built for an 8×
A100 node but adapts to 4 visible GPUs.)

## What the run should now cover
Beyond the IATC reconstruction loop: this run is the **full mark4 pipeline** test, and
should now also exercise the **apm-structure-match** stage (see
`apm-structure-match-design.md`) — the local CPU baseline is logged (median 14% scope
coverage, type+multichar). GPU is only needed there if we use the pgvector/embedding
matcher (Rob's pattern) rather than typed overlap.

## Runbook (prepared scripts: `scripts/linode-4gpu-setup.sh` + `linode-4gpu-run.sh`)
On SSH, in order:
1. **Verify box**: `nvidia-smi` (4 cards), `command -v nvcc` (CUDA toolkit on image?),
   disk ≥200 GB, RAM ≥64 GB.
2. **Get code+data on the box**: rsync `futon6` from dev (or git clone) incl.
   `data/iatc-candidates/` (the 10 held-out papers + manifest).
3. **`linode-4gpu-setup.sh`** → installs vLLM 0.23.0, serves 70B-AWQ TP=4 on :8000;
   auto-detects nvcc and applies the eager/flashinfer-off fallback only if absent.
   Waits for readiness (~70s) and prints per-card memory.
4. **`linode-4gpu-run.sh`** → IATC reconstruction loop (70B) → `loop-run-70b/`.
5. **Owner review**: `iatc_argcheck.bb` + `substance_gate.py`, faithfulness spot-checks,
   non-uniform distribution; **compare 70B vs the 8B baseline (8B auto-failed 6/10)**.
6. Optional: the **apm-structure-match** CPU stage (scope extract + match) — runs on the
   box or dev; GPU only if we use the pgvector/embedding matcher.

## Asks from Joe to resume
1. Re-provision the 4-GPU box (powered down) — **with the CUDA toolkit on the image**
   this time — and hand over **IP + root SSH**.
2. Confirm the 70B checkpoint (the ungated AWQ candidate above, unless you prefer one).

## Source
Mined from claude-4 transcript `34299d44-…` (2026-06-16, pre-crash). The Agency JVM
crashed during the OOM but **restarted cleanly** (new pid, ~16:50+); XTDB Evidence
Landscape is back and its durable store survived — these provisioning facts can be
**cross-checked against the Landscape** (not yet done).
