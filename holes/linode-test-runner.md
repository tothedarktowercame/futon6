# Linode staged test runner — shake out the NEW pipeline on real GPU (before Rob)

*Author: claude-1, 2026-06-17. The integration test of the *whole* new checker pipeline on real
GPU, staged small (10 → 20 papers), before the superpod hand-off. Companion to
`proofcheck-run-invocation.md` (the content + preregistration) and distinct from
`warp-superpod-parallel-runner.md` (different hardware → different parallelism). Gated on Joe's
send (provision the 4-GPU box).*

## 0. Why this is needed (and why it's NOT the superpod runner)

Run #2 was a **no-GPU** pass over *existing* `loop-run-70b` graphs. The new pipeline — producer
(anatomy → candidates → **70B IATC**) **+** the checker spine (rungs → R2d → CAS-SEL → CAS-CERT) —
has **never run end-to-end through the 70B and out the other side**. The Linode test does that on
real GPU, small and staged, to shake out integration dust before Rob/superpod.

**The Linode's parallelism is the opposite of the superpod's** (from `linode-4gpu-setup.sh` + the
go-live prereg):
- 4× RTX 4000 Ada (20 GB each). The 70B-AWQ-INT4 runs **TP=4 at ~18.8 GB/card**, `--gpu-memory-
  utilization 0.95` → **the model nearly fills all 4 GPUs as ONE server.**
- ⇒ **No device-sharding, no extra replicas, no mem-util headroom** (Rob's "more copies per GPU"
  fits the superpod's near-empty A100s, NOT this). The **only GPU-parallelism lever is vLLM
  continuous-batching of concurrent requests** to the single TP=4 server.
- The **producer (CPU) + checker spine (CPU)** parallelize across the box's CPU cores, independently.

So the Linode runner = **request-concurrency on one shared model + CPU-core parallelism**, where
the superpod runner = **device/GPU sharding + CPU sharding**. Complementary, not the same code.

## 1. The one genuinely-new piece: a concurrent IATC driver

Everything else **composes from what exists**. The new bit is small and specific:
- Today's `mark3_iatc_loop.py` processes papers **sequentially** (one candidate → 70B → gate →
  retry). On the Linode that leaves the GPUs batching-starved (one request in flight).
- **New:** a bounded-concurrency driver that issues **M papers' IATC requests at once** (asyncio /
  thread pool, M ≈ 8–16) against the single `OPENAI_BASE_URL` server, so vLLM **batches them**. M is
  a flag (`--concurrency`), bounded so the KV-cache (the ~1 GB/card headroom at util 0.95) doesn't
  OOM. This is the entire "parallel runner" delta for the Linode.
- Keep determinism where it matters: the per-paper gate/retry logic is unchanged; only the
  scheduling is concurrent. Output is per-paper, order-independent.

## 2. Runner stages (compose existing + the new driver)

```
0. PROVISION + SERVE   linode-4gpu-setup.sh   (StackScript reboot → vLLM 70B TP=4, ~70s)   [exists]
1. PRE-FLIGHT GATE     pipeline_witness.py --witness <each pid>   (producer seams conform)   [exists]
2. PRODUCER (CPU‖)     anatomy → candidates over the batch, parallel across cores            [exists]
3. IATC (GPU, batched) the NEW concurrent driver: M papers async → the one TP=4 server        [NEW, small]
4. GATE (CPU)          iatc_repair + argcheck + substance + rung-2 sidecar                     [exists]
5. CHECKER (CPU‖)      iatc_semcheck (R2a/b/c+R2d) → cas_cert  → per-paper CAS-CERT            [exists]
6. SCORECARD           cas_cert --graph-dir <run> → verdict+confidence vectors + residual map  [exists]
7. DECOMMISSION        linode-cli destroy when GPU stages done; checker can finish locally     [exists]
```

## 3. Staged plan (10 → 20, resumable)

- **Stage A — 10 papers (smoke):** the goal is *does it run end-to-end without dust*, not stats.
  Pick 10 with full anatomy artifacts (the witness gate passes 1–4). Watch: vLLM serves, the
  concurrent driver batches (nvidia-smi util up, no KV OOM), graphs emit, gate runs, CAS-CERT
  certificates come out with the verdict+confidence shape from run #2.
- **Stage B — +20 papers (confirm):** resume on a fresh 20 (the make-like `runnable()`/sidecar
  check skips Stage A's done papers). Confirm throughput holds with concurrency, and the scorecard
  is sane across 30. This is the "no surprises at slightly larger N" check.
- **Resumable** throughout — a paper whose `.rung2.edn` / cert exists is skipped.

## 4. Preregistration (what the FIRST real GPU run validates — score against it)
- **L1 end-to-end:** all 10 (then 30) papers traverse producer→IATC→gate→checker→CAS-CERT with **no
  stage erroring**; certificates emit with `verdict` + `confidence{level,limiting_factors}`.
- **L2 enrichment, not raw (C5 from run #1):** these candidates carry the **full enrichment**
  (grounding/scopes/expository), unlike run #1's raw-source candidates. Predict: warrant-resolution
  + anchor-faithfulness **higher** than run #1's `loop-run-70b` baseline (warrant ≈ 6/28). This is
  the size-vs-enrichment confound finally tested on real output.
- **L3 batching works:** with `--concurrency M`, GPU util stays high without KV OOM; wall-clock per
  paper drops vs sequential. (The honest Linode throughput number — report it.)
- **L4 honesty holds at the producer's edge:** CAS-CERT verdicts on the *fresh* graphs trace to real
  rung verdicts (spot-check ≥3, as in the run-#2 review); confidence is `medium` (symbol/technique
  N/A) and names those grains.
- **L5 determinism of the checker** (not the 70B): re-running the checker over the same graphs is
  byte-identical (the 70B itself isn't deterministic; the *checker* must be).

## 5. What to build vs compose
- **Build (small):** the concurrent IATC driver (§1) — `--concurrency M`, bounded async over the one
  vLLM server. ~a focused script wrapping `mark3_iatc_loop`'s per-paper call. Dispatchable.
- **Compose:** provision/serve, witness pre-flight, producer, gate, checker, scorecard,
  decommission — all exist; the runner is a thin shell stitching them + the staged 10/20 control.
- **Relationship to the superpod fork:** the *checker spine* is identical on both; the *producer
  scaling* differs (Linode = request-batch on shared model; superpod = device-shard). The Linode
  test de-risks the pipeline; the superpod fork scales the substrate. Lessons (concurrency limits,
  enrichment payoff) carry forward.

## 6. Recommendation
Build the **concurrent IATC driver** now (the one new piece — CPU-testable against a stub/local
endpoint before any box), and assemble the thin staged-runner shell. Then the Stage-A 10-paper run
is a single send-gated command when Joe provisions the box. Everything downstream of the 70B is
already built, reviewed, and deterministic.

## Concurrent driver build note — codex-3

Implemented `scripts/mark4_iatc_concurrent.py` as the Linode request-concurrency driver.  It reuses
`mark3_iatc_loop.py`'s per-paper prompt/backend/repair/gate/rung-2/retry functions and only changes
the scheduler: a `ThreadPoolExecutor` bounded by `--concurrency` feeds multiple papers to the single
OpenAI-compatible vLLM endpoint.  Output remains per-paper (`<paper>.edn` + `<paper>.rung2.edn`) and
completed papers are skipped on resume.

CPU stub gates passed: `python3 -m py_compile scripts/mark4_iatc_concurrent.py` and
`pytest -q tests/test_mark4_iatc_concurrent.py`.  The tests compare concurrent stub output against
one-paper-at-a-time sequential output, assert the in-flight backend calls never exceed `M`, verify
input-order independence, and check resume skips existing graph/rung2 pairs.  Real GPU throughput and
KV-cache-safe `--concurrency` remain Linode Stage-A measurements.
