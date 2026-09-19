# Running all of category theory (~5,000 papers) on what's free now

Target set by Joe 2026-09-19: 2 nodes available, 8 GPUs. 5,000 papers = all of CT.
**The purpose of the run is to make a quality assessment possible** — there has not
yet been enough scale to judge quality. That constraint drives everything below.

## The answer: it fits, and it needs no quality changes

Measured from the 0919b probe, per paper: **54 LLM calls, 75.8k prompt tokens,
14.8k output tokens**. At 5,000 papers that is 270,416 calls, **379M prompt
tokens and 74M output tokens**.

Note the ratio: **prompt:output is 5.1:1**. This workload is prefill-dominated,
not decode-dominated. The S3 `steps` call re-sends the whole candidate plus the
nodes it just received, which is where a third of the prefill goes.

Hardware, from `futon6/holes/warp-superpod-parallel-runner.md` (not guessed):
**A100-80GB**, Rob's device-pinned convention of **1 shard per GPU**
(`CUDA_VISIBLE_DEVICES=k -> <out>-shard-k`), runner default model **70B-AWQ**.
AWQ matters: ~40 GB of weights fits one A100-80GB outright, so one full replica per
GPU with ~40 GB left for KV cache — no tensor-parallel split needed, and Rob's
existing sharding convention applies to the 70B unchanged.

Proof-count budget, reconciled with `holes/mark7-superpod-run-playbook.md` §4:

| basis | proofs at 5,000 papers | prompt | output |
|---|---|---|---|
| playbook low | 27,000 | 238M | 46M |
| probe rate (0.67 x 15.5/paper) | 51,925 | 382M | 75M |
| playbook head rate (0.91 x 15.3) | 69,615 | 484M | 96M |
| playbook high | 64,000 | 451M | 89M |

**The probe independently confirms the playbook's yield estimate.** The playbook
derived ~15.3 proofs per contributing paper from the top-100 most-cited; the probe
gives 124 candidates from 8 contributing papers = **15.5**. Two different samples,
the same number.

What does *not* agree is the **contributing rate**: 91/100 for the top-cited head,
but only 8/12 here. That is the S1 zero-proof defect, and it swings the window
budget by ~35% — which is another reason to keep those papers in and measure it.

### The probe ran on a fallback box, not this hardware

Joe, 2026-09-19: the 8-GPU nodes were not available when the probe ran, so it went
through a single serial endpoint on different hardware. **The probe's ~23 s/call is
therefore a property of that box, not of the pipeline**, and it sets neither a floor
nor a ceiling on the 8x A100 path. Nothing in the run records what it ran on —
`host-config.jsonl` captures the model tag and endpoint but no GPU model, count,
serving stack or concurrency, which is exactly how that rate invites misreading.

What survives unchanged is the part that does not depend on hardware: the **token
counts**, measured from artifact sizes. Those are what the window budget should rest on.

### State the window as a throughput bar, not a predicted runtime

Since no measurement of the real allocation exists yet, the useful form of the answer
is the aggregate throughput that has to be beaten to finish 5,000 papers in 20 hours:

| basis | proofs | prompt | output | needed out tok/s (8 GPU) | per GPU | needed prefill tok/s | per GPU |
|---|---|---|---|---|---|---|---|
| playbook low | 27,000 | 238M | 46M | 639 | **80** | 3,308 | 414 |
| probe yield rate | 51,925 | 382M | 75M | 1,041 | **130** | 5,301 | 663 |
| playbook high | 64,000 | 451M | 89M | 1,236 | **155** | 6,267 | 783 |
| head rate | 69,615 | 484M | 96M | 1,327 | **166** | 6,716 | 840

Single-stream 70B-AWQ on one A100-80GB is ~20-30 output tok/s. So even the most
pessimistic row asks for about **5-6x single-stream per GPU** — which is what batching
at concurrency 32-64 routinely delivers, usually by a wide margin. On 2 nodes the bar
halves again.

This is a bar that can be confirmed or refuted in half an hour of GPU time, and it does
not depend on any estimate of mine about A100 throughput. Run the calibration, compare
against the table, and the window question is settled rather than argued.

## What to change: the serving layer, and nothing else

Because this run is what makes the quality assessment possible, **anything that
changes prompts or models changes the thing being measured.** Freeze the pipeline.

1. **Serve the 70B with vLLM, concurrency 32–64 per replica.** The probe ran
   `[1/124] … [124/124]` strictly serially against a single endpoint at
   `127.0.0.1:11436` — ~10–20 output tok/s sustained, i.e. single-stream decode.
   This is the entire gap. Identical model, identical prompts, identical gates.
2. **`--enable-prefix-caching`.** The S3 steps call's prompt begins with the same
   ~2,165-token candidate the nodes call just used. Automatic prefix caching makes
   that shared prefix nearly free — roughly a third off prefill, no code change and
   no effect on outputs.
3. **Make the client concurrent.** The batching server does nothing if the loop
   still awaits one response at a time. This is the one code change required, and
   it touches scheduling only, not prompts.
4. **Sample the per-paper HTML renders instead of emitting all of them.** The probe
   wrote 37 MB of render for 8 papers (15 MB for `0806.1324` alone); at 5,000 papers
   that is ~23 GB of the ~50 GB total. Rendering is presentation, not pipeline —
   sampling it costs nothing scientifically.

## What NOT to change for this run

Two things I recommended in `EFFICIENCY.md` are wrong for *this* run, given its purpose:

- **Do not cascade 8B→70B.** It is a sound lever later, but running a mixed-model
  pipeline through the first CT-wide assessment would measure the cascade rather than
  the pipeline. Land the quality baseline on 70B first; then the cascade has something
  to be compared against.
- **Do not gate out the zero-proof papers.** Four of twelve papers identified 0 proofs
  at S1 — including two that had proof-move cues. Skipping them would save ~20% of S4
  and hide the defect. At 5,000 papers, *how often S1 finds no proofs* is one of the
  most valuable numbers the run can produce. Keep them in and measure it.

Collapsing S3's two calls into one is a genuine ~35% prefill win, but it rewrites the
prompt — so it belongs after the baseline, or as an A/B on a held-out slice, not in
the assessment run.

## Before you book the window

- **Calibrate, don't guess.** Run 200 candidates at concurrency 1, 8, 32 and 64 and
  measure achieved tok/s and time-to-first-token. Half an hour of GPU time converts
  every estimate above into a real number, and tells you the replica/TP split to use.
- **Instrument.** The probe recorded no per-call latency or token counts anywhere —
  `metrics.jsonl` is quality metrics only. Emit prompt tokens, completion tokens and
  latency per call. Without it the next capacity question is guesswork again.
- **Pin and record the exact model.** The runner's default is **70B-AWQ**, but the
  probe's `host-config.jsonl` records only the tag `llama3.1:70b` against an endpoint
  on port 11436 — no quantization, no serving stack. For a run whose whole purpose is
  a quality baseline, "which 70B, at what precision" cannot be left implicit. Record
  model, quantization, vLLM version and concurrency in `host-config.jsonl`.
- **Use the runner that already exists.** `warp-superpod-parallel-runner.md` §6c
  already specifies `--shards`, `--devices`, `--copies-per-gpu` (memory-sized auto),
  `--model` (default 70B-AWQ) and `--resume`; `linode-test-runner.md` §3 already says
  to watch "vLLM serves, the concurrent driver batches (nvidia-smi util up, no KV
  OOM)". None of this needs designing — it needs the 8-GPU nodes, which now exist.
- **Record the hardware in the run.** `host-config.jsonl` should carry GPU model and
  count, serving stack and version, concurrency and `--copies-per-gpu`. The 0919b probe
  is a worked example of why: without it, a rate produced by a constrained fallback box
  reads as if it were the pipeline's own.
- **Try `--copies-per-gpu 2`.** Rob's own throughput insight (same doc: an A100-80GB
  at 100% util but 11/82 GB used is not maxed) applies directly. With AWQ weights at
  ~40 GB there is not room for two full replicas plus KV, so the realistic version of
  his point here is *one replica with a large KV cache and high concurrency*. Measure
  both; his instinct to fill the memory is right even if the shape differs.
- **Check resume.** S10 already logged one `interrupted` attempt in a 3.5h run. Over
  ~18h and 270k calls an interruption is likely. Per-candidate outputs are already
  persisted under `artifacts/graphs/.attempts/`, so resume looks feasible — verify it
  by killing and restarting a short run before trusting it at scale.
- **Budget ~50 GB** of run output at 5,000 papers (~43 GB artifacts + render),
  or ~27 GB with renders sampled.

## Confirmed by ivan (Rob's agent), 2026-09-19 22:1x, Matrix "Private Federation Proof"

Every inference in this document about the probe's serving path is now confirmed
first-hand, and one of them upgrades from guess to fact:

- **The probe used Ollama, not vLLM.** Model tag `llama3.1:70b` is Ollama's default
  70B GGUF at **~4-bit**, a single endpoint at `127.0.0.1:11436`, single-stream, with
  `OLLAMA_CONTEXT_LENGTH` pinned to **32768**. It is *not* the runner's 70B-AWQ default.
- **The probe ran on ONE Slurm-allocated GPU** — the 8-GPU nodes were unavailable.
- ivan's own conclusion, unprompted: *"treat the probe as a functional S1–S12 pass,
  not a pinned quality/precision baseline."*
- The vLLM + `--enable-prefix-caching`, one-replica-per-GPU, concurrency 32–64
  direction is endorsed as "well-matched to a prefill-dominated workload", and they
  have a **governed superpod vLLM-ensure path** for it already.
- Still with Rob, not ivan: the free-node GPU count, A100-80GB confirmation, the
  replica/concurrency choice, the 130–166 tok/s per GPU target, and go-ahead for the
  200-candidate calibration. ivan has surfaced all of them to him.

### What the one-GPU fact does to the arithmetic

It sharpens the throughput bar into a measured multiple instead of an assumed one.
The probe's ~20 output tok/s was **one GPU, single-stream, Ollama 4-bit GGUF**. The bar
is 130–166 output tok/s per GPU across 8 GPUs. So the batching speedup required
per GPU is about **7–8x single-stream** — and that is measured on their hardware,
not estimated from a generic A100 figure. vLLM at concurrency 32–64 routinely exceeds
it, and AWQ under vLLM should start from a higher single-stream baseline than
Ollama's GGUF, so the true multiple needed is likely lower still.

### The consequence nobody should miss

The probe's quality numbers — grounding 66.06%, checker 242/242, substance 121/121,
accretion 0.114 → 0.275 — were produced by a **4-bit GGUF at 32k context under
Ollama**, not by the 70B-AWQ-under-vLLM the pipeline is built around. So the CT-wide
run will not merely be faster on different hardware; it may land at different quality.
That is an argument *for* running it, not against: there is currently no pinned quality
baseline at all, and this run creates the first one. It also means the probe's numbers
must not be quoted as the pipeline's quality, in either direction.
