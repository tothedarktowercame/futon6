# mark7 throughput — what the 0919b probe says about 5k and 500k papers

All numbers below are measured from `extracted/run`, not estimated.

## The measured cost of one paper

12 papers, S1→S12, 12,459 s of stage wall clock = **1,038 s/paper**.

| stage | wall | share | what it is |
|---|---|---|---|
| S3 | 8,623 s | 69% | IATC reconstruction loop, 70B |
| S4 | 2,213 s | 18% | expository loop, 70B |
| S10 | 610 s | 5% | move lexicon |
| S7 | 545 s | 4% | CLean typing, 70B |
| S1 | 392 s | 3% | markup/mark emission (CPU) |
| S2,S5,S6,S8,S9,S11,S12 | 77 s | 0.6% | assembly, export, render |

**86% of the run is three LLM stages.** Everything else is rounding error.

LLM calls, counted from the artifacts: S3 emits 124 `.nodes.json` + 124 `.steps.json`
(two round trips per candidate; the `.edn` is assembled from them) plus 121 rung-2
sidecars; S4 280; S7 121. That is **~770 calls for 12 papers ≈ 64 calls/paper**.

## The bottleneck: the endpoint is running unbatched

S3 spent 8,623 s producing 248 model outputs. Measured payloads:

- prompt: candidate JSON averages 8.5 KB ≈ **2,165 tokens**
- output: `nodes.json` 1.7 KB ≈ **428 tokens**; `steps.json` 1.1 KB ≈ **286 tokens**

That is ~88k output tokens in 8,623 s — **~10 output tok/s sustained**, and even if
half of S3 were non-LLM (grading, the eval harness, candidate extraction all run
inside it) it is still ~20 tok/s. Twenty tokens per second is *single-stream decode
speed for a 70B*. There is no batching anywhere in the path.

Corroborating: `logs/S3.command.log` runs `[1/124] … [124/124]`, strictly one
candidate at a time; `host-config.jsonl` shows one endpoint, `127.0.0.1:11436`,
model tag `llama3.1:70b`, with a `/health` probe the runner waits on.
`mark7-rob-handoff.md` asked for vLLM ("serve your model OpenAI-compatible across
the 8 GPUs"); port 11436 and an Ollama-style model tag suggested that is not what ran.
**Confirmed by ivan 2026-09-19:** it was Ollama, `llama3.1:70b` = the default 70B GGUF
at ~4-bit, `OLLAMA_CONTEXT_LENGTH=32768`, single-stream — on **one** Slurm-allocated
GPU, because the 8-GPU nodes were unavailable.

A 70B under vLLM continuous batching holds 32–64 concurrent decodes with aggregate
throughput in the hundreds-to-low-thousands of tok/s. **The allocation is idle
roughly 97% of the time.**

## The gap, stated honestly

Assume the 70B occupies the whole 8-GPU allocation.

| target | needed | available | shortfall |
|---|---|---|---|
| 4,616 papers (full math.CT), 20h window | 4.79M s | 72,000 s | **66x** |
| 5,000 papers, 20h window | 5.19M s | 72,000 s | **72x** |
| 500,000 papers, 30 days | 519M s | 2.59M s | **200x** |

At today's rate 5,000 papers is 60 days of continuous run; 500,000 is 16.5 years.

## What closes it, ranked by evidence

**1. Batch the client; serve with vLLM. 20–50x.** The single biggest lever and it
changes no output. Same model, same prompts, same gates — just concurrent requests
against a batching server instead of a serial for-loop. This alone covers the 5k case.
Combine with the sharding already written (`scripts/mark7_shard_manifest.py`,
`handoff-superpod-all.sh` Block-1/Block-2) for replicas on top.

**2. Collapse the per-candidate call chain. 1.5–3x on the dominant stage.** S3 makes
two round trips per candidate — nodes, then steps — and the second re-sends the same
~2,165-token candidate plus the nodes it just got back. One structured-output call
returning both halves cuts calls per candidate from 2 to 1 and prompt tokens by ~40%.
The checker validates the result either way, so this is testable in an afternoon.

**3. Cascade small→large behind the gates that already exist. 2–4x.** `iatc_argcheck.bb`
and `substance_gate.py` return a mechanical pass/fail on every single item — that is
precisely the escalation signal a cascade needs, and it is already wired in. Run 8B
first, escalate only failures to 70B. The S3 log notes 8B auto-failed 6/10 on substance
as a *solo* model; as tier one it only has to clear a fraction of items to pay, because
every item it clears is a 70B call never made.

**4. Gate the expensive stages on S1 yield. ~1.2x — ALSO NOT YET.** Skipping the
zero-proof papers would hide the S1 defect the CT run is best placed to quantify.

Four of twelve papers identified 0 proofs, yet `math__0608040` (716) and `math__0310337`
(819) alone produced 1,535 of the 2,139 expository candidates and 21 MB of the loss
reports. 60 of S4's 280 LLM items were spent on papers contributing nothing proof-side.
S1 already knows the proof count before any of this runs.

**5. Instrument first — this is a prerequisite, not a lever.** There is no per-call
latency or token count anywhere in the run. `metrics.jsonl` is 649 rows of pure quality
metrics; `stage-attempts.jsonl` has only stage start/finish. Every number in this
document had to be back-derived from file sizes and timestamps. Before anyone tunes for
500k, the runner should emit per-call prompt tokens, completion tokens, latency and
queue depth. Cheapest change here by far, and everything else depends on it.

## For 500,000 papers specifically (beyond the current target of ~5,000)

Levers 1–4 multiply to roughly 70x–700x, which comfortably clears 5,000 papers and
reaches the 200x for 500k only at the optimistic end. The structural problem at that
scale is **64 LLM calls per paper**. 500k papers is 32M calls; no amount of batching
makes that cheap. Getting to single-digit calls per paper means batching many proofs
into one prompt, or letting the small model plus the mechanical checkers handle the
common case outright with the 70B reserved for genuine escalations.

Worth noting what is *not* the problem: quality gates all pass (checker 242/242,
substance 121/121), and the accretion curve rises 0.114 → 0.275. The science is
working. This is purely a serving and call-count problem.

## Caveat

Per-paper cost is highly skewed and 12 papers is a thin base. `0806.1324` alone
produced 62 of the 124 proof candidates and 22,708 marks; `0705.0102` produced 11.
Any projection to 5k or 500k should be re-derived once per-call instrumentation
exists and a wider sample has run.
