# Linode 1-GPU test run — IATC model loop on ~10 papers (2026-06-16)

**Owner:** Joe + claude-4 · **Status: PREPARED, awaiting Linode provisioning (Joe)**
**Goal:** prove the validated reconstruction loop end-to-end on a real model before
committing to the superpod. codex-4's honest H2 finding: a script can *select*
candidate passages but cannot *reconstruct* the warranted DAG — that needs an LLM.
This run validates: scripted candidate-extraction → **LLM reconstruction** →
self-gate (checker + substance) → owner review.

## What is already prepared (committed, no GPU needed)
- **10 test papers** — gh200 ids with layer-(a) marks, **excluded from the accepted
  15-graph pilot** so this is a fair held-out test. Candidates in
  `data/iatc-candidates/` (one `*.candidate.json` per paper + `manifest.json`),
  each carrying the proof-passage **source window** + **binder-context** (variable
  typings) the model reads.
- **`scripts/mark3_extract_candidates.py`** — the scripted half (passage selection
  salvaged verbatim from the rejected generator; emits reading material, no graph).
  Re-run anytime: `python scripts/mark3_extract_candidates.py`.
- **`scripts/mark3_iatc_loop.py`** — the model loop: few-shot prompt (18 valid IATC
  seeds: 3 checker fixtures + 15 pilot) + source window → LLM → parse EDN →
  `iatc_argcheck.bb` AND `substance_gate.py` → retry (≤3, gate errors fed back) →
  emit → cross-item batch substance gate. Plumbing validated locally with
  `--backend stub` (10/10 gated, batch-substance PASS); only the model call is unproven.
- **Gates** (auto, in-loop): `scripts/iatc_argcheck.bb` (structural) +
  `scripts/substance_gate.py` (anti-shell: filler/vacuity + cross-item template/
  canned-warrant collapse).

## What Joe provisions
A **1-GPU Linode**. The default model is **LLaMA-class ~8B** (`meta-llama/Llama-3.1-8B-Instruct`),
which fits a single ~24GB GPU; a single 40/80GB GPU allows a larger or less-quantized
model. Serve it with **vLLM's OpenAI-compatible server** (this is the OpenAI *wire
protocol*, not OpenAI the provider — the model is local LLaMA on the Linode GPU):

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct --port 8000     # serves /v1/chat/completions
```

## The run (after the model is serving)
```bash
cd futon6
git pull                      # get this branch (candidates + harness + gates)
OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=x \
  .venv/bin/python scripts/mark3_iatc_loop.py \
      --candidates data/iatc-candidates \
      --out data/iatc-argument-graphs/loop-run \
      --backend openai --model meta-llama/Llama-3.1-8B-Instruct
```
Output: `data/iatc-argument-graphs/loop-run/<id>.edn` per gated paper; per-paper
attempts in `.attempts/`. The loop prints `N/10 gated PASS · batch-substance PASS/FAIL`.

## Acceptance / owner review (claude-4, after the run)
The gates are necessary-not-sufficient (that's the whole H1/H2 lesson). On return I will:
1. confirm gate results (checker + substance, per-item and cross-item batch);
2. **faithfulness spot-check** ≥3 graphs against source at the cited line anchors
   (the method that caught the H2 shells: node ↔ source-claim, relation ↔ prose
   connective, missing-warrant ↔ genuine elision);
3. check the node/edge/hole **distribution is non-uniform** (no template collapse).

**Success = ** most of 10 gate-PASS **and** faithfulness holds on the spot-checks
**and** non-uniform distribution. That greenlights scaling the loop (the rest of
gh200 + the giants) — on bigger compute, same harness, same gates.

## Notes
- If the 8B model produces malformed EDN often, the loop's 3-attempt retry feeds the
  gate error back; if pass-rate is low, bump the model size (the 1-GPU run is exactly
  to measure this) before scaling.
- Nothing here is superpod-specific; the superpod's only added role is scale.
