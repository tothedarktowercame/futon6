# LLM backend pluggability + cost at scale (Joe, 2026-06-17)

*"openai" in the pipeline = the OpenAI-compatible API *protocol*, not the vendor. All four LLM
call-sites read one env knob; the default is the **local 70B**. This note records the cost picture
at arXiv scale and the per-role backend strategy, because it's a real planning input for Rob.*

## It's local-first by default (not "retargetable" — already retargeted)

Every `call_openai` reads `OPENAI_BASE_URL` (default `http://localhost:8000/v1`) + `OPENAI_API_KEY`
(default `"x"`, a dummy — local vLLM ignores it). No `api.openai.com` is hardcoded anywhere.
- The four sites: `mark3_iatc_loop.py` (the **producer ③** — the heavy one), `cas_select.py` (the
  checker **Tier-1 verify**), `mark3_expository_loop.py` (⑤), `sfc_symbol_grounding.py` (SFC2b).
- To use a hosted API you'd **opt in** (`OPENAI_BASE_URL=https://api.openai.com/v1` + a real key).
  The out-of-box path is the local vLLM 70B (`vllm.entrypoints.openai.api_server`).

## Cost at scale — the dominant cost is the producer, and it's local

The LLM call budget is dominated by the **IATC producer** (~1× + retries per paper, ~5K tokens/call):
- math.CT (~5K papers): ~6.5K calls ≈ 32M tokens.
- all arXiv (~500K papers): ~650K calls ≈ 3.25B tokens.

| backend | math.CT (~32M tok) | arXiv (~3.25B tok) | nature of cost |
|---|---|---|---|
| **local 70B** (vLLM, owned/rented GPU) | GPU-hours | GPU-hours | compute-time (≈ free on owned superpod) |
| hosted big (GPT-4o-class, ~$4–5/1M) | ~$130–160 | **~$13–16K** | per-token API billing |
| hosted small (4o-mini-class, ~$0.6/1M) | ~$20 | ~$2K | per-token API billing |

**So Joe's instinct is right and the design already answers it:** at arXiv scale a paid API is
expensive ($K–$10K+), but the dominant cost (IATC) runs **local by default**, so scaling is a
**GPU-time** question (Stage-A measures the Linode throughput; the superpod amortizes it), not an
API-bill question. A hosted backend is a knob you'd flip only for a **small quality experiment**.

## What different backends do "in principle" — and how we measure it

Backend choice is a **quality × cost × reproducibility** trade, and it **splits by role**:
- **Producer (heavy, structured generation):** model quality compounds — a stronger model likely
  yields better-anchored, more-complete argument graphs → higher warrant-resolution / fewer orphans
  / fewer R2a flags. This is where quality buys the most.
- **Checker verify / rung-3 judge (bounded classification):** small calls on the ~27% residue; a
  cheap/local model likely suffices. **You can point the two roles at different `OPENAI_BASE_URL`s**
  — strong+costly for the producer, cheap+local for the bounded verify. A real cost lever the
  env-knob architecture already supports (per-role endpoint).
- **Reproducibility:** a pinned local model (temp 0, fixed seed) is a stable scientific instrument;
  a hosted API can silently change model versions under you, shifting your graphs. For a standing
  conformance instrument, local-pinned is the safer default.

**The elegant part — CAS-CERT is the instrument to MEASURE the backend difference.** The checker
spine is deterministic and model-free, so running the *same* papers through *different* producer
backends and diffing the CAS-CERT scorecards (warrant-resolution, miswires, orphans, thin moves)
is an **objective backend comparison**. "What difference does GPT-4o vs the local 70B make?" is not
a guess — it's a 10-paper experiment whose yardstick we already built. (A natural Stage-A+ add-on:
run the same 10 through two backends, diff the certs.)

## Recommendation
Stay local-first (it's the default + the cost answer). Treat hosted backends as an opt-in **quality
probe** on small batches, measured by CAS-CERT. If/when quality matters more than GPU-time on a
sub-corpus, the per-role split (strong producer / cheap verify) is the cost-efficient config — and
it needs no code change, just two `OPENAI_BASE_URL`s.
