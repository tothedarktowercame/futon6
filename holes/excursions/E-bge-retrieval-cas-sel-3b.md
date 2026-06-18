# E-bge-retrieval-cas-sel-3b

Author: claude-1, 2026-06-18. Bounded Linode experiment, modelled on
`E-70B-on-raw-control-arm.md`. Owns the runner `scripts/linode-bge-retrieval.sh`
and the payload `scripts/cas_sel_3b_embed_experiment.py`.

## Why this experiment exists

CAS-SEL-3's Tier-0 retrieval is classical **hotword overlap**, with a measured
ceiling: recall@4 = **15/22**, full-pool ceiling **19/22** — three fixture steps have
**zero lexical overlap** with their correct pattern (`a93J05/s3` "z=z₀+mω₁+nω₂" →
`quotient-by-irrelevance`; `a96J01/s2` → `construct-auxiliary-object`; `b97J01/s6`).
CAS-SEL-3b proposed an embedding modality to lift this. On the dev box, **bge-small**
gave union recall **17/22** and recovered **none** of the 3 zero-overlap steps, and
**bge-large — the spec's model — was killed loading** (1.3 GB; dev-box memory; the
OOM lesson in miniature). So the question is unresolved on the dev box.

**This experiment resolves it on the bigger box, and it is discriminating:**
- **bge-large recovers the 3 zero-overlap steps** → the ceiling was **model size**;
  ship the embedding modality for CAS-SEL-3b and re-pin the honest-recall test up.
- **bge-large recovers none** → the ceiling is **text-vs-structure**: those matches
  are *structural*, not lexical or semantic-text (a step's prose shares nothing with
  its pattern; the link is the argument shape). That is the empirical case for the
  **R-GCN / structure-first** direction (final-checklist §6) — text retrieval, at any
  size, plateaus there.

Either outcome is decisive and cheap to get.

## Design

- **Standalone embedding job.** It does **not** serve or need the 70B. Runs on **CPU**
  by default (`DEVICE=cpu`) so it never contends with a vLLM 70B that may be filling
  the GPUs at TP=4. The corpus is tiny (39 patterns + 22 steps) — the only real cost is
  the model load + a one-time ~1.3 GB HF download for bge-large. (Small-data BGE is the
  README-embeddings "cheap, safe" case — *not* the big-data recompute that OOM'd.)
- **Portable, no futon3 needed on the box.** The payload reads the committed pattern
  snapshot `data/cas-select/pattern-texts.json` (39 patterns) when futon3 (cas_select's
  live source) is absent, and the fixture steps from the futon6 checkout. On the dev box
  it uses the live patterns and refreshes the snapshot.
- **Asymmetric BGE retrieval** (the documented mistake to avoid): pattern texts embed as
  passages; step texts embed as queries with the instruction prefix
  `"Represent this sentence for searching relevant passages: "`. (No prefix for MiniLM.)
- **Compares** bge-large vs bge-small vs MiniLM × two pattern-text representations
  (`title+conclusion+hotwords`, `full`), each with an **embedding-collapse audit**
  (cosine-to-mean std; `<0.01` collapse, `<0.05` mild — `audit-graph-embeddings.py`).
  The collapse to avoid is **R-GCN-specific**; BGE-text is the validated escape — the
  audit here just confirms the pattern vectors discriminate.

## What needs to run

On the provisioned box, from the futon6 checkout (`$REPO`):

1. `bash scripts/linode-bge-retrieval.sh` — ensures `sentence-transformers` in the venv,
   then runs `cas_sel_3b_embed_experiment.py` for each model × representation, prints a
   headline table + the discriminating verdict, writes per-model JSON to `$OUTDIR`
   (default `/tmp/cas-sel-3b`). Env overrides: `REPO/VENV/PYTHON/DEVICE/MODELS/REPRS/OUTDIR`.

## Artifacts

| what | where |
|---|---|
| per-model results | `/tmp/cas-sel-3b/<model>__<repr>.json` (recall, zero-overlap recovery, collapse audit) |
| pattern snapshot (committed) | `data/cas-select/pattern-texts.json` |
| dev-box baseline | bge-small: hot 15/22 · embed 12/22 · union 17/22 · recovered NONE · collapse mild (0.0226) |

## How to read the result

- **bge-large `accept=YES`** (all 3 recovered, union > 19/22) → model size; ship embedding
  CAS-SEL-3b, re-pin `test_tier0_retrieval_recall_is_honest` upward, note the embedding
  modality is model-free *of the generative LLM* (an embedding model is a separate tier).
- **bge-large `accept=no`** (recovers NONE) → text-vs-structure ceiling → the R-GCN /
  structure-first direction (§6) is the real path for those steps; record it as the
  decisive evidence and keep CAS-SEL-3b on hotword-only for now.
- **`collapse=mild/COLLAPSE`** on the pattern vectors → the representation
  under-discriminates; `--repr full` is the first lever.

## Status

Specified + runner and payload written; **payload validated on the dev box** with
bge-small (reproduces hot 15/22 · embed 12/22 · union 17/22 · recovers none · collapse
mild). The bge-large arm is what this box adds. **Send-gated to Joe** (box time). Hand
the runner to the agent orchestrating the Linode session; it is independent of the 70B
steps and can run alongside or alone.
