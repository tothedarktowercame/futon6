# E-70B-on-raw-control-arm

Author: claude-2, 2026-06-18. Bounded experiment (RAW-CTL on the proofcheck
readiness card). Owns the runner `scripts/linode-4gpu-run-raw.sh`.

## Why this experiment exists

The mark4 go-live comparison contrasted an **enriched 70B** run against a
**blind 8B** run — and concluded enrichment helps. But that comparison moved
**two variables at once**: the *enrichment* (inlined deterministic anatomy:
symbol typings, scopes, proof-moves, definitions) **and** the *model size*
(70B vs 8B). So the win can't be attributed to enrichment alone — it's
confounded with model size.

**RAW-CTL isolates the enrichment variable.** Hold the model fixed (the same
70B), hold the candidates fixed, and strip **only** the enrichment. Whatever
delta remains vs `loop-run-70b` is attributable to enrichment alone.

The question it answers is the **cost-at-scale** one: before we run the 70B over
all of arXiv, do we need to pay for the enrichment pipeline, or does the raw 70B
do just as well? (Priority was deliberately lowered after the rung-2 / discursive
reframe — generator quality matters less than the checking — but this is the
clean answer to the enrichment cost question.)

## Design

- **Tightest possible control.** The raw candidates are *derived from the exact
  enriched candidates the enriched arm used* (`data/iatc-candidates`), emptying
  only the `enrichment` array → `data/iatc-candidates-raw`. Same source windows,
  same binders, same line ranges; the single changed variable is the anatomy.
  (This beats re-extracting, which could drift the windows.)
- **Zero source edits.** `mark3_iatc_loop.py`'s precondition gate
  (`require_enriched`) accepts a candidate iff `schema == "iatc-candidate/v2-enriched"`
  **and** an `enrichment` key is present; `render_enrichment` renders an empty
  array as "(no deterministic anatomy detected in this window)". So the raw
  candidates **retain the schema string as a gate token**, set `enrichment: []`
  (the real control variable), and carry an explicit
  `"_control_arm": "raw-no-enrichment"` marker. The arm is unambiguous via that
  marker + the `…-raw` input/output dirs.
  - *Cleaner alternative (not used, to keep the handoff a single script):* add an
    `"iatc-candidate/v2-raw"` schema to the gate's accepted set and relax it.

## What needs to run, in sequence

All ON the provisioned 4-GPU Linode, from the `futon6` checkout (`$REPO`).

1. **Provision + serve the 70B** — `scripts/linode-4gpu-setup.sh`. Brings up
   the venv and serves `Meta-Llama-3.1-70B-Instruct-AWQ-INT4` via vLLM
   (tensor-parallel 4) as `mark4-70b` on `:8000`. (~15 min.)
2. **Enriched arm (baseline)** — `scripts/linode-4gpu-run.sh`. (Re)extracts the
   enriched candidates into `data/iatc-candidates` and runs the IATC loop →
   `data/iatc-argument-graphs/loop-run-70b` + `mark3-eval-*-70b`. **Required
   first** — RAW-CTL derives its candidates from, and compares against, this.
3. **Raw control arm** — `scripts/linode-4gpu-run-raw.sh` (this excursion). It:
   - derives `data/iatc-candidates-raw` from `data/iatc-candidates` (enrichment
     stripped);
   - runs the IATC loop (same 70B) → `data/iatc-argument-graphs/loop-run-70b-raw`;
   - runs the non-fatal eval tail → `mark3-eval-*-70b-raw`;
   - generates `cas_cert` certs for both arms and prints both eval summaries.
   - Invoke: `bash scripts/linode-4gpu-run-raw.sh` (env overrides
     `PORT/MODEL/REPO/PYTHON/ENRICHED_CANDS/ENRICHED_OUT/OUT/...`; same defaults
     as the enriched runner).
4. **Compare** — diff the two arms on the eval summaries
   (`mark3-eval-summary-70b.md` vs `…-70b-raw.md`: grounding / expository /
   prior-vs-posterior) and the `cas_cert` certs (aggregate gate + concept-grain
   + proof-grain; the run-#2 enriched concept-grain baseline is mean **0.867**).

## Artifacts

| Arm | graphs | eval | cert |
|---|---|---|---|
| enriched | `loop-run-70b` | `mark3-eval-*-70b` | `/tmp/enriched.cert.json` |
| raw | `loop-run-70b-raw` | `mark3-eval-*-70b-raw` | `/tmp/raw.cert.json` |

## How to read the result

- **Raw ≈ enriched** (substance / grounding / concept-grain hold up) →
  enrichment is **not needed** before the arXiv 70B. Drop it from the scale path;
  that's the cost win.
- **Raw degrades** → enrichment earns its keep; budget for it at scale.

## Status

Specified + runner written and verified (syntax + dry-run: 10 raw candidates,
253 enrichment marks stripped, all gate-pass). **Send-gated to Joe** (GPU spend).
Box provisioned 2026-06-18; setup in progress. Hand the runner to the agent
orchestrating the tests; run after steps 1–2 above.
