# REPRODUCING — the math.CT structural-mining results

**Companion to:** `HANDOFF-superpod.md` (operator instructions),
`capability-proof-arxiv.pdf` (claims A1–A14 with warrants, in
`futon3c/holes/labs/M-diagramprover/`), and
`holes/excursions/E-superpod-hardening.md` (the hazard ledger, 40 entries).

## What "reproducing" means here

There is an LLM inside stages S3, S4 and S7, so byte-identical output across
hosts is not on offer and is not claimed. Reproducibility is defined at the
**invariant level**: a reproduction is a run whose *gate verdicts* and
*replay-harness checks* agree with ours, and whose corpus-level counts land
near the reference values below. Individual generations, retry counts, and
wall-clock are expected to vary.

Keep two words distinct (we do):

- **Reproduce** — same corpus, same model, same pipeline commit → same gate
  verdicts and invariant passes, counts near reference.
- **Replicate** — different model or serving stack → the A14 experiment
  (registered, not yet witnessed). Count differences there are *findings*,
  not failures.

Most of the enforcement is already in the pipeline rather than in this
document: the two entry gates are mandatory with no override flag, every
stage transition is ledgered under a corpus id, and a stage whose upstream
has no passing ledger entry for *this* corpus is refused rather than run.

## Pins

| what | value |
|---|---|
| pipeline | `futon6` @ `ce9cd20` (or later; the stepper + gates are the contract) |
| corpus manifest | `holes/mark7-16.ids.txt` — 16 arXiv ids, sha256 `8cfb8461353f55aeeccf1b5bf8812d3e33f80207cf6cf81c89fc1b441e045a9d` |
| corpus label | `math-ct-e2e-16` (16 papers: 12 originally declared + 4 discovered in the shared output directory by identity check I1 — see the capability proof §Replay) |
| concept substrate | `data/warp/{concept-index,def-snippets,defined-index,concept-usage}.json` + `data/concept-encyclopedia-ct.json` (~68 MB, in-repo) |
| model | GLM-4.5-Air, Q4 quantisation, served by llama.cpp's OpenAI-compatible endpoint |
| sampling / timeouts | as hardcoded in the loop scripts at the pinned commit (`mark3_iatc_loop.py`, `mark3_expository_loop.py`, `clean_box_typing.py`; `FUTON6_LLM_TIMEOUT` overrides the 300 s / 120 s defaults) |
| reference host ("Zone") | 256 GB CPU-only box, 32 cores; **4.1 tok/s** single-stream decode, 1.58× aggregate on two shards |

**Not yet recorded, wanted:** the exact GGUF file hash and the llama.cpp
build/version of the reference run. Neither was captured at run time. The
next run on Zone should record both here; until then, "same model" means
"GLM-4.5-Air Q4 under a current llama.cpp."

**Not shipped:** the arXiv source tarballs (per-paper licenses vary; we do
not redistribute). Fetch the 16 e-prints yourself from arXiv by the manifest
ids and point `FUTON6_EPRINTS` at the directory. The corpus is defined by the
manifest, not by any directory listing — that distinction is load-bearing
(hazard H35).

## The four tiers, cheapest first

### Tier 0 — replay attestation (seconds, no model, no eprints)

Given a run directory (ours or yours), the replay harness re-derives the
accounting from the artifacts:

```bash
.venv/bin/python scripts/replay_e2e.py --run-dir data/runs/<run-id> \
    --ids holes/mark7-16.ids.txt --corpus-id <corpus-id>
```

Eleven checks in four families — conservation (C1–C2), identity (I1–I3),
shape (S1–S3), persistence (P1–P3) — each tagged FAIL / WARN / not-yet-
applicable. **Agreement criterion: exact.** A reproduction of the accounting
is 11/11 non-FAIL. (Provenance imperfections WARN; artifact corruption
FAILs. `--through S<n>` scopes the suite to a partial run.)

Pass every path flag explicitly (`--run-dir --graphs --steps --clean --ids
--corpus-id`): the script's built-in defaults refer to the reference host's
legacy shared directories, an ids file not in this repo, and the pre-I1
corpus label `math-ct-e2e-12` — relying on them off-Zone will refuse or, worse,
attest the wrong artifacts.

### Tier 1 — environment attestation (~minutes, model, no eprints)

```bash
.venv/bin/python scripts/conformance.py \
    --endpoint $OPENAI_BASE_URL --model $MODEL --json conformance.json
```

Six checks; the decisive one asks the model a banana's colour under a schema
whose enum is `["purple", "octagonal"]` — **"purple" is the passing answer**
(the grammar overrode the model; "yellow" means your stack ignores
`response_format` and every LLM stage will emit templates). Two checks are
negative — they assert that a degenerate graph is refused and that a failed
stage exits non-zero. **Agreement criterion: all six verdicts match ours.**
The throughput figure is host-specific by design: it is not compared, it is
*collected* — please report it. This tier alone is a publishable data point:
it is the first thing we ask of any new host, and `conformance.json` is that
host's methods section.

### Tier 2 — same-model reproduction (hours to days, model + eprints)

```bash
export FUTON6_EPRINTS=/path/to/eprints  OPENAI_BASE_URL=...  MODEL=...
.venv/bin/python scripts/linode_stepper.py --plan --profile <linode|superpod>
.venv/bin/python scripts/linode_stepper.py --run --profile <profile> \
    --run-id <fresh-id> --corpus-id math-ct-e2e-16
```

Preflight and conformance run first, mandatorily. **Agreement criterion:**
both gates pass; 12/12 stages ledgered; Tier-0 replay 11/11 non-FAIL on the
resulting run directory; counts land near the reference values below.

We have **one** reference observation, so tolerance bands are honestly
uncalibrated. Report your deltas; the second run is what calibrates the
bands. Reference values (Zone, `math-ct-e2e-16`):

| quantity | reference |
|---|---|
| S1 anatomy | 16/16 papers, 320,337 marks, 0 failures *(no model — expect exact)* |
| S3 proof graphs | 98 gated PASS (argcheck + substance) |
| S3 first-pass retry rate | ~37% (reconstructions 36.7–37.8%; now emitted in-loop as `retry-rate-$RUN_ID.json` — quote yours from that file only) |
| S4 expository regions | 280/280 gated PASS |
| S5 verdicts | well-formed 6 · partial 82 · weak-extraction 10; strategy moves 202 grounded / 564 thin / 52 ungrounded |
| S7 typing | 94 typed / 4 cycle-rejected / 0 failed; entropy gate: mean off-diagonal cosine 0.02 (ceiling 0.85) |
| S8 export | 481 nodes / 304 edges |
| S10 lexicon | 732 distinct entries / 737 moves (zero reuse at n=16 — a starting point, not a finding) |
| S11 canon | 97 definitions → 77 shapes, 0 coverage gaps |
| S12 accretion | proof-move grounding 0.118 → 0.272, rising; expository 0.033 → 0.094 |

**Expected red, not your problem:** S3 also writes
`data/runs/$RUN_ID/anchor-faithfulness.txt`, which **exits non-zero by
design** while hazard H38 (checker and graphs on different line bases) is
open. Reference: 64.3% pass raw; 86.3% excluding the 25 frame-mismatch
graphs. A frame-mismatch report from your run is useful data, not a failure.

### Tier 3 — replication (different model / serving stack)

The A14 experiment. Everything in Tier 2 applies **except** the reference
counts: with a different model, only the gate verdicts and replay invariants
must hold. Divergences in the counts are the result — please report them
alongside your `conformance.json`.

## What to send back

Same list as the handoff, at any tier reached:

1. `conformance.json` (Tier 1+) — especially the throughput figure
2. `data/runs/$RUN_ID/` — phase ledger + artifacts (Tier 2+)
3. `retry-rate-$RUN_ID.json` (Tier 2+)
4. `anchor-faithfulness.txt` — red as described (Tier 2+)

## Honest boundaries

- n=1 reference run; every band above is a point estimate.
- The reference run's artifacts live on the Zone host, not in this repo;
  the repo carries the pipeline, gates, manifests, and substrate.
- The model pin is incomplete (no GGUF hash, no llama.cpp version) until the
  next Zone run records them.
- The full hazard ledger — 40 ways a stage can appear to succeed while
  failing, 37 closed — is `holes/excursions/E-superpod-hardening.md`, and it
  is the authority on itself. If a gate refuses your host and you believe the
  gate is wrong, that is a bug report we want, not a thing to route around;
  there is deliberately no override flag.

## Cleanups / TODO

1. **Fix `replay_e2e.py`'s defaults.** They currently point at the reference
   host's legacy shared directories, an ids file not in this repo
   (`holes/mark7z-e2e.ids.txt`), and the pre-I1 corpus label
   `math-ct-e2e-12`. Off-Zone they refuse or attest the wrong artifacts;
   the warning in Tier 0 above papers over what should be a code fix
   (run-scoped defaults, or no defaults at all — required flags).
2. **Confirm the manifest of record.** This document pins
   `holes/mark7-16.ids.txt` as *the* manifest for `math-ct-e2e-16`. If a
   canonical 16-paper list exists elsewhere (on Zone, or under another
   name), reconcile before anyone external runs Tier 2 against this one.
3. **Record the model pin.** GGUF file hash + llama.cpp build of the next
   Zone run, into the Pins table above (already flagged there).
