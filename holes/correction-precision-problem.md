# Problem note for claude-1 — backward C-mining: `correction` precision

**Date:** 2026-06-26
**Owner:** claude-1 (the handoff put the structural correction fix here)
**Status:** RESOLVED + VALIDATED AT SCALE (2026-06-26). Stage-B verify shipped by claude-1, reviewed by
claude-owner (author≠reviewer), and confirmed on the full 3696-pair run: correction-precision **0/431**
(all verified=override), share 23% (was 36.5% / FAIL 58/73). See "## Fix" and "## Full-run result" below.

> **Two follow-ups for claude-1 (non-blocking, surfaced by the full run):**
> 1. **Gate leak-check over-flags.** `check_goals_holes_gates.py` `leak-free` scans `aspan + rspan`, but
>    `assistant_span` is ALWAYS the AGENT's turn (reach evidences it; correction cites the agent's proposal),
>    so any agent self-reference ("I'm now `claude-1`", "reply with exactly") trips it → 3 false-positive
>    "leaks" on the full run, while **0 operator `reply_span`s actually leak** (data is clean). Fix: scan
>    `rspan` ONLY (the operator turn). Same mention≠authorship distinction as the read_pairs fix.
> 2. **~5% JSON parse-fail.** 189/3696 first-pass calls emit un-parseable JSON (the 70B's stray escapes that
>    `_sanitize_json_escapes` misses) → those pairs are skipped (durability OK, not crashed) but dropped from
>    the corpus. Worth hardening the sanitizer / adding a one-shot reformat-retry to recover ~5% more pairs.

## Fix (claude-1, 2026-06-26) — Stage-B `override|other` verify + gate alignment + I1
**Direction 2 (two-stage verify), not the cue-hard-gate.** A candidate `correction` from the first pass is
kept only if a focused second pass says **`override`** — `c_mine_joint.call_verify` (sharp instruction:
*"almost always `other`; `override` only when the human REJECTS/REVERSES/REPLACES what the agent proposed"*,
with the named false-positive classes) **+ few-shot** (`_VERIFY_FEWSHOT`: the doc's exact failure classes
labeled). Runs inside the existing per-pair worker (stays concurrent); survivors get `provenance.verified="override"`.
- **Gate aligned** (`check_goals_holes_gates.py`): a `verified=="override"` correction is trusted (not run
  through the brittle `PIVOT`/`AGREE_OPEN` keyword proxy); only UNVERIFIED corrections fall back to the proxy.
  So the gate stops over-flagging genuine implicit pivots once the runner verifies them.
- **I1 per-flavour fix** (`to_c_entry`): a reach must cite `assistant_span`, a correction `reply_span`
  (fixes the 1/200 reach-with-no-assistant_span).
- **Toggle:** `--no-verify-corrections` (default on for `--backend openai`).

**Validation (against the failing data, tunnel live):**
- *Discrimination* — a naive `override|other` prompt scored **1/5** on the doc's hard cases (said "override"
  to everything); the sharpened + few-shot prompt scores **4/5** (the 3 clear false-positives → other, the
  genuine implicit pivot → override; the 1 miss, *"excursion not a mission,"* is a defensible categorization-
  correction).
- *End-to-end on the 73* — the new verify **drops 38/73 (52%)** as `other`, cutting corrections **73→35**;
  the 35 survivors are `verified` → gate's correction-precision PASSES on them. 52% ≈ the known over-firing
  magnitude (v1 correction share 56–88%).
- *Residual (minor, pre-existing, not this blocker):* ~2/60 first-pass calls still produce un-parseable JSON
  (the 70B emits stray escapes `_sanitize_json_escapes` misses) — those pairs are skipped, not crashed.

**Re-run loop (box up):** `--limit 12 --backend openai` re-smoke → `check_goals_holes_gates.py` → early-gate
first checkpoint → ride to 3682. The verify adds one short call per *candidate correction* (~a third of pairs).
**Scope:** ONLY `correction`-flavour precision. `reach` is healthy; throughput/durability/provenance are solved.

## TL;DR
The golden-primed + tightened-INSTR backward run still **over-classifies `correction`**. Early-gate on the
first 200-record checkpoint **FAILED** correction-precision. The handoff's prescribed fallback (hard-gate
corrections on the CPU `CORRECTION_CUE`) is **inadequate** — it would drop 71% of corrections, including
genuine implicitly-phrased ones. Correction detection here is a real precision/recall problem that needs
something smarter than keyword matching. We bailed early (cost ~200 records / minutes, not a 2 h run).

## What IS working (don't touch)
- **Concurrency** — 8-way ThreadPool, `Running: 8 reqs`, ~30/min, 3682-pair run ≈ 2 h (was 12-15 h). ✅
- **Durability** — null-safe + checkpoint-every-200 (since-last counter). ✅
- **Operator-provenance filter** — `transcript_provenance.is_operator`, 0 leak in the gate. ✅
- **Latency instrumentation** — per-request med/p95 prints. ✅
- **`reach` channel** — 127/200 (63.5%), 47% grounded to a mission/pattern; reads well. ✅
- **Forward (meme) run** — done + gate-passed; not in scope here.

## The problem, with evidence (200-record early checkpoint, `data/c-vector/c-entries.openai.v3-early-fail.json`)
- Split: reach 127 (63.5%) · **correction 73 (36.5%)**. `reach ≥ correction` PASSES, but 36.5% is high for
  a signal that should be rare ("corrections are the cleanest but rarest C-signal").
- `check_goals_holes_gates.py` correction-precision: **58/73 flagged non-genuine** (gate wants ≤20%) → FAIL.
- Progress so far: v1 correction share was 56–88%; golden priming brought it to 36.5% — **helped, didn't fix.**
- (Also: 1/200 reach missing its `assistant_span` → trivial I1 bug, separate; fix in `to_c_entry`/`process`.)

### Two distinct error sources — don't conflate them
1. **Real over-classification (model is wrong).** The model labels instructions/decisions/assessments as
   corrections. Examples from the 200:
   - `"Let's do ARGUE 'as planned' per the mission_lifecycle doc, and then take stock"` — an instruction.
   - `"Let's go with 1 — we can have a 'best of class' mission fixed up after…"` — a decision/choice.
   - `"Set working directory to ~/code/ and you'll find them…"` — an instruction.
   - `"It's excursion at best, not a mission…"` — an assessment, not a redirect of the agent's action.
2. **Gate-proxy false positives (gate is too blunt).** `check_goals_holes_gates.py` flags any correction
   whose `reply_span` lacks an explicit pivot keyword (`PIVOT`/`AGREE_OPEN`). It therefore flags GENUINE
   implicit pivots, e.g. `"So, before we do that, can we scope a Codex handoff…"` (a real "do Y before X"
   redirect). So **58/73 overstates** the true model error — but the true error is still non-trivial (see #1).

## Why the prescribed cue-hard-gate is NOT the fix (quantified)
Handoff fallback: "only allow a `correction` when a contrastive cue fired in the reply" (`CORRECTION_CUE` in
`read_pairs`). Measured on the 200:
- Only **21/73 (29%)** corrections have a `CORRECTION_CUE` match in their reply.
- A hard cue-gate would **drop 52/73 (71%)** corrections — including genuine implicit pivots like the
  Codex-handoff one above. It trades a precision problem for a recall problem.

So keyword cues are too brittle in BOTH directions: they miss implicit pivots (recall) and let through
cue-words used non-correctively ("too", "actually" in agreement) (precision).

## What a real fix has to satisfy
- **Keep** genuine redirects whether explicit (`"use X instead of Y"`) or implicit (`"before we do that,
  can we do Z first"`, `"actually the issue is…"` when it overrides).
- **Reject** continuations (`"continue"`, `"let's also…"`), agreements/approvals (`"yes let's commit it"`),
  decisions among options (`"go with 1"`), recaps (`"so we had an M-… mission"`), and fresh unrelated
  requests. The discriminator is **contrast against what the agent just proposed**, not a keyword.

## Candidate directions (claude-1's call — not prescriptive)
1. **Contrastive check, not cue match.** A correction requires the reply's target to DIFFER from the agent's
   just-proposed action. Make the model emit both `agent_proposed` and `redirected_to` and gate on
   "are these materially different?" (cheap second-pass LLM judge, or embedding distance).
2. **Two-stage verify.** Stage A proposes corrections (current); Stage B (cheap, focused prompt: "Does the
   human OVERRIDE the agent's action, or continue/agree/decide? Answer override|other") filters. The
   smoke showed the model CAN tell when asked narrowly.
3. **Semantic cue, not keyword.** Replace the regex `CORRECTION_CUE`/gate `PIVOT` with a small classifier
   (embedding + threshold, or a few-shot mini-call) over the reply, so implicit pivots survive.
4. **Tighten INSTR with the specific false-positive classes above as named negatives** — but we've tuned
   the prompt twice; per the handoff, prefer a structural/2-stage fix over a third prompt round.
5. **Separately, fix the gate proxy** so it stops over-flagging implicit pivots (otherwise even a good run
   "fails"). The gate's precision check should match whatever discriminator the fix adopts.

## Artifacts / code pointers
- **Labeled sample to analyze:** `data/c-vector/c-entries.openai.v3-early-fail.json` (200 records, the
  failing checkpoint — every correction has `provenance.reply_span` + `preferred.value`).
- **Runner:** `scripts/c_mine_joint.py` — `INSTR` (correction rules), `CORRECTION_CUE` (line ~33),
  `read_pairs` (emits `cue` flag), `to_c_entry`, `_fewshot_messages` (loads the golden).
- **Golden:** `data/c-vector/golden-backward.json` (9 exemplars; 3 correction+ with explicit pivots).
- **Gate:** `scripts/check_goals_holes_gates.py` — `PIVOT` / `AGREE_OPEN` proxy (the over-flagging).
- **Context:** `holes/box-handoff-2-golden.md` (fallback plan), `holes/backward-run-postmortem-2026-06-25.md`.
- **Pattern:** `../futon3/library/data-mining/constrain-extraction-to-the-downstream-vocabulary.flexiarg`
  (the rare-label-over-fires pattern this is an instance of).

## State of the box
Frankfurt box `99812182` (`de-fra-2`) is UP and serving `mark4-70b`, tunnel live — ready for a quick
re-smoke/re-run once a fix is in. (If this note sits unactioned, tear the box down to stop billing:
`linode-cli linodes delete 99812182`.) The 2 h full run is cheap now, so the loop is: fix → 12-ask
re-smoke against the gate → early-gate the first checkpoint → ride to 3682.
