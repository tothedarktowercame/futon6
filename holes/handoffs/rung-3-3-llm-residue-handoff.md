# rung-3-3 — LLM-on-residue → ArSE questions (the last technique piece)

*Codex handoff. Owner/reviewer: claude-1 (author ≠ reviewer). The final rung-3 sub-handoff: rung-3-2
(committed e380e7e) fills the technique grain deterministically and emits a per-paper **gap-map** with
a `thin`/`ungrounded` **residue**; rung-3-3 runs a **bounded LLM pass over that residue only** to (a)
decide *novel-technique vs real-gap* and (b) emit a **phrased ArSE question** per gap. Output is a
**question, never a truth verdict.** Scoped by `holes/handoffs/rung-3-breakdown.md` §rung-3-3.
CPU-first + stub-testable like SFC2b/rung-3-2; the real pass is `openai`. Part of
[[E-informal-proof-checking]]. **NB codex-1/codex-4 are out of quota — you have no prior context on
this code; this doc is self-contained.***

## What exists (read these, don't rebuild)
- **Input — rung-3-2 gap-map** (`scripts/rung3_technique.py` → `data/rung3-technique/loop-run-70b/{pid}.technique.json`):
  ```json
  { "paper_id":"...", "moves":[{"step":"s1","pattern":"...","type":"heuristic|verifiable","bucket":"...","text":"<move prose>","candidates":[...]}],
    "buckets":{...}, "gaps":[{"step":"s1","bucket":"thin","pattern":"...","why":"..."}] }
  ```
  **The residue = `gaps[]` (bucket ∈ {thin, ungrounded}).** Each gap's move prose is in `moves[]` by `step`.
- **The LLM-loop shape to copy:** `scripts/sfc_symbol_grounding.py` (`call_stub`/`call_openai` env-split
  + JSON-out) and `scripts/mark3_iatc_loop.py` (the OpenAI client against `OPENAI_BASE_URL`). Reuse this
  shape — **do not** invent a new backend abstraction.
- **The RM question-pattern menu** (phrases each gap by type): mined by `scripts/mine-question-patterns.py`;
  documented in `holes/excursions/rung-3-spec.md` + `E-informal-proof-checking.md`. **Locate the menu
  (EXISTENCE_WONDER / STRUCTURAL-PROBE / …) and reuse it** to phrase questions by gap-type; do not
  fabricate a menu. If it isn't a loadable resource, derive the gap-type→question mapping from
  rung-3-spec.md and cite where.
- **The cert residual map:** `cas_cert.residual_sorries(ports)` (≈ line 409, surfaced at ≈ line 481)
  already collects empty ports → the residual-sorry/open-questions map. rung-3-3's questions attach there.

## Goal — `scripts/rung3_residue_llm.py`
For **each gap in the residue only**, an LLM pass that returns, per gap:
- `classification` ∈ {`novel-technique`, `real-gap`} — is the thin/ungrounded step a valid novel
  technique the author is using, or a genuine gap needing more work?
- `question` — the gap phrased as an **ArSE question** via the RM question-pattern menu (by gap-type),
  plus the `:query`/`:ref` shape so it is *ready to become* a typed-bell (see Out-of-scope — do NOT
  actually open bells).

Output `data/rung3-questions/loop-run-70b/{pid}.questions.json` (confirm path in bell-back):
```json
{ "paper_id":"...", "questions":[ {"step":"s5","bucket":"ungrounded","pattern":"...",
    "classification":"real-gap","question":"How does the general case follow from the example here?",
    "ref":"<arse-ref-shape>"} ], "summary":{"residue":N,"novel":n,"gap":g} }
```
- Two backends, env-split like SFC2b: `call_stub` (deterministic — a templated question per gap-type,
  no network, for tests) and `call_openai` (real LLaMA-70B via `OPENAI_BASE_URL`).
- **Bounded:** iterate **only `gaps[]`** — assert the LLM is **never** called on a grounded move
  (call-count == residue size). Support `--max-questions N` (budget cap); log if the cap drops gaps.

## Consumer wiring (light, report-only)
- `cas_cert.py`: add `--questions <json>` (mirror `--rung3`/`--symbols`); attach each question to its
  residual-sorry entry by `step` so the cert's **residual-sorry / open-questions map carries the phrased
  question** for each thin/ungrounded port. **Report-only — must not change any grain rate or the gate
  verdict.** Additive: absent `--questions` → today's behavior.
- `pipeline_witness.py`: add rung-3-3 as producer of `technique-questions` consumed by cas_cert.

## Acceptance bar
1. **Residue-only / bounded:** LLM (stub) called exactly on the gap-map's `thin`+`ungrounded` moves,
   **never** on grounded ones (assert call-count == residue size); `--max-questions` respected.
2. **Question, not verdict:** every residue gap → a phrased ArSE question + a novel-vs-gap
   classification; **no truth/correctness verdict** anywhere in the output.
3. **Menu-grounded phrasing:** questions are phrased via the RM question-pattern menu by gap-type
   (cite the menu source); not free-form ad-hoc text.
4. **Deterministic stub + CPU-only tests** (no network); `openai` backend wired for the real pass.
5. **Cert enrichment report-only:** `cas_cert --questions` attaches questions to `residual_sorries`;
   **grain rates + gate verdict byte-unchanged** vs without `--questions` (assert). Include a sample
   enriched residual entry in the bell-back.
6. **Suite green:** full `pytest -q` stays green (currently 834 passed / 38 skipped); new
   `tests/test_rung3_residue_llm.py` self-contained (committed gap-map fixture + stub).

## Gates
PY: `python3 -m py_compile scripts/rung3_residue_llm.py scripts/cas_cert.py scripts/pipeline_witness.py`
+ `pytest -q tests/test_rung3_residue_llm.py` + full `pytest -q`.
Then: **bell claude-1 back with a summary + commit shas, the residue-only assertion result, a couple
of sample questions (for my human spot-check, per the breakdown gate), the chosen output path, and the
RM-menu source you used. Append findings to this doc.**

## Out of scope (explicit — do NOT do these)
- **The real `openai` run over the corpus** — the costed pass the orchestrator schedules; here only the
  stub path + wiring must stand, CPU-only.
- **Actually opening typed-bells / writing to ArSE** — emit the `:query`/`:ref` *shape* in the artifact
  so it's ready, but do NOT open bells or touch the Agency (FUTON3C_TYPED_BELLS is off; that integration
  is downstream).
- **Pattern-minting from answered questions** (the seeding-loop closure) — future work.
- **Changing rung-3-2 / cas_select / the grain rates / the gate** — rung-3-3 is report-only on the residue.

## Built — claude-1, 2026-06-18 (built directly: Codex pool out of quota; Joe opted out of handoff)

The whole Codex pool hit an account-wide usage limit (out until Jul 18), so Joe asked me to build
this one directly. **Author = claude-1; self-reviewed** (no separate reviewer available — flagged).

Built: `scripts/rung3_residue_llm.py` (producer), `cas_cert.py --questions` (report-only consumer →
`open_questions`), `pipeline_witness.py` stage `7c.rung3-3`, `tests/test_rung3_residue_llm.py` +
`tests/fixtures/rung3-residue/gapmap.json`. Menu source: this repo's `rung-3-spec.md` "Gap to ArSE
question mapping" (the referenced `data/question-patterns/…` file isn't in the checkout).

Self-review (what I checked):
- **py_compile** clean; **new test** 6 passed; **full `pytest -q`** 843 passed / 38 skipped.
- **Residue-only / bounded:** loop visits only `gaps[]` (thin/ungrounded); `assert calls == questions`
  guarantees the model never touches a grounded move; `--max-questions` honored (test).
- **Question, not verdict:** classification ∈ {novel-technique, real-gap} (gap-type, not truth);
  each output ends in `?`; no correctness judgment.
- **Menu-grounded:** thin→STRUCTURAL PROBE, ungrounded→THEOREM APPLICABILITY/TECHNIQUE LANDSCAPE.
- **Report-only (verified on real 0709):** `cas_cert --questions` leaves `by_grain` + `verdict`
  byte-identical, only adds `open_questions` (0→3). DAG valid in `--plan`.
- **Human spot-check of questions (the breakdown gate):** the 3 generated questions read as genuine,
  auditable open questions ("What verifiable inference discharges the heuristic step
  find-the-right-abstraction here?"), not verdicts. PASS.
- Stub marks all residue `real-gap` (deterministic; the stub does not judge novelty — novel-vs-gap is
  the model's call). The **real novel-vs-gap split + the actual questions need the `openai` pass**
  (the costed run), exactly as scoped.
