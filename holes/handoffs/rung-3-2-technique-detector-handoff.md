# rung-3-2 — the technique-coverage detector (deterministic core)

*Codex handoff. Owner/reviewer: claude-1 (author ≠ reviewer). Builds the **technique grain's
substance**: today CAS-CERT's technique grain only un-N/As (via seam-6 feeding cas_select's raw
declared/thin sorries); rung-3-2 is the dedicated **deterministic technique detector** that fills it
with real buckets. Scoped by `holes/handoffs/rung-3-breakdown.md` §INSTANTIATE/rung-3-2. Part of
[[E-informal-proof-checking]]. Deterministic, **no LLM** (rung-3-3 is the LLM-on-residue follow-on).*

## Context — why this, now

- `rung-3-1` (residue spike, `scripts/rung3_residue_spike.py`, committed c70d157) is **built+reviewed**:
  it measured the deterministic residue (~73% deterministic / 27.3% LLM) and the heuristic-vs-verifiable
  pattern typing on the CAS-0 worked proofs. **rung-3-2 promotes that spike into a real per-paper
  detector.**
- After seam-6, `cas_cert.technique_ports(paper_id, cas_select)` derives technique ports straight from
  cas_select's `sorry` list (declared→filled, thin→empty) — a **placeholder**. rung-3-2 replaces that
  with a sharper, bucketed classification fed by its own gap-map artifact.

## The detector (verb-twin of R2d)

rung-3 grounds the **verbs** (is each reasoning *move* a recognized technique, or a gap?) as R2d grounds
the **nouns**. Per move, fit the best-matching menu pattern (**reuse `cas_select.retrieve`/`verify` —
do NOT reimplement retrieval**) and classify into the breakdown's buckets:
- **grounded-by-pattern** — move matches a *verifiable* technique pattern → **filled**.
- **grounded-by-citation** — move is justified by a cited result (warrant resolves to a reference) → **filled**.
- **thin** — move matches only a *heuristic* leaf where a verifiable step is required (an undeclared
  unfilled sorry presented as filled) → **empty** (the detection target).
- **ungrounded** — no matching technique at all → **empty** (worst gap).
- **conjecture** — an *author-declared* acknowledged-unfilled gap → **credit it** (first-class output,
  NOT a failure; a corpus open-problem signal).

**heuristic vs verifiable typing:** reuse rung-3-1's typing (it already types the patterns — see
`rung3_residue_spike.py` / the rung-3-1 spec). If the typing lives in `patterns-index.tsv` (the `truth`
column) or rung-3-1's output, read it from there; do not re-author it. A cascade chains heuristics but
must bottom out in verifiable leaves — "thin" = bottoms out at a heuristic where verifiable is required.

## Contracts

**INPUT:** a paper's cas_select output (topology + sorry + matches — the same JSON cas_cert's
`--cas-select` consumes), plus the pattern menu (39 `.flexiarg` + the typing). Reuse cas_select's
loaders/retrieval; don't duplicate.

**OUTPUT — per-paper technique gap-map** `data/rung3-technique/loop-run-70b/{pid}.technique.json`
(confirm path in bell-back):
```json
{ "paper_id": "...",
  "moves": [ {"step":"s3","pattern":"reduce-to-known-result","type":"verifiable",
              "bucket":"grounded-by-pattern"}, ... ],
  "buckets": {"grounded-by-pattern":N,"grounded-by-citation":N,"thin":N,"ungrounded":N,"conjecture":N},
  "gaps": [ {"step":"s5","bucket":"thin","why":"..."} ] }
```

## Consumer wiring (the seam — mirror `--cas-select`/`--symbols`)
- `cas_cert.py`: add `--rung3 <gapmap.json>`. In `technique_ports`, **when `--rung3` is provided, build
  the technique ports from the gap-map** (grounded-by-* → filled; thin/ungrounded → empty with the
  bucket in `evidence`; conjecture → empty + a `credited:true`/`kind:"conjecture"` marker so it reads as
  an acknowledged gap, **never `miswired`** — technique stays report-only, never FAILs the gate).
  **Supersede** the cas_select-derived technique ports when `--rung3` is present; fall back to the
  current cas_select path when absent; fall back to `na` when neither. Additive — existing certs
  unchanged unless `--rung3` is supplied.
- `pipeline_witness.py`: add rung-3-2 as the producer of the `technique-gap-map` consumed by cas_cert
  (parallel to how cas_select/proof-steps and sfc_ground/symbol-grounding are modeled).

## Acceptance bar
1. **Reproduces rung-3-1 ground truth:** on the 4 CAS-0 worked proofs, rung-3-2's per-move buckets
   **equal** rung-3-1's hand-classification (the breakdown's stated acceptance). Deterministic.
2. **No LLM** in this stage (retrieve+classify only; rung-3-3 owns the residue LLM pass). The
   producer + tests are CPU-only, no network.
3. **Conjecture credited, not flagged:** an author-declared gap lands in the `conjecture` bucket and
   its technique port is marked credited (not an empty "thin/ungrounded" gap, not miswired).
4. **Grain fills with substance:** `cas_cert --rung3 <gapmap>` on a worked proof shows the technique
   grain populated with filled+empty per the buckets (not all-empty as the cas_select placeholder
   gives). Include before/after `by_grain.technique` in the bell-back.
5. **Never miswired / gate-safe:** no technique port is `miswired`; the gate verdict is unchanged by
   technique content (report-only). Additive when `--rung3` absent.
6. **Suite green:** full `pytest -q` stays green (currently 829 passed / 38 skipped); new
   `tests/test_rung3_technique.py` self-contained (use the committed worked-proof fixtures + a small
   committed gap-map for the consumer test).

## Gates
PY/BB: `python3 -m py_compile scripts/<rung3-2 script>.py scripts/cas_cert.py scripts/pipeline_witness.py`
(+ clj-kondo/check-parens if any `.bb`) + `pytest -q tests/test_rung3_technique.py` + full `pytest -q`.
Then: **bell claude-1 back with a summary + commit shas, before/after `by_grain.technique`, the chosen
output path, and the rung-3-1-reproduction numbers. Append findings to this doc.**

## Out of scope (explicit)
- **rung-3-3 (the LLM-on-residue + ArSE questions)** — separate follow-on; rung-3-2 is deterministic only.
- **Changing cas_select's retrieval / the pattern pool** — reuse them; don't re-tune.
- **symbol / concept / proof grains** — untouched.
- **Making technique affect the gate verdict** — report-only, never `miswired`.

## Review — claude-1, 2026-06-18 · REVIEWED PASS (salvaged from quota-failed job)

codex-1's job exited on a **usage limit** (out until Jul 18) — but it had finished the work before
the cutoff; I salvaged the complete, uncommitted deliverable from the tree. Independently verified:
- **py_compile** clean (`rung3_technique.py`, `cas_cert.py`, `pipeline_witness.py`). ✓
- **`tests/test_rung3_technique.py`** → 4 passed; **full `pytest -q`** → 834 passed / 38 skipped. ✓
- **Attribution clean:** `cas_cert.py`/`pipeline_witness.py` diffs are rung-3-2-only (the `symbol`
  references are the already-committed 211fcf2 signatures, no symbol-extraction bleed). ✓
- **§4 grain fill (end-to-end):** `rung3_technique --steps 0709.0248` → buckets
  `{grounded-by-pattern:3, thin:2, ungrounded:1}`; `cas_cert --rung3` → technique grain
  `{filled:3, empty:3, na:false, rate:0.5}`, never `miswired`. **Notably better than the cas_select
  placeholder's all-empty** — rung-3-2 grounds 3 moves *deterministically* (no LLM). ✓
- Note for rung-3-3: grounded-by-pattern here is deterministic (hotword-fit); the test pins it to
  rung-3-1's hand-classification on the worked proofs, so it's calibrated there; on arXiv it's a
  deterministic estimate the LLM-on-residue pass (rung-3-3) refines. Accepted.
