# SFC2b → CAS-CERT — wire the symbol grain (the second N/A seam)

*Codex handoff. Owner/reviewer: claude-1 (author ≠ reviewer). Sibling to
`seam6-cas-segment-handoff.md`. Closes the **second** (and last) N/A grain in the CAS-CERT
certificate: `symbol` (rung SFC2b). The producer exists (`scripts/sfc_symbol_grounding.py`, built
this session); the cert's consumer is a hardcoded stub. Same "producer exists, consumer doesn't
read it" shape as seam-6. Part of [[E-informal-proof-checking]] / [[M-symbol-grounding]].*

## The gap (verified, today)

A real cert (`cas_cert.py --graph .../0709.0248.edn`) reports the symbol grain N/A:
```
symbol: {na: true, rung: SFC2b}   limiting_factor: "symbol grain N/A — SFC2b not built"
port: {grain: symbol, state: na, evidence: "SFC2b not wired into CAS-CERT yet"}
```
The cause is in `cas_cert.py`: `symbol_ports()` (lines ~94-106) takes **no input** and always
returns a single hardcoded `state="na"` port. Meanwhile `sfc_symbol_grounding.py` already *produces*
per-formula groundings. This handoff connects them — exactly as `technique_ports(paper_id,
cas_select)` already consumes the CAS-SEL output.

## What exists (don't rebuild)

`scripts/sfc_symbol_grounding.py` — SFC2b, the producer:
- `ground(formula, context, backend, model) -> dict` returns schema `sfc-symbol-grounding/v0`:
  ```json
  { "schema": "sfc-symbol-grounding/v0", "backend": "...", "structure": {...},
    "groundings": [ {"symbol": "f", "binding": "...", "evidence": "...", "status": "grounded"} ],
    "summary": {"symbols": N, "grounded": g, "undefined_in_context": u, "unsupported": s} } ```
- `status` ∈ **grounded** (binding proposed *and* its `evidence` appears **verbatim** in the
  context — `check()` enforces this), **undefined-in-context** (no binding found), **unsupported**
  (a binding was proposed but its evidence is NOT in the context — i.e. rejected/hallucinated).
- Two backends, env-selected: `call_stub` (deterministic, no network — for tests) and `call_openai`
  (real LLaMA-70B via `OPENAI_BASE_URL`). **The `check()` verbatim-evidence step is deterministic
  regardless of backend** — that's the trust anchor: an LLM-proposed binding only counts as
  `grounded` if its evidence is literally in the source text.

The per-paper context already exists on disk in the **candidate** artifact
(`data/iatc-candidates/{pid}.candidate.json`, schema `iatc-candidate/v2-enriched`):
- `source-window` (str) — the verbatim prose/tex window → **this is the `context`** for `ground()`
  and for the `check()` evidence test.
- `binder-context` (list[str]) — the symbol-introduction lines for the move.

## Goal

Make the `symbol` grain live: a per-paper symbol-grounding artifact + a real `symbol_ports()` that
consumes it, so the cert's symbol grain goes N/A → populated (filled/empty), wired through
`pipeline_witness`. Mirror the seam-6 shape and the existing `--cas-select` consumer pattern.

## Two pieces

### A. Producer-driver (per-paper) — bridges SFC2b's per-formula API to a per-paper doc
`scripts/sfc_ground_paper.py` (or a `--candidate` mode on the existing script):
- Input: a `{pid}.candidate.json`. Use `source-window` as the `context`; take the formula/symbols
  from the move (the math in `source-window` / `binder-context` — reuse SFC2b's existing
  `structure_and_holes`/`symbols_in` rather than re-extracting).
- Run `ground(...)` and aggregate to a per-paper doc:
  `data/symbol-grounding/loop-run-70b/{pid}.symbols.json` (confirm path in bell-back), shape:
  `{paper_id, groundings: [ {symbol, binding, evidence, status} ], summary: {...}}`.
- `--backend stub` (default, for tests, no LLM) and `--backend openai` (real grounding). **Like
  seam-6, the wiring + stub path must be CPU-testable and deterministic; the real grounding pass
  is LLM-backed (the cost Joe accepted for this grain).**

### B. Consumer (deterministic) — the actual seam-closer in `cas_cert.py`
- Add `--symbols <json>` (mirror the existing `--cas-select` arg + how `main` loads it).
- Rewrite `symbol_ports(symbols)` to consume the per-paper doc and emit one port **per grounded
  symbol**, mapping `status` → `state`:
  - `grounded` → **filled** (evidence verbatim-verified)
  - `undefined-in-context` → **empty**
  - `unsupported` → **empty**, with the rejected binding in `evidence` for transparency.
  - **Do NOT map anything to `miswired`** — `miswired` FAILs the gate, and a missing/hallucinated
    symbol binding is an honest *gap*, not a structural contradiction. (Flag for review if you
    think a genuine symbol-level contradiction signal exists.)
- When `--symbols` is absent, keep returning the single `na` port (so today's behavior and the
  9 existing certs are unchanged unless symbols are supplied — additive, like `--cas-select`).
- `grain_summary`/`by_grain` already computes filled/empty/rate generically — once `symbol_ports`
  emits real ports, the symbol grain's `na` flips to false and `rate` populates with no other edits.

### Wiring
- `pipeline_witness.py`: add the SFC2b producer as a stage producing `symbol-grounding` consumed by
  `cas_cert`; reflect the symbol grain in `--plan`. (It's a cert *input*, parallel to cas_select —
  model it the same way cas_select/proof-steps is modeled.)

## Acceptance bar

1. **Consumer determinism:** given a fixed `{pid}.symbols.json`, `cas_cert --symbols <f>` emits a
   symbol grain with `na: false` and a `rate` matching grounded/(grounded+undefined+unsupported);
   byte-identical on re-run.
2. **Status mapping correct:** a doc with one of each status yields filled/empty/empty respectively;
   nothing maps to `miswired`; the gate verdict is unchanged by symbol content (symbol never FAILs).
3. **Producer (stub) reproducible:** `sfc_ground_paper.py --backend stub` on
   `data/iatc-candidates/0709.0248.candidate.json` emits a valid `sfc-symbol-grounding`-shaped
   per-paper doc, deterministic, no network.
4. **Evidence-check honesty preserved:** a `grounded` port's `evidence` string actually appears in
   the candidate's `source-window` (assert it — this is the whole trust basis).
5. **N/A → populated end-to-end:** `cas_cert` on 0709.0248 with `--symbols` shows the symbol grain
   populated and the `limiting_factors` no longer lists "symbol grain N/A". Include the before/after
   `by_grain.symbol` in the bell-back.
6. **Additive / suite green:** without `--symbols`, certs are unchanged (symbol still `na`); full
   `pytest -q` stays green (currently 821 passed / 38 skipped).

## Tests (self-contained, no network, no GPU)
`tests/test_sfc_cert_wiring.py`:
- A small committed fixture `{pid}.symbols.json` with one grounded / one undefined-in-context / one
  unsupported → assert the §1/§2 port states + grain `na:false` + rate.
- `sfc_ground_paper.py --backend stub` on the committed 0709.0248 candidate → §3/§4 (schema, byte
  determinism, grounded-evidence-in-source-window).
- Keep cas_cert's existing tests green (the `na`-when-absent path).

## Gates
PY: `python3 -m py_compile scripts/cas_cert.py scripts/sfc_ground_paper.py
scripts/sfc_symbol_grounding.py scripts/pipeline_witness.py`
+ `pytest -q tests/test_sfc_cert_wiring.py` + full `pytest -q` green.
Then: **bell claude-1 back with a summary + commit shas, the before/after `by_grain.symbol`, and the
chosen output-dir path. Append findings to this doc.**

## Out of scope (explicit)
- **The real (openai) grounding run over the corpus** — that's the costed LLM pass the orchestrator
  schedules; here only the stub-backed wiring + determinism must stand.
- **Improving SFC2b's grounding quality / prompt** — the producer is fixed; this only consumes it.
- **CAS-SEL / technique grain** — handled by `seam6-cas-segment-handoff.md`; don't touch it.
- **Making symbol affect the gate verdict** — symbol is report-only (never `miswired`).

## Findings

### SFC2b CAS-CERT wiring BUILT (codex-4, 2026-06-18)

Added `scripts/sfc_ground_paper.py` as the per-paper producer driver. It reads
`data/iatc-candidates/{pid}.candidate.json`, uses `source-window` as grounding context,
derives the formula from explicit formula fields or inline math in that window, and calls the
existing `sfc_symbol_grounding.ground()` path. Default output directory:
`data/symbol-grounding/loop-run-70b`; materialized stub artifact:
`data/symbol-grounding/loop-run-70b/0709.0248.symbols.json`.

Wired `scripts/cas_cert.py --symbols <json>` as an additive consumer. Without `--symbols`, the
single `symbol` N/A port is unchanged. With symbols supplied, each grounding becomes a symbol
port: `grounded -> filled`, `undefined-in-context -> empty`, `unsupported -> empty`; no symbol
status maps to `miswired`, so symbol content remains report-only and does not alter the gate.
`scripts/pipeline_witness.py --plan` now includes `5a.sfc_ground(SFC2b)` producing
`symbol-grounding`, consumed by `8.cas_cert`.

0709.0248 before/after:
- before `by_grain.symbol`: `{"empty": 0, "filled": 0, "miswired": 0, "na": true, "rate": null, "rung": "SFC2b"}`
- after `by_grain.symbol`: `{"empty": 5, "filled": 3, "miswired": 0, "na": false, "rate": 0.375, "rung": "SFC2b"}`
- gate unchanged: `FAIL -> FAIL`; limiting factor changes from `symbol grain N/A` to report-only
  low solidity.

Validation:
- `python3 -m py_compile scripts/cas_cert.py scripts/sfc_ground_paper.py scripts/sfc_symbol_grounding.py scripts/pipeline_witness.py`
- `pytest -q tests/test_sfc_cert_wiring.py` -> `4 passed`
- `pytest -q` -> `829 passed, 38 skipped`
- `python3 scripts/pipeline_witness.py --plan` -> DAG order valid; SFC2b producer is upstream of CAS-CERT.

## Review — claude-1, 2026-06-18 · REVIEWED PASS (wiring) (commit 211fcf2)

Independently re-ran:
- **Suite:** `pytest -q` → 829 passed / 38 skipped (matches; +4 vs the 825 baseline). ✓
- **Before/after:** symbol grain `na:true` → `{filled:3, empty:5, na:false, rate:0.375}`. ✓
- **Mapping correct:** read `symbol_ports` — grounded→filled, undefined/unsupported→empty,
  unknown→empty; **no path emits `miswired`**, so the gate can never FAIL on symbol content
  (verified gate unaffected). Additive confirmed: other certs (e.g. 0705.0452) stay `na` without
  `--symbols`. ✓
- **Trust-anchor mechanism:** checked every `grounded` port's `evidence` against the candidate's
  `source-window` → 3/3 verbatim, 0 violations. The deterministic `check()` plumbs through. ✓

**ACCEPTED for what the handoff scoped (the wiring/seam).** The consumer, determinism,
never-miswired, additivity, and the verbatim-evidence mechanism are all correct.

**Quality caveat — flagged, NOT a wiring defect (extraction was out of handoff scope):** the stub's
extracted "symbols" for 0709.0248 are noise — `*`, `Ap`, `Cab`, `\\id`, `and`, `share`. The driver
runs `inline_math($…$)` over the prose window then `symbols_in` (built for clean formulas), so on
real arXiv tex it picks up prose fragments. Consequences:
1. The committed `0709.0248.symbols.json` **filled:3 / rate:0.375 is STUB NOISE**, not a real
   measurement — `grounded` under stub means "stub fabricated an evidence substring that is
   trivially in-context," so §4 under stub tests a near-tautology. The real trust test happens only
   under the `openai` backend.
2. **Even the openai run won't yield a trustworthy symbol grain until symbol extraction is
   sharpened** (filter prose tokens like `and`/`share`; require genuine math symbols). That's the
   real prerequisite — a clean follow-on (producer-internal; the cert wiring is done).

Verdict: wiring **PASS**; symbol-grain *substance* gated on (a) the openai pass and (b) sharper
symbol extraction. No fixes to the wiring required.
