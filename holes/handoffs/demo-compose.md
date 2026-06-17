# Handoff — DEMO-COMPOSE (the per-paper anatomy → proof-check demo)

*Owner: claude-1 (spec + review). Priority: **demo-critical path** (Joe, 2026-06-17 —
"get a demo up ASAP, then we have something concrete to test against"). Composes only
already-built green pieces; one new assembly+render script.*

## Goal

For each demo paper, render — as **one HTML artifact** — the whole spine end-to-end,
so Joe can *test against* it (is the structure faithful? is the rung-2 verdict right?
where is it wrong?):

1. **Imported concepts** (rung −1) — the concepts the paper uses, with **defined?**
   coverage (from `sfc_concept_coverage` / `def-snippets` / `defined-index`).
2. **A worked `:structure`** (the definition layer) — pick one formula-defined concept
   the paper uses, run `sfc_def_structure.bb` on its def-snippet formula → the Clojure
   `:structure`, with `:grounding` holes shown honestly (the LLM-grounding to-do).
3. **The argument + rung-2 verdict** — the paper's IATC graph rendered (reuse
   `build_iatc_goldens` / `dp_anatomy_html`), plus the **`iatc_semcheck` rung-2 profile**:
   R2a/R2b/R2c per-check status + the **residual sorries highlighted** (orphan nodes,
   anchor flags, self-loops = "where we're least sure").

## Inputs (all exist on disk)

- IATC graphs: `data/iatc-argument-graphs/loop-run-70b/{id}.edn` (+ `.attempts/` for the
  one substance-fail `0708.2185`).
- rung-2 profile: `bb scripts/iatc_semcheck.bb <graph>` (EDN/`--edn`) — per-graph
  R2a/R2b/R2c + reasons + profile.
- `:structure`: `bb scripts/iatc_def_structure.bb -` … actually `scripts/sfc_def_structure.bb`
  on a def-snippet formula.
- concepts + coverage: `data/warp/concept-usage.json` (`paper_concepts[id]`),
  `data/warp/def-snippets.json` (`snippets`), `scripts/sfc_concept_coverage.py`.
- source text / anchors: `data/showcases/ct-anatomy/golden/fable-{id}-dp-emacs.json`.

## Build

`scripts/build_proofcheck_demo.py` (Python orchestrator; reuse `dp_anatomy_html` render
fns + the `build_iatc_goldens` window/marks pattern). For each paper in the set:
- gather its concept list + which are defined (coverage panel);
- pick one formula-defined concept (concept ∈ `paper_concepts` ∩ `snippets`); run
  `sfc_def_structure.bb` on its formula → `:structure`; on parse-failure try the next
  concept, and if none parse, fall back to the L-closure exemplar labelled "capability";
- render the IATC graph (existing engine) + a **rung-2 verdict panel** from
  `iatc_semcheck` (R2a/R2b/R2c status + residual-sorry reasons highlighted).
Output: `data/showcases/ct-anatomy/proofcheck-demo/index.html` (reuse the dp-demo CSS).

**Paper set (show the range):** `0706.1286` (clean — R2a .857/R2b pass), `0708.2067`
(orphan-node residual sorries), `0709.0248` (R2a proposition-anchor flag), `0708.2185`
(substance-fail self-loop, from `.attempts`). Parameterize so any/all 10 can be rendered.

## Acceptance

- One HTML, per-paper sections, each showing: imported-concepts+coverage, a real
  `:structure` (with `:grounding` holes visible), the argument graph, and the rung-2
  verdict with **residual sorries called out**. Screenshot-verified.
- Deterministic; reuses the built tools (no reimplementation of checks/transducer).
- The four featured papers visibly differ (clean vs orphan vs anchor-flag vs self-loop).

## Gates
PY (`pytest` for any new helper) + headless-Chrome screenshot check. Bell claude-1 back
with a summary + commit shas; append findings to this doc.

## Findings — DEMO-COMPOSE (codex-1)

Implemented `scripts/build_proofcheck_demo.py`, composing the existing green
pieces rather than reimplementing checks:

- concept coverage via `scripts/sfc_concept_coverage.py` helpers and the existing
  definition substrates;
- definition structure via `bb scripts/sfc_def_structure.bb -`;
- source-window / IATC mark projection via `build_iatc_goldens` helpers and
  `dp_anatomy_html.render_marked_source`;
- rung-2 verdicts via `bb scripts/iatc_semcheck.bb --out ...` (`--include-attempts`
  for the 0708.2185 attempt graph).

Artifacts:

- HTML: `data/showcases/ct-anatomy/proofcheck-demo/index.html`
- Screenshot: `holes/handoffs/screenshots/proofcheck-demo.png`

Featured papers rendered:

- `0706.1286`: clean baseline shape, with only anchor residual notes.
- `0708.2067`: closure failure with orphan nodes called out.
- `0709.0248`: R2a proposition-anchor residual, including the
  `extensional-category` line-1510 flag.
- `0708.2185`: `.attempts/0708.2185.attempt2.edn` self-loop substance failure.

Remaining gaps:

- The structure panel is intentionally honest about the current parser: it shows
  real `:ungrounded` holes, and some selected formula snippets are still noisy.
- The imported-concepts panel shows the first 80 paper concepts to keep the demo
  scannable; the coverage metric still uses the full paper concept set.
- `iatc_semcheck.bb` currently writes EDN with `--out`; there is no literal
  `--edn` flag, so the composer uses the green `--out` interface.

Gates passed:

- `python3 -m py_compile scripts/build_proofcheck_demo.py`
- `pytest -q tests/test_build_proofcheck_demo.py` (`3 passed`)
- `pytest -q tests/` (`775 passed, 38 skipped`)
- Headless Chrome screenshot:
  `google-chrome-stable --headless=new --disable-gpu --no-sandbox
  --window-size=1440,1800 --screenshot=holes/handoffs/screenshots/proofcheck-demo.png
  file:///home/joe/code/futon6/data/showcases/ct-anatomy/proofcheck-demo/index.html`
