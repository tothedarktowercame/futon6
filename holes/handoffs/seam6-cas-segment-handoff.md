# Seam-6 — the IATC-graph → proof-steps segmenter (`cas_segment.py`)

*Codex handoff. Owner/reviewer: claude-1 (author ≠ reviewer). Closes the one open seam in the
checker spine: stage 6 `cas_select` consumes `proof-steps`, but nothing produces a segmentation of
an **arXiv IATC graph** into steps — so CAS-SEL/CAS-CERT's topology branch is N/A on every arXiv
paper. This is the piece that **merges the arXiv producer track into the APM cascade track**. Part
of [[E-informal-proof-checking]].*

## The gap (verified, today)

`python3 scripts/pipeline_witness.py --plan` reports:

```
[6.cas_select] cpu gap  consumes ['proof-steps'] ✗ unmet ['proof-steps']
NOTE: no stage produces a segmentation of an arXiv IATC graph into steps.
```

cas_select today reads pre-segmented steps only from `tests/fixtures/cas-select/{pid}.steps.json`
— hand-built from the 4 APM worked examples. There is **no path from a real arXiv IATC graph to
steps**. Everything downstream (CAS-SEL retrieve/verify/assemble, the CAS-CERT topology+sorry ports)
is built and tested; it just has no arXiv input.

## Goal

Build `scripts/cas_segment.py` — given an **IATC argument graph** (`.edn`), deterministically emit a
`{pid}.steps.json` in the exact schema cas_select already consumes, so the arXiv witnesses flow
through stage 6 → 7 → 8.

**Deterministic, no LLM.** This is the Tier-0 segmenter (CPU-testable, no GPU). An LLM prose-cleanup
pass is an explicit *follow-on*, out of scope here (see below).

## Contracts (both fixed, on disk)

**INPUT — IATC graph** (`data/iatc-argument-graphs/loop-run-70b/{pid}.edn`), top-level keys
`paper/id, passage/id, source, nodes, edges, holes`:
- `nodes`: `[{id, kind, text, source:{lines:[a b]}}]` — `kind` ∈ `:claim` etc.; `text` is the math
  prose; `id` is a keyword like `:extensional-category`.
- `edges`: `[{id, kind:`​`:infer`​`, relation, premise, warrant:{kind,text}, conclusion, source:{lines}}]`
  — an **inference move**: `premise` and `conclusion` are node-ids; `warrant` is the reasoning
  (or `{:kind :missing-warrant}`).

**OUTPUT — steps doc** (the schema cas_select's `load_steps` expects — see
`tests/fixtures/cas-select/a93J05.steps.json`):
```json
{ "paper_id": "0709.0248",
  "steps": [ {"id": "s1", "text": "<one reasoning step, as math prose>"}, ... ] }
```

## Recommended approach (the contract is the hard requirement; algorithm is guidance)

1. **Parse the graph** with the structured edn loader (`r2d_concept_coverage.load_edn` — already
   used across the spine), **not** regex. Build a `{node-id -> node}` map.
2. **One step per inference edge**, in **source-line order** (tie-break on edge `id` for
   determinism). For each edge synthesize **real math prose by resolving node-ids to `node.text`** —
   e.g. `"{premise.text}; therefore {conclusion.text}"`, and append the warrant when it carries
   content (`"(because {warrant.text})"`; skip for `:missing-warrant`). This is the crucial
   difference from `rung3_residue_spike.edge_move_text`, which emits a **debug string** (`"edge :e-…
   relation :implies from premise :foo…"`) — that string is useless to the hotword retriever; we
   want the underlying mathematics so Tier-0 retrieve has lexical surface to match.
3. **Setup steps:** also emit a step for any node that is never a `conclusion` of an edge (a
   given/axiom with no derivation), text = `node.text`, placed by its own source line. This keeps
   constructions/definitions ("Let P be …") in the step list — they carry the
   `construct-auxiliary-object` / `unfold-the-definition` patterns.
4. **Re-id** the merged, source-ordered list as `s1, s2, …`.
5. **Reuse what exists:** the ordering / `:lines` extraction logic is already proven in
   `rung3_residue_spike.py` (`source_lines`, `warrant_text`); lift the *idea*, but operate on parsed
   structures, not text blocks.

### Wiring (so the seam actually closes)
- Write to a real data dir, **not** `tests/fixtures/`: suggest
  `data/cas-select-steps/loop-run-70b/{pid}.steps.json` (adjustable — confirm in the bell-back).
- Give `cas_select.py` a way to read produced steps (it's fixtures-only today): add a `--steps
  <file>` / `--steps-dir <dir>` arg alongside the existing `--fixtures`, so `select_proof` can run
  on a segmenter output. Keep the fixture path the default (don't break the 4-proof tests).
- Update `scripts/pipeline_witness.py` stage `6.cas_select`: flip `status` `gap → built`, point
  `path` at the new dir, and add `cas_segment` as the producer of `proof-steps`. After this,
  `--plan` must show stage 6 with **no `✗ unmet`**, and `--witness 0709.0248` must trace 6 → 7 → 8.

## Acceptance bar

1. **Schema + determinism:** `cas_segment.py data/iatc-argument-graphs/loop-run-70b/0709.0248.edn`
   emits a valid steps doc (loadable by `cas_select.load_steps`); re-running is **byte-identical**.
2. **Faithful prose, not debug strings:** each step's `text` contains resolved `node.text` math
   prose (assert it does **not** contain the literal substring `"relation :"` / raw node-ids).
3. **Order + coverage:** steps are source-line ordered; every inference edge is represented, plus
   the given/setup nodes; re-id'd `s1..sN` contiguously.
4. **Seam closed:** after the wiring, `pipeline_witness.py --plan` shows stage 6 `built` with no
   unmet input, and `--witness 0709.0248` reports stage-6 PASS (not GAP). Include the `--plan`
   output in the bell-back.
5. **End-to-end smoke:** `cas_select` runs on the produced 0709.0248 steps and returns a topology +
   sorry list without error (numbers are not asserted — the honest Tier-0 recall ceiling stands;
   this only proves the seam carries data). Optionally show the resulting CAS-CERT topology grain
   flips from N/A → populated for that paper.
6. **Existing tests stay green** — the 4-proof CAS-SEL fixtures and the whole suite (`pytest -q`,
   currently 821 passed / 38 skipped) are unaffected; the new `--steps` path is additive.

## Tests (self-contained, no network, no GPU)
- New `tests/test_cas_segment.py`: run the segmenter on a **committed** small graph (0709.0248 is
  already in `data/iatc-argument-graphs/loop-run-70b/`) and assert §-acceptance 1–3 (schema, byte
  determinism on a second call, source-line ordering, edge+setup coverage, prose-not-debug,
  contiguous re-id). Use the live graph file as fixed input (it's committed test data, not live
  substrate — acceptable here, unlike the live-`data/warp` coupling we just had to re-pin).

## Gates
PY: `python3 -m py_compile scripts/cas_segment.py scripts/cas_select.py scripts/pipeline_witness.py`
+ `pytest -q tests/test_cas_segment.py tests/test_cas_select.py` + full `pytest -q` stays green.
Then: **bell claude-1 back with a summary + commit shas, the `pipeline_witness --plan` output
showing the closed seam, and the chosen output-dir path. Append findings to this doc.**

## Out of scope (explicit — do NOT do these)
- **Tier-1 LLM prose cleanup** of the synthesized step text (a bounded IATC-graph→clean-prose call).
  That's the quality follow-on; the deterministic resolve-node-text version is the seam-closer and
  must stand alone, CPU-only.
- **Re-segmenting the APM proofs** — their hand-built fixtures stay authoritative.
- **CAS-SEL/CAS-CERT logic changes** — they're built; this only feeds them.

## Findings — codex-1, 2026-06-18

Implemented `scripts/cas_segment.py` as a deterministic IATC EDN graph to CAS-SEL
steps producer. It parses with `r2d_concept_coverage.load_edn`, emits one step
per inference edge plus setup steps for nodes that are not edge conclusions, sorts
by source lines, and re-ids contiguously as `s1..sN`. Edge steps resolve node ids
to node prose and skip missing-warrant placeholders, so the output is math prose
rather than `rung3_residue_spike` debug strings.

Output directory chosen and wired: `data/cas-select-steps/loop-run-70b/`.
Generated steps for the 9 committed `loop-run-70b` graphs. For `0709.0248`, the
segmenter emits 6 steps and `cas_select.py --backend stub --steps
data/cas-select-steps/loop-run-70b/0709.0248.steps.json` runs without error
(`topology=[]`, 6 thin induce rows under stub/no oracle).

Wiring:

- `cas_select.py` now accepts additive `--steps <file>` and `--steps-dir <dir>`
  inputs; the existing fixture default is unchanged.
- `pipeline_witness.py --plan` now includes `5b.cas_segment` producing
  `proof-steps`, and `6.cas_select` is `built` with no unmet inputs.
- `pipeline_witness.py --witness 0709.0248` reports `5b.cas_segment` PASS,
  `6.cas_select` PASS, and `SEAM GAPS: none`.

Gates passed:

- `python3 -m py_compile scripts/cas_segment.py scripts/cas_select.py scripts/pipeline_witness.py`
- `pytest -q tests/test_cas_segment.py tests/test_cas_select.py` (`9` passed)
- `pytest -q` (`825` passed, `38` skipped)
