# WARP-ORCH breakdown — wire the warp / tapestry layer

*Breakdown of the `WARP-ORCH` "needs breakdown" card in `holes/proofcheck-readiness.html`
(Phase A · structure-first concepts). Owner excursion: **E-structure-first-concepts** (claude-1).
Drafted by claude-loop, 2026-06-17 — DRAFT for review, not dispatched. Decomposes the
unspecified item into a liveness audit + a runner + two wiring steps, evidence-first.*

## IDENTIFY — the gap

~15 `warp_*` / concept scripts produced ~15 data artifacts (`data/warp/`, dated **Jun 13–17**)
that form a clean ~6-stage DAG from corpus → concept substrate. **But no runner orchestrates
them** — they were run by hand, once, in the right order. The card's flag: *liveness unverified*
(do the scripts still run? are the on-disk outputs stale relative to current inputs?) and
*orchestration unspecified*. The whole **concept-first foundation** (rung −1 `SFC1`, `build_term_prior`,
`build_concept_encyclopedia`, and the tapestry descent the cascade's genealogical `select` needs)
consumes this layer, so it must be **live, current, and rebuildable**, not a frozen pile.

## MAP — the dependency DAG (reconstructed from docstrings + I/O reads)

```
S0  inputs        golden marks (261 DP papers, data/showcases/ct-anatomy/golden/) + 9742 eprints (.tex)
S1  raw scans     warp_concordance  → concordance.json (100MB, Jun13)
                  warp_citations    → citations.json   (14MB,  Jun13)
                  warp_bib          → bib-index.json   (108MB, Jun13)
S2  defined       warp_defined_pass (←concordance)             → defined-index.json (12MB, Jun14)
S3  hitlist       warp_hitlist      (←concordance, defined-index)→ hitlist.json      (845KB, Jun14)
S4  spread        warp_def_snippets (←hitlist)                  → def-snippets.json  (2MB,  Jun14)
                  warp_concept_usage(←hitlist, scans 9742 eprints)→ concept-usage.json(21MB, Jun14)
                  warp_concept_embed(←hitlist)                  → concept-embed.npy + concept-carpet-pos.json
S5  graph         warp_concept_graph(←hitlist, def-snippets)    → concept-graph.json (13KB, pagerank, Jun14)
S6  higher        mark3_thread_tapestry (←golden, citations)    → per-concept phylogeny  [DESCENT]
                  build_concept_encyclopedia (←usage, snippets, graph, nLab/NNexus) → concept-encyclopedia-ct.json (226KB, Jun16)
---- overlays (inspection-only; pre-superpod card 3.3 — NOT on the concept-first spine) ----
    warp_paper_landscape (t-SNE) · warp_or_curvature · warp_salingaros · warp_greatest_hits · warp_debt_report
---- consumers (the concept-first foundation — downstream of WARP-ORCH, not part of it) ----
    build_term_prior · sfc_concept_coverage (SFC1) · (eventually) R2d concept-coverage-of-proofs
```

**Two live-state signals to reconcile (don't assume frozen):**
- `data/warp/concept-index.json` (15MB) was **written today, 19:26** — something is *actively*
  producing it (likely the SFC / concept work). The liveness audit must attribute it and the
  runner must not clobber live work.
- Everything else is Jun 13–16 — candidate-stale if any S0 input changed since.

**`mark3_thread_tapestry` is special:** its output is a *per-concept phylogeny* — "git-blame for a
concept," typed by `definition` / `cited-activation` (explicit import) / `uncited-activation` (implicit
import) / `redefinition`, woven over the citation graph. **That is exactly the citation-descent relation
the cascade's genealogical `select` requires** (a paper inherits its imports'/citations' patterns). So
WARP-ORCH is not only a structure-first prerequisite — it also produces the descent graph for Phase D
(`CAS-SEL`).

## DERIVE — the orchestration

A single deterministic runner over the DAG, **make-like**, with a manifest and an audit mode:

- **Topological run.** `warp_run` runs S1→S6 in dependency order. Default = the **spine** (S1–S6);
  `--overlays` opts into the inspection layer.
- **Make-like freshness + manifest.** Each stage declares `(inputs, output, script)`. Skip a stage
  when its output exists and is newer than all its inputs (or an input **content-hash** is unchanged).
  Write `data/warp/warp-manifest.json`: per stage `{script, inputs, input-hash, output, built-at,
  rows, status}`. Re-runs are cheap (only stale stages rebuild) and the manifest *is* the answer to
  "is the layer current?".
- **Audit mode (`--audit`, read-only).** Without rebuilding, report per stage: *script compiles/runs?
  output present? output stale vs inputs? row count?* → the liveness table the card asks for.
- **Promote the descent artifact.** `mark3_thread_tapestry`'s phylogeny becomes a named, queryable
  output of the runner (not a side script), documented as the descent relation for `CAS-SEL`.
- **Determinism.** All stages are classical scans (regex tokenize + set lookup, df inversion, PageRank)
  — same inputs → same outputs, no GPU, no agents.

## ARGUE

> **IF** the concept-first foundation (and the cascade's genealogical `select`) must run off a *current,
> rebuildable* concept substrate,
> **HOWEVER** ~15 scripts + artifacts sit un-orchestrated on disk with liveness unverified, run-by-hand
> order, and one artifact being actively rewritten,
> **THEN** first *audit liveness* (read-only), then wire a *make-like runner + manifest* over the
> confirmed DAG, promoting the tapestry phylogeny to a first-class descent artifact,
> **BECAUSE** the pieces already exist (don't rewrite them — orchestrate them), the DAG is deterministic
> and cheap, and a manifest turns "is it live?" from archaeology into a one-command answer that
> downstream (SFC1, R2d, the cascade) can depend on.

## VERIFY — acceptance for the whole breakdown

1. `warp_run --audit` emits a per-stage liveness table (live / stale-output / broken) + the **confirmed**
   I/O DAG (correcting any edge this MAP got wrong).
2. `warp_run` rebuilds the spine deterministically from S0; a no-input-change re-run is a **no-op**
   (manifest skip); the manifest records freshness for every stage.
3. `sfc_concept_coverage` (SFC1) reproduces its **100% / 98.4%** off the orchestrated outputs (proves the
   foundation consumes the runner, not hand-run stragglers).
4. The tapestry phylogeny is emitted as a named artifact and documented as the `CAS-SEL` descent relation.

## INSTANTIATE — sub-handoffs (car-of-sequence; do WARP-ORCH-1 first)

### WARP-ORCH-1 · Liveness audit (read-only) · CPU · PY
**Goal:** answer "liveness unverified." For each `warp_*` / concept script: (a) does it import/compile;
(b) smoke-run on a small slice (or dry-run) — does it still run against current inputs; (c) does its
declared output exist and is it newer than its inputs; (d) row/key counts. **Attribute the active
`concept-index.json` (who writes it, is it canonical or a stray).**
**Output:** a liveness table + the **confirmed DAG** (this MAP's edges are reconstructed from grep —
WARP-ORCH-1 makes them authoritative). Read-only; no rebuilds.
**Acceptance:** table covering all spine scripts; every S0→S6 edge confirmed or corrected.

### WARP-ORCH-2 · The runner + manifest · CPU · PY
**Depends on** WARP-ORCH-1. **Goal:** `warp_run` (topological, make-like skip, `data/warp/warp-manifest.json`),
default spine / `--overlays` opt-in / `--audit` (folds in -1). Do **not** rewrite the stage scripts —
shell out to them in order. Guard the active `concept-index.json` (no clobber of live work — coordinate
with the SFC owner).
**Acceptance:** deterministic rebuild from S0; idempotent re-run is a no-op; manifest records every stage's
freshness; `--audit` reproduces -1's table.

### WARP-ORCH-3 · Promote the tapestry descent artifact · CPU · PY
**Goal:** put `mark3_thread_tapestry` in the runner and emit its per-concept phylogeny as a **named,
queryable** artifact (e.g. `data/warp/concept-phylogeny.json` / `.edn`), documented as the **citation-descent
relation** the cascade `select` consumes (`a paper inherits its imports'/citations' patterns`). Bridges
WARP-ORCH → Phase D `CAS-SEL`.
**Acceptance:** phylogeny artifact built by `warp_run`; one worked example of "concept C reached paper P via
cited-activation from paper Q"; documented as the `CAS-SEL` descent input.

### WARP-ORCH-4 · Close the SFC join · CPU · PY
**Goal:** confirm `build_term_prior` + `build_concept_encyclopedia` + `sfc_concept_coverage` (SFC1) consume
**the runner's** outputs (not hand-run files); SFC1 coverage reproduces off `warp_run`. This makes the
concept-first foundation a single `warp_run && sfc_concept_coverage` chain.
**Acceptance:** end-to-end `warp_run` → SFC1 reproduces 100% / 98.4%; the foundation has no hand-run step.

**Gates (all):** PY (`pytest` where logic is added) + report the numbers. Coordinate with the
E-structure-first-concepts owner before WARP-ORCH-2 touches `data/warp/` (active `concept-index.json`).
