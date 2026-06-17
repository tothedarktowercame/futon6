# WARP-ORCH-1 Liveness Audit

Audit date: 2026-06-17
Scope: read-only liveness audit for the WARP / concept S0-S6 spine plus
`build_term_prior`, `sfc_concept_coverage`, and the active
`data/warp/concept-index.json`.

No `data/warp/` rebuild was run. Smoke checks used `.venv/bin/python` and wrote
only to `/tmp/futon6-warp-orch1` or in-memory temporary fixtures.

## Summary

- All audited Python scripts compile with `.venv/bin/python -m py_compile`.
- Current WARP artifacts are partly live and partly stale by mtime:
  `concordance.json`, `def-snippets.json`, and `concept-graph.json` are stale
  against their declared inputs.
- `mark3_thread_tapestry.py` has no named WARP output today; its default output
  is `tmp/mark3-threads/ct-threads.json`, which is absent.
- `data/warp/concept-index.json` is canonical SFC-D3 output from
  `scripts/sfc_concept_index.py`, not a stray. The file is currently modified in
  the working tree and has mtime `2026-06-17 19:52:35 +0100`; guard it in
  WARP-ORCH-2.
- Current `sfc_concept_coverage.py` over the live inputs reports
  top-100 `100/100 = 100.0%`, top-500 `499/500 = 99.8%`.

## Liveness Table

| Stage | Script | Compiles | Smoke run | Declared output | Present | Freshness | Rows / keys |
|---|---|---:|---|---|---:|---|---:|
| S1a | `warp_concordance.py` | yes | PASS: `--limit 1`, empty DP/anatomy dirs, temp out | `data/warp/concordance.json` | yes | STALE: newest golden `2026-06-15 19:43:45` > output `2026-06-13 20:19:53` | `terms=173109` |
| S1b | `warp_bib.py` | yes | PASS: `--paper 0704.0502`, temp out-dir | `data/warp/bib-index.json`, `data/warp/bib/` | yes | fresh: eprints newest `2026-02-20 02:17:21` < output `2026-06-13 16:13:27` | `papers=9742`, `bibitems=256030` |
| S1c | `warp_citations.py` | yes | PASS: `--limit 1 --identity-limit 1 --no-write` | `data/warp/citations.json` | yes | fresh vs eprints + bib index/dir | `edges=30426`, `cited_by=5512` |
| S2 | `warp_defined_pass.py` | yes | PASS: `--probe 0704.0502` | `data/warp/defined-index.json` | yes | fresh: eprints newest `2026-02-20 02:17:21` < output `2026-06-14 11:08:28` | `concept_to_papers=179990` |
| S3 | `warp_hitlist.py` | yes | PASS: tiny temp main with current-shape inputs | `data/warp/hitlist.json` | yes | fresh vs concordance + defined-index | `hitlist=3802`, `frontier=1` |
| S4a | `warp_def_snippets.py` | yes | PASS: helper smoke; no CLI dry-run, main hard-codes live hitlist | `data/warp/def-snippets.json` | yes | STALE: hitlist `2026-06-14 12:02:18` > output `2026-06-14 11:39:01` | `snippets=972`, `papers_scanned=9742` |
| S4b | `warp_concept_usage.py` | yes | PASS: tiny temp main with patched `W`/`EPRINTS` | `data/warp/concept-usage.json` | yes | fresh vs hitlist + eprints | `paper_concepts=9737`, `papers_scanned=9742` |
| S4c | `warp_concept_embed.py` | yes | PASS: helper smoke; tiny two-concept main is below 2D layout dimensionality | `data/warp/concept-embed.npy`, `data/warp/concept-carpet-pos.json` | yes | fresh vs hitlist + def-snippets + graph | `positions=3802`, `npy_bytes=730112` |
| S5 | `warp_concept_graph.py` | yes | PASS: tiny temp main with current-shape inputs | `data/warp/concept-graph.json` | yes | STALE: hitlist `2026-06-14 12:02:18` > output `2026-06-14 11:39:02` | `n_nodes=1000`, `n_edges=5499`, `authority=120` |
| S6a | `mark3_thread_tapestry.py` | yes | PASS: `--self-test` | default `tmp/mark3-threads/ct-threads.json`; no WARP artifact | no | missing | self-test: `concepts_with_threads=1`, `n_papers=3` |
| S6b | `build_concept_encyclopedia.py` | yes | PASS: `--n 1 --out /tmp/...` | `data/concept-encyclopedia-ct.json`, `data/concept-encyclopedia/ct/` | yes | fresh vs term-prior + background + snippets + graph + defined-index | `entries=200` |
| consumer | `build_term_prior.py` | yes | PASS: `--max-papers 1 --out /tmp/...` | `data/term-prior-ct.json` | yes | STALE by whole golden-dir mtime: newest golden `2026-06-15 19:43:45` > output `2026-06-15 18:24:43` | `terms=2459715` |
| consumer | `sfc_concept_coverage.py` | yes | PASS: temp report output | `holes/excursions/sfc-concept-coverage.md` | yes | fresh vs current WARP/SFC inputs | top-100 `100.0%`, top-500 `99.8%` |
| SFC-D3 | `sfc_concept_index.py` | yes | PASS: `--concept natural transformation --no-write` | `data/warp/concept-index.json` | yes | fresh vs usage + snippets + defined + encyclopedia; active working-tree file | `concepts=3500`, `genuine=3107`, `defined=3190` |

## Confirmed I/O DAG

The authoritative source-level DAG is:

```text
S0 corpus inputs
  eprints: /home/joe/code/storage/futon6/data/arxiv-math-ct-eprints
  anatomy: /home/joe/code/storage/futon6/data/ct-anatomy-v0
  golden: data/showcases/ct-anatomy/golden
  background: data/background-corpus-index.json

S1a warp_concordance.py
  inputs: eprints, anatomy, golden
  outputs: data/warp/concordance.json

S1b warp_bib.py
  inputs: eprints
  outputs: data/warp/bib-index.json, data/warp/bib/*.json

S1c warp_citations.py
  inputs: eprints, data/warp/bib-index.json, data/warp/bib/
  outputs: data/warp/citations.json

S2 warp_defined_pass.py
  inputs: eprints
  outputs: data/warp/defined-index.json

S3 warp_hitlist.py
  inputs: data/warp/concordance.json, data/warp/defined-index.json
  outputs: data/warp/hitlist.json

S4a warp_def_snippets.py
  inputs: data/warp/hitlist.json, eprints
  outputs: data/warp/def-snippets.json

S4b warp_concept_usage.py
  inputs: data/warp/hitlist.json, eprints
  outputs: data/warp/concept-usage.json

S5 warp_concept_graph.py
  inputs: data/warp/hitlist.json, data/warp/def-snippets.json
  outputs: data/warp/concept-graph.json

S4c warp_concept_embed.py
  inputs: data/warp/hitlist.json, data/warp/def-snippets.json,
          data/warp/concept-graph.json
  outputs: data/warp/concept-embed.npy, data/warp/concept-carpet-pos.json

S6a mark3_thread_tapestry.py
  inputs: data/showcases/ct-anatomy/golden,
          data/concept-encyclopedia/ct,
          data/warp/cite-resolution/
  current output: tmp/mark3-threads/ct-threads.json (missing)
  needed WARP-ORCH-3 output: named data/warp/concept-phylogeny.* artifact

S6b build_concept_encyclopedia.py
  inputs: data/term-prior-ct.json,
          data/background-corpus-index.json,
          data/warp/def-snippets.json,
          data/warp/concept-graph.json,
          data/warp/defined-index.json
  outputs: data/concept-encyclopedia-ct.json,
           data/concept-encyclopedia/ct/*.edn

consumer build_term_prior.py
  inputs: data/showcases/ct-anatomy/golden
  outputs: data/term-prior-ct.json

consumer sfc_concept_coverage.py
  inputs: data/warp/concept-usage.json,
          data/warp/def-snippets.json,
          data/warp/defined-index.json,
          data/concept-encyclopedia-ct.json,
          data/warp/concept-graph.json
  outputs: holes/excursions/sfc-concept-coverage.md

SFC-D3 sfc_concept_index.py
  inputs: data/warp/concept-usage.json,
          data/warp/def-snippets.json,
          data/warp/defined-index.json,
          data/concept-encyclopedia-ct.json
  outputs: data/warp/concept-index.json,
           holes/excursions/sfc-concept-index.md
```

## Material Corrections To The Breakdown MAP

1. `warp_defined_pass.py` does not consume `concordance.json`; it scans eprints.
2. `warp_def_snippets.py` consumes both `hitlist.json` and eprints, not hitlist
   alone.
3. `warp_concept_embed.py` consumes `hitlist.json`, `def-snippets.json`, and
   `concept-graph.json`; the breakdown listed only hitlist.
4. `mark3_thread_tapestry.py` does not consume `data/warp/citations.json`; it
   consumes `data/warp/cite-resolution/`, `data/showcases/ct-anatomy/golden`,
   and `data/concept-encyclopedia/ct`.
5. `build_concept_encyclopedia.py` does not consume `concept-usage.json`; it
   consumes `term-prior-ct.json`, `background-corpus-index.json`,
   `def-snippets.json`, `concept-graph.json`, and `defined-index.json`.
6. `concept-index.json` is downstream SFC-D3, not part of the S1-S6 spine. It is
   canonical and actively modified; WARP-ORCH-2 must not clobber it.

## Concept-index Attribution

`data/warp/concept-index.json` is written by `scripts/sfc_concept_index.py`.
Git history attributes the original artifact and writer to commit
`546298ed1704f330f4a180ab72133dcb6c0fc869`:

```text
546298e sfc: add concept-to-papers index
 data/warp/concept-index.json | 1 +
 scripts/sfc_concept_index.py | 243 +++++++++++++++++++++++++++++++++++++++++++
```

The artifact is canonical for SFC-D3: it inverts `concept-usage.json` into
concept -> paper lists and attaches the SFC1 genuine/defined flags. Current
working-tree status shows it modified, with mtime `2026-06-17 19:52:35 +0100`;
this audit did not rebuild or write it.

## Gate Notes

- No helper logic was added, so `pytest tests/` was not required by this
  handoff.
- No Clojure/Babashka files were touched; clj-kondo/check-parens are N/A.
- `data/warp/` was not intentionally written. Smoke outputs were under
  `/tmp/futon6-warp-orch1`.
