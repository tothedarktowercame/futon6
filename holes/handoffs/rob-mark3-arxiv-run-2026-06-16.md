# Rob handoff — mark3 arXiv pipeline run (fresh future batch + eprints)

**Date:** 2026-06-16 · Joe + claude-4 · **Status: DRAFT (one eprint-wiring gap flagged below)**

**Naming note:** this "mark3" pass is the **deterministic CPU arXiv pipeline**
(`scripts/superpod-job.py`) re-run on a fresh batch to confirm recent fixes. It is
*not* the neural/IATC layer (that work — concept encyclopedia, IATC argument graphs,
embeddings, the LLM reconstruction loop — is now **mark4**, in progress separately).
This run needs no reorganization beyond pulling the updated runner.

## What Rob runs

1. **Update the runner:**
   ```bash
   cd ~/futon6 && git pull   # latest master
   ```
2. **Use a FRESH future arXiv batch** — not an already-completed early batch. Build the
   next one (storage-aware) and pull it to the superpod:
   ```bash
   ssh chicago mark2 build          # builds next batch
   ssh chicago mark2 next           # show the ready batch id
   scp chicago:~/mark2/inbox/batch-XYZ.tar.gz .   # then: ssh chicago mark2 pulled XYZ
   ```
3. **Unpack** as usual. A batch ships `batch-XYZ.jsonl` **and** an `eprints/` directory
   (LaTeX sources). Keep them together — the runner auto-detects `<batch-dir>/eprints/`.
4. **Run the pipeline:**
   ```bash
   python ~/futon6/scripts/superpod-job.py \
     --arxiv-jsonl batch-XYZ.jsonl \
     --site arxiv.math \
     --output-dir ./output/ \
     --paper-eprint-dir ./eprints/ \
     --paper-hg-eprint-dir ./eprints/ \
     --distinctor-eprint-dir ./eprints/
   ```
   **Why the explicit `--*-eprint-dir` flags** (read this): the runner's auto-detect
   currently wires the batch-local eprints to the **term-discovery** stage *only*
   (`superpod-job.py` ~6636-6648 sets `--discover-terms-eprint-dir` and nothing else).
   The **paper stages (5c/5d/Stage 6)** — the ones that drive claim/proof recovery — fall
   back to abstracts unless their eprint dirs are passed. Passing the three flags above
   makes *all* stages use the local LaTeX, which is what the validation below checks.
   (If/when the auto-detect is completed to wire all eprint dirs — see "Owner follow-up"
   — the bare command from step 4 will suffice.)

## Outputs to inspect (`./output/`)
- `pattern-tags.json` — Stage 3 LLM pattern tags
- `reverse-morphogenesis.json`
- `geometry.json`
- `paper-hypergraphs.json`
- `manifest.json` + the per-stage status summaries

## What this run validates (the specific checks)
1. **Stage 3 is no longer mostly empty on papers** — `pattern-tags.json` should carry
   real per-paper tags for the bulk of the batch, not near-empty records.
2. **Stage 6 has explicit records, not silent null coverage** — the Stage-6 section of
   the manifest/status should report explicit per-paper coverage (present/empty with a
   reason), never silently null.
3. **`geometry.json` is present** — it is produced on master (`superpod-job.py` ~5439);
   confirm the file exists and is non-trivial.
4. **Older-source papers recover claims/proofs better with local eprints** — for papers
   whose `eprints/` LaTeX is present (esp. older submissions), `paper-hypergraphs.json`
   should show richer claim/proof structure than an abstract-only run. (`manifest.json`
   records `paper_text_source: eprints|metadata` per the paper stages — confirm it says
   `eprints` for papers that have them.)

## Contract back to us
Return `output/` (or just the five artifacts above + `manifest.json`) plus the
stage-status summary. We grade against the four checks; the eprint-source field in the
manifest is the quick tell for check 4.

## Owner follow-up (claude-4, not Rob's problem)
- **Complete the eprint auto-detect** so `--arxiv-jsonl` alone wires *all* eprint dirs
  (discover-terms + distinctor + paper + paper-hg), making step-4's bare command match
  Joe's intent. Small change to the ~6636 block; pending the mark4 branch-thrash settling
  before it lands on master.
