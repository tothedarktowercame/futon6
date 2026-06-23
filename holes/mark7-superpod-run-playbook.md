# mark7 — full math.CT Superpod run playbook (20-hour window)

**Drop-in ready for whenever Rob books a 20h Superpod slot.** One run, full math.CT, every
lesson from the mark5/mark6 work baked in, instrumented as an **accretion sweep** so the
window yields the *whole* improve-as-we-run curve for every tier — not just endpoints.

---

## 1. The shape of it

- **Corpus:** **primary** math.CT — `holes/math-ct-full.ids.txt` (4,510 papers whose
  `primary_category = math.CT`, of ~4,616 in the archive; ordered **chronologically** = the
  natural accretion order). A further **4,252 cross-listed** papers (primary in
  math.AT/RT/QA/AG/… but tagged math.CT) are staged and available to *enrich the S2 substrate*
  without being run subjects — add them if we want the broad corpus.
- **Single host:** everything (S1–S12) runs on the Superpod after one STAGE rsync. No
  dev/box split (that was a data-staging gap, not a topology — mark6 lesson).
- **Accretion sweep:** process in chronological order; checkpoint every metric at log-spaced
  n (1/10/100/1k/…/full). At ~27k all-proofs (4,510 papers × ~6), full completion in 20h is
  *plausible* with 8-GPU batching — but the design doesn't depend on it: whatever the window
  reaches, the checkpoints give rising curves. *Coverage is a bonus; the curve is the product.*

## 2. Lessons baked in (audit — all wired)

| lesson | where |
|---|---|
| single-host STAGE (~68 MB substrate, **dereference symlinks**) | stepper STAGE step |
| RETRIEVE all outputs before teardown (mark6 lost CLeans/B) | stepper RETRIEVE step |
| whole-paper mining, all-proofs (not 1/paper) | S3 `--all-proofs` |
| macro DERIVED from methods (not 70B-tagged) | `clean_box_typing` |
| per-graph isolation (illegal EDN doesn't sink the batch) | S3/S5 loops |
| SFC parses Π/Σ/λ binders | `sfc_def_structure.bb` |
| structure embedding widened (z-norm + method-bigrams; sim 0.74→0.01) | `clean_structure_embed` |
| comprehension scoped to run-corpus (floor→slope) | `clean_comprehension --substrate-papers` |
| 3 normalization tiers + reground + whole-paper CLean | S10–S11 |
| completeness ledger (S2 corpus-fresh, no cross-corpus reuse) | stepper ledger |

## 3. The run (turnkey)

```bash
# S0 + STAGE (from dev) — provision the alloc, serve LLaMA TP=8, rsync substrate + eprints
futon6/.venv/bin/python scripts/linode_stepper.py --plan --profile superpod   # review
# ... provision, then rsync -L the STAGE manifest (incl. holes/math-ct-full.ids.txt) to scratch ...

# S1..S12 ON THE HOST (set IDS=holes/math-ct-full.ids.txt, CORPUS=math-ct-full@<date>):
RUN_ID=mark7  CORPUS=math-ct-full  \
  futon6/.venv/bin/python scripts/linode_stepper.py --run --profile superpod \
    --run-dir data/runs/mark7 --corpus-id math-ct-full --run-id mark7
#   halts at each gate; ledger refuses any stage whose upstream didn't run for this corpus.

# RETRIEVE (from dev) BEFORE releasing the alloc — pull graphs, CLeans, object B,
#   embed+ingest, expository graphs, metrics+ledger, the lexicons, the accretion curves.
```

Stage order: `S0 · STAGE · S1 anatomy · S2 concepts · S3 IATC(all-proofs) · S4 expository ·
S5 comprehension · S6 paper-graph(B) · S7 CLean-embed · S8 export · S9 APM · S10
lexicon+reground · S11 structural+whole-paper · S12 accretion-sweep · RETRIEVE`.

## 4. 20-hour budget (the GPU stages are the pole)

Process **chronologically**; rely on vLLM batch concurrency across 8 GPUs. Rough shares:

- **S3 IATC** (~27k proofs) and **S7 box-typing** (~27k) are the cost; **cap S4 expository**
  (sample ~30 regions/paper, not all — one paper had 466) so it doesn't dominate.
- CPU stages (S1/S2/S5/S6/S8–S12) are cheap and parallel.
- If batching gives ~20× single-stream, the full corpus is plausible in the window; if not,
  the sweep simply checkpoints a chronological prefix. **Either outcome is a usable result.**
- Hard rule: **RETRIEVE before teardown.** A pulled prefix > a lost full run.

## 5. RETRIEVE manifest (don't lose the EDN)

`data/iatc-argument-graphs/mark7` (graphs) · `holes/clean-mark7` (CLeans EDN) ·
`data/iatc-paper-graphs/mark7` (object B → unblocks whole-paper canonicalization, todo #17) ·
`data/showcases/clean-mark7-demo` (embed + ingest for Rob) · `data/expository-scope-graphs/mark7` ·
`data/runs/mark7` (metrics + ledger + the harvested lexicons + accretion curves).

## 6. Learning goals (what the run answers)

The run is designed to produce, in one window, the curves that turn assertions into evidence:

1. **The accretion curves** (S12) — does each tier's metric *rise then converge* with corpus
   size? concept-coverage, reference-resolution, proof-move grounding, expository-move
   recognition, structural-compression ratio, comprehension (now run-scoped → should finally
   *slope*, not floor).
2. **Move-lexicon convergence** (S10) — how large is math.CT's inference/expository move
   vocabulary, and where does it saturate? (the corpus's own reasoning repertoire)
3. **Structural shape census** (S11) — how many canonical definition shapes does math.CT
   reduce to (compression ratio at scale)? which constructs still hit SFC coverage gaps?
4. **Whole-paper archetypes** (S11) — do papers cluster into a finite set of structural
   archetypes (the paper-level macro signature, settled in mark6)?
5. **The structure embedding for Rob** (S7, now widened) — cross-paper structural twins at
   full-corpus scale, the "this proof argues like that one" index.
6. **Anchor-confidence distribution** — across IATC/expository/SFC, how much of what we
   harvest is high-confidence vs flagged (the per-layer quality floor).

Every one of these is a *curve or census over the corpus*, so a partial 20h sweep still
teaches us where math.CT's reasoning, exposition, and definitional structure converge.
