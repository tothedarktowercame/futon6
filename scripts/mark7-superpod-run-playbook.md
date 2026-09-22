# mark7 — full math.CT Superpod run playbook (20-hour window)

**Drop-in ready for whenever Rob books a 20h Superpod slot.** One run, full math.CT, every
lesson from the mark5/mark6 work baked in, instrumented as an **accretion sweep** so the
window yields the *whole* improve-as-we-run curve for every tier — not just endpoints.

---

## 1. The shape of it

- **Corpus:** **primary** math.CT — `holes/math-ct-full.ids.txt` (all **4,616** papers with
  `primary_category = math.CT`; Rob has all of arXiv math on the Superpod, so we're not limited
  to dev-staged eprints). Cross-listed math.CT papers (primary in math.AT/RT/QA/AG/…) are
  available to *enrich the S2 substrate* without being run subjects — add if we want the broad corpus.
- **Order = citation-ranked** (`math-ct-full.ids.txt`, most-cited-first; 2,700 have inbound
  citations, backbone `math__0608040` @767). This makes the sweep a **backbone-saturation**
  curve (how fast do the top-N most-cited papers cover math.CT's reasoning) and front-loads the
  highest-value papers so an incomplete window still gets the important ones. **Chronological**
  alternative (`math-ct-chrono.ids.txt`) gives the historical-accretion curve — running both and
  comparing is itself a result.
- **Single host:** everything (S1–S12) runs on the Superpod after one STAGE rsync. No
  dev/box split (that was a data-staging gap, not a topology — mark6 lesson).
- **Accretion sweep:** process in chronological order; checkpoint every metric at log-spaced
  n (1/10/100/1k/…/full). At ~27k all-proofs (4,510 papers × ~6), full completion in 20h is
  *plausible* with 8-GPU batching — but the design doesn't depend on it: whatever the window
  reaches, the checkpoints give rising curves. *Coverage is a bonus; the curve is the product.*

## 2. Lessons baked in (audit — all wired)

| lesson | where |
|---|---|
| single-host STAGE — **only** ~68 MB substrate + futon3 (eprints already on the Superpod; **dereference symlinks**) | stepper STAGE step |
| RETRIEVE all outputs before teardown (mark6 lost CLeans/B) | stepper RETRIEVE step |
| whole-paper mining, all-proofs (not 1/paper) | S3 `--all-proofs` |
| macro DERIVED from methods (not 70B-tagged) | `clean_box_typing` |
| per-graph isolation (illegal EDN doesn't sink the batch) | S3/S5 loops |
| SFC parses Π/Σ/λ binders | `sfc_def_structure.bb` |
| structure embedding widened (z-norm + method-bigrams; sim 0.74→0.01) | `clean_structure_embed` |
| comprehension scoped to run-corpus (floor→slope) | `clean_comprehension --substrate-papers` |
| 3 normalization tiers + reground + whole-paper CLean | S10–S11 |
| completeness ledger (S2 corpus-fresh, no cross-corpus reuse) | stepper ledger |
| **units listed in the S3 prompt, ids cut from their text** (contract `mark7-v4`) | `candidate_spans.unit_id`, `mark3_iatc_loop.render_units` |
| **quote-vs-gloss measured per run** (`quote-agreement.json`) | S3 tail |
| **S1b strategies**: bindings + defined terms per window, with the rule that chose each | `markup_strategies`, fed through both candidate extractors |
| **S4 regions carved from S1's environments** (author macros included) | `expository_region_extract(..., marks)` |
| **S4 cap spends exposition first**, in-proof prose only if room is left | `select_even` |
| **S4 cap scales per paper** (`FUTON6_EXPOSITORY_CAP_PER_PAPER=scaled`) | `run_manifest.scaled_cap` |

## 2a. What changed after the 20260921 run (read before reusing artifacts)

The last run's own output showed three things worth fixing before the next window:

- **S3 cited units it had never seen.** The prompt listed no units, while the schema
  accepted ids `s1..sn` built at prompt time, so 87% of nodes cited unit `s_i` as node
  `i`, and of the nodes whose gloss could be checked only 14% quoted the clause the
  gloss described. Every gate passed throughout. Units are now listed in the prompt
  under ids cut from their own text (`L350-b4ee`), and the rate is measured into
  `quote-agreement.json` at S3.
- **Candidates must be re-cut.** Contract `mark7-v4` refuses candidates whose units
  carry no ids rather than falling back to positional ones; re-run
  `mark3_extract_candidates.py`. Reusing a 20260921 candidate directory will stop the
  run and say so.
- **One cap for every paper.** 30 regions was the budget for a six-page note and for
  a 40,000-line book alike: 0806.1324 carves 209 regions and S4 read 30, while
  0708.2185 carves 27 and lost nothing. Set
  `FUTON6_EXPOSITORY_CAP_PER_PAPER=scaled` and each paper gets
  `round(6*sqrt(regions))`, clamped to [12, 120]. Over the 12-paper run that reads
  674 regions of 1,370 where the fixed cap read 354; a short paper is read whole and
  no single paper can take more than 120 calls. The rule, and the cap each paper got,
  are recorded in the manifest and in the S4 selection accounting.
- **S4 was reading almost nothing.** Regions came from a fixed list of `\begin{...}`
  names, so a paper whose environments are author macros (`\df`, `\prf`) yielded one
  region: 0705.0102 gave 33 of 690 body lines. Carving from S1's environments gives 71
  regions, 211 lines. With a cap, exposition is selected before prose inside proofs.

## 3. The run (turnkey)

```bash
# S0 + STAGE (from dev) — provision the alloc, serve LLaMA TP=8, rsync substrate + eprints
futon6/.venv/bin/python scripts/linode_stepper.py --plan --profile superpod   # review
# ... provision, then rsync -L ONLY the ~68MB substrate + futon3 patterns + holes/math-ct-full.ids.txt
#     (NO eprint download — Rob has all of arXiv math on the Superpod already) ...

# S1..S12 ON THE HOST (set IDS=holes/math-ct-full.ids.txt, CORPUS=math-ct-full@<date>):
#   -u: unbuffered, or the log looks stalled while healthy (E-superpod-hardening H6)
#   --from S1 --reuse S0 STAGE: boot steps never ledger-record; without --reuse the
#   ledger BLOCKS S1 (H2). --ids is load-bearing (below).
RUN_ID=mark7  CORPUS=math-ct-full  \
  futon6/.venv/bin/python -u scripts/linode_stepper.py --run --profile superpod \
    --from S1 --reuse S0 STAGE --ids holes/math-ct-full.ids.txt \
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

- **S3 IATC**, **S4 expository**, and **S7 box-typing** dominate cost. S4 is
  uncapped by default; one paper had 466 regions. Setting
  `FUTON6_EXPOSITORY_CAP_PER_PAPER=N` before the run pins a per-paper cap in the
  manifest: regions are chosen at even spacing in source order, and every
  unselected region is accounted as `deferred`. A capped run qualifies only that
  declared scope; deferred regions are not accepted work.
- **Budget the proof count from measurement, not the old ~6/paper guess** (§7 H4):
  the top-100 most-cited yielded 1,525 all-proofs candidates from 91 contributing
  papers (~15.3/contributing paper). If the head rate held corpus-wide that's
  4,616 × 0.91 × 15.3 ≈ **64k proofs (2.4× the ~27k planned)**; the tail likely
  yields fewer, so the true count sits between 27k and 64k. Plan the window
  arithmetic against the upper half of that range and let S12's checkpoints
  carry a non-completion gracefully.
- CPU stages (S1/S2/S5/S6/S8–S12) are cheap and parallel.
- If batching gives ~20× single-stream, the full corpus is plausible in the window; if not,
  the sweep simply checkpoints a chronological prefix. **Either outcome is a usable result.**
- Hard rule: **RETRIEVE before teardown.** A pulled prefix > a lost full run.

## 5. RETRIEVE manifest (don't lose the EDN)

All outputs are under `--run-dir`, described by `run-manifest.json`.
Package the completed prefix, copy the archive to durable storage, and verify
that retrieved copy before releasing the allocation:

```bash
python3 scripts/retrieve_run.py pack --run-dir data/runs/mark7 --output /scratch/mark7.tgz
# Transfer /scratch/mark7.tgz to durable storage, then on the receiving host:
python3 scripts/retrieve_run.py verify /durable/mark7.tgz --extract-to /durable/mark7
```

The default prefix is S12. Use `--through S<n>` only for an explicitly partial
run whose S1–S<n> stages passed. Missing outputs, checksum mismatches, and replay
warnings/failures prevent successful verification. See the
[manifest/resume/retrieval guide](../docs/mark7-run-manifest.md). A partial
archive is evidence of that prefix, not acceptance of a complete build.

## 6. Learning goals (what the run answers)

The run is designed to produce, in one window, the curves that turn assertions into evidence:

1. **The accretion curves** (S12) — does each tier's metric *rise then converge* with corpus
   size? concept-coverage, reference-resolution, proof-move grounding, expository-move
   recognition, structural-compression ratio, comprehension (now run-scoped → should finally
   *slope*, not floor). With **citation-ranked order** this is a **backbone-saturation** curve:
   how few of the most-cited papers already cover most of math.CT's reasoning (the Pareto/
   diminishing-returns shape) — and how that differs from the chronological-accretion curve.
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

## 7. Pre-flight findings from the Zone CPU probe (2026-08-05) — fix before the window

A quality-probe run of this exact playbook on a CPU-only 256 GB box (GLM-4.5-Air
via llama.cpp's OpenAI endpoint, top-100 citation-ranked papers, run-id
`mark7z`) surfaced three things Rob's window would otherwise hit or want:

1. **S2 now checks substrate/corpus compatibility.** The runner executes
   `warp_substrate_check.py --ids <frozen-corpus>` followed by
   `coverage_inline.py --concepts data/warp/concept-usage.json --field paper_concepts`.
   It is not a stub and cannot be reused or manually marked done. A substrate
   mismatch requires repair before continuing; the historical Zone workaround
   is not an accepted invocation.
2. **Hardcoded LLM timeouts assume GPU throughput.** `mark3_iatc_loop.py`,
   `mark3_expository_loop.py` (300 s) and `clean_box_typing.py` (120 s) now
   read `FUTON6_LLM_TIMEOUT` (defaults unchanged). Slow endpoints need it;
   the Superpod won't, but batch congestion might.
3. **Serving configuration is shared.** Set `OPENAI_BASE_URL`, `MODEL`, and
   `FUTON6_PYTHON_CMD` as described in the
   [host configuration guide](../docs/mark7-host-configuration.md). The runner
   and S3 wrapper use the same endpoint and interpreter; conformance must pass.
4. **Census fact:** top-100 citation-ranked papers yield **1,525 all-proofs
   candidates from 91/100 papers** (~15 extractable proofs/paper among the
   most-cited — 2.5× the ~6/paper the 20 h budget in §4 assumed; re-check the
   window arithmetic for the full 4,616).
