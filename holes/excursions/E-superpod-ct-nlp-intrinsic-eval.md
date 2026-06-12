# E-superpod-ct-nlp-intrinsic-eval — evaluate the Superpod CT NLP pipeline *on its own terms*

**Date:** 2026-06-12 · Joe + claude-6 · **Status: IDENTIFY + first findings**
**Spawned from:** the DarkTower formalization probe (`futon5a/.../E-exotype-ct-grounding`,
`futon3c/.../E-arse-ct-probe`) showed the math.CT scan is *useful downstream* — a codex swarm
grounded and formalized CT concepts from it. That is a **practical/extrinsic** evaluation. This
excursion adds the **intrinsic** one: is the neural NE/relation + scope extraction *itself* sound,
judged against its own statistical and structural expectations — not just by whether downstream
agents can use it.

## HEAD (one line)

The scan works in practice; this asks whether it works *in principle* — do the mined named
entities and scopes match the distributions M-prior-mathematics predicted (Zipf/Pareto), where do
they deviate (the over-detection "surprise"), and is the between-`$...$` scope-binding good enough?

## Why now (Joe, 2026-06-12)

- Downstream use is validated, but some papers showed **low scope counts**, and the
  `$...$` (math-mode) recognition — surfaced by yesterday's **first-proof work with fable** —
  "probably wasn't great yet."
- M-prior-mathematics built **step-1** (the CT term prior, full corpus, 2026-05-31) but its
  **step-2 posterior-vs-prior test was never run**. Joe's three questions ARE that test plus two
  extensions (scopes; `$...$`).

## Findings so far (2026-06-12, evidence-grounded)

Run over `futon6/data/ct-term-prior.json` (`unigram_df`, 9,742 docs, ~1.98M vocab) +
`futon6/data/first-proof-scope-summary.json`.

1. **The prior's MODE is contaminated — it is *not* CT.** Top terms by df:
   `document, definition, category, section, theorem, proof, theory, have, example, first, set,
   …, see, case, only, given, some`. These are English/LaTeX scaffolding, not CT named entities.
   This **contradicts M-prior-mathematics's stated premise** ("legitimate CT terms — functor,
   morphism, pretopos — sit at the mode"): stopwords sit at the mode. As a label-free NER prior
   over *raw unigrams*, the head is junk; `functor/morphism/pretopos` are far down-rank.

2. **The latex-emph over-detection is confirmed and rank-visible** (the exact failure the mission
   names): `objects` (rank 28, df 8735), `left` (44), `right` (54), `full` (124), `version` (155);
   cf. `storage/arxiv-paper-hg-gpu/candidate-new-terms.jsonl` (`objects` 404, `left` 401 — both
   dominantly from `\emph{}`).

3. **Zipf slope is shallow (s≈0.56, R²≈0.88) — BUT the test used the wrong variable.** It was run
   on **document frequency**, which *saturates* (top-20 each in ~90%+ of docs), flattening the head
   by construction. Classic Zipf is over **term frequency** (occurrence counts). So 0.56 is **not a
   verdict** — re-run on tf (and over extracted NEs, not all unigrams) before concluding anything
   about power-law shape. *(Discipline note: this is the kind of df-vs-tf slip the intrinsic eval
   must avoid; recorded so it isn't repeated.)*

4. **`$...$` scope-binding is weak on proofs.** first-proof: **74.0% floating expressions**
   (433/585 not bound to a scope) vs an **18.3% nLab baseline** — ~4× worse. Also 12 vacuous
   scopes, 340 orphans, 183 externally-bound (resolved against Mathworld 64 / Planetmath 46 /
   Wikipedia 39 / EoM 23 / …). Refinement target is concrete and measurable.

## The intrinsic-eval program (4 threads; sequence by car-of-sequence)

- **T1 — fix the Zipf/Pareto test (cleanest, ~immediate).** Re-run on **term-frequency**, over the
  **extracted entities** (stopword-filtered), not raw unigrams. Report the power-law exponent +
  goodness-of-fit. The generic-word head predicts a much cleaner fit once stopwords are stripped.
  *Assurance question:* does the NE distribution follow the law M-prior-mathematics assumed?
- **T2 — posterior-vs-prior (M-prior step-2; thread 1).** Surprise-score each mined NE against the
  prior; the over-detected junk (`objects`, `left`, "Stable Marriage everywhere") should surface as
  high-surprise. Operationalizes the mission's guardrail; quantifies junk-tail mass.
- **T3 — scope distributions (thread 2).** Distribution of scopes-per-paper across the 9,916 scan;
  is it Pareto? Are the **low-scope-count papers** extraction failures or legitimately short/sparse?
  Use `apm-proof-scope-audit.json`, `first-proof-scope-audit.json`, `diffsub-scopes.json`,
  `efe-scopes.json`, the 30-paper `showcases/ct-anatomy/` audit.
- **T4 — `$...$` refinement (thread 3).** Improve between-dollar-sign scope-binding on first-proof;
  demonstrate the **74%-floating number drop** toward the 18.3% nLab baseline. The named, falsifiable
  improvement.

## Honest caveats / discipline

- **df ≠ tf** (finding #3) — the proper Zipf variable is term frequency; do not report df-slopes as
  power-law verdicts.
- **The prior is over raw unigrams, not NEs.** A useful NER prior needs stopword filtering or to be
  built over the *extracted entity set*. The current head being stopwords is the single biggest
  intrinsic-quality flag.
- **Descriptive, not a single score.** Intrinsic eval *describes* where the extraction is sound vs
  contaminated (per-stage, per-distribution); resist collapsing to one number.
- **M-prior premise needs revisiting** (finding #1) — flag back to M-prior-mathematics: the
  "CT-terms-at-the-mode" claim doesn't hold for the raw-unigram prior.

## Success criterion

For each thread, a *concrete distributional finding* (not a vibe): T1 — a tf/NE Zipf fit with
exponent + R²; T2 — a ranked junk-tail by prior-surprise with mass %; T3 — scopes-per-paper
distribution + a verdict on the low-count papers (failure vs sparse); T4 — a measured drop in
floating-expr %. The deliverable is a per-stage **quality map** of the pipeline, plus the specific
fixes each finding implies.

## Relations
- `futon6/holes/missions/M-prior-mathematics.md` — the prior (step-1 built); this runs its step-2 +
  feeds back the contaminated-mode finding.
- `futon6/data/ct-term-prior.json` / `mission-term-prior.json` / `topic-prior-msc.json` — the priors.
- `futon6/data/first-proof-scope-summary.json` + `-audit.json` — fable's first-proof work (T4).
- `futon5a/holes/excursions/E-exotype-ct-grounding.md`, `futon3c/holes/excursions/E-arse-ct-probe.md`
  — the **practical/downstream** eval this complements.
- M-bayesian-structure-learning — the "accumulate posteriors not counters" frame M-prior instances.
- Skolem scope audit (agent memory) — vacuous/unused/free scopes; T3 overlaps (12 vacuous in first-proof).
