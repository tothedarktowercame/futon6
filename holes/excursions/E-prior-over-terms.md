# Excursion: E-prior-over-terms

**Date:** 2026-06-15
**Owner:** Claude owner (end-to-end; Agency down → built directly, gate-verified)
**Parent mission:** [M-prior-mathematics](../missions/M-prior-mathematics.md)
  — this is the concrete instance of the corpus base-rate prior applied to the
  **prose concept-term layer** of the DP pipeline (not symbols, not the NER
  Stage-5 tail).
**Siblings:** [E-discover-terms-as-dictionary-construction](E-discover-terms-as-dictionary-construction.md)
  (learn-and-promote framing), [E-iatc-model](E-iatc-model.md) (the reasoning
  layer this term layer sits beneath); DC-1 / DC-11 in
  `../dp-defect-catalogue.md`.

## 1. The tension (IDENTIFY)

The concept layer (DC-1 "terms not noticed" → DC-11 "ground terms to an
authority") now over-fires in two opposite ways, both visible on `0807.1872`:

- **OVERFED** — a qualifier or framing word is glued onto a real term:
  *"interesting abelian category"*, *"neither the category"*. The head term
  (`abelian category`) is real; the modifier is not part of it.
- **HUNGRY** — the head noun is truncated, dropping the part that makes it a
  named object: *"category of modules"* where the paper means
  *"category of modules over a ring"*.

A third, *"localization of spaces"*, is neither overfed nor hungry — it simply
isn't an established term at all (a one-off descriptive phrase).

A curated stopword list (the `_LEAD_STOP`/`_CLAUSE_CUT` whack-a-mole in
`_trim_phrase`) cannot decide these: "interesting" is a perfectly good word, and
"of modules" is a perfectly good continuation. The decision is **statistical, not
lexical** — *does this exact phrase recur across the corpus?*

## 2. The discriminator, measured (MAP / VERIFY)

Document-frequency over a 900-paper CT sample (raw substring, case-insensitive):

| phrase | papers | reading |
|---|---:|---|
| `interesting abelian category` | **1** | hapax → not a term (overfed) |
| `abelian category` | **197** | at the mode → real term |
| `localization of spaces` | **2** | barely recurs → vague, suppress |
| `category of modules` | **95** | real, but… |
| `category of modules over` | **61** | …64% of those carry the longer form → hungry |
| `triangulated category` | 112 | real |
| `left adjoint` | 399 | real |

"interesting abelian category" appears **precisely once** — Joe's prediction,
confirmed. Document-frequency separates the four cases cleanly where no trim rule
can. This is the same surprise signal that catches "Stable Marriage Problem
detected everywhere" in the parent mission — over-detection and hapax-detection
are both base-rate anomalies.

## 3. The argument (ARGUE)

```
IF:      a concept candidate should be marked only when it names a real,
         recurring mathematical object.
HOWEVER: lexical trim rules cannot tell a glued qualifier ("interesting") from a
         genuine modifier ("abelian"), nor a truncated head ("category of
         modules") from a complete one.
THEN:    validate every candidate against a corpus document-frequency prior —
         trim overfed phrases to their highest-df recurring core, extend hungry
         ones to a recurring fuller form, and drop hapax junk.
BECAUSE: a real term recurs (df at the mode); a hapax / glued phrase does not
         (df 1–2). Document-frequency IS the learn-and-promote criterion, so no
         curated list is needed and the rule transfers to any MSC class.
```

## 4. The build (INSTANTIATE)

Two pieces, mirroring how the **macro** recognizer-registry already works
(`build_recognizer_registry.py` → `ct-recognizer-registry.json`):

### `scripts/build_term_prior.py`
The prose-term analogue of the macro registry. Iterates a corpus of golden
papers, emits content-bounded 1–4-grams (first & last word non-stop), counts
**document-frequency** (how many papers contain each), prunes to `df ≥ K`, writes
`data/term-prior-<msc>.json` = `{_meta, df:{term: papers}}`.

**MSC-repeatable by construction** — nothing CT-specific is hardcoded:
```
build_term_prior.py --golden-dir <DIR> --msc <NAME> --out data/term-prior-<NAME>.json
```
The superpod blast re-points `--golden-dir`/`--out` per MSC class; the concept
layer selects the active index via `DP_TERM_PRIOR` (default `term-prior-ct.json`).

### `dp_enrich._prior_normalize`
Runs over each concept candidate before de-nesting:
- **overfed/hapax** (heuristic candidates): longest contiguous **head-anchored**
  subphrase with `df ≥ floor`; none → drop. `interesting abelian category` →
  `abelian category`; pure hapax → gone.
- **hungry**: extend with following words while the longer form recurs
  (`df ≥ floor` and `≥ 0.4 × base`) and ends on a content word (never on a
  preposition — guarded by `_BAD_TAIL`).
- **authoritative** candidates (author `\emph{}` = DC-2, lexicon hit = DC-11):
  **extend-only, never trimmed or dropped** — so C-TERM-COVERAGE and grounding
  cannot regress.
- **no index present** → pass is a NO-OP (graceful degradation; gates hold on a
  fresh checkout).

Floor is `DP_TERM_PRIOR_FLOOR` (default 4: recurs in ≥4 papers).

## 5. Learn-and-promote (the phase this opens)

A candidate that is **high-df but not in the lexicon** is a *promotion
candidate* — a real term the curated lexicon is missing. The df index is exactly
the signal Joe asked for ("learn new terms as we go and promote them"): the
corpus, not a hand list, decides termhood. Wiring promotion candidates back into
the lexicon (and grounding them as `corpus-df`) is the natural next step.

## 6. Known limits / follow-ups

- The shipped CT index is built at `MAX_N=4` from a 4000-paper sample. Hungry
  extension to forms longer than 4 words (e.g. the full *"category of modules
  over a ring"*) needs a rebuild at larger `MAX_N`; the logic already handles it,
  it just needs the bigger index. Cheap re-run (parameterized).
- df threshold and the `0.4×` extend ratio are first-cut; tune against the
  posterior-vs-prior test in the parent mission.
- The prior is CT-fit: a guardrail for the CT runner, not a universal arXiv
  prior (parent mission §2 caveat). New/emerging CT terms rare in the fit corpus
  can look anomalous — the trending-vs-hallucination hole, unchanged here.

## 7. Verification (to confirm; author ≠ reviewer pending Agency)

- [ ] gates hold: `W-NEST-SCOPE=0`, `wellformed_errors=0`, `symbol_tagged=1.0`,
      `math_coverage=1.0`, `symbol_grounded` does not fall.
- [ ] `term_grounded` / `terms_concept` do not regress; spurious-concept count
      drops on `0807.1872` (interesting abelian category, localization of spaces).
- [ ] DP_TERM_PRIOR unset / index absent → byte-identical to pre-change output.
- [ ] author-emphasis and lexicon concepts unchanged (extend-only honored).
