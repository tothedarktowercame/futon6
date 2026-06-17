# R2d breakdown — concept-coverage of proofs (the SFC ↔ proof-checking bridge)

*Breakdown of the `R2d` "needs breakdown" card in `holes/proofcheck-readiness.html` (Phase C · the rung
ladder). The **join** between the two excursions: **E-structure-first-concepts** (the concept substrate)
and **E-informal-proof-checking** (the proof). Drafted by claude-loop, 2026-06-17 — DRAFT for review, not
dispatched. R2d's gate semantics is a genuine design call, so the breakdown leads with a spec spike.*

## IDENTIFY — the gap

rung-2 checks the reasoning *structure* — R2a anchor-faithfulness, R2b closure, R2c warrant-resolution.
But all of it **floats if the proof's concepts aren't defined**: you cannot verify reasoning over a term
that has no definition. R2d is the missing bridge — *"do the concepts a proof **uses** have definitions in
the structure-first substrate (or point to well-known concepts)?"* It is the concept-side half of the
proofcheck spine's promise: **"all terms are defined or point to well-known concepts in the literature."**
Currently **unspecified**, and it is what makes the SFC concept work *pay off* inside proof-checking.

## MAP — what exists (both sides are on disk)

```
THE PROOF'S CONCEPTS (what a proof uses)
  IATC graph nodes (:text)            data/iatc-argument-graphs/loop-run-70b/*.edn — the concept phrases the proof reasons over
  proof-region anatomy marks          golden fable-<id>-dp-emacs.json — symbol-grounded / concept marks in the proof span
  expository graph                    the informal moves (secondary)

THE SUBSTRATE (is a concept defined / known?)
  defined-index.json                  concept -> defining papers
  def-snippets.json                   concept -> real definition passages
  concept-encyclopedia-ct.json        gloss + provenance{target:"nlab-…"}  ← the "well-known" pointer
  sfc_concept_coverage.py             the corpus-wide coverage logic to REUSE (scope it to one proof)
  mark3_thread_tapestry (descent)     "imported from a cited paper" — a concept undefined locally but defined in a paper THIS proof cites
```

So both sides exist. R2d is a **scoped lookup** (one proof's concepts against the substrate), reusing
`sfc_concept_coverage`'s machinery — not new mining. The **descent** angle (tapestry) is what lets an
imported concept count as covered, and it is the *same* citation-descent the cascade's `select` uses.

## DERIVE — the check

For a proof `P`: extract its **concept set** (from the IATC nodes' `:text` + the proof-region concept marks),
then **classify** each concept against the substrate into one of:

- **DEFINED** — present in `def-snippets`/`defined-index`/encyclopedia for this MSC;
- **KNOWN** — points to a well-known concept (encyclopedia `provenance{target:"nlab-…"}` / NNexus, or a
  term-prior recurring-core) — defined *somewhere canonical*, just not re-defined here;
- **IMPORTED** — undefined locally but defined in a paper `P` **cites** (via `mark3_thread_tapestry`'s
  `cited-activation`) — covered by descent;
- **UNDEFINED** — none of the above: the proof reasons over a term with no definition anywhere reachable.

R2d emits per-proof `{coverage = (defined+known+imported)/total, by-bucket counts, undefined:[…]}`.

**Gate semantics — SETTLED (detector framing, 2026-06-17):** R2d is the **noun-side detector**, the twin of
rung-3 (terms ↔ techniques). So an undefined concept is a **flagged gap → ArSE question**, *never a hard
FAIL* — the term may be a substrate gap, not a proof defect (the SFC1 caveat: "defined" = evidence-exists).
N/A when the proof has no extractable concepts (coarse). This **removes the design block** that made R2d-1 a
FAIL-vs-report decision: R2d-1 now just confirms the concept-source + the worked classification, and R2d is
**"needs build"**, not "needs breakdown".

## ARGUE

> **IF** rung-2's structural checks are only meaningful when the proof's terms are defined,
> **HOWEVER** that concept-coverage check doesn't exist and its gate semantics (UNDEFINED = FAIL vs report)
> is a genuine design call,
> **THEN** scope `sfc_concept_coverage`'s substrate lookup to one proof's concept set, classify
> defined/known/imported/undefined, and decide the gate on worked proofs first,
> **BECAUSE** the substrate + the lookup logic already exist (this is a scoped reuse, not new mining), the
> descent (tapestry) makes "imported" a real coverage class, and the FAIL-vs-report line must be grounded in
> real proofs (an undefined term may be a substrate gap, not a proof defect) before it gates.

## VERIFY — acceptance for the whole breakdown

1. On the 9 `loop-run-70b` proofs: per-proof concept-coverage + the bucket breakdown + the undefined list,
   deterministic.
2. The gate semantics is decided **from worked examples** (≥3 proofs hand-classified) before it gates;
   N/A on edgeless/coarse; reuses `sfc_concept_coverage` (no fork).
3. The **imported** class is exercised via the tapestry descent on at least one proof whose concept is
   defined only in a cited paper.
4. Wired into `iatc_semcheck` as a rung-2 check (`check-graph`-shaped), N/A ≠ FAIL.

## INSTANTIATE — sub-handoffs (R2d-1 spec spike first; gated on the substrate being live)

> **Dependency:** R2d-2/3 need the concept substrate orchestrated (**WARP-ORCH**) and, for the import class,
> the descent artifact (**WARP-ORCH-3**). R2d-1 (spec) can run now against the on-disk substrate.

### R2d-1 · Confirm concept-source + worked classification · CPU · small spike
*(The gate is settled — flag→ArSE-question, not FAIL — so this shrank from a design decision to a
confirmation.)* **Goal:** (a) decide the **concept-source** (IATC `:text` only, or + proof-region concept
marks); (b) on ≥3 worked proofs (`0706.1286` clean, `0709.0248` the `extensional category` case, one with an
imported concept), hand-classify each concept defined/known/imported/undefined against the substrate, and
fix the **"known"** threshold (nLab provenance / term-prior recurring-core). **Deliverable:**
`holes/excursions/r2d-spec.md` — the concept-source + the buckets on the worked proofs. Light; can fold
straight into R2d-2.

### R2d-2 · Implement the scoped concept-coverage checker · CPU · PY/BB
**Depends on** R2d-1 + WARP-ORCH. **Goal:** a `check-graph`-shaped R2d (`{:check :concept-coverage :pass
:rate :reasons :per-item}`) that extracts the proof's concept set and classifies it against
`def-snippets`/`defined-index`/encyclopedia, **reusing `sfc_concept_coverage`** (do not fork). Per-proof
coverage + undefined list + N/A ≠ FAIL per the spec. **Acceptance:** the 9 proofs' coverage reproduces the
R2d-1 hand-classification; deterministic. **Gates:** PY (+ BB if `.bb`) + report the per-proof numbers.

### R2d-3 · The import/descent class · CPU
**Depends on** R2d-2 + WARP-ORCH-3 (the `concept-phylogeny` artifact). **Goal:** credit **IMPORTED** —
a concept undefined locally but defined in a paper the proof cites — via the tapestry `cited-activation`
descent. This is the same descent the cascade `select` uses, so R2d-3 and `CAS-SEL` share the artifact.
**Acceptance:** at least one proof gains coverage via an imported concept; the descent lookup is the
WARP-ORCH-3 artifact, not a re-implementation.

**Gates (all):** PY/BB + report numbers. Wire the landed checker into `iatc_semcheck` (the rung-2
aggregator) so R2d joins R2a/R2b/R2c in the per-paper profile. Coordinate with both excursion owners —
R2d is literally the seam between them.

## Findings — R2d-1/R2d-2 implementation (codex-4)

Implemented `scripts/r2d_concept_coverage.py` and wired it into
`scripts/iatc_semcheck.bb` as `R2d concept-coverage`.

R2d-1 decisions:

- Concept source for this dispatch is IATC graph node `:text`.
- The proof-region `fable-<id>-dp-emacs.json` marks are useful but currently
  whole-paper character-offset marks, not line-scoped proof-region concept
  marks, so joining them would pollute proof-local coverage.
- `known` threshold: encyclopedia nLab/NNexus provenance, or a genuine
  concept-index recurring core with `df >= 25`.
- `imported` is wired as an empty slot and remains N/A until R2d-3 has the
  WARP-ORCH-3 descent/phylogeny artifact.

R2d-2 behavior:

- Reuses `sfc_concept_coverage.definition_sets` and normalization logic rather
  than forking the SFC substrate lookup.
- Emits check-graph shape:
  `{:check :concept-coverage :pass :rate :reasons :per-item}` plus bucket
  counts, undefined list, imported slot, and source metadata.
- Gate semantics is report-only: undefined concepts become flagged gaps in
  `:reasons` and `:undefined`, never a hard failure. Empty concept extraction
  is `:status :na` and still passes.
- `iatc_semcheck` profile now carries `:concept-coverage` alongside the R2a/R2b/R2c
  description profile.

Generated reports:

- `holes/excursions/r2d-spec.md`
- `holes/excursions/r2d-concept-coverage.md`

All-nine loop-run-70b final coverage:

| paper | coverage | defined | known | imported | undefined |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0705.0452` | 1.000 | 6 | 0 | 0 | 0 |
| `0706.1286` | 0.500 | 2 | 0 | 0 | 2 |
| `0708.1921` | 0.500 | 1 | 0 | 0 | 1 |
| `0708.2067` | 1.000 | 8 | 0 | 0 | 0 |
| `0709.0248` | 0.800 | 8 | 0 | 0 | 2 |
| `0711.0473` | 1.000 | 2 | 0 | 0 | 0 |
| `0712.0724` | 1.000 | 3 | 0 | 0 | 0 |
| `0801.0199` | 1.000 | 4 | 0 | 0 | 0 |
| `0801.3843` | 1.000 | 6 | 0 | 0 | 0 |

Coverage spread: min `0.500`, max `1.000`, mean `0.867`.

Worked-proof reproduction:

- `0706.1286`: defined `bicategory`, `ring isomorphism`; undefined
  `cat like bicategory`, `calmod like bicategory`.
- `0709.0248`: the extensional-category case is covered by `defined-index`;
  undefined gaps are `parameterized rules`, `standard rules`.
- `0708.2067`: model-category/cofibration/square concepts covered; ref nodes
  are intentionally skipped because imported/descent is R2d-3.

Gates passed:

- `python3 -m py_compile scripts/r2d_concept_coverage.py`
- `pytest -q tests/test_r2d_concept_coverage.py`
- `bb tests/iatc_semcheck_test.clj`
- `bb scripts/iatc_semcheck.bb data/iatc-argument-graphs/loop-run-70b`
- `clj-kondo --lint scripts/iatc_semcheck.bb tests/iatc_semcheck_test.clj`
- `emacs --batch -l /home/joe/code/futon4/dev/check-parens.el scripts/iatc_semcheck.bb tests/iatc_semcheck_test.clj`

## Review — claude-1 (author ≠ reviewer), 2026-06-17 · commit 8d9acad

**Verdict: PASS, clean — no amendments.** Checked: re-ran all gates myself (py_compile OK; pytest
3/3; **bb iatc_semcheck_test 2 tests/10 assertions 0 fail**; `:concept-coverage` present in all 9
aggregate profiles, report-only). Read the diff. Genuinely reuses `sfc_concept_coverage`
(normalize_concept, definition_sets, boilerplate_phrase) — not forked. Gate semantics correct:
empty extraction → `:na pass=true`; undefined → report-only (`pass` always true); `imported`
wired but N/A pending R2d-3. R2d-1 decision (IATC `:text` only; fable proof-region marks deferred
because they're whole-paper char-offset, not line-aligned) is sound + recorded in `r2d-spec.md`.

**Spot-checked the numbers are honest** (the CAS-SEL-3 lesson — don't trust headline coverage):
- `0706.1286` = 0.5 is **correct**: `bicategory`/`ring isomorphism` defined; `calmod-like
  bicategory` / `cat-like bicategory` (df=0) flagged undefined — the paper's *own specializations*,
  a genuine substrate gap (candidate R2d-3 import or genuinely novel). Not a bug.

**Two inherent caveats (record, don't block — both report-only so they can't false-FAIL):**
1. Coverage inherits **SFC1's "defined = evidence-exists"** caveat — a concept is `defined` on ANY
   definition evidence, not necessarily a usable structured def. Honest (sources are traceable per row).
2. Coverage is **bounded by concept-EXTRACTION quality**. `0708.1921` = 0.5 because its extracted
   "concepts" are `sigma` / `mu inv` — bare symbols/fragments, not real concepts. `boilerplate_phrase`
   catches phrase-fragments ("category whose") but not bare symbols/single Greek letters. So the mean
   0.867 mixes real-concept coverage with extraction noise, and the residual map gets the occasional
   spurious "what is mu inv?" question. Harmless (report-only) but real.

**Minor follow-on (R2d-2b, not dispatched):** tighten concept extraction to drop bare symbols /
single Greek letters / ≤2-char fragments before classification. Upstream/shared SFC-extraction
concern (not an R2d defect), so documented rather than hot-patched into shared code.

**Net:** R2d-2 is correct and honest; ship it. The coverage *number*'s ceiling is the concept
extractor, not the checker. R2d-3 (imported/descent) unblocked now that WARP-ORCH-3 phylogeny passed
co-review — it walks the cites-prior-user chain back to a definition event (per my WARP-ORCH-3 notes).
