# DP fleet plan of work — capabilities to have online when Joe's back

Agreed 2026-06-13 with Joe before he stepped away. Orchestrated by claude-1
(moderated: 2 Claude + 2 Codex executing at once, rotating the backlog).
Goal of the run: drive grounding coverage UP and well-formedness errors to 0,
leaving only irreducible debt (real definition holes). Read the trajectory in
`holes/loss-ledger.md` (committed, appended every loop tick).

Legend: ✓ online · ▶ in flight · ☐ targeted this run.

## Weft — per-paper structure (coverage ⊥ invariants)

ONLINE (✓):
- math-lexeme classification; role-gap → concept (NNexus/nLab)
- Let-binder + "$X$ is a $Y$" + conjunct binders (definiendum/definiens)
- superpod scope manifest (40-type detector)
- $-span + display-equation math envelope (R1, every `$…$`/eqnarray is a scope)
- symbol tagging + grounding (use→binder edge)
- binder/scope ATOMICITY + NESTING + sentence clamp (well-formedness CLEAN)
- informal proof-move layer (deferred-verification / suffices / WLOG)
- quantifier + where/with grounding; defined-in-paper grounding
- non-symbol-token classifier (layout/text-mode excluded from the denominator)
- label/ref/cite reference-graph harvest

TARGETED THIS RUN:
- ▶ per-capability REFACTOR (`scripts/dp_capabilities/`) — the enabler that
  ends merge-conflicts (codex-vscode, under a byte-identical gate)
- ☐ Galois alias-layer — `in_mathlib` matcher connects prose names to
  `IsGalois`/`PointedGaloisObject` etc. (corrects false-DEBT corpus-wide)
- ☐ subscript/superscript & function-application grounding (more C-SYM-GROUND)
- ☐ residual well-formedness → absolute 0 (the 1 W-SENTENCE + C-MATH-NONNULL)
- ▶ scale DP generation — more papers into the golden set so coverage is
  measured representatively (not on a handful)

## Warp — corpus second layer (cross-paper)

ONLINE (✓):
- W1 bibliography extraction — all 9742 math.CT eprints (256k bibitems, 287k cites)
- W3 concordance — 78,286 terms, 23.5M defined/used observations across the corpus

TARGETED THIS RUN:
- ▶ W2 citation graph — linkage repair (first pass landed at only ~2%; raising
  via arXiv-id extraction + normalized title/author matching)
- ▶ corpus-DEBT report — the payoff: concepts used across many papers but
  defined in NONE and absent from Lean∪PlanetMath∪nLab (cross-paper Skuld)
- ☐ (stretch) concordance query surface / shuttle integration

## Discovered during the run (added per Joe's standing instruction)

- ☐ **DEBT-report concept-filtering** (found 2026-06-13): the corpus-DEBT
  `debt_frontier` is keyed on raw cseq macros (`\coker`, `\hocolim` = real
  notation; `\frontmatter`, `\vfuzz` = layout noise), not concept terms. Must
  filter to the concept vocabulary (definienda / multi-word terms / authority-
  resolvable) so it surfaces real definition-holes (the "Galois object" class),
  not control sequences. → codex-3.
- ☐ **W2 re-run with the improved matcher** (found 2026-06-13): `warp_citations.py`
  was improved but `citations.json` is stale at 2% — regenerate over the corpus
  and verify the new linkage rate honestly. → codex-4.

## Coverage target

"Good" = grounding well up from today's ~52% across a representative corpus,
with well-formedness at 0 and only irreducible C-DEFINIENS-DEBT remaining.
"Perfect" approaches full grounding. The loop continues until then or claude-1
judges diminishing returns; mission-close stays Joe's call.

## How claude-1 logs progress while you're away

Every loop tick: run `scripts/log_loss.py "<note>"` → appends a timestamped row
to `holes/loss-ledger.md` (committed) and `data/loss/loss-log.jsonl` (live).
On return, `holes/loss-ledger.md` is the trajectory; `git log` shows what
capabilities landed (each detector commit names the capability + before/after
numbers); the claim ledgers (`data/loss/claims.jsonl`, `data/warp/claims.jsonl`)
show who did what.
- ☐ **memory-safe batch runner** (found 2026-06-13): `dp_batch.py` OOM'd at 8/200
  (loads concept-authority/nlab/mathlib in-process per paper, leaks). Needs
  subprocess-per-paper isolation before scaling the corpus. → codex-3.
- ☐ **display-defined (`:=`) symbol grounding** (found 2026-06-13): symbols
  defined in display equations (the codiagonal `:=` class, R6) are still
  ungrounded — a real C-SYM-GROUND lever. → claude-3.
- ☐ **wf generalization** (found 2026-06-13, HIGH PRIORITY): scaling golden 32->62
  exposed 838 well-formedness errors (was 1) — the atomicity/nesting/sentence
  invariants were overfit to hand-tuned papers. Bucket stale-golden vs real
  detector bugs; drive corpus wf back to 0. → claude-4. NOTE: regenerate
  scale-gen's new papers AFTER this fix lands.
- ☐ **appositive typing** (found 2026-06-13, BIG LEVER): "a Hopf algebra $H$",
  "the category $\C$" — type-noun immediately followed by the symbol. claude-2's
  residue analysis: this is ~78% of the ungrounded tail and structurally distinct
  from Let/is-a/quantifier/where (why the tail stalled). High-precision when
  anchored to a type-noun lexicon + head position. → claude-3.
- ☐ **noise context-classification** (found 2026-06-13; GOVERNANCE: claude-1):
  ~20% of residue is noise (prose-in-math, env/layout names). Exclude it
  DETECTOR-side via a CONTEXT test (prose inside \text/\intertext/captions/
  layout → text-mode/layout kind), NOT a checker spelling-denylist — keep the
  "non-math by WHERE it sits, not what it spells" principle. The checker's
  existing exclusion then applies honestly (+~5pp, denominator-correction).

## Ceiling reached (2026-06-13, claude-2 residue re-analysis)

Corpus grounding ~70%, wf 0. claude-2's verdict: near the STRUCTURAL CEILING
(~mid-to-high 70s%). Construct-based grounding largely exhausted. TWO final
levers, then the remainder is honest irreducible debt (bound indices, dummy/
generic variables — record as irreducible-debt, do not chase):
- ☐ boundary-parity root fix ($$/$ tokenizer) — ~+6pp denominator correction
  + kills the wf-error family at root. SHARED tokenizer (anatomy_v0_sweep) —
  strictly tightening; claude-1 reviews. → claude-2.
- ☐ conservative appositive lexicon extension (point/element/pair/vector/...,
  exclude form/relation/theory/number) — a few pp. → claude-3.
After these: grounding axis structurally COMPLETE; loop consolidates.

## Breadth phase (2026-06-13, loop re-invoked after ceiling)

Grounding axis structurally complete (75%, wf 0). Loop continues on BREADTH:
scale toward the full 9742 math.CT eprints so the grounding number + DEBT
frontier are archive-representative. Grounding % expected to stay ~75% (new
papers average the ceiling); the value is representativeness + a comprehensive
concept-DEBT frontier. Throttled subprocess batches; warp artifacts refreshed
as the corpus grows. → codex-3 scaling; claude-1 measures + logs each tick.

## Breadth stress-test finding (2026-06-14): one pathological outlier

The 949-paper scale showed 4215 wf errors — but the diagnostic proved they are
NOT stale (persist after current-detector regen) and come almost ENTIRELY from
ONE paper: 1001.4071, a 1.2M-char book-length doc with unusual `$$...$$` spans
that embed LaTeX comments (`$$}%%ascii three dollars...`). All 4215 are W-ATOMIC
(bind/typed boundaries inside these weird display spans that the snap pass
doesn't handle). The OTHER 948/949 papers are wf-CLEAN.
- ☐ **outlier-span handling** (real detector-generalization bug, HIGH value but
  not loop-cheap): make the snap pass handle long/comment-embedded `$$` display
  spans so bind/typed boundaries snap out. Until then, 1001.4071 is a KNOWN
  outlier — the corpus is effectively wf-clean (948/949). Do NOT exclude it to
  hide the number (that's gaming); fix the span handling or report it as a
  flagged outlier. → a focused detector task / Joe's prioritization.
NOTE: the "clean full regen" ledger row mislabeled this as staleness; it is a
real single-paper outlier. The corpus wf is honestly "0 except 1 outlier paper".

