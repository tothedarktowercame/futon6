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
