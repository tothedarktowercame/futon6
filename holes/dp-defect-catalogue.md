# DP defect-class catalogue — the dual of the recognizer registry

**Started:** 2026-06-15 (Joe + Opus, from a visual QC pass on the dp-demo).
**Why this file:** `check_invariants.py` reports `0905.0595` **clean** (0 wf
errors, 100% symbol-tagged) while the rendered page is visibly wrong. The
checker is at *its* fixpoint, not at *correct* — it only enforces ~5 invariant
families. The defects an eye/LLM notices are **classes the checker doesn't yet
encode.** This catalogue is where a noticed class is named *once* so it can be
recognised *corpus-wide* — "notice a defect class once, recognise it forever",
the dual of the recognizer registry (`M-distributed-proofreaders` §3).

**The loop this serves:** eye/LLM notices a class → named here → promoted to a
deterministic `C-*`/`W-*` invariant in `check_invariants.py` → re-mine = re-run
the (cheap) checker over already-emitted `fable-*-dp-emacs.json` → the class
lights up corpus-wide as dispatchable work → detector fix → never regress.
"Re-mining runs over the *annotated* texts" = the LLM/checker reads `text +
marks`, not raw TeX.

## Severity / disposition key
- **promote-to-invariant** — recurring, mechanically checkable → new checker rule.
- **detector-fix** — capability exists, just unwired or mis-firing.
- **feature** — new capability (a representation we don't emit yet).

---

## DC-1 — terminology not noticed  · detector-fix (merge) · DOMINANT
**STATUS: first pass landed 2026-06-15** (Opus, direct — Agency down, no Codex to
bell). `scripts/dp_enrich.py` is a **post-annotation enrichment pass** over the
stored `text + marks` (read-only w.r.t. the live `golden/` run): it merges a
`concept` mark layer from `build_golden_paper`'s prose detectors, run at render
time by `dp_anatomy_html`. `CONCEPT_ENDINGS` extended with CT vocabulary
(subcategory/colimit/system/limit/monad/adjunction/transformation/equivalence).
Result on `0905.0595`: 61 concept terms (45 distinct); all of Joe's flagged
lines now tag (L136/146/147/163/166/176). Coverage UNCHANGED (tagged 1.0,
grounded .6234, wf 0, math 1.0 — checker ignores the prose layer). Demo set:
66–214 concept spans/paper. **Residual:** ~2/61 noisy phrases with proper-noun
heads + mid-phrase glue ("Quillen equivalent to the …") — needs real
NP-chunking, a follow-on. **Not yet promoted** to a `C-TERM-COVERAGE` invariant
(would need the term layer in the persisted JSON, not just render-time).
This pass IS the "re-mine over annotated texts" pattern — the template for DC-3/DC-4.

**PROMOTED TO INVARIANT 2026-06-15** (Joe: "needs to be an invariant — rolled
out everywhere"). Two halves, both by Opus (Agency down; Joe spot-checks):
- **Detector half:** `dp_paper_view` now persists the `concept` layer into the
  mined JSON (`build()` appends `dp_enrich.concept_marks`, defensive try/except
  so it can never crash the mine). `dp_enrich.enrich` made idempotent (skips if
  a `concept` layer is already present) so render-time + persisted don't double.
- **Checker half:** `C-TERM-COVERAGE` in `check_invariants.py`. Independent
  locator = AUTHOR EMPHASIS (`\textit/\emph/\textbf/...`), which the detector
  does NOT key on → not self-grading. Every emphasised prose phrase must carry a
  concept/definiendum/definiens mark; uncovered = `extend-coverage` debt. New
  scalars `terms_emphasised`, `term_coverage`. Surfaced in the demo stat panel.
- **Rollout:** ride the re-mine (no bulk rewrite, no race with the live mine).
- **Finding:** `term_coverage` 2–13% across the demo set even when enriched —
  emphasis marks the DEFINITION site, and the concept detector notices terms in
  passing but misses them there. So **C-TERM-COVERAGE is exactly the DC-2
  measurement**; DC-2 (detector keys on emphasis) is the detector response that
  drives this dial up. The adversarial loop is wired end-to-end.


Named math concepts in prose carry no mark.
- **Evidence:** `0905.0595` L136 (*canonical functor*, *small full subcategory*),
  L146 (*dense*, *subcategory*, *canonical colimit*), L147 (*canonical functor*),
  L173 (*cofibrantly generated*), L176 (*complete and cocomplete category*,
  *model category structure*).
- **Cause:** `dp_paper_view.py` descends into math-mode symbols; the prose-concept
  layer (`build_golden_paper.py` + `concept_authority.py`, 130 960 terms) is never
  merged into the DP marks. This view shows one of two layers.
- **Authority coverage (checked):** model category 198 · weak factorization system
  20 · dense functor 16 · cofibrantly generated 4 · **canonical functor 0**
  (paper-local — needs DC-2).
- **Fix direction:** run the term-spotter over non-math prose spans, merge as a
  `concept`/`term` mark layer. Then an invariant `C-TERM-COVERAGE` over
  authority-known terms makes the gap measurable corpus-wide.

## DC-2 — definiendum mis-attribution  · detector-fix + promote
**STATUS: landed 2026-06-15** (Opus, direct). The detector now keys on author
emphasis (`dp_enrich._EMPH_RE`, identical to `check_invariants.EMPH_RE`): every
`\textit{}`/`\emph{}`/… phrase becomes a `concept` mark (source `emphasis`),
kept even single-word ("dense"). Detector + checker share `sweep.math_spans` for
"where math is" and an identical term gate (skip emphasis inside math, and skip
emphasised *sentences* — `endswith(".")`/`". "` — which are stress not terms).
Result: `term_coverage` 0% → **100%** on all 7 demo papers, `C-TERM-COVERAGE`
debt → 0, symbol/wf untouched. **Showcase pointer: 0905.0595 L146** — `dense`
now renders `k-concept` ("term: dense — author-emphasised term"). The C-TERM
dial is now a real convergence axis, not a permanent debt.


The wrong token is bound as definiendum; `\textit{}`/`\emph{}` emphasis ignored.
- **Evidence:** `0905.0595` L146 "A small full subcategory `$\ca$` … is called
  *dense*" → detector bound **`$\ck$`** as definiendum; the defined *term* is
  "dense", predicated of `$\ca$`.
- **Fix direction:** treat `\textit{X}`/`\emph{X}` adjacent to "is called / is
  said to be / we call" as defining term X. Doubles as the recogniser for the
  paper-local terms DC-1 can't get from the authority.
- **Promote:** `W-DEFD-EMPH` — an "is called *X*" with no definiendum mark on X.

## DC-3 — Let–Then implication split across the sentence boundary  · feature/promote
**STATUS: landed 2026-06-15** (Opus, direct). Detector `detect_implications`
(dp_paper_view) emits an `implies` scope pairing hypothesis ⟹ conclusion across
the ". " boundary; kind `implies` is non-structural so it is exempt from
W-SENTENCE/W-ATOMIC (it is meant to span them) while the inner let-binder stays
clamped. Checker `C-IMPL-PAIR`: a sentence-initial Then/Hence/Thus with a nearby
Let/Given/Suppose and no `implies` scope = debt. On 0905.0595: 5 implications
found, `C-IMPL-PAIR` debt 0, wf still 0. **Showcase pointer: L163** (k-impl
scope). **Nested-blockquote rails** (Joe's forum-thread idea) shipped in the
renderer: each multi-line scope (env/*, `implies`, multi-line let-binder) draws a
coloured left rail, outer→inner — **L158–159 shows `[env, impl]`** (a Let–Then
inside a theorem env, two stacked rails). The showcase now re-mines in-memory
(`dp_anatomy_html --remine`) so it reflects the current detector without writing
golden/.
Render refinement (Joe, 2026-06-15): the `implies` scope draws ONE continuous
underline across the whole Let…Then (text-decoration flows through nested marks +
abutting segments, `skip-ink:none`) so it reads as a single scope; the detector
emits `kw-hyp`/`kw-con` keyword marks so `Let` (bold blue, binder class) and
`Then` (bold purple, inference class) are styled by syntax class.
Rails refinement (Joe, 2026-06-15): rails are now CONTINUOUS full-height bars
(own grid column, flex/grid stretch, rows abut → one unbroken line per scope);
and ONLY containing scopes draw rails (env/* + implies). Binders no longer draw
a rail — a binder that runs past an environment boundary (0905.0595 L172–173:
let-binder L172–173 vs env/proof L165–172) was making the blockquote cross inline
markup; binders keep their inline underline only.


A theorem "Let … . Then …" becomes two disconnected binders instead of one
hypothesis→conclusion.
- **Evidence:** `0905.0595` L163 "Let `$\ck$` be a cocomplete bounded category.
  **Then** `$(\ck,\Iso)$` is a … weak factorization system." → two independent
  `let-binder` marks; the implication is lost.
- **Cause:** `W-SENTENCE` (correctly) clamps the binder at ". " — but nothing
  re-joins hypothesis and conclusion.
- **Fix direction:** an **implication scope** (Given/Let … . Then/Hence/Thus …),
  exempt from the sentence clamp the way `env/*` already is, lexically triggered
  by the consequent connective. A logical layer *above* binding.
- **Promote:** `C-IMPL-PAIR` — a sentence-initial "Then/Hence/Thus" with a prior
  "Let/Given/Suppose" and no implication scope linking them.

## DC-4 — structured math parse for mouse-over verification  · feature
Parse complex math envs into an S-expr / Content MathML tree so hover *confirms*
the parse, not just the colouring.
- **Evidence/target:** `0905.0595` L138 `E_\ca: \ck \to \Set^{\ca^{\op}}` →
  `(: E_\ca (-> \ck (presheaf \ca)))`; L142 `E_\ca K = \hom(-,K)/\ca^{op}`.
- **Asset:** LaTeXML already emits Content MathML (`:latexml-fragment-parse`,
  satisfied). This is the backlog's deferred "latexml deep parse" — pulled
  forward, **scoped to verification tooltips, not grounding.**
- Also exposes the DC-? mis-tokenisations (`gf` = g∘f L104, `QR` = Q·R L218)
  structurally — a parse won't glue juxtaposed identifiers into one symbol.

## DC-5 — interwoven text+math structure  · feature (generalises DC-4)
Display equations grammatically embedded in a prose sentence.
- **Evidence:** `0905.0595` L136–144 — "the canonical functor [DISPLAY] assigns
  to each object `$K$` the restriction [DISPLAY] of its hom-functor [INLINE]…"
  is one grammatical unit spanning prose + two displays.
- **Fix direction:** a single parse spanning prose and math; the tooltip/structure
  view should show the sentence skeleton with the displays as constituents.

---

## DC-9 — environment scope truncation + markup-inclusion inconsistency  · landed
**STATUS: landed 2026-06-15** (Opus). The nLab-wiki scope detector truncated long
proofs (a `pos+400` clamp → 0905.0595 proof L165–**172** missing its last line
L173 and `\end{proof}`) and missed custom env names (`coro`/`propo` got no
scope while `proof`/`lemma` did). Replaced as the env source by
`detect_tex_environments` (dp_paper_view): exact `\begin{NAME}…\end{NAME}`
matching, nesting-safe, ALL env names canonicalised (Joe's call), **delimiters
INCLUDED** (Joe's call — rail brackets `\begin`→`\end`). Skips display-math envs
(already math scopes) and whole-paper wrappers (`document`/`abstract`, whose rail
would cover everything). nLab `env/` scopes filtered out of the manifest so there
is one env source. Result on 0905.0595: proof L165–**174** (full), `propo`/`coro`
now scoped consistently, corollary L180–183 with the Let–Then nested inside
(`[env, impl]`). Coverage unchanged (env marks non-structural). Caveat: nLab env
scopes carried theorem-statement fields that the simple tex-env tip drops — fine
for the showcase; revisit if a consumer needs them.

## Mis-tokenisation classes (found before Joe's list; folding in)
- **DC-6 — multi-letter run glued into one symbol.** `LETTER_RUN` tags `gf`
  (L104, `h=gf` = g∘f) and `QR` ("the composition `QR`") as single symbols,
  inflating `symbol_tagged=1.0`. **promote:** `W-SYM-JUXTAPOSITION`.
  **STATUS: landed 2026-06-15** (Opus, direct). Detector splits a BARE italic
  multi-letter run into single-letter symbols (shared `mathalpha_regions` /
  `is_script_run` discriminators in math_envelope: NOT for `\mathrm{Hom}` operator
  names, NOT for `^{op}` script modifiers, NOT digit-bearing runs). Checker:
  PIECEWISE symbol coverage (a run tiled by several marks still counts tagged, so
  the LETTER_RUN denominator is unchanged) + `W-SYM-JUXTAPOSITION` debt for any
  remaining bare multi-letter symbol mark. On 0905.0595: `gf`→g·f (L104),
  `QR`→Q·R, **grounded 0.62 → 0.71** (split letters ground), symbol_tagged 1.0,
  wf 0, juxtaposition debt 0. **Showcase pointer: L104** (hover g/f: "split from
  'gf'").
  Two refinements from controlled A/B (DP_NO_JUXT_SPLIT toggle, grounded must not
  fall): (1) **split only UNGROUNDED wholes** — a grounded multi-letter unit is a
  NAME the binder resolved (sloppy bare `Ab`=category, not A·b); splitting it
  dropped 0710.2254 −1.2pp, so grounded units are kept (and the checker flags
  only UNGROUNDED bare runs). (2) **W-SYM-JUXTAPOSITION scoped to in-span marks** —
  the detector tokenizes per-file while the checker re-tokenizes the concatenated
  text; `$`-parity differs across joins (xy-pic displays), so the checker was
  flagging 848 out-of-(checker-)span diagram tokens (`dr`/`ur`/`Gf`). Gating on
  the checker's own spans removed the divergence. A/B final: grounded ≥ baseline
  on all (0710 +0.0, 0711 +0.017, 0905 +0.091), W-JUXT 0.
- **DC-7 — truncated / conflicting definiens.** `\ck` defined 3× with different
  definiens ("cocomplete bounded category" / "complete" / "complete"), grabbing
  fragments of "complete and cocomplete locally presentable category".
  **promote:** `C-DEFINIENS-CONFLICT` (one definiendum, ≥2 non-subsuming definiens).
- **DC-8 — definiens grabs layout** (spans a hard `\n`). Cosmetic; **promote:**
  `W-DEFINIENS-NEWLINE`.

## Promotion order (loss-per-work, provisional)
1. DC-1 merge (capability exists; clears the most visible loss).
2. DC-2 + DC-6 + DC-7 (cheap deterministic invariants, high signal).
3. DC-3 implication scope (new structure, high conceptual value).
4. DC-4 → DC-5 (the structured-math features; largest, most generative).

## The reasoning layer (IATC) — successor to the object-layer fixes

DC-1…DC-9 made the **object** layer correct (terminology, definienda, symbols,
environments). The **reasoning** layer (inferences as typed arrows over claims,
nested per IATC, references resolved to bindings) is written up separately in
`holes/excursions/E-iatc-model.md` — the IATC model, what we detect
(`detect_inferences` / `detect_enumerate_anaphora`), the standoff rendering
principles, a worked Clojure example on 0905.0595 L185–193, the completeness gap
vs the spec, and a verification checklist. Grounded in Corneli et al. 2019
(arXiv:1803.06500).

## DC-11 — lexicon-grounded term spotting (symbol grounding for math TERMS) · landed
**STATUS: landed 2026-06-15** (Opus). Joe's caveat: terms like `left adjoint`,
`enough projectives`, `long exact sequences` were UNMARKED on 0807.1872 despite
being in nLab/NNexus — because the prose spotter (`hole_marks`) is a hand-coded
`CONCEPT_ENDINGS` SUFFIX heuristic and never consulted the 130 960-term lexicon
(`concept_authority` only used `resolve()` for symbol role-gaps). Fix
(`dp_enrich.lexicon_marks`, NNexus-style: first-word index + longest match):
match multi-word lexicon terms + a curated single-word set against prose, emit
`concept` marks **grounded** to their authority entry (`grounded` field). Proper
names filtered (all-Title-Case multi-word = a person, e.g. nLab mathematician
pages — "Carles Casacuberta"; eponymous terms like "Bousfield localization"
survive via the lowercase head-noun). New checker dial **`C-TERM-GROUND`** +
scalars `terms_concept`/`term_grounded` (mirrors C-SYM-GROUND). Result: 0807
concept marks 59 → 126; `term_grounded` 0.52–0.76 across the demo set; gates
unchanged (W-NEST-SCOPE 0, wf 0, symbol_tagged 1.0). Follow-on (Joe): **learn new
terms as we go and promote them** rather than rely on the existing list — start
by curating the curated list (the 1815 `candidate-terms`, which is noisy with
common words). Prerequisite for the EATC / exposition survey (a dense, grounded
object layer keeps the "negative space" clean).

### DC-11 precision (2026-06-15): concept-phrase clause-cut
The ungrounded SUFFIX heuristic (`hole_marks`) over-matched clauses ending in a
concept word — "short note we produce such a category" (Joe). `_trim_phrase` now
collapses a hole-phrase to its trailing noun phrase by cutting at the last
clause-marker (we/that/is/are/construct/prove/need/exist/such a/…), so clauses
drop to their NP (often single-word → filtered). The grounded (lexicon) layer is
clean; this only tightens the ungrounded heuristic layer, which `C-TERM-GROUND`
measures (0807 term_grounded≈0.73 — the ~27% ungrounded are these heuristic
phrases). Real reduction comes from the learn-and-promote phase (grow the lexicon),
not more clause-cut whack-a-mole.

## DC-12 — overfed / hungry concept phrases  · base-rate prior · landed
The clause-cut whack-a-mole above could not decide two opposite errors:
**OVERFED** ("interesting abelian category" — a qualifier glued to a real head)
and **HUNGRY** ("category of modules" where the paper means "…over a ring").
Lexically, "interesting" and "of modules" are both fine; the decision is
*statistical* — does this exact phrase recur? Measured df over 900 CT papers:
"interesting abelian category" = **1** paper (hapax → not a term);
"abelian category" = **197**; "localization of spaces" = 2 (vague, drop);
"category of modules" = 95 vs "category of modules over" = 61 (hungry).

Fix = the prose-term base-rate prior (E-prior-over-terms, the concrete instance
of M-prior-mathematics applied to terms): `build_term_prior.py` builds a corpus
document-frequency index (`data/term-prior-<msc>.json`); `dp_enrich._prior_normalize`
trims overfed phrases to their highest-df recurring head-anchored core, extends
hungry ones, and drops hapax junk. Author-`\emph{}` (DC-2) and lexicon hits
(DC-11) are authoritative (extend-only, never trimmed/dropped → C-TERM-COVERAGE
safe). No index present → NO-OP (gates hold on fresh checkout). MSC-repeatable:
re-point `--golden-dir`/`--out` per class; `DP_TERM_PRIOR` selects the active
index. df is also the **learn-and-promote** signal — a high-df phrase absent from
the lexicon is a promotion candidate. See `holes/excursions/E-prior-over-terms.md`.
