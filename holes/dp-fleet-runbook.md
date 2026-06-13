# DP fleet runbook — structure-mine arXiv by playing coverage ⊥ invariants

**Status: LIVE 2026-06-13.** Orchestrated by claude-1. This is the
operational scale-up of `M-distributed-proofreaders` from one-paper interactive
to many-Claude continuous. If you are a Claude agent belled to this file, this
is your standing task until arXiv (CT corpus first) is fully structure-mined or
claude-1/Joe stands you down.

## The goal

Drive the math-paper structural markup (the "DP" capability) to convergence
across the corpus: **coverage → ceiling, well-formedness errors → 0**, with the
irreducible-debt floor (real definition holes) left as honest residue. The
artifact is the markup itself + a falling loss dashboard.

## The two scripts (do not confuse their roles)

- `scripts/dp_paper_view.py` — the **DETECTOR**. Proposes structure, emits
  `data/showcases/ct-anatomy/golden/fable-<paper>-dp-emacs.json`.
  Run: `python3 scripts/dp_paper_view.py <paper> --with-binders --with-scopes --with-concept-authority --with-xref`
- `scripts/check_invariants.py` — the **CHECKER / loss dashboard**. Reads only
  the emitted JSON, emits a typed violation list.
  Run: `python3 scripts/check_invariants.py <paper>` or `--corpus`.

**Author ≠ reviewer, even in code.** You FIX the detector. You NEVER edit the
checker to make a violation vanish — that is gaming the loop and fails review.

## The loop you run (bounded batches, never silent)

1. **CLAIM** before working — append to `data/loss/claims.jsonl` one line:
   `{"agent":"<you>","claim":"<capability-or-paper-batch>","at":"<iso>","state":"open"}`.
   First read the file; if your target is already `open` by another agent, pick
   another. (data/ is gitignored — this is transient coordination state.)
2. **MEASURE** — run the checker (`--corpus`, or on your batch) and read the
   dashboard. The dominant debt-class × cheapest fix = your next move (the
   DP-discipline: clear the most loss per unit work).
3. **FIX** — improve the detector (or bring a capability online). Small,
   principled changes; mirror the existing idioms in `dp_paper_view.py`.
   - If the fix is substantial coding, **hand off to an idle Codex** (codex-2/3/4):
     `python3 ../futon3c/scripts/agency_send.py --from <you> --to codex-N --kind bell` (prompt on a quoted heredoc).
     Always pass `--from <you>`. Include goal, files, acceptance bar, gates,
     and "bell <you> back with summary + shas." Then YOU review its diff
     (read it, re-run the checker yourself) before accepting — a real gate.
4. **VERIFY (the gates)** — `python3 -m py_compile` the changed script;
   regenerate the affected papers; re-run the checker. **Acceptance: your
   target violation-class drops AND coverage does not regress** (best-guess
   grounded must not fall, `symbol_tagged` stays 1.0, `math_coverage` stays
   1.0). If coverage dropped, you traded it for well-formedness — back out.
5. **RECORD** — set your claim line `state:"done"` with the before/after
   numbers; commit the detector change (Co-Authored-By: Claude Fable 5
   <noreply@anthropic.com>); never commit `data/`.
6. **CHECKPOINT** — bell claude-1 with {capability, papers touched, dashboard
   delta, shas}. Then take the next car. Stop the batch and await
   instruction if you hit a judgment call, a coverage regression you can't
   resolve, or the `DP-FLEET-OFF:` sentinel from claude-1/Joe.

## Capability backlog — bring the highest-value online FIRST (Random Access)

Hand-made map on 0809.2517 (✓ live · – exists in another view, unwired · ·
unbuilt). Rank by loss-cleared-per-unit-work, not list order:

- **C1 macro-table → classifier join** (cheapest, widest): the per-paper macro
  table is built but not consulted when classifying tokens, so the whole
  author-macro tail (`\C \Hom \Set \Cat \id \op …`) is FALSE-unknown across
  hundreds of papers. Single-site fix, biggest corpus-wide debt cleared. See
  `holes/anatomy-v0-loss-backlog.md` C1–C3. **Start here if unclaimed.**
- **defined-in-paper** (–): harvest `\newtheorem`/definition envs + "we
  call/define/denote … by" so definienda introduced in-paper GROUND their
  later uses → kills the dominant `C-SYM-GROUND` debt and `canon holes`.
- **TeX environments** (–): theorem/proof/lemma/definition/remark env scopes
  (env/* — legitimately multi-sentence; the checker exempts them from the
  sentence clamp). Cheap, structural, enables defined-in-paper.
- **label/ref/cite harvest** (·): `\label/\ref/\eqref/\cite` → the in-paper
  reference graph; lets a deferred claim point at its target (and pairs with
  the informal-proof-move layer: a hedge + the equation it defers).
- **canon holes** (–) / **C-DEFINIENS-DEBT**: surface terms resolving in
  neither Lean/PlanetMath/nLab — the three-Norn DEBT cell, real definition
  holes. Mostly reporting; the honest floor, not a bug to "fix".
- **latexml deep parse** (·): heavy display/diagram parse (GrCalc, xy-pic).
  LAST — defer per Joe ("right chops, but not on our first outing").

## Invariants the checker enforces (your well-formedness target)

- W-ATOMIC: no non-math (binding/structural) scope boundary inside a `$…$` /
  display span — math spans are atomic.
- W-NEST: structural scopes nest or are disjoint, never partially overlap.
- W-SENTENCE: a non-env scope must not cross ". " (the period is English).
- C-MATH-NONNULL: every `$…$` carries a math mark (R1, hungry-$).
- C-SYM-TAGGED / C-SYM-GROUND: every letter-run in math is tagged; ungrounded
  is explicit debt.

Coverage pushes tagging up; well-formedness punishes sloppy tagging; the
fixpoint where both hold is correct markup. Hold both — never trade.

## Hard constraints

- Never restart the futon3c JVM. Never commit `data/`. Cursor-safe reload only
  (`paper-anatomy-reload`) if you touch the Emacs side. Verify a checker
  finding is real before dispatching it as work (a buggy checker dispatches
  garbage — claude-1 already hit and fixed one: a naive `$` regex; use
  `sweep.math_spans`, the shared tokenizer).
