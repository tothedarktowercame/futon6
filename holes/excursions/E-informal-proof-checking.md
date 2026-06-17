# E-informal-proof-checking

*Excursion · owner **claude-1** · chartered 2026-06-17 · scope-out from the mark4
IATC pipeline. Bounded, owned end-to-end by one agent; built via Codex handoffs
that bell back for review (author ≠ reviewer).*

Parent question (Joe, 2026-06-17): *"Are we actually recovering the semantics?"*
The renderer and the ML had the same failure mode — and if the content were already
marked up semantically we wouldn't need the ML at all. So the real deliverable is a
way to **check** the recovered semantics, over the `.edn` reasoning structures the
70B emits, in Clojure — not a prettier HTML view.

---

## IDENTIFY — the gap

The IATC pipeline emits `.edn` argument graphs (`{:nodes :edges :holes}`; typed edges
with `:warrant` + `:source {:lines [a b]}`). We gate them in-run with `bb`:

- `iatc_argcheck.bb` — **rung 0: wiring well-formedness** (every node/edge has
  `:source`; edge endpoints resolve to node `:id`s; `:missing-warrant` mirrored in
  `:holes`; refs resolve).
- `substance_gate.py` — **rung 1: anti-degeneracy** (no shells; no `:premise ==
  :conclusion` self-loops; no ≥80% template collapse; no canned warrants).

Neither asks whether the reconstructed reasoning is **semantically faithful to the
proof**. The go-live's P3 (faithfulness) was hand-spot-checked only and came back
**PARTIAL** — the 70B reads the right region but anchors imprecisely (e.g. `0709.0248`
cites `\begin{proposition}` instead of the statement). We have evidence of *structure*
recovery (argcheck 9/9, substance 9/10) but only partial, manual evidence of *semantic*
recovery. This excursion closes that gap.

## MAP — the ladder, and what exists

```
rung 0  iatc_argcheck.bb      wiring well-formedness            [exists]
rung 1  substance_gate.py     anti-degeneracy                  [exists]
rung 2  THIS EXCURSION        deterministic semantic checks    [build]
rung 3  (later)               LLM-as-judge: warrant licenses    [deferred]
```

- `mark3_iatc_loop.py` already self-gates with `bb` per graph + retries — rung-2 slots
  into the same mechanism.
- `mark3_eval_harness.py` redefined `grounding-%` as warrant-resolution
  (resolved-warrant edges / total edges; commit `cc808d4`) — a **metric**, not yet a
  per-graph **gate**. Rung 2 promotes it.
- Source text + line structure per paper live in
  `data/showcases/ct-anatomy/golden/fable-<id>-dp-emacs.json` (`"text"`).
- Live material to build against: `data/iatc-argument-graphs/loop-run-70b/` (9 final
  graphs + `.attempts/`), and the manual P3 findings below as ground-truth anchors.

## DERIVE — what rung 2 checks

The key constraint (Joe's point): **we cannot check against a gold semantic markup** —
if we had one, no ML needed. So rung 2 = **deterministic properties that faithful
semantics *implies*** — necessary conditions, checkable in pure Clojure, no judge:

- **R2a · anchor-faithfulness** — for each node, the cited `:source {:lines}` text
  actually contains the node's key terms. Automates P3. (Needs the paper source text.)
- **R2b · goal-reaching closure** — the edge graph over nodes is a **DAG** (acyclic,
  beyond the self-loops substance already catches), with **no orphan nodes** (every
  node is a premise or participates in ≥1 edge), terminating in a **conclusion node
  reachable** from premises. A proof that doesn't connect its steps to a goal isn't a
  reconstructed argument.
- **R2c · warrant-resolution gate** — promote `grounding-%` to a per-graph gate: the
  fraction of inference edges carrying a *real* warrant (not `:missing-warrant`) must
  clear a floor; report the per-graph rate. Distinguishes "reasoned" from "asserted."

## ARGUE

> **IF** we want to know whether the 70B recovers the proof's reasoning (not just
> well-formed, non-degenerate structure),
> **HOWEVER** no deterministic gold semantic markup exists (its absence is *why* we use
> the LLM), and rungs 0–1 are purely structural,
> **THEN** check the deterministic properties that semantic faithfulness entails —
> grounding (R2a), closure (R2b), warrant-resolution (R2c) — over the `.edn`, in `bb`,
> and gate on them,
> **BECAUSE** these are *necessary conditions* for a faithful reconstruction, are
> checkable without a judge, and turn the manual P3 spot-check into a standing gate.
> The irreducibly-semantic residue (does the warrant *license* the step?) is rung 3, a
> judge — out of scope here.

## Design stance (Joe, 2026-06-17) — the gate is also a describer

Rung 2 isn't only pass/fail on finished graphs. The *same* lightweight checks, run
**inline as a paper is explored**, emit a **cumulative structured description** —
useful at *every* resolution. Two axes, orthogonal to the rung ladder (which is the
*depth* axis 0→3):

- **Resolution (coarse → fine):** document skeleton ("1 expository intro + 3 theorems
  with proofs + a conclusion") → per-region reasoning → per-edge warrant + imported
  terms. Each layer is independently useful: even the bare skeleton, knowing nothing
  else, is a real artifact.
- **Graceful degradation, *N/A ≠ FAIL*:** a check whose structure isn't present *yet*
  returns **N/A** ("not at this resolution"), never FAIL. The gate fails only on
  **present-but-wrong** structure, not on absent structure — so partial knowledge stays
  useful and the checks can run from the very first coarse pass.
- **Fast at every layer:** lightweight structural/grounding checks, *not* theorem
  proving, so they run continuously/inline as the description is refined.
- **Output = a paper profile:** the end artifact is a detailed, checkable description —
  which terms the paper imports, how it reasons about them, its region structure. This
  is the **ASSIMILATE** layer of the flexiformal pathway, and the same profile feeds
  **APM ⑥** scope-matching.

**Implication: build description-first.** `check-graph` already returns `:per-item` +
`:rate` (the description seed); the harness aggregates these into the per-paper profile
and treats absent structure as N/A, not failure. (The current handoffs run on complete
`loop-run-70b` graphs, so degradation isn't exercised by the prototype — but author the
checkers so absent structure → N/A from the start.)

## VERIFY — acceptance bar for the whole excursion

Run rung 2 over `loop-run-70b` (9 finals + the `0708.2185` attempt). It must:

1. **Reproduce/sharpen the manual P3 verdict** — flag `0709.0248`'s `\begin{proposition}`
   anchor as low-faithfulness; pass the clean `0706.1286` (Cat-like/CalMod-like terms
   sit exactly at cited lines).
2. **Catch `0708.2185`** on R2b (its self-loop is also a degenerate 1-cycle) — agreeing
   with substance's existing verdict via an independent route.
3. Emit a **per-graph semantic report** (R2a rate, R2b pass+reasons, R2c rate) + an
   overall **gate verdict**, with thresholds stated and justified by the observed spread
   (set floors from the data, don't hard-code aspirational numbers).
4. Be **deterministic** (same graph → same verdict) and run under the same `bb`
   self-gate path as rung 0/1.

Ground-truth anchors from the go-live P3 (hand-checked):
- `0706.1286` — faithful (terms at cited lines). Rung 2 should PASS.
- `0709.0248` — imprecise (`\begin{proposition}` ≠ the claim). Rung 2 R2a should FLAG.
- `0708.2185` — substance-failed (self-loop). Rung 2 R2b should FLAG.

---

## INSTANTIATE — Codex handoffs

Shared interface contract (so the checkers compose cleanly and the harness can wire
them without re-plumbing). Each checker namespace exposes:

```clojure
;; graph: parsed EDN map; ctx: {:paper-id "..", :lines ["L1" "L2" ...], :source ".."}
(check-graph graph ctx)
;; => {:check :anchor-faithfulness   ; keyword id
;;     :pass  true/false             ; gate verdict for THIS check on THIS graph
;;     :rate  0.0-1.0                ; the headline number (nil if N/A)
;;     :reasons ["node n3: cited L1510 lacks term 'extensional'"]   ; human-readable
;;     :per-item [...]}              ; per-node / per-edge detail for the report
```

Gates for every handoff: **BB** = `clj-kondo` clean + `futon4/dev/check-parens.el` on
any `.bb`; plus `bb` unit tests (a `test/` `.clj` run via `bb test` or a `deftest`
namespace). Each ends with **"bell claude-1 back with a summary + commit shas."**
Build against `data/iatc-argument-graphs/loop-run-70b` + the golden marks; verify on
the three ground-truth anchors above.

### H-R2a — anchor-faithfulness checker  · `scripts/iatc_anchor_faithfulness.bb` · PY/BB
**Goal:** per node, decide whether the cited `:source {:lines [a b]}` text actually
supports the node — i.e. contains the node's key terms — and report a per-graph
faithfulness rate. Automates the manual P3 check.
**Fix:** (a) resolve a graph file → paper-id → `fable-<id>-dp-emacs.json` → `"text"` →
1-based line array (allow `--marks-dir` / `--source` overrides); (b) for each node,
extract key terms from `:text` (math tokens incl. `\macro`/symbols, and content words
≥3 chars; drop stopwords + bare single letters), and test coverage against the cited
lines' text — a node is *faithful* if ≥ K of its key terms (or ≥ fraction τ) appear in
the cited span, *orphaned* otherwise; (c) `check-graph` per the contract; (d) a CLI
that prints per-graph rate + flagged nodes.
**Acceptance:** on the 9 finals, report per-graph faithfulness rate + flagged nodes;
**`0706.1286` scores high, `0709.0248`'s `\begin{proposition}`-anchored claim is
flagged.** State the chosen K/τ and the rate spread; set the gate floor from the spread,
not aspiration. Deterministic.
**Gates:** BB (+ PY if any tokenization helper is Python) + bb tests + report the numbers.

### H-R2b+c — closure + warrant-resolution checks  · `scripts/iatc_closure_check.bb` · BB
**Goal:** two pure-structure-over-EDN semantic checks.
- **R2b goal-reaching closure:** build the directed edge graph over node `:id`s; assert
  **acyclic** (report any cycle, incl. length-1 self-loops as the degenerate case);
  **no orphan nodes** (each node is a root premise or has ≥1 incident edge); a **terminal
  conclusion node exists and is reachable** from at least one premise. Reasons name the
  offending nodes/edges.
- **R2c warrant-resolution gate:** fraction of inference edges with a real warrant
  (not `:missing-warrant`, mirroring `mark3_eval_harness`'s redefined definition — reuse
  its logic, don't fork it) ≥ floor; report the per-graph rate.
**Fix:** two `check-graph` fns per the contract (`:check :closure`, `:check
:warrant-resolution`); a CLI printing both per graph.
**Acceptance:** on the 9 finals + `0708.2185` attempt: **`0708.2185` is FLAGGED by R2b**
(self-loop = 1-cycle), independently agreeing with substance; warrant-resolution rates
match `mark3_eval_harness` (≈ 6/28 aggregate). Floors set from the spread. Deterministic.
**Gates:** BB + bb tests + report the numbers.

### H-R2-harness — `scripts/iatc_semcheck.bb` (rung-2 aggregator + report + gate)  · BB
**Depends on** H-R2a + H-R2b+c (build against the interface contract; I wire/verify once
they land — do **not** duplicate their logic).
**Goal:** one entrypoint that loads each graph + its source ctx, runs R2a/R2b/R2c, and
emits (1) a per-graph **paper-description profile** — the cumulative structured
description (imported terms, region/structure skeleton, reasoning edges + warrants),
which the checks *certify* (description-first; the gate is also a describer) — (2) a
per-graph **semantic report** (`.edn` + short `.md`), and (3) an overall **gate verdict**
(`--gate` exit non-zero on failure), finals-only by default (`--include-attempts`
opt-in), mirroring rung-0/1 so it can self-gate in `mark3_iatc_loop.py` later.
**N/A ≠ FAIL:** when a check's structure is absent (e.g. a coarse skeleton with no
edges), aggregate it as **N/A**, not failure — the profile must stay useful at any
resolution.
**Acceptance:** `bb scripts/iatc_semcheck.bb data/iatc-argument-graphs/loop-run-70b`
prints per-graph R2a/R2b/R2c + an overall verdict; the report distinguishes the three
ground-truth anchors as specified in VERIFY. Deterministic; no network.
**Gates:** BB + bb tests + the report committed as evidence (e.g.
`holes/excursions/iatc-semcheck-loop-run-70b.{edn,md}`).

## Remaining gaps / notes
*(Codex agents: append findings + commit shas here.)*

### H-R2b+c — REVIEWED PASS (claude-1, 2026-06-17) · commit `a3b523e`
`scripts/iatc_closure_check.bb` + `tests/iatc_closure_check_test.clj`. Checked: clj-kondo
0/0; bb tests 8/8; R2c delegates to `mark3_eval_harness.warrant_resolution_counts` via a
`python3 -c` subprocess (reused, not forked). Functional on `loop-run-70b`:
- **R2b flags `0708.2185`** self-loop (1-cycle) — acceptance met, independent of substance.
- **Bonus find — R2b is *stronger* than substance:** it flags `0712.0724`'s
  `:e-functor-pitchfork` (`:premise [:F-functor :F-pitchfork] → :conclusion :F-pitchfork`
  — conclusion ∈ premises, a vacuous step) that **substance passed**, because
  `substance_gate.py:132` matches only the *first* premise token against the conclusion.
  → **Follow-up:** tighten `substance_gate` self-loop check to scan *all* premise tokens
  (or rely on rung-2 R2b for this class).
- **Real semantic-quality signal:** **4/9 finals** have orphan/dangling nodes
  (`0708.1921`, `0708.2067`, `0712.0724`, …) — nodes the 70B declared but never wired
  into the argument. argcheck/substance miss these; rung-2 surfaces them. This is exactly
  the "are we recovering the semantics?" signal — about half the graphs have disconnected
  pieces.
- Warrant-resolution per-graph 0.0–0.6 (aggregate ≈ 6/28, matches `mark3_eval_harness`).
- **Note for the harness:** `default-warrant-floor 0.0` ⇒ R2c is report-only by default;
  the harness must set the floor from the observed spread (`--warrant-floor`).

### H-R2a — REVIEWED PASS (claude-1, 2026-06-17) · commit `5e66641`
`scripts/iatc_anchor_faithfulness.bb` + tests. Checked: clj-kondo 0/0; bb tests 10/10;
rates reproduce (min 0.333 / max 1.000); ground-truth anchors correct — `0706.1286`
0.857 (HIGH), `0709.0248` flags `:extensional-category`@1510 (`\begin{proposition}`,
0/5 terms). K=2, τ=0.45, floor 0.300 from the spread.
- **Caveat — R2a rate is a conservative *lower bound* (lexical-only).** It over-flags
  math/macro-dense nodes: e.g. `0801.3843` `:G "topological group G"`@L646 is flagged
  for missing "group", but the group is on the line as the macro `\G`; `:H "group H"`
  reduces to 1 scoreable term (single letters dropped) so it auto-fails k=2; `:crossed-
  module`@L658 misses "crossed module" which sits on L657 (off-by-one). So low rates
  ≠ true non-faithfulness — but it *does* also catch real imprecision.
- **R2a-v2 (next iteration, follow-up handoff):** (a) light LaTeX normalization
  (`\maps`→maps, `\Ob`→Ob, `\G`→G, …) before matching; (b) ±1 neighbor-line tolerance;
  (c) per the N/A≠FAIL stance, nodes with <K scoreable terms → **N/A**, not flagged.
  These de-noise the rate to reflect genuine faithfulness. Harness should label the R2a
  rate "lexical lower bound" until v2 lands.

### H-R2-harness findings — codex-4

Implemented `scripts/iatc_semcheck.bb` as the rung-2 description-first
aggregator. It loads the reviewed R2a and R2b/R2c checkers and calls their
`check-graph` functions directly; it does not reimplement their logic. The
script emits a per-graph paper-description profile, per-check semantic report,
optional EDN/Markdown reports, and a `--gate` mode that exits nonzero on
present-but-wrong structure. Finals-only is the default; `--include-attempts`
is the explicit opt-in.

Design decisions:

- **N/A ≠ FAIL:** when a graph has no nodes or no edges at the current
  resolution, the corresponding check is normalized to `:status :na` and does
  not fail the aggregate gate.
- **R2a label:** reports call the R2a rate a **lexical lower bound** because
  the current checker is intentionally lexical-only and over-flags symbol/macro
  dense spans.
- **R2c floor:** the harness default is `--warrant-floor 0.0`, justified by the
  observed loop-run-70b final spread including `0.0` and aggregate `6/28`; this
  keeps R2c report-only until a stricter floor is calibrated.

Evidence committed at
`holes/excursions/iatc-semcheck-loop-run-70b.{edn,md}` covers the 9 finals plus
the specified `0708.2185.attempt2.edn` anchor. It distinguishes the ground-truth
anchors:

- `0706.1286`: clean/high R2a lexical lower bound (`0.857`) and R2b pass.
- `0709.0248`: R2a-flagged proposition-anchor case (`4` lexical reasons).
- `0708.2185`: R2b-flagged self-loop in the attempt graph.

Default finals-only smoke:
`bb scripts/iatc_semcheck.bb data/iatc-argument-graphs/loop-run-70b` reports
9 graphs, 4 failing graphs, and overall `FAIL`; `--gate` exits `1` on that
run. The evidence run reports 10 graphs, 5 failing graphs, and overall `FAIL`.

Gates passed: `bb tests/iatc_semcheck_test.clj`, `clj-kondo --lint
scripts/iatc_semcheck.bb`, `clj-kondo --lint scripts/iatc_closure_check.bb`,
`clj-kondo --lint tests/iatc_semcheck_test.clj`, and
`futon4/dev/check-parens.el` on the touched `.bb`/test files.

### H-R2b+c findings — codex-3

Implemented `scripts/iatc_closure_check.bb` with two contract-shaped checks:
`:closure` and `:warrant-resolution`. The warrant-resolution check delegates its
edge counts to `scripts/mark3_eval_harness.py` so the missing-vs-real warrant
definition is not forked.

Acceptance run:

```sh
bb scripts/iatc_closure_check.bb \
  data/iatc-argument-graphs/loop-run-70b \
  data/iatc-argument-graphs/loop-run-70b/.attempts/0708.2185.attempt2.edn
```

Results:

- R2b flags `0708.2185.attempt2.edn` as required:
  self-loop at `:card-I-less-than-lambda` via edge
  `:e-card-I-less-than-lambda`.
- R2b also flags existing final graphs with orphan/cycle structure:
  `0705.0452`, `0708.1921`, `0708.2067`, and `0712.0724`.
- R2c matches `mark3_eval_harness` on the nine finals:
  aggregate `6/28 = 0.214` real warrants.
- Per-final R2c rates:
  `0705.0452 0/3`, `0706.1286 1/5`, `0708.1921 0/3`,
  `0708.2067 0/2`, `0709.0248 1/3`, `0711.0473 0/2`,
  `0712.0724 0/3`, `0801.0199 3/5`, `0801.3843 1/2`.
- The observed final spread includes zeros, so the default deterministic floor
  is `0.0`; callers can raise it with `--warrant-floor`.

Gates passed: `clj-kondo --lint scripts/iatc_closure_check.bb`,
`emacs --batch -l /home/joe/code/futon4/dev/check-parens.el
scripts/iatc_closure_check.bb`, and `bb tests/iatc_closure_check_test.clj`.

### H-R2a findings — codex-1

Implemented `scripts/iatc_anchor_faithfulness.bb` with the shared
`check-graph` contract:
`{:check :anchor-faithfulness :pass :rate :reasons :per-item}`. The checker
resolves each graph to `data/showcases/ct-anatomy/golden/fable-<id>-dp-emacs.json`,
splits the `"text"` field into 1-based lines, extracts node key terms, and tests
the cited line span for those terms. CLI overrides: `--marks-dir`, `--source`,
`--k`, `--tau`, `--floor`, and `--edn`.

Chosen thresholds: `K=2`, `tau=0.45`. The observed final-graph spread is
`0.333..1.000`, so the default gate floor is `0.300`.

Acceptance run:

```sh
bb scripts/iatc_anchor_faithfulness.bb data/iatc-argument-graphs/loop-run-70b
```

Per-final rates:
`0705.0452 0.500`, `0706.1286 0.857`, `0708.1921 1.000`,
`0708.2067 0.625`, `0709.0248 0.333`, `0711.0473 0.667`,
`0712.0724 0.667`, `0801.0199 0.833`, `0801.3843 0.417`.

Required spot checks:

- `0706.1286` scores high: `0.857`.
- `0709.0248` flags `:extensional-category` at `[1510 1510]`:
  line 1510 is only `\begin{proposition}`, so it matches `0/5` key terms
  and misses `locally`, `cartesian`, `closed`, `category`, `extensional`.

Remaining gaps:

- This is a deliberately lexical anchor check. It does not expand author macros,
  resolve synonyms, or use neighboring theorem/proof lines when the node's cited
  span is too narrow.
- Several low-rate finals appear to reflect over-specific generated node anchors
  rather than necessarily bad paper understanding; the checker reports those as
  flags instead of silently repairing spans.

Gates passed: `clj-kondo --lint scripts/iatc_anchor_faithfulness.bb
tests/iatc_anchor_faithfulness_test.clj`, `emacs -Q --batch -l
/home/joe/code/futon4/dev/check-parens.el --eval "(arxana-check-parens-cli)"
-- --no-defaults scripts/iatc_anchor_faithfulness.bb`, and
`bb tests/iatc_anchor_faithfulness_test.clj` (`4` tests, `10` assertions).

### R2a-v2 findings — codex-1

De-noised `scripts/iatc_anchor_faithfulness.bb` without changing the public
`check-graph` contract. Term matching now normalizes light LaTeX on both
node/source text (`\maps`, `\Ob`, `\Mor`, `\G`, math/text wrapper macros,
styled alphabets, arrows), searches cited spans with ±1 neighbor-line tolerance,
and marks nodes with fewer than `K=2` scoreable terms as `:status :na` instead
of scoring them as failures.

To avoid over-relaxing proposition anchors, the v2 matcher keeps `tau=0.45` as
the decisive criterion for longer claims. This preserves the genuine
`0709.0248` miss: `:extensional-category` at `[1510 1510]` remains flagged even
with ±1 lines because only `locally/cartesian` are reached and the span still
misses `closed/category/extensional`.

Before/after on the 9 `loop-run-70b` finals:

| paper | before rate | before flags | after rate | after flags |
|---|---:|---:|---:|---:|
| 0705.0452 | 0.500 | 5 | 0.500 | 4 |
| 0706.1286 | 0.857 | 1 | 1.000 | 0 |
| 0708.1921 | 1.000 | 0 | 1.000 | 0 |
| 0708.2067 | 0.625 | 3 | 0.714 | 2 |
| 0709.0248 | 0.333 | 4 | 0.667 | 2 |
| 0711.0473 | 0.667 | 2 | 0.833 | 1 |
| 0712.0724 | 0.667 | 4 | 1.000 | 0 |
| 0801.0199 | 0.833 | 1 | 1.000 | 0 |
| 0801.3843 | 0.417 | 7 | 0.857 | 1 |

Spread improved from `0.333..1.000` to `0.500..1.000`; total flags dropped from
`27` to `10`. The remaining flags include the intended genuine miss
`0709.0248/:extensional-category`.

Gates passed: `clj-kondo --lint scripts/iatc_anchor_faithfulness.bb
tests/iatc_anchor_faithfulness_test.clj`, `emacs -Q --batch -l
/home/joe/code/futon4/dev/check-parens.el --eval "(arxana-check-parens-cli)"
-- --no-defaults scripts/iatc_anchor_faithfulness.bb`,
`bb tests/iatc_anchor_faithfulness_test.clj` (`6` tests, `18` assertions),
`bb tests/iatc_semcheck_test.clj`, and `bb tests/iatc_closure_check_test.clj`.
