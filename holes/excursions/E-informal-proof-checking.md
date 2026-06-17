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

## Open questions — the cascade→sorry→wiring reframe (Joe, 2026-06-17)

Surfaced connecting this excursion to the `render_run` rendering work and the
**`cascade → sorry → wiring`** fold (`early-closures.md`; the four stages
`discharge`/`cascadeFeed`/`answer`/`compose`). The reframe: the rung ladder reads
as a **per-proof fold** rather than a linear pipeline —

- **paper = a sorry** (hole-topology: claims without supports, terms without
  definitions, dangling nodes — the 4/9 incomplete graphs are under-wired sorries);
- **checks = the cascade menu** (R2a anchor, R2b closure, rung-−1 term-coverage, and
  proof-shape-specific checks: induction-schema, diagram-chase commutativity, …);
- **`select` = match the menu to *this* proof's topology** → *each proof gets its own
  cascade of checks*, which a fixed `{R2a,R2b,R2c}`-applied-uniformly cannot do;
- **wiring = the filled model** (every claim wired to a support, every term to a
  definition-or-known-concept); **residual sorries = the "where we are least sure" map**.
- Target is an **IR for the LLM**, not only a human view: `render_run` and the IR are
  the *same partially-wired graph* under two projections (human-legible vs serialized);
  "gate-is-a-describer / N/A≠FAIL" = an unfilled port is N/A, a FAIL is a *mis-wire*.
- Output beyond the per-paper profile: a **conformance certificate** ("this model is a
  well-formed wiring — all claim/term ports filled") + the residual-sorry uncertainty map.

**These are recorded as questions — not yet answered (Joe):**

- **Q1 — where does the cascade *menu* live?** First-class `.flexiarg` patterns in the
  library (so `select` is pattern-matching the proof's sorry-topology, reusing the
  existing pattern machinery, and the menu is growable), **or** a checker registry
  extending the rung-2 `check-graph` contract (closer to what's already built)? Open.
- **Q2 — where does `select` get the topology to match against?** Presumably the coarsest
  pass — the document skeleton + which concepts/claims exist (the first `render_run`
  frame). If so the build-up *order* is load-bearing: **coarse frame → select cascade →
  fill → re-render**. Is that the sequencing? Open.

**claude-1 leanings (proposed — Joe to ratify; recorded, not closing the questions):**
- *Q1 — split the two concerns.* The **checks** are a **registry** of `check-graph`-shaped
  functions (extends the built rung-2 contract — executable, deterministic, tested,
  composable); the **select rule** is **`.flexiarg` patterns** matching the proof's
  sorry-topology → a check set. Registry for *what executes*, `.flexiarg` for the
  *growable shape→cascade* mapping. This avoids forcing executable `bb` checks into
  patterns, and avoids hard-coding selection in code — each is used for what it's good at.
- *Q2 — yes, the coarse frame, and it's load-bearing.* `select` reads the coarse pass
  (skeleton + which concepts/claims/proof-shapes exist) — which is exactly the
  **resolution axis's first artifact** (the "describer / N/A≠FAIL" stance). So the coarse
  pass isn't only a cheap description; it's the **controller**: coarse frame → select
  cascade → fill → re-render. The resolution axis and the control flow are the same spine.
- *Connection:* this is the live futon `sorry = wiring-diagram-with-a-gap` vocabulary, not
  a new metaphor — `select`-per-topology is what makes "each proof its own cascade"
  precise, and the residual-sorry map is the uncertainty output rung-3 (the judge) would
  then prioritize.

**Seed inventory — candidate menu *content* (claude-loop; Joe-ratified as a candidate seed, 2026-06-17).**
Complements claude-1's Q1 split (which gives the menu's *structure*: registry = what executes,
`.flexiarg` = the growable shape→cascade map) by answering *where the patterns come from*. The
recursion (a paper's cascade inherits the patterns of its imports/citations) must bottom out in a
hand+mined seed — and it already largely exists on disk:

- **Pólya heuristics** — canonical, **to author** as `.flexiarg` (not in-repo yet; confirmed by
  search). "Work backwards" = reverse morphogenesis itself; "a related problem?" = CONNECTION_SEEKING;
  specialize/generalize = GENERALIZATION_PROBE. The hand-written ceiling of the empirical set.
- **The RM question-pattern survey (prior work, on disk)** — 12 content-facing + 8 process-facing
  named patterns with MO frequency + First-Proof productivity:
  `holes/handoffs/question-asking-pattern-mining-from-mo-rm-2026-03-06.md`. Corpora live: MO
  `storage/mo-processed-gpu/reverse-morphogenesis.json` (95K entries), math.SE
  `storage/math-processed-gpu/reverse-morphogenesis.json` (805K). Survey done; the full
  embed→cluster→name pass (`data/question-patterns/`) was **not** run.
- **The close-reading expository taxonomy** (`expository-superpod-vocab.edn`, 16 typed-hole scopes)
  \+ the **25 math-informal proof patterns** + `futon3/library/futon-theory/reverse-morphogenesis.flexiarg`.

**Name-level convergence (evidence the menu is real, not a method artifact):** the MO content
patterns coincide with the expository taxonomy almost name-for-name — CONNECTION_SEEKING ↔
`connection/bridge-analogy`, OBSTRUCTION_IDENTIFICATION ↔ `obstruction`, EXISTENCE_WONDER/CONJECTURE
↔ `open-problem/status`, the "why" ↔ `rationale/telos`. MO-question-mining, expository, and IATC are
three views of one menu.

**The genuinely new build = the IATC+expository → pattern compiler** — which is the *reused RM
mining method* (embed situations/moves → cluster → name → cross-validate vs First-Proof productivity)
pointed at the IATC graphs + expository moves instead of MO situations. Method specced + survey-proven;
only the input substrate changes. The other new build is the genealogical `select` (citation/import-
descent indexing of which seed patterns each paper inherits — `warp_citations/bib` is the descent graph).

**Prior art — the same cascade, already prototyped over the *stack itself* (Joe).**
`futon3c/holes/excursions/pipeline-pattern-cascade.html` (a Moran-style control sketch) runs this
exact apparatus over futon's own **missions + `futon3/library` patterns**: mission clusters = basins,
**cited patterns *warrant* basins**, hollow nodes = **missing patterns/scans** (= residual sorries),
joined by **strong vs speculative/missing edges**. Three transfers are direct:

- its **inductive attachment rule** — *start from attested nodes → attach the next basin by concrete
  cited-pattern overlap (before embedding proximity) → split a basin when it is mixed* — is exactly
  the **seed bottoming-out + growth of `select`** (attested = the seed; overlap = citation/import
  descent; split = per-proof refinement);
- its "because" panel states our thesis verbatim — *"holes mark where the cascade needs an **adviser**,
  not a static roadmap"* — i.e. **per-proof-assembled, not a fixed pipeline**;
- its forward-model **pattern-seeding loop** — *repeated held-sorry shapes seed real library patterns
  rather than remaining one-off exceptions* (Pudding held sorries → future cards) — is the
  **mining-from-residue loop** in stack form, and the source of new menu patterns.

It even carries the projection point: the **piano-roll trace is "a time projection *as a view over the
semilattice*, not the ontology itself"** — the same claim as `render_run` being a *projection* of the
partial wiring. **Swap (missions, library-patterns, capabilities) → (papers/proofs, reasoning-patterns,
certified-properties) and the whole apparatus transfers.** So the cascade reframe is not new
machinery — it is this stack-level prototype pointed at proofs.

## Sequencing — producer vs checker, and the concept-first reordering (Joe, 2026-06-17)

How `proofcheck-readiness.html`'s items sequence relative to the previously-prototyped
`pre-superpod-pipeline-readiness.html` (①–⑦). The initial intuition — a machine-facing `render_run`
*between each superpod phase* — dissolved once the workflow swung **concept-first**.

**The two maps are producer vs checker, not two halves of one line.**
- **`pre-superpod` (①–⑦) is the *substrate producer*, and it's largely done.** ① anatomy + ④ IATC + ⑤
  expository already ran (GOLIVE: 9/10 70B graphs + the mined patterns); it collapses to one card
  (`GOLIVE`) in the new map — upstream input, not the spine.
- **`proofcheck` is the *checker* — the new spine** (the rung ladder −1 → 3), which consumes the
  producer's output.

**What concept-first reorders.** The old pipeline had **② Concept substrate as a weak middle stage**
(precision 0.108, stage 2 of 7). Concept-first **promotes it to the foundation and splits it into three
rungs that come *before* the reasoning checks bite**: rung −1 (term defined? `SFC1` ✓) → **:structure lift**
(`SFC2a` ✓ / `SFC2a-v2`) → **symbol grounding** (`SFC2b` = M-symbol-grounding). The **join is `R2d`** (do a
proof's concepts have definitions in the substrate?) — *unbuildable until the concept foundation exists*.
That is the concrete dependency the swing creates: build out the old ② first, because R2d and the cascade
both consume it.

**Integrated order:**
1. *Concept foundation* (CPU, cheap, mostly specced): `SFC1` ✓ → `SFC-D3` / `SFC-NORM` →
   `SFC2a` ✓ / `SFC2a-v2` → `SFC-AGG` (needs schema); `WARP-ORCH` (wire the live-on-disk layer).
2. *Symbol grounding* (LLM, bounded): `SFC2b`.
3. *Proof-checking rungs 0–2* (CPU, **already built**) — run NOW on existing graphs; wire `R2-wire`,
   de-noise `R2a-v2`, calibrate `R2c`; **then `R2d`** ← gated on (1).
4. *Cascade + residue* (last): settle Q1/Q2 → `CAS-SEL` + `CAS-CERT` → rung-3 judge.

**The non-obvious bit:** rungs 0–2 are structural, need no concepts, and **can grade today** — but the
*semantically interesting* checks (`R2d`, the cascade `select`) sit **behind** the concept foundation.
"Concept-first" ≠ "do concepts before you can check anything"; it = the cheap structural checks run now,
while the semantic checking is gated on the substrate, so build that first.

**It resolves the earlier ④/⑤-vs-① sequencing question.** The concept substrate (rung −1) is the
foundation; reasoning (④/⑤) is gated on it (reconstruct arguments only over grounded concepts); checking is
downstream; **`render_run` is a *projection*, not a phase** — which is why the new map demotes `RENDER` to
deferred. The machine-facing IR first imagined *between* phases turned out to be the **checker's cumulative
description profile** (the gate-is-a-describer output); `render_run` is a human view of that, not the spine.

## CAS-0 — empirical pattern-pool seeding (worked examples · claude-1, CPU, no GPU)

Joe's call (2026-06-17): settle the cascade's open questions **empirically, not by fiat** —
you cannot match on patterns you do not have, and the pool is thin (36 math-informal, the
RM corpora un-mined). So work real **APM proofs** end-to-end — induce each one's **sorry +
wiring**, find/write the patterns its steps need — and let Q3/Q4/Q5 *emerge*. This section
is the running log; full write-ups under `cas0-worked-<id>.md`.

### Checkpoint #1 — `apm-a93J05` (doubly-periodic entire ⇒ constant) · `cas0-worked-a93J05.md`
A "reduce-to-known-theorem"–shaped proof (Liouville). **5/5 steps matched the existing 36
math-informal patterns — zero new patterns needed** (construct-auxiliary-object →
reduce-to-known-result(EVT)+estimate-by-bounding → quotient-by-irrelevance →
local-to-global → reduce-to-known-result(Liouville)). Matches verified against the patterns'
text, not forced.

**The mechanism this surfaced** (the answer to "induce a sorry + a wiring"):
- the matched patterns' **conclusions chain into the wiring** (the argument DAG); and
- **the residual sorries are exactly the matched patterns' `HOWEVER` clauses left
  undischarged** — each `.flexiarg` names its own proof obligation, the informal proof
  asserts the conclusion and skips it. (`quotient-by-irrelevance`'s "verify well-defined on
  equivalence classes" → the sorry "ω₁,ω₂ tile ℂ"; `local-to-global`'s "verify the pieces
  patch" → "f(z)=f(z₀)".) So pattern-matching is a **principled sorry generator**, not a
  hand-curated hole list.

**What it answers empirically** (recorded against the open questions above):
- **Q3 (topology vocabulary):** a proof's topology *is* its sequence of matched patterns —
  no separate hand-authored taxonomy. `select` = which patterns match the steps.
- **Q5 (deterministic vs judge):** match + HOWEVER-readout is mostly deterministic; the only
  judgement is "does this step really instantiate this pattern?" (a bounded verify spot).
- **Refinement, not a gap:** `reduce-to-known-result` fired twice with different cited
  theorems (EVT, Liouville) → the named theorem is a **slot** (`:cites`), not a new pattern.
- **Pool signal:** 1 proof, 0 new patterns — but a canonical shape the 36 cover well; the
  gaps will only show on a *different* shape (construction / induction / diagram-chase).

### Checkpoint #2 — `apm-a96J01` (uniformly-convergent series, sup-norms diverge) · `cas0-worked-a96J01.md`
A construction/existence proof — chosen to stress the pool where #1 (reduce-to-known) didn't.
**It did: 4/5 steps matched, and the 5th yielded the first new pattern.** The load-bearing
move — *disjoint supports* so the bumps don't interfere — is captured by none of the 36
(`exploit-symmetry` is WLOG; `quotient-by-irrelevance` collapses; `local-to-global` *patches
overlapping* pieces — this proof is its **dual**). So I wrote
`math-informal/separate-into-independent-pieces` (`[✂️/分]`, registered in
`resources/sigils/patterns-index.tsv`): engineer disjoint/independent support → cross-terms
vanish → a global aggregate property collapses to a per-piece one (disjoint bumps, orthogonal
vectors, independent RVs, disjoint cycles).

**Confirms across shapes:**
- The **mechanism is shape-independent** — #1 (reduce-to-known) and #2 (construction) both
  induce wiring (patterns' conclusions) + sorry (patterns' undischarged `HOWEVER`s) the same
  way. (Same locus claude-loop's rung-3 section below names: a "thin" leaf = an undischarged
  HOWEVER — the sorry generator and the rung-3 thinness signal are the *same* mechanism.)
- **Q4 (static vs adaptive):** the pool **grows demand-driven from worked proofs** (#1 +0,
  #2 +1) — the prior-art "pattern-seeding loop" is real, and seeding is per-proof, not a bulk
  mining pass. We wrote the *one* pattern this proof needed, not a speculative taxonomy.
- `construct-an-explicit-witness` + `construct-auxiliary-object` co-fire ⇒ `select` matches
  pattern **families with parameters** (cf. #1's `reduce-to-known` `:cites` slot), not atoms.

### Checkpoint #3 — `apm-b97J01` (finite p-groups: nontrivial center & nilpotent) · `cas0-worked-b97J01.md`
A hard, multi-part **induction** probe — chosen to answer the question we flagged: does the
induction *schema* need a finer hole than the one coarse pattern? **Two findings:**
- **Finding A (the answer): no finer hole.** `induction-and-well-ordering` covered part (d)
  cleanly (induct on order; the footnote does it as plain strong induction; the
  strictly-increasing-chain-in-finite-`G` = the well-founded-termination instance). The
  "base case *is* the inductive engine" framing is a nice *instance*, not a missing schema.
  A negative result that argues against pre-building the menu by fiat — let proofs demand it.
- **Finding B (second new pattern): counting.** Part (c)'s labelled "Key Insight" — the
  class-equation divisibility (`pⁿ = |Z(G)| + Σ[G:C_G(xᵢ)]`, every non-central term `≡0 mod
  p` ⇒ `p | |Z(G)|`) — is a **counting argument**, and the 36 had **no** counting pattern
  (no double-counting/pigeonhole/orbit-counting). Wrote
  `math-informal/count-over-a-decomposition` (`[🧮/数]`, registered): split a quantity over a
  decomposition, control all-but-one part with a shared congruence/bound/vanishing, read off
  the residual (class equation, Burnside, Cauchy/Sylow, inclusion–exclusion, pigeonhole).

**Running tally (3 proofs):** mechanism **3/3 shape-independent** (reduce-to-known /
construction / induction+counting) — wiring = patterns' conclusions, sorry = patterns'
undischarged HOWEVERs, every time. **Pool +0,+1,+1 → 38** math-informal. New observation:
(d) cites the proof's *own* part (c) as the "known result" ⇒ `reduce-to-known-result`'s
`:cites` slot spans **internal lemmas**, not just external theorems.

**CAS-SEL-1 is now writable on these worked examples** — the step→pattern→(wiring,sorry)
procedure is concrete and repeatable across 3 shapes. Recommend one more shape
(diagram-chase or ε-δ analysis), then write the spec spike *on* the corpus.

### Checkpoint #4 — `apm-a96J04` (AC monotone maps null sets to null sets) · `cas0-worked-a96J04.md`
The one-more-shape: an **ε-δ / measure-theory** proof, a region #1–#3 never touched. **Third
new pattern, plus a structural read on the pool.**
- **New pattern: `epsilon-of-room`** (`[🤏/微]`, "Give Yourself an Epsilon of Room",
  registered). The closer "`m*(f(E)) ≤ ε` for every ε, let ε→0 ⇒ `=0`" — the central
  manoeuvre of analysis — had no home in the 36 (`estimate-by-bounding` produces *a* bound;
  `optimise-a-free-parameter` *tunes* ε; this *sends ε→0*). Rest of the proof = `unfold-the-
  definition` ×3 (AC / monotone / null-set) + `estimate-by-bounding` (subadditivity).
- **Finding A — analysis was the under-covered region, and the discovery rate hasn't
  saturated.** Tally **+0,+1,+1,+1 → 39**. #1 was 0-new *because it reduced to a named theorem*
  (Liouville) — the 36 cover "which big theorem to cite" but skew toward **strategy**
  (reduce-to-known / induct / contradict / construct) and were thin on **execution idioms**
  (make-pieces-independent, count-over-a-decomposition, epsilon-of-room — all minted here).
  The pool is **not saturated** at the analysis frontier.
- **Finding B — but the *mechanism* is saturated (4/4 shapes), so CAS-SEL-1 is specifiable
  now.** The step→pattern→(wiring,sorry) procedure is identical across all four; the spec does
  **not** depend on pool completeness (the pool grows demand-driven during RUN — the seeding
  loop). **Recommend: write CAS-SEL-1's spec spike on the 4-proof corpus; let proofs keep
  minting patterns.**

**Corpus:** a93J05 (reduce-to-known) · a96J01 (construction) · b97J01 (induction+counting) ·
a96J04 (ε-δ measure). New patterns: `separate-into-independent-pieces`,
`count-over-a-decomposition`, `epsilon-of-room`.

## rung-3 = technique-grounding — the verb-twin of R2d; "thin" = an undischarged HOWEVER at a heuristic leaf (Joe + claude-loop, 2026-06-17)

CAS-0's finding — **residual sorry = the matched pattern's undischarged `HOWEVER`** — is, read from the
proof-checking side, exactly **rung-3**. Two reframes follow, and they relocate rung-3.

**rung-3 is the verb-twin of R2d.** R2d grounds the *nouns* (is each term defined / known?); rung-3 grounds
the *verbs* (is each move a known technique, or a gap?). Both are **detectors of gaps, not arbiters of
truth** — rung-3 never asks "is this step true" (that *is* the mathematics); it asks "is this step grounded
in a recognized technique, or thin?". So rung-3 = **technique-pattern-coverage of the moves**, the way R2d is
concept-coverage of the terms. (Buckets parallel R2d's defined/known/imported/undefined:
grounded-by-pattern / grounded-by-citation / thin / ungrounded.)

**Not every undischarged HOWEVER is equal — the heuristic/verifiable split is what makes a cascade a
*mathematical* one.** Technique patterns have a type:
- **heuristic** — justifies a *strategy* ("reduce to an easier case", "consider the generic point"). Guides
  discovery; does **not** justify a step. Pólya-grade, RM-question-grade.
- **verifiable** — justifies an *inference* ("by [theorem] whose hypotheses hold", a computation, a def).

The *same* pattern can be either (`reduce-to-known-result` is a heuristic as a *strategy*, verifiable once
the reduction map is exhibited). **A cascade chains heuristics but must bottom out in verifiable leaves.** So
**"thin"** is sharp: a load-bearing step whose HOWEVER is discharged only at the *heuristic* level where a
*verifiable* step is required, the verifiable step not exhibited ("this is hard, so pick an easier example"
used *as if it were the proof*). That is what "more work to even explain the technique" means — exhibiting the
verifiable step the heuristic gestured at. **This sharpens "pattern cascade in mathematics" against a generic
one: the leaves must be verifiable, or there's a gap there.**

**The sorry typology (in the wiring vocabulary):**
- **verifiable step** = a *filled* sorry;
- **conjecture** = an *author-declared, acknowledged-unfilled* sorry — **credit it** (the author was honest);
  a corpus-wide **map of stated open problems + their dependents** is a first-class output, not a failure
  (ties to the expository `open-problem/status` move);
- **thin step** = an *undeclared* unfilled sorry presented as filled — **the detection target**.

**The detector asks, it does not answer — and that is where the two halves of the menu meet.** A detected gap
becomes an **ArSE question** ("how does the general case follow from the example here?"), and the **RM
question-pattern menu** (EXISTENCE_WONDER, STRUCTURAL PROBE, …) is the vocabulary for phrasing it by gap-type.
The proof-patterns (verbs) *find* the gap; the question-patterns *phrase* it. "Likely gap"/"likely hole" is
the honest label; an ArSE question that recurs across papers is a research frontier; one that gets answered
mints a new **verifiable** pattern (the pattern-seeding loop, now typed heuristic-vs-verifiable).

**"How much LLM" is empirical (extends CAS-0 Q5).** CAS-0 found match + HOWEVER-readout mostly deterministic,
the only judgement being "does this step instantiate this pattern?". So don't fix the LLM fraction a priori —
**measure the deterministic residue** on real moves: that residue *is* the LLM's share. And don't be precious
about the LLM on the residue (patterns→design works with an LLM, harder deterministically — futon5
experiments) **because the output is a question, not a verdict**, and conjectures give partial ground truth
(authors already flag many gaps).

**Relocation:** rung-3 therefore **moves from Phase C (the rung ladder) to Phase D (the cascade)** — it is the
**per-edge instance of `select`** (which technique pattern fits this move, what is the residual?), it
*depends on* the menu and *feeds* it, and its residue is the gap. No longer "the last rung"; the edge-grain of
the cascade. Breakdown: `holes/handoffs/rung-3-breakdown.md` (rewritten to this framing).

**The grounding ladder — SFC2b ⊂ R2d ⊂ rung-3 (Joe, 2026-06-17).** Symbol grounding (SFC2b) is the **base
case** of the *same* detector at the finest grain: bind each **symbol** to its domain/codomain, the way R2d
binds **terms** to definitions and rung-3 binds **moves** to techniques. One check — *grounding coverage* — at
three grains (symbol / concept / technique), all detectors (grounded / known / undefined → flag → ArSE
question, N/A ≠ FAIL). **Gaps propagate up the grain:** `f(x,y)` dropped into a paper with no further comment
fails *symbol* grounding (`x, y` untyped; `f`'s domain/codomain unknown — SFC2a's `:grounding :hole`s
unfilled), and *therefore* fails *concept* coverage (you cannot say what the concept `f` even **is** without
its domain/codomain — ℝ→ℝ vs a functor are different concepts), and *therefore* no step using it is
technique-checkable. So the foundational order **is** the grain order — ground symbols before concepts before
techniques (why the spine runs rung −1, symbol-grounding included, before rung 2/3). "Symbol grounding is the
simple case" = it is the **base case, not the easy one**: binding one symbol to its local domain is a small
bounded LLM task, but the residue (the LLM's share) grows with grain. And the nuance: grounding is about
**types**, not mention — "`x` appears on line 5" isn't grounding; "`x ∈ X`, the 𝒱-category" is. That is why
SFC2a emits `:grounding :hole` and SFC2b *fills* it; an unfilled hole **is** the uncommented `f(x,y)`.

## CP — the discursive core (Joe + claude-loop, 2026-06-17; claude-1 converging from CAS-0)

The whole design has resolved into one shape. **Every check is a question, and the paper is a discourse.**
The grounding ladder (SFC2b symbol / R2d concept / rung-3 technique) and the cascade are all asking the same
*kind* of thing — *what is x? · is this term defined? · does this move instantiate a technique?* — and a
proof's meaning is precisely **the structure of questions it can answer** (the dialogical / BHK reading; it is
also why there is no static gold markup to check against — the parent question's deeper answer). The
detectors are **question-askers**; the substrate (concept encyclopedia, def-snippets, IATC graphs, the
pattern menu) is the **answerable database**; ArSE is the **discourse medium**.

**What stops this from being endless AI chat — and it is load-bearing — is that *scopes turn a question into
a graph query* (Joe).** We have been using scopes throughout (the `scope` mark layer, APM ⑥ scope-match, the
scope-kind hierarchy), and that is exactly the bridge:
- **termination** — a scoped question (lookup `x`'s binding in *this* scope) *returns or doesn't*; it does not
  recurse into more chat;
- **grounding** — the answer comes from the database, not the model's imagination;
- **a determinate gap** — an *empty* query is a real open question, not an LLM "I'm not sure" — the honest
  residual sorry.

So the **LLM is invoked only on an empty scoped query**, and — the convergence point — **its answer becomes a
new binding in a scope** (a query result that folds back into the database). That is why the discourse
*converges* rather than chats forever: every answered question grows the queryable substrate and shrinks the
frontier; endless-chat is what you get when answers evaporate, and scopes make answers **persist as data**.

**The sharpening:** the grain ladder **is** a ladder of scopes — a symbol's binding-scope ⊂ a concept's
definition-scope ⊂ a technique's pattern-scope ⊂ the proof scope — so grounding-at-grain-*g* *is*
querying-the-scope-at-grain-*g*, the three detectors are **one query operation over three nested scopes**, and
the `f(x,y)` propagation is just scope nesting (an empty inner-scope query poisons the outer). Scopes are not
new machinery; a scope already *is* a typed region with bindings, i.e. a mini-database. We have been building
the query substrate the whole time; the discursive reading names what it is *for*.

**Net:** detectors = questions · substrate = the answerable database · **scopes = the query bridge (the
saving grace)** · ArSE = the discourse medium · LLM = residue-only, and its answers persist as scope bindings.
That two routes reach it — CAS-0's worked proofs and the grounding-ladder — is evidence it is the design's
actual core, not a framing imposed on it.

**CAS-0 evidence for the discursive core (claude-1).** The four worked proofs + the CAS-SEL-1 spec
instantiate this shape concretely — and not by analogy, with a falsifiable test:

- **"Every check is a question" is literal — the question *is* the HOWEVER.** CAS-0's finding was
  sorry = a matched pattern's undischarged `HOWEVER`. A pattern's HOWEVER is phrased as exactly the
  question the check asks: `quotient-by-irrelevance` → *"is the quotient well-defined?"*,
  `local-to-global` → *"do the pieces patch?"*, `count-over-a-decomposition` → *"is the decomposition
  exhaustive?"*, `epsilon-of-room` → *"does the bound hold for every ε?"*. A proof's sorry list = its
  open questions = "the structure of questions it can(not yet) answer."
- **Tier-0 retrieve *is* "scope → graph query."** Hotword lookup over the pattern pool (the answerable
  database) returns a match or doesn't — termination + grounding, no recursion into chat. The pattern
  pool is the technique-scope; the match is the binding.
- **"Empty query = a real open question, not LLM-unsure" — now a falsifiable test.** CAS-SEL-3
  acceptance #2: simulate the *pre-mint* 36-pattern pool and the empty-query (`induce_queue`) set must
  equal **exactly** the three steps where the pool genuinely lacked a pattern (disjoint-support,
  class-equation-divisibility, ε-arbitrary) — and nothing else. An empty scoped query reproduces a real
  gap, deterministically; it is not the model hedging. That test *is* the determinate-gap claim.
- **"LLM only on empty queries; its answer becomes a binding; the discourse converges" = the three-tier
  cost gradient.** Tier-2 induce fires only on `:none`, and its output (a new `.flexiarg`) is a new
  technique-scope binding that grows the pool. Convergence is *measured*: the induce/discovery rate was
  **0,1,1,1 and falling** — every answered question shrank the frontier. (And the gate — author ≠
  reviewer before a binding enters the pool — is what keeps the persisted answers trustworthy.)
- **The grain ladder = nested scopes, with the `:cites` slot as the nesting operator.** A technique
  pattern's slot queries *down*: `reduce-to-known-result`'s `:cites` slot in a93J05 resolves to a
  named theorem (concept scope); in **b97J01 it resolved to the proof's *own* part (c)** — an internal
  scope binding. So the slot is literally the `f(x,y)` scope-nesting (a technique query answered by a
  concept/lemma query), and an unresolved slot poisons the outer step's sorry — exactly the propagation
  claimed.

**The certificate IS the substrate made per-paper — and its display is a reader's guide (Joe, 2026-06-17).**
The answerable substrate (profile + scope bindings + sorry/gap map), taken **per paper**, *is* the
conformance certificate (**CAS-CERT**) — not a separate artifact. Two faces of one object:
- **machine (CAS-CERT):** `{conformance: are the symbol/term/technique ports grounded? · coverage-by-grain ·
  residual-sorries: [the open questions] · value-signals}` — feeds rung-3 + the loop;
- **human (the guide):** the same, *displayed* as a per-paper reader's orientation —
  - **read this first** — what it is about: imported concepts + region skeleton (noun + structure layers);
  - **start here** — the entry point: the main claim + dependency order (the IATC DAG roots→goal + the import
    descent — read these lemmas / cited results first);
  - **open questions we couldn't figure out yet** — the gaps: thin steps (rung-3) · undefined terms (R2d) ·
    ungrounded symbols (SFC2b) · the author's own conjectures — i.e. the residual-sorry map / ArSE questions;
  - **why it's likely valuable** — centrality (concept pagerank) · novelty (introduces vs only applies) · the
    connections it makes (the `connection/bridge-analogy` moves) · the conjectures it raises · % grounded.

So **RENDER displays CAS-CERT**: render_run / DEMO-COMPOSE are the *surface*, the certificate is the
*content*, and "two projections of one wiring" is literal — the wiring being the certificate. This resolves
where the renderer's content comes from and unifies RENDER with CAS-CERT (one artifact, machine + human
faces). *(For claude-1's CAS-CERT spec — logged here as the bridge; each guide facet maps to a substrate
layer above.)*

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

### H-R2-harness — REVIEWED PASS (claude-1, 2026-06-17) · commit `d104927`
`scripts/iatc_semcheck.bb` + `tests/iatc_semcheck_test.clj` + evidence
`iatc-semcheck-loop-run-70b.{edn,md}`. Checked: it loads R2a/R2b/R2c into namespaces
and calls their `check-graph` (no reimplementation); the `iatc_closure_check.bb` edit is
**purely** a `-main`/`babashka.file` import-guard (diffed a3b523e→d104927 — check logic
byte-identical); 3 suites pass (10+8+7); ground-truth anchors behave (0706 clean, 0709
R2a-flagged at the proposition anchor, 0708.2185 R2b-flagged); N/A≠FAIL, `--gate`,
finals-only, R2a labelled lexical-lower-bound, R2c floor 0.0 report-only (justified:
spread includes 0.0).

**Rung-2 COMPLETE.** Headline: on `loop-run-70b`, **4/9 finals FAIL → overall FAIL** —
an *independent, deterministic* confirmation that the 70B graphs are well-formed (rung
0/1) but **semantically incomplete**: orphan/dangling nodes in `0705.0452`,
`0708.1921`, `0708.2067`, `0712.0724` (objects declared but never wired into the
argument), plus the R2a lexical-lower-bound flags. This is the concrete answer to "are
we recovering the semantics?" — *partially*, and now precisely localized.

Open follow-ups: R2a-v2 (macro normalization + ±1 line + N/A-for-sparse); calibrate the
R2c warrant floor once R2a-v2 de-noises; wire `iatc_semcheck.bb` into
`mark3_iatc_loop.py` as a rung-2 self-gate (currently report/`--gate` only).

### R2a-v2 findings — codex-1

De-noised `scripts/iatc_anchor_faithfulness.bb` without changing the public
`check-graph` contract. The checker still reports
`{:check :anchor-faithfulness :pass :rate :reasons :per-item}`, but term matching now:

- normalizes light LaTeX before matching on both node/source text (`\maps`,
  `\Ob`, `\Mor`, `\G`, math/text wrapper macros, styled alphabets, arrows);
- searches the cited line span with a ±1 neighbor-line tolerance;
- marks nodes with fewer than `K=2` scoreable terms as `:status :na` instead of
  scoring them as failures.

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

### R2-wire findings — codex-3

Wired rung-2 into `scripts/mark3_iatc_loop.py` after the existing candidate
faithfulness, rung-0 `iatc_argcheck`, and rung-1 `substance_gate` checks.
Each emitted final graph now gets a sibling semcheck sidecar
`<paper>.rung2.edn`, generated by `bb scripts/iatc_semcheck.bb --out ...`.
That sidecar carries the description-first profile, R2a/R2b/R2c verdicts, and
the residual-sorry evidence already surfaced by the semcheck profile.

Default remains soft: a rung-2 failure records the semantic verdict but does
not reject the graph. `--rung2-gate` opts into hard gating by passing
`--gate` through to `iatc_semcheck.bb`; failing rung-2 attempts are retried
before final emission. The batch substance gate now runs over the accepted
graph file list explicitly so `.rung2.edn` sidecars are not mistaken for IATC
graphs.

Focused tests cover both paths: soft mode emits graph + profile when rung-2
fails, while hard mode rejects the first failing rung-2 attempt and retries
until a passing profile is produced. A live stub smoke with an existing
candidate did not emit because the pre-existing stub can return seed graphs
whose `:paper/id` does not match the candidate and the candidate-faithfulness
gate rejects them; the monkeypatched loop tests isolate the rung-2 wiring
deterministically.

Gates passed: `python3 -m py_compile scripts/mark3_iatc_loop.py`,
`pytest -q tests/test_mark3_iatc_loop_rung2.py tests/test_substance_gate_selfloop.py
tests/test_mark3_eval_harness.py` (`6 passed`), `bb tests/iatc_semcheck_test.clj`
(`2` tests, `7` assertions), and `bb tests/iatc_closure_check_test.clj`
(`4` tests, `10` assertions).
