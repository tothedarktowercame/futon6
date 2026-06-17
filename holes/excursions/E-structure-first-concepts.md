# E-structure-first-concepts

*Excursion · owner **claude-1** · chartered 2026-06-17 · the **foundational first step
beneath** [[E-informal-proof-checking]]. Owned end-to-end; built via Codex handoffs that
bell back for review.*

Parent question (Joe, 2026-06-17): **"Do we even know what the terms mean?"** Proof-
checking presupposes it — you cannot check reasoning *over* concepts you can't define.
So before rung-2 ("does the reasoning hold?") comes rung **−1**: "is there a definition
for each concept the proof uses?"

Lineage: Joe's `~/Xi.tex` (2002, 14k lines) — a canonical operator/definition namespace
(`\newcommand` operator names + `\theoremstyle{definition}`), the hypertext-math /
PlanetMath seed. The "structure-first" framing (Joe ↔ Rob): cheaply rank the most-used
concepts, ensure *those* have definitions + usage examples, build the per-concept map
incrementally — priority by usage, not by paper order.

---

## IDENTIFY — the gap

E-informal-proof-checking's R2a/R2d ask whether a proof's nodes are grounded and
coherent. But all of that floats unless the **concepts themselves are defined**.
"Structure first" inverts the old tapestry (post-hoc "where do these concepts appear")
into an **upstream prioritizer**: cheaply find the commonest concepts → guarantee those
are defined → annotate them everywhere → grow the map per-concept. The pieces largely
exist but are partial, un-wired, and stop at the informal layer.

## MAP — what exists (live on disk, un-orchestrated)

- **`warp_concept_usage.py`** — cheap classical all-corpus scan (9742 eprints →
  `{paper: [concepts]}`, `concept-usage.json`). Invert → per-concept document frequency.
  Raw df is noisy (boilerplate "there exists"/"more generally" outrank real concepts);
  **`build_term_prior.py`** (OVERFED/HUNGRY/HAPAX) is the de-noiser.
- **`def-snippets.json`** (972 concepts → real definition passages) + **`defined-index.json`**
  (concept → defining papers, 12 MB).
- **`concept-encyclopedia-v0`** (`data/concept-encyclopedia-ct.json`, 200 CT concepts) —
  per concept: `df`, `pagerank`, `used_papers`, `depends_on`, `gloss {paper, text}`
  (EXPLORE-level definition passage), `components {genus, differentiae[].refs}`
  (genus–differentia; `refs` = the concept's *imports*), `defined_in {n_papers, sample}`,
  and **`:structure` = an explicit mark3 HOLE** (the deep ∀/∃ typed defining property —
  the ASSIMILATE/CANALIZE target). Its own note: *"cheap structure-first scaffold."*
- **`~/Xi.tex`** — the canonical-definition-namespace lineage.

**Gaps:** MSC-scoped (200 CT concepts, not the corpus-wide 3737); un-orchestrated
(scaffold, not a maintained incremental substrate); the `:structure` typed layer is a
hole; **no coverage check** ("of the most-used concepts, which are actually defined?");
not bridged to proof-checking.

## DERIVE — deliverables

- **D1 (foundational gate) · concept-definition COVERAGE.** For the top-N most-used
  concepts (df-ranked, term-prior-de-noised), do we have a definition (`gloss` /
  def-snippet)? Report coverage % + the **ranked list of high-priority UNDEFINED
  concepts**. This is "do we know what the terms mean?" made checkable and cheap.
- **D2 · corpus-wide genuine-concept ranking** — df inversion over `concept-usage.json`,
  de-noised by the term-prior, optionally re-weighted by `pagerank` centrality. Beyond
  the 200 CT concepts.
- **D3 · the `concept → papers` annotate-everywhere index** as a first-class artifact
  (the inverse of `concept-usage`; "where does each concept show up").
- **D4 (deferred) · prioritized enrichment loop** — fill missing glosses; lift
  `gloss → components → :structure` (the EXPLORE→ASSIMILATE move), common-concepts-first.
  **Now de-risked on both ends (2026-06-17):**
  - *Feedstock confirmed* (H-SFC1, reviewed PASS): top-100 **100%** / top-500 **98.4%**
    of genuine concepts have definitional evidence, and the bulk carry *real* glosses
    (encyclopedia + def-snippets), not just `defined-index`. Caveat: "defined" =
    *evidence exists*, not *usable structured definition* — and ~8/500 "undefined" are
    normalization noise (`non commutative`, `unit counit`, `algebra topology`) to
    merge/clean before enrichment.
  - *`:structure` lift has a deterministic LaTeXML path* (live demo, L-closure example):
    `latexmlmath --cmml` parses the defining formula to Content MathML that maps
    **mechanically** to Clojure — recognizing `conditional-set` (set-builder),
    `evaluated-at` (restriction), `for-all`, `eq`, `⇒`, `≅`, action. The *only* gap was
    the **typed multi-var binder** `∀f,g:X→Y.` (sole un-parse; everything else clean).
    Three small deterministic rules close it — **no LLM for the expression layer**:
    (1) binder-normalize `∀f,g:T.φ → ∀f.∀g.φ` (capture `T` aside);
    (2) symbol dictionary (`approx→cong`, `evaluated-at→restrict`, `⋅→action`);
    (3) relational-chain regroup (`A⇒B≅C` → `A⇒(B≅C)`).
    Worked target: `(= (overline M) (conditional-set (∈ x X) (forall [f g] (: (→ X Y))
    (implies (= (restrict f M) (restrict g M)) (cong (· f x) (· g x))))))`.
- **D5 (deferred) · bridge to [[E-informal-proof-checking]]** — **R2d concept-coverage of
  proofs**: do the concepts a proof *uses* have definitions in the substrate? A proof
  over undefined terms is unverifiable; this couples the two excursions.

## ARGUE

> **IF** proof-checking must know what the terms mean,
> **HOWEVER** the definition substrate is partial (200 CT concepts), un-wired, and stops
> at the informal `gloss` (the `:structure` typed layer is a hole),
> **THEN** first *measure and close* concept-definition coverage for the most-used
> concepts — structure-first, priority by usage,
> **BECAUSE** you cannot check reasoning over undefined terms, and ranking by usage puts
> the scarce enrichment effort where it buys the most downstream proof-checking.

## Design stance (Joe, 2026-06-17) — each concept is a map-reduce aggregator

A structure-first concept is **not a single canonical definition**; it's a **reduce over
per-paper grounded instances**:
- **map** — each paper's *use* emits a per-paper grounded instance (the H-SFC2b
  grounding: `f(x,y)` over ℝ in one paper, over ℂ in another; `≅` = iso-in-`Y` here, …).
- **shuffle** — the `concept → papers` index (D3) groups instances by concept.
- **reduce** — fold the instances into the entry, **keeping variation as a family**, not
  an error: surface the common core (the `genus`) vs the varying axis (a `differentia`,
  e.g. domain ∈ {ℝ, ℂ}), and generalize ("over a field `K`") where the fold can.

Consequences:
- **Variation is data, fixed "in post."** Divergent groundings across papers aren't
  contradictions to resolve eagerly — they're a family reconciled at the reduce.
  Polysemy-tolerant by construction.
- **Incremental + order-independent (a monoid).** Each new paper just emits instances;
  the aggregator merges. Matches "useful at every resolution" — a new paper only refines.
- **Robust to grounding noise.** No single H-SFC2b grounding needs to be right; quality
  is the *reduced consensus*, not any one source. (Softens H-SFC1's "evidence ≠ quality"
  caveat: quality emerges from aggregation.)
- This **is** [[project_symbol_grounding_25_year_framing]]'s shape: per-paper *defeasible*
  bindings (map) + cross-paper meta-learning (reduce). It also gives the encyclopedia's
  `genus`/`differentiae` a **computational origin** (genus = reduced core; differentiae =
  the variant axes discovered by the fold).

**Schema implication:** the per-concept entry carries `instances [{paper, grounding}]`
+ a reduced `{genus, variant-axes}`, not a lone definition. Shapes **D3** (map/shuffle)
and **D4 / H-SFC2b** (the per-paper map outputs and the reduce that families them).

## VERIFY — acceptance bar

D1 coverage run reports, deterministically and cheaply (no GPU): of the top-N
df-ranked **genuine** concepts (term-prior working — the list is real concepts, not
"there exists"), what % have a `gloss`; and a plausible ranked list of high-priority
**undefined** concepts. Sanity: the CT-encyclopedia's 200 should mostly read "defined";
the gap is the long tail + non-CT MSC areas.

---

## INSTANTIATE — Codex handoffs

### H-SFC1 — concept-definition coverage + genuine ranking  · `scripts/sfc_concept_coverage.py` · PY
**Dispatch now.** Cheap, classical, evidence-first; reuses existing artifacts + the
term-prior.
**Goal:** (a) corpus-wide genuine-concept ranking — invert `data/warp/concept-usage.json`
to per-concept document frequency, de-noise with `build_term_prior.resolve_phrase`
(drop OVERFED-to-generic / boilerplate), optionally re-weight by `concept-graph.json`
pagerank; (b) **definition-coverage**: for the top-N (default 200/500), is the concept
present in `def-snippets.json` / `defined-index.json` / `concept-encyclopedia`? report
coverage % + the ranked list of high-priority **undefined** concepts.
**Files:** `scripts/build_term_prior.py` (reuse — don't fork), `data/warp/concept-usage.json`,
`data/warp/def-snippets.json`, `data/warp/defined-index.json`,
`data/concept-encyclopedia-ct.json`, `data/warp/concept-graph.json`.
**Acceptance:** coverage % at N∈{100,500} + the top-K undefined genuine concepts; the
de-noised top list is real concepts (not boilerplate — term-prior demonstrably working);
deterministic. Commit a short evidence report under `holes/excursions/`.
**Gates:** PY (`pytest` + report the numbers).
**When done:** bell claude-1 back with summary + shas; append findings here.

### H-SFC2 — the `:structure` lift, in two layers (Joe, 2026-06-17)

The `gloss → :structure` lift splits cleanly by *who does what*, and puts the LLM only
where it earns its keep:

- **H-SFC2a (deterministic, dispatch now) · LaTeXML → Clojure skeleton.**
  `scripts/sfc_def_structure.bb`. Shell `latexmlmath --cmml` on a definition's defining
  formula → Content MathML → map mechanically to a Clojure s-expr, applying the 3 rules
  from D4: (1) binder-normalize `∀f,g:T.φ → ∀f.∀g.φ` (capture `T` aside), (2) symbol
  dictionary (`approx→cong`, `evaluated-at→restrict`, `⋅→action`), (3) relational-chain
  regroup. Output = the skeleton **plus a list of ungrounded symbols/operators**
  (`X, Y, M, ·, ≅, f, g`) each marked `:grounding :hole`.
  *Acceptance:* the L-closure example yields the D4 worked target; ≥3 more formula-
  defined concepts (drawn from `def-snippets.json`) parse to Clojure; ungrounded-symbol
  list emitted; deterministic. *Gates:* BB (clj-kondo + check-parens + bb tests).
- **H-SFC2b (LLM, deferred) · symbol grounding.** Fill each `:grounding :hole` by binding
  the symbol/operator to its **per-paper domain meaning** read from the surrounding prose
  + the paper's own definitions (`X` = "$\V$-category", `·` = the action defined at
  eqn N, `≅` = iso in `Y`). This **IS** [[project_symbol_grounding_25_year_framing]] /
  M-symbol-grounding applied at the definition layer — so use its stance: bindings are
  **per-paper defeasible strategies**, checkable against the paper's *own* definitions
  ("does this symbol resolve to something the paper defines?" — flag "uses undefined
  symbol" otherwise), and cross-paper learning stays meta (no global symbol table).
  Don't reinvent it; charter against M-symbol-grounding's yardstick.

### H-SFC-D3 — concept → papers "annotate-everywhere" index  · `scripts/sfc_concept_index.py` · PY
The **map/shuffle** stage of the map-reduce (concept = monoid): the inverse of
`concept-usage` (which is paper→concepts), as a first-class queryable artifact.
**Goal:** build `data/warp/concept-index.json` = `{concept → {df, papers:[ids],
genuine:bool, defined:bool, sources:[...]}}` over the full corpus, **reusing**
`sfc_concept_coverage` helpers (`invert_usage` for df, `genuine_ranking` for the
term-prior de-noise, `definition_sets`/`attach_coverage` for defined?+sources) — note
`invert_usage` only *counts*, so the new bit is collecting the actual **paper lists**
per concept. Plus a query CLI: `--concept "natural transformation"` → its papers +
metadata; `--paper 0706.1286` → its concepts (forward direction for symmetry).
**Scope:** shuffle only — concept→paper-list + flags. The per-paper *grounded instance*
and the **reduce** (`{genus, variant-axes}`) are D4/SFC2b, not here.
**Files:** reuse `scripts/sfc_concept_coverage.py` + `scripts/build_term_prior.py` (don't
fork); inputs `data/warp/concept-usage.json` (+ def-snippets/defined-index/encyclopedia
for flags).
**Acceptance:** index built over all ~9742 papers / ~3737 concepts; df equals the
inversion; genuine/defined flags consistent with SFC1's numbers; a sample query returns
the right papers; deterministic. Commit a short evidence report under `holes/excursions/`.
**Gates:** PY (`pytest` + report the numbers). Bell claude-1 back; append findings here.

### Deferred (spec later — car-of-sequence)
- **H-SFC3** D5 R2d proof-concept-coverage bridge into E-informal-proof-checking.
- Normalization/merge pass for mis-segmented undefined concepts (`non commutative`,
  `unit counit`, `algebra topology`) before any enrichment.
- Optional: re-run `warp_concept_usage` on the current corpus for a fresh ranking.

## Remaining gaps / notes
*(Codex agents: append findings + commit shas here.)*

### H-SFC1 findings — codex-1

Implemented `scripts/sfc_concept_coverage.py`, reusing
`scripts/build_term_prior.py::resolve_phrase` for the genuine-concept pass.
The script inverts `data/warp/concept-usage.json` to document frequency,
filters boilerplate / generic fragments, optionally applies
`data/warp/concept-graph.json` pagerank weighting, and checks definition
presence across `def-snippets.json`, `defined-index.json`, and
`concept-encyclopedia-ct.json`.

Evidence report committed at `holes/excursions/sfc-concept-coverage.md`.

Acceptance run:

```sh
python3 scripts/sfc_concept_coverage.py --top-k-undefined 30
```

Coverage:

- Top 100 genuine concepts: `100/100 = 100.0%` defined.
- Top 500 genuine concepts: `492/500 = 98.4%` defined.

De-noising checks:

- `natural transformation` ranks `#2`.
- First ranked concept containing `adjoint` is `left adjoint` at `#3`.
- `there exists` and `more generally` are filtered out of the genuine ranking.

Top undefined genuine priorities from the run:
`non commutative`, `unit counit`, `n categories`, `quasi inverse`,
`quasi isomorphisms`, `algebra topology`, `hom spaces`, `quasi isomorphic`,
`generated objects`, `functors between categories`.

Remaining gaps:

- Coverage is lexical over existing substrates. A hit in `defined-index.json`
  is evidence of a definition-like occurrence, not a proof that the definition
  is conceptually good.
- Singular/plural and notation variants are not fully lemmatized yet, so
  `natural transformation` / `natural transformations` style splits remain.
- Some undefined priorities are genuine but normalization-poor phrases
  (`non commutative`, `algebra topology`, `n categories`); H-SFC2 should
  normalize or merge these before enrichment.

Gates passed: `python3 -m py_compile scripts/sfc_concept_coverage.py`,
`pytest -q tests/test_sfc_concept_coverage.py` (`3 passed`), and full
`pytest -q tests/` (`772 passed, 38 skipped`).

### H-SFC-D3 findings — codex-4

Implemented `scripts/sfc_concept_index.py`, the D3 map/shuffle artifact for
structure-first concepts. It builds `data/warp/concept-index.json`, the inverse
of `data/warp/concept-usage.json`, as:
`{concept -> {df, papers, genuine, defined, sources}}`.

The script reuses the SFC1 helpers instead of forking them:

- `invert_usage` supplies the document-frequency denominator;
- `genuine_ranking` supplies the term-prior de-noised `genuine` flag;
- `definition_sets` / `attach_coverage` supply `defined` and `sources`.

Acceptance run:

```sh
python3 scripts/sfc_concept_index.py
```

Evidence report committed at `holes/excursions/sfc-concept-index.md`.

Results:

- Indexed `3737` concepts over `9737` papers with concepts (`9742` scanned).
- `df` and `len(papers)` validate exactly against `invert_usage`.
- De-noised genuine concepts: `3340`.
- Concepts with definition evidence: `3382`.
- SFC1 consistency from the index: top-100 genuine concepts `100/100 = 100.0%`
  defined; top-500 `492/500 = 98.4%` defined.
- Sample `--concept "natural transformation"`: `df=4882`, `genuine=true`,
  `defined=true`, sources `concept-encyclopedia`, `def-snippets`,
  `defined-index`, and includes `0706.1286`.
- Sample `--paper 0706.1286`: `293` concepts, including
  `natural transformation`.

Scope boundary: this is only the D3 shuffle (concept → paper-list + flags). It
does not build per-paper grounded instances or the genus/variant-axis reduce.

Gates passed: `python3 -m py_compile scripts/sfc_concept_index.py`,
`pytest -q tests/test_sfc_concept_index.py`.

### H-SFC2a findings — codex-3

Implemented `scripts/sfc_def_structure.bb`, a deterministic
LaTeXML-Content-MathML to Clojure `:structure` transducer. The script shells
`latexmlmath --cmml=- -`, parses the returned Content MathML with a small
dependency-free tag parser, and applies the D4 deterministic rules:

- binder-normalize `\forall f,g:T` into single binders while preserving
  `{:vars ["f" "g"], :type "T"}`;
- symbol dictionary: `approx -> cong`, `evaluated-at -> restrict`,
  `conditional-set`, `for-all`, `\Rw -> \Rightarrow`;
- relational-chain regroup for the observed `A => B ~= C` shape.

The L-closure worked example emits the target:

```clojure
(= (overline M) (conditional-set (∈ x X)
  (forall [f g] (: (→ X Y))
    (implies (= (restrict f M) (restrict g M))
             (cong (· f x) (· g x))))))
```

Evidence report committed at
`holes/excursions/sfc-def-structure-evidence.md`, including three additional
formula-defined concepts from `data/warp/def-snippets.json`: `fibrant
replacement`, `homotopy category`, and `homotopy equivalence`.

Remaining gaps:

- H-SFC2a intentionally leaves symbols/operators as `{:grounding :hole}`;
  H-SFC2b must bind those per paper.
- The generic Content-MathML mapper is deliberately small; it handles the
  observed formula layer but does not yet canonicalize all LaTeXML fallback
  operators such as implicit `times` from juxtaposition.

Gates passed: `clj-kondo --lint scripts/sfc_def_structure.bb`,
`emacs --batch -l /home/joe/code/futon4/dev/check-parens.el
scripts/sfc_def_structure.bb`, and `bb tests/sfc_def_structure_test.clj`.

### H-SFC2a — REVIEWED PASS (claude-1, 2026-06-17) · commit `bc38163`
Checked: clj-kondo 0/0; bb tests 6/6; **L-closure reproduces the D4 target *live*** (via
`bb scripts/sfc_def_structure.bb -`), with `:normalized-formula` + `:binder-captures
[{:vars [f g] :type "X→Y"}]` confirming the binder-normalize rule, and the ungrounded
list emitted. Generalizes across the 4 report shapes (set-builder/∀/⇒, typed map,
functor-app, ≅). Meets the handoff bar — **PASS**.

**Generalization gaps I found by testing a formula NOT in the report** (`\{ n \in
\mathbb{N} \mid \exists k . n = 2k \}`) — these are H-SFC2a-v2, not blockers, but one is a
*correctness* concern:
- **`\exists` is silently mangled** → `(formulae-sequence ( k) …)`; the existential is
  **dropped**, not flagged. *Must* gain an `∃`-normalize analogous to the `∀` rule (or at
  minimum mark `∃` as an unhandled `:hole`) before the transducer is trusted beyond the
  ∀-class — a silent quantifier drop yields a *wrong* `:structure`.
- `\mathbb{N}` → `(* \mathbb N)` (spurious juxtaposition-times + ungrounded `\mathbb`);
  add `\mathbb{X}`/`\mathcal{X}` → blackboard/script symbol to the dictionary.
- LaTeXML's `formulae-sequence` join leaks into output; canonicalize/strip it.
- (Known, by design) juxtaposition `Qf` → `(* Q f)`: mult-vs-application ambiguity —
  correctly deferred to H-SFC2b grounding (Q-as-functor ⇒ application).

### H-SFC2a-v2 findings — codex-4

Widened `scripts/sfc_def_structure.bb` without replacing the H-SFC2a transducer:
the original LaTeXML Content-MathML path and three deterministic rules remain,
with a new canonicalization pass for the reviewed gaps.

Closed gaps:

- `\exists k . φ` now canonicalizes from LaTeXML's `formulae-sequence` fallback to
  `(exists [k] φ)` instead of silently dropping the quantifier.
- Typed/multi-variable existentials normalize analogously to the existing binder
  rule: `\exists x,y:X\to Y. x=y` emits
  `(exists [x] (exists [y] (= x y)))` and captures
  `{:vars ["x" "y"], :type "X\\to Y"}`.
- `\mathbb{X}` / `\mathcal{X}` normalize to styled mathematical symbols before
  LaTeXML where possible, with a fallback that collapses LaTeXML's
  `(* \mathbb X)` / `(* \mathcal X)` artifacts.
- Unrecognized `formulae-sequence` shapes become explicit `:hole` structures
  rather than leaking the ambiguous operator into `:structure`.

The reviewed regression formula now emits:

```clojure
(conditional-set (∈ n ℕ) (exists [k] (= n (* 2 k))))
```

The H-SFC2a L-closure target remains unchanged.

Gates passed: `clj-kondo --lint scripts/sfc_def_structure.bb`,
`emacs --batch -l /home/joe/code/futon4/dev/check-parens.el
scripts/sfc_def_structure.bb`, and `bb tests/sfc_def_structure_test.clj`.

### H-SFC-AGG findings — codex-1

Implemented `scripts/sfc_concept_aggregate.py`, the v0 reduce stage over
`concept-index.json` plus definition sources. The run writes the committed
adjunction fixture at `data/warp/sfc-adjunction-fixture.json` and evidence report
at `holes/excursions/sfc-concept-aggregate.md`.

Results:

- GC surface→core retention emits the required bad-term examples while retaining
  their paper support: `all functors -> functor` (`df=1028`), `any two -> pair`
  (`df=3661`), `each other -> relation` (`df=3445`).
- Adjunction fixture assembled from all requested source families:
  `PlanetMath=2`, `nLab=8`, `arxiv-def-snippets=14`.
- Reduce recovers `genus = "adjunction F⊣G"` and the three non-contextual
  definition framings: `hom-set-natural-bijection`, `unit-counit-triangle`,
  and `universal-arrow`.
- Chosen variant-axis schema: `lean-family-v0`. It keeps a structure-like
  `genus`, stores every observed source/paper use as an `instance`, and records
  equivalent definition framings as a labelled family under
  `variant_axes[].variants`. Equivalence between framings is represented as
  explicit `iff-lemma` bridge holes, mirroring Lean's broad pattern of
  `structure`/`class` fields, `instance`s, and separate `Iff`/defeq bridges.

Remaining gaps:

- The `iff-lemma` bridges are recorded, not proved.
- Framing classification is prose/keyword based; formula grounding remains an
  H-SFC2b responsibility.
- Encyclopedia-v0 seeds the genus where usable, but the adjunction entries have
  noisy/missing genus fields, so this fixture uses the hand-recognised fallback
  `adjunction F⊣G`.

Gates passed: `python3 -m py_compile scripts/sfc_concept_aggregate.py`,
`pytest -q tests/test_sfc_concept_aggregate.py`, and full `pytest -q tests/`.
