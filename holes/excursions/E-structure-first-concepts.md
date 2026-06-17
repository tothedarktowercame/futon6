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

### Deferred (spec after we see D1's coverage numbers — car-of-sequence)
- **H-SFC2** D4 enrichment loop (fill glosses; `gloss→components→:structure` lift).
- **H-SFC3** D5 R2d proof-concept-coverage bridge into E-informal-proof-checking.
- Optional: re-run `warp_concept_usage` on the current corpus for a fresh ranking
  (the inversion itself needs no re-run).

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
