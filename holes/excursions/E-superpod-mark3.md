# E-superpod-mark3 — handoff spec for Rob (LLM/neural phases over all arXiv)

**Date:** 2026-06-16 · Joe + Claude owner · **Status: OUTLINE / planning**
**For:** Rob (rjmeyers @ superpod.smu.edu — 20× DGX A100, 160 GPUs; batch partition
18 nodes / 2-day walltime). Model: LLaMA-class (Joe).
**Predecessors:** the CPU pipeline (futon6 `dp_paper_view` + `dp_enrich` +
`dp_anatomy_html`), [E-iatc-model](E-iatc-model.md),
[E-prior-over-terms](E-prior-over-terms.md),
[E-superpod-ct-nlp-intrinsic-eval](E-superpod-ct-nlp-intrinsic-eval.md),
[M-prior-mathematics](../missions/M-prior-mathematics.md), and the close-reading
pilot (`futon3c/holes/excursions/close-reading/`).

## HEAD

mark3 is the **LLM/neural layer (b)** on top of the deterministic CPU layer (a).
Layer (a) marks *what is in the text* (symbols, math, IATC illatives in running
prose, expository scopes, term grounding) deterministically and lint-gated.
mark3 adds *what the prose elides or only implies* — typed grounding, coreference,
warranted argument graphs, paraphrase-level scope recognition — at **arXiv scale**.
Everything mark3 emits is **standoff** (char/line-anchored, nothing restated) so
it merges into the existing renderer as additional mark layers + `.edn` graphs.

## Input contract (what Rob receives, per paper)

- `text` (the mined source) + the **layer-(a) marks** (`fable-<id>-dp-emacs.json`):
  symbols (grounded/ungrounded, incl. named operators), math spans, claims,
  inferences (`register: deductive|body`), exposition regions, concept terms,
  binders, env scopes. These are mark3's *scaffold* — it enriches, never re-derives.
- Per-MSC term-prior (`term-prior-<msc>.json`) + the NNexus/nLab index.
- Scope is **all arXiv**: layer-(a) detectors + term-prior are per-MSC and
  generalize by re-pointing the corpus; the mark3 phases are MSC-agnostic (they
  read mathematical prose). math.CT (the gh200 + the 9,742-paper corpus) is the
  pilot/benchmark; other MSC classes follow by re-pointing.

## Output contract

- Additional **standoff mark layers** merged into the same JSON (typed grounding,
  anaphora links, classified scopes) — char-anchored, no restated text.
- **`.edn` argument graphs** per passage (the IATC schema; see Phase D).
- Every annotation carries a **confidence**; low-confidence ones route to review
  (consent gate), not silent acceptance.

## Phases

### P1 — Bound-term grounding (ground-to-type)
Every ungrounded `symbol` mark → its **type**, inferred from the binder + usage
("Let $\\ba$ be a flock" ⇒ `a,b` : objects of $\\ba$; $f(a,b)$ ⇒ args typed by the
domain). **Why superpod:** needs reading the binding context, not a lexicon lookup.
**Output:** `symbol-grounded` + type gloss. **Eval:** grounding-% lift (1005.2653
25% → ~98% target; see the mockup). The remaining ungrounded after layer-(a)'s
operator + binder + NNexus grounding are exactly the bound variables P1 targets.

### P2 — Anaphora / coreference resolution
Resolve "the above", "this", "such", definite descriptions to antecedents — the
general case beyond layer-(a)'s classical enumerate-item resolver. **Why superpod:**
coreference. **Output:** `anaphor` marks with resolved target span.

### P3 — Expository scope: coverage + classification (incl. alternate phrasings)
Two jobs, both neural:
1. **Coverage** — *general neural NLP on the expository sections* (Joe): the
   close-reading consolidation reached only **34.72% of expository sentences**.
   A sentence-level expository-move model (embed each sentence; classify into the
   minted hierarchy or "none") pushes coverage up by generalizing past keyword/
   synonym matching to **paraphrase / embedding similarity** — this is the
   "alternate phrasings for recognition of scopes" item.
2. **Classification + discovery** — bin each expository scope into the minted
   kinds (`connection/*`, `rationale/*`, `obstruction/*`, …); DP-style
   mint-pressure proposes new kinds when a cluster fits none.
**Seed labels:** the 47.8k gh200 agent proposals (weak labels) + the 6
close-readings (427 gold records) + the seed hierarchy's `:synonyms`.
**Eval:** coverage %, classification agreement, discovery-curve saturation —
ties directly into [E-superpod-ct-nlp-intrinsic-eval](E-superpod-ct-nlp-intrinsic-eval.md).

### P4 — IATC → Clojure/EDN argument graphs
Reconstruct the warranted argument DAG with **explicit warrants + honest typed
holes** (`:missing-warrant` naming what the prose elides), line-anchored standoff.
**Already de-risked:** the self-gating Codex pool produced **13/13 checker-PASS**
graphs in the pilot — so mark3 can run *generate → check → PASS* against the same
checker (or a model fine-tuned on the 13 pilot graphs as few-shot). **Output:** the
`.edn` schema (`:nodes` :object/:claim/:meta, `:edges` :infer with
:premise/:given/:conclusion/:warrant, `:holes`). The renderer already shows these.

### P5 — Recurring-non-term filtering (NP-head precision)
Drop recurring descriptive phrases the corpus-df prior **can't** catch because
they recur ("study of categories" df=26, "wishes to study" df=5). **Why superpod:**
needs NP-head / POS structure to tell a term from a descriptive phrase — the
precision boundary layer-(a) explicitly punts to mark3.

### P6 — Citation / cross-paper resolution
Resolve in-text `[N]` to the cited arXiv paper and import its claims → a
**cross-document** argument graph. All-arXiv scope makes this the payoff phase:
warrants discharged by results in *other* mined papers.

## "Anything else" — additional candidates
- **Confidence calibration** (cross-cutting): every mark3 annotation carries a
  trust score; thresholds route to the consent gate / human review.
- **Eval harness** (cross-cutting, prerequisite): grounding-%, expository-coverage-%,
  checker-PASS-%, agreement-with-pilot, prior-vs-posterior (M-prior step-2). Without
  it the run isn't measurable. Extends E-superpod-ct-nlp-intrinsic-eval.
- **EATC** (Exposition Anchoring Theory + Content): formalize the expository
  pattern language (P3's target vocabulary) from mission-lifecycle phases +
  math-informal flexiargs — the expository analogue of IATC.
- **Math-structure parse** (different axis): a Content-MathML-ish parse of complex
  `$...$`/display expressions for mouse-over structural verification (Joe's old
  DC-4/DC-5). Neural seq-to-tree; optional.

## Boundary — what stays CPU vs goes to mark3

| concern | CPU layer (a) — done | mark3 — superpod |
|---|---|---|
| symbol grounding | named operators, binders, NNexus singles (provenance-gated) | **bound variables → type** (P1) |
| reasoning | illatives in running prose (`register`) | **warranted graphs + holes** (P4) |
| expository | structural regions + synonym/vote classification | **paraphrase coverage + discovery** (P3) |
| terms | corpus-df prior (kills hapax/overfed) | **recurring-non-term NP-head** (P5) |
| reference | enumerate-item anaphora; cite marks | **general anaphora (P2); cross-paper (P6)** |

## Scale ladder — what each level can do that the one below can't (Joe)

The phases above are mostly *per-paper* enrichment. But some capabilities **only
exist at corpus scale** — they're not "a better per-paper algorithm," they're
impossible on one paper. The organizing insight:

- **per-paper = PARSE** (local structure: the layer-(a)+(b) marks).
- **per-MSC = NORMS** ("is this normal *for this field*?"). Embeddings here =
  an MSC semantic space → synonymy, anomaly/surprise.
- **whole-arXiv = CONNECTIONS** ("where *else* does this idea live?"). Embeddings
  here = a universal space → cross-field transfer.

### Per-MSC — only-at-corpus capabilities
- **Base-rate priors** (M-prior-mathematics, term-prior — *built*): what's a real
  term vs a hapax; the over-detection "surprise" signal. A paper can't know its own
  base rate. Drives P5 (overfed/hungry/junk).
- **Paraphrase / synonym clusters via embeddings**: the *alternate phrasings* item —
  "Fourier transformation functor" ≈ "Fourier transform" ≈ "Fourier functor" cluster
  in the MSC space. This is the engine for P3 coverage and for normalizing term
  variants. Per-paper sees one phrasing.
- **Vocabulary + scope-kind DISCOVERY** (learn-and-promote; mint-pressure): a term
  or a new expository kind exists *because it recurs* across the field. n=1 can't
  mint. (Partly done — gh200 votes minted 5 kinds.)
- **Canonical-definition consensus + DEVIATION**: ground a term to the field's
  consensus definition — and flag when a paper *redefines* a standard term (the
  interesting signal). Per-paper has only the author's phrasing.
- **Notation priors**: what a symbol typically denotes in the field.

### Whole-arXiv — only-at-corpus capabilities
- **Cross-field concept transfer**: "convolution" / "groupoid" / "Fourier" link
  across CT, functional analysis, physics (Day convolution ↔ classical convolution).
  Universal embeddings bridge fields; an MSC space can't.
- **Global citation / claim graph** (P6 at full scale): a warrant in a CT paper
  discharged by a result in an algebra or topology paper — the cross-document
  argument web. The payoff of all-arXiv scope.
- **Global entity resolution**: canonical theorems (Brown representability, Yoneda),
  objects, authors resolved across all of math.
- **Universal retrieval / analogy**: "the topology analogue of this construction."
- **Diachronic / emergence**: trending terms over time — resolves the
  trending-vs-hallucination hole (M-prior §2.3).

### Implication for the phases
Several phases gain a corpus dimension Rob should build explicitly:
P1 grounding ← per-MSC notation/definition priors, whole-arXiv cross-field types;
P3 ← per-MSC paraphrase clusters + kind discovery (embeddings ARE the engine);
P5 ← per-MSC base-rate (done) *plus* "doesn't cluster near any known term ⇒ not a
term"; P6 ← whole-arXiv by nature. **Embeddings are a first-class mark3 artifact**,
trained per-MSC (norms) and globally (connections) — not just a per-paper feature.

## The loom — a nested, self-similar structure (Joe)

The scales aren't a flat ladder; they're a weave, and the same pattern recurs at
every scale (fractal). Four **views over one dependency graph**:

- **weft** — *per-paper*: the threads woven across one page (the layer-(a)+(b)
  marks: symbols, IATC, expository, the argument graph).
- **warp** — *across-papers / per-MSC*: the tension lines set up first that every
  weft weaves through (the field's norms — priors, embeddings, vocabulary, the
  citation structure). The warp is the frame; the weft is the picture.
- **thread** — *per-concept phylogeny*: follow ONE concept longitudinally — its
  definitions, redefinitions, specializations across papers and time (git-blame
  for a concept). A diachronic cut through the warp.
- **tapestry** — *cross-classification interface map*: not what's *inside* a paper
  or class, but what interfaces it **exposes outward** — its **imports**
  (dependencies it loads) and **exports** (the definitions/results others cite).

### The self-similar recursion (the load-bearing insight)
The same two structures — **interface (import/export)** and **dependency
(reasoning)** — appear at every scale:

| scale | import / export | dependency / reasoning |
|---|---|---|
| sentence | "By [3], X" = a micro-import | premise → conclusion + warrant |
| paragraph / intro | the intro *loads the field's libraries* | motivates: goal → construction |
| paper | bibliography = dependency manifest; results = exports | the proof DAG |
| MSC / arXiv | the citation graph = the package ecosystem | cross-document argument web |

So a paper's **introduction is BOTH expository (a weft scope) AND a library-load
(a tapestry import)** — the same act seen at two scales. And crucially: **the IATC
argument graph and the citation graph are the same dependency structure at
different scales.** A `:warrant` edge and a `[N]` citation are the same kind of
edge — a dependency — one resolved *in-paper*, one *out-of-paper*.

### Practical upshots for mark3
- **A typed hole (`:missing-warrant`) IS an unresolved import.** This unifies P4
  and P6: resolving a hole = linking the dependency *down* (to an in-paper premise)
  or *out* (to a cited / corpus result). P6 discharges P4's holes against the
  whole-arXiv graph. One mechanism, two scales.
- **Interface extraction = the tapestry.** Each scope/paper has a computable
  interface: **imports** = citations + "by [N]" warrants + `connection/*`
  expository scopes; **exports** = its definienda + `universal-property/*` scopes
  (what makes it citable). The tapestry is the import/export graph over all arXiv.
- **One representation, four views.** Store nodes + `depends-on`/`exposes` edges
  at nested scales; weft/warp/thread/tapestry are *queries* over it, not separate
  pipelines. The argument-graph `.edn` schema (P4) already has the shape — extend
  its edges to cross paper boundaries (P6) and the same store serves all four.
- Embeddings remain the substrate, but the *graph* is the object: embeddings
  cluster/link nodes; the dependency edges (warrants→citations) are the weave.

## The concept encyclopedia — structure-first NOUNS (Joe)

The interface is a **double layer**:
- **citation imports** — explicit, → *papers* (resolved by P6).
- **concept imports** — implicit, → *core disciplinary knowledge*. An advanced
  paper *uses* "2-morphism" without citing it: it's assumed background, not a
  reference. In a paper→citation graph this edge is invisible; in a paper→concept
  graph it points at a bare string. **It has no structured resolution target.**

**Proposal: build the target as we go — a "cheap PlanetMath" per-MSC.** Not
"concept X is defined in paper Y" (an edge), but an actual **semi-formalised
definition of the object** — exactly as we semi-formalise *reasoning* into IATC
`.edn` graphs. The IATC graphs formalise the **verbs** (inferences); this
formalises the **nouns** (the objects/types). Same discipline: structured,
standoff-to-provenance, honest typed holes for what's not pinned down.

**Scope:** the top ≈2000 concepts per MSC — and the **term-prior already ranks
them** (df). ~2000 is a bounded, one-time per-MSC superpod synthesis job.

### Why this beats a paper→concept graph
A usage graph tells you *which* concepts a paper touches; it can't tell you *what
they are*, so you can only count. With structure-first definitions you can
**reason about contents**: type-check an IATC warrant against the object's
structure ("this step treats $\\hat K$ as a left adjoint — does the def support
that?"), resolve concept-imports, and follow the **concept-dependency graph**
(2-morphism → 2-category → category → …). It is a per-MSC **type system /
ontology**, not a bag of strings — the real meaning of "structure-first."

### Entry schema (semi-formal, `.edn`, like the IATC graphs)
```clojure
{:concept/id "2-morphism" :msc "18"
 :kind :morphism                         ; object|morphism|property|construction|operation
 :signature "a morphism between parallel 1-morphisms f,g : A → B in a 2-category"
 :depends-on [:2-category :1-morphism :parallel]   ; concept-import edges (the deps)
 :structure {:source :1-morphism :target :1-morphism
             :constraint "parallel: shared 0-cell source and target"}
 :operations [:vertical-composition :horizontal-composition :identity-2-cell]
 :defining-property "..."                 ; the discriminating condition
 :provenance {:nlab "nlab-…" :nnexus "nnexus:…" :defined-in ["pid…"] :df 619}
 :holes [{:kind :underspecified :wanted "coherence laws not stated here"}]}
```

### How it fits the rest
- **Resolves the implicit interface layer**: a concept-import now points at an
  encyclopedia entry — the double layer is fully resolved (citations→papers,
  concepts→entries).
- **Makes "canonical-definition consensus" concrete** (the per-MSC capability):
  the entry IS the field's consensus definition, synthesised from the corpus's
  defined-in-paper occurrences + nLab/NNexus; **deviation detection** = a paper
  whose usage doesn't match the entry's structure.
- **The warp's concept layer**: the shared library every weft (paper) imports
  from; exports (a paper's definienda / `universal-property/*` scopes) accrete
  into it.
- **Typed substrate for P1/P4/P6**: P1 grounds a symbol to a concept *with
  structure*; P4 warrants type-check against it; P6 + concept-imports together
  discharge holes.

**v0 BUILT (2026-06-16, cheap/classical):** `scripts/build_concept_encyclopedia.py`
→ `data/concept-encyclopedia-ct.json` — **200 top-CT concepts**, assembled from
existing artifacts (term-prior df + NNexus/nLab index + warp/def-snippets +
concept-graph PageRank + defined-index), no fresh mine. Stats: 156/200 carry an
nLab/NNexus authority link; 113/200 have a clean in-corpus *definitional* sentence
(e.g. abelian category, triangulated category, presheaf — textbook-quality); 112/200
carry concept-dependency edges; every entry has a `:semi-formal` hole for the deep
structure (signature/type/defining property). **Telling pattern:** the ~half with
weak in-corpus definitions are largely the *core-assumed* concepts (natural
transformation, functor) that papers DON'T re-define — they lean on nLab provenance
+ the hole, which directly validates the concept-import thesis. This is the noun-side
analogue of the 13 IATC argument graphs: a concrete target set for the superpod to
deepen.

**GOLDEN few-shot seeds (2026-06-16):** `data/concept-encyclopedia/ct-golden/` —
6 hand-authored, fully-formalised entries in the APM-Xi structure-first form
(`:genus` / typed `:given` / structural `:data` / `:axioms` as ∀/∃ statements with
`:refs`, **no holes**): functor, natural-transformation, adjunction, monad,
abelian-category, and **yoneda-lemma**. The noun-side analogue of the 13 IATC
argument graphs — what the cheap scaffold's `:formalise` holes get deepened into.
README gives the superpod handoff contract + self-gating check.

**Theorems are scopes too (Joe, 2026-06-16).** A *scope* is typed `:given` (with
relations) → a produced output. A **definition** produces defining `:axioms`; a
**theorem** produces a `:conclusion` relation justified by a `:proof` — same shape,
`:kind :theorem` (see `yoneda-lemma.edn`). This is the **hinge** between the
encyclopedia (nouns) and the IATC graphs (verbs): a theorem's `:proof` IS a
verb-side IATC graph; its interface is explicit — it **imports** its hypotheses +
the lemmas it `:uses`, and **exports** its `:conclusion` + downstream constructs —
so it is a first-class node in the dependency tapestry, and a proof's
`:missing-warrant` hole is an unresolved lemma-import. Theorem *statements* are
already detected as `env/theorem`/`env/lemma` scopes (layer a); mark3 adds the
structure-first treatment of the statement (givens → conclusion) + the proof link.

**Build (mark3, per-MSC, full):** for each of the top-≈2000 concepts, synthesise the
entry from (nLab/NNexus seed) + (its defined-in-paper passages across the corpus)
+ (usage contexts), emit `.edn` with honest holes; extract `:depends-on` to build
the concept-dependency graph. One-time per MSC; refreshed as the corpus grows.

## Restructure: Codex handoffs now vs superpod-only (2026-06-16)

The IATC pilot showed that "superpod" work is often a **self-gating Codex pool**:
per-item EDN output + a checker + golden few-shot. Most of the wishlist factors
that way. Discriminator: **Codex** if (structured output + checker + seeds +
per-item scope); **superpod** if (embedding model / neural training / arXiv-scale).

### Codex handoffs (buildable now) — each: input · seeds · self-gating checker · out

**Wave 0 — substrate (do first; gates the rest)**
- **H0a Eval harness** — compute grounding-%, expository-coverage-%, checker-PASS-%,
  prior-vs-posterior. Input: golden marks + run output. No seeds; it IS the gate.
- **H0b Noun-side checker** — verify a concept/theorem `.edn`: every `:refs`/
  `:depends-on` resolves to another entry; every `:axiom`/`:conclusion` is
  well-formed. (Verb-side IATC checker already exists.)

**Wave 1 — the type substrate (the others check against it)**
- **H1 Concept encyclopedia deepening** — fill the 200 `:formalise` holes
  (`data/concept-encyclopedia/ct/*.edn`) into the golden form. Input: scaffold
  entry (gloss + provenance + corpus passages + dep edges). Seeds: the **6 golden**
  (`ct-golden/`). Checker: H0b. Out: deepened `.edn`. *Prime candidate — exact
  IATC-pilot shape.*

**Wave 2 — per-item, parallel, check against the encyclopedia**
- **H2 IATC argument graphs at scale** — the 182 gh200 (+ the 18 giants).
  Seeds: the **13 pilot graphs**. Checker: the §6 IATC checker. Proven.
- **H3 Theorem formalisation** — give `env/theorem`/`env/lemma` statements the
  givens→conclusion treatment + link the proof graph. Seed: `yoneda-lemma.edn`.
  Checker: H0b + proof resolves to an IATC graph.
- **H4 Ground-to-type** (P1) — per paper, bind each ungrounded symbol to its type.
  Seed: the `1005.2653` mockup. Checker: each type resolves to an encyclopedia
  entry or an in-paper binder. **Acceptance = the symbol-grounding star's
  pre-registered bar** (`M-symbol-grounding.md §5a`): Layer-3 ≥50% precision on
  real output + coverage-at-confidence vs the mockup's projection + gates hold.
  H4 is what flips that star from in-progress to satisfied.
- **H5 Anaphora** (P2) — per paper, resolve "the above"/"this"/definite
  descriptions. Checker: antecedent span exists + type-compatible. (Needs 1 golden.)
- **H6 Expository classification** (P3, classification half) — bin scopes into the
  minted kinds. Already a Codex pool (the 47.8k gh200 proposals). Checker: kind ∈
  hierarchy or mint-pressure proposal.
- **H7 Citation resolution** (P6, per-paper half) — resolve `[N]` to the cited
  arXiv id. Mostly mechanical; checker: id exists in corpus.

### Wave 3 — CODE handoffs (we write + sample-validate now; Rob RUNS at scale)
The "superpod" items are superpod-only in **execution** (GPU / all-arXiv); the
**code is ours**. These are Codex handoffs whose deliverable is a *pipeline/script
+ a small-scale validation* (on the gh200 / a sample), runnable as-is by Rob at
full scale. Gate = "runs end-to-end on the sample + sanity metrics hold."
- **H8 Embedding pipeline** (per-MSC + global). Write the train/infer script —
  **BGE, with hard negatives** (not R-GCN; see superpod-embeddings note). Validate:
  retrieval sanity on gh200 (known synonyms cluster). Rob: train on all-arXiv.
- **H9 Deviation detector** — code comparing a paper's usage to the encyclopedia /
  embedding norms; flags redefinition / non-standard notation. Validate on a few
  hand-picked deviation cases. (Depends on H1 + H8 outputs.)
- **H10 Expository-coverage model** — train/eval a sentence-level expository-move
  classifier on the gh200 votes (weak labels) + the 6 close-readings (gold).
  Validate: held-out coverage/F1. Rob: run over all-arXiv. (Lifts the 34.72%.)
- **H11 Cross-document graph builder** — resolve citations + assemble the global
  dependency/claim graph. Validate on a gh200 subgraph. Rob: all-arXiv.
- **H12 Diachronic / emergence** — emerging-term detector over the corpus+time.

### Genuinely execution-only (NOT us)
Only the **large-scale GPU runs** of H8–H12 — training embeddings on all-arXiv,
batch inference at scale. That is compute, fully parameterised by our code; nothing
here is unwritten on our side.

### Ordering
H0 → H1 → {H2…H7 EDN handoffs ∥ H8 embeddings} → {H9, H10, H11 once H1/H8 land}.
The Wave-1/2 Codex output (encyclopedia, classifications, graphs) is the structured
substrate the Wave-3 code compares/trains against — deviation (H9) and coverage
(H10) only become possible once it exists. So everything up to the GPU runs is a
Codex/us deliverable; the superpod's unique role is scale execution of H8–H12.

## APM as inline evaluation (Joe, 2026-06-16) — the capstone eval

The UT-Austin prelim corpus (APM: `storage/apm/`, 489 problems over four
subjects — analysis, algebra, functional analysis, topology) is a **small,
curated, multi-MSC, ground-truthy** set: ideal to run the *whole* method against,
end-to-end, not one layer. It folds in as a **late stage** of the mark3 run — it
needs the encyclopedia + IATC + grounding layers to exist first (there must be
something to resolve imports *against*). It also feeds the held
`ai-passes-prelims` star (substrate `apm-prelim-corpus-substrate`, already attested).

**Already real (the "early analysis"):** `futon3c/data/apm-informal-proofs/`
(117 NL proofs); `futon6/data/apm-proof-scope-{audit,summary}.json` — 76 proofs
scope-marked, **1.3% floating-expr** (scopes bind well), **2341 externally-bound
vs 1620 orphan symbols (~59% grounded)**, and a **Lean cross-ref** per problem
(sorry-free 4 / sorry-carrying 11 / no-lean 61).

**Why it's a strong eval — it exercises every layer with a checkable target:**
- **scope markup** (layer a) — *evidenced* (1.3% floating, multi-MSC).
- **symbol grounding** (P1/H4) — the **1620 orphans are a second, non-CT
  measurement surface** for the symbol-grounding bar (§5a of M-symbol-grounding):
  a generalization test beyond gh200/CT. Current baseline ~59% externally-bound.
- **IATC graphs** (P4/H2) — mark up the 117 proofs as argument graphs; validate
  faithfulness via the checker **+ the Lean cross-check** (the sorry-free/-carrying
  proofs give formal ground truth for whether the graph's structure matches a real
  proof).
- **import resolution / tapestry** (P6/H11 + encyclopedia) — *this is the
  "structure-based approach to relate proofs to the literature" Joe asked for*: a
  prelim proof imports known theorems/concepts; resolving its externally-bound
  symbols + warrants to the encyclopedia/arXiv tapestry, and checking they resolve
  *correctly* (we know what a standard prelim proof depends on), is an end-to-end
  test of the whole stack against ground truth.

**Honest split:** scope markup is evidenced; the IATC-graph + import-to-literature
eval is **projected** — a late-stage handoff (**H13**, extends H0a): run H2/H4/H11
over APM as a held set, score against the Lean cross-ref + standard-result ground
truth, report coverage-at-confidence per subject. Because APM is tiny vs arXiv but
ground-truthy, it is the **inline evaluation of the whole method** — and a
multi-MSC generalization check the gh200 (CT-only) can't give.

## Handoff logistics
- **Self-gating checker** offloads structural review (the pilot's key efficiency:
  by the time output reaches a human, structure is guaranteed). Ship the checker
  with the run.
- **Few-shot seeds:** the 13 pilot `.edn` graphs, the minted expository hierarchy
  (`expository-scope-hierarchy.edn` + consolidation), the 6 close-readings.
- **Benchmark:** the gh200 (already CPU-rendered); dp-demo + the
  `1005.2653-superpod-mockup.html` are the **target visualization** of a fully
  enriched paper. Relocate the `.edn` graph dir into `futon6/data/` so the run has
  one drop location and the renderer auto-picks-up.
- **Partition:** batch (18 nodes / 2-day) for the full corpus; short for dev.
