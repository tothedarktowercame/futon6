# Pre-Superpod Handoff — Classical DP Mining, Cross-MSC Validated (2026-06-14)

**To:** Rob (has the full arXiv eprint corpus locally; we don't).
**From:** futon6 warp/weft loom.
**Ask:** run the *classical* (CPU-only, no GPU) structure-mining pipeline over the
full corpus, one MSC class at a time, with the checker as the per-class
acceptance gate. This is the "floor" pass that precedes any GPU work.

Everything below is reproducible from `scripts/` in this repo; no `data/` is
shipped (it's gitignored and corpus-sized). The numbers cited are from runs on
math.CT (9742 papers) plus a 50-paper cross-MSC probe off the local mark2 inbox.

---

## 1. Why this is ready — the validation

### Structural floor generalizes *perfectly* off math.CT
`scripts/warp_crossmsc_demo.py` sampled 5 papers each from 10 non-CT MSC classes
(the date-sorted mark2 inbox, genuinely all-of-math), ran the detector + checker:

```
class      n  grounded  tagged   math  wf-err
math.AG    5      50%   100%  100%       0
math.NT    5      33%   100%  100%       0
math.CO    5      62%   100%  100%       0
math.AP    5      49%   100%  100%       0
math.PR    5      60%   100%  100%       0
math.DG    5      50%   100%  100%       0
math.RT    5      69%   100%  100%       0
math.GT    5      72%   100%  100%       0
math.QA    5      80%   100%  100%       0
math.LO    5      74%   100%  100%       0
```

- **wf-errors = 0, tagged = 100%, math-coverage = 100% on every class.** The
  scope/binder/math/atomicity detection is domain-agnostic and robust. The
  well-formedness-overfit risk we saw while scaling *within* CT does **not**
  recur cross-domain.

### Grounding varies by lexicon distance — expected, and the run spec fixes it
- Grounding tracks proximity to the *CT-derived* concept lexicon: categorically
  adjacent domains ground high (QA 80, LO 74, GT 72, RT 69); far domains lower
  (NT 33, AP 49, AG 50, DG 50).
- NT's 33% is **not** the method failing — it's the CT hitlist not covering
  number theory. The detector's binders/appositives are domain-agnostic; only
  *grounding* resolves against a lexicon. **Fix: each MSC class builds its own
  lexicon** (§3, step B). Run that way, NT grounds against NT concepts.

### Concept extraction is unbiased (parity)
Within CT, at validation-time snapshot the DP-marked (~1k) vs classical-only
(~8.7k of 9742) papers were at parity on what the extractor sees — concepts/paper
median 99 vs 103, defs/paper 39 vs 43. So landscape placement by concept-usage is
fair corpus-wide. The DP-marked set *was* a biased **sample** (citation in-deg
mean 8.6 vs 2.5; median year 2009 vs 2020, because it was the early + most-cited
papers) — which is exactly why the rollout marks **all** papers: it erases the
selection bias. DP-exclusive metrics (grounding%, aliveness) should not be read as
corpus-representative until the whole class is marked.

> Status (2026-06-14): the math.CT mark-all is **already running locally** (a
> 4-way `dp_batch --shard` pass over the ~8.7k unmarked CT papers, ~12h), so the
> CT selection bias is being closed now and CT will land fully marked. Rob's
> full-corpus run is therefore primarily the **other** MSC classes; he can redo
> CT on his complete corpus if he wants the canonical version, but it isn't the
> bottleneck.

---

## 2. The pipeline (scripts, in order)

Weft (per-paper structure) and warp (cross-corpus second layer). Detailed
runbooks: `holes/dp-fleet-runbook.md`, `holes/warp-runbook.md`.

**Weft — per paper:**
1. `scripts/dp_paper_view.py` — detector. `build(pid, with_binders, with_scopes,
   with_ca, with_xref)` → marks (scopes, binders, math envelope, symbols,
   proof-moves). Reads eprints; capability modules in `scripts/dp_capabilities/`.
2. `scripts/check_invariants.py` — checker. `check_paper(pid, {text, marks})` →
   coverage (`symbol_grounded`, `symbol_tagged`, `math_coverage`, `symbols`) +
   `wellformed_errors`. **Coverage ⊥ well-formedness**: never trade one for the
   other; fix the detector, never the checker.

**Warp — cross-corpus (after a class is weft-marked):**
3. `warp_defined_pass.py` — EMPH/DEFENV/CALL concept-definition scan → defined-index.
4. `warp_concordance.py` — term → {paper, role}.
5. `warp_hitlist.py` — defined-index ∩ concordance → groundable concept hitlist.
6. `warp_concept_usage.py` — corpus-wide 1-3gram concept usage per paper.
7. `warp_citations.py` / `warp_bib.py` — citation edges (W2).
8. `warp_concept_graph.py` — definition-dependency graph + PageRank authority.
9. `warp_concept_embed.py` — multiplicity embedding (no GPU/superpod needed).
10. Landscape/overlays (optional, for inspection): `warp_paper_landscape.py`
    (t-SNE), `warp_or_curvature.py` (#1 terrain), `warp_salingaros.py` (#2
    aliveness), `warp_greatest_hits.py` (scope-district portrait).

---

## 3. Run spec for the full corpus

**A. Partition by MSC class.** Process each primary-category class independently.
This is also the natural unit for the per-class gate (§4) and floor/ceiling
decoupling (the classical floor here; GPU ceiling later).

**B. Build the lexicon PER CLASS.** Run steps 3–5 (defined-pass → concordance →
hitlist) *within each class* so grounding resolves against that domain's own
vocabulary. Do **not** reuse the CT hitlist cross-domain — that's what caps NT
at 33%. (Cross-class concept sharing is a later, optional merge step.)

**C. Mark ALL papers** in the class (not a cited/early subsample) — removes the
selection bias documented in §1.

**D. Quarantine pathological inputs, don't hide them.** One CT paper (1001.4071,
~1.2M chars, malformed `$`+comment spans) hangs the detector ~2h. Skiplist
convention: `holes/dp-outlier-skiplist.txt` (documented, counted, not silently
dropped). Apply a per-paper wall-clock timeout and append timeouts to the
skiplist with their reason.

---

## 4. Acceptance gate (per class)

A class passes when, over its marked papers:
- `wellformed_errors == 0` (hard gate — structural correctness; held on all 10
  probe classes).
- `math_coverage` and `symbol_tagged` ≈ 100% (held on all 10).
- `symbol_grounded` reported but **not** gated to a fixed threshold — it's
  lexicon-relative; report the per-class distribution instead. Use it to decide
  where a richer lexicon (or later GPU grounding) buys the most.

`check_invariants.py` produces all of these; aggregate them per class the way
`warp_crossmsc_demo.py` does (its table is the template).

---

## 5. Outputs to return

Per class: the marks (weft), the warp artifacts (defined-index, concordance,
hitlist, concept-usage, citations, concept-graph, concept-embed), and the
per-class coverage/wf aggregate table. The coverage table is the headline
deliverable — it's what tells us the floor held and where grounding needs help.

---

## 6. Successor model stage — Argument-to-Clojure compilation

This is **not** part of Rob's CPU-only floor run above. The handoff above marks
the corpus and proves the deterministic substrate is stable. The next superpod
stage should consume that substrate and compile mathematical prose into a small
Clojure/EDN IATC graph that can be checked.

This is technically parallel to the earlier MathOverflow reverse-morphogenesis
stage, but conceptually simpler:

```
MO reverse morphogenesis:
  visible answer/thread -> infer the question/problem-situation that would make it natural

paper argument -> Clojure:
  visible proof/expository prose + DP marks -> infer the typed argument graph
  that the prose is already performing
```

The model does not invent the mathematical situation from scratch. It is anchored
by source loci, theorem/proof environments, binders, quantifiers, references,
citations, and illative cue words (`implies`, `follows from`, `therefore`,
`means`, `by`, etc.). The desired output is an explicit argument object with
typed holes where a warrant or role is not recoverable.

### Proposed stage contract

**Input per passage**
- Raw LaTeX span with line offsets.
- DP marks from the weft pass: environments, binders, quantifiers, definitions,
  math atoms, symbols, references, citations, claim/inference/anaphor marks when
  present.
- Optional close-reading records of the form `(scope, query)` from
  `holes/excursions/close-reading/*.close-reading.md`.

**Output per passage**
```clojure
{:paper/id "0905.0595"
 :passage/id "0905.0595:prop0.2:proof:190-208"
 :source {:lines [190 208]
          :kind :proof
          :text "..."}
 :nodes [...]
 :edges [...]
 :holes [...]}
```

Suggested node/edge vocabulary, deliberately small at first:

```clojure
;; node kinds
:claim        ;; proposition asserted by the paper
:ref          ;; reference to a local enumerated/theorem/labelled item
:object       ;; mathematical object or construction
:definition   ;; definitional content
:warrant      ;; theorem/definition/rule licensing an inference
:meta         ;; bibliographic or expository aside, not object-layer content

;; inference roles
:given        ;; contextual fact required to apply a warrant
:premise      ;; proposition doing the inferential work
:conclusion   ;; proposition established
:assume       ;; subproof assumption
:contradicts  ;; contradiction target
:depends-on   ;; weaker support edge when the precise warrant is a hole
```

**Checker gates**
- EDN parses.
- Every node and edge has a source locus.
- Every edge endpoint resolves to an existing node.
- Every `:ref` resolves to a local label/item/theorem/citation or is listed in
  `:holes`.
- `:meta` nodes are not used as object-layer `:conclusion`s.
- Subproof scopes are properly nested.
- Warrant gaps are explicit: `{:kind :missing-warrant ...}`, not silently
  omitted.

### Hand-built seed examples from the six DP demo papers

The demo index is:
`data/showcases/ct-anatomy/dp-demo/index.html`

It links six current-run anatomy pages:
`0807.1872`, `1012.1220`, `0905.0595`, `0801.2567`, `1005.2653`,
`0711.1761`. The examples below are intentionally hand-built seed shapes, not
extractor output. They are meant to force the Clojure schema to become concrete
before a superpod run.

#### Example A — `0905.0595`, proof of prop0.2, object inference + meta aside

Source passage: the proof uses enumerated conditions `(1)` and `(2)` from
`\cite{AR}`. The paper then says that `(2)` implies non-presentability, explains
that `(2)` is not explicitly stated in AR, and derives it from the graph
construction.

```clojure
{:paper/id "0905.0595"
 :passage/id "0905.0595:prop0.2:proof:190-208"
 :source {:lines [190 208] :kind :proof}
 :nodes
 [{:id :ar-1
   :kind :ref
   :label "(1)"
   :text "for each regular cardinal lambda, delta_lambda is a colimit cocone"
   :source {:lines [198 199]}}
  {:id :ar-2
   :kind :ref
   :label "(2)"
   :text "id_1 does not factorize through any component of delta_lambda"
   :source {:lines [200 200]}}
  {:id :delta-colimit
   :kind :claim
   :text "delta_lambda is a colimit cocone for each lambda"
   :source {:lines [202 202]}}
  {:id :not-presentable
   :kind :claim
   :text "1 is not lambda-presentable for any regular lambda"
   :source {:lines [202 203]}}
  {:id :ar-meta
   :kind :meta
   :text "Condition (2) is not stated explicitly in AR"
   :source {:lines [203 203]}}
  {:id :no-map-from-1
   :kind :claim
   :text "there is no morphism from 1 to a non-terminal object of ca"
   :source {:lines [203 204]}}
  {:id :ca-graphs
   :kind :definition
   :text "ca is the full subcategory of Gra consisting of graphs A without any morphism B_i -> A"
   :source {:lines [204 207]}}
  {:id :loop-gives-constant-map
   :kind :claim
   :text "a morphism 1 -> A means a loop in A and consequently a constant morphism B_i -> A"
   :source {:lines [207 208]}}]
 :edges
 [{:id :e-presentability
   :kind :infer
   :relation :implies
   :given [:ar-1 :delta-colimit]
   :premise :ar-2
   :warrant {:kind :missing-warrant
             :text "presentability would make id_1 factor through a component of the filtered colimit"}
   :conclusion :not-presentable
   :source {:lines [202 203]}}
  {:id :e-ar-2
   :kind :infer
   :relation :follows-from
   :premise :no-map-from-1
   :conclusion :ar-2
   :meta [:ar-meta]
   :source {:lines [203 204]}}
  {:id :e-no-map
   :kind :infer
   :relation :contradiction
   :given [:ca-graphs]
   :assume {:kind :claim
            :text "there is a morphism 1 -> A for non-terminal A in ca"}
   :premise :loop-gives-constant-map
   :contradicts :ca-graphs
   :conclusion :no-map-from-1
   :source {:lines [204 208]}}]
 :holes
 [{:kind :missing-warrant
   :edge :e-presentability
   :wanted :presentability-factorization-rule}]}
```

This example is the main schema stress test: `:given` must differ from
`:premise`, and a bibliographic aside must be `:meta`, not a conclusion.

#### Example B — `0807.1872`, Freyd extension classes imply non-small Hom

Source passage: the proof constructs a proper class of non-isomorphic modules
`M_i`, each fitting into an exact sequence, then concludes that
`\Ext^1_A(Z,Z)` is a proper class. The next corollary uses
`\Hom_D(A)(Z,TZ) ~= \Ext^1_A(Z,Z)` to obtain a proper class of morphisms in the
derived category.

```clojure
{:paper/id "0807.1872"
 :passage/id "0807.1872:L1.3-C1.4:487-527"
 :source {:lines [487 527] :kind :proof}
 :nodes
 [{:id :modules-Mi
   :kind :object
   :text "modules M_i with underlying group Z plus Z and one nonzero phi_i"
   :source {:lines [487 494]}}
  {:id :pairwise-nonisomorphic
   :kind :claim
   :text "the M_i are pairwise non-isomorphic as R-modules"
   :source {:lines [494 494]}}
  {:id :proper-class-Mi
   :kind :claim
   :text "there is a proper class of non-isomorphic modules M_i"
   :source {:lines [495 497]}}
  {:id :short-exact-sequence
   :kind :claim
   :text "each M_i fits in 0 -> Z -> M_i -> Z -> 0"
   :source {:lines [497 502]}}
  {:id :proper-ext
   :kind :claim
   :text "Ext^1_A(Z,Z) is a proper class"
   :source {:lines [503 503]}}
  {:id :hom-ext-iso
   :kind :claim
   :text "Hom_{D(A)}(Z,TZ) is isomorphic to Ext^1_A(Z,Z)"
   :source {:lines [523 523]}}
  {:id :proper-hom
   :kind :claim
   :text "there is a proper class of morphisms Z -> T Z in D(A)"
   :source {:lines [525 527]}}]
 :edges
 [{:id :e-nonisomorphic
   :kind :infer
   :relation :because
   :premise :modules-Mi
   :warrant {:kind :claim
             :text "the ordinal j for which phi_j is nonzero changes with i"}
   :conclusion :pairwise-nonisomorphic
   :source {:lines [494 494]}}
  {:id :e-ext
   :kind :infer
   :relation :therefore
   :premise [:proper-class-Mi :short-exact-sequence]
   :warrant {:kind :missing-warrant
             :text "short exact extensions classify Ext^1 classes"}
   :conclusion :proper-ext
   :source {:lines [495 503]}}
  {:id :e-corollary
   :kind :infer
   :relation :implies
   :premise [:hom-ext-iso :proper-ext]
   :conclusion :proper-hom
   :source {:lines [523 527]}}]
 :holes
 [{:kind :missing-warrant
   :edge :e-ext
   :wanted :ext-classification-of-short-exact-sequences}]}
```

This example tests a common proof pattern: construction of many witnesses,
classification by a standard invariant, then transport across an isomorphism.

#### Example C — `1005.2653`, transfer along `h` gives quantum groupoid

Source passage: the Fourier functor `\hat K` is identified as restriction along
the canonical Kleisli functor `h`. It preserves internal homs and is
conservative; therefore `[H,Vect_k]` inherits `*`-autonomous monoidal biclosed
structure, and `\hat K=[h,1]` is a quantum groupoid.

```clojure
{:paper/id "1005.2653"
 :passage/id "1005.2653:fourier-transfer:70-77"
 :source {:lines [70 77] :kind :expository-proof}
 :nodes
 [{:id :h
   :kind :object
   :text "canonical Kleisli functor h : A^op tensor A -> H"
   :source {:lines [70 72]}}
  {:id :khat-restriction
   :kind :claim
   :text "the Fourier functor K-hat is restriction along h, i.e. [h,1]"
   :source {:lines [70 73]}}
  {:id :khat-preserves-homs
   :kind :claim
   :text "K-hat preserves both left and right internal homs"
   :source {:lines [73 73]}}
  {:id :khat-conservative
   :kind :claim
   :text "K-hat is conservative"
   :source {:lines [74 74]}}
  {:id :source-star-autonomous
   :kind :claim
   :text "[A^op tensor A,Vect_k] is star-autonomous monoidal biclosed"
   :source {:lines [74 74]}}
  {:id :target-star-autonomous
   :kind :claim
   :text "[H,Vect_k] is star-autonomous monoidal biclosed"
   :source {:lines [74 74]}}
  {:id :quantum-groupoid
   :kind :claim
   :text "K-hat=[h,1] is a quantum groupoid in the sense of [3]"
   :source {:lines [77 77]}}]
 :edges
 [{:id :e-transfer-star-autonomy
   :kind :infer
   :relation :transfer
   :given [:khat-restriction :khat-preserves-homs :khat-conservative]
   :premise :source-star-autonomous
   :warrant {:kind :missing-warrant
             :text "conservative hom-preserving restriction reflects/transfers the biclosed star-autonomous structure"}
   :conclusion :target-star-autonomous
   :source {:lines [70 74]}}
  {:id :e-quantum-groupoid
   :kind :infer
   :relation :therefore
   :premise [:target-star-autonomous :khat-restriction]
   :warrant {:kind :citation :target "[3]"}
   :conclusion :quantum-groupoid
   :source {:lines [75 77]}}]
 :holes
 [{:kind :missing-warrant
   :edge :e-transfer-star-autonomy
   :wanted :restriction-transfer-of-star-autonomous-biclosed-structure}]}
```

This example tests a different shape from proof-by-contradiction: property
transfer along a functor, with a citation as warrant.

### Schema stabilization gate

Before running this as a superpod stage, use the six demo papers as a hand-built
calibration set. If the three examples above do not stabilize the vocabulary,
expand the seed set rather than prompting at scale:

1. Add at least one example from each linked demo paper.
2. Force coverage of these shapes: `:definition`, `:construction`,
   `:transfer`, `:contradiction`, `:proper-class`, `:density/full-embedding`,
   `:open-status`, `:bibliographic-meta`.
3. Only freeze the prompt/schema once the checker can classify every failure as
   either parse failure, unresolved reference, missing warrant, role error, or
   unsupported shape.

At that point the stage becomes a superpod job: batch passages by MSC class,
emit EDN, run the deterministic checker, and keep rejected passages as typed
training examples for the next schema iteration.

---

## 7. Reproduce the validation locally
```
.venv/bin/python scripts/warp_crossmsc_demo.py [batch.tar.gz] [K-per-class]
# default batch: ~/code/storage/mark2/inbox/batch-007.tar.gz, K=5
```
