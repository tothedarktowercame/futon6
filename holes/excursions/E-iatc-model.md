# Excursion: E-iatc-model — an IATC reasoning layer over mined math papers, rendered as standoff bars-and-arrows

**Date:** 2026-06-15 · owner: Opus (this session) · paired (Joe + Opus)
**Status:** IDENTIFY/INSTANTIATE — working prototype on one paper (`0905.0595`),
rendered live in the DP-anatomy demo. Captured at Joe's request for an
independent Codex verification pass (author ≠ reviewer).
**Repo:** futon6. Detector: `scripts/dp_paper_view.py`. Render-time enrichment:
`scripts/dp_enrich.py`. Renderer: `scripts/dp_anatomy_html.py`. Checker/lint:
`scripts/check_invariants.py`. Shared scope helpers: `scripts/dp_capabilities/wellformed.py`.
**Prior art:** Corneli, Martin, Murray-Rust, Rino Nesin & Pease, *Argumentation
Theory for Mathematical Argument* (Argumentation 2019; arXiv:1803.06500) — the
IATC ("Inference Anchoring Theory + Content") spec. Sibling: `holes/dp-defect-catalogue.md`
(the DP defect classes DC-1…DC-10), `holes/missions/M-distributed-proofreaders.md`.

## HEAD (one line)

A math proof is reasoning; the deterministic DP markup captures the **object
layer** (symbols, definitions, environments) but leaves the **inference layer**
as negative space. This excursion adds that layer as **standoff annotation** —
claims (propositions) connected by typed illative **arrows**, nested per IATC's
"subgraph in a statement slot", with references resolved to their bindings —
and renders it as continuous pink blockquote bars with standalone arrow rows.

## 1. The tension (why this exists)

`check_invariants.py` reports `0905.0595` **clean** (0 well-formedness errors,
100% symbols tagged) while the rendered page is visibly reasoning that we don't
model. Joe's framing: coverage is now dense enough that the *uncovered* parts
are coherent **negative space** — and that negative space is **reasoning**,
exactly what IATC is meant to cover. So instead of adding more hand invariants
(DC-1…DC-9 were object-layer fixes), bring in IATC as the framework for the
reasoning layer wholesale.

## 2. The IATC model (faithful to the paper)

IATC markup has five grammatical categories (paper §3, Tables 1–2):

- **`perf[…]`** performatives / illocutionary force: Assert, Agree, Challenge,
  Retract, **Define**, Suggest, Judge, Query, QueryE.
- **`rel[…]`** inferential structure: **implies**, equivalent, **not**,
  conjunction, has_property, instance_of, indep_of, case_split, wlog.
- **`value[…]`** heuristic judgments: easy, plausible, beautiful, useful.
- **`meta[…]`** reasoning tactics: goal, strategy, auxiliary, analogy,
  implements, generalise.
- **`struct[…]`** content-focused relations: **used_in**, reform, instantiates,
  expands, sums, cont_summand.

Structure: performatives have **slots** filled by statements or objects; a
statement slot can be filled by a **(possibly disconnected) subgraph** — that is
how a *lemma* / a nested derivation is represented (paper §3, "subgraph in a
statement slot"). Crucially IATC **anchors** content to locutions and points
named objects into the text via `struct[used_in]` — **it never restates text**.
The locution layer (the source) appears once; everything else is overlay.

### What we implement vs. the spec (completeness, honest)

| IATC | Our artifact | Status |
|------|--------------|--------|
| locution layer (anchored, not restated) | the source text, marks by offset | ✓ standoff |
| statement / proposition (I-node) | `claim` marks | ✓ |
| `rel[implies]` + RA | `inference` marks (relations: implies / means / follows-from / consequently / **by**) | ~ partial (implies-family only) |
| `perf[Define]` | `definiendum`/`definiens` | ✓ |
| reference / `struct[used_in]` | enumerate-item **anaphora** (`label` antecedent + `anaphor` ref) | ~ partial (enumerate items only) |
| nesting = subgraph-in-slot (lemma) | inference **nest** depth (containment + chain) | ✓ for the cases seen |
| `perf[Assert/Suggest/Judge/Challenge/Query]` | — | ✗ not yet |
| `rel[not/equivalent/has_property/case_split/…]` | — | ✗ |
| `value[…]`, `meta[…]` | — | ✗ |

Verdict: we have the **content/proposition layer** + a slice of the
**inferential-relation layer** + an enumerate reference layer. The
**performative** layer (IAT's core anchoring) and the strategic layers
(`value`/`meta`) are absent. ~2 of 5 categories. A start, not complete.

## 3. What we detect (the marks)

All in `dp_paper_view.py` unless noted, emitted as standoff marks
`{start,end,layer:"dp",kind,…}` over the paper's `text`:

- **`detect_implications`** — `Let … . Then …` implication scopes (`implies`
  kind) + `kw-hyp`/`kw-con` keyword marks. (DC-3.)
- **`detect_inferences(text, marks)`** — the in-proof illative layer. Pivots:
  `implies (that)`, `means (that)`, `follows from (the fact that)`,
  `consequently`. Each → an `inference` mark carrying `subj_span`, `obj_span`,
  `nest`, and `fields [subject, relation, object]`; plus `claim` marks for the
  operands. Sentence-local clause extraction; restricted to env/proof regions.
- **`following/by ⟨ref⟩`** (inside `detect_inferences`) — a small embedded
  inference: *by premise-reference P, conclusion C* (e.g. L185 "following (1),
  δ_λ is a colimit cocone"). The reference rides with the arrow.
- **`detect_enumerate_anaphora(text)`** — `\item[(N)]` BINDS label (N) to its
  content (`label` mark = antecedent); each later `(N)` resolves to its
  **nearest preceding** same-label item (`anaphor` mark, short range). The IATC
  reference layer.
- **`detect_tex_environments`** — `\begin{NAME}…\end{NAME}` environment scopes
  (DC-9), delimiters included.

### Nesting (the part most worth verifying)

In `detect_inferences`, an inference **B nests one level inside A** when either:
- **containment** — B's arrow midpoint lies inside A's `subj_span` or `obj_span`
  (e.g. `following (1)` inside the `implies` premise); or
- **chain** — B's premise IS A's conclusion (overlapping spans, A before B),
  guarded so a containment pair is not also read as a chain (e.g. `consequently`
  inside `means`'s conclusion).
Depth (`nest`) is propagated to a fixpoint (4 passes). Rendered as the count of
nested pink rails.

## 4. The rendering principles (`dp_anatomy_html.py: render_marked_source`)

The renderer is **pure standoff**: the source text is rendered **once**; nothing
is restated (this was the hard-won correction — an earlier "broken-out block"
that copied claim text was abandoned as un-IATC). Rules:

1. **Sequential row numbers** — every rendered row is numbered `L1…LN` so any row
   is referenceable (source `\n`s within a reasoning region are collapsed to
   spaces so an operand is one flowing row).
2. **Reasoning regions** = tight clusters of claim+inference spans that do NOT
   cross a `. ` sentence boundary; each region absorbs its trailing period.
3. **Within a region**: each claim/operand is a row; the illative is a
   **standalone arrow row** (`=relation=>`, magenta, top CSS precedence); LHS
   and RHS split cleanly before/after it.
4. **Pink (magenta) bars** = `r-claim` rails, one per nesting depth; continuous
   within a region; a **nested** inference adds an extra indented pink rail.
5. **Per-scope breaks** — an env rail breaks (gap) only when *that* environment
   starts (so an inner scope ending, e.g. an enumerate, never breaks the outer
   proof rail); the pink rail breaks only between regions; descriptive prose
   between inferences sits outside the bars (no pink).
6. **A pink region requires illative structure** — a reasoning region draws a
   pink bar only if it contains at least one inference *arrow*. A `claim` mark
   with no arrow (e.g. an imperative "Now consider the category …" mis-tagged as
   a claim) is NOT a reasoning region and draws no bar — a bar without an arrow
   reads as a dangling annotation. Enforced render-side by filtering claim-only
   regions (2026-06-15; the defect recurred on 4/7 demo papers). The mis-tagged
   claim itself is a separate detector-precision question (it stays a claim mark,
   just unbracketed).
7. **One arrow, decorated once** — a single logical arrow can be SPLIT into
   adjacent `k-inf` spans when another mark overlaps it (e.g. the anaphor on the
   ref in "following (1)"). The `=`/`=>` decorations are per-span `::before`/
   `::after`, so a naive split doubles them: `=following =>=(1)=>`. Fixed CSS-side
   by decorating the *run* once — `=` only before the first segment
   (`.k-inf + .k-inf::before{content:""}`), `=>` only after the last
   (`.k-inf:has(+ .k-inf)::after{content:""}`) — so it reads `=following (1)=>`
   (2026-06-15, on `0905.0595` L191). A normal single-span arrow is unaffected.

## 5. Worked example — `0905.0595`, the proof of prop0.2 (rows ~L185–193)

Source: *"Since, following (1), δ_λ is a colimit cocone for each λ, (2) implies
that 1 is not λ-presentable for any regular λ. Condition (2) is not stated
explicitly in [AR] but it follows from the fact that there is no morphism from 1
to a non-terminal object of 𝒜."*

What our marks encode, as Clojure data:

```clojure
;; implies step (nest 0)
(infer :implies
  :premise  (and (infer :by                       ; "following (1)" — nest 1 (containment)
                   :premise    (ref (1))
                   :conclusion "δ_λ is a colimit cocone for each λ")
                 (ref (2)))
  :conclusion "1 is not λ-presentable for any regular λ")

;; follows-from step (nest 0, separate region; conclusion stated first in prose)
(infer :follows-from
  :premise    "there is no morphism from 1 to a non-terminal object of 𝒜"
  :conclusion (ref (2)))

;; reference layer (anaphora → enumerate bindings)
(bind (1) "For each regular cardinal λ, there is a λ-filtered diagram D_λ:𝒟_λ→𝒦
           whose only compatible cocones δ_λ are trivial ones (codomain 1)")
(bind (2) "For each λ, id₁ does not factorize through any component of δ_λ")
```

Visual mapping: each `(infer …)` is a pink region; depth = pink-rail count
(`by` is ▌▌ inside `implies`'s ▌); the arrow is a standalone magenta row;
`(ref n)` are blue anaphors resolving to the `(bind n …)` items.

### The logic-first model we actually want

Codex review (2026-06-15): the rendered standoff marks are useful evidence, but
the next artifact should be a small **Clojure IATC content graph** that we can
reason about and check directly. HTML bars are a view; the graph is the object.

For this passage, the clean model has two connected arguments plus a
bibliographic aside. The crucial distinction is that "following (1),
`δ_λ` is a colimit cocone..." is not the premise of `(2) implies ...`; it is a
**given/context condition** needed to apply the presentability warrant. The real
inference is:

```clojure
(infer :presentability-contradiction
  :given [(ref (1))
          (claim "δ_λ is a λ-filtered colimit cocone for each regular λ")]
  :premise (ref (2))
  :warrant "If 1 were λ-presentable, id_1 would factor through some component of the λ-filtered colimit cocone δ_λ"
  :conclusion "1 is not λ-presentable for any regular λ")
```

Separately, the passage justifies why `(2)` is true. "Condition (2) is not
stated explicitly in [AR]" is not part of the object-layer inference; it is
metadata attached to the assertion of `(2)`.

```clojure
(meta :bibliographic-aside
  "Condition (2) is not stated explicitly in AR")

(infer :derive-condition-2
  :premise "there is no morphism from 1 to a non-terminal object of A"
  :warrant "a factorization of id_1 through a component of δ_λ would require a morphism 1 -> that component"
  :conclusion (ref (2)))
```

The final sentences prove the premise of that second inference:

```clojure
(infer :no-map-from-1-to-nonterminal-A
  :assume "A is a non-terminal object of 𝒜"
  :premise
    (infer :definition-of-A
      :premise "𝒜 is the full subcategory of Gra consisting of graphs A with no morphism B_i -> A"
      :conclusion "if A ∈ 𝒜, then there is no morphism B_i -> A")
  :subproof
    (infer :contradiction
      :assume "there is a morphism 1 -> A"
      :step
        (infer :graph-semantics
          :premise "there is a morphism 1 -> A"
          :conclusion "A has a loop")
      :step
        (infer :constant-map
          :premise "A has a loop"
          :conclusion "there is a constant morphism B_i -> A")
      :contradicts "A ∈ 𝒜 has no morphism B_i -> A")
  :conclusion "there is no morphism from 1 to a non-terminal object of 𝒜")
```

As a dependency sketch:

```text
(1) + δ_λ colimit cocone
        given-context
             |
(2) no id_1 factorization through δ_λ components
             |
             v
1 is not λ-presentable for any regular λ


definition of 𝒜
+ morphism 1 -> A means loop in A
+ loop in A gives constant B_i -> A
             |
             v
no morphism 1 -> non-terminal A in 𝒜
             |
             v
(2)
```

This suggests the first Clojure IATC library vocabulary should include at least:

- `:given` — contextual facts required to apply a warrant.
- `:premise` — the proposition doing inferential work.
- `:warrant` — an implicit theorem, definition, or rule connecting premise to
  conclusion.
- `:meta` — bibliographic or expository comments, not object-layer claims.
- `:assume` / `:subproof` / `:contradicts` — nested derivations, especially
  contradiction arguments.
- `:conclusion` — the claim established by the inference.

So the answer to the open questions below is now: yes, add `:given` distinct
from `:premise`; yes, trim the `follows-from` conclusion to `(ref (2))`; and
yes, prioritize a checkable Clojure graph over further HTML renderer refinement.

## 6. Open questions / what remains to verify

1. **Resolved: `given` vs `premise`.** Add `:given` distinct from `:premise`.
   In the prop0.2 passage, `(2)` is the premise doing the inferential work;
   `following (1), δ_λ is a colimit cocone...` is context needed to apply the
   presentability warrant.
2. **Resolved: `follows-from` conclusion.** Trim the object-layer conclusion to
   `(ref (2))`; model "Condition (2) is not stated explicitly in [AR]" as
   `:meta`, not as the conclusion.
3. **Containment vs chain nesting** — verify the guard in `detect_inferences`
   doesn't mis-nest (e.g. an outer inference nesting under an inner one). Check
   `nest` levels on a second paper.
4. **Completeness** — the performative / value / meta layers are absent (§2). Is
   the current slice the right *first* slice, or should `perf[Assert]` (the
   default force) come before more `rel` types?
5. **Extraction precision** — sentence-local clause boundaries are heuristic;
   anaphor resolution is nearest-preceding within 2500 chars. Check false
   positives on a paper with equation numbers "(1)" that are NOT enumerate refs.

## 7. Verification checklist (commands)

- Regenerate the demo from the live detector (no `golden/` write, no JVM):
  `.venv/bin/python scripts/dp_anatomy_html.py 0905.0595 --remine`
- Lint gate (must hold): re-mine in-memory, run `check_invariants.check_paper` →
  expect `W-NEST-SCOPE == 0`, `wellformed_errors == 0`, `symbol_tagged == 1.0`,
  `symbol_grounded == 0.7143` (no coverage regression from the reasoning layer —
  `claim`/`inference`/`anaphor`/`label` are non-symbol, non-structural kinds).
- Inspect the marks: `inference` nest levels, `claim` spans, `anaphor` tips, and
  confirm `scope_crossings(marks) == []` (clean nesting — `wellformed.py`).
- Author ≠ reviewer: this excursion + the code were written by Opus; the gate
  numbers above are the falsifiable claims to re-derive.

## 7b. Breadth census (2026-06-15) — the overfitting check

Before declaring the model good, we measured IATC anchors across the **whole
dp-demo set** (7 papers), not just the tuned region. Per-paper inference /
claim / anaphor / binding counts, nesting depth, scope-crossings, wf:

| paper | infs | claims | anaph | binds | crossings | wf |
|---|---|---|---|---|---|---|
| 0710.2254 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0711.1761 | 2 | 4 | 0 | 42 | 0 | 0 |
| 0801.2567 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0807.1872 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0905.0595 | 5 | 9 | 9 | 10 | 0 | 0 |
| 1005.2653 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1012.1220 | 0 | 0 | 0 | 0 | 0 | 0 |

**Finding: the model is OVERFIT to `0905.0595`.** Five of seven papers yield
zero anchors; `0711.1761` finds 42 enumerate bindings but resolves 0 anaphors
(its reference style doesn't match the `(N)` resolver). No regressions, though —
`scope_crossings == []` and `wf == 0` on every paper, so the IATC layer + the
nesting reconciler are safe corpus-wide. This is exactly the
Distributed-Proofreaders trap (optimising one snapshot). The DP-for-IATC agenda
the census implies, loss-ranked:

1. **Illative coverage** (dominant) — the lexicon (`implies/means/follows from/
   consequently/by` + `Let…Then`) barely matches other papers. Measure the
   illatives that actually occur (`thus/hence/therefore/so/since/because/it
   suffices/we obtain/…`) and cover them. The reasoning-layer analogue of the
   "707K false unknowns" census in `M-distributed-proofreaders`.
2. **Anaphor resolution** — generalise beyond `(N)` nearest-preceding/2500-char
   (0711's 42 bindings, 0 resolutions).
3. **Exposition heuristic** — text at a nesting layer not covered by a sub-mark,
   flanked by marked siblings, is exposition (Joe). Examples in `0905.0595`:
   L194–197 (mid-proof), L226–231 (between proof and next prop), L249–251 (end
   matter). A candidate `exposition` kind; surfaces how much reasoning we are
   NOT capturing once anchors are denser.

Method: `scripts` build per paper + `check_invariants.check_paper`; counts over
`marks` by `kind`. Re-derivable; Codex should reproduce the sparsity and confirm
no crossings.



### Layer (a) broadening — DP iteration 1 (2026-06-15)

Measured the connectives that actually occur in demo proof regions (don't guess
the lexicon): `by`(37) `from`(20) `given`(19) `if/then`(14/13) `since`(13)
`so/thus/hence/therefore`(23) `we have/obtain`(6). Covered the dominant clean
shapes: **consequent-markers** (thus/hence/therefore/so/whence/accordingly —
premise = prior sentence, the proof-chain backbone), **if-then**, and
**equivalent / if-and-only-if** (`rel[equivalent]`). Anchor count per paper,
before → after:

| paper | before | after | note |
|---|---|---|---|
| 0710.2254 | 0 | 4 | equivalent×4 |
| 0711.1761 | 2 | 10 | so/thus/follows-from |
| 0801.2567 | 0 | 4 | equivalent, then |
| 0807.1872 | 0 | 0 | proofs ~383 chars (little reasoning) |
| 0905.0595 | 5 | 14 | +thus/hence/equivalent; maxnest 2 |
| 1005.2653 | 0 | 0 | NO proof/theorem envs (correct) |
| 1012.1220 | 0 | 0 | proof written as `Proof.` text, not `\begin{proof}` — missed |

4/7 papers now anchored (was 2). Gates unchanged on every paper
(`scope_crossings==[]`, `wf==0`, grounding stable). Remaining gaps are
STRUCTURAL, the next DP items:
- **text-style proofs** (`Proof.` / `\noindent{\bf Proof.}`) — env detection and
  the proof-restriction miss them (1012.1220).
- **reasoning outside `\begin{proof}`** — the proof-restriction excludes
  expository derivations; pairs with the exposition heuristic.
- `by`/`from`/`since`/`given` (the biggest raw counts) are premise/given/warrant
  markers with messier shapes — partly covered (`following/by ⟨ref⟩`), deferred.

Wiring: `detect_inferences`/`detect_enumerate_anaphora` run unconditionally in
`dp_paper_view.build()`; `dp_batch.py` calls `build(**FLAGS)` with all flags on —
so the live CPU runner emits the broadened layer (a) on its next iteration with
no further wiring.


### Layer (a) broadening — DP iteration 2: structural environments (2026-06-15)

0807.1872 surfaced two structural problems (Joe):
1. **Preamble tagged as content.** 0807's preamble is 48% of the file
   (10,794/22,589 chars) — all `\newcommand`/`\def`/`\newenvironment`. The env
   detector was matching the `\begin{...}` inside `\newenvironment{...}{...}`
   *definitions* as environment *uses* (92 spurious marks: env/beweis, env/redu,
   env/notation, …). **Fix:** drop every mark starting before `\begin{document}`
   in `build()` — the preamble is definitions, not content. Corpus-wide; 0807
   preamble marks 92 → 0, no gate change on any paper.
2. **Author-defined proof delimiters.** 0807's proofs are `\prf…\eprf` (5 each),
   invoking `\newenvironment{beweis}{\noindent{\bf Proof}…}{…□}` — not
   `\begin{proof}`, so they were invisible. **Fix:** `_ENV_CANON` maps
   multilingual proof-env names (beweis/pf/demo/preuve → proof), and
   `detect_proof_macros` recognises author proof-delimiter macro PAIRS
   (`\prf…\eprf`, `\bpf…\epf`, …) in the body as proof regions. 0807: 0 → 6
   inferences across 3 proofs.

After both: **5/7 demo papers anchored** (0710/0711/0801/0807/0905); gates clean
(`scope_crossings==[]`, `wf==0`, grounding stable). Remaining zeros are honest:
1005.2653 has no proof/theorem envs at all; 1012.1220 writes its proof as
`Proof.` plain text (no env, no macro pair) — the next structural item, alongside
relaxing the proof-restriction for expository reasoning.

## 8. Relationship to the catalogue & missions

The object-layer fixes (DC-1 terminology merge, DC-2 emphasis-definiendum,
DC-6 symbol juxtaposition, DC-9 environment scopes) are recorded in
`holes/dp-defect-catalogue.md` with their invariants. This excursion is the
**reasoning-layer** successor: where the catalogue made the *nodes* (objects,
defs, scopes) correct, IATC adds the *edges* (inferences) and *references*
(anaphora). Candidate promotion: a mission to build out the missing IATC
categories (performatives first) and export the marks as an IATC content graph
(the Clojure form in §5 as an actual artifact).
