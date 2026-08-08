# TN: Reconciling the two "cycle detectors" (R2b closure vs S7 CLean typing)

Status: COMPLETE (analysis 2026-08-08). All claims below measured on zone-joe
against `~/code/futon6/data/iatc-argument-graphs/run/` (98 graphs, `*.rung2.edn`
excluded) and `~/code/futon6/holes/clean-run/` (88 typed CLeans), plus the S7 run
logs `mark7z-s7.log` / `mark7z-s7-retry.log` / `mark7z-s7-sweep.log`.

## TL;DR

The two detectors agree far more than the 4/10 overlap suggests — **the
disagreement is almost entirely artifact, not graph topology**:

- **A-only 6 = `:given`-blindness.** Every one of A's six extra cycles routes
  through an edge's `:given` field. Detector B's loader reads only `:premise`,
  never `:given`, so those arcs simply do not exist in the CLean skeleton.
  Five of the six are `given ∋ conclusion` "self-loops" that are arguably
  extraction artifacts, not circular reasoning.
- **B-only 6 = a serialization bug, not cycles at all.** Every one of B's six
  extra rejects has an infer edge with a **vector-valued `:conclusion`**;
  `iatc_to_clean.cid()` assumes a scalar and renders the Python list repr into
  the EDN (`:produces :[Keyword(...), ...]`), which fails `clean_argcheck` gate
  **G1 (unreadable EDN)** — not G7 (cycle). The log line
  `REJECT <pid>: not a DAG comb (e.g. cyclic-equivalence)` is wrong for all six.
  Their inference arcs are acyclic; 3 of the 6 pass A's entire closure gate.
- **Overlap 4 = the genuine premise-level cycles**, visible to both detectors,
  and all four have the shape of an equivalence/iff proof flattened into two
  implications ("cyclic-equivalence"), not vicious circularity.

Also: the framing "Detector A rejects exactly 10" is true only counting
**cycle** findings. The full R2b closure gate FAILs **40/98** (orphan nodes
dominate). The 10-vs-10 symmetry is a coincidence of counting conventions.

## 1. What each detector actually tests (verified against code, not comments)

### Detector A — `check-closure` in `scripts/iatc_closure_check.bb`

Object: a **node-level digraph over the argument graph's nodes**, with arcs
derived from **ALL edges regardless of `:kind`** — `graph-arcs` (lines 60–67)
iterates `(:edges graph)` with no kind filter. Arc sources are every id found
under `:from :given :premise :assume :depends-on :contradicts :meta`
(`edge-sources`, lines 52–54); targets under `:to :conclusion`. `endpoint-ids`
recurses into vectors, so list-valued fields (incl. vector `:conclusion`)
flatten into multiple arcs without complaint.

FAIL reasons (any one suffices, lines 150–164): (1) cycle (self-loops checked
first); (2) orphan node (in-degree 0 AND out-degree 0); (3) no root; (4) no
terminal; (5) terminals unreachable from roots.

So the task framing needs two corrections: A does **not** operate on
"inference arcs" (it mixes in support/contradict/meta edges and the
`:given`/`:assume`/`:depends-on` context fields), and "A rejects 10" refers to
the cycle-reason subset only.

Measured (zone-joe, `bb scripts/iatc_closure_check.bb` over the 98): 40 files
FAIL closure; exactly 10 have a `cycle:` reason, matching the given A-list.
Cycle kinds: 5 self-loops, 5 proper 2–3-cycles.

### Detector B — S7, `clean_box_typing.py` + `clean_argcheck.bb`

Object: the **CLean skeleton** from `iatc_to_clean.build_skeleton` (lines
115–151), built as follows:

- `load_graph` (lines 90–112) keeps **only `:kind :infer` edges**; infer edges
  missing `:id` or `:conclusion` are **silently skipped** (line 102–103 — this
  bites: `0708.2185__p1 :e-cone-reflection-assumption` and
  `0806.1324__p4 :e-sub-lemma-stated` both lack `:conclusion` and vanish).
  Only `:premise` is read for antecedents. **`:given`, `:assume`,
  `:depends-on`, non-infer edges: all invisible to B.**
- Each surviving infer edge becomes a box (consumes = premise claim ids,
  produces = the conclusion claim id via `cid()`). Wires = claim flow between
  boxes (S→T iff S produces a claim T consumes), self-wires skipped. The
  skeleton is thus (approximately) the **line graph of the premise→conclusion
  claim-flow relation of the inference subgraph** — a projection of A's object
  that forgets `:given`-arcs, non-infer edges, and conclusion-less infer edges.
- `clean_argcheck.bb` gates G1–G8. **Only G7 is acyclicity** (Kahn over wires).
  G1 = parseable EDN + required keys; G2 box ids/method/text; G3 copar; G4 wire
  endpoints; G5 port typing; G6 hole fields; G8 shape. Any nonzero exit makes
  `clean_box_typing.py` (line 159–162) delete the output and log
  `REJECT <pid>: not a DAG comb (e.g. cyclic-equivalence)` — the message names
  only one of eight possible causes.

"Missing from `holes/clean-run/`" could in principle also mean load error or
LLM typing failure (logged FAIL, not REJECT). Measured from the S7 logs: the
2026-08-06 first run (`mark7z-s7.log`) lost 87 graphs to a llama-server outage
(`Connection refused`), but the retry (`mark7z-s7-retry.log`: rejected 9) and
sweep (`mark7z-s7-sweep.log`: rejected 1, `0706.1286__p10`) re-processed them;
the union of REJECTs is exactly the given B-10 and there are no residual
typing-FAILs. So **all 10 B-missing are genuine `clean_argcheck` rejects.**

### Formal relationship

A wire-cycle in B always implies a premise→conclusion node cycle in A's graph
(boxes in a wire cycle chain conclusions into premises). The converse fails
because A's arc set is a strict superset: `:given`-arcs, non-infer edges,
`:contradicts`/`:meta` arcs. So on cycles, **B ⊆ A necessarily** — modulo B's
independent non-cycle reject reasons (G1–G6, G8), which is where all six B-only
rejects actually come from.

## 2. Per-file diagnosis (all 16 flagged graphs)

Method: A's per-file reasons from a fresh closure run; B's gate identified by
regenerating each rejected skeleton with
`.venv/bin/python scripts/iatc_to_clean.py <graph> --out /tmp/tn-diag/<pid>.clean.edn`
(untyped — structural gates don't depend on typing) and running
`bb scripts/clean_argcheck.bb /tmp/tn-diag/`.

| pid | A cycle finding | B gate result | mechanism |
|---|---|---|---|
| 0706.1286__p26 | `:unit-isomorphism -> :companions-psfr -> :unit-isomorphism` | **G7 cycle** | premise-level 2-cycle (both see it) |
| 0706.1286__p3 | `:prop-2nd-cond -> :iso-condition -> :prop-2nd-cond` | **G7 cycle** | premise-level 2-cycle |
| 0806.1324__p4 | `:frac-set-lemma -> :fraction-reduction -> :small-set-of-fractions -> :frac-set-lemma` | **G7 cycle** | premise-level 3-cycle (plus a second: `bijection-claim ↔ {surjectivity, injectivity}`) |
| 0905.2621__p11 | `:acyc-plus-def -> :acyc-co-containment -> :acyc-plus-def` | **G7 cycle** | premise-level 2-cycle |
| 0706.1286__p7 | 3-cycle `:lax-framed-functor -> :natural-bijection -> :vertical-full-faithful -> …` | PASS (typed, in clean-run) | closing arc is `:e-given-from-vertical-horizontal`'s **`:given [:vertical-full-faithful :horizontal-full-faithful]`** — invisible to B |
| 0708.1921__p2 | self-loop at `:lax-natural-transformations` via `:e-lax-natural-transformations` | PASS | edge has `:given [:tilde-p-r-natural :lax-natural-transformations]` and `:conclusion :lax-natural-transformations` — `given ∋ conclusion` |
| 0708.2185__p1 | self-loop at `:subobject-split` via `:e-diagram-commutes` | PASS | `:given [:subobject-split]`, `:conclusion :subobject-split` |
| 0806.1324__p15 | self-loop at `:H-bar-definition` via `:e-H-bar-existence` | PASS | `:given [:tilde-H-definition :H-bar-definition]`, `:conclusion :H-bar-definition` |
| 0806.1324__p16 | self-loop at `:q-star-adjunction` via `:e5` | PASS | `:given [:q-star-adjunction]`, `:conclusion :q-star-adjunction` |
| 2311.05789__p0 | self-loop at `:unit-coherence-diagram` via `:e-coherence` | PASS | `:given [:unit-coherence-diagram]`, `:conclusion :unit-coherence-diagram` |
| 0706.1286__p10 | no cycle (closure FAIL: orphan `:g-strong`) | **G1 unreadable** | vector `:conclusion` → `:produces :[Keyword(g0-has-left-adjoint), Keyword(g1-has-left-adjoint), Keyword(lf1-iso-f0l), Keyword(rf1-iso-f0r)]` |
| 0706.1286__p24 | no cycle (closure PASS 1.000) | **G1 unreadable** | `:produces :[Keyword(unit-isomorphism-bad-square), Keyword(odot-preserves-opcart-bad-square)]` |
| 0708.2185__p2 | no cycle (closure PASS 1.000) | **G1 unreadable** | `:produces :[Keyword(rlp-implies-injective), Keyword(injective-implies-rlp)]` |
| 0806.1324__p9 | no cycle (closure FAIL: 4 orphans) | **G1 unreadable** | `:produces :[Keyword(ker-ga-im-l), Keyword(im-ga-ker-l)]` |
| 0905.0465__p0 | no cycle (closure PASS 1.000) | **G1 unreadable** | `:produces :[Keyword(dual-space-exists), Keyword(coevaluation-only-finite-dim)]` |
| 0905.0465__p5 | no cycle (closure PASS 1.000) | **G1 unreadable** | `:produces :[Keyword(z0-equiv-data), Keyword(nondegenerate-eta)]` |

Cross-check: a corpus-wide scan (using the pipeline's own `_edn_safe` loader)
finds vector `:conclusion` in **exactly** those six files and no others, and
finds **zero** true `premise == conclusion` (X⇒X) infer edges in the whole
corpus. All six A-only graphs have typed outputs in `holes/clean-run/`.

## 3. Worked examples

### A-only, worked end to end: `0708.1921__p2` (self-loop that types cleanly)

The graph's edge (verbatim from `data/iatc-argument-graphs/run/0708.1921__p2.edn`):

```clojure
{:id :e-lax-natural-transformations, :kind :infer, :relation :because,
 :given [:tilde-p-r-natural :lax-natural-transformations],
 :premise :oxlax-theorem,
 :warrant {:kind :claim, :text "naturality of tilde-p and tilde-r plus definitions …"},
 :conclusion :lax-natural-transformations, …}
```

- Detector A: `edge-sources` includes `:given`, so it emits the arc
  `:lax-natural-transformations → :lax-natural-transformations`; `self-loop`
  fires ⇒ `cycle: self-loop at node :lax-natural-transformations via edge
  :e-lax-natural-transformations`.
- Detector B: `load_graph` reads only `:premise`; the box for this edge is
  `consumes [:oxlax-theorem] produces :lax-natural-transformations`. No
  self-reference exists at any point; the wire graph is a DAG; all gates pass;
  `holes/clean-run/0708.1921__p2.clean.edn` exists with `:clean/typing-source :llama`.

Reading the text fields: the `:given` is the *statement being established*
("p and r are components of lax natural transformations…") listed as context
for a because-relation. The self-loop is an extraction quirk (the LLM citing
the conclusion among its givens), not the paper reasoning circularly. Same
shape in all five self-loop cases.

### A-only, the one non-self-loop: `0706.1286__p7`

Premise-only arcs: `:def-ff → :given-condition → :lax-framed-functor →
:natural-bijection → {:vertical-full-faithful, :horizontal-full-faithful}` — a
DAG. The cycle A reports closes only through
`:e-given-from-vertical-horizontal` = `{:given [:vertical-full-faithful
:horizontal-full-faithful], :premise :given-condition, :conclusion
:lax-framed-functor}`. This is the two directions of a biconditional (the
forward direction proved from the definition, the converse deriving
full-faithfulness *from* lax-framedness) sharing nodes — an iff flattened into
implications. B sees only the premise arcs and types it (5 boxes, DAG).

### B-only, worked end to end: `0706.1286__p10` (acyclic but rejected)

The graph has an infer edge whose `:conclusion` is a **vector** of four claim
ids. `cid(x)` (`iatc_to_clean.py` line 57–59) handles `int` and keyword only;
a vector falls through to `kw(x)` and stringifies the Python object. The
rendered skeleton (regenerated) contains, at `:clean/boxes` box `:e-backward`:

```
:produces :[Keyword(g0-has-left-adjoint), Keyword(g1-has-left-adjoint), Keyword(lf1-iso-f0l), Keyword(rf1-iso-f0r)]
```

`clean_argcheck.bb` G1 fails with `unreadable EDN — Invalid token: :`, exit 1,
and `clean_box_typing.py` logs `REJECT 0706.1286__p10: not a DAG comb (e.g.
cyclic-equivalence)`. The inference arcs are acyclic (A finds no cycle; its
closure FAIL for this file is an unrelated orphan, `:g-strong`).

### B-only, second example: `0708.2185__p2`

Same mechanism: `:produces :[Keyword(rlp-implies-injective),
Keyword(injective-implies-rlp)]` → G1. This graph **passes A's entire closure
gate at rate 1.000** (and warrant-resolution at 1.000) — by A's lights it is
one of the cleanest graphs in the corpus, yet S7 drops it, mislabeled cyclic.
Note the conclusion names: `rlp-implies-injective` / `injective-implies-rlp` —
again an equivalence, this time expressed as a two-element conclusion vector
rather than as two edges (which is why it did NOT produce a cycle).

### Overlap, for contrast: `0706.1286__p26`

`:e-psfr` `{:premise [:unit-isomorphism :comp-isomorphism], :meta
[:easy-check]} → :companions-psfr`; `:e-unit-iso` `:premise :companions-psfr →
:unit-isomorphism`. Premise-level 2-cycle ⇒ A reports it; in B, box `e-psfr`
consumes `c:unit-isomorphism` which box `e-unit-iso` produces while consuming
`c:companions-psfr` which `e-psfr` produces ⇒ wire cycle ⇒ G7. Both fire, and
correctly. Content-wise it is again equivalence-shaped ("X are companions iff
the unit/composite 2-cells are isomorphisms").

## 4. Adjudication

**Neither detector is computing "circular reasoning"; they compute different
structural properties of different objects, and both of their labels mislead.**

1. **They test different graphs.** A: node digraph over all edge kinds and all
   seven source fields. B: wire graph over infer-edges' premise→conclusion
   claim flow only (a projection that deletes `:given`, `:assume`,
   `:depends-on`, `:contradicts`, `:meta`, non-infer edges, and
   conclusion-less infer edges). The projection can only destroy cycles, never
   create them — confirmed empirically: B's cycle set (4) ⊂ A's cycle set (10).
2. **Their reject sets are unions over different failure classes.** A's
   headline "10" is cycles only (the full gate fails 40/98, mostly orphans).
   B's "10" is 4 cycles + 6 instances of one serialization bug surfacing as
   unparseable EDN. The apparent 6/6 asymmetry decomposes exactly:
   A-only 6 = `:given`-arcs B cannot see; B-only 6 = the vector-`:conclusion`
   bug A is immune to (its `endpoint-ids` flattens vectors).
3. **Which notion should a capability-proof reader care about?** B's G7 — a
   cycle in premise→conclusion claim flow — is the right formalization of
   *inferential* circularity for a proof skeleton. A's cycle test is
   over-inclusive as a circularity check: treating `:given` (and `:contradicts`,
   `:meta`) as inference sources manufactures cycles out of context references;
   5 of A's 10 findings are `given ∋ conclusion` artifacts of extraction. But
   A's closure gate as a whole is testing something else and also legitimate:
   *argument-graph well-formedness* (connectivity, roots, terminals) at the
   extraction layer, where `:given` arcs rightly count as references.
4. **Is either simply wrong?** Two genuine defects, both on the B side of the
   pipeline (in the producer, not the checker):
   - `iatc_to_clean.py` mishandles vector `:conclusion` (6/98 graphs dropped
     for a non-topological reason). The graphs are well-formed by A's checker;
     the multi-conclusion shape is expressible (one box per conclusion, or a
     product claim). This is a real corpus loss: 3 of the 6 pass the full R2b
     closure gate.
   - The REJECT log message asserts "not a DAG comb (e.g. cyclic-equivalence)"
     for any nonzero `clean_argcheck` exit; for 6 of the 10 observed rejects
     that diagnosis is false. `clean_argcheck` itself reports precise per-gate
     errors — the driver discards them (`>/dev/null 2>&1`).
   On the A side, no bug, but a labeling hazard: its "cycle" findings on this
   corpus are dominated by `:given` artifacts, and every genuine premise-level
   cycle found by either detector (the overlap 4) is equivalence/iff structure
   ("cyclic-equivalence"), not question-begging. A reader of the capability
   proof should treat "cycles detected" claims as *extraction-artifact +
   equivalence-flattening* findings unless a premise-level cycle is exhibited.

## 5. Side findings (not part of the question, but measured en route)

- **The ";; NOTE: dropped N self-loop inference(s) (vacuous step, X⇒X)" comment
  is wrong in all 4 files that carry it** (`0706.1286__p29`, `0708.3398__p25`,
  `0905.2621__p1`, `2311.05789__p1`): the vacuous test
  `all(p == e["conclusion"] for p in e["premise"])` is vacuously true for
  **empty-premise** edges, and every flagged edge in those files is
  empty-premise, not X⇒X (the corpus contains zero X⇒X edges). Nothing is
  actually dropped either — the count only feeds the comment; the boxes are
  still emitted.
- B silently skips infer edges missing `:conclusion` (2 observed:
  `0708.2185__p1 :e-cone-reflection-assumption`, `0806.1324__p4
  :e-sub-lemma-stated`), so even its infer-subgraph is a proper subgraph.
- 48/98 graphs use `:given` on at least one edge — the A/B object divergence is
  corpus-wide, not confined to the 6 disagreement cases.
- The 2026-08-06 S7 outage run (`mark7z-s7.log`, 87 connection-refused FAILs)
  was fully recovered by the retry + sweep runs; the current clean-run/88 state
  reflects `REJECT`s only.

## 6. What was not determined

- Whether the five `given ∋ conclusion` self-loops are one systematic LLaMA
  extraction habit or five accidents — would need the source passages.
- What the intended semantics of vector `:conclusion` is (the extractor prompt
  wasn't examined); the six cases all look like two-directions-of-an-iff or
  multi-part claims.
- Whether A *should* include `:contradicts`/`:meta` as arc sources (design
  intent unknown; on this corpus `:meta [:easy-check]` in `0706.1286__p26`
  contributes an arc from a node that is arguably not a claim).
