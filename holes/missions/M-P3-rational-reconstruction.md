# Mission: P3 Rational Reconstruction

**Date:** 2026-03-06
**Status:** IDENTIFY
**Owner:** futon6, with pattern discipline from futon3
**Depends on:** First-proof P3 material (complete), synthetic QA batch
  (complete), draft provenance standard (futon3, complete)
**Enables:** Provenance standard prototype validation, math-strategy pattern
  library seed, futon6 pattern discipline model

## 1. IDENTIFY

### Motivation

Problem 3 (existence of a CTMC with modified Macdonald polynomial stationary
distribution) has a complete solution sketch (`problem3-solution.md`), a
reviewer assessment (REVIEWER.md: 2 major, 1 medium finding), a Mermaid wiring
diagram, Codex repair cycles, and 8 synthetic QA pairs decomposing the proof
into sub-questions. The mathematical content exists.

What's missing is the *decision trace*. The solution was developed ad hoc:
steps were chosen, decompositions were made, notation bridges were asserted —
but no record exists of *why* those moves were made, what alternatives were
considered, or what patterns (if any) governed the proof strategy. The
reviewer found gaps (star/non-star bridge asserted not proved, irreducibility
too compressed, positivity needs citation) — but we can't trace those gaps
back to the decision that produced them.

This mission replays the P3 proof development with pattern discipline,
producing:

1. **Virtual PSRs** — at each proof decision point, what pattern would have
   been selected? If no pattern exists, write one.
2. **Virtual PURs with mathematical scope** — after each step, what was the
   outcome? Scope is the proof step (s1, s4, s6, etc.)
3. **Alternative decompositions** — did the pattern-theoretic framing suggest
   different sub-problems, orderings, or strategies than the original?
4. **New patterns** — seed a `math-strategy/` namespace in the pattern library
   with reusable proof strategy patterns discovered during reconstruction.

### Theoretical anchoring

- **Draft provenance standard** (`futon3/docs/draft-provenance-standard.md`):
  PUR carries scope, PSR scope optional, graph is system of record. This
  mission is the primary prototype.
- **Rational reconstruction** (M-futon-enrichment precedent): replay
  development as if discipline had been in place, recording findings as
  timestamped layers. Each replay pass is itself evidence.
- **Baldwin cycle**: individual learning (this proof needed this move) →
  population knowledge (pattern library gains a reusable strategy).

### Scope

**In scope:**
- Replay P3 proof decomposition with virtual PSR/PUR
- Write new `math-strategy/` patterns as discovered
- Compare pattern-guided decomposition with original
- Record mathematical scope annotations on PURs
- Assess whether reviewer-identified gaps would have been caught by
  pattern discipline

**Out of scope:**
- Actually fixing the P3 gaps (that's a separate repair mission)
- Building tooling for automated scope detection (future work)
- Replaying other first-proof problems (follow-on missions)
- Ingesting results into futon1a hyperedge store (Phase 3+ of
  M-futon-enrichment)

### Completion criteria

1. Every decision point in the P3 proof has a virtual PSR (pattern chosen
   or pattern gap identified)
2. Every proof step has a virtual PUR with mathematical scope
3. At least 3 new `math-strategy/` patterns written to futon3/library/
4. A comparison table: original decomposition vs pattern-guided
   decomposition, noting where they diverge
5. Assessment: would pattern discipline have caught the reviewer's 3
   findings earlier?

### Source material

- `futon6/data/first-proof/problem3-solution.md` — the proof
- `futon6/data/first-proof/problem3-wiring.json` — wiring diagram
- `futon6/data/first-proof/problem3-v1.mmd`, `problem3-v2.mmd` — Mermaid
- `futon6/data/synthetic-qa/synth-p3-*.json` — 8 synthetic QA pairs
- `futon6/data/synthetic-qa/problem3-prompts.jsonl` — generation prompts
- `futon6/REVIEWER.md` §Problem 3 Findings — 3 findings (2 major, 1 medium)
- `futon3/docs/draft-provenance-standard.md` — the standard being prototyped
- `futon3/library/math-informal/` — 25 existing math patterns (content, not
  strategy)

### Relationship to other missions

- **M-futon-enrichment** — this mission produces the first mathematical
  provenance data. If/when we ingest it into the hyperedge store, it
  connects the enrichment pipeline to mathematical reasoning.
- **M-three-column-stack** — the math column is currently the sparsest.
  Pattern-backed proof provenance would populate it with structured edges.
- **Future P3 repair mission** — the gaps identified here (and potentially
  reframed by pattern analysis) feed directly into a repair mission.
- **Future Pn reconstruction missions** — P7, P8 (critical), P2 (critical)
  are candidates for the same treatment once the method is validated on P3.

## 2. MAP

### Q1: Decision points in the P3 proof

The proof has 7 sections (solution.md) corresponding to 8 wiring nodes:

| Node | Section | Decision | What was chosen |
|------|---------|----------|-----------------|
| p3-problem | — | Problem framing | Existence question for CTMC with F*/P* stationary law |
| p3-s1 | §2 | Construction strategy | Exhibit the t-PushTASEP explicitly |
| p3-s2 | §2 (cont.) | Formalization | Write explicit generator q(η,η') |
| p3-s3 | §3 | Nontriviality argument | Show rates don't use F* values |
| p3-s4 | §4 | Main theorem | Invoke AMW Theorem 1.1 |
| p3-s5 | §5 | Notation bridge | Assert F*=F at q=1 (or cancel in ratio) |
| p3-s6 | §6 | Sanity check | Compute n=2 case explicitly |
| p3-s7 | §7 | Conclusion | Compose s3+s4+s5 |

The key strategic decisions are at **s1** (choosing to construct rather
than prove abstractly), **s4** (choosing which theorem to invoke), and
**s5** (choosing to assert rather than prove the notation bridge).

### Q2: Existing patterns that apply

| Decision point | Candidate pattern | Fit |
|----------------|-------------------|-----|
| p3-s1 (construct chain) | `math-informal/construct-an-explicit-witness` | Strong: "build the object step by step" |
| p3-s4 (invoke AMW thm) | `math-informal/reduce-to-known-result` | Strong: "transform until it matches a known theorem" |
| p3-s6 (n=2 check) | `math-informal/try-a-simpler-case` | Strong: "reduce parameters to smallest non-trivial" |
| p3-s5 (notation bridge) | `math-informal/transport-across-isomorphism` | Weak: notation conventions aren't isomorphisms |
| p3-s1→s4 (local rates→global stationary) | `math-informal/local-to-global` | Medium: the CTMC has local transitions and a global stationary law, but the connection is via the theorem, not via a gluing argument |

**Gaps** (no existing pattern fits well):

- **p3-s5**: The notation bridge is a *convention alignment* move — showing
  two notations refer to the same object (or differ by a known factor).
  No existing pattern addresses this. Candidate: `math-strategy/convention-bridge`.
- **p3-s3**: The nontriviality argument is a *meta-mathematical* move —
  showing the construction doesn't circularly depend on the target.
  Candidate: `math-strategy/non-circularity-check`.
- **p3-s7**: The conclusion is a *composition* move — assembling independent
  pieces (construction, theorem, notation) into a complete answer.
  Candidate: `math-strategy/compose-independent-lemmas`.

### Q3: Synthetic QA decomposition

The synthetic QA generated 4 step-level entries (problem, s1, s4, s6),
each targeting a gap identified by the reviewer:

| Synth step | Wiring node(s) | Gap targeted | Reviewer finding |
|------------|----------------|--------------|------------------|
| p3-problem | p3-s5 | Star/non-star bridge asserted not proved | Finding 1 (major) |
| p3-s1 | p3-s1 | Truncated geometric needs Pieri justification | — (not flagged) |
| p3-s4 | p3-s2/s4 | Irreducibility claimed not proved | Finding 2 (major) |
| p3-s6 | p3-s4 | Positivity needs direct citation | Finding 3 (medium) |

Note: the synthetic QA skipped s2, s3, s5, s7 — these are the nodes
that are either clarifications (s2), meta-arguments (s3), the notation
bridge itself (s5, which is the *subject* of the p3-problem QA), or the
conclusion (s7, which composes rather than discovers).

### Q4: Wiring diagram dependency structure

The dependency graph (from `problem3-wiring.json`, 10 edges) reveals:

```
p3-problem ← p3-s1 (reform: construct candidate)
                ├── p3-s2 (clarify: explicit generator)
                ├── p3-s3 (assert: nontriviality)
                ├── p3-s6 (exemplify: n=2 sanity)
                └── p3-s4 (reference: applies to same dynamics)
                      └── p3-s5 (clarify: notation bridge)
p3-problem ← p3-s4 (assert: AMW theorem)
p3-problem ← p3-s7 (assert: conclusion)
                ├── references p3-s3
                └── references p3-s4
```

Two independent paths converge on the problem:
- **Construction path**: s1 → s2/s3/s6 (build the chain, show it works)
- **Theorem path**: s4 → s5 (invoke the result, bridge the notation)
- **Conclusion**: s7 composes both paths

This two-path structure is itself a pattern: construct an explicit object
AND invoke a theorem about it. The construction path provides nontriviality;
the theorem path provides the stationary distribution. Neither alone
suffices.

### Q5: Reviewer findings → decision points

| Finding | Severity | Decision point | Pattern gap |
|---------|----------|----------------|-------------|
| Star/non-star bridge asserted | Major | p3-s5 | No `convention-bridge` pattern; the PSR would have flagged "I'm asserting this, what pattern governs assertion?" |
| Irreducibility too compressed | Major | p3-s7 (cites s3 but s3 doesn't prove irreducibility) | The `compose-independent-lemmas` pattern would have required checking each piece is actually proved |
| Positivity needs citation | Medium | p3-s4 | `reduce-to-known-result` pattern says "verify missing hypotheses" — the PSR should have listed positivity as a hypothesis to check |

### Ready vs missing

**Ready (no new work needed):**
- P3 proof content: 7 sections, 8 wiring nodes, 10 edges
- Synthetic QA: 8 files covering 4 gap-targeted steps
- Existing patterns: 3 strong fits (construct-witness, reduce-to-known,
  try-simpler-case)
- Reviewer findings: 3 findings mapped to decision points
- Wiring diagram: full dependency graph

**Missing (the actual work):**
- 3 new patterns to write: `convention-bridge`, `non-circularity-check`,
  `compose-independent-lemmas` → **DONE** (written during DERIVE)
- Virtual PSR for each of the 7 decision points
- Virtual PUR with mathematical scope for each step
- Comparison analysis: original vs pattern-guided decomposition
- Counterfactual assessment: would patterns have caught the 3 findings?

## 3. DERIVE

### 3.1 New patterns

Three new patterns written to `futon3/library/math-strategy/`:

| Pattern | Sigil | Addresses |
|---------|-------|-----------|
| `math-strategy/convention-bridge` | 🌉/桥 | p3-s5: notation bridge between F* and F |
| `math-strategy/non-circularity-check` | 🔄/非 | p3-s3: construction doesn't use target |
| `math-strategy/compose-independent-lemmas` | 🧩/合 | p3-s7: assembling the final answer |

Each follows the `math-informal/` flexiarg format:
IF/HOWEVER/THEN/BECAUSE with NEXT-STEPS. The key design choice: these are
*strategy* patterns (how to structure a proof move) not *content* patterns
(what mathematical concepts to invoke). The `math-informal/` namespace
already has content patterns; `math-strategy/` captures proof architecture.

### 3.2 Virtual PSR/PUR format

A virtual PSR/PUR reconstructs the pattern discipline that *would have*
operated at each decision point. The format extends the standard PSR/PUR
with mathematical scope:

**Virtual PSR:**
```
## vPSR (Virtual Pattern Selection Record)
- **Step**: p3-s1 (construct chain)
- **Pattern chosen**: `math-informal/construct-an-explicit-witness`
- **Candidates considered**: construct-an-explicit-witness,
    use-probabilistic-method, reduce-to-known-result
- **Rationale**: The problem asks "does there exist"; construction gives
    the strongest answer and produces a computable object.
- **Scope**: p3-s1 (§2 Lemma — explicit chain construction)
- **Scope granularity**: proof-step
- **Confidence**: high (construction is the natural move for existence)
```

**Virtual PUR:**
```
## vPUR (Virtual Pattern Use Record)
- **Step**: p3-s1 (construct chain)
- **Pattern**: `math-informal/construct-an-explicit-witness`
- **Scope**: p3-s1 (§2 Lemma — explicit chain construction)
- **Actions taken**: Defined t-PushTASEP on n-site ring with rates 1/x_j,
    truncated geometric species selection, push-cascade termination.
- **Outcome**: success — explicit chain constructed
- **Prediction error**: low — construction is standard for this class
- **Gap detected**: The truncated geometric probability t^(k-1)/[m]_t is
    stated but its derivation from Pieri coefficients is not shown here
    (delegated to a separate step). The construct-witness pattern says
    "verify each required property" — this verification is incomplete.
- **Reviewer alignment**: No reviewer finding on this step directly, but
    the incomplete verification is the kind of gap that accumulates.
```

**Scope granularity for mathematics** (discovered during MAP):

| Granularity | Example | When to use |
|-------------|---------|-------------|
| `problem` | P3 (existence of CTMC) | Top-level claim |
| `proof-step` | p3-s1 (construct chain) | Named step in the proof |
| `sub-step` | p3-s1/Pieri-derivation | Specific argument within a step |
| `expression` | e2 (F_{η^(k)} formula) | Particular formula or equation |

Most vPURs will be proof-step scoped. Sub-step scope is for when a single
step contains multiple distinct arguments (like s1 containing both the
construction and the probability justification).

### 3.3 The full virtual replay

Each of the 8 wiring nodes gets a vPSR + vPUR. The replay proceeds in
wiring dependency order (leaves first, root last):

| Order | Node | Pattern | Key question |
|-------|------|---------|--------------|
| 1 | p3-s1 | `construct-an-explicit-witness` | What object to build? |
| 2 | p3-s2 | `construct-an-explicit-witness` (cont.) | How to formalize it? |
| 3 | p3-s3 | `non-circularity-check` | Is the construction independent of the target? |
| 4 | p3-s4 | `reduce-to-known-result` | Which theorem applies? |
| 5 | p3-s5 | `convention-bridge` | Do the notations match? |
| 6 | p3-s6 | `try-a-simpler-case` | Does n=2 work? |
| 7 | p3-s7 | `compose-independent-lemmas` | Do the pieces assemble? |

Note: p3-problem (the question itself) does not get a PSR — it's the
input, not a decision. The decomposition into s1–s7 is itself a strategic
decision, but it's the *proof architecture*, not a single pattern
application. The two-path structure (construction + theorem) is an
emergent property of the pattern choices, not a pattern applied upfront.

### 3.4 Comparison method

After the replay, produce a comparison table:

| Aspect | Original (ad hoc) | Pattern-guided (replay) |
|--------|-------------------|------------------------|
| Decomposition | 7 sections | Same 7 nodes (or different?) |
| Notation bridge | Asserted (§5) | `convention-bridge` demands proof |
| Nontriviality | 3 sentences (§3) | `non-circularity-check` demands ingredient trace |
| Conclusion | Narrative (§7) | `compose-independent-lemmas` demands dependency graph + trace |
| Positivity | Mentioned in passing | `reduce-to-known-result` says "verify missing hypotheses" |
| Irreducibility | "optional" (§7) | `compose-independent-lemmas` would flag: is this piece needed or not? |

The comparison is not about whether the pattern-guided version is
"better" — it's about whether the patterns *surface gaps earlier*.
If a PSR at s5 would have said "I need `convention-bridge` but I'm
asserting instead of proving," that's a detection the original missed.

### 3.5 Counterfactual assessment design

For each reviewer finding, answer:

1. **Which vPSR covers this decision point?**
2. **Does the pattern's HOWEVER clause warn about this specific risk?**
3. **Does the pattern's THEN clause prescribe the action the reviewer
   recommended?**
4. **Would the vPUR's "gap detected" field have flagged the issue?**

If all four answers are yes for a finding, the pattern discipline would
have caught it. If some are no, that reveals either a pattern gap (the
pattern doesn't warn about this risk) or a discipline gap (the pattern
warns but the practitioner might skip the check).

### 3.6 IF/HOWEVER/THEN/BECAUSE for key design decisions

**D1: Strategy patterns vs content patterns**

IF: The `math-informal/` namespace has 25+ content patterns but no
proof-architecture patterns.
HOWEVER: Mixing strategy and content in one namespace would blur the
distinction between "what mathematical concept to use" and "how to
structure the proof move."
THEN: Create a separate `math-strategy/` namespace for proof architecture.
BECAUSE: The distinction mirrors code patterns: `storage/` patterns govern
how to build a storage system (strategy), while `futon-theory/` patterns
describe what the theory says (content). Keeping them separate lets you
compose them: "use construct-an-explicit-witness (strategy) to build a
chain whose properties are governed by Macdonald polynomial theory
(content)."

## 4. ARGUE

### Patterns governing this mission

This mission is itself governed by patterns — not the `math-strategy/`
patterns it creates, but higher-level patterns that shaped the decision
to do this work and the method used. Making these explicit is the ARGUE
phase's job: showing that the design is coherent with the stack's
theoretical commitments, not just workable.

**`enrichment/rational-reconstruction` [日/引]** — the direct ancestor.
M-futon-enrichment used rational reconstruction to build the codebase's
self-representation through layered replay rather than bulk import. This
mission applies the same method to mathematical proof rather than code:
replay P3's development as if pattern discipline had been in place,
recording findings as evidence layers. The three advantages transfer:
replayability (vPSR/vPUR records can be queried), incremental correction
(early virtual records may be revised as we learn what scope granularity
works), and methodology transfer (if it works for P3, it works for P7,
P8, and beyond).

**`futon-theory/baldwin-cycle` [🔃/三]** — the theoretical frame. The
three new `math-strategy/` patterns are Baldwin Phase 1 artifacts:
learned behaviors that emerged from examining P3. If the virtual replay
confirms they catch real gaps, they enter Phase 2 (assimilation into the
library as reusable patterns). If future missions confirm they work
across problems, Phase 3 (canalization — they become standard proof
hygiene, expected of every proof, not optional heuristics). This mission
is explicitly Phase 1: exploration. We do not yet know if these patterns
are good.

**`futon-theory/retroactive-canonicalization` [🔄/溯]** — the deeper
logic. The P3 proof was developed ad hoc. The virtual reconstruction
retroactively identifies which decisions *were* pattern applications
(even though no pattern was named at the time) and which were pattern
*gaps* (decisions where no pattern existed and the proof suffered for
it). The reviewer's 3 findings are evidence that certain patterns were
structurally necessary — the proof broke precisely where discipline was
absent. Retroactive canonicalization says: the structure was always
there; the naming capacity to see it was not. The vPSRs provide that
naming capacity after the fact.

**`f6/pattern-as-strategy` [🎯/策]** — the bridge to agents. The
`math-informal/` patterns are already positioned as agent-operational
strategies (per Corneli 2014 §10.6). The `math-strategy/` patterns
extend this from content strategies ("try induction") to architectural
strategies ("check that your construction isn't circular"). If agents
can use `convention-bridge` as a prompt to verify notation equivalences
before asserting them, that's the pattern-as-strategy vision operating
at a higher level of proof structure.

### Why this design, not alternatives

The obvious alternative is: just fix the P3 gaps directly. Write the
missing bridge lemma, add the irreducibility proof, cite positivity.
This would be useful but would teach us nothing about methodology.
The rational reconstruction approach costs more upfront but produces
transferable artifacts (patterns, vPSR/vPUR format, scope granularity
model) that apply to all 10 first-proof problems and to future
mathematical work in the stack.

A second alternative: apply pattern discipline only to *new* proofs
going forward, skip the reconstruction. This avoids the hindsight bias
of virtual records but loses the opportunity to validate the method
against known ground truth (the reviewer's findings). P3 is a
controlled experiment: we know where the gaps are, so we can test
whether the method would have found them.

### Trade-off summary

| Given up | Gained |
|----------|--------|
| Speed (reconstruction is slower than direct repair) | Validated methodology that transfers to Pn |
| Certainty (virtual records are retrospective, not real-time) | Controlled test against known reviewer findings |
| Simplicity (could just fix P3) | Three reusable patterns + scope granularity model + vPSR/vPUR format |

### Generalization notes

The method generalizes along two axes:

1. **To other first-proof problems.** P7 and P8 have critical gaps; P2
   has a critical gap. Each could receive the same treatment. The
   patterns discovered here (convention-bridge, non-circularity-check,
   compose-independent-lemmas) may apply directly or need extension.

2. **To non-mathematical domains.** The vPSR/vPUR format with domain-
   specific scope levels works for any domain where decisions are made
   and outcomes evaluated. The provenance standard's code scope
   (repo/namespace/var) and mathematical scope (problem/step/sub-step/
   expression) are two instances of the same structure. A third instance
   (e.g., for design decisions, business strategy) would confirm the
   pattern is general.

**D2: Virtual vs actual PSR/PUR**

IF: We want to reconstruct pattern discipline on an already-completed proof.
HOWEVER: Actual PSRs record real-time decisions; virtual PSRs reconstruct
decisions after the fact, with hindsight.
THEN: Prefix with "v" (vPSR, vPUR) and include a "reviewer alignment"
field that explicitly notes whether the pattern would have caught known
issues. This makes the reconstruction honest about its retrospective nature.
BECAUSE: The value of the reconstruction is methodological (does the
standard work?) not historical (what actually happened). The "v" prefix
prevents confusion with live pattern discipline.

**D3: Scope granularity for mathematics**

IF: The draft provenance standard defines code scope levels
(repo/namespace/var/file).
HOWEVER: Mathematical proofs have different natural units (problem/step/
sub-step/expression).
THEN: Define parallel mathematical scope levels (see §3.2 table).
BECAUSE: The provenance standard should be domain-agnostic in its
*mechanism* (PUR carries scope) but domain-specific in its *vocabulary*
(what counts as a scope level). Mathematics and code have isomorphic
structures at different granularities.
