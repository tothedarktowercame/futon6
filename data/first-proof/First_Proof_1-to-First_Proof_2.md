# First Proof 1 → First Proof 2: Changelog and Readiness

What changed between the end of Sprint 1 (Feb 14, 2026) and now (Feb 15),
and how it positions us for a better First Proof 2 in approximately one
month.

## Sprint 1 recap (from sprint-review.tex)

- 55 hours, 341 commits, 3 agents (Joe / Claude / Codex)
- 4/10 correct, 4/10 correct direction, 2/10 wrong
- Both wrong answers on NO-answer problems (YES bias + domain-depth gap)
- Infrastructure: bare wiring diagrams (typed nodes + edges), Codex
  verification prompts generated per node, numerical stress testing
- Weaknesses identified:
  1. No systematic falsification protocol
  2. No CT-backed verification of proof structure
  3. Wiring diagrams had IATC edge types but no discourse, port, or
     categorical annotation — so structural checks were manual
  4. Literature mining was intensive but keyword-biased (P4: missed
     hyperbolic Hessian family; P7: only searched YES direction)
  5. No formal PSR/PUR discipline (proposed but not implemented)

## What we built since (Feb 14–15)

### 1. CT reference dictionary (nlab-wiring.py)

Extracted from 20,441 nLab pages. Eight categorical pattern types
(adjunction, equivalence, fibration, kan-extension, limit, monad,
natural-transformation, universal-property) with:
- Required and typical links per pattern
- Discourse signatures (which component types co-occur)
- Diagram counts
- Link weights (definition-ref vs. prose-ref frequencies)

**Sprint 1 gap addressed:** Literature mining was keyword-biased. The CT
reference provides a structured index of mathematical relationships — not
just keywords but typed connections between concepts, weighted by how they
co-occur in 20K pages of reference mathematics.

### 2. Hierarchical wiring assembly (assemble-wiring.py)

Three-level nested wiring diagrams for SE/MO threads:
- **Thread level:** IATC performative edges (assert, challenge, clarify, ...)
- **Categorical level:** per-node CT pattern detection with IDF weighting
- **Diagrammatic level:** commutative diagram extraction

Per-node enrichment:
- Discourse annotations (scopes, wires, ports, labels)
- Input/output port extraction from mathematical text
- Port matching across edges (term overlap + CT reference boost)
- NER term spotting from 19K-term kernel

Validated on 200 SE/MO threads with sharp 2x2 differentiation:
MO-CT 3.2 cat/thread (72%), SE-CT 2.7 (80%), MO-MP 0.4 (28%),
SE-MP 0.1 (10%).

**Sprint 1 gap addressed:** Wiring diagrams were flat (node type + edge
type only). Now they carry discourse structure, port connectivity, and
categorical context — the verifier has something to check.

### 3. CT-backed verifier (ct-verifier.py)

Four verification checks per edge:
1. **Categorical consistency** — do both endpoints' CT patterns co-occur?
2. **Port compatibility** — are source output ports type-compatible with
   target input ports?
3. **IATC-discourse alignment** — does the edge type (assert, challenge,
   ...) match the source node's discourse markers?
4. **Reference completeness** — how many of the claimed CT pattern's
   required/typical links appear in the node's text?

Live mode watches a directory for new wirings and verifies as they arrive.

**Sprint 1 gap addressed:** Structural checks were manual (reviewer reads
wiring diagram, spots inconsistencies by eye). Now automated: the verifier
scores every edge and reports exactly why it fails.

### 4. Proof wiring enrichment (enrich-proof-wiring.py)

Bridges the gap between bare proof wirings (like problem4-wiring.json)
and the verifier's expectations:
- Adds discourse annotations via proof-specific markers ([ERROR] →
  adversative, [PROVED]/QED/COMPLETE → consequential, WLOG → such-that)
- Extracts equation ports from body text (symbol = expression → bind/let,
  expression >= 0 → constrain/such-that)
- Runs categorical detection against CT reference
- Computes port matches across edges

Result on Problem 4: verification score 0.006 (bare) → 0.238 (enriched).
24/41 edges port-compatible, 15/41 IATC-aligned.

**Sprint 1 gap addressed:** The proof wirings we built during Sprint 1
were structurally rich (good IATC typing, evidence strings, hyperedges)
but lacked the annotation layers needed for automated verification. Now
we can enrich any proof wiring and immediately get a verification report.

### 5. Superpod Stage 7 upgrade (superpod-job.py)

Stage 7 now produces CT-backed wiring when the reference exists:
- Loads CT reference + NER kernel automatically
- Converts SEThread dataclass → normalized dict → build_thread_graph()
- Streaming JSON write (memory-safe for 567K threads)
- Falls back to legacy thread_performatives when no CT reference

**Sprint 1 gap addressed:** The superpod pipeline was set up for
large-scale processing but produced only flat performative wiring. Now it
produces the same rich 3-level wiring at scale. When we run on 567K
math.SE + 100K MathOverflow threads, every thread gets CT-backed wiring
from the start.

## How this helps First Proof 2

### Systematic falsification (Sprint 1 lesson #1)

The verifier enables automated falsification. For each proof step:
- **IATC alignment check** catches structural tells: an `assert` edge with
  no consequential discourse in the source is a flag. A `reference` edge
  with no citation-like marker is a flag. These are exactly the kind of
  "logically sound but mathematically wrong" arguments that tripped P1.
- **Port compatibility check** catches premise flow errors: if a step
  claims to use a result from an earlier step but the port types don't
  match, that's a structural gap.
- **Completeness check** catches thin claims: a node claiming
  `cat/adjunction` but missing most required links is under-substantiated.

In Sprint 1, falsification was ad-hoc ("What if the answer is NO?"). In
Sprint 2, we can run `ct-verifier.py verify` after every proof draft and
get a scored report of structural weak points.

### Richer proof wirings from the start (Sprint 1 lesson #3)

Sprint 1 wirings were built post-hoc. For Sprint 2:
1. Build the wiring diagram as the proof develops (not retroactively)
2. Run enrichment after each major step
3. Run verification continuously (live mode)
4. Use verification failures to guide the next cycle

This turns the verifier into a real-time proof assistant, not a post-mortem
tool.

### Better literature mining (Sprint 1 lesson #4)

The CT reference + NER kernel enable structured literature queries:
- When a node's categorical detection fires (e.g., `cat/adjunction`), the
  reference provides required_links and typical_links — these are
  domain-specific search terms, not generic keywords
- When a port match score is low between two connected nodes, the missing
  terms identify what concept is needed to bridge them
- The IDF weighting distinguishes ubiquitous terms (category, functor)
  from discriminative ones (adjoint, monad) — searches focus on the latter

For P4's near-miss: if the node claiming a Cauchy-Schwarz endpoint had
been enriched with port analysis, the "missing input port" would point
toward the Jacobian contraction step — the exact bridge we couldn't find.

### Coaching interventions become data-driven (Sprint 1 lesson: P6)

The P6 coaching intervention worked by forcing layer enumeration. With
CT-backed wiring, layer enumeration is partially automated:
- Discourse annotations identify which layers are active (consequential
  wires = deductive layer, adversative = falsification layer, clarifying =
  definitional layer)
- Categorical detections identify which mathematical frameworks are in play
- Port connectivity reveals which steps are well-connected vs. isolated

When the verifier shows a cluster of failing IATC checks in one part of
the proof, that's a signal to reframe — the same signal that the P6
coaching intervention provided manually.

### YES-bias detection (Sprint 1 lesson: P1, P7)

The two wrong answers shared a pattern: constructive argument from false
premise (P1) or unfounded conditional discharge (P7). The verifier
can flag these:
- A node with `wire/consequential` discourse ("therefore", "QED") but
  low port-compatibility scores on its incoming edges has weak foundations
- A node claiming a categorical pattern but with 0.0 completeness score
  is asserting structure it hasn't justified

Neither check is a proof of incorrectness. But both are structural tells
that warrant explicit NO-direction testing — exactly the protocol the
sprint review recommends for Sprint 2.

## Infrastructure ready for Sprint 2

| Component | Status | What it does for FP2 |
|-----------|--------|---------------------|
| `nlab-ct-reference.json` | Ready | 8 CT patterns, 20K pages, structured index |
| `assemble-wiring.py` | Ready | 3-level enrichment pipeline |
| `ct-verifier.py` | Ready | 4-check automated verification |
| `enrich-proof-wiring.py` | Ready | Bridge bare proof → enriched proof |
| Superpod Stage 7 | Ready | CT-backed wiring at scale (567K threads) |
| NER kernel (19K terms) | Ready | Term spotting for ports + completeness |
| 200-thread validation set | Ready | 2x2 differentiation confirmed |
| PSR/PUR skills | Ready | `/psr` and `/pur` in Claude Code |

## What's still needed before Sprint 2

1. **Run superpod on math.SE + MathOverflow** — produces the large-scale
   CT-backed wiring corpus that enables richer literature mining during
   the sprint itself
2. **Build a "proof assistant" mode** — wraps enrich → verify → report
   into a single command that runs after each proof step, integrated with
   the sprint workflow
3. **Falsification protocol** — formalize the "test the NO direction"
   step as a schedulable protocol primitive, not an ad-hoc intervention.
   Trigger: when verification score plateaus, force a NO-direction cycle
4. **Domain-depth index** — for each First Proof 2 problem, pre-identify
   the specialized subfields and pre-load the relevant CT patterns and
   NER terms. The goal: never be keyword-blind in the way that missed
   hyperbolic Hessian contraction for P4
5. **Calibration tracking** — maintain running confidence estimates per
   problem, updated after each verification cycle

## Timeline

- **Now → Week 1:** Run superpod on full math.SE + MO (CPU-only, ~48h).
  Build proof-assistant mode wrapper.
- **Week 2:** Pre-load domain knowledge for announced FP2 problems (if
  topics are revealed early). Otherwise, build the domain-depth index
  from the CT reference + arXiv abstracts.
- **Week 3:** Dry-run the full Sprint 2 protocol on a practice problem
  (e.g., re-do P4 with CT-backed infrastructure, target: find the
  all-n bridge that Sprint 1 missed).
- **Week 4:** Sprint 2.
