# Mission: Artificial Stack Exchange

**Date:** 2026-02-25
**Status:** IDENTIFY (mission proposal)
**Origin:** Joe Corneli, SFI Complexity Postdoctoral Fellowship proposal (~2019).
Reformulated as a computational mission for the futon stack, 2026-02-25.
**Owner:** futon6, with dependencies on futon3c (social loop), futon3b (gates),
futon3a (pattern search), futon4 (hypergraph), futon5 (AIF wiring), futon7
(world interface)
**Cross-ref:** M-self-representing-stack (futon4), M-futon3-last-mile (futon3c)

## Motivation

The original proposal aimed to realise Turing's vision of artificial agents
that are "able to converse with each other to sharpen their wits." The vehicle
was an agent-based model of Stack Overflow: agents that ask and answer
questions about a technical corpus, improve a shared knowledge graph, and
compete/cooperate within institutional rules.

Five years later, the futon stack has built much of the infrastructure the
proposal described — but the pieces haven't been assembled into the system
the proposal envisioned. LLMs have made the NLP components tractable (the
proposal's M4-M6 milestones are now off-the-shelf), which shifts the
challenge from knowledge extraction to agent architecture, institutional
design, and self-improvement.

**The computational mission:** Wire the existing futon infrastructure into a
running Artificial Stack Exchange where agents ask questions, answer them,
evaluate the results, improve a shared knowledge graph, and — crucially —
write new agents and new institutions. The system that results should be
the futon stack's primary engine of self-improvement.

## Theoretical Anchoring

### What the Proposal Got Right

The proposal identified six structural layers, all of which now have futon
infrastructure:

1. **Content extraction** (reading technical texts into knowledge
   representations). Now: futon6 P0 (25 math-informal patterns, 496/535
   entries tagged), P7 (114K physics QA imported, NER kernel built).

2. **Active inference** (agents actively seek to maximise evidence for
   their sensory model). Now: futon2 (reference AIF implementation,
   golden microtraces settled), futon5 (three exotypes with wiring
   diagrams).

3. **Tangled Program Graphs** (teams of programs networked together,
   fitness scored jointly). Now: hyperedges as frozen dynamics
   (M-self-representing-stack DERIVE, 2026-02-24). Hyperedges carry
   topology, not just membership — they ARE the tangled program graph,
   with Clojure-inside-EDN providing the evaluable structure.

4. **IAD institutions** (rules, norms, and strategies constraining agent
   interactions). Now: futon3b gate pipeline (G5→G0, typed evidence,
   structured rejections), futon3c social pipeline (presence → auth →
   mode → dispatch → persist). The gates ARE institutions — they define
   what's OBLIGED, FORBIDDEN, and PERMITTED.

5. **Baldwin effect** (population-level learning from individual
   selection). Now: futon3b L1 canonicalizer (tension observer + library
   evolution). Individual agent work → evidence → pattern library update
   → future agents inherit improved patterns.

6. **Self-play** (agents asking and answering to improve a shared graph).
   Now: f6/self-play-loop pattern (Asker/Answerer/Critic, graph updates
   gated by critic scores). Defined as a pattern but not yet instantiated
   computationally.

### What the Proposal Deferred

Two milestones were explicitly placed late in the timeline and have no
futon infrastructure yet:

- **M16: Agents writing agents.** The system can run agents but cannot
  produce new ones. An agent that identifies a gap in its own capabilities
  and writes a specialist sub-agent to fill it would be a qualitative
  leap. The peripheral model (futon3c) provides the envelope; the
  question is whether an agent can design a new peripheral spec and
  register a new agent to inhabit it.

- **M17: Agents writing institutions.** The gates (futon3b) are
  programmable, but no agent has ever created a new gate rule. An agent
  that observes a failure pattern (e.g., low-quality code passing G3)
  and proposes a new gate constraint would close the institutional
  self-improvement loop. IAD says institutions are rules about rules —
  M17 is the meta-level.

### The Noöme

The proposal described Stack Exchange as a portion of "the human noöme,
a noetic heritage that is as necessary for our collective survival as the
microbiome is to individual persons." The self-representing stack
(M-self-representing-stack) is the futon stack's noöme — its own
knowledge heritage made navigable and queryable. The Artificial Stack
Exchange is the process by which that heritage grows.

### Q ⊢ A and the Forum

The proposal noted that Q&A has the structure of Kolmogorov's calculus of
problems: Q ⊢ A, where obtaining a useful answer reduces to asking a
suitable question. futon3c's Forum (proof trees with posts as proof steps)
already has this structure — a thread is a derivation, a post is a step,
and the thread's fruit is the answer. The Artificial Stack Exchange is
the Forum made autonomous: agents post questions, other agents post
answers, and the proof tree grows.

### Koans and Stepping Stones

The proposal cited W. Brian Arthur on starting from "the simplest possible
tasks and growing in complexity." futon6's koans concept (M9 in the original
timeline) provides this — simplified exercises that scaffold agent learning
before tackling real questions. The O-0 classical baseline (PLAN.md) is
itself a koan: reproduce Corneli (2014) Chapter 6 before attempting
anything more ambitious.

## What Exists (MAP preview)

| Proposal Component | Futon Infrastructure | Status |
|---|---|---|
| Q&A corpus | futon6 P7: 114K physics.SE QA pairs | Imported, NER-tagged |
| Knowledge graph | futon4 Arxana + futon1a XTDB | Operational, hyperedge write path plumbed |
| Pattern library | futon3/library + futon3a search | 50+ patterns, search works but not wired to agents |
| Agent framework | futon3c agency registry + peripherals | Running (Claude, Codex, Tickle) |
| Evidence store | futon1a XTDB backend | Operational, queryable via API |
| Gate pipeline | futon3b G5→G0 | Tested (30 tests, 107 assertions) |
| Baldwin cycle | futon3b L1 canonicalizer | Tested, not wired to social layer |
| AIF loop | futon2 (ants), futon5 (wiring diagrams) | Reference impl settled |
| Self-play loop | f6/self-play-loop pattern | Defined, not instantiated |
| Sandbox | Drawbridge /eval, codex CLI | Available |
| GitHub API | futon7 (started 2026-02-24) | Early stage |
| Forum | futon3c forum module | Tested, bridge scripts exist |
| Classical baseline | O-0 (PLAN.md) | In progress |

## Scope In

### Phase 1: Self-Play Instantiation

Wire the self-play-loop pattern into a running system:

- **Asker agent** queries the knowledge graph (Arxana/futon1a) for gaps:
  isolated nodes, missing cross-links, ungrounded annotations. Generates
  questions targeting those gaps.
- **Answerer agent** uses graph context + pattern library (futon3a) +
  corpus (futon6 P7 data) to construct answers.
- **Critic agent** evaluates Q&A pairs using the gate pipeline (futon3b)
  as quality control. Scores gate through G5→G0.
- Graph updates committed only when critic score exceeds threshold.
- Evidence entries emitted for every Q&A cycle (via futon3c evidence
  landscape).
- 100-iteration pilot: measure graph growth (new relations, confidence
  scores, gap markers).

### Phase 2: Institutional Modelling

Model Stack Exchange institutions as gate rules (futon3b):

- Map SE reputation mechanics to gate constraints (who can ask, who can
  answer, what qualifies as "vetted").
- Map SE voting to evidence weighting in the proof-path store.
- Map SE tags to pattern library categories (futon3a).
- Implement at least one IAD-style rule: "ATTRIBUTES of participants who
  are OBLIGED/FORBIDDEN/PERMITTED to ACT under specified CONDITIONS."
- Test: does the institution improve Q&A quality vs. unconstrained self-play?

### Phase 3: Agents Writing Agents (M16)

The qualitative leap:

- An agent running in the self-play loop identifies a capability gap
  (e.g., "I can't answer questions about topology because I have no
  topology-specific patterns").
- The agent proposes a new specialist agent: defines a peripheral spec
  (tool constraints, capability envelope), registers it with the agency
  registry, and delegates topology questions to it.
- The new agent inherits the knowledge graph and pattern library but has
  a narrower scope.
- Evidence: does the specialist outperform the generalist on its domain?

### Phase 4: Agents Writing Institutions (M17)

The meta-level:

- An agent observes failure patterns in the self-play loop (e.g.,
  "questions about X consistently get low critic scores").
- The agent proposes a new gate rule: "questions about X require at least
  two evidence citations from the corpus."
- The rule is added to the gate pipeline (futon3b) and tested.
- Evidence: does the new rule improve Q&A quality on topic X?

### Phase 5: GitHub Integration (via futon7)

Connect the Artificial Stack Exchange to the real world:

- Agents ask and answer questions about ongoing development in futon
  repos (not just static corpus).
- Questions triggered by GitHub events (new PRs, failing CI, open issues).
- Answers grounded in codebase via reflection API (futon3c).
- The Stack Exchange becomes the stack's own support forum.

## Scope Out

- Replacing Stack Overflow (this is a model, not a product)
- GPU-intensive NLP (use LLMs via existing agent invoke paths)
- Multi-user human participation (future; start agent-only)
- Formal verification of Q&A (use gate pipeline as quality proxy)

## Relationship to the Original Proposal

| Original Milestone | Futon Mission/Component | Status |
|---|---|---|
| M1 Gather data | futon6 P7 | Done (114K QA pairs) |
| M2 Argumentation analysis | futon6 P0 (25 patterns) | Done |
| M3 Process model analysis | futon3/library patterns | Done (50+ patterns) |
| M4 ML/NLP bootcamp | Superseded by LLMs | N/A |
| M5 Match Q↔A | futon3a similarity search | Exists, needs wiring |
| M6 Hierarchical ML | Superseded by LLMs | N/A |
| M7 Active Inference bootcamp | futon2 settled | Done |
| M8 Agent modelling + sandbox | futon3c agency + Drawbridge | Done |
| M9 Curate koans | O-0 classical baseline | In progress |
| M10 Crowdsourced exercises | — | Not started |
| M11 Agent-written questions | Phase 1 (Asker agent) | This mission |
| M12 Publication: IJCAI | — | Deferred |
| M13 IAD modelling | Phase 2 | This mission |
| M14 SFI collaborators | — | Ongoing (Rob on futon6) |
| M15 Contributor infrastructure | futon3c + futon7 | Partial |
| M16 Agents writing agents | Phase 3 | This mission |
| M17 Agents writing institutions | Phase 4 | This mission |
| M18 Contest | — | Future (futon7 event?) |
| M19 Publication: AI | — | Deferred |
| M20 GitHub API | Phase 5 (futon7) | This mission |
| M21 SFI collaborators | — | Ongoing |
| M22 Tutoring study | — | Future |
| M23 Publication: Science | — | Deferred |

## Completion Criteria

1. Self-play loop runs for 100 iterations, producing measurable graph growth
2. At least one IAD-style institution improves Q&A quality vs. baseline
3. At least one agent-written agent outperforms a generalist on its domain
4. At least one agent-written gate rule improves quality on its topic
5. At least one question is triggered by a real GitHub event and answered
   with codebase-grounded evidence

## Relationship to Adjacent Missions

| Mission | Relationship |
|---|---|
| M-self-representing-stack (futon4) | The knowledge graph this mission populates is the one that mission makes navigable |
| M-futon3-last-mile (futon3c) | The agent flows this mission uses are the ones that mission wires |
| O-0 classical baseline (futon6) | The koan that grounds Phase 1 |
| M-improve-irc (futon3c) | Transport reliability for multi-agent self-play |
| futon7 (various) | World interface for Phase 5 |

## Source Material

| Source | What We Take |
|---|---|
| SFI Complexity Postdoc proposal (Corneli, ~2019) | Vision, milestone structure, theoretical anchoring |
| Corneli (2014) Chapter 6 | O-U learning event model, vocabulary trajectories |
| f6/self-play-loop pattern | Asker/Answerer/Critic architecture |
| f6/negative-space-duality pattern | Formal/informal dual layer analysis |
| math-informal/parametric-tension-dissolution | Structural tension resolution |
| gauntlet/world-is-hypergraph | The stack IS the game world |
| futon-theory/reverse-morphogenesis | ← operator: infer what comes next from current form |
| Arthur (2004) SFI working paper | Stepping stones, simplest-first |
| Arthur (2017) "autonomous economy" essay | Self-organizing, conversational, dynamic |
| Ostrom, IAD framework | Rules, norms, strategies for agent institutions |
| Turing (1948) "sharpen their wits" | The original vision |
