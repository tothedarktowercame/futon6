# Anatomy of a Futonic Mission (from the HEAD down)

*Preprint skeleton — DOCUMENT phase of E-mission-head. Companion to "A First
Proof Sprint" (futon6). Format: running example, not monograph — the primary
source is `futon6/holes/missions/E-mission-head.md` and its sibling artifacts;
this paper reads them rather than repeating them.*

**Status:** skeleton (2026-06-10, Joe + Fable). The §5 synthesis is the novel
content; everything else narrates artifacts that already exist.

---

## 1. Two ascents of one mountain

On 2026-06-10 a mission was developed from a single greenfield HEAD to a
verified design in one working session — with two instruments watching it the
whole way. The first instrument read its **structure**: a live panel showing
the mission's typed scopes — phases, sections, patterns, selection records —
materializing as each section was written, absent phases rendered as ghosts.
The second read its **life**: the same HEAD compiled to an 8+32-bit sigil,
birth vitals, falsifiability conditions, a survived adversarial bout, and a
deadline clause with teeth.

The Daumal-affine observation: these are two ascents of the same mountain by
different faces, and each can report the other's summit. Neither reading
reduces to the other; the paper's contribution (§5) is their synthesis.

## 2. The running example

`E-mission-head.md` — a mission whose subject is missions: "make the mission
HEAD a typed object." Its development is unusually well-instrumented because
it instrumented *itself*: every phase it grew was captured by the tooling it
was building. Primary artifacts (the paper's data):

| artifact | reading |
|---|---|
| `E-mission-head.md` (the doc, HEAD→DOCUMENT) | both — the common source |
| substrate-2 mission-scope hyperedges (31 scopes) + `*mission-overview*` | scope |
| `E-mission-head.aif.edn` (sigil, vitals, F-nodes, amendments) + `-contra` | AIF |
| `meme.db` `diagram/E-mission-head` (4+3 arrows) + `cascade/E-mission-head-argue` (3+3) | both |
| `mission_head_invariants.clj` (VERIFY V0, 5/5) | AIF (verified) / scope (a scope) |
| `closure-folds.edn` entries incl. the first `:success false` | the loop's grain |

<!-- vitals: holes/missions/E-mission-head.md -->

## 3. The scope reading (anatomy)

The mission as a nested system of typed regions anchored in text: the
eightfold spine, loose-sections, pattern/PSR/PUR records, the plain-argument
sub-scope; detected (`mission_scope_detect.py`), ingested to substrate-2,
rendered as the Scratch-Map. Third-person, structural, *what the mission is
made of*. Key device: the **ghost line** — a canon phase with no scope yet. The primary
source's opening theme, as the scope typography renders it:

<!-- excerpt: holes/missions/E-mission-head.md :: ## HEAD -->

## 4. The AIF reading (physiology)

The mission as a lifeform seeded from its HEAD: the recast to pattern form
(Golemization), the compiled sigil, health-at-birth, falsifiers authored
before constructions, the Calculemus self-argument, the wire-or-die clause.
First-person (the mission's own model of itself), dynamical, *what the
mission wants and whether it is well*. Key device: the **open arrow** — a
typed gap with its RHS specified and its construction owed. The moment the
lifeform first spoke — the sigil compilation, from the source:

<!-- excerpt: holes/missions/E-mission-head.md :: ### 3.3.1 v0.1 RESULT — the Golemization run (2026-06-10) -->

## 5. The synthesis (the new thing)

The two readings are not analogues; they are **coupled as observation and
model**:

> **The scope reading is the lifeform's observation space; the AIF reading is
> its generative model. A mission is well-formed exactly when the two cohere,
> and the coherence is computable.**

The mapping is exact, not poetic:

| scope object | AIF object | the coupling |
|---|---|---|
| HEAD section | generative seed (priors) | the compiler (bridge) |
| eightfold spine | policy stages | the canon is a *prior over trajectories* |
| **ghost line** | **prediction error** | canon prior expects a phase; observation (the doc) lacks it — ∅ is literally the residual |
| in-passing closure | a satisfied prediction by another path | posterior update without the expected observation |
| open arrow | expected free energy not yet discharged | A2 prices its persistence |
| PSR / PUR | policy selection / outcome observation | the pur binder makes the loop legible |
| falsifier (F-node) | precision channel | what would *count* as error |
| vitals (health.json) | interoception | computed FROM the scope-readable doc |
| the panel | exteroception | the body as seen |

Three consequences worth the paper:

1. **The body-schema claim.** The H1–H3 wiring (health computed from the doc
   → displayed in the panel → seeded into the head) is not plumbing; it is
   the *binding* of interoception to exteroception — the point at which the
   mission acquires a body schema. Before it, two instruments; after it, one
   organism with two senses.
2. **Coherence is checkable, so DOCUMENT can be earned.** Disagreements
   between the readings are findings, mechanically surfaced: a scope the
   model doesn't want (decorative structure), a want with no scope
   (unobservable desire — exactly the unminted-pattern catch), a ghost the
   model says is satisfied (the in-passing DOCUMENT closure). The session
   produced instances of all three.
3. **The synthesis discharges the cascade's own largest gap.** The mission's
   ARGUE cascade carries the unminted candidate
   `two-projections-of-one-quantity`. The synthesis *is* that pattern's
   construction: scope-reading and AIF-reading as two projections of one
   underlying typed-arrow object (substrate-2 hyperedges + meme.db arrows +
   the package are three stores of *one* structure). Publishing the paper
   mints the pattern; the preprint is its flexiarg's BECAUSE.

And the Daumal closure, which the stack supplies on its own: the points where
two independent ascents *agree* are the only points one may call real — and
the stack already has a name and a discipline for externally-checkable value
found on the mountain: **the peradam**. The synthesis predicts where peradams
form: at reading-agreements. "One descends, one sees no longer, but one has
seen."

And the argument in the register anyone can read — the mission's own
plain-language statement, boxed as it appears in the source:

<!-- excerpt: holes/missions/E-mission-head.md :: ### 4.1 Plain-language argument (the version anyone can read) -->

## 6. Method appendix (planned)

The session as the Cyborg learning loop run at operator scale: fold records
(incl. the first `:success false`), the closure standard, the bell-handoff
INSTANTIATE, the review gates. Cross-refs: E-scope-audit (the instrument's
own audit trail), C-falsifiable-missions §7 (the closure schema), E-ground-G
(what the satisfaction conditions discharge against).

## 7. Relation to "A First Proof Sprint"

The Proof Sprint documents a *proof* finding its way through the stack; this
paper documents a *mission* doing the same — one level up: the unit of work
examining the unit-of-work concept. Same house, different floor.

---

*TODO (next sessions): §3/§4 prose passes over the primary source; figures
(the panel at frames 1→6; the bout's cross-edge diagram; the vitals card);
§5 worked dis/agreement instances with store queries; decide venue + whether
the two-projections flexiarg ships with the paper or before it.*
