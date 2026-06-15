# Anatomy of a Futonic Mission (from the HEAD down)

*Preprint skeleton — DOCUMENT phase of E-mission-head. Companion to "A First
Proof Sprint" (futon6). Format: running example, not monograph — the primary
source is `futon6/holes/missions/E-mission-head.md` and its sibling artifacts;
this paper reads them rather than repeating them.*

**Status:** draft v2 (2026-06-10, Joe + Fable) — level-0 prose now holds to the
plain-language standard (the ARGUE §4.1 discipline applied to the whole paper:
boxes may speak the system's native language; the prose around them may not).

---

## 1. Two ascents of one mountain

In the software system we work in, the unit of work is called a *mission*: a
single document that begins with a short statement of intent (its *HEAD*) and
grows, phase by phase, into a finished piece of work. On 10 June 2026 one
mission went from a bare statement of intent to a verified design in a single
working session — and two different instruments watched it happen.

The first instrument read the mission's **structure**. As each section was
written, a live outline showed the document's parts — its phases, sections,
and cited methods — appearing as nested, colored blocks, with the phases that
had *not* been written yet shown as pale "ghost" lines. The second instrument
read the mission's **life**. The same statement of intent was compiled into a
small numerical fingerprint, from which the system computed a health report
for the newborn work: how confident, what was missing, what would count as
failure — together with stated conditions under which the work would declare
itself dead.

In René Daumal's *Mount Analogue*, a mountain is climbed by parties on
different faces, and what makes the summit real is that the ascents agree.
The two readings here are two such ascents: neither reduces to the other, and
their agreement — made precise in §5 — is the point.

## 2. The running example

The running example is a mission whose subject is missions themselves: *make
the opening statement of intent a typed, checkable object*. Its development
is unusually well documented because it documented itself — every phase it
grew through was captured by the very tooling it was building. The primary
sources (everything below is on disk and quoted from, never paraphrased):

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

Read structurally, a mission is a nested system of typed regions of text —
*scopes*. The standard life-cycle phases form its spine; sections nest inside
phases; records of which methods were chosen, and how they fared, nest inside
sections. A detector finds these regions automatically and a panel draws them
as colored blocks inside blocks (the same typography used in this PDF). This
reading is third-person and anatomical: it says what the mission is *made
of*. Its most useful device is the **ghost line**: a life-cycle phase that
the standard form expects but the document does not yet contain, drawn in
pale outline. Here is the running example's opening statement, exactly as the
structural typography renders it:

<!-- excerpt: holes/missions/E-mission-head.md :: ## HEAD -->

## 4. The organism reading (physiology)

Read the other way, the mission is a small organism. (In-house this is
called the *AIF reading*, after **active inference** — a framework from
computational neuroscience in which an organism maintains a model of itself
and acts to keep that model's predictions true. Nothing below requires more
of the theory than that sentence.) Its statement of intent
is rewritten in a four-part design-pattern form (*if / however / then /
because*) and compiled into a numerical fingerprint — the system borrows the
word *sigil*, after an earlier experiment in which simulated ants were
animated by installing exactly such fingerprints. (To actually compile one for
a mission of your own, see `futon5/README-sigils.md` — the compilation is a
script, `head_exotype_probe.py`, not a hand-authored value.) From the sigil the system
computes birth vitals; the design states in advance what observations would
prove it wrong; the work even argues for its own existence against the
strongest objections we could write down, and carries a clause that declares
it dead, publicly, if its parts are not put to use by a deadline. This
reading is first-person and physiological: it says what the mission *wants*
and whether it is *well*. Its most useful device is the **open arrow**: a
named gap whose target is specified but whose construction is still owed.
The moment the organism first spoke — the fingerprint compilation, from the
source:

<!-- excerpt: holes/missions/E-mission-head.md :: ### 3.3.1 v0.1 RESULT — the Golemization run (2026-06-10) -->

## 5. The synthesis (the new thing)

The two readings are not merely parallel descriptions; they are **coupled,
the way observation and model are coupled in a living thing**:

> **The structural reading is the organism's observation space; the organism
> reading is its model of itself. A mission is well-formed exactly when the two cohere,
> and the coherence is computable.**

The mapping is exact, not poetic:

| structural object | organism object | the coupling |
|---|---|---|
| statement of intent (HEAD) | the model's starting beliefs | the fingerprint compiler |
| the standard phases | stages of a plan | the standard form is an *expectation about how work unfolds* |
| **ghost line** | **prediction error** | the standard form expects a phase; the document lacks it — the pale line *is* the mismatch |
| a phase satisfied in passing | a prediction met by an unexpected route | belief updated without the expected observation |
| open arrow | an unmet commitment the model still expects | the deadline clause prices its persistence |
| method chosen / how it fared | action selected / outcome observed | the records make the learning loop legible |
| stated failure condition | what would *count* as error | written before the construction, never after |
| the vitals card | interoception (the body sensing itself) | computed *from* the structurally-readable document |
| the outline panel | exteroception (the body as seen) | the same document, drawn |

Three consequences worth the paper:

1. **The body-schema claim.** Late in the session, three pieces of wiring
   were added: the health report is computed from the document, displayed in
   the outline panel, and fed into the mission's model of itself. That is not
   plumbing; it is the binding of inner sense to outer sense — the point at
   which the mission acquires a body schema. Before it, two instruments;
   after it, one organism with two senses.
2. **Coherence is checkable.** Disagreements between the readings are not
   embarrassments but findings, surfaced mechanically: structure the model
   has no use for (decoration); a want with no corresponding structure (a
   desire nothing can observe — the session caught a cited method that had
   never actually been written down this way); a ghost the model says is
   already satisfied (one phase was closed in passing, inside another
   section, and the detector learned to see it). The session produced live
   instances of all three.
3. **The synthesis pays one of its own debts.** When the mission argued for
   its existence, its argument leaned on a design principle that turned out
   never to have been written into the method library: *two projections of
   one quantity*. The synthesis above is precisely that principle's missing
   construction — the structural reading and the organism reading are two
   projections of one underlying object, stored three ways. Publishing the
   argument writes the missing principle down.

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

How the session was actually run: a human operator and several AI agents
sharing one learning loop — every method-use recorded with its outcome
(including the record-keeping's first honestly-logged *failure*), build work
handed to worker agents with acceptance criteria and reviewed on return, and
each claim checked against the system's stores before being written here.
Technical cross-references for in-house readers are kept in the source
repositories rather than in this text.

## 7. A second ascent: a mission read mid-life

The running example above was born instrumented — it documented itself from
its first sentence. The harder test of the two readings is a mission that
lived first and was analyzed later. On 11 June 2026 the structural
instruments were pointed at `M-futonzero-generative` — a mission chartered
two days earlier, advanced through twenty live War-Machine flights, its
work products built by three agents — and the detector recovered
**33 typed scopes** from a document nobody wrote for the instruments.

What the structural reading found, mid-life: the mission's *behavioral
constraints* came back as typed objects — each "scope out" bullet ("No
reward-trained policy update", "No hidden scheduler") is an anchored scope
with outward polarity, machine-visible at its exact line; its two hard
gates and its checkpoint came back as sections; and its life-cycle came
back honestly thin — HEAD and IDENTIFY present, every later phase a ghost
line, which is *true*: this mission advances by external flights, not by
writing its own phases. The organism reading agrees from the other face:
the twenty flights were this mission's metabolism, and the document only
records what the metabolism deposited.

And the instrument reported its own boundary, which is the finding worth
the section: the mission's **Remaining-Work checklist — the seven counted
holes that the flights actually closed and the War Machine actually
counted — is invisible to the scope reading.** The checkboxes detect as
nothing at all; meanwhile a separate counter (the mission registry's)
reads the same lines by regex, and the flights' acts changed them without
either instrument seeing one object. Three systems touch the same holes;
none share them. The mid-life analysis thus ends the way the first ascent
did — with the synthesis paying a debt by *naming* it: the counted hole
wants to become a first-class scope, so that a hole's count, a flight's
act, and a panel's display are one thing read three ways. That is the
next construction, and this paragraph is its preregistration.

## 8. Relation to "A First Proof Sprint"

"A First Proof Sprint" follows a single mathematical proof finding its way
through this same system. The present account sits one level up: it follows
the *unit of work itself* — the mission — through the same passage, the
container examining the concept of containment. Same house, different floor.

## 9. Reprise and technical explainer (supporting material)

*This section reprises the account above in the system's native,
category-theoretic vocabulary — the language the level-0 prose deliberately kept
in boxes. It is written to travel as **supporting material**: a publication can
carry §§1–8 as the paper and this as a technical supplement. Every construction
named below is on disk; where a leg is schematic or still owed, it says so. The
datatype used here is formalized by the companion mission `M-typed-holes`
(futon3c; Lean in `mathlib4/DarkTower/`), of which the running mission
E-mission-head is itself a worked example (`M-typed-holes-example-mission-head.md`;
Lean `DarkTower/Examples.lean::MissionExample`).*

**R1 — the object (reprise of §§1, 3, 4): a mission is a BV-typed wiring
diagram with typed holes.** The structural reading of §3 is, precisely, a
**hyperedge with typed holes** — a position in the polynomial functor category
**Poly**, carrying a set of *typed directions* (the holes). Concretely:

- the life-cycle phases (§3's "spine") are a **non-commutative `seq` comb**,
  `⟨HEAD; IDENTIFY; …; DOCUMENT⟩` (`DarkTower/BV.lean`, `Examples.lean::lifecycle`);
- the two ascents of §1 — scope and organism — are a **`copar`** (parallel
  composition held together), `copar(scope, organism)`
  (`Examples.lean::readings`); a mission is well-formed exactly when the two
  branches cohere, which is the §1 "summit is where the ascents agree" stated
  categorically;
- §3's **ghost line** and §4's **open arrow** are both **typed holes**: the
  ghost line is a hole typed by an *expected phase* (`hungry_for = expected-phase`),
  the open arrow a hole typed by a *named-but-unbuilt target*
  (`DarkTower/TypedHole.lean`; `M-typed-holes-example-mission-head.md` §2).

**R2 — the act (reprise of §4): "fill" is the discharge counit; prediction
error is an unfilled hole.** §4's organism *acts* to keep its self-model's
predictions true. In the native vocabulary that act is **`fill`**: supplying a
typed hole with a well-typed filler. A **prediction error is exactly an
unfilled typed hole**, and the AIF action that resolves it is one application of
`fill` — formally the **counit** of the discharge comonad (`DarkTower/Fill.lean`,
`DarkTower/Discharge.lean`). So §4's "the body acts to reduce surprise" and the
stack's "discharge a hole" are one operation under two descriptions.

**R3 — the synthesis is a construction, and it is now built (reprise of §5).**
§5.3 records a debt: the synthesis leans on the principle *two projections of
one quantity*, which had never been written into the method library, and §5
claims the paper itself is "that principle's missing construction." This
supplement reports the construction **discharged**. The one underlying object is
the **typed-hole/`fill` datatype**; the structural and organism readings are two
of its **projections**; and `M-typed-holes` built the single runtime
operator — `fill(hole, filler, kind=…)` — that **six** such projections route
through (answer/query, reply/bell, ground/symbol, compose/comb, discharge/proof,
cascade-feed/mining), each a view, none a reimplementation
(`futon3c/scripts/fill.py`; the coverage proof `DarkTower/Coverage.lean`:
`Fintype.card Projection = 6`, no orphan by construction). The §5 coupling table
is then literally a list of **typed coherence-wires across the `copar`** — the
BV interaction that makes the two readings one object rather than two
descriptions. (Each row of §5's table is one such wire; see
`M-typed-holes-example-mission-head.md` §2(c).)

The construction was not merely defined but **witnessed at runtime**: routing the
six projections through the one `fill` produced typed records in the witness
store (ArSE) — *proof = witnessed fill*, the stack's I5 — so the claim "the two
readings are two projections of one operator" is checkable on the running system,
not only in Lean.

**R4 — the mid-life preregistration came true (reprise of §7).** §7 closes by
*preregistering* a construction: the counted Remaining-Work hole "wants to become
a first-class scope, so that a hole's count, a flight's act, and a panel's
display are one thing read three ways," because "three systems touch the same
holes; none share them." That is the **same disunity** R3 resolves: three (here,
six) surfaces of one hole with no shared operator. The single `fill` is the
shared operator the preregistration asked for; the predicted next construction
has since landed. (Caveat preserved from §7: making the *displayed/counted* hole
literally the same object as the scope is the wiring D1 demonstrates in principle;
the full last-mile UI/registry unification is its own follow-on.)

**R5 — diagrams and their provenance.** Two diagram families illustrate this
supplement. (i) The **lifecycle `seq` + readings `copar`** string diagram is
*native* to this paper's running example (E-mission-head = `MissionExample`), so
it carries no caveat. (ii) The **Poly position-with-directions**, the **lens
(`comb`) composition**, and the **answer = `fill`/discharge counit** diagrams are
drawn from a *different* worked example — the grounding of arXiv 0809.2517
(`DarkTower/Examples.lean::PaperExample`) and the cascade→sorry→wiring fold
(`DarkTower/FirstFlightsExample.lean`) — and **must be labelled as such**: they
show the same machinery on a mathematics-paper instance, not on this mission.

**R6 — what is proved vs. demonstrated vs. owed (honesty ledger).** The
datatype, the six-projection coverage, and the discharge/fill laws are **Lean-
proved** (`DarkTower/`, `lake build` green, 0 `sorryAx`). The single runtime
`fill` and all six adapters routing through it are **demonstrated live** (witnessed
in ArSE this session). Still **schematic or owed**: the `cascade-feed` leg is
covered as a static satiety grade, not yet a hungry→sated *transition*; the BV
`medial`/`switch` rules are schematic (full Guglielmi-BV fidelity deferred); a
formal `copar`-interaction functor is future work; the substrate-2a data is a
first cut (filler *quality* tracks its QA baseline, not the wiring). Nothing here
is claimed beyond these lines.

---

*TODO (next sessions): §3/§4 prose passes over the primary source; figures
(the panel at frames 1→6; the bout's cross-edge diagram; the vitals card);
§5 worked dis/agreement instances with store queries; decide venue + whether
the two-projections flexiarg ships with the paper or before it.*
