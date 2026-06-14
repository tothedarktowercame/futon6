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
animated by installing exactly such fingerprints. From the sigil the system
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

---

*TODO (next sessions): §3/§4 prose passes over the primary source; figures
(the panel at frames 1→6; the bout's cross-edge diagram; the vitals card);
§5 worked dis/agreement instances with store queries; decide venue + whether
the two-projections flexiarg ships with the paper or before it.*
