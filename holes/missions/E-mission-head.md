# Excursion: E-mission-head — make the mission HEAD a typed object

**Date:** 2026-06-10
**Status:** :greenfield — HEAD only, by design. This excursion is itself the
demonstration: it is being developed live with the `*mission-overview*` panel
open, so the Scratch-Map builds up from scratch as each phase section lands.
**Spawned:** from E-scope-audit session 1 (W10 + W11), Joe + Fable, unclocked.

## HEAD (Joe, 2026-06-10, verbatim sense)

A mission's HEAD is doing two things at once.

First, in a Jazz or Free Jazz sense, the HEAD is the **theme around which the
whole mission is improvised** — ideally in a fully automated way, by following
the **patterns of improvisation** that Joe intends to learn from the futon5
work. We are not there yet.

Second, each mission is a **virtual peripheral**, and the HEAD is an **AIF
head**: it should be **mappable to the usual AIF terminal vocabulary** —
priors, preferences, observations, policies — which would allow us to
understand, in typed terms, **what it means to satisfy the requirements of
the mission**.

Concretely, the HEAD should therefore carry an extraction of **concepts** —
per the **Interest Network** — named entities like "storage", "agents",
"patterns", "meme graph", "futon3b", "pattern cascades", obtainable just by
*reading the HEAD*. The current kernel extraction is unigram-grade; the
upgrade is Interest-Network-grade extraction over the mission corpus, feeding
the mission-mode concept lane that already exists. The satisfaction-conditions
half connects to E-ground-G: a HEAD mapped to AIF terminal vocabulary is what
grounded-G ultimately discharges against.

## 1. IDENTIFY

### The gap (Joe, 2026-06-10, verbatim sense)

Missions right now are treated as **project management tools** — fine, but
that ignores that project management tools *are* "peripherals" in the
peripheral-spec sense: **constrained execution envelopes**. Gantt charts,
Google Calendars, project plans — it checks out. The problem: those are all
very boring, and **Missions are way cooler — because missions are fun.** They
give you a sense of **purpose, not just progress**. And we don't know how
they will develop until we run them — that is the improvisatory part.
`mission-lifecycle.md` gives the rough shape, but (for example) we don't
always know in advance how we will VERIFY a design — we can do it in a clever
way depending on how the mission shows up. Unlike a Gantt chart, **we can't
know in advance.**

The gap as experienced inside futonic development: a mission's HEAD is both
the **thematic statement of intent** *and*, implicitly, **a small essay
describing the mission's purpose**. With those ingredients we should be able
to build an **AIF model of the mission** (per `futon5/README-aif+.md`), so the
mission becomes an **AIF "lifeform" the moment it is seeded** — we could see
whether it is strong or weak, missing context, what its boundaries are, all
at the start; and by the time it is built it should look *healthy*, the
structural weaknesses ameliorated. Benefits: **#1 automation, #2 quality,
#3 integration** with the AIF-hierarchies modelling.

### Where the gap is already palpable in the stack (evidence pass, 2026-06-10)

1. **`futon2/holes/M-aif-head.md` — COMPLETE 2026-03-15: "AIF Heads for Every
   Peripheral (Mission Peripheral First)."** The strongest evidence: this gap
   was *a closed mission three months ago* ("organs without a body — no shared
   generative model"), yet the gap is still felt. Either it built something
   narrower than the felt need, or it built the right thing and the practice
   never inhabited it (the meme.arrow failure mode: revive-vs-replace applies).
   **First question of MAP: what did M-aif-head actually deliver, and why
   doesn't today's mission tooling feel it?**
2. **Missions already ARE peripherals in code** —
   `futon3c/src/futon3c/peripheral/mission.clj` + `mission_backend.clj` (the
   Mission Peripheral wraps mission-domain operations). The envelope exists;
   what's missing is the *generative model inside it* — the peripheral
   constrains execution but holds no model of the mission's own health.
3. **Hand-built AIF models of missions already exist** —
   `futon5/data/missions/*-exotype.edn` (aif2, coordination, social,
   evidence-landscape). These are exactly "AIF model of a mission" — but
   hand-authored, for four missions, not *seeded from the HEAD*. The gap =
   the missing compiler from HEAD → exotype.
4. **`futon3/library/futon-theory/mission-interface-signature.flexiarg`** —
   the single most-attested pattern in the corpus (≈236 attestations in
   `pattern-attestation.json`): mission-as-typed-interface is already the
   stack's strongest habit, waiting to be formalized.
5. **The capability star-map** (`futon0/.../M-capability-star-map.graph.edn`)
   types missions by `:scope`/`:produces` — a proto-AIF signature (what it
   senses / what it acts on) — and the WM already does EFE over it. A
   seeded-at-birth lifeform model is what that machinery is starving for
   (cf. WM "starved of input sources").
6. **The watcher already parses the essay half** — `parse-mission-md` extracts
   `:mission/summary` (first paragraph) into substrate-2 `mission-doc`
   hyperedges, where it sits **unused for any model**. The HEAD's raw material
   is in the store today.
7. **E-ground-G + this excursion's HEAD**: grounded-G needs satisfaction
   conditions to discharge against; a HEAD mapped to AIF terminal vocabulary
   *is* those conditions. The two excursions meet exactly here.
8. **mission-mode/scope-audit session 1** (`futon6/holes/E-scope-audit.md`):
   the lifecycle is now *visible* (phases, ghosts, in-passing closures) but
   purely descriptive — the panel can show a mission's shape, and has no
   notion of its *strength*. The UI is a lifeform-viewer with no lifeform.

### IDENTIFY exit

The gap is real, felt, and **eight-times palpable**; the decisive unknown is
item 1 (what M-aif-head delivered vs. what is lived-in). → MAP should open
with the M-aif-head autopsy, then inventory the exotype schema against what a
HEAD actually contains.

## 2. MAP (2026-06-10) — autopsy + the futon5 improvisation apparatus

### 2.1 M-aif-head autopsy: built, 0% wired — the head never attended the birth

M-aif-head (COMPLETE 2026-03-15, futon3c commit `51ae8eb`) delivered **real,
well-designed code**:
- `futon3c/src/futon3c/aif/mission_head.clj` (242 ln) — `MissionAifHead`,
  EFE + softmax action selection over a bounded action space;
- `futon3c/src/futon3c/aif/observe.clj` (186 ln) — **ten observation
  channels** (phase-progress, obligation-satisfaction, structural-law
  compliance, prediction-divergence, …) — i.e. exactly the "is this mission
  strong or weak" vocabulary this excursion wants;
- `futon3c/src/futon3c/aif/invariant.clj` (92 ln) — C9: *every peripheral has
  an AIF head*, as an executable check;
- the `:on-cycle-complete` hook installed in `peripheral/cycle.clj:159`.

**And none of it is alive**: zero callers for any export; the hook is
installed but no peripheral config ever passes a callback;
`mission-domain-config` (`peripheral/mission.clj:144`) was never updated; the
`validate-phase-advance` call site uses the 2-arity (no head, no state). Three
handoffs never landed (H-1 AifHead protocol in futon2 — claimed, file absent;
H-2 inventory loader; H-8 portfolio effect sink). **The mission designed the
head but never wired it into the Mission Peripheral's birth ceremony.** The
felt gap = the distance between "the peripheral IS an AIF organism" (promised)
and "AIF scaffolding sits on disk unexercised" (actual). Same failure mode as
meme.arrow (built ≠ inhabited). **Verdict: REVIVE** — the architecture is
sound and load-bearing (cited by later missions); the work is integration,
not invention.

### 2.2 futon5: the patterns-of-improvisation apparatus (ACTIVE — latest commit 2026-06-09)

- **The 256 "universal" patterns exist as the II Ching**: 64 hexagrams × 4
  energy levels = 256 physics rules, library at `futon3/library/iiching/`
  (skeleton: `futon3/library/iching/`, 64 files) — *the pattern library lives
  in futon3, the play lives in futon5*, as stated. Machinery:
  `futon5/256ca.el` (2014 meta-CA, 256 sigils), `src/futon5/hexagram/lift.clj`
  (6×6 exotype matrix eigendecomposition → 6 eigenvalue signs → hexagram lines).
- **The bridge is built**: `scripts/pattern_exotype_bridge.py` (Feb 2026) maps
  **all 791 library patterns → 8-bit exotypes + 36-bit xenotypes** via
  MiniLM(384) → PCA(32) + Ridge, trained on 320 anchors; held-out hexagram
  bit-accuracy 57.9%, Hamming 3.37±1.33; output
  `resources/pattern-exotype-bridge.edn`.
- **The ants↔CA↔CT triangle is working code**, with real transformations:
  pattern→exotype (bridge), exotype→MMCA rule (`mmca/exotype.clj`),
  exotype→ant-AIF deltas (`cyber_mmca/core.clj`), pattern→CT wiring diagram
  (`scripts/pattern_to_wiring.clj`; 6 compiled wirings in
  `holes/e-warranted-play/wiring/`, 2026-06-09), xenotype evaluation registry
  (`xenotype/interpret.clj`, 17 scoring modes incl. filament/EOC).
- Known tension from the play: coupling × diversity are anti-correlated
  (Pareto at gen 12) — relevant later if mission-lifeforms get scored.

### 2.3 The conjecture: mission scopes as early evidence of the 256

Assessment: **credible, unproven, and now testable.** The shape alignment is
real — a 36-bit xenotype is a *situational binding* (where/what/why/who a
pattern applies), which is what a mission scope *is*; pattern
IF/HOWEVER/THEN/BECAUSE parallels the eightfold phases. What does not yet
exist: any explicit hexagram↔scope-type mapping artifact, and the scope
detector does not ingest futon5's enumeration. The honest framing: the binder
vocabulary (11 types) is the *empirical* end and the II Ching (256) the
*universal* end; the test is whether observed scope-instances cluster onto a
small stable set of binding-shapes that embed near hexagram anchors in the
bridge's space.

### 2.4 The synthesis MAP hands to DERIVE (the hot finding)

**The HEAD→AIF-model compiler partially exists already.** A mission HEAD is
text; `pattern_exotype_bridge.py` embeds arbitrary pattern text into an
8-bit exotype + 36-bit xenotype *today*. So the v0 of "mission becomes an AIF
lifeform the moment it is seeded" is a composition of parts on disk:

> HEAD text → (bridge) → exotype/xenotype → hexagram lift → run/score
> (interpret.clj, 17 modes incl. EOC) → **health-at-birth readout** →
> seed `MissionAifHead`'s 10 observation channels (the revived M-aif-head)
> → display in mission-mode (the lifeform-viewer that's waiting).

Every arrow except the first and last two exists as working code. DERIVE's
question is not "can this be built" but **which composition is the honest
v0** — and what "health" means at seeding time (strong/weak, missing context,
boundaries) in terms of the ten observe.clj channels + the xenotype scoring
modes.

### 2.5 This mission's own wiring diagram — written into the meme store (2026-06-10)

Joe's steer: we should build a **wiring diagram per mission** — the
cascade→sorry→wiring-diagram evidence discipline being developed with
claude-3 — and hold mission development to **the same standard as the AIF
learning loop** ("did it help?", realized closure, no laundering). *"We" (Joe)
are the Cyborg version of that same learning loop.* No time like the present:
§2.4's chain **is** a wiring-diagram-with-gaps, so it now lives as arrows in
the live futon3a meme store (`futon3a/meme.db`), scope-tagged
**`diagram/E-mission-head`** (the R3 named-composite container isn't built
yet; the tag is the v0 grouping):

- **4 × `:constructed`** (mode `:construction`, evidence in rationale):
  head-text→exotype/xenotype (the bridge); encoding→hexagram (lift);
  encoding→evaluation-score (interpret.clj); aif-head-code→ten-channels
  (M-aif-head's dormant deliverable).
- **3 × `:open`** — the mission's sorries, RHS specified:
  score→**health-at-birth-readout**; readout→**seeded-beliefs** (the unwired
  birth ceremony, §2.1); readout→**mission-mode-lifeform-lane** (the viewer
  awaiting its lifeform).

Closing this excursion honestly = `promote!`-ing those three arrows
`:open→:constructed` with real constructions — measurable in the store, per
the E-ground-G standard (closure events, not assertions). Note in passing:
minting required applying the meme schema's own pending `advances_cap`
migration (schema.clj's exact DDL; locked WM-policies contract §4a) — the
store's first non-claude-4 write found the seam honest.

## 3. DERIVE — opened with a candidate runner (2026-06-10)

### 3.1 The probe: this HEAD, through the default chain

`futon5/scripts/head_exotype_probe.py` (new; the **candidate runner**) runs
the bridge's *default* representation on a mission HEAD — deliberately, so
representation failures surface honestly (Joe's prediction: "our default way
of doing that wouldn't actually work well"). Run on this excursion's own HEAD
(1190 chars):

- **exotype `00011100`** → rotation=0, match-threshold=1,
  invert-on-phenotype=true, mix=0 — but **bit-confidence 0.29 (LOW)**:
  per-bit proba `0.13 0.44 0.43 0.61 0.56 0.55 0.25 0.32` — four of eight
  bits are near coin-flips. Three bits are confident (b0=0, b6/b7≈0); the
  middle of the rule is noise.
- **nearest anchors are all weak** (cos ≈ 0.35; nothing close): exotype-242/243,
  **hexagram-38 Kui 睽 (Opposition/estrangement)** and **hexagram-49 Ge 革
  (Revolution/molting)** — poetically apt for a mission about a felt gap and
  missions-becoming-lifeforms, but at Δcos 0.002 the poetry is not signal.
- **xenotype (36-bit): not computable at all** — the bridge derives it from
  IF/HOWEVER/THEN/BECAUSE sections, and a HEAD has none. Plus a length
  mismatch (1190-char prose vs short pattern texts the anchors were trained on).

**Prediction confirmed: the default representation is weak here** — and now
that's a measured fact with a runner, not a vibe.

### 3.2 The representation finding (the convergence)

The failure is *structural*, and the fix is something this excursion already
wanted for independent reasons: **the xenotype path requires sectioned text,
and W11's HEAD→AIF-terminal-vocabulary mapping IS a sectioning of the HEAD**:

> intent/theme ≈ IF (priors) · the felt gap ≈ HOWEVER (observation
> divergence) · the move ≈ THEN (policies) · the warrant ≈ BECAUSE
> (preferences/values)

So the representation change that makes the bridge work on HEADs and the AIF
mapping that makes satisfaction conditions typed are **the same move**: parse
(or author) the HEAD into AIF-terminal sections, embed section-wise, and the
36-bit xenotype — the *situational binding*, i.e. the lifeform's boundary —
becomes computable. v0.1 of the runner = section the HEAD, re-run, compare
bit-confidence against today's 0.29 baseline (falsifiable: if sectioning
doesn't raise it, the weakness is deeper than structure).

### 3.3 The recast (Joe steer, 2026-06-10): HEAD as design pattern — Golemization

> "The HEAD should be recast as a design pattern. That's exactly what we did
> for AIF ants — we installed sigils (II Ching patterns) into them, to
> **Golemize** them."

The ants precedent (`futon5/src/futon5/cyber_ants.clj`, `cyber_mmca/core.clj`):
an ant is animated not by prose but by an installed **sigil** — the pattern
already compiled to its 8-bit animating word, translated into AIF deltas. The
Golem's shem. Same move for missions: author (or project) the HEAD in the
**IF/HOWEVER/THEN/BECAUSE** pattern form, and the bridge's section path gives
the full 36-bit xenotype — the sigil to install in the mission peripheral at
birth. **This excursion's HEAD, recast:**

**IF:** you want missions to be living things — purpose, not progress —
improvised around a theme the way a jazz head seeds a performance, ideally
automatable by following learned patterns of improvisation.

**HOWEVER:** missions are currently treated as project-management tools —
Gantt-grade constrained execution envelopes; the HEAD is prose; nothing can
read a mission's strength, weakness, missing context, or boundaries at
seeding time; satisfaction of the mission's requirements is untyped.

**THEN:** recast the HEAD as a design pattern and compile it — install its
sigil (exotype + xenotype via the pattern-exotype bridge) into the mission
peripheral at birth, Golemizing the mission: an AIF model seeded from the
theme, with health observable from day one.

**BECAUSE:** each mission is a virtual peripheral and the HEAD is its AIF
head; a HEAD mapped to AIF terminal vocabulary yields typed satisfaction
conditions (what grounded-G discharges against); and the ants showed that
installing II-Ching sigils into AIF agents is sufficient to animate them.

### 3.3.1 v0.1 RESULT — the Golemization run (2026-06-10)

The probe's pattern-mode ran on §3.3's recast (same projector, section-wise):

| section | bits | conf | nearest anchor | cos |
|---|---|---|---|---|
| IF | `00011100` | 0.28 | hexagram-49 **Ge 革** (molting/revolution) | 0.284 |
| HOWEVER | `00011001` | 0.36 | hexagram-13 **Tongren 同人** (fellowship) | 0.306 |
| THEN | `10101100` | 0.32 | iiching/exotype-248 | **0.454** |
| BECAUSE | `11001000` | 0.37 | hexagram-31 **Xian 咸** (mutual influence) | **0.461** |

**xenotype-32 `00011100·00011001·10101100·11001000`, mean-conf 0.33** (baseline 0.29).

Findings, honestly weighted:
1. **Qualitative unlock confirmed: the sigil now exists.** The xenotype was
   *uncomputable* before the recast; the Golem has a complete shem.
2. **Bit-confidence lift is modest** (+0.04). Sectioning helps; it does not
   transform the projection. (Caveat: v0.1 reused the whole-pattern projector
   per section; the bridge's official path trains per-section projectors —
   some headroom there.)
3. **The real signal moved into anchor proximity, asymmetrically**: THEN and
   BECAUSE land at cos ≈ 0.46 — a third stronger than the whole-text best
   (0.349) — while IF/HOWEVER stay weak (≈ 0.29). Reading: **the actionable
   half of a HEAD (move + warrant) projects into pattern-space well; the
   intent/gap half is mission-specific prose the pattern-grain anchors don't
   cover.** Supports next-car 3: HEAD-grain anchors (missions with known
   outcomes) for the IF/HOWEVER half.
4. The oracle stayed coherent: intent→*molting*, gap→*fellowship*,
   warrant→*mutual influence*. Recorded as flavor, not evidence.

### 3.3.2 The lifeform package: `E-mission-head.aif.edn` (2026-06-10)

Joe: an AIF+ lifeform should ship with a **`.aif.edn` package** — they are
supposed to be able to **contend in a Calculemus battle** with other AIFs. So
this mission now has one (sibling file, canonical AIF+ v2 shape per
`futon5a/holes/stories/leaf-argument.aif.edn`): the **sigil** (exotype +
xenotype-32 + per-section anchors), **birth vitals** (`:health` — completeness
32/36, bit-confidence 0.33, the proximity asymmetry, construction-state 4/3),
the recast as spine+claims, four **falsifiability nodes** (F1 discharged-PASS
= the v0.1 result; F2 decorative-readout, F3 uninformative-sigil, F4
intent-not-typeable — all spec-only), and support/falsifies edges.

**Contendability proven**: staged via `futon5a/scripts/run_aif2_contest.clj`
("Calculemus!") against `leaf-argument.aif.edn`. Verdict `indeterminate` —
correctly: no cross-edges were authored, and the package's structural metrics
match the canonical exemplar's. A real bout = authoring attack edges against
an opponent's claims; that is a deliberate future move, not tonight's.
(Runner quirk found: `(drop 1 *command-line-args*)` eats the first flag under
`bb`; invoke with a dummy leading arg.) Also recorded: **Skolemization as an
alternative Shem-computation** — the three `:open` arrows are existentials;
naming their witness functions is another way to write the animating word.
*"Skolemization: tired → Golemization: wired."*

### 3.4 DERIVE next cars (superseded in part by §4 amendments)

1. **v0.1 runner**: section this HEAD per §3.2 (hand-sectioned first — cheap),
   re-run, report Δconfidence + whether a xenotype materializes.
2. If Δ is real: the `health-at-birth-readout` open arrow gets its first
   component (bit-confidence + anchor-proximity + xenotype-completeness as
   health channels), and the HEAD-authoring convention gains a typed template.
3. Anchor-set question for later: 1190-char HEADs may want HEAD-grain anchors
   (missions with known outcomes) rather than pattern-grain ones — the
   mission-corpus equivalent of the 320-anchor set.

## 4. ARGUE (2026-06-10) — the lifeform's self-argument: why should I exist?

**Form (a first):** this ARGUE was conducted *as a Calculemus bout* — the
excursion's own `.aif.edn` lifeform contending against a **steel-manned
adversary package** (`E-mission-head-contra.aif.edn`) whose every claim is
true: the **graveyard base-rate** (M-aif-head 0% wired; meme.arrow 5 months
empty), the **Goodhart surface** (health-at-birth invites sigil-gaming), the
**jazz objection** (typing the HEAD Gantt-ifies the very thing that makes
missions fun), and the **thin-signal objection** (+0.04 mean-confidence).
Runner: `futon5a/scripts/run_aif2_contest.clj`.

**Round 1 — the arena caught a real flaw:** `a-thesis-attacks: 0,
b-thesis-attacks: 1`. The lifeform rebutted every contra pillar but never
struck the contra's thesis — it argued like a defender, not a contender
(undercutting supports ≠ defeating the conclusion).

**Round 2 — the thesis-strike, and symmetry:** added `n0 →attacks→ nC0`:
*remaining prose is not the safe default — it is the status quo that already
failed, measurably* (the eight-times-palpable gap; the WM starving; untyped
satisfaction giving grounded-G nothing to discharge against; M-aif-head dying
**invisibly** precisely because nothing measured its non-wiring). Result:
**Symmetry: mutual** (thesis 1:1, 6 vs 5 cross-edges).

**What the contra earned — amendments codified in the package:**
- **A1 (from nC2/Goodhart):** the health readout **never gates birth, never
  enters any optimization target** (G, EFE, fitness) — observation + sparse
  audit only. The Xbox-achievement discipline, now an invariant; violating it
  triggers the lifeform's own F2.
- **A2 (from nC1/graveyard):** **wire-or-die** — if the three `:open` arrows
  in `diagram/E-mission-head` are not promoted with real constructions within
  the next two working sessions on this thread, the lifeform self-declares
  `:moribund` in its own package. The graveyard claim is load-bearing, not
  dismissed.

**Verdict: `indeterminate` — and that is the honest ARGUE exit.** Both sides'
falsifiability nodes are `:spec-only`; existence is warranted *conditionally*:
the lifeform lives exactly as long as F2–F4 stay undischarged-against-it and
A2's clock is respected. The strongest defense made (nBEC vs nC3) stands as
the excursion's core position: *a jazz head IS a typed object — changes, key,
form — and the solos are wild because the theme is stable; the ants were
animate BECAUSE of their sigils.*

**ARGUE exit → operator ratification.** Per house discipline the phase closes
on Joe's call: ratify the amendments (esp. A2's two-session clock) and the
conditional verdict, or send the lifeform back to the arena.

### 4.1 Plain-language argument (the version anyone can read)

When we start a piece of work, we write a short statement of what it is for.
Right now that statement is just words: nothing else in the system can read
it, check it, or use it. So nobody can tell, at the start, whether a piece of
work rests on solid ground or shaky ground — and when work quietly stalls,
nothing notices. This has already happened here: a major piece of work was
"finished" on paper three months ago, and nobody saw that it was never
actually connected to anything.

The proposal: when work begins, turn its opening statement into a small
summary the system can read — what the work wants, what is in the way, what
it will do, and why. From that, the system can show, from day one, how strong
or weak the work looks, what is missing, and what would count as finishing.
As the work proceeds, the same readout shows whether it is getting healthier.

What we checked before believing this: we tried it on this very piece of
work. It partly worked — the "what to do" and "why" parts came through
clearly; the "what we want" part needs better reference material, which we
now know how to gather. We also put up the strongest objections we could
find, and answered each one:

1. *"These summaries will be built and then ignored, like last time."*
   Last time, nothing showed that the finished work was never put to use.
   This time there is a deadline: if the pieces are not in use by then, the
   work is marked as failed, in a place everyone can see.
2. *"People will write their statements to score well."* The readout is never
   used as a target or a gate. It is information only, checked occasionally
   and independently.
3. *"This will make creative work bureaucratic."* The summary describes only
   the starting point, never the path. The path stays free — that freedom is
   the point of working this way, and it is kept on purpose.

Every claim above comes with a stated test that would prove it wrong, and the
first of those tests has already been run and passed.

Why this is needed, in one line: **work should be able to say what it is for,
in a form that lets anyone — or anything — check how it is doing.**

### 4.2 The argument from the pattern language — a small cascade

The most important version was missing: the ARGUE rebuilt as a **Pattern
Language Cascade** — the argument as a composition of *real library patterns*,
checked into the arrow store (`futon3a/meme.db`, scope-tag
**`cascade/E-mission-head-argue`**, 3 `:constructed` + 3 `:open`).

**The demonstrated chain (real patterns, composition shown by this ARGUE):**

> `futon-theory/mission-interface-signature` (work declares a readable,
> checkable interface — the thesis joint; ≈236 attestations, the corpus's
> strongest habit)
> → `peripherals/read-existing-seam-before-implementing` (build it by
> reviving M-aif-head's channels + the exotype bridge — the graveyard answer)
> → `realtime/liveness-heartbeats` (revived parts must emit liveness or die
> visibly — amendment A2 *is* this pattern applied to mission wiring)
> → `mission-coherence/logic-model-before-code` (falsifiers stated before
> infrastructure: F1–F4 authored first; F1 already ran — the thin-signal answer)

**The three coverage gaps (`:open` candidate-pattern arrows — the argument
needs them; the library lacks them):**

1. **`candidate-pattern/two-projections-of-one-quantity`** — prose HEAD and
   compiled sigil as two projections of one intent. *Audit finding:* this was
   cited as `structure/two-projections-of-one-quantity` in M-memes' PSR — a
   PSR leaning on a pattern that **was never minted**. The cascade exposes it.
2. **`candidate-pattern/measure-never-target`** — amendment A1 (readouts are
   information, never gates or optimization targets); recurs across peradams,
   health readouts, attestation — undocumented as a pattern.
3. **`candidate-pattern/stable-theme-enables-free-improvisation`** — the core
   defense against Gantt-ification; load-bearing in two missions, not in the
   library.

Per the grounded-learning loop's own economics (C-falsifiable-missions §7 A3),
coverage-gap patterns are the **highest-information units in the store** —
so the cascade's gaps are not weaknesses of the argument but its most
valuable output: three candidate patterns, each born with a use-site and a
rationale. This is the cascade→sorry→wiring discipline applied to an
*argument* rather than code — more work checked into the store, as intended.

## 5. VERIFY (opened 2026-06-10) — two derivations of one argument

**ARGUE exit (Joe, 2026-06-10):** ratified — the four-voice ARGUE (bout +
amendments + plain-language + pattern cascade) passes out of ARGUE into VERIFY.

### V1 — the convergence probe (the centerpiece; per combining-methods-as-diagnostic)

Our §4.2 cascade was **designed** top-down from the bout. claude-3's **Build 1**
(`futon3a/holes/labs/M-memes-arrows/cascade_construct.py`, phylogeny-grounded,
reviewed PASS — C-falsifiable-missions) **assembles** cascades bottom-up from
the prior distribution over the pattern library. Run Build 1 over this
mission's hole (ψ ≈ "make the mission HEAD a typed, checkable object") and
**diff the two derivations**. The disagreement is the diagnostic:

- **P1 (recoverability):** the assembled cascade ranks the four designed
  patterns (`mission-interface-signature`, `read-existing-seam…`,
  `liveness-heartbeats`, `logic-model-before-code`) high. If yes → the
  designed argument is *recoverable from the prior* — not idiosyncratic; the
  pattern language supports it. If no → either the assembler is blind here or
  the argument leans on unattested structure. Both are findings.
- **P2 (the gap test — sharpest):** the assembler **cannot propose** the three
  `:open` candidate patterns (they are not in the library, hence not in the
  prior). Prediction: the diff shows holes exactly at the three gap joints.
  *Unless* it surfaces near-neighbors that could serve — in which case the
  candidates shrink or die before minting. This is how candidate patterns
  earn their flexiargs: survive the near-neighbor search.
- **P3 (the duals):** Closure 01 showed assembly over-selects (26 patterns,
  6 phylogeny-orphans); Closure 06 notes design under-covers (4 nodes). The
  diff measures both vices on one hole — calibration data for Build 2's
  proposal-scoring (C-falsifiable-missions §7 A2: the cascade is a proposal
  distribution; score its ranking of the used set).

**Form:** follow-up excursion (candidate name `E-cascade-convergence`),
cross-referenced to claude-3's Build 1 — it verifies this mission AND
calibrates their assembler; one run, two consumers. Build 1 may still be in
motion; the excursion waits on its owner's nod, not on this mission.

### V2 — the standing obligations (already armed)

- **A2's wire-or-die clock**: the three `:open` arrows in
  `diagram/E-mission-head` (health-readout, seeded-beliefs, lifeform-lane) —
  promote within two working sessions or the lifeform self-declares
  `:moribund`. This is VERIFY's liveness check, running by construction.
- **F2–F4** (in `E-mission-head.aif.edn`): decorative-readout,
  uninformative-sigil, intent-not-typeable — spec-only falsifiers awaiting
  their runs; F4's run is the HEAD-grain-anchors experiment (§3.4 car 3).

### V0 — the logic model: VERIFIED (2026-06-10)

Joe's steer: don't wait on Build 1 — the established VERIFY is the core.logic
model, and the pattern specifying it (`mission-coherence/logic-model-before-code`)
is **already the fourth pattern in this mission's own ARGUE cascade**. The
argument prescribed its own verification; running it:

`futon3c/src/futon3c/logic/mission_head_invariants.clj` — five structural
invariants over an abstract lifeform trace:

| invariant | encodes | adversarial caught |
|---|---|---|
| `measure-never-target` | A1: no gate/optimize decision consumes the health readout | ✓ |
| `wire-or-die` | A2: at the horizon, no `:open` arrow while claiming `:alive` | ✓ |
| `mode-crossing` | promotion requires a construction; attestation never crosses | ✓ |
| `sigil-provenance` | sigils derive from HEAD sections; hand-set bits forbidden | ✓ |
| `falsifier-first` | every construction's claim has an *earlier* falsifier | ✓ |

**`run-verify` ⇒ `:verified? true`** — conforming witness 0 violations;
all 5 adversarial fixtures caught by their own category (including the
subtle one: a falsifier authored *after* its construction is caught by
ground-step comparison, not just absence).

V1 (the Build-1 convergence probe) stays open as the *empirical* complement —
optional, parallel, on claude-3's timetable; V0 verifies the design now.

### PUR — Pattern Use Record

- Pattern: `mission-coherence/logic-model-before-code`
- Actions taken: encoded the mission's 5 design invariants (A1, A2,
  mode-crossing, sigil-provenance, falsifier-first) as core.logic+pldb over an
  abstract lifeform-event trace; authored conforming witness + one adversarial
  fixture per invariant; ran offline.
- Outcome: success — witness clean, 5/5 adversarials caught.
- Prediction error: minimal — one encoding note: the falsifier-*ordering*
  check resisted pure relational form and needed ground projection (min over
  falsifier steps); the house exemplar had no temporal-order invariant, so
  this is a small extension of the idiom worth carrying back to the pattern.

## 6. INSTANTIATE (opened 2026-06-10) — three arrows, three bells

The INSTANTIATE work *is* A2's three `:open` arrows, dispatched as parallel
bell handoffs from **fable-1** (newly Agency-registered) to the three idle
codex agents — contract-first so the consumers build against the schema while
the producer implements it. The shared contract: **`<mission>.health.json`**
(sibling of the mission doc; sigil + health + degraded mode for prose-only
HEADs; schema fixed in the bells).

| handoff | arrow | agent | job |
|---|---|---|---|
| **H1** — health emitter (`head_exotype_probe.py --emit-health`) | score → health-readout | codex-1 | `…227827-461` |
| **H2** — the revive: wire MissionAifHead into the birth (`:on-cycle-complete`, 4-arity `validate-phase-advance`, `seed-from-health`) + A1-as-test | readout → seeded-beliefs | codex-2 | `…264432-462` |
| **H3** — vitals block in the overview panel (♥ conf · xeno% · section marks · reading) | readout → lifeform-lane | codex-3 | `…264707-463` |

Each bell carries: goal, `:in`/`:out`, acceptance bar, gates (clj-kondo +
check-parens + named tests where applicable), the **hard constraints** (no
touching the running JVM/Emacs; health never flows into gate/optimize — the
V0 `measure-never-target` invariant restated as a handoff term), and
**"bell fable-1 back with summary + commit shas."**

**Review protocol (fable-1, on each bell-back):** read the diff, re-run the
gates, check H2 against `mission_head_invariants.clj`'s invariants, state what
was checked; fix small findings directly (carve-out b). **On acceptance:
`promote!` the arrow in `diagram/E-mission-head` with the commit sha as the
construction ref** — the A2 clock is satisfied by exactly these three
promotions, and each promotion is a closure event for the fold ledger.

## 7. DOCUMENT (opened 2026-06-10) — "Anatomy of a Futonic Mission (from the HEAD down)"

Joe's frame: what this session produced is special — the same mission reached
from **two perspectives** (the scope reading in `*mission-overview*`; the AIF
reading in the lifeform package), a result Daumal would find satisfying in its
own right. DOCUMENT = a **preprint**, companion to "A First Proof Sprint":
running-example format, this mission doc as primary source, no verbatim
repetition. **The missing piece that makes it worth writing: a synthesis of
the two readings.**

Skeleton drafted: `futon6/holes/anatomy-of-a-futonic-mission.md`. The §5
synthesis thesis: *the scope reading is the lifeform's observation space; the
AIF reading is its generative model; a mission is well-formed exactly when
the two cohere — and the coherence is computable* (ghost lines are literally
prediction errors against the canon prior; the H1–H3 wiring is the
interoception↔exteroception binding — the body schema). Bonus discharge: the
synthesis IS the construction for the cascade's unminted
`two-projections-of-one-quantity` — publishing mints the pattern. Peradams
form at reading-agreements.

DOCUMENT remains open until the preprint is real and Joe calls the close
(mission-close is the operator's call); INSTANTIATE bell-backs land first.

### 6.1 INSTANTIATE closed (2026-06-10) — three bells, three folds, A2 satisfied

All three handoffs landed and passed fable-1 review (the bells rang back but
the return path was mis-aimed — fable-1's resume ran in the wrong project
store; root-caused via the job receipts, re-grafted with explicit cwd; the
*work* was flawless):

- **H1 PASS** (`futon5@17e9b53`): `--emit-health` producer; both artifacts
  verified incl. degraded mode. The vitals it computed match §3.3.1 exactly.
- **H2 PASS + one review fix** (`futon3c@9c4033b`): the M-aif-head **revive**
  — `seed-from-health`, `:on-cycle-complete` wired, 4-arity refusal surface,
  and **A1 implemented as a runtime check-law** (beyond the bar). Fix:
  `degraded?` now checks the xenotype, not the sigil container (+ contract-
  shaped fixtures); 5 tests / 21 assertions green.
- **H3 PASS** (`futon3c@5b87192`): `♥ vitals conf 0.33 · xeno 89% · ⊗IF
  ⊗HOWEVER ✓THEN ✓BECAUSE` — rendered live in `*mission-overview*` from the
  H1 artifact. The lifeform-viewer has its lifeform.

**All three arrows `promote!`d `:open → :constructed`** with commit shas as
construction refs — `diagram/E-mission-head` now reads **7 constructed / 0
open. A2's wire-or-die clause is satisfied with the clock barely started;
the lifeform keeps `:alive` honestly.** Three fold records added (the H2 one
credits `read-existing-seam-before-implementing` — the revive was the fold).
Residue for claude-4's seam: `promote!`'s CH2 substrate-2 emission wants a
`:payload` on the arrow — promotions succeeded, secondary emission skipped.
