# Anatomy of a WM Flight — a workup from three specimens

*Third in the anatomy series (Mission, Proof, now Flight). 2026-06-11,
fable-1 (ground control), from the first twenty flights' primary sources.
Preregistration baseline: `futon6/holes/early-closures.md` — the closure
grammar (cascade / sorry / construction + fold record) is what we expected
to find inside a flight; §5 checks whether we did.*

## 1. The object

A **flight** is one turn of the War Machine pilot loop: an agent inhabits
the pilot peripheral, reads the live field, chooses a velocity, possibly
acts, measures what the field did, and writes itself into the record. It is
the loop-grade unit of work the way a mission is the campaign-grade unit
and a proof-cycle is the task-grade unit. Twenty specimens exist
(`futon3c/holes/PILOTS-LOG.md` Turns 1–20), each with machine artifacts.

## 2. The organs, and where each lives on disk

| organ | what it is | artifact |
|---|---|---|
| **field-read** | the ranked differential dT at begin (116 actions, each with G, counterfactual G, rationale) | γ frame `:dT-snapshot` (`futon3c/data/repl-traces/live-*.edn`) |
| **velocity** (v) | the chosen action, with attribution (guarded-top / operator-directed / chosen-target) | frame `:v`, `:v-attribution` |
| **warrant** | why THIS v — pattern-warrant, standing contract, operator direction | PILOTS-LOG EVAL + proto-PSR; consent-gate cg-id in frame |
| **prediction** | the field's own G for v, plus the frozen counterfactual | frame `:predicted-discharge`, `:predicted-constant` |
| **begin-state** | the durable stash that survives process death | `live-*.begin.edn` |
| **the act** | real work in the world (a commit, a doc edit, a registry change) — or a HOLD | evidence-ref (commit shas); or a discipline event instead |
| **witness** | the evidence-ref without which `:executed?` throws | frame `:evidence-ref` |
| **measurement** | realised G from a SETTLED post-field read, tagged for admissibility | frame `:realised-discharge`, `:realised-source`, `:realised-read`, `:independent?` |
| **out-of-band gradient** | refusals, declines, merges — the losses G-vs-G can't see | `futon3c/data/discipline-events.edn` |
| **self-record** | the human-readable turn + the typed γ pair | PILOTS-LOG entry; frame `:trace` |

## 3. Three specimens (primary sources, quoted)

### Specimen A — Turn 9: the clean flight (everything worked)

The γ frame, verbatim (dT elided):

> `{:independent? true, :realised-source :measured, :v {:type
> :advance-mission, :target "M-daily-scan", :open-hole-count 4, ...},
> :predicted-discharge -4.1629..., :realised-discharge -4.1235...,
> :prediction-error 0.039..., :predicted-constant -4.0839...}`

And the log's PUR:

> "LESSON APPLIED from cycle 8: verified the target had a pilot-CLOSEABLE
> counted hole (not operator-gated) BEFORE choosing... Settled-read
> protocol honored: forced a confirming scan to avoid the cycle-6
> transient trap. Witness = the committed fix."

Anatomy notes: every organ present and load-bearing. The dual prediction
shows the two models disagreeing in OPPOSITE directions around the realised
value — the constant model "off in the opposite direction (state-blind:
predicts no move)" — which is what makes this single pair evidential.
The act was a real bounded code fix (futon7 `701522d`), the measurement
waited for two agreeing scans, and the per-hole increment (0.039) was BORN
here and confirmed by Turn 10. **A flight in full health.**

### Specimen B — Turn 4: the censored success (the pipe worked, the number lied)

The log, verbatim:

> "**PRINT / FOUND.** field moved, sorry dropped from ranked-actions,
> first independent pair _[predicted G=-4.73, realised G=-4.73 ...]_"

predicted == realised to the digit, error 0.0 — because the discharge
SUCCEEDED so completely that the target vanished and the apparatus copied
predicted into realised. The flight was healthy as an *act* (real mint,
real registry discharge, real field movement) and pathological as a
*measurement* (a censored observation wearing a perfect score). Caught
within the hour; the organ it forced into existence is `:realised-source
:target-absent-fallback` + verdict exclusion. **The anatomy lesson:
success of the act and validity of the measurement are separate organs,
and a flight can have one without the other.**

### Specimen C — Turn 2: the refusal (a complete flight with no act)

The log (hand-backfilled — this flight predates auto-DOCUMENT) and the
discipline event:

> "Live WM recommended **`open-mission M-capability-star-map`**
> (G=−5.698)... Resolved the fork by **pattern-warrant**..."
> `{:discipline/event :teleport-refused, :predicted -5.698..., :note "WM
> top recommendation was an un-earnable teleport... Caught by V2
> earned-closure discipline at PRINT"}`

No frame, no pair, no commit — and it is the single highest-value flight
of the twenty: the refusal exposed the forward model predicting the spawn
of the already-born, which forced the class conversion, which made every
later measurement possible. **A flight's value is not its discharge; the
out-of-band gradient organ exists precisely because the best flights
sometimes produce only a typed refusal.**

## 4. The two readings

| organ | REPL reading (operational) | AIF reading (the charter's) |
|---|---|---|
| field-read | READ: dT snapshot | perception of the niche |
| velocity + warrant | EVAL: choose v, cite why | policy selection under priors |
| act / hold | PRINT: do, or refuse | action — incl. epistemic refusal |
| measurement | LOOP: settled re-read | outcome observation (L1: dynamics) |
| witness + out-of-band | evidence discipline | the L2 seam: what witnessed outcomes will train |
| self-record | the log entry | belief update made public |

The synthesis (same move as the Mission anatomy): the REPL reading is the
flight as the pilot *does* it; the AIF reading is the flight as the
charter *evaluates* it; a flight is well-formed when each REPL organ has
its AIF counterpart actually filled — and the three specimens show the
three ways that coherence is earned: by measurement (A), by honest
invalidation (B), by refusal (C).

## 5. The preregistration check (vs early-closures.md) — REVISED

*(First draft of this section claimed the mapping by prose analogy — the
M-pilot-appearance repertoire without the M-memes-arrows content-hygiene
(Joe's catch, 2026-06-11). A grammar-mapping claim is earned by
CONSTRUCTING the objects in the stores, not by tabling the analogy. Done
now for the specimens; the split below marks what is demonstrated vs what
remains asserted.)*

**Demonstrated, store-grade:**
- **Specimen A's cascade, actually run**: `construct_cascade(ψ = the
  day-counter hole)` → size 1, **C=0.369**, sole pattern
  `scan-coherence/mission-anchored-scan` (rel 0.369). The library had
  almost nothing for this hole — and honestly, the fold used none of it:
  plain code closed it. Fold record written to `closure-folds.edn`
  (`wm-flight/turn-9-daily-scan-item-368`, `:used []`) — the same shape as
  closure-01's "investigation closure, no fold". Note the hygiene point
  inside the record: the flight's disciplines (settled-read,
  target-verification, witness) are LOOP organs, not fold patterns;
  putting them in `:used` would launder loop-hygiene into
  pattern-utility.
- **Specimen B's fold record** (`wm-flight/turn-4-two-projections-mint`):
  `:used ["structure/two-projections-of-one-quantity"]` with the real
  discharge refs — arrow `arr-4d50ce67-10b` promoted WITH payload, CH2
  sorry-ref `futon3a/sorry/meme-arrow-969d5eb3d8b6b363`. Fold-success and
  measurement-validity recorded as separate facts at the fold grain.
- **The closure grammar's three states exist for B in the actual stores**:
  cascade (`cascade/E-mission-head-argue` scope-tags on the provenance
  arrow), sorry (registry entry + arrow→sorry-doc in substrate-2),
  construction (`futon3@c1c0325` in the arrow payload).

**Still asserted, not yet constructed:**
- Specimen A's hole as a substrate-2 subset: item 368 lived as a counted
  doc-hole, never as a store sorry — the hole→store half of the grammar is
  REAL only for holes that pass through the registry/meme-store (B), and
  merely analogical for counted mission holes (A). Closing that gap is the
  hole-granularity work (counted sub-holes as store objects).
- Specimen C has no fold record: a refusal folds nothing — whether
  refusals deserve their own typed place in the folds file (as β-grade
  evidence about the FIELD rather than about patterns) is an open
  recording-discipline question.

**What survives of the original claim:** the flight still ADDS the
price-tag organ (dual-model predicted-vs-realised with admissibility
tags), now backed by A's actual frame; and the fold-record half of the
preregistration is now demonstrated rather than asserted. The hole-state
half is demonstrated for store-borne holes only.

## 6. What the schematic (M-pilot-appearance) doesn't show

The schematic says READ→EVAL→PRINT→LOOP. The specimens show the flight's
actual mass is in organs the schematic has no boxes for: target
*verification* before choice (A's "verified pilot-closeable BEFORE
choosing"), the settled-read *wait* (A forced a confirming scan), the
witness *requirement* (no ref, no executed), the *hold path* as a
first-class outcome (C), the begin-state *durability* (flights survive
process death), and the *counterfactual logging* (every prediction carries
the model it argues against). The schematic is the skeleton; these are the
immune system — and arcs 1–2 demonstrate the immune system was built one
infection at a time.

## Open

- Specimens not yet worked: a build-dispatch flight (T7/T10 — the
  three-agent shape), a pattern-mint flight under the new mint-PSR rule,
  and a future L2-witnessed flight (none exist yet — pudding-G1).
- The daily scan wants the same treatment (per Joe's 無 on §367): its
  organs include term-provenance, which flights don't have.
- This doc feeds the PDF/preprint pipeline like its siblings if wanted
  (render_mission_pdf.py handles the format).
