# M-pheromone-field — designed stigmergic signals over the live EFE map (T1 follow-on)

**Status: IDENTIFY (2026-07-05). Chartered by Joe (emacs-repl) as the T1 follow-on
from M-live-efe-map ("it's OK to run M-pheromone-field as a follow on"). Charter
drafted by claude-18. Owner: TBD.**

## The claim gate (inherited, binding)

From M-live-efe-map's C5: *this layer may be called a pheromone field only once agent
behavior demonstrably responds to it; until then it is attention, displayed.* This
mission's whole job is to EARN that name: build deposit/decay semantics AND close the
behavioral loop, or close cleanly having shown the loop isn't warranted.

## What already exists (the substrate — build on, don't duplicate)

All landed 2026-07-04/05 under M-live-efe-map:

- **The field surface**: `/api/alpha/live-efe-map` + the live overlay in
  `futon6/scripts/mission_efe_field.py` (10s poll; 🛸 sessions, 🚀 operator centroid,
  WM-attention pulses).
- **A slow pheromone already running**: live pattern attestation — deposit = each
  A→B turn's `context-retrieval` evidence; decay = the rolling 60-day window
  (`refresh_pattern_attestation.sh`, daily 05:30); field effect = road ink now, spring
  geometry weekly (Sunday 05:00 re-solve, "continental drift" — operator-decided
  cadence 2026-07-05).
- **A fast proto-pheromone (in flight)**: coordination-edge overlay — who belled whom,
  fading with age (codex-1, dispatched 2026-07-05; rendering fade only, no semantics).
- **Self-clocking dispatches**: `agency_send.py --mission` → `clock-dispatch!` — agent
  presence on districts is now cheap and durable.

## The gap this mission closes

The substrate signals are all *derived* (retrievals, ticks, bells happen anyway; the
map counts them). A pheromone proper is a *designed* signal: an agent (or the WM)
**deposits** on a district intentionally or as a side effect of visiting; the field
**decays** by a chosen law; other agents **perceive** it through their normal context
surfaces and are (sometimes) drawn. Three design questions, roughly in order:

1. **Deposit semantics.** What deposits, how much, and where does the accumulator
   live? Candidates: session-presence per clocked district (per tick alive), turn
   volume, WM tick weight, explicit `deposit` tool. The accumulator wants to be
   server-side (the live-efe-map endpoint's JVM) with durable snapshots — NOT
   substrate-2 hot-path writes (I-0 discipline; cf. parked_on.clj's disk-backed
   pattern).
2. **Decay law.** Half-life per signal class; evaporation must be cheap (lazy decay
   computed at read time beats a sweeper).
3. **Perception + response (the hard one).** Where do agents *encounter* the field?
   Candidates: `boot_context`/`mission_context` gaining a "high-pheromone districts
   near your clock" line; the operator bulletin; headless session prompts. The
   success criterion is one demonstrable case: an agent's district/mission choice
   changes because of the field, and the trace shows it. **Guardrails**: attraction
   never auto-dispatches — perception may inform, only the operator (or a
   consent-gated lane, WM-I4) arms action. Autonomous-behavior coupling inherits the
   WM-overnight apparatus guardrails.

## Kill criteria (a clean kill is a success)

- If designed deposits turn out indistinguishable in effect from the derived
  attestation signal (agents respond the same or not at all), close as: *attestation
  IS the pheromone; no separate mechanism warranted* — and record that as the T1
  answer.
- If perception surfaces exist but no behavioral response can be demonstrated in N
  honest trials, the "pheromone" name stays unearned; the deposit/decay machinery is
  mothballed, not left running dark.

## Relations

- Parent: `M-live-efe-map` (T1). Siblings: E-feature-constellation (enacted-pattern
  clusters), E-evidence-flow (which store the deposits/counts live in — its Q1–Q5
  answers constrain the accumulator design), M-capability-star-map (what the WM
  credits). The ants correspondence stays structural: pheromones → this mission;
  colony behavior claims stay out until the loop closes.
