# M-live-efe-map — agents on the EFE landscape, ants-style

**Status: VERIFY complete (2026-07-04) — HEAD and IDENTIFY ratified by
Joe 2026-07-04 evening; MAP verified the inventory and found the
frontier gap; DERIVE locked the two-tier placement/freshness rule;
ARGUE defended it; Joe ratified DERIVE+ARGUE (D-frontier, D-method, C3
two-tier, C4/C5 inspection-only) 2026-07-04 and VERIFY ran the design's
claims against the live systems same session (claude-18). All checks
pass; three operational findings recorded for INSTANTIATE. C3 cadence
resolved by Joe: ≈daily; the one-off re-embed ran same session
(coordinates 212 → 248, BGE file regenerated, endpoint serving the new
set live).**

## HEAD

### Operator-voice anchor

Dictated by Joe over emacs-repl, 2026-07-04 evening, across two turns of a
riff; wording preserved:

> "One of the issues that we looked at in the aif-wiring-explainer is R15
> 'higher level — nested model' and another is R11 'Other agents — multi
> brain'. It has occurred to me that we have seen this before with the AIF
> ants simulator. The ants, by using pheromone, develop a kind of
> collective perception. It also occurred to me that the EFE landscape map
> gives us something very akin to the layout of 'food' in the ant
> simulator. The 'food' here is reducing free energy. Whereas previous
> ~/code/p4ng generation AIF controllers were per-agent, now we could do
> something a bit cheaper to get started: simply register our agents
> against the landscape and see where they are working. We could
> eventually start to push some of the global AIF model 'into' the agents,
> simply by looking at how agent/session information maps to
> landscape/mission/pattern material. We have those cross-maps now via
> things like session-mode. Furthermore, unlike the ants, we can create
> more interagent/intersession signals as needed."

And from the opening turn of the riff:

> "What if we let the little ship represent *my* cursor and let it move
> around the map a bit? And what if we made the map *live* so that it
> showed which agents are running around on it? Then suddenly the EFE map
> starts to look *almost exactly* like the AIF Ants map! Rather than
> running around gathering food, our agents would be running around fixing
> missions, gathering stars and upgrading badges, etc — and over time, the
> landscape would shift noticeably."

The rocket precedent: `futon6/data/mission-efe-field-embed.html` already
drew a ship showing *where* M-emacs-cursor-peripheral sat in the
embedding — a mission that was finished, was "fun and satisfying as an
interactive mission," and whose code was later reused for better websocket
support in another Emacs integration. The ship now gets to be *Joe*.

### What is already felt to be true

- The ants correspondence is structural, not decorative: pheromones →
  collective perception is the R11/R15 mechanism already demonstrated once
  in this stack, and the EFE landscape is the food layout with "food =
  reducing free energy."
- The cheap first step is real: agent→mission clock edges are already
  persisted to substrate-2 (`clock-dispatch!` posts hyperedges), the
  invoke-jobs ledger streams live per-agent activity (built 2026-07-04,
  M-custom-harness), and the embedding coordinates exist. "Register agents
  against the landscape and see where they are working" is a join, not a
  build.
- The cross-maps needed to later push the global AIF model *into* agents
  exist (session-mode; agent/session ↔ landscape/mission/pattern
  material). Per-agent controllers (the p4ng generation) are the
  *eventual* shape, deliberately not the starting one.
- Unlike ants, the signal vocabulary is ours to extend: interagent /
  intersession signals can be *designed* as needed (bells/whistles and the
  coordination ledger are the existing precedents).
- Adjacent settlement carried in: the remaining aif-wiring-explainer
  badges are like M-capability-star-map's stars — not needed to *run* the
  WM (the overlay-as-prosthetic finding). On this map they become field
  features: legibility, not requirements.

### Anti-glibness discipline

- **Gamification guard** (standing operator feedback): "gathering stars
  and upgrading badges" must remain a *description of the display*, never
  the objective function. The defensible frame is inspection surface —
  the map answers "where is everyone and why." If agent behavior ever
  optimizes for map features rather than mission outcomes, that is a
  defect, not emergence.
- **Don't claim collective perception before the signals exist.** Dots on
  a map is telemetry; collective perception requires the loop where
  field-state actually modulates agent behavior. Name the difference in
  every phase.
- **Landscape shift must be earned**: the EFE field moving because work
  completed is the point; the field moving because a visualization
  parameter changed is noise dressed as signal.

### Working-economy position

Underwrites: a live R11/R15 evidence surface for the AIF R-contract
(collective perception as an inspectable phenomenon rather than a claim);
WM legibility for the operator (the E-wm-operator-lane concern, made
spatial); the operator-not-sovereign principle made visible — Joe's ship
is one embodied agent among the others, on the same field.

Underwritten by: the M-custom-harness observability layer (registry,
invoke-jobs events, `recent_coordination`); clock-lineage agent→mission
hyperedges; the EFE embedding (`mission-efe-field-embed.html`) and the
possible-world-regulator EFE sweep for eventual recompute; the finished
M-emacs-cursor-peripheral (cursor stream + reusable websocket code);
session-mode cross-maps; I-0 (serve from the one JVM).

### Clarity-gap / carried-forward tensions

- **T1 — signal design:** which interagent/intersession signals count as
  pheromone (evaporating, additive, field-borne) vs. which are just
  messages? The bells/coordination ledger are message-like; a pheromone
  layer may want deposit/decay semantics. Deliberately unresolved.
- **T2 — push-into-agents boundary:** at what point does "looking at how
  agent/session info maps to landscape material" become a per-agent AIF
  controller, and is that a later phase of this mission or the p4ng
  successor mission's territory?
- **T3 — cursor privacy/grain:** the ship is Joe's cursor. What grain is
  shown (file? mission? repo?), and does it stream always or only when
  invited? (The HUD/Bayesian-model work is adjacent prior art.)
- **T4 — landscape recompute cadence:** live-per-event, per-checkpoint,
  or nightly? "Over time, the landscape would shift noticeably" suggests
  slow is acceptable and maybe preferable.
- **T5 — badge/star ontology:** field features vs. capability claims —
  the capability-overlay read-contract exists; this map must not fork it.

### Provenance

Operator riff over emacs-repl, 2026-07-04 evening (two turns), captured
same evening by claude-16 with Joe's wording quoted verbatim; the
connective context (which infrastructure already exists, the stigmergy /
niche-construction reading) is from the same conversation. Intake method:
live dialogue, not interview.

**HEAD exit:** awaiting Joe's recognition that this is faithful to the
mission's live shape. IDENTIFY should begin from the "smallest live
slice" sketched in-conversation: one aggregating endpoint (agent →
clocked mission → embedding coordinates) plus the static embed polling
it — real agents visible on the real map before any ship, any pheromone,
any recompute.

## IDENTIFY (2026-07-04, evening — fused with a thin MAP)

Operator framing (Joe): "IDENTIFY and MAP are rather similar — this is
more of an 'integration' mission than a fresh build… The main 'gap' for
IDENTIFY to look at is: what existing features do we need to integrate to
make this work?" Accordingly, the gap statement IS the integration
inventory, and MAP reduces to verifying the rows marked *verify*.

### The gap: an integration inventory

| Piece | Provides | Status | Integration point |
|---|---|---|---|
| EFE field embed (`futon6/data/mission-efe-field-embed.html`, 2.4 MB, data inlined; `mission-carpet-pos-embed.json` as probable coordinate table) | the landscape: mission positions + EFE field, incl. the rocket precedent | exists, static | becomes the base layer; live layer overlays it — the inline field is NOT regenerated per event (*verify: which artifact is the coordinate source of truth*) |
| Clock-lineage agent→mission hyperedges (`futon3c/src/futon3c/agency/clock_lineage.clj` `clock-dispatch!` → substrate-2) | where each agent IS on the mission map, durably | exists, live (writes confirmed 2026-07-04: zai-9's clock) | the join key: agent → mission-id → embedding coords |
| Registry + invoke-jobs ledger (`/api/alpha/agents`, `/api/alpha/invoke/jobs` with live text/tool events, M-custom-harness) | who is alive; what each agent is doing right now | exists, live | dot state: idle/invoking + activity caption |
| `recent_coordination` / coordination ledger | who belled whom — the message-trails | exists, live | optional edge overlay (the proto-pheromone; real deposit/decay semantics stay OUT, T1) |
| Session-mode cross-maps (`futon3c/emacs/session-mode.el` + session↔mission↔pattern material) | the agent/session ↔ landscape/mission/pattern mapping for the eventual push-into-agents step | exists | read-side only in this mission (T2 boundary) |
| M-emacs-cursor-peripheral code (descendant probably `futon3c/emacs/futon-agency-ws.el` — *verify location*) | Joe's cursor stream | finished mission, code reused once already | ship position = cursor file → mission via enrichment layer (`enrich-file`) → coords; grain + opt-in per T3 |
| One serving JVM (I-0) + WebArxana static serving | where the endpoint and page live | exists | ONE new aggregating endpoint (agent → clocked mission → coords); page polls it |
| Capability stars / badges overlays (M-capability-star-map read-contract) | field features for legibility | exists | rendered as landscape features only; MUST consume the existing read-contract, never fork it (T5) |

**The only genuinely new code:** the aggregating endpoint (a join of
three existing reads) and the overlay/polling layer on the embed page.
Everything else is wiring to things that already run.

### Completion criteria

- **C1 (the smallest live slice):** real agents visible on the real map
  at their clocked missions, dot state reflecting live invoke activity,
  served from the one JVM; `pgrep java` unchanged (I-0 witness).
- **C2 (the ship):** Joe's cursor appears as the rocket at
  file→mission grain, opt-in streaming (T3 resolved by operator choice).
- **C3 (freshness contract):** an explicit, recorded decision on map
  freshness — the operator's candidate: re-embed at session end vs.
  slower cadence (T4) — implemented at whatever cadence is chosen. The
  base field may be stale-but-dated; agent positions are always live.
- **C4 (guard):** the gamification guard holds under review — no agent
  behavior or prompt optimizes for map features; the map remains an
  inspection surface. (A review criterion for every phase, not an
  artifact.)

### Scope out (hard)

Pheromone deposit/decay semantics (T1 — follow-on mission if wanted);
per-agent AIF controllers (T2 — the p4ng-successor's territory);
landscape recompute machinery beyond the chosen C3 cadence; any write
path from the map into agent behavior.

### Relationship to other missions

Consumes: M-custom-harness (observability layer — this is its second
driving-grade consumer), M-capability-star-map (overlay read-contract),
M-emacs-cursor-peripheral (finished; code lineage), possible-world
regulator (eventual re-embed machinery, C3). Feeds: the aif-wiring
R-contract evidence surface (R11/R15), E-wm-operator-lane (spatial
operator legibility), and — if T1/T2 later open — the p4ng-successor
per-agent controller line.

### Owner and dependencies

Owner: unassigned (candidates: claude-driven integration, or a second
zai driving experiment once M-futon1b-port's verdict is in). Repos:
futon6 (map + doc), futon3c (endpoint + Emacs side), futon5/futon2
(overlay contracts). Operator gate: Joe (T3 privacy, C3 cadence,
phase ratifications).

**IDENTIFY exit:** Joe agrees the gap-as-inventory is right and the
scope cut (in: integration + one endpoint; out: pheromones, controllers,
recompute machinery) matches his intent. MAP then = verifying the two
*verify* rows and confirming the smallest-slice join returns sane data.

### IDENTIFY addendum — the War Machine layer (operator, same evening)

Joe, wording preserved:

> "The one extra thing that I think we could add at the IDENTIFY stage is
> showing 'the War Machine itself' somehow — insofar as we are now in a
> position to get a live reading on *it* not just on the agents. This is
> what I have been working on with claude-10 (turning the
> wm-achievement-ledger / flight log / ./p4ng/sequel-notebook.org and
> other similar documents that attempt to show what cascades + sorries +
> wiring diagrams are being evaluated and computed by the war machine)…
> all of *that* could be redisplayed live on the same surface. In that way
> the War Machine itself could be acting as a kind of 'pheromone field'
> for the agents — this cascade, this sorry, this wiring diagram — these
> ones are deemed important enough to draw in at the sketch level. So,
> interactive sessions might go check them out, headless / autonomous
> sessions might be drawn to them for other reasons."

**What this changes.** T1 (signal design) is partially *resolved by
reframe*: pheromone v1 needs no new deposit/decay mechanism — the WM's
current evaluation activity IS the field. "What the WM is attending to"
renders on the map at sketch level; agents (interactive or headless)
perceive it and may be drawn. This is also where R11 and R15 meet on one
surface: the agents are the multi-brain (R11), the WM is the nested
higher-level model (R15), and the map shows both *and their coupling*.
Designed interagent signals with true deposit/decay semantics remain the
T1 follow-on, now with a working field to extend rather than a blank.

**Inventory rows added:**

| Piece | Provides | Status | Integration point |
|---|---|---|---|
| WM live endpoints (`/api/alpha/war-machine`, `/api/alpha/aif-stack/live`, port 7070) | live reading on the WM itself | exist, live | the WM-attention feed |
| wm-achievement-ledger (*verify location*) | what the WM has achieved / is crediting | exists | field feature: achievement marks |
| Flight log (`flight.spec.edn` records; `flight_scope_view.bb`, `flight-mode.el` as existing projections) | which flights / steps are in evaluation | exists | "currently evaluating" markers (*verify: current record source*) |
| `p4ng/sequel-notebook.org` + cascade figures | cascades being computed, with artifacts | exists, session-logged | sketch-level cascade markers |
| Sorries (sorry-arrow contracts) + wiring diagrams (futon5) | the WM's open holes and structures under evaluation | exist | sketch-level markers; "deemed important enough to draw in" |
| claude-10 workstream (Evidence Landscape ← live WM info; session `23b70755…`) | the sibling effort turning these documents into live display | in progress | this mission CONSUMES/ALIGNS with it — same rule as T5: do not fork the sibling's contract |

**Completion criterion added — C5 (the WM-attention layer):** the map
renders, live and at sketch level, which cascades / sorries / wiring
diagrams the WM is currently evaluating, sourced from the live endpoints
and ledgers above. Claim discipline per the HEAD: this layer may be
*called* a pheromone field only once agent behavior demonstrably responds
to it; until then it is the WM's attention, displayed.

## MAP (2026-07-04, late evening — verification of the inventory)

Operator opened MAP with a live fact: the evidence-viewer
(`…:7071/evidence-viewer/index.html`) now shows `author = war-machine`,
`tag = wm-tick` — "the live emission works," with the caveat that the WM
can do more than is currently logged.

**Verified this session (claude-16, against the live store and files):**

- **wm-tick feed: live and rich.** 18 entries, author `war-machine`,
  subject `{:ref/type :agent :ref/id "war-machine"}`. Each tick body
  carries `:enacted` and `:target` (MISSION IDS — the join to map
  coordinates is direct), plus `:G :expected-G :realized-G`,
  `:gates {:pass :fail}`, `:candidates`, `:mode` (e.g. "stop-the-line"),
  `:decision` (e.g. "advance-mission"), `:trigger`. C5 is renderable from
  this feed as it exists today: WM attention = pulses at the
  enacted/target missions' coordinates, annotated with G.
- **Coordinate source of truth:** `futon6/data/mission-carpet-pos-embed.json`
  — a dict of mission-id → [x y], 212 missions (embed generation
  2026-06-12). The 2.4 MB field HTML carries its data inline; the JSON is
  the join table.
- **Cursor lineage:** `futon3c/emacs/futon-agency-ws.el` confirmed as the
  websocket descendant (cursor machinery present).
- **WM document layer:** achievement ledger =
  `futon2/scripts/wm_achievement_ledger.bb`; flight-log projection =
  `futon3c/scripts/flight_scope_view.bb` (records per `flight.spec.edn`).

**The MAP finding — the frontier gap.** The smallest-slice join was
tested against tonight's actually-active missions: **neither
M-futon1b-port nor M-custom-harness has coordinates** — both are newer
than the June-12 embedding. The map's most active inhabitants are exactly
the ones the static field does not know yet. Consequences: (1) the C3
freshness contract is not a nicety — without it the liveliest agents are
invisible; (2) DERIVE inherits a named decision, **D-frontier: how to
render agents clocked to unmapped missions** (margin "frontier zone"
placement, provisional nearest-neighbor embedding, or hidden-until-
re-embed — each with different honesty properties). The ants echo is
noted without being leaned on: new work is off the map until the
landscape re-learns it — the white-space-scout's territory.

**MAP exit:** inventory verified, join sanity established (including its
instructive failure mode), C5's feed confirmed live. Ready for DERIVE on
operator ratification: its opening decisions are D-frontier, the C3
cadence, and the endpoint's exact join shape.

## DERIVE (2026-07-04, late evening)

### Operator direction

Joe, on the frontier gap: "the embedding is great, but we could *also*
compute location relative to the embedding cheaply, to find approximate
locations without rerunning the whole embedding all the time."

### D1 — Frontier Placement

**Rule:** an unmapped mission is never hidden merely because the June-12
embedding does not contain it. It receives an approximate position
computed *relative to* the frozen coordinate set, with provenance attached.
The approximation is a live display fact, not a replacement for a real
re-embed.

Placement is attempted in this order:

1. **Relational centroid (default, graph-native):** collect mapped anchor
   missions related to the unmapped mission by mission-scope tree links,
   `relates-to` binders, explicit mission references, clock/coordination
   adjacency, or consumed-by / feeds relationships in mission docs. Place
   at the weighted centroid of those anchors.
2. **Native BGE kNN (when a mission vector exists):** if the mission has a
   record in `futon3a/resources/notions/bge_mission_embeddings.json`,
   find nearest mapped mission vectors by cosine distance and place at
   their inverse-distance-weighted centroid.
3. **Text BGE kNN (fallback, EFE-blind):** if no stored mission vector
   exists and relational anchors are absent or too weak, embed the mission
   text with the same BGE family and place by kNN against mapped mission
   docs.
4. **Frontier shelf (last resort):** if no honest anchor exists, render
   the mission in a labelled off-map frontier band, not at an invented
   coordinate.

The current coordinate source is
`futon6/data/mission-carpet-pos-embed.json` (212 missions). The embed
layout was generated by `futon6/scripts/mission_carpet_variants.py` as
BGE cosine distance to metric MDS, then rendered by
`futon6/scripts/mission_efe_field.py embed`. The prior
`M-efe-bge-followon-actions` result is binding here: 2D BGE-MDS positions
are a useful lens but not a trustworthy ground metric. Therefore DERIVE
uses high-dimensional cosine for kNN ranking and only projects the final
weighted result into the existing 2D display.

### D2 — Approximation Formula

For a target mission `m`, let `A(m)` be mapped anchors with coordinates
`p(a)`. Each anchor receives a non-negative weight:

`w(a) = 3*scope_edge + 2*explicit_ref + 2*clock_or_coordination + bge_knn + 1*doc_mention`

These are **arbitrary starting weights**, not learned constants and not
settled theory. They are intentionally exposed as VERIFY-time tunables;
the same provenance-honesty required of approximate coordinates applies
to the coefficients that produce them. In v1 each term is zero when
absent, and `bge_knn = max(0, cosine(m,a) - tau)` for native/text BGE
methods; `tau` is likewise a starting threshold to tune under VERIFY,
not a hidden invariant. The approximate coordinate is:

`p_hat(m) = sum_a w(a) * p(a) / sum_a w(a)`

If `sum_a w(a) = 0`, the mission goes to the frontier shelf. If all
positive-weight anchors are themselves approximate, the result is marked
`:anchor-depth 2+` and rendered with lower confidence; no chain of
approximations may silently become an embedded coordinate.

The endpoint returns the full placement record, not just `[x y]`:

`{:mission-id, :x, :y, :placement, :method, :anchors, :anchor-depth,
:confidence, :as-of, :source-coordinate-set}`

`placement` is one of `:embedded`, `:approximate`, or `:frontier-shelf`.
`method` is one of `:embedded`, `:relational-centroid`,
`:native-bge-knn`, `:text-bge-knn`, or `:unanchored-frontier`.

### D3 — Freshness Contract

C3 becomes two-tier:

- **Live tier:** agent and WM positions update continuously from clocks,
  invoke-jobs, and wm-ticks. New missions get approximate placement within
  the same session they become active, without rerunning the whole
  embedding.
- **Re-anchor tier:** a slow full re-embed retires approximate positions
  to embedded positions. The DERIVE default is session-end or slower; an
  implementation may expose the cadence as an operator setting, but it
  must record the coordinate-set timestamp used by every rendered point.

This preserves the HEAD discipline: landscape shift is earned by changed
substrate and explicit re-anchoring, while live approximation is labelled
as frontier telemetry.

### D4 — Endpoint Shape

One read-only aggregating endpoint is enough for INSTANTIATE:

`GET /api/alpha/live-efe-map`

It should return:

- `:coordinate-set` — source file, variant (`embed`), generated timestamp
  if known, and mission count.
- `:agents` — connected / recently active agents joined through
  agent-clock or clock-lineage to mission placement records, plus
  invoke-jobs activity state.
- `:war-machine` — latest wm-tick attention records, with `:enacted` and
  `:target` joined to mission placement records and G terms preserved.
- `:cursor` — Joe's ship only when opt-in cursor streaming is active,
  joined by file→mission enrichment and then mission placement.
- `:frontier` — unmapped or approximate missions with placement provenance
  and anchors, so the UI can render them as frontier rather than pretending
  they are part of the old embedding.

The endpoint must not write to substrate-2 and must not influence agent
selection. Any later pheromone/controller loop is a different mission.

### D5 — Render Contract

Approximate and embedded positions must be visually distinct:

- embedded mission: solid point / normal hub
- approximate mission: dashed ring or hollow marker, tooltip naming method
  and anchors
- frontier shelf: labelled off-map band
- WM attention: pulse or halo at enacted/target missions, annotated with G
- agent: live dot at its current clocked mission, with invoking/idle state
- Joe cursor ship: opt-in rocket/ship at file→mission placement

The display may use ants language in captions only as analogy. Until the
field changes agent behavior, the claim is "WM attention and agent
telemetry on the EFE landscape," not "collective perception."

### DERIVE Exit

DERIVE is ready to exit when Joe ratifies:

- D-frontier = approximate out-of-sample placement, not hidden-until-
  re-embed.
- D-method = relational centroid first; native BGE kNN when available;
  text BGE fallback; frontier shelf only when unanchored.
- C3 = live approximation plus slow explicit re-anchor.
- C4/C5 remain inspection-only: no prompt or scheduler optimizes for map
  features, and the WM-attention layer is display until a later mission
  closes the loop.

## ARGUE (2026-07-04, late evening)

### Pattern cross-reference (`futon3/library/`)

The DERIVE design is not just a convenient join. It is the consequence
of treating the live EFE map as an observer-stage peripheral over
existing typed evidence.

| Design element | Pattern | How it supports |
|---|---|---|
| Read-only v1: endpoint + overlay, no writes, no agent steering | `peripherals/read-only-first-then-extend` | This mission wants the later possibility of pheromone/controller loops, but the first safe landing is observation only. The map subscribes to clocks, jobs, wm-ticks, and cursor state; it writes nothing back and therefore cannot corrupt substrate or compete with agent ownership. |
| Canonical endpoint instead of a side channel | `peripherals/canonical-typed-event-vs-side-channel` | Cursor/agent/WM presence must travel through canonical HTTP/WS surfaces and typed payloads, not a quick private wire. The design's single `/api/alpha/live-efe-map` aggregator preserves replay, audit, consent, and multi-client reuse. |
| Live layer as typed observation vector | `aif/structured-observation-vector` | Agent dots and WM attention are not prose captions. They are normalized observations: agent, session, mission placement, activity state, G terms, provenance, timestamp. This is the minimum shape that can later feed AIF without reinterpretation. |
| Approximate placement as provenance-bearing proposal | `sidecar/proposal-ledger-for-fuzzy-output` | Out-of-sample placement is fuzzy. Treating `[x y]` as fact would overclaim. DERIVE therefore returns method, anchors, confidence, depth, and source coordinate set with every approximate point; the UI renders the proposal as approximate until a re-embed promotes it. |
| Frontier shelf and dashed markers | `futon-theory/honest-map-over-flattering-counter` | Hiding unmapped missions flatters the old embedding; pretending approximate points are embedded flatters the new display. The honest move is to show the frontier as frontier, even when that makes the map look incomplete. |
| WM attention and coordination edges visible as topology | `system-coherence/present-graph-topology-not-adjacency-lists` | The live surface's claim is relational: who is clocked where, what the WM is attending to, what missions connect current activity. Rendering edges/pulses on the landscape is more faithful than per-agent status lists. |
| Anti-gamification guard | `agent/environment-over-optimization` + `aif/no-self-certification` | The map creates an environment agents can inspect; it must not become the target they optimize. No behavior may count as better merely because the map itself says so; verdicts still need external mission outcomes and evidence the display did not manufacture. |
| VERIFY shape | `system-coherence/turn-design-into-checks` + `futon-theory/mission-lifecycle` | ARGUE should hand VERIFY checkable claims: read-only, canonical surface, approximate provenance, no hidden active missions, no map-driven scheduler writes, and no collective-perception claim before behavior changes. |

### Theoretical coherence

IDENTIFY said this is an integration mission, not a fresh build: register
agents against the EFE landscape and show the WM's attention on the same
surface. DERIVE follows that exactly. It joins existing clocks, invoke
jobs, wm-ticks, cursor lineage, and coordinates; the only new mechanism
is the read-side aggregator needed to make those existing signals
co-visible.

The ants analogy also becomes precise without becoming glib. Ant
pheromones matter because they are field-borne signals that affect later
movement. This mission does not yet close that loop. It displays the
field candidates: agent telemetry and WM attention. Calling them
"collective perception" would be premature; rendering them as structured
observations is the correct precondition for a later loop that can be
verified.

The frontier-placement choice is the critical argument. MAP found that
the liveliest current missions are absent from the June-12 coordinate
set. Therefore hidden-until-re-embed would make the map systematically
blind to live work, while fake exact placement would violate the
landscape-shift discipline. Approximate out-of-sample placement is the
only design that keeps the map live and honest: visible now, labelled as
approximate, and retired by explicit re-anchoring.

The BGE caution from `M-efe-bge-followon-actions` is also preserved. The
2D BGE-MDS map is a useful display lens but a poor ground metric; DERIVE
therefore uses native high-dimensional cosine only to choose neighbours,
then projects the weighted result into the existing 2D surface. This is
not a workaround around the weak 2D embedding. It is the standard
out-of-sample move with the distortion named and contained.

### Trade-offs

- **Observation before control:** we give up immediate pheromone
  deposit/decay and per-agent controller behavior. In return, the first
  implementation has almost no blast radius and produces the evidence
  needed to design those loops later.
- **Approximate now, exact later:** we give up the purity of only showing
  embedded missions. In return, active frontier work appears while its
  uncertainty remains inspectable.
- **One aggregator endpoint:** we give up bespoke UI-specific feeds. In
  return, Emacs, web, and later agents can consume one canonical contract.
- **High-dimensional kNN, 2D display:** we give up reading precise
  semantic adjacency from the visible map. In return, neighbour selection
  uses the metric where it actually has signal, while the display stays
  compatible with the existing field.
- **Inspection surface, not reward channel:** we give up the tempting
  "agents chase badges/stars" game. In return, the map remains a
  diagnostic surface rather than a new self-certifying objective.

### Generalization

The design generalizes to any stale-but-useful map over a live substrate:
keep the stable base layer, add read-only live overlays, place new items
by provenance-bearing approximate extension, and visibly distinguish
approximate from embedded facts. This applies to mission maps,
capability-star maps, proof landscapes, and later pattern/cascade maps.
The general rule is: **do not block live telemetry on full recomputation,
and do not let cheap approximation masquerade as ground truth.**

### Plain-language argument

Joe's original image was right: the EFE landscape can become a live
working map, not just a static picture. The little ship can show where
Joe is working; the other agents can appear as moving dots at the
missions they are clocked to; and the War Machine's current attention can
show up as pulses on the same terrain. That makes the surface start to
resemble the old ants simulator in the important way: many actors moving
over one field, with the field showing where useful work seems to be.

But the first honest version is still an inspection surface, not a mind
or a game. "Stars," "badges," and "food" are display language for
mission state and free-energy reduction; they are not the thing agents
should optimize. Likewise, dots on a map are only telemetry. They become
collective perception only after the displayed field demonstrably changes
how agents choose work. Until that loop exists, the right claim is more
modest and stronger: Joe, the agents, and the War Machine can finally be
seen on the same live landscape.

The map also has to stay honest about its own age. The existing embedding
is valuable, but tonight's active missions are newer than it, so hiding
them would make the liveliest work invisible. Pretending they have exact
embedded positions would be just as bad. The derived design therefore
places new missions near related known missions, marks those positions as
approximate, and later replaces them with exact positions when the
landscape is re-embedded. The frontier remains visible as frontier.

So the argument is: make the map live by joining what already exists,
make it safe by keeping the first version read-only, and make it truthful
by labelling approximate placement instead of laundering it as fact. That
is the smallest version that preserves the HEAD: Joe's ship, live agents,
War Machine attention, and the ants-style field all appear, while the
mission refuses the glib leap from "we can see the field" to "the field
already controls behavior."

**ARGUE exit:** ready for Joe review. The design is forced by the
constraints: live work must be visible, approximate placement must be
labelled, the first landing must be read-only, and any later pheromone or
controller loop must be verified as behavior-changing rather than claimed
by analogy.

*Ratified by Joe over emacs-repl, 2026-07-04: D-frontier (approximate
placement, never hidden), D-method (relational centroid → native BGE kNN
→ text BGE → frontier shelf), C3 (live approximation + slow explicit
re-anchor), C4/C5 (inspection-only).*

## VERIFY (2026-07-04, claude-18 — design claims run against live systems)

Method: every checkable claim ARGUE handed over was exercised against
the running stack (ports 7070/7071, live stores, real files), not
re-read from the doc. `pgrep java` untouched; all reads bounded
(`&limit`, timeouts).

### Checks

- **V1 — coordinate table.** `futon6/data/mission-carpet-pos-embed.json`
  loads: 212 missions, x ∈ [200, 3178], y ∈ [200, 3400]. The frontier
  gap reproduces exactly: M-futon1b-port, M-custom-harness, and
  M-live-efe-map itself all absent. **PASS** (MAP finding confirmed).
- **V2 — wm-tick join (C5).** `GET /api/alpha/evidence?author=war-machine`
  (port 7070) returns 18 entries, tags `wm-tick`/`wm-click`/`wm-cron`,
  bodies carrying `:enacted :target :G :expected-G :realized-G :gates
  :mode :decision :trigger`. Distinct enacted/target missions =
  {M-bayesian-structure-learning, M-first-flights} — **both mapped**;
  the WM-attention layer joins to coordinates with zero placement
  machinery needed today. **PASS.**
- **V3 — native BGE kNN (D-method step 2).**
  `futon3a/resources/notions/bge_mission_embeddings.json` exists: 229
  records, 1024-dim vectors, plus `cross_refs` per record (a bonus
  relational-anchor source). **But it is June-12 vintage too** — none of
  the frontier missions have vectors. **PASS with finding F1.**
- **V4 — relational centroid (D-method step 1) dry-run.**
  M-custom-harness's doc references resolve to two mapped anchors
  (M-kangaroo, M-agency-hardening); the D2 formula yields [1437.6,
  447.8] — in-range, sane. M-futon1b-port's dominant reference (12×) is
  M-custom-harness, itself unmapped: the `:anchor-depth 2+` rule is
  exercised by the very first real frontier pair, not a hypothetical.
  **PASS.**
- **V5 — agent join (C1).** `clock/clocked-on` hyperedges live on 7071
  (`?type=clock/clocked-on&limit=10`): endpoints agent↔target,
  `:hx/props` carry `:mission-id`, `:session-id`, and a witness record
  (rule, source file, edit-count) — provenance for free. **PASS with
  finding F2.**
- **V6 — WM live endpoints.** `/api/alpha/war-machine` responds
  (scheduler running; snapshot warming after the 19:29 restart —
  `retry-after` honored, structure as expected incl. r14-gamma
  verdicts); `/api/alpha/aif-stack/live` returns the full frame set.
  **PASS.**

### Findings for INSTANTIATE

- **F1 — BGE staleness tracks embed staleness.** The mission-embedding
  file is regenerated with the map, so D-method step 2 can never fire
  for exactly the missions that need placement most. Relational
  centroid (step 1) is the workhorse; step 2 is a re-anchor-tier
  benefit, not a live-tier one. The re-anchor tier (C3) should
  regenerate `bge_mission_embeddings.json` alongside the coordinate
  set. Text-kNN (step 3) requires a live BGE encoder call — only worth
  wiring if step 1 proves too weak in practice.
- **F2 — clock targets include excursions.** Live `clock/clocked-on`
  edges point at excursions (e.g. `E-repl-continuations`) as well as
  missions. The endpoint must handle non-mission targets: v1 rule =
  render on the frontier shelf labelled as excursion (they have no
  mission coordinates by construction); do not silently drop them.
- **F3 — evidence read path.** The wm-tick read is
  `GET /api/alpha/evidence` on **7070** (futon3c transport); 7071 has
  only the POST route, and `…:7071/evidence-viewer/index.html` currently
  404s (asset-dir not mounted in the running instance — noted for Joe,
  not a blocker for this mission since the 7070 read path works).

### Verdict

The DERIVE design survives contact with the live substrate. Nothing in
the argument broke; the two stress points VERIFY was most suspicious of
(frontier placement with no BGE vectors, anchor-depth chains) both
occur in tonight's real data and are both handled by rules DERIVE
already wrote. INSTANTIATE can proceed: the `GET /api/alpha/live-efe-map`
aggregator (D4 shape, plus the F2 excursion branch) and the overlay on
the embed page (D5 render contract).

**VERIFY exit:** complete. Awaiting Joe's go for INSTANTIATE.

### VERIFY addendum — C3 cadence resolved; one-off re-embed executed (2026-07-04, claude-18)

**Operator decision (Joe, emacs-repl):** re-embed ≈daily if the map is
in use; run a one-off now so the rest of the build works against
up-to-date data. This resolves the C3 cadence question (DERIVE's
"session-end or slower" → **≈daily**).

**One-off re-embed executed, full chain (the same chain a daily job
would run):**

1. `clojure -M -m futon.missions index` (futon3a) — mission_records
   refreshed from substrate-2 `code/v05/mission-doc` hyperedges:
   274 → **278 records**, all three frontier missions present with
   summaries. (Why the Jul-3 nightly missed them: the docs were
   ingested by the commit watcher on Jul-4 — pipeline fine, timing.)
2. `embed_text.py --json --model BAAI/bge-large-en-v1.5` —
   `bge_mission_embeddings.json` regenerated (278 × 1024-dim; prior
   file backed up as `.bak-20260612`). Sanity: unchanged missions
   reproduce their old vectors at cosine ≈ 1.000 (median); the 0.81
   tail is missions whose docs changed — earned shift only.
3. `mission_carpet_variants.py` — coordinate family regenerated:
   `mission-carpet-pos-embed.json` **212 → 248 missions** (243 BGE-placed,
   5 citation-placed; MDS stress 723.9).
4. `mission_efe_field.py embed` — field HTML re-rendered.

**Post-conditions verified live:** `GET /api/alpha/live-efe-map` serves
the 248-mission coordinate set without restart (it re-reads the JSON);
M-custom-harness lands at [2038, 2583] adjacent to its strongest
relational anchor M-kangaroo [2173, 2738] — the semantic placement
agrees with the D2 dry-run's relational prediction; the frontier shelf
drains to exactly two residents, both excursions (E-repl-continuations,
E-monster-to-joey) — the F2 rule rendering non-mission clock targets
honestly. `pgrep java` untouched throughout (I-0).

**For INSTANTIATE:** the ≈daily job = steps 1–4 above verbatim (step 1
already runs nightly at 04:30 via the notions index; steps 2–4 need
adding). Every rendered point's `:source-coordinate-set` timestamp
requirement (D3) is now exercised by real data two generations apart.

### VERIFY addendum 2 — T3/C2 resolved: 🛸 sessions, 🚀 centroid (2026-07-04)

**Operator decision (Joe, emacs-repl):** "Let's use 🛸 to indicate
positions of the *sessions* (e.g. the current session would be clocked
in on M-live-efe-map). As for rocket, how [about] positioning it at the
*average* of all active sessions?"

This **supersedes C2's original shape** (cursor → file→mission grain,
opt-in streaming) and dissolves T3 rather than answering it: there is
no cursor stream at all, hence no privacy grain to choose and no
opt-in machinery to build. The sessions ARE the 🛸 layer (the agent
dots, already live); the operator's 🚀 renders at the unweighted
centroid of *active* sessions — the operator's position is derived
from the work being driven, which is the operator-not-sovereign
principle made literal geometry.

**Implemented same session (claude-18, commit 9cd5005, futon3c):** the
endpoint returns a top-level `:ship` — active = status invoking/idle
(alive this server epoch, not merely restored) with an embedded or
approximate placement; synthetic frontier-shelf coordinates excluded;
no live contributors → `:ship nil` (honest absence, no placeholder).
Verified live: ship at [2196.9, 2712.9], the exact midpoint of the two
active sessions (claude-16 on M-custom-harness, zai-9 on
M-futon1b-port). Overlay glyph semantics (🛸 per session, 🚀 at
`:ship`, centroid never computed client-side) amended into the codex-2
INSTANTIATE handoff.

## INSTANTIATE (2026-07-04, late evening — complete)

**Build (codex-2, belled with inline spec + ship amendment):**
- `ad80ccb` (futon6) — live overlay in the *generator*
  (`mission_efe_field.py`, so re-embeds carry it): 10s polling of
  `/api/alpha/live-efe-map`, placement-honesty rendering (solid vs
  dashed ring vs labelled frontier band), WM attention pulses annotated
  with G, graceful "live layer offline" badge; plus
  `daily_reembed.sh` (atomic mv, record-parity check).
- `66b5ce7` (futon6) — the ship amendment: 🛸 glyph per session
  (invoking bright/animated, idle dim), 🚀 at the endpoint's `:ship`
  with "operator centroid of N active sessions" tooltip; centroid never
  computed client-side.

**Review (claude-18, the gate):** read both diffs in full; re-ran every
claimed PASS. Chromium against the live endpoint: badge "live layer
on · 8 agents · 18 WM", 8 🛸 sessions at their clocked districts, 1 🚀
at the centroid, frontier band holding the two excursions;
`daily_reembed.sh` executed end-to-end (exit 0, 278 records, parity
held). Header language checked against C4: "WM attention and agent
telemetry on the EFE landscape" — no pheromone/collective-perception
claims anywhere in the UI. Findings fixed directly (`bf04096`):
drawWarMachine deduped to the latest tick per mission+slot (18 ticks
had stacked 31 pulses on 2 missions — visual dishonesty about
attention breadth); `mission_efe_scope_dump.py` added to
`daily_reembed.sh` (without it new missions get coordinates but no
districts and never render — the F-finding from the static-render
request, now closed structurally).

**C3 armed:** crontab `30 5 * * *` → `daily_reembed.sh` (after the
04:30 notions index), logging to `futon2/logs/daily-reembed.log`.

**Completion criteria:** C1 ✓ (live agents on the real map, one JVM,
`pgrep java` untouched all session) · C2 ✓ in its superseded 🛸/🚀
form · C3 ✓ (≈daily, armed) · C4 ✓ (reviewed at every phase; the map
optimizes nothing) · C5 ✓ (WM attention live, deduped, G-annotated).
Claim discipline holds: this is WM attention and agent telemetry on
the EFE landscape — the collective-perception claim stays unclaimed
until a later mission demonstrably closes the behavior loop.

**Mission state: built and live. Close gate: Joe.**

## Post-INSTANTIATE operator decisions (2026-07-05)

**Live attestation (C3 deepened):** `pattern-attestation.json` was a frozen one-off
(June 8, ad-hoc `bb futon0.report.pattern-density` dump); now refreshed daily from
the live evidence store (`refresh_pattern_attestation.sh`, 60-day rolling window,
pinned to localhost — the laptop's `FUTON3C_EVIDENCE_BASE` default points at lucy's
thin store, see E-evidence-flow). Road ink is now RELATIVE to the current economy's
strongest road. First fresh dump inverted the economy: recorded-handoff 0→63,
idempotent-handoff 0→44; baseline-cyber-ant 100→7.

**Geometry cadence = WEEKLY (Joe, 2026-07-05):** attestation feeds the layout
springs, so fresh attestation can move districts. Decision: punctuated weekly
re-solve ("continental drift") — spatial memory holds between drifts, each drift is
itself informative. Cron: Sunday 05:00 `mission_carpet.py` (full re-solve) before
the 05:30 daily render. **INTERIM: runs on the laptop.** Target: run the weekly
re-embed + re-solve on lucy so the map doesn't depend on the laptop being open —
requires at-least-weekly `futon-sync` (exists: `futon0/scripts/futon-sync.clj`,
manifest-driven) plus unresolved dependencies: BGE model + venv on lucy,
`mission_records.json` provenance, and WHICH substrate-2/evidence store the scope
dump and attestation count against (lucy's stores are currently thin and disjoint
from the laptop's — E-evidence-flow Q1–Q5 gate this move).

**Dispatch self-clocking:** `agency_send.py --mission` added (the server side
existed since D1/O3); dispatched agents now clock on job completion and appear as
🛸 without manual intervention.

**Coordination-edge overlay (inventory row, "optional edge overlay"):** dispatched
to codex-1 2026-07-05 (bell b465807b, itself carrying `--mission M-live-efe-map`):
endpoint `:coordination` key from the mesh-edge ledger + fading cyan threads
between placed saucers. Rendering fade only — deposit/decay semantics stay out.

**T1 chartered as follow-on:** `M-pheromone-field.md` (same directory) — deposit /
decay / perception-response semantics, claim gate inherited (the name "pheromone"
must be earned by demonstrated behavioral response). T2 remains p4ng-successor
territory.
