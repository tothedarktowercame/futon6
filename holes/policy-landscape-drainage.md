# The policy landscape as drainage basins (Joe, 2026-06-12 ~19:25)

Design note, end of the WM-piloted-flight day. The current EFE field
(mission-efe-field*.html) is MISSION-centric — patterns are a hidden
layer. Joe: "It would be good to have separate pictures of the POLICY
landscape. Once we have our 55 cascades: look at them as DRAINAGE
BASINS associated with the patterns that they closed. That ought to
give an idea of 'optimisation over policies' — on the view that the
selected cascade DID drain, could we have done it better some other
way?"

## The reading

- A typed hole is a depression in the landscape — an EFE minimum
  waiting for flow (hunger as potential).
- Patterns are the channel segments; the substrate metric (claude-3's
  curvature over the cascade graphs) is the TOPOGRAPHY — what is near,
  what is steep.
- A mined cascade is the REALIZED drainage network of its basin: the
  composition that actually carried the discharge from have to want.
  "It drained" is a fact (912/1320 artifacts verified — the
  retrodictive witness).
- G-over-policies, pictorially: for each drained basin, enumerate
  counterfactual drainage routes (alternative cascades reachable in
  the same topography), score both under the metric, and ask: better
  some other way? Retrodictive, zero operator cost, outcomes known.

## Why this is the right first G picture

It inverts the usual burden. Instead of asking the WM to PROPOSE
policies and waiting to learn if they were good (slow, expensive,
confounded), we hold the outcome fixed (the basin drained) and vary
the policy (the route). The 55-79 mined cascades are exactly a
training set of (basin, realized-route, outcome=drained) triples;
the contest harness's per-arm-best generalizes to per-basin
best-of-alternatives vs history. This is claude-1's circumstances-v1
("retrodictive gold") given its native visualization.

## Composition with the running lanes

- claude-3's cascade adapter gives the topography over mined graphs.
- claude-1's retrodictive test IS the drainage comparison; the picture
  is its display.
- The BV connective pass (miner v2) types the channels: ⊗ tributaries
  (parallel), ◁ mainstem (sequential), ⅋ braided channel (tight
  coupling), × distributary choice. A drainage network with typed
  reaches is a policy in the formal sense.
- Display: the embed-layout legibility win (metric-salt addendum)
  applies — basins drawn in embedding space for sense-making, scored
  in graph space for judgment.

Owner: unassigned (candidate next-pass item once adapter + v1.1 miner
land). The picture's name, for the file when it exists:
pattern-drainage-field.html.

## Three traps, folded in (critique round, 2026-06-12 evening — descending importance)

**1. Fact-versus-model: only one direction of disagreement is
informative.** The realized route has an OUTCOME; counterfactuals have
only metric scores. Metric-prefers-alternative does NOT mean history
was suboptimal — it marks a point where either the metric or the route
is wrong, and retrodiction cannot say which. So the drainage field
generates CALIBRATION FLAGS, not training labels. Guard explicitly
against feeding metric-preferred counterfactuals into the prior as if
they were better outcomes — that is CH1 self-reference re-entering
through the retrodictive door (the prior would learn the metric's
counterfactual tastes, not reality's). The honest training signal runs
the OTHER way: where the metric scores the realized route poorly and
it drained anyway, THE METRIC owes an update.
Schema consequence: comparison output carries :flag
{:metric-prefers-alt | :metric-disconfirmed-by-drainage}, never a
:better-route label.

**2. The training set is censored — drained basins only.** Learning
from (basin, route, drained) is learning from won games: it can make
successful drainage more efficient; it cannot teach which basins stay
dry or which routes fail. Mine the NEGATIVE class: abandoned cascades,
never-closed holes, moribund missions. The dry-basin corpus is already
sitting in the miner's skip list (133 skipped-not-completed) plus the
first :success false closure-fold record (its seed). Thinner data,
honest class.

**3. Non-stationarity: cascades reshape the terrain they drain.**
Closing holes flips capabilities, mints patterns, changes precursor
structure — the unlocking thesis itself. Realized routes drained PAST
terrain; scoring alternatives against CURRENT topography compares
incommensurables (some alternatives were unreachable then; some
realized moves are unrepeatable now). Required: PALEO-TOPOGRAPHY —
terrain state commit-pinned per basin at drainage time, reconstructed
from git history (the retrospective-reconstruction lane already
established commits as the temporal signal). The metaphor supports the
fix: rivers carve canyons; real hydrologists reconstruct the old
surface before judging the old channel.

Build-order consequence: trap 3 makes terrain-pinning a PREREQUISITE
of the comparison, not a refinement — without paleo-topography the
flags of trap 1 are unreadable. Order: pin terrain per basin → run
comparisons → emit flags (1) → grow the negative corpus (2) in
parallel.
