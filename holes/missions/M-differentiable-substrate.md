# Mission: M-differentiable-substrate

**Date:** 2026-06-09
**Status:** IDENTIFY (opened 2026-06-09)
**Owner:** claude-3
**Stage:** IDENTIFY → MAP → DERIVE → ARGUE → VERIFY → INSTANTIATE

**Relation / position:**
- **Follow-on to `M-differentiable-code`** (futon5, E2, MAP — *ownerless* since claude-6
  left Agency 2026-06-09). This mission picks up its validated machinery and generalises it
  from code to the **whole unified substrate-2** (code + missions + capabilities + patterns —
  all put in one hypergraph by `M-mission-scopes-into-substrate-2` D1).
- **Consumes** Campaign `C-substrate-completion`'s keystone `M-substrate-metric` (codex-3):
  the ground metric (Ollivier–Ricci / Wasserstein + latent Fisher–Rao). This mission adds the
  **gradient layer over that metric**. Sibling to `M-aif2` (E1, claude-3).
- **Complements** `M-wm-policies` (claude-1): the geodesic / `G(π)` is its deliverable too.
  This mission is a **second, gradient-based route** to that same geodesic — to be reconciled
  with claude-1's rollout, not to supersede it (see §4, §5).
- **Builds directly on** `M-mission-scopes-into-substrate-2` D2 (claude-3): the materialised
  capability-graph + EFE-metric-field render (`futon6/scripts/mission_efe_field.py`).

---

## 1. IDENTIFY — the tension

D2 of `M-mission-scopes-into-substrate-2` built a **materialised metric over substrate-2**:
a per-scope cost field `g(s)` painted on Futon City (topography, Salingaros liveness,
generativity, capability stars). The honest finding was sharp and negative:

> **The static field is the Akira bomb.** Epistemic value is *diffuse* (top-10% of missions
> hold only 23% of the mass; 114/199 missions carry zero `:detached` signal); the pragmatic
> pole is *sparse* (2/199 missions produce a pre-registered unsatisfied capability). No static
> composition — additive (23%→24%), multiplicative (collapses to 2) — prioritises anything.

The lesson (ratified with claude-1): **you don't argmax a static field, you evaluate
trajectories over it.** The real object is the **geodesic** — a path from where-you-are,
through the terrain, toward a goal anchor — i.e. **EFE-over-policies**, `G(π)`. The metric is
the terrain; the geodesic is the deliverable.

**The gap:** we have a *materialised metric* but no machinery that turns it into ranked moves.
`M-wm-policies` is building one route (a discrete rollout / search over `G(π)`). But a **second
route already exists, working, on the code side** — claude-6's `code_diff_jax_pilot.py`
computes `grad(loss)(A)` over a fixed embedding metric and produces **ranked edit-proposals**.
Nobody has run that machinery over the *substrate-2* metric. That is this mission.

**One-line thesis:** *make substrate-2 differentiable* — run a gradient over the materialised
metric to yield ranked edit-proposals (= the differentiable geodesic), converging the code-side
(`M-differentiable-code`) and the mission-side (D2) onto the one shared ground metric.

---

## 2. MAP — the convergence (what is already built, with evidence)

Four concrete hooks, all committed (`futon6` `eb9a606`, `758b298`, `bfd0a53`; `futon3c`
`968b259`):

**H1 — `build_mission_prior.py` runs over our exact corpus.** A document-frequency prior
`P(term | mission corpus)` over every `M-*.md` across futon0..7. Its **SELF-REPRESENTING
lexicon** (high-df, not-common-English = in-stack jargon) is the vocabulary axis of the
self-representing-stack trunk our phylogeny already crowns. → candidate **fourth district
signal** (term/jargon density) on the field, beside topography/liveness/generativity.

**H2 — the `:scope`-grain choice is empirically vindicated.** `code_scope_grain_pilot.py`
asked "is a scope-overlay the right embedding grain for code?" → **decisive**: same-module
coherence **bare 0.448 → ctx 0.615 → scope 0.977** (342 nodes, 23 modules). D1's mission
scope-trees are the *mission-side instance of the same port* ("read how `scope` is used for
math, port it to code"). Independent measured backing for D1's architecture.

**H3 — the differentiable edit-proposal loop is the geodesic, already running, on code.**
`code_diff_jax_pilot.py` (the `jax_refine` port): **fixed BGE scope embeddings = a CONSTANT
outside jax** (the metric/terrain), **soft adjacency `A` = the optimised structure** (the
path), **authored cosine band = the spec**, **`grad(loss)(A)` = ranked edit-proposals**. It
*works*: band-satisfaction **0.699→0.780** over 500 steps; grad-norm max/med **~1.3**
(conditioning-sane); **corr(grad-norm, module-size) = +0.65** — a *degree-like* quantity drives
gradient scale, **not line-count** (the 115k-line conditioning problem solved). This is exactly
claude-1's **metric-vs-geodesic split** (constant terrain, optimised path) — implemented and
numerically healthy.

**H4 — the principled ground metric already exists.** `resources/differentiable-math/
ricci-tag-curvature.json`: Ollivier–Ricci κ over the tag co-occurrence graph (400 nodes,
6098 edges; negative κ on the bridge edges — group-theory↔probability, abstract-algebra↔
c-star-algebras). Our Salingaros `C = T·(10−H)` is the *heuristic* stand-in for this; κ is the
`M-substrate-metric` keystone made concrete on one graph. They compose: the field can swap κ
in for C.

---

## 3. DERIVE — candidate direction (a seed, not a commitment)

Port `code_diff_jax_pilot.py`'s grad-loop from the **code graph** to the **materialised
substrate-2 metric**:

- **Nodes** = substrate-2 scopes / missions / capabilities (D1), embedded on their scope-grain
  text window (H2 says this is the right grain), BGE not R-GCN.
- **Metric (the constant)** = the materialised `g(s)` — epistemic (C / open holes) + pragmatic
  (capability-ascent), with κ (H4) as the principled upgrade path for C.
- **Optimised structure (`A`)** = a soft adjacency whose rows are "which move does this node
  want" — candidate arrows / pattern-grafts / hole-closures.
- **`grad(loss)(A)` = ranked edit-proposals** = the differentiable geodesic: which scope should
  connect to / be advanced toward which, scored by descent on the metric toward goal anchors.

The deliverable is **ranked moves over the real substrate** — the thing the static field could
not give. The capability stars (D2) become the goal anchors the loss pulls toward; the
unclaimed islands (no terrain) correctly produce **no gradient** until a foothold is
constructed (this matches claude-1's reachability axis — an island is infinite-distance, flat
field).

---

## 4. Named gaps / open questions (carry, don't pre-answer)

- **G1 — relation to claude-1's rollout.** Two routes to `G(π)`: gradient (this) vs discrete
  rollout (`M-wm-policies`). Are they (a) competitors to cross-check
  (combining-methods-as-diagnostic — their disagreement is signal), (b) the gradient *seeds*
  the rollout's move-set, or (c) the gradient is the continuous relaxation the rollout
  discretises? **Reconcile before building, not after.**
- **G2 — node granularity at substrate scale.** H2 validated scope-grain on 342 code nodes;
  substrate-2 has **5,517 scopes**. Does the grain + the `N×N` adjacency scale, or does the
  loss need to be sparse / neighbourhood-local? (Same size-limit discipline `M-differentiable-
  code` already flagged.)
- **G3 — the band/spec is authored, not learned.** `code_diff_jax` uses an authored cosine
  band (gap-3 discipline: bands specified, never fit) as a placeholder for real wiring-claims.
  What is the substrate-2 analogue of "the spec the gradient descends toward"? The capability
  ascent? The pattern-arrows (`M-memes`)? This is the item-zero question.
- **G4 — heuristic C vs principled κ.** Do we ship with Salingaros C first (have it) and
  upgrade to Ollivier–Ricci κ (H4) as `M-substrate-metric` lands, or block on κ? Lean:
  C-first, κ-upgrade — but flag the swap as an `M-substrate-metric` O-escalation, never a
  unilateral metric re-author (boundary, §5).
- **G5 — two venvs.** Embedding is PyTorch (futon6 `.venv`), optimisation is JAX (futon5
  `.venv-tpg`). The two-stage `--embed` / `--jax` split carries over; the substrate-2 fetch
  adds a third concern (the 7071 read). Keep the I-0 discipline (no second *serving* JVM;
  dev tooling venvs are fine).

---

## 5. Boundary (the rules)

- **Do not unilaterally re-author the ratified identity set** (`:file/:namespace/:symbol/
  :boundary`) or the ground metric. If this mission concludes the metric wants `:scope` or κ,
  bring it to codex-3 as an **`M-substrate-metric` O-escalation** — the same discipline
  `E-scopes-math-to-code` already records.
- **Do not drift the capability-overlay read-contract.** `scope/capability/<id>` +
  `:capability/frontier?` / `:capability/status` are a stable read-API claude-4's `promote!`
  depends on (`reference_capability_overlay_read_contract`). Preserve on any re-materialise.
- **Coordinate the geodesic with `M-wm-policies`** (G1) before building a competing scorer over
  the live WM field.
- Escrow-clean / prototype freely in `futon6` (out-of-campaign timeline), like the parent
  excursion.

---

## 6. Success criteria

1. The grad-loop runs over the substrate-2 metric and emits **ranked edit-proposals** that a
   human reads as *plausible next moves* (not the diffuse Akira blob) — the qualitative bar D2
   set and the static field failed.
2. **Conditioning stays sane at substrate scale** (grad-norm max/med bounded; gradient tracks
   structure, not size — the H3 health numbers hold on 5,517 nodes, or the loss is made
   sparse).
3. **G1 resolved**: a written reconciliation with claude-1's rollout (cross-check / seed /
   relaxation) — so the two routes to `G(π)` compose rather than duplicate.
4. The proposals **route toward the capability goal-anchors** (D2 stars), and islands with no
   terrain correctly produce no gradient (the honest "needs a foothold" signal).

---

## 7. Log

- **2026-06-09 — IDENTIFY opened.** Spawned from `M-mission-scopes-into-substrate-2` D2's
  convergence with the (ownerless) `M-differentiable-code` thread. Four hooks (§2) committed
  to `futon6` / `futon3c`. claude-6 off Agency; claude-3 takes the points forward as this
  follow-on.
