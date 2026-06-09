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

### 3.1 The emit interface (ratified 2026-06-09 — the contract claude-4's rollout consumes)

This route's output is a **ranked candidate move-set** in EDN. A *move* is a candidate BHK-arrow
(keyed by `(have, want)` endpoints, aligning with claude-4's arrow store). The rollout consumes
the top-k as its branching set and uses `:prior` as the PUCT branching weight.

```clojure
{:emit/at      <unix-ts>
 :emit/metric  {:compose :additive :epistemic :C-holes :pragmatic :cap-ascent :C-variant :salingaros}
 :emit/k       <int>
 :moves [{:move/id    "<have>->. <want>"     ; stable key — dedupe, cross-check, return-channel
          :move/class :close-hole | :graft-pattern | :advance-capability | :centre-mess
          :have       "scope/<id>"           ; arrow tail (current state node)
          :want       "scope/<id>" | "scope/capability/<id>"   ; arrow head (target)
          :advances-cap "<cap-id>" | nil     ; -> promote! GET scope/capability/<id>, route on :frontier?
          :score      <float>                ; raw first-order gradient magnitude (mass-gain)
          :prior      <float in [0,1]>       ; softmax(:score) over the set = PUCT prior P(s,a)
          :delta-g    <float>                ; first-order predicted metric descent (vs path-integral)
          :confidence :claimed-substrate | :conjectural   ; summit/real vs island/seeded-:open arrow
          :rank       <int>}                 ; first-order rank — cross-checked against the path rank
         ...]}
```

Field rationale: `:have/:want` = arrow endpoints (claude-4's keying + sorry-arrow
unify-on-promotion); `:advances-cap` = the capability-overlay read-contract hook (do not drift
`:capability/frontier?`/`:status`); `:prior` = a **distribution**, not just a top-k cut (the
AlphaZero policy head); `:delta-g` = the first-order number the rollout's `G(π)` is cross-checked
against; `:confidence` = the island/summit trust split (a conjectural move is *proposed* terrain
— the rollout discounts it); `:move/class` = which transition `T` applies (close a `:detached`
hole / graft an arrow / flip capability-status via promote!). **Return channel (reserved):** the
stable `:move/id` lets the rollout report realized `G(π)` per move back as the training target for
this route's loss — closing the AlphaZero loop (`reward = peradam` enters here).

**Two claude-4 constraints (ratified 2026-06-09), both already compatible:**
- **Sim-on-copy / no `:7071` writes during search.** This route's emit is a **static data
  artifact** — a snapshot of scored moves + `:emit/metric` version, consumed *once*. The rollout
  sims entirely on its copy of the cap-overlay with zero live dependency on this route or on
  7071 during search. The return channel (R2) is strictly *post*-search (training), never
  mid-search.
- **Reachability is the ROLLOUT's gate, not mine.** claude-4 consumes a *reachable* move-set
  (open arrows whose `:have` is reached by some `:constructed` arrow). So this route emits
  `:prior` over a **broad candidate set** (every move it can score off the metric, including
  not-yet-reachable ones — flagged `:confidence :conjectural`, the islands); the **rollout
  intersects with its currently-reachable set per sim-node and renormalizes `:prior` over the
  survivors.** Emitting the superset is deliberate: constructing an arrow mid-rollout opens new
  reachable `:have`s, and the prior must already cover them — so the prior is a *function over
  the candidate space*, the rollout supplies the moving reachable mask.

---

## 4. Named gaps / open questions (carry, don't pre-answer)

- **G1 — relation to claude-1's rollout. ✅ RESOLVED 2026-06-09 (the AlphaZero split;
  M-wm-policies §3, futon2 `b473b33`; coordinated by claude-1).** The two routes **COMPOSE,
  not compete** (G1 options b+c *both*): this gradient route is the **policy prior** —
  fast / global / first-order, it ranks candidate MOVES (which single edits descend the
  metric toward the goal-anchors); claude-4's discrete rollout is the **search** — it consumes
  this route's top-k proposed moves as its **branching set** and evaluates actual PATHS
  (path-integral `G(π)` over sequences, the combinatorial structure a first-order gradient
  can't see). `value = G(π)`, `reward = peradam`. **Interface:** this route EMITS ranked
  candidate moves; the rollout CONSUMES the top-k (shape in §3.1). Where the first-order rank
  disagrees with the path-integral rank = the **cross-check diagnostic** *and* the
  **policy-improvement training signal** (the full AlphaZero loop — search improves the prior,
  prior guides search). Ownership: claude-3 = gradient/prior, claude-4 = rollout/search,
  claude-1 = coordination.
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
- **G6 — `:centre-mess` has no transition T (deferred to M-memes).** Of the four move-classes,
  three are atomic arrows with a clean forward-model step (claude-4's shared kernel):
  `:close-hole` → promote `:open→:constructed`; `:advance-capability` → promote + cap-flip on
  `:capability/frontier?`; `:graft-pattern` → mint a new `(have,want)` arrow (which opens new
  reachable `:have`s — the superset-prior reason). `:centre-mess` is **not** atomic — it's a
  compound graph-rewrite over a cluster (raise coherence H, lower Salingaros C), whose mechanism
  (pattern→wiring→structure) isn't built yet (M-memes territory). **v1 resolution (ratified with
  claude-4 2026-06-09):** keep it as a *visible candidate* carrying its g-cost but mark it
  **`:move/terminal? true`** — the rollout does not expand through it. Defining a toy T would
  fabricate dynamics the simulator can't run (the regulator lesson). Promote to a real T when
  M-memes delivers the pattern→structure mechanism; that's when `:centre-mess` becomes
  expandable. (All other moves carry `:move/terminal? false`.)

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
3. **G1 resolved ✅ (2026-06-09)**: the AlphaZero split — this route is the policy prior, the
   rollout is the search, composing via the §3.1 emit interface. The remaining build-bar is that
   a real grad-loop *emits* the §3.1 shape and claude-4's rollout consumes it.
4. The proposals **route toward the capability goal-anchors** (D2 stars), and islands with no
   terrain correctly produce no gradient (the honest "needs a foothold" signal).

---

## 7. Log

- **2026-06-09 — IDENTIFY opened.** Spawned from `M-mission-scopes-into-substrate-2` D2's
  convergence with the (ownerless) `M-differentiable-code` thread. Four hooks (§2) committed
  to `futon6` / `futon3c`. claude-6 off Agency; claude-3 takes the points forward as this
  follow-on.
