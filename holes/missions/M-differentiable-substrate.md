# Mission: M-differentiable-substrate

**Date:** 2026-06-09
**Status:** INSTANTIATE — scope-grain v2 producer live + wired to claude-4's rollout (see Checkpoint 1)
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

## Checkpoint 1 — the gradient policy-prior is live, useful, and wired (2026-06-09)

**The big step.** This mission went, in one session, from IDENTIFY (the static-field "Akira
bomb" — a metric that can't prioritise) to a **working, *useful* gradient policy-prior** that
emits real ranked moves over substrate-2 and is consumed by claude-4's discrete rollout. The
AlphaZero split is fully realised on the gradient side: prior (this) + search (claude-4).

**Built + working:**
- **Producer** (`scripts/diffsub_emit.py`, two-stage BGE→JAX) at **scope grain**: 5532 real
  substrate-2 scopes (verbatim ids + `anchor/passage` embeddings) + 33 capabilities, sparse-kNN
  N=5565 k=20, Option-A clean-detached metric. Conditioning sane (grad-norm max/med 1.50).
  [`801dc62` base (codex-2) + `34776c5` precursor-chain (claude-3)]
- **Locked emit interface** (§3.1): the policy-prior contract, drift-proof across three agents.
- **Reachability seam = depth:** 55 moves = **44 chained close-holes** (real scope-ids,
  precursor-chained by canonical phase — e.g. hypergraph-operator depth-5) + **3 reachable cap
  summits** + **7 intended-dark islands**. Root taxonomy **21 mission / 3 capability / 7
  conjectural / 0 drift**, verified against claude-4's by-construction handshake.
- **A *useful* prior** (`c0ee162`): the hot-swappable-metric side-by-side proved the metric was
  NOT the bottleneck (all variants → uniform prior); the real lever was the prior softmax
  temperature. z-norm + temperature took the prior **uniform → peaked** (entropy-norm 0.85, top
  move 10% mass vs 1.8% uniform). Metric (`option-a`/`sharp`/`liveness`) and temperature
  (`DIFFSUB_PRIOR_TEMP`) are now pluggable axes.

**Cross-agent state.** claude-4's consumer is built + tested (depth-5 chain PASS, 3-way root
classifier, zero-drift handshake) and waiting on the producer bytes, which are committed on
branch `diffsub/scope-grain-v2`. **End-to-end is GREEN — SEAM CLOSED** (claude-4 witness futon2
`holes/labs/e-rollout-v2-e2e.clj`, `3d41d26`): zero-drift handshake (21/3/7/0 exact), transitive
reachability 44/44 close-holes lit from the 21 seeds + 3/3 summits + 7/7 islands dark, and the
hypergraph-operator depth-5 chain unrolls live. The AlphaZero split now runs producer -> consumer
-> rollout. (claude-4's clarification: soft scores don't touch reachability/depth — structural,
all green — only *selection* among equally-reachable chains; metric-sharpening improves which
chain the search prefers, R2 is the principled fix.)

**Open (next moves).** (1) claude-4 runs the move-set end-to-end (expect 44 chained close-holes +
3 summits + 7 dark islands). (2) The rollout-side metric/temperature experiment — which prior
best guides PUCT. (3) **v3:** the Option-B `:backfill` move-class (the 290 pattern coverage-gaps)
+ κ (H4) as the principled metric upgrade. (4) **R2:** the policy-improvement return channel —
claude-4's realized `G(π)` per `:move/id` trains this prior, closing the AlphaZero loop.

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
          :prior      <float in [0,1]>       ; the POLICY HEAD P(s,a) — claude-4's R1 renormalizes THIS over reachable survivors (RATIFIED 2026-06-09; NOT recomputed from :score)
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
  prior guides search). **Framing (Joe, 2026-06-09):** the two routes are the **AlphaZero
  featureset — BOTH required, not redundant**; the prior makes the search tractable and the
  search sharpens the prior. The rank-disagreement cross-check is a *minor side-benefit*, NOT
  the reason for two routes. Ownership: claude-3 = gradient/prior, claude-4 = rollout/search,
  claude-1 = coordination. Both build in **parallel** (Joe opened the codex pool 2026-06-09).
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
- **2026-06-09 — DERIVE producer built + reviewed (PASS).** codex-2 built
  `scripts/diffsub_emit.py` (futon6 `29fe492`) — two-stage BGE→JAX, **G2 v1 grain =
  mission+capability** (230 nodes; scope-grain deferred — `/tmp/scopes.json` has no per-scope
  ids), **sparse N×k kNN** adjacency (k=20), **G4 Salingaros-C** (κ deferred). claude-3 review
  (author≠reviewer): re-ran `--jax` (deterministic — reproduces exactly); **EDN key-sets
  IDENTICAL to the stub** (claude-4's consumer unaffected, `:emit/stub? false`); **conditioning
  sane — grad-norm max/med = 1.37** (cf code_diff_jax 1.3 → criterion-2 ✓), corr(grad-norm,
  degree)=+0.18; **summit-score-med 0.27 ≫ island 0.043** (criterion-4 ✓ — reachable frontier
  caps get strong gradient, conjectural islands near-zero); satisfaction 0.58→0.88. Output
  `data/diffsub-moves.edn` (19 moves). **v2 findings (not blockers):** (a) `:graft-pattern`
  dropped — needs pattern nodes (scope/pattern-grain), absent at mission grain; (b)
  `:close-hole` `:have`/`:want` are mission-grain placeholders (`scope/<stem>/detached#…`), NOT
  real per-scope ids — they won't join claude-4's reachability until scope-grain lands; the
  capability moves carry REAL ids. Criteria 1/2/4 met (crit-3/G1 = the AlphaZero split, already
  resolved). **Next:** scope-grain v2 (real scope-ids → close-hole + graft-pattern join
  reachability); the policy-improvement return channel (R2) once claude-4's rollout reports
  realized G(π).
- **2026-06-09 — scope-grain v2 started (branch `diffsub/scope-grain-v2`).** Foundation:
  `scripts/diffsub_scope_dump.py` (`3c2af59`) dumps **5532** substrate-2 scopes with REAL ids +
  verbatim passages (`anchor/passage`) + anchor-state. **FINDING:** only **44** scopes are
  genuinely anchor-detached (the real open holes); the v1 render's 461 conflated those with ~417
  `pattern`(290)/`source-material`(112)/`relates-to`(12) links that read `:detached` only because
  the cited flexiarg/source isn't a substrate-2 endpoint — coverage-gap artifacts, not open work.
  Scope-grain **cleans the epistemic signal** (a chunk of the Akira-bomb diffuseness). **METRIC =
  Option A** (Joe): clean anchor-detached only; the coverage-gap `:backfill` move-class deferred
  to a v3 demo. Producer scope-grain extension dispatched to codex-2 (job 327) — emits real
  scope-ids on `:close-hole` so they join claude-4's reachability (the v1 gap). claude-3 reviews
  on bell.
- **2026-06-09 — scope-grain v2 producer DONE (claude-3 took the baton on codex-2's Agency
  timeout).** codex-2 completed the scope-grain build (`--grain scope`: 5532 real scopes + BGE
  passage embeddings, sparse-knn N=5565 k=20, all 44 detached as `:close-hole` with REAL verbatim
  scope-ids) but hit the 30-min timeout before commit; I committed its base (`801dc62`) + added
  the **precursor-chain** (`34776c5`). Chain: eightfold-phase holes chain by canonical order
  (mission → derive → argue → verify → document → instantiate), so closing earlier phases unlocks
  later = the rollout's search depth. Root taxonomy verified against claude-4's handshake: **21
  mission / 3 capability / 7 conjectural / 0 drift**; 9 chain-links; hypergraph-operator depth-5.
  Handed to claude-4 (job 335) for end-to-end. Conditioning max/med 1.50 (sane). **Open:** the
  gradient landscape is FLAT at scope-grain (sat 0.585→0.599, 44 detached in 5565) — structure
  strong, `:score`/`:prior` soft; metric-sharpening (class/frontier weighting) is the follow-up
  if the rollout leans on prior magnitudes.
- **2026-06-09 — hot-swappable metric + the side-by-side payoff (Joe: "don't fix blind").** Added a
  metric registry (`option-a`/`sharp`/`liveness`, `--metric`, tagged in `:emit/metric :C-variant`)
  + a `--experiment` harness (same structure, different scores; variant files
  `diffsub-moves-<name>.edn`). The experiment **immediately redirected the fix**: all three metric
  variants gave a **uniform** prior (entropy-norm 1.000, prior-max = 1/55) despite real score
  signal (score-cv 0.36, summit≫island 6×) — so the metric was NOT the bottleneck. The real lever
  was the **prior softmax over small-magnitude gradient scores** flattening the ranking. Fix:
  z-normalize scores + temperature (`DIFFSUB_PRIOR_TEMP`, default 2.0) → prior **uniform →
  peaked** (entropy-norm 0.85, top move 10% mass vs 1.8% uniform; meaningful top-5). Handshake
  unchanged (21/3/7). The metric stays a pluggable axis for future experiments; temperature is the
  immediate tunable claude-4's PUCT can dial. Lesson: the experiment caught a wrong-knob fix.
- **2026-06-09 — END-TO-END GREEN, seam CLOSED.** claude-4 ran the full rollout against the
  landed producer (`diffsub-moves.edn`, 55 moves); witness futon2 `holes/labs/e-rollout-v2-e2e.clj`
  (`3d41d26`). Zero-drift handshake exact (21 mission / 3 capability / 7 conjectural / 0 drift);
  transitive reachability 44/44 close-holes lit from the 21 seeds, 3/3 cap summits reachable, 7/7
  islands dark; live best-rollout depth-5 (hypergraph-operator mission→derive→argue→verify→
  document→instantiate). v1's 3-reachable → 44 chained + 3 + 7. **The AlphaZero split runs
  producer→consumer→rollout — the mission's central thesis is demonstrated end-to-end.**
  claude-4 clarified the soft-score impact: it does NOT affect reachability/depth (structural,
  all green), only *selection among equally-reachable chains* — so metric-sharpening (the `sharp`
  variant / temperature) improves *which* chain the search prefers, and R2 (realized G(π) per
  move-id → gradient training) is the principled fix; v1 forward-only as agreed.

---

## 8. R2 — the policy-improvement loop (v2 design, drafted 2026-06-09)

**Thesis.** Close the AlphaZero loop. v1 is forward-only (prior → search). R2 makes it
bidirectional: claude-4's search reports what it *concluded* per move, and this route trains its
prior to predict that — so the prior improves from experience, the soft-score problem dissolves
into *learning* (not hand-tuning), and `reward = peradam` finally does work. The change in kind:
a fixed heuristic becomes a learning system.

### 8.1 The return contract (claude-4 → claude-3) — the gating co-design

Per rollout-batch, claude-4 emits the search's per-move statistics (NOT just the path outcome —
credit assignment must be honest):

```clojure
{:return/at        <ts>
 :return/from-gen  <prior-generation searched with>   ; staleness guard
 :return/rollouts  <int>                                ; batch size
 :moves [{:move/id  "<have>-><want>"
          :visits   <int>     ; N(s,a) = the SEARCH'S IMPROVED POLICY π  (the training target)
          :q        <float>   ; Q(s,a) = mean realised value of paths through it (value signal)
          :selected <int>}    ; times on the best-rollout path
         ...]}
```
The **visit-count `:visits`** is load-bearing — it IS the search's improved policy (AlphaZero's
target). `:q`/`:selected` are value signals (for a later value head).

### 8.2 The training mechanism (this route)

A **Bayesian blend of a cold-start inductive prior + a learned refinement**:
- **Cold-start** = the current gradient-over-metric (H3). Zero-data inductive bias.
  `g(s) = w·features(s)`, features = [det, frontier, class-weight, log(gen), log(degree)]; w is the
  hand-set variant coefficients today.
- **Learned** = fit those same 5 coefficients w so the resulting prior matches claude-4's
  visit-count policy π — **policy-distillation** (cross-entropy of my prior vs the search's π over
  the shared move-set). **The metric experiment harness becomes the training harness**: the
  side-by-side *enumerated* w-variants; R2 *optimises* w by gradient on the distillation loss.
  Same parameterisation, learned not guessed.
- **Blend (the posterior)** = `prior = (1−α)·cold + α·learned`, α grows with accumulated rollout
  evidence. Gradient-prior = prior-over-weights; search outcomes = likelihood; blend = posterior
  (the Bayesian-structure-learning reliability update, instantiated).

Start linear-in-features (interpretable, few-shot-robust — rollouts are expensive, signal sparse).
A richer learned head is a later capacity upgrade once data accumulates.

### 8.3 The clock + versioning

Batched: after B rollouts, claude-4 emits the aggregated return → refit w → re-emit
`diffsub-moves.edn` tagged `:emit/gen <n>` → claude-4 searches the new prior for batch n+1. Each
return carries `:return/from-gen` so stale returns are down-weighted. Loop tick = a batch.

### 8.4 Exploration preservation (don't collapse the search)

- Keep α < 1 (cold-start gradient-prior always contributes) — preserves inductive-bias exploration.
- Temperature floor on the learned prior (don't over-peak).
- claude-4's PUCT keeps its own exploration term + value head — the search explores regardless.

### 8.5 Open design questions

- **R2-Q1 (the gate):** does claude-4's rollout expose per-move `N(s,a)`/`Q`, or only best-path
  `G(π)`? Co-design FIRST — visit-counts are the honest credit-assignment target.
- **R2-Q2:** the α schedule (how fast to trust learned weights vs the inductive prior).
- **R2-Q3:** cross-mission generalisation — features are generic (det/frontier/class/gen), so
  learned w *should* transfer; verify (else the loop is per-mission, narrower).
- **R2-Q4:** sparse-signal robustness — favour the 5-weight model + strong prior until data earns
  more capacity.
- **R2-Q5 (`reward = peradam`):** is `:q` the realised peradam (the typed-witness fruit a *closed*
  hole emits)? Ties R2 to the pudding-prover peradam + sorry-arrow contracts.

### 8.6 Why R2 over v3

v3 (`:backfill` + κ) gives the static prior *better inputs* — incremental, solo. R2 makes the
apparatus *improve itself* — the change in kind overnight-autonomy needs. The experiment harness +
the reserved `:move/id` join key mean the hooks already exist. R2 next; v3 a clean solo follow-up.
- **2026-06-09 — R1 contract evolution RATIFIED (`:prior` is the policy head).** claude-4
  (re-checking my sharpening at my nudge) found its R1 was recomputing `:prior` as
  `softmax(:score)` — and since `:score` is flat at scope-grain (spread 0.0127 → uniform), it
  **silently re-flattened my sharpened policy head** before PUCT. So the z-norm+temperature work
  was correct but discarded. Fix (claude-4 `b38f8d3`): R1 now consumes the producer's `:prior`
  directly (renormalized over each node's reachable survivors), `softmax(:score)` only as
  fallback when `:prior` is absent. RATIFIED — this IS the faithful split: `:prior` = the policy
  head (my gradient), `:score`/`:delta-g` = the value. Verified effect: at t=0 over 39 reachable
  survivors prior max-mass 2.56% (uniform) → 12.9% (~5× peak). **My sharpening now drives claude-4's
  branching**, and the two pluggable axes (`:C-variant` {option-a/sharp/liveness}, `DIFFSUB_PRIOR_TEMP`)
  are live for its PUCT A/B — unblocking the rollout-side metric/temperature experiment.
