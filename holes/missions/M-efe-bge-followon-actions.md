# M-efe-bge-followon-actions

Date: 2026-06-12
Status: HANDOFF-READY. Exploration by the claude owner-session; reviewed by fable-2 (whistle,
2026-06-12). Offered to and scoped for **fable-2** to execute. This single document is the
hand-off: it says what we did, what we learned, what is *not yet proven*, and the bundle to run.

## One-line

We rebuilt mission/pattern embeddings on BGE and asked how the embedding should drive the
`mission-efe-field` carpet. The answer is a **family of competing layouts, not one** — and the
family already shows that *semantic similarity* and *citation/pattern coupling* are
**near-orthogonal** metrics over the missions. That disagreement is the finding. This doc carries
the diagnostics needed to trust (or reject) the embed layout, and the honest disagreement-field
work that follows.

## Background / what prompted this

- BGE-large-en-v1.5 (1024-d) now co-embeds missions + patterns (1302 vectors;
  `futon3a/resources/notions/bge_{mission,pattern}_embeddings.json`), replacing MiniLM.
  (Separate finding: Hamming retired as a metric for the iiching codebook — orthogonal to this.)
- The `mission-efe-field` carpet was laid out **purely** by a force-directed citation +
  shared-pattern graph — *no embedding at all*. "Redo with the new embedding" is therefore not a
  swap; it *introduces* the semantic signal as a layout driver.
- **Joe's organizing frame:** the carpet springs, the BGE embedding, and the EFE terrain are
  competing **projections of a latent mission-metric we don't yet know how to define**. Don't pick
  one — generate the family and read the (dis)agreement. (memory:
  feedback_projections_converge_on_metric; ties to project_substrate_metric.)

## What we built

`futon6/scripts/mission_carpet_variants.py` emits four layouts in one coordinate frame
(`mission-carpet-pos-{force,embed,springs,seed}.json`), rendered by
`mission_efe_field.py <variant>` → `mission-efe-field-{variant}.html`:

- **force** — citation + pattern-road springs (baseline; ≈ the current carpet)
- **embed** — BGE cosine → metric **MDS** ("the embedding IS the map")
- **springs** — force layout + BGE-cosine springs (top-6 neighbours, cos > 0.45)
- **seed** — BGE-seeded, then graph-relaxed

## What we learned so far

1. **force vs embed pairwise-distance agreement is Spearman ρ = 0.088 — near-orthogonal.**
   Citation/pattern *coupling* and BGE *semantic affinity* measure genuinely different things over
   the missions. Full matrix (ρ over 195 embedded missions):

   ```
              force     embed   springs      seed
      force  +1.000    +0.088    +0.433    +0.532
      embed  +0.088    +1.000    +0.217    +0.266
    springs  +0.433    +0.217    +1.000    +0.536
       seed  +0.532    +0.266    +0.536    +1.000
   ```

2. **The embed layout *looks* ~3× richer** (1820 vs ~580 contour segments) and Joe confirmed it is
   visibly much richer than the baseline — **BUT THIS IS UNPROVEN.** See (a): MDS of cosine
   distances among high-dimensional-sphere vectors can produce a uniformly-spread disc, which yields
   more contour segments *for free*, independent of any real semantic structure. Do not report
   "embed is richer" as a result until the shuffle control below settles it.

3. **springs/seed land closer to the graph than the embedding** (ρ ≈ 0.43–0.54 with force vs
   ≈ 0.22–0.27 with embed). A naive blend lets the dominant graph springs win — the "compromise" is
   really a vote for one side, not an average of the two metrics.

## fable-2's review — the corrections and the plan

### (a) Prove or disprove the embed "richness" before trusting it

- The reported MDS stress **464 is RAW** sklearn stress (`normalized_stress="auto"` silently
  disables normalization for metric MDS) → uninterpretable. Compute **Kruskal stress-1** =
  `sqrt(stress / Σ dᵢⱼ²)`; > ~0.15–0.20 means the 2D map is known-poor.
- **Decisive — shuffle control:** permute the embedding↔mission assignment, re-run MDS, count
  contour segments. If shuffled embeddings *also* give ~1800 segments, the richness is a property of
  MDS-on-cosine matrices, **not** of the missions.
- Also: **k-NN trustworthiness/continuity** (k ≈ 10; terrain is local, so this matters more than
  global ρ); **restart stability** (vary the seed, measure Procrustes distance between layouts —
  real structure survives, artifact reshuffles); **intrinsic dimensionality** (classical-MDS
  eigenspectrum — if 10+ dims are needed, *any* 2D map is mostly distortion and only cluster
  membership, not adjacency, should be read off it).

### (b) Keep it bimetric — do not merge

- Citation/pattern graph = *realized coupling* (where work actually flowed — trade routes); BGE =
  *topical affinity* (what is about the same thing — geography). ρ = 0.088 means these are
  near-independent; a single merged ground metric would **erase** that fact.
- Recommendation for the substrate-metric thread (project_substrate_metric): make the ground
  structure **bimetric** — both distances first-class per pair, and the **disagreement** a derived
  first-class field. Find any latent metric by **external arbitration, not weighted averaging**.
- Same anti-pooling discipline as the cascade work's F8 (outputs of different generators never
  silently pool).

### (c) The disagreement field — done honestly

- Raw force→embed displacement vectors are **meaningless without Procrustes alignment first** (MDS
  output carries arbitrary rotation/reflection/scale; the current script has no alignment).
- Weight each mission's displacement by **local projection quality** (per-point stress /
  trustworthiness) so the field separates "the metrics genuinely disagree here" from "the 2D
  projection is just bad here."
- Deliverable: a ranked list of **unexploited adjacencies** — semantically-near / graph-far
  missions that *should* be talking and never have.
- t-SNE: acceptable as a *lens*, wrong as a *base layer* (it destroys global distance and must never
  feed the terrain metric). Prefer UMAP if adding one, kept visually distinct from the carpet.

## The bundle to execute (= fable-2's offered scope)

1. **Diagnostics:** Kruskal stress-1 + shuffle control + k-NN trustworthiness + restart-Procrustes +
   intrinsic-dimensionality → a verdict on whether the embed richness is real or artifact.
2. **Align:** Procrustes-align all variants to a common frame; compute per-mission projection
   quality.
3. **Disagreement field:** over the aligned frame, quality-weighted → the ranked
   unexploited-adjacency list.
4. **Bimetric substrate proposal:** both distances + the disagreement as first-class fields, written
   up against project_substrate_metric.

## Acceptance test (the empirical arbiter)

`O-cross-mission-unlocking` (pudding-prover registry): *"does a capability flip on one mission open
reachable :have's on another?"* The top of the unexploited-adjacency list (semantically-near /
graph-far pairs) is exactly where emergent cross-mission unlocking should appear **if** the BGE
metric carries causal signal. **Pass condition:** the disagreement-field ranking predicts
cross-mission unlocking better than chance and better than the graph metric alone. This arbitrates
the two metrics empirically rather than aesthetically.

## Artifacts

- `futon6/scripts/mission_carpet_variants.py` — family generator + cross-method agreement matrix.
- `futon6/scripts/mission_efe_field.py` — now takes a variant arg (`force|embed|springs|seed`).
- `futon6/data/mission-efe-field-{force,embed,springs,seed}.html` — the four rendered fields.
- `futon6/data/mission-carpet-pos-{variant}.json` — the four position sets (common frame).
- `futon3a/resources/notions/bge_mission_embeddings.json` — 1024-d BGE, `basename`-keyed (clean
  `M-…` stems; 195/212 carpet missions have a vector, 17 are citation-placed).

## Open decisions for Joe

- Adopt the **bimetric** framing into the substrate-metric thread (project_substrate_metric)?
- Wire **O-cross-mission-unlocking** as the standing acceptance test for "does the semantic metric
  carry causal signal"?

## Step 1 RESULTS — diagnostics verdict (fable-2, 2026-06-12; ALL MEASURED)

Run: `futon6/scripts/mission_embed_diagnostics.py` (controls rendered through the real
`mission_efe_field.py <variant>` pipeline; canonical `mission-efe-field.html` untouched).

**VERDICT: the embed layout's "3× richer terrain" is an MDS-on-cosine ARTIFACT, not semantic
signal.** Random gaussian unit vectors (1024-d, same mission count), pushed through the identical
MDS → placement → terrain pipeline, produce **more** contour segments than the real BGE embedding:

| layout | contour segs |
|---|---|
| force (baseline, re-measured) | 602 |
| embed (real BGE, re-measured) | 1820 |
| ctrl-shuf 1–3 (labels permuted) | 2008 / 2360 / 2480 |
| ctrl-rand 1–3 (random vectors) | **3230 / 2928 / 2952** |

Contour count is a property of MDS-of-cosine point spread, not of the missions. The real
embedding actually yields *less* terrain than noise — genuine cluster structure concentrates
points, while near-equidistant noise spreads into the uniform disc that maximizes level-set
complexity.

Supporting measurements (same run):

- **Kruskal stress-1 = 0.381** (the reported 464 was raw sklearn stress) — far above the
  ~0.15–0.20 "known-poor" threshold. The 2D map preserves little of the 1024-d metric.
- **Intrinsic dimensionality:** classical-MDS eigenspectrum needs **12 / 33 / 47 dims** for
  50/80/90% of variance; the 2-D share is **14.2%**. Any 2D map of this space is mostly distortion.
- **Restart instability:** Procrustes disparity 0.43–0.72 vs the seed-7 layout across seeds 1–5 —
  the specific positions reshuffle substantially run-to-run.
- **Local signal survives:** trustworthiness 0.768 / continuity 0.824 (k=10) — ~77% of 2D
  neighbours are true semantic neighbours. Cluster *membership* is partially readable; adjacency
  and terrain are not.
- **The high-d metric itself has real content:** BGE cosine-distance cv = 0.150 vs 0.031 for
  random vectors — 5× the relative spread. The signal is in the 1024-d metric, not its 2D shadow.

**Consequence for steps 2–3 (methodological correction):** the unexploited-adjacency RANKING must
be computed in the **native metrics** (high-d BGE cosine vs graph distance), not from 2D
displacement — the 2D projection is too lossy and too unstable to carry it. The Procrustes-aligned
2D disagreement field remains useful as a *visualization lens* (quality-weighted per (c)), but the
deliverable ranking should never pass through the projection.

Artifacts added: `scripts/mission_embed_diagnostics.py`;
`data/mission-carpet-pos-ctrl-{shuf,rand}{1,2,3}.json` + `data/mission-efe-field-ctrl-*.html`
(gitignored data/, regenerable).

## Provenance

claude owner-session built the family + agreement matrix; fable-2 supplied the (a)/(b)/(c) review
that corrected the premature "embed is richer" claim and the unaligned disagreement-field sketch.
This doc is the synthesis handed back to fable-2 to execute.
