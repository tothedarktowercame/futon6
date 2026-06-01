# Mission: Differentiable Mathematics — a ground-metric pilot on real math embeddings

**Date:** 2026-05-31
**Status:** IDENTIFY → pilot; claude-2 owns end-to-end (speculative-sequel scout)
**Checkpoint (2026-05-31):** both metric halves demonstrated on real math data —
continuity (BGE NN geometry) + curvature (Ollivier–Ricci marks bottlenecks,
hypothesis CONFIRMED). Committed. Next: BGE-grounded κ recompute (join the halves).
**Relation:** speculative sequel to `futon5 M-differentiable-code` (the code swarm:
claude-3, claude-6, codex-3). We work in **their future timeline** — they are
building the embeddings M-differentiable-code needs; we already HAVE the
equivalent for mathematics, so we can pilot the ideas faster and *show the kind
of thing that becomes possible* once the embeddings exist.
**Campaign:** informs (does NOT join) `futon3c C-substrate-completion`'s keystone
`M-substrate-metric`. A scout/demonstrator, not an escrow member.

## 1. Why this exists

`C-substrate-completion` charters a **ground metric** on substrate-2 with a
verified contract of derived objects: **curvature** (Ollivier–Ricci over
Wasserstein), **continuity** (a smooth embedding for gradients), **information
geometry** (Fisher–Rao latent). Its #1 shared blocker (CONSTITUTION §122) is
**node-granularity + a continuous embedding** — the thing M-differentiable-code
must build before it can take a single gradient.

**Mathematics already has that, built.** The Feb-2026 superpod run left, on this
laptop (verified 2026-05-31, shapes via mmap):

| artifact | shape / size | what it is |
|---|---|---|
| `storage/math-processed-gpu/embeddings.npy` | **(805200, 1024) float32** | BGE (`bge-large-en-v1.5`) **textual** embeddings, 805k math.SE entities |
| `storage/math-processed-gpu/hypergraph-embeddings.npy` | **(1068183, 128) float32** | **structural** embeddings, 1.07M hypergraph nodes (R-GCN — see caveat) |
| `storage/math-processed-gpu/structural-similarity-index.faiss` | 522M | queryable structural index |
| `storage/math-processed-gpu/relations.json` | 158M | the typed graph the metric is defined ON |
| `storage/math-processed-gpu/hypergraphs.json` | 17G | full hypergraph (slice via CT subset, don't load whole) |
| `storage/math-processed-gpu/thread-wiring-ct.json` | 9.3G | CT-slice wiring |

So the pilot can build the **actual campaign deliverable** — a ground metric with
curvature + continuity — on real math data NOW, and feed the design lessons back
to `M-substrate-metric` before the code swarm reaches the same wall.

## 2. The pilot (what we demonstrate)

The metric contract, on math embeddings, scoped to a tractable slice:
1. **Continuity:** the BGE (805200,1024) array IS the continuous embedding —
   distance `d(x,y)=‖emb(x)−emb(y)‖` (or cosine) is already smooth and queryable.
   Demonstrate sane nearest-neighbour geometry on a slice.
2. **Curvature:** Ollivier–Ricci `κ(x,y)=1−W₁(μ_x,μ_y)/d(x,y)` over the hypergraph
   `relations.json`, with `d` from (1). Negative κ on bridges/bottlenecks =
   "ameliorate here" = the principled tension signal aif2 wants. Demonstrate κ
   marks meaningful structure (a bottleneck concept between two dense clusters).
3. **Feed back:** whatever we learn about node-granularity, curvature behaviour,
   and the textual-vs-structural blend goes to `M-substrate-metric` as evidence
   for the STANDARD-ARGUE/VERIFY contract — the "kind of thing that's possible."

## 3. Caveats named up front (verified this session — do NOT trip these)

1. **The structural (1068183,128) array is R-GCN output.** This session
   established (codex-2-verified, `futon6/technote-arxiv-mining.md`) that futon6's
   R-GCN structural embeddings **collapsed to cosine ~1.0**; BGE replaced them for
   retrieval. So build the **continuity/metric on the (805200,1024) BGE array**;
   treat the 128-d structural array as the *cautionary* input, not the trusted
   one. (Same "measurement instrument not learned objective" warning as
   C-substrate-completion §119 and [[project_differentiable_code]] gap #3.)
2. **Two node universes, not row-aligned:** 805,200 textual entities vs 1,068,183
   hypergraph nodes. Joining them needs the `*-ids.json` / `hypergraph-thread-ids.json`
   maps — verify the alignment, never assume it.
3. **Scale:** 3.1G + 17G. Work on the **CT slice** via mmap; never load whole.

## 4. First concrete probe

Cheapest real demonstration that the continuous metric exists and is sane:
mmap the BGE array, take a small slice, compute nearest-neighbours for a known
math entity, and check the neighbours are semantically coherent. If yes, the
continuity half of the metric is demonstrably real on math data — then build up
to a small curvature computation over a relations.json subgraph.

## First probe RESULT (measured 2026-05-31)

Ran the continuity demonstration on the real BGE array (`embeddings.npy`,
mmap'd, first 20,000 entities; row i ↔ entity i, alignment verified against
`entities.json` which is a JSON array of multi-line objects — a *streaming*
decoder is required, naive per-line parse fails). Cosine nearest-neighbours:

- "Why is $1$ not a prime number?" → top-5 all primality questions (0.74–0.79)
- "If eigenvalues are positive, is the matrix positive definite?" → top-5 all
  positive-definite-matrix questions (0.81–0.87)
- a Lebesgue-measurable-function question → measurable/continuous-limit
  neighbours (0.77–0.81)
- the Fibonacci question → Binet's formula, nth term, computing digits (0.77–0.82)

**Finding:** the **continuity half of the campaign's ground metric already
exists, on real math data** — `d(x,y)=‖emb(x)−emb(y)‖` over the BGE array is
smooth, queryable, and semantically coherent with ZERO building. This is exactly
the continuous embedding M-differentiable-code's swarm is still constructing; the
math timeline has it now. Geometry over 20k vectors (dim 1024) runs in ~0.2s.
Next: scale the slice / add the FAISS index, then the curvature half
(Ollivier–Ricci over a `relations.json` subgraph) — the harder, more novel piece.

## Curvature probe RESULT — hypothesis CONFIRMED (measured 2026-05-31)

**Does negative Ollivier–Ricci κ mark meaningful bottleneck concepts? YES.**
Built the tag co-occurrence graph from `relations.json` (1,985,936 `tagged-with`
edges over 805,200 questions; tags adjacent iff co-tagging ≥30 questions, top 400
tags), computed κ=1−W₁(μ_x,μ_y)/d with exact earth-mover per edge via scipy LP.
Script: `scripts/ricci_bottleneck_pilot.py`; output
`resources/differentiable-math/ricci-tag-curvature.json` (6,098 edges, 176s).

**Most NEGATIVE κ = genuine cross-area bridges** (the bottleneck signal):
group-theory↔probability (−0.274), cardinals↔linear-algebra (−0.270),
abstract-algebra↔calculus (−0.233), algebraic-topology↔combinatorics (−0.240),
computability↔number-theory (−0.230). Each connects two otherwise-distant
subfields — exactly "ameliorate here" connector concepts.

**Most POSITIVE κ = tight intra-community pairs:** c-star-algebras↔operator-algebras
(+0.586), field-theory↔galois-theory (+0.532), linear-algebra↔matrices (+0.530),
lie-algebras↔lie-groups (+0.503) — near-synonym pairs inside one area.

**Subtle confirmation the metric is doing real work:** `c-star-algebras` is
*negative* with ideals/abstract-algebra but *positive* with operator-algebras —
correctly capturing that the same node is tightly bound to one neighbour while
bridging into another. That role-distinction is precisely what aif2's
tension-proposer wants (propose at the bridge, not the dense core).

**Campaign import:** both halves of `C-substrate-completion`'s ground-metric
contract are now demonstrated on real data — **continuity** (BGE NN geometry) and
**curvature** (Ollivier–Ricci marks bottlenecks). The keystone `M-substrate-metric`
can cite this as evidence the contract is achievable, before the code swarm has
embeddings. Caveat: this used a hop-distance ground metric (d=1 adjacent / 2 else)
as a cheap proxy; the full contract wants d from the continuous embedding — the
natural next step is to recompute κ with BGE-distance as the ground metric and
check the bottleneck ranking survives.

## INCIDENT 2026-05-31 — BGE-grounded recompute OOM'd the box; BACKED OFF

The next probe (BGE-distance as the ground metric, tag centroids from
`embeddings.npy`) was attempted and **OOM'd the machine, pressuring the serving
futon3c JVM into swap.** Root cause: the script `f.read()` the full 2.3G
`entities.json` into RAM (regex over the whole file) + held an 805k qid→row dict
+ mmap'd the 3.1G array + q2tags, on a box already running a ~13Gi JVM. The
runaway was killed; the JVM survived and recovered (HTTP 200 on :7070 after); the
offending script was deleted unrun-from-git (never committed).

**This violated the mission's OWN caveat #3 (slice, never load whole) and
[[feedback_no_synchronous_heavy_drawbridge_calls]] (don't run heavy compute that
can starve the shared serving JVM).** Decision (Joe): **back off** the BGE-grounded
recompute on this machine.

**If/when resumed, hard constraints:** (a) NEVER read entities.json whole — stream
the qid→row map and discard as you go, or precompute it once to a small sidecar;
(b) build tag centroids incrementally over mmap row-slices, never materialise the
full 3.1G array; (c) run on a machine that is NOT hosting the futon3c JVM, or with
a hard memory cap (ulimit/cgroup); (d) ideally hand this to the superpod, not the
laptop. The hop-distance curvature result (committed, `192c120`) already
demonstrates the bottleneck signal — the BGE-grounded version is a refinement, not
a prerequisite, so there is no urgency that justifies risking the box.

## Relations

- `futon5 M-differentiable-code` — the present-timeline sibling; we scout its future.
- `futon3c C-substrate-completion` / `M-substrate-metric` — the keystone this informs.
- [[project_differentiable_code]], [[project_prior_mathematics]] — sibling futon6/5 work.
- `futon6/src/futon6/graph_embed.py` — how the structural embeddings were built (R-GCN).
