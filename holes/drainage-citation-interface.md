# Drainage-basin apparatus → citation & concept-usage graphs (interface sketch)

**Design only — NOT built this excursion (E-drainage-basin-policy-landscape, kept
bounded). Owner: claude-3. Records how the same basin/flow/paleo-topography method
attaches to the literature graphs so a later pass can wire it.**

The policy-landscape layers (`scripts/paleo_topography.py`, `scripts/drainage_flow.py`)
are deliberately generic in shape: a **basin** is a graph with a *realized route*,
an *outcome*, and a *drainage time*; **terrain** is a metric over that graph;
**paleo-topography** pins the terrain to the drainage time from a temporal signal.
Only the temporal signal and the "outcome" differ between domains. So the same
three layers transpose onto our two literature graphs with one substitution each.

## The transposition

| layer | mission cascades (built) | citation graph | concept-usage graph |
|---|---|---|---|
| **basin** | one mined cascade (a closed typed hole) | a citation community / a paper's incoming-citation neighborhood | a concept cluster (terms co-occurring across papers) |
| **realized route** | ordered pattern-cite chain | the actual citation path an idea propagated along | the actual sequence of papers that used the concept |
| **flow / discharge** | verified closing artifacts | citation accumulation (in-degree grown over time) | usage accumulation (downstream invocations of the term) |
| **terrain (metric)** | Ollivier-Ricci curvature over the cascade graph | OR-curvature over the citation subgraph ("what is near/steep" in idea-space) | OR-curvature over the term co-usage graph |
| **paleo-topography pin** | git closing-commit date of the mission | **arXiv id encodes the date** (`YYMM.nnnnn`) — pin terrain by paper date | concept **first-use date** (earliest paper using the term) |
| **outcome ("it drained")** | retrodictive artifact verification | the citation actually happened (the edge exists) | the concept was actually reused downstream |

## Data already in hand

- `data/warp/citations.json` — `edges: [{from, to, via}]` (+ `cited_by` inverse,
  `stats`). `from`/`to` are arXiv ids, so **the temporal signal is free**: the
  paper date is in the id (`0704.1378` → 2007-04). No git reconstruction needed —
  the literature is already commit-pinned by arXiv. Paleo-stratigraphy = sort by
  id-date, exactly as `paleo_topography.py` sorts basins by drainage time.
- `data/warp/concordance.json` — `schema: warp-concordance-v1`, `terms: {...}`,
  `generated_at`. A term's earliest-paper date is its first-use stratum.

## What a later pass would do (NOT now)

1. **Basin extraction**: community-detect the citation graph (e.g. label-propagation
   over `edges`) → each community is a basin; or take per-paper incoming neighborhoods.
2. **Paleo-pin**: for each basin, the terrain is the OR-curvature over the
   sub-graph *as of the basin's epoch* (papers up to the basin's median date) —
   reuse `substrate_metric_e1_curvature` exactly as `paleo_topography.py` reuses
   the cascade adapter; the only new code is the citation→edges projection
   (a sibling of `substrate_metric_cascade_adapter.project_cascade`).
3. **Flow**: in-degree accumulation per node over the stratigraphy = discharge.
4. **Calibration flags, unchanged**: the *realized* citation path has an outcome
   (it happened); counterfactual paths (reachable alternatives in the same epoch)
   have only metric scores. Emit `:metric-disconfirmed-by-drainage` when the
   metric scored the realized path as a bottleneck yet the citation propagated,
   and `:metric-prefers-alt` (review flag, never a label) otherwise. The **CH1
   self-reference guard carries over verbatim** — a metric-preferred uncited path
   is not "a citation that should have happened."

## The one honest caveat (carried from trap 1)

For the citation graph the "outcome" is *survivorship-biased in the same way the
cascades are* (trap 2): we see citations that happened, not the papers that
*should* have cited and didn't. The negative class — plausible-but-absent
citations — is the literature analogue of the dry-basin corpus, and like it must
be mined separately before any of these flags become training signal rather than
calibration. Build the positive layers first; flag, do not label; mine the
negative class before learning.

## Interface contract (so the later pass is a thin adapter)

A domain plugs in by providing four things; everything else (paleo ordering, flow,
flag classification, the CH1 guard) is reused:
```
basin_id            -> str
graph_edges(basin)  -> [{a, b, relation}]        # for the curvature metric
realized_route(basin) -> [node, ...]             # the path that "drained"
drainage_time(basin)  -> iso8601                  # arXiv id-date | concept first-use
outcome(basin)        -> {verified, total}        # the retrodictive "it drained"
```
The mission-cascade adapter is the reference implementation of this contract.
