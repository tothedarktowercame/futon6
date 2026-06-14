# E-drainage-basin-policy-landscape — paleo-topography first

**Excursion (bounded, single-owner). Spun out 2026-06-14. Owner: a Claude
agent. Bell claude-1 back with results + shas.**

## Goal
Build the drainage-basin policy landscape per `holes/policy-landscape-drainage.md`
— **paleo-topography first** (reconstruct the underlying terrain/history before
overlaying current flow). Read that hole as the charter; this note just scopes
it as a bounded handoff.

## Scope / steps
1. Read `holes/policy-landscape-drainage.md` fully; extract its intended model
   (basins = clusters, ridges = divides, flow = accumulation, paleo-topography =
   the historical/underlying structure laid down first).
2. Build the paleo-topography layer first, then the drainage/flow on top.
3. **Cross-link to the CT work (Joe's flag, for LATER not now):** the same
   basin/flow apparatus may apply to our **citation graph**
   (`data/warp/citations.json`, 30k edges) and **concept-usage graph** (the
   concordance, `data/warp/concordance.json`) — basins = concept/citation
   clusters, flow = citation/usage accumulation, paleo-topography = the
   historical layering of the literature. Note the interface; do NOT build it
   this excursion (keep bounded) — just record how the method would attach.

## Acceptance
Paleo-topography layer built; drainage/flow on top; a short section on the
citation-/concept-graph application interface (design only). Commit artifacts +
this note.

## RESULTS (claude-3, 2026-06-14) — DONE

Built, all three traps discharged:
- **Paleo-topography FIRST** — `scripts/paleo_topography.py` -> `data/paleo-topography.json`.
  81 basins (mined cascades), each terrain commit-pinned to its git DRAINAGE
  commit (closing commit of the mission source), ordered by drainage time = the
  stratigraphy (2026-02-18 -> 2026-06-12). Terrain = OR-curvature reused from
  `futon3c/.../substrate_metric_cascade_adapter` (commit-pinned snapshot, not live
  HEAD — trap 3). 81/81 drainage-dated, 43 with a realized pattern-cite route,
  verified-discharge mean 0.78 (the retrodictive "it drained" witness).
- **Drainage / flow ON TOP** — `scripts/drainage_flow.py` -> `data/drainage-flow.json`.
  Reads the paleo terrain (never HEAD). Per basin: discharge (verified artifacts,
  1103 total), reachable counterfactual sibling routes (pattern-overlap AND drained
  at-or-before — trap 3 at route grain), and a CALIBRATION FLAG (trap 1):
  22 metric-concordant, 21 metric-prefers-alt, 0 metric-disconfirmed-by-drainage,
  38 unroutable. **Never a :better-route label**; only :metric-disconfirmed-by-
  drainage carries a learning signal and it points AT THE METRIC; CH1 self-
  reference guard documented in-code. Negative class (trap 2) REPORTED (136 dry
  basins) not modelled — bounded.
- **Interface sketch (design only)** — `holes/drainage-citation-interface.md`:
  how the same basin/flow/paleo apparatus attaches to `data/warp/citations.json`
  (basins=citation communities, paleo-pin=arXiv id-date — temporal signal is free)
  and `data/warp/concordance.json` (concept clusters, paleo-pin=first-use). A
  four-method interface contract makes the later pass a thin adapter. NOT built.

Artifacts (`data/*.json`) are gitignored fleet-wide -> committed the SCRIPTS +
the two design docs + this note; the JSON regenerates from the scripts.
Shas in the checkpoint bell to claude-1.

## Constraints
Never restart the futon3c JVM. Co-Authored-By: Claude Fable 5
<noreply@anthropic.com>. Bell claude-1 back with {what was built, the
citation-graph interface sketch} + shas.
