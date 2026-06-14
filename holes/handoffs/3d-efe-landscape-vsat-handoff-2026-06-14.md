# Handoff — 3D EFE Landscape in VSAT (task #8)

**For:** a remote Claude (federated Agency).
**Owner / bell back to:** claude-1 (futon6 session) with a summary + commit SHAs.
**Cross-repo:** source semantics live in **futon6**; the deliverable lands in **~/vsat** (a separate repo — commit there).

## Goal
Port the 2D mission-EFE landscape into VSAT's 3D apparatus: render the per-step
metric field g(s) over Futon missions as a **navigable 3D terrain**, with missions
and capabilities placed on it — same semantics as the 2D reference, in WebGL/VR.

## Source of truth (futon6 — read, don't reimplement the semantics)
`scripts/mission_efe_field.py` is the canonical 2D renderer. Reproduce ITS meaning:
- **Field** = metric g(s) scatter-added on a grid → topographic surface. In 3D this
  becomes the **terrain height** (high cost = high/rough ground; low = basins).
- **Layout** `POS` from `data/mission-carpet-pos.json` → the (x, y) of each mission;
  height comes from the field at that point.
- **Colormap** `TERR` (7 bands, deep-blue→amber) → terrain material banding.
- **Capabilities** from `data/capability-graph.json`: claimed (minted by a mission),
  unclaimed ⭐ (registered goal, no minting mission → endpoint), and the
  projection-layer `GROUNDED_BY` anchoring (display-only; curated EDN untouched).
- **Roads** `data/mission-carpet-roads.json`; the **warm "lasso"** = recency-weighted
  activity field (optional in v1).
- Inputs also include `data/efe-scopes.json` (from `mission_efe_scope_dump.py`).

Run the 2D version first to see what you're matching: `python scripts/mission_efe_field.py`
→ `data/mission-efe-field.html`.

## Target (~/vsat — A-Frame)
- The apparatus is **VSATLATARIUM** (`/story/vsatlatarium`, A-Frame/WebGL+VR). Read
  `~/vsat/CLAUDE.md` (VSATLATARIUM section) and `~/vsat/README.md` first.
- **Reuse existing infra**, don't rebuild: stabilcam (billboard text), troika-text
  (outlined labels), the force-directed/dome layout patterns, hover rings, turntable
  pattern (rotate world, not camera).
- **Data feed**: VSAT reads JSON from `~/vsat/data/` (see `data/futon-profile.json`
  + `data/futon-profile-HANDOFF.md` for the existing futon→vsat feed convention).
  Emit an `efe-landscape.json` feed from the futon6 data above; render it scene-side.
- **Heads-up:** there's a `~/vsat/PAUSE` file — check whether the app is paused/
  deploy-gated and coordinate before assuming it's live.

## Suggested mapping (2D → 3D)
- terrain heightmap from the g(s) grid (the same scatter-add); `TERR` bands → material.
- missions = entities at (x, y, fieldHeight); label via troika-text/stabilcam.
- capability stars = distinct 3D glyphs; keep the claimed / unclaimed-⭐ / grounded
  distinction visible (color or shape).
- contours optional as iso-bands draped on the surface.

## Acceptance bar
- A navigable 3D EFE landscape in VSATLATARIUM that **faithfully reproduces the 2D
  semantics** (height = metric cost, TERR banding, mission positions = POS,
  capability claimed/unclaimed/grounded distinction).
- **Verify with Playwright** (VSAT has `playwright.config.ts`): load the scene, drive
  the camera, screenshot it — attach the screenshot to the bell-back. Don't ask Joe
  to switch tabs.

## Gates (must clear)
- VSAT's own checks: `biome` lint (`biome.json`) + the Playwright/test suite green;
  **don't break VSATLATARIUM** or existing stories.
- Honor the futon6 data contract (don't mutate the curated capability EDN — the field
  is display-only, same rule as `mission_efe_field.py`'s `GROUNDED_BY`).
- **Export-safety (fresh lesson, 2026-06-14):** bound the geometry. The 2D greatest-hits
  export once emitted ~1.6M DOM nodes and OOM'd the browser; we fixed it by AGGREGATING
  (≤6 glyphs/paper). A-Frame chokes the same way on too many entities — aggregate /
  instance / LOD; do not emit one entity per raw mark. If you need per-mark detail at
  scale, rasterise, don't DOM it.

## Coordination
- This is cross-repo: cite the futon6 source commit you matched against; commit the
  deliverable in **~/vsat**. Bell **claude-1** back with: summary, the vsat commit
  SHAs, the Playwright screenshot, and any semantics you had to interpret/deviate on.
