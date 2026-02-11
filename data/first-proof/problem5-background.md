# Problem 5 Background Note

Date: 2026-02-11

## Purpose

This note records the information sources used to revise `problem5-solution.md`.
It is a provenance summary only (no new research).

## Source Inventory

### 1) Local corpus mining (MathOverflow / Math.StackExchange dumps)

Primary local data files:
- `se-data/mathoverflow.net/Posts.xml`
- `se-data/mathoverflow.net/Users.xml`
- `se-data/math.stackexchange.com/Posts.xml`
- `se-data/math.stackexchange.com/Users.xml`

Relevant extracted hits were documented in:
- `data/first-proof/library-research-findings.md`

Most useful Problem 5 local hits:
- MO 467605 (answer on homotopical combinatorics): transfer systems, indexing systems, and `N_infty` context.
- MO 462316 (answer): transfer-system bibliography and 2024 survey pointers.
- MO 430939 (question): explicit modern `N_infty`-operad framing.
- MO 164210 + answers 164350/164363: abstract and model-level descriptions of geometric fixed points.
- MO 427280 (question): modern discussion of slice-filtration subtlety.

How these were used:
- To confirm transfer/indexing systems are the right language for “incomplete” equivariant structure.
- To confirm geometric fixed-point detection is structural and not tied to one point-set model.
- To identify that the old draft’s `rho_H^O` object was not clearly standard in this literature stream.

### 2) In-repo synthesis notes

- `data/first-proof/polya-reductions.md`
- `data/first-proof/library-research-findings.md`

How these were used:
- `polya-reductions.md` flagged the exact uncertainty: subgroup restrictions vs inventing an “O-regular representation.”
- `library-research-findings.md` provided the curated local-corpus evidence list that guided the rewrite.

### 3) Core paper references used in the tightened writeup

- Blumberg–Hill, arXiv:1309.1750  
  Role: grounding for `N_infty` operads and indexing-system/transfer-data viewpoint.

- Hill–Yarnall, arXiv:1703.10526  
  Role: regular-slice generator pattern and geometric-fixed-point connectivity characterization used as baseline template.

- Hill–Hopkins–Ravenel (slice framework)  
  Role: classical background context for regular slice filtration.

## What changed because of these sources

1. Removed nonstandard `rho_H^O`-based formulation from the Problem 5 solution.
2. Replaced with a transfer-restricted regular-slice definition (restrict subgroup indexing, keep regular-slice cell template).
3. Stated connectivity criterion as a restricted-subgroup version of the regular geometric-fixed-point test.
4. Synced this logic into:
- `data/first-proof/problem5-solution.md`
- `scripts/proof5-wiring-diagram.py`
- `data/first-proof/problem5-wiring.json`
- `data/first-proof/problem5-v1.mmd`

## Limits

- This note does not claim a full literature-complete “incomplete slice filtration” theorem survey.
- It only records sources actually used to produce the current Problem 5 revision in this repo.
