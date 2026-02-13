# Problem 6: Track Belief Summary from Local MathOverflow Data

Date: 2026-02-13
Primary evidence file: `data/first-proof/problem6-mo-track-evidence.md`

## Method

- Corpus scanned: `se-data/mathoverflow.net/Posts.xml`.
- Questions scanned: 152,893.
- Each track used an independent keyword bundle.
- "Anchored strong" means at least 2 track-keyword hits plus graph/matrix/spectral context.

## Anchored-Strong Signal (higher is better)

| Track | Anchored strong hits | Signal |
|---|---:|---|
| E (expansion / conductance / spectral geometry) | 60 | strongest |
| C (interlacing / KS / mixed characteristic) | 8 | moderate |
| A (SR / determinantal / negative dependence) | 6 | moderate-low |
| F (BSS-like potential + matrix concentration) | 4 | moderate-low |
| B (hyperbolic barrier) | 1 | weak |
| D (near-rank-1 reformulation) | 0 | very weak |

## Practical interpretation for strategy confidence

1. Track E has clear external precedent on MO (many graph-spectral threads around Cheeger/expansion/conductance).
2. Tracks C and A have meaningful but niche support; they are plausible secondary tracks.
3. Track F has limited direct precedent but enough related threads (BSS lineage and matrix concentration) to keep as a backup.
4. Tracks B and D currently look weak from MO precedent alone; they need either new theory or strong internal empirical breakthroughs to justify top priority.

## Suggested priority (belief-driven)

1. E (primary)
2. C (secondary)
3. A (secondary)
4. F (backup)
5. B (exploratory only)
6. D (de-prioritize unless new evidence appears)

## Caveat

This is a precedent scan, not a correctness proof. It measures how much related machinery appears in MathOverflow discussions, not whether the track closes GPL-H.
