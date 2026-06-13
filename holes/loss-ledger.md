# Loss ledger — DP fleet coverage trajectory

Appended every loop tick by `scripts/log_loss.py` (claude-1). Read top-to-bottom
for the story: grounding % should rise, well-formedness errors should fall to 0,
debt is mostly the dominant ungrounded-symbol count (and irreducible definition
holes). See `holes/dp-fleet-plan.md` for the capability targets.

| time (local) | papers | grounded | wf-errors | debt | note |
|---|---|---|---|---|---|
| 2026-06-13T16:46:40 | 32 | 52% | 1 | 47176 | baseline at handoff: weft wf-clean + grounding/quantifier/defined-in-paper/refs online; warp W1+W3 done, W2-linkage + corpus-DEBT in flight; per-capability refactor in flight |
| 2026-06-13T16:54:19 | 32 | 52% | 1 | 47176 | refactor landed (behavior-preserving, gate-verified); non-symbol classifier lifted 0809.2517 78%->86%; warp DEBT report + W2 committed; +2 discovered items (DEBT concept-filter, W2 re-run) |
| 2026-06-13T17:22:11 | 32 | 53% | 1 | 46928 | round 2 landed: Galois alias (DEBT->Lean), sub/superscript grounding, DEBT concept-filter, W2 linkage re-run; all self-committed on split modules |
| 2026-06-13T17:52:39 | 62 | 57% | 838 | 95629 | round 3: display-:= grounding, wf->0 (stale-golden cleared), memory-safe batch runner, corpus scaled 32->62 papers, DEBT summary written |
| 2026-06-13T18:32:29 | 231 | 57% | 859 | 324767 | round 4: corpus scaled 62->227; residue analysis (tail NOT irreducible); citation-DEBT bridge (0 internal resolutions) |
| 2026-06-13T20:16:38 | 261 | 70% | 0 | 220054 | FULL REGEN by claude-1 (agents stalled on it 3x): appositive + noise-context + current wf applied corpus-wide |
| 2026-06-13T20:51:24 | 261 | 70% | 0 | 219924 | round 5: def-equation/name-verb grounding (+0.5pp); residue re-analysis -> NEAR ACHIEVABLE CEILING ~70%; warp DEBT refreshed |
