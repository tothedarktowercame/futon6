# Loss ledger — DP fleet coverage trajectory

Appended every loop tick by `scripts/log_loss.py` (claude-1). Read top-to-bottom
for the story: grounding % should rise, well-formedness errors should fall to 0,
debt is mostly the dominant ungrounded-symbol count (and irreducible definition
holes). See `holes/dp-fleet-plan.md` for the capability targets.

| time (local) | papers | grounded | wf-errors | debt | note |
|---|---|---|---|---|---|
| 2026-06-13T16:46:40 | 32 | 52% | 1 | 47176 | baseline at handoff: weft wf-clean + grounding/quantifier/defined-in-paper/refs online; warp W1+W3 done, W2-linkage + corpus-DEBT in flight; per-capability refactor in flight |
