# Dry-basin hitlist audit — :checkpoint-only band (2026-06-13, claude-1 seat)

Independent audit of the four `:checkpoint-only` candidates the dry-basin
miner ranked at the top of `data/dry-basins/_hitlist.json`. Each `:via`
was checked against the mission's actual Status line + closing text, not
trusted from the quote alone. **Verdict: 0 of 4 are clean ready-to-close.**
Do not bundle-close any of these on the hitlist's say-so.

| rank | mission | miner :via | TRUE state | verdict |
|---|---|---|---|---|
| 1 | M-explore-aiqa | "Mission complete through DOCUMENT. Ready to close…" (verbatim, line 209) | **Status line 11: "CLOSED 2026-06-04 (Joe's call)"** | ALREADY CLOSED — miner misclassified a closed mission as a dry basin |
| 2 | M-war-machine-frontend-upgrade1 | "either landed or explicitly deferred, the mission is ready to close" (line 569) | Status: **Open**; only 2/8 phases (map, document); the :via is a conditional close-*criterion*, not a verdict | FALSE POSITIVE — genuinely open |
| 3 | M-bounded-in-flight-state | "All four exit criteria met:" | Status: open, with a deliberate **2026-05-04 reopen** into INSTANTIATE; explicit unmet exit: "close again only after the operator verifies a fresh commit visibly increments the author's session balance" | GENUINELY OPEN — :via predates the reopen |
| 4 | M-single-locus | "documentation-only; do not rename live family." (line 81) | Status: "INSTANTIATE complete for the mission-home slice; siblings handed to Codex via GitHub issue #64" | PARTIAL — core done, siblings outstanding; :via is a constraint note, not a close statement |

## Root cause (miner advance-typing weakness)

The `:advance-type` heuristic keys on close-*language* ("ready to close",
"exit criteria met", "complete through DOCUMENT") anywhere in the body,
without:
1. **reading the canonical `Status:` line** — would have caught #1 (CLOSED)
   and #2/#3 (Open);
2. **detecting already-CLOSED missions** — #1 should never have been in the
   dry-basin (not-completed) set at all; the triple-miner's completed-detector
   ALSO missed it (it's in neither corpus cleanly). A `Status:.*CLOSED`
   check belongs in both miners.
3. **respecting reopen markers** — #3's "reopen" note inverts an earlier
   "exit criteria met"; the latest reopen wins.

This is a loss-surface item for the dry-basin miner (fits the anytime/
DP-improvement doctrine): the `:checkpoint-only` typing is currently
unreliable because it reads language, not state.

## The one thread worth pulling now

#3 M-bounded-in-flight-state's unmet exit IS the bug the WM panel still
shows: its reopen (G-1, "mana-earning unwired", 27 commits with `Block:`
footers but every session `:earned 0`) is exactly the all-zero Session-AIF
balances on this morning's War-Machine panel. So this mission is not
closeable — but it names, precisely, the fix for a live WM instrument
gap. That's a real `:ratified-car`-shaped piece of work (the
`record-mana!` call sites / Block-completion → mana-credit wiring),
mis-filed as `:checkpoint-only`.

## Recommendation to Joe

- Close nothing from this band blind. #1 is already closed (no action).
  #2, #3, #4 are open/partial and need real review, not a rubber stamp.
- The valuable find is #3 → the mana-earning wire-up, which would also
  un-zero the WM panel. Candidate to dispatch.
- Feed the three root-cause checks (Status-line read, CLOSED-detection,
  reopen-marker) back to the dry-basin + triple miners (loss-surface
  backlog, task #14-adjacent).
