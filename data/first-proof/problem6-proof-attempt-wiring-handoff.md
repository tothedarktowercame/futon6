# Handoff to Claude: Problem 6 Proof-Attempt Wiring (Overall + Zoom)

Date: 2026-02-12
Author: Codex

## What was added

1. `data/first-proof/problem6-proof-attempt-wiring.json`
- Wiring diagram for the full current proof-attempt structure.
- Captures the high-level reduction pipeline:
  - leverage-threshold/Turan reduction,
  - Case 2b isolation,
  - trace-route ceiling,
  - barrier/L2* reduction,
  - MSS/KS gap map,
  - final open bridge statement.

2. `data/first-proof/problem6-proof-attempt-gpl-zoom-wiring.json`
- Zoomed wiring focused only on the theorem-level bridge.
- Central question node is:
  - `H1-H4 => min_{v in R_t} ||Y_t(v)|| <= theta`.
- Includes:
  - hypotheses nodes H1..H4,
  - sufficiency-to-closure node,
  - two failure-mode challenge nodes (trace-averaging too weak, strong budget condition fails).

3. `scripts/proof6-proof-attempt-wiring-diagram.py`
- Generator that emits both JSONs in futon wiring schema (`nodes`, `edges`, `hyperedges`, `stats`).
- Current defaults write to the two paths above.

4. `data/first-proof/problem6-claude-handoff.md`
- Appended file pointers to the new overall and zoom wiring diagrams.

## Why this helps

- The overall diagram makes the global argument graph inspectable and easier to discuss across contributors.
- The zoom isolates exactly one open implication and its local dependencies/challenges.
- This should reduce ambiguity when proposing the next theorem search or proof tactics.

## How to regenerate

Run:

```bash
python3 scripts/proof6-proof-attempt-wiring-diagram.py
```

Optional custom outputs:

```bash
python3 scripts/proof6-proof-attempt-wiring-diagram.py \
  --overall-out data/first-proof/problem6-proof-attempt-wiring.json \
  --zoom-out data/first-proof/problem6-proof-attempt-gpl-zoom-wiring.json
```

## Validation done

- `python3 -m json.tool data/first-proof/problem6-proof-attempt-wiring.json`
- `python3 -m json.tool data/first-proof/problem6-proof-attempt-gpl-zoom-wiring.json`

Both parse successfully.

