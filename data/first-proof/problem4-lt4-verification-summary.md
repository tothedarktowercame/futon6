# Problem 4 (n<4) Codex Verification Summary

## Run metadata
- Date: 2026-02-11
- Command:
  - `python3 scripts/run-proof-polish-codex-p4-lt4.py --limit 8`
- Model: `gpt-5.3-codex`
- Output JSONL:
  - `data/first-proof/problem4-lt4-codex-results.jsonl`
- Prompts JSONL:
  - `data/first-proof/problem4-lt4-codex-prompts.jsonl`

## Headline result
- Processed: `8`
- Verified: `5`
- Plausible: `0`
- Gap: `2`
- Error: `1`

## Per-node outcomes
- `p4-problem`: `verified`
- `p4-s5`: `verified`
- `p4-s5a`: `verified`
- `p4-s5b`: `verified`
- `p4-s5c`: `verified`
- `p4-s5d`: `gap`
- `p4-s6`: `error`
- `p4-lt4-synthesis`: `gap`

## Key flagged issues for follow-up
1. `p4-s5d` equality characterization is too broad.
   - Verifier says surplus equality requires `u=v=0` (under `s,t>0`), and the extra case `s=t, u=-v` with nonzero `u` is not valid.

2. `p4-s6` overstates n=3 strictness.
   - Core superadditivity proof is accepted.
   - "strict" should be qualified because equality can occur (e.g., centered symmetric cubics with `u=v=0`).

3. `p4-lt4-synthesis` also marks strictness/equality conditions as the unresolved gap.

## Suggested patch target in proof text
- `data/first-proof/problem4-solution.md`
  - Replace unqualified "strict" language for n=3 with a statement that explicitly handles equality cases.
  - Tighten equality-condition statement in the Titu/Engel step.
