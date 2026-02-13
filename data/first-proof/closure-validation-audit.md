# Closure vs Validation Audit (Cross-Problem)

Date: 2026-02-13

Purpose:
- check whether "closed/fully solved" labels are backed by current
  node-level validation artifacts;
- flag premature closure claims for quality-control correction.

Criteria used in this audit:
- `validated`: parseable node-level verifier artifact exists with no unresolved
  verifier-level `gap/error` flags;
- `partial validation`: verifier artifact exists but contains unresolved
  `gap/error` outputs;
- `pending validation`: no current node-level verifier artifact found.

## Status Table

| Problem | Prior manuscript status | Node-level validation artifact | Audit status | Notes |
|---|---|---|---|---|
| P1 | closed / fully solved | `problem1-codex-results.jsonl` | partial validation | 9 outputs: 0 verified, 2 plausible, 7 gap, 0 error |
| P2 | closed / fully solved | `problem2-codex-results.jsonl` | partial validation | 10 outputs: 0 verified, 3 plausible, 6 gap, 1 error |
| P3 | closed / fully solved | `problem3-codex-results.jsonl` | validated (existence scope) | 9 outputs: 2 verified, 7 plausible, 0 gap, 0 error |
| P4 | partial | `problem4-lt4-codex-results.jsonl` | partial validation | status already partial; validator includes gaps/errors |
| P5 | scope-limited | `problem5-codex-results.jsonl` | partial validation | status already limited; validator includes gaps |
| P6 | partial | `problem6-codex-results.jsonl` (empty) | pending validation | file exists but no outputs |
| P7 | closed / fully solved | `problem7-codex-results.jsonl` | partial validation | 9 outputs: 0 verified, 4 plausible, 5 gap, 0 error |
| P8 | closed / fully solved | `problem8-codex-results.jsonl` | partial validation | 11 outputs: 0 verified, 5 plausible, 6 gap, 0 error |
| P9 | closed / fully solved | `problem9-codex-results.jsonl` | partial validation | 12 outputs: 1 verified, 4 plausible, 7 gap, 0 error |
| P10 | conditional | `problem10-codex-results.jsonl` | validated (conditional) | 15 outputs: 8 verified, 7 plausible, 0 gap, 0 error |

## Immediate QC Implication

Current high-level manuscript labels should distinguish:
- `closed in manuscript form, with node-level validation artifacts now present but still containing unresolved gaps`
  (P1, P2, P7, P8, P9);
- `existence path validated under scoped criterion (no remaining validator gaps in current run)`
  (P3);
- `conditional with completed node-level verification artifact`
  (P10).

## Evidence Files

- `data/first-proof/latex/first-proof-monograph.tex`
- `data/first-proof/latex/intro-making-of.tex`
- `data/first-proof/problem3-codex-results.jsonl`
- `data/first-proof/problem4-lt4-codex-results.jsonl`
- `data/first-proof/problem5-codex-results.jsonl`
- `data/first-proof/problem6-codex-results.jsonl`
- `data/first-proof/problem1-codex-results.jsonl`
- `data/first-proof/problem2-codex-results.jsonl`
- `data/first-proof/problem7-codex-results.jsonl`
- `data/first-proof/problem8-codex-results.jsonl`
- `data/first-proof/problem9-codex-results.jsonl`
- `data/first-proof/problem10-codex-results.jsonl`
