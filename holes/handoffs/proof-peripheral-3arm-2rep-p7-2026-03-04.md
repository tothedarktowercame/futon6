# Proof Peripheral 3-Arm Rerun (P7, 2 Replicates, 2026-03-04)

Goal: test whether processed corpus structure adds value beyond raw/MO retrieval, while tracking latency and resumability behavior.

## Setup

Runner:

- `scripts/run-proof-polish-codex-p7.py`

Common controls:

- model: `gpt-5.3-codex`
- reasoning effort: `medium`
- web search: `disabled`
- timeout: `--timeout-seconds 300`
- retries: `--max-retries 2`
- resumable append: `--resume`

Arms:

1. processed-only
   - `--math-se-dir se-data/math-processed`
2. MO-only
   - `--math-se-dir se-data/mathoverflow.net`
3. processed+MO (dual-path hint)
   - `--math-se-dir se-data/math-processed --extra-corpus-dir se-data/mathoverflow.net`

Outputs:

- `data/first-proof/problem7-codex-results-exp-processed-rep{1,2}.jsonl`
- `data/first-proof/problem7-codex-results-exp-mo-only-rep{1,2}.jsonl`
- `data/first-proof/problem7-codex-results-exp-processed-plus-mo-rep{1,2}.jsonl`

Scoring used for comparison:

- `verified=2`, `plausible=1`, `gap/error=0`

## Per-replicate results

- processed rep1: `verified=4, plausible=2, gap=3, error=0`, score=`10/18` (avg `1.111`)
- processed rep2: `verified=3, plausible=2, gap=4, error=0`, score=`8/18` (avg `0.889`)

- MO-only rep1: `verified=4, plausible=1, gap=4, error=0`, score=`9/18` (avg `1.000`)
- MO-only rep2: `verified=3, plausible=4, gap=2, error=0`, score=`10/18` (avg `1.111`)

- processed+MO rep1: `verified=3, plausible=2, gap=3, error=1`, score=`8/18` (avg `0.889`)
- processed+MO rep2: `verified=2, plausible=5, gap=2, error=0`, score=`9/18` (avg `1.000`)

Note: processed+MO rep1 contains one parse failure row (`p7-s4`) with `attempts=2`, `timed_out=true`.

## Aggregate across both replicates (18 judgments per arm)

- processed-only
  - status: `verified=7, plausible=4, gap=7, error=0`
  - score sum: `18` (avg `1.000`)
  - retries/timeouts: `attempts>1 = 3`, `timed_out=true rows = 3`
  - elapsed: total `3792.4s`, mean `210.7s`, median `188.8s`, max `591.9s`
  - references: total `69` (`mathoverflow.net=47`, `math.stackexchange.com=22`, MO share `68.1%`)

- MO-only
  - status: `verified=7, plausible=5, gap=6, error=0`
  - score sum: `19` (avg `1.056`)
  - retries/timeouts: `attempts>1 = 0`, `timed_out=true rows = 0`
  - elapsed: total `1767.2s`, mean `98.2s`, median `88.6s`, max `164.9s`
  - references: total `66` (`mathoverflow.net=57`, `math.stackexchange.com=9`, MO share `86.4%`)

- processed+MO
  - status: `verified=5, plausible=7, gap=5, error=1`
  - score sum: `17` (avg `0.944`)
  - retries/timeouts: `attempts>1 = 1`, `timed_out=true rows = 1`
  - elapsed: total `2558.4s`, mean `142.1s`, median `111.4s`, max `600.1s`
  - references: total `66` (`mathoverflow.net=59`, `math.stackexchange.com=7`, MO share `89.4%`)

## Pairwise node-level comparison (18 rep-node units)

Using score ordering `verified > plausible > gap/error`:

- processed vs MO-only: processed better `1`, worse `2`, tie `15`
- processed vs processed+MO: processed better `4`, worse `3`, tie `11`
- MO-only vs processed+MO: MO-only better `3`, worse `1`, tie `14`

Interpretation: quality signal is weak and highly tie-dominated at this sample size.

## Main takeaways

- The missing third arm concern is now addressed: all three arms ran with two replicates.
- No clear quality win for processed structure in this experiment.
- MO-only is slightly best on average score and clearly best on latency/stability.
- processed-only shows a real long-tail latency issue (timeouts/retries) that can bias wall-clock and throughput.
- Reference sourcing is MO-dominant in all arms, but not fully stable in mix; processed-only shifts toward more Math.SE references.

## Runner hardening update

`run-proof-polish-codex-p7.py` now also supports resume recovery controls:

- `--resume-rerun-failures` re-runs rows with parse/invalid outputs during resume.
- `--resume-rerun-timed-out` re-runs rows previously marked `timed_out=true`.

This closes a practical gap where prior `--resume` treated parse-failed rows as completed.

## Recommendation for next superpod iteration

- Use MO-only as operational baseline for research-facing proof checks.
- Keep processed+MO as candidate arm, but only after additional latency work or retrieval-quality instrumentation.
- If budget allows, add replicate 3 before making a structural-retrieval claim.
- Add reference-quality auditing (not just site counts): uniqueness, direct relevance, and citation correctness.
