# Superpod Run Readiness Contract (2026-03-04)

Purpose: make each `superpod-job` run self-describing and machine-checkable,
so we can decide quickly whether a run is ready for downstream work.

## What changed

`superpod-job.py` now writes explicit run contract fields to `manifest.json`:

- `stage_status`: per-stage lifecycle record (`completed` / `skipped` / `prompt_only`) with `skip_reason` when skipped.
- `health_gate_thresholds`: thresholds used for this run.
- `health_issues`: health warnings collected during execution.
- `readiness`: summary status + issue count + `preflight` mode.

## New gates

New configurable gates (CLI flags):

- `--gate-stage6-parse-rate-min` (default `0.10`)
- `--gate-stage7-categorical-rate-min` (default `0.03`)
- `--gate-stage7-port-rate-min` (default `0.04`)
- `--gate-stage9b-val-acc1-min` (default `0.98`)

Existing behavior remains:

- `--preflight`: fail-fast on any health warning.
- Without `--preflight`: warnings are recorded in `health_issues` and run continues.

## Recommended preflight command

Use this before expensive production runs:

```bash
cd /home/joe/code/futon6
python3 scripts/superpod-job.py /path/to/Posts.xml \
  --site math.stackexchange \
  --output-dir /path/to/outdir \
  --preflight \
  --gate-stage6-parse-rate-min 0.10 \
  --gate-stage7-categorical-rate-min 0.03 \
  --gate-stage7-port-rate-min 0.04 \
  --gate-stage9b-val-acc1-min 0.98
```

## Evaluator compatibility

`evaluate-superpod-run.py` now reads and reports:

- `readiness`
- `stage_status`
- `health_issues`

It also validates stage-status contract completeness and missing skip reasons.

## Operational expectation

A run is "ready for next-step consumption" when:

- `manifest.readiness.status == "pass"`
- `manifest.health_issues` is empty
- required downstream stages show `stage_status[*].status == "completed"`

If status is `warn`, resolve the listed issue(s) before treating outputs as
production-grade.
