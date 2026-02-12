# GPL-H Formal Targets: Run Summary (n<=96, c_step=1/3)

Date: 2026-02-12
Runner: Codex

## Commands

```bash
python3 scripts/verify-p6-phase-structure.py --nmax 96 --c-step 0.3333333333 --seed 20260212
python3 scripts/verify-p6-phase2-m0.py --nmax 96 --c-step 0.3333333333 --seed 20260212
python3 scripts/verify-p6-double-counting.py --nmax 96 --c-step 0.3333333333 --seed 20260212
python3 scripts/verify-p6-trace-budget.py --nmax 96 --c-step 0.3333333333 --seed 20260212
python3 scripts/verify-p6-fresh-dbar-bound.py --nmax 96 --c-step 0.3333333333 --seed 20260212
python3 scripts/verify-p6-allscores-bound.py --nmax 96 --c-step 0.3333333333 --seed 20260212
python3 scripts/verify-p6-avg-charpoly.py --nmax 96 --c-step 0.3333333333 --seed 20260212
```

Raw logs are in `/tmp/p6-formal-target-runs/`.

## Results by target

### G1 (Phase-2 structure / m0 ratio)

From `verify-p6-phase2-m0.py`:

- Total Case-2b instances: `632`
- Phase-2 step rows: `581`
- `m0/n` on Phase-2 rows:
  - min `0.9524`
  - mean `0.9900`
  - max `1.0000`

Double-counting bound on Phase-2 rows:

- `dbar_dc` max `0.628019`
- `dbar_dc >= 1`: `0/581`

### G2 (greedy leverage-degree proportionality)

From `verify-p6-double-counting.py`:

- Rows: `581`
- Leverage ratio `avg_lev(S_t) / avg_lev(I0)`:
  - min `0.6800`
  - mean `0.9990`
  - max `1.0450`
- `dbar_provable` max `0.6280`
- `dbar_provable >= 1`: `0/581`

From `verify-p6-fresh-dbar-bound.py`:

- `dbar_fresh_bound` max `0.628019`
- `dbar_fresh_bound >= 1`: `0/581`

### G3 (avg charpoly route)

From `verify-p6-avg-charpoly.py`:

- Case-2b instances: `632`
- Nontrivial rows: `581`
- Violations (`largest_root >= 1`): `0`
- Largest root:
  - max `0.476190`
  - mean `0.272114`
  - p95 `0.400332`

## Global checks

From `verify-p6-allscores-bound.py`:

- Case-2b instances: `647`
- Nontrivial rows: `569`
- Worst `max_score`: `0.675353`
- Worst `dbar`: `0.683230`
- Rows with any `score > 1`: `0`
- Rows with `dbar >= 1`: `0`

From `verify-p6-phase-structure.py`:

- Phase 1 covers full horizon: `421/647` (`65.1%`)

## Interpretation

- Empirically, all three formal targets are consistent with GPL-H closure at this scale.
- The remaining gap is formal/theorem-level (not computational): proving the
  structural step that forces the Phase-2 regime and `dbar < 1` from H1-H4.
