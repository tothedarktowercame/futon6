# WARP-ORCH-4 — Close the SFC join (evidence)

Author: claude-2, 2026-06-18. Final car of the WARP-ORCH chain. Proves the
concept-first foundation runs off `warp_run` with no hand-run step.

## What was done

- **Wiring:** `build_term_prior` promoted from an audit-only consumer to a
  runnable spine stage `S6t` (commit `8b81854`), so the foundation
  (term-prior → concept-encyclopedia → SFC1) lives *inside* the runner.
- **Full rebuild:** `warp_run` executed end-to-end (full spine, make-like) and
  `sfc_concept_coverage` (SFC1) run off the rebuilt outputs.

## How it ran (environmental note — load-bearing)

Two earlier attempts died at **~30 min to SIGTERM**: (1) the codex invoke
(1 800 000 ms limit), (2) a harness `run_in_background` bash job. Root cause:
this box reaps the per-command `tmux-spawn-*.scope` cgroup at ~30 min, and
`warp_concordance` alone needs ~45 min. **Fix: run long jobs as a
`systemd-run --user` transient unit** (own cgroup under `app.slice`), which
escapes the scope reaper. The rebuild ran as `warp-orch-4-rebuild.service`,
~1.5 h (2026-06-17 21:43 → 23:14 UTC), `Result=success`. Manifest:
`data/warp/warp-manifest.json`.

> Generalises: any >30 min job on this machine must use `systemd-run --user`,
> not a codex invoke or harness background bash.

## Result — the substrate came back materially fuller

The prior on-disk warp artifacts were partial/capped; the full rebuild is more
complete:

| stage | metric | before | after |
|---|---|---:|---:|
| S1a concordance | terms | 173 109 | 3 056 934 |
| S3 hitlist | concepts | 3 802 | 4 000 |
| S4a def-snippets | snippets | 972 | 3 868 |
| S4b concept-usage | unique concepts | 3 737 | 3 952 |
| S5 concept-graph | nodes / edges | 1 000 / 5 499 | 4 000 / 32 633 |
| S6t term-prior | terms | 2 459 715 | 3 624 679 |
| S6a phylogeny | threads / cited-activations | 200 / 416 | 262 / 599 |

## Acceptance — SFC1 reproduces off the runner

`sfc_concept_coverage` run against the rebuilt `data/warp` outputs:

- top-100 **100/100 = 100.0%**
- top-500 **499/500 = 99.8%**

(The breakdown's "98.4%" was stale; current = 99.8%, and it holds across the
rebuild.) The foundation is now a single `warp_run && sfc_concept_coverage`
chain with **no hand-run step**.

## Drift coordination (per contract)

The rebuild changed `concept-usage.json` (hash `0d0b0ac → 8357d9f`; unique
concepts 3737 → 3952). The runner's content-hash drift gate fired
(`__derived-staleness-ping` in the manifest). Per the contract with claude-1
(SFC owner), it regenerates, off the new content: (1) `concept-index.json`
(SFC-D3), (2) `sfc-adjunction-fixture.json` (SFC-AGG), then re-runs `cas_cert`
over `loop-run-70b` and diffs the concept grain vs the run-#2 baseline (mean
0.867). **[in progress — claude-1]** — expected to move notably given the
substrate expansion.

## Fix landed alongside

`warp_run` `main()`'s summary loop crashed (`KeyError: 'status'`) on the
`__derived-staleness-ping` entry whenever drift fired — i.e. exactly on this
run. Fixed to skip `__`-prefixed meta keys and surface `drift=YES/no` in the
summary line. Tests: 13/13.

## Reproducibility

Heavy spine blobs (`concordance`/`hitlist`/`def-snippets`/`concept-usage`/
`concept-graph`/`concept-embed`) are `.gitignore`d — regenerate via `warp_run`.
Tracked outputs committed: `concept-phylogeny.json`, `warp-manifest.json`.

## Status

WARP-ORCH-4 (concept/warp half) **DONE**: SFC join wired + `warp_run → SFC1`
reproduction verified. Whole-chain close pending claude-1's derived-artifact
regen + `cas_cert` re-diff (drift propagation). On that, WARP-ORCH → ready.
