# E-mission-mining — retrospective triple-mining of the mission corpus

**Date:** 2026-06-13 · owner: claude-1 seat (this session) · spun out of the
WM piloted flight `futon3c/holes/flights/F-wm-piloted-2026-06-12.md` (sortie 11).
**Status:** RUN — miners built and run once over the corpus; now in the
DP/anytime improvement loop (mine → measure loss → fix the worst → re-mine).

## Charter

Mine the futon mission corpus for the triple each mission is a derivation of
— (typed hole, term, wiring diagram) — plus the dry-basin negative class and
the forward hitlist. Feeds three downstream lanes: the substrate metric
(retrodictive gold), the drainage policy-landscape (`policy-landscape-drainage.md`),
and the anatomy paper's §8 third ascent. Imperfect by measurement, not by
suspicion — loss classes seed their own improvement passes
(`dark-tower-x-triples-synergies.md` for the BV-comb upgrade).

## What has landed

- **Triple miner** (`scripts/mission_triple_miner.py`, futon6 d61dc33 v1.1):
  81 completed-mission triples → `data/mission-triples/*.edn`. Schema pinned
  in `golden-graphs/SCHEMA.md`. v1.1 fixes: checkpoint harvest (0→17),
  HEAD/IDENTIFY/MAP fallback (21 → `:reconstructed-thin`), artifact resolution
  (912→1052 verified of 1320). Reviewed PASS (task #11).
- **Dry-basin miner** (`scripts/dry_basin_miner.py`, futon6 4f4978e): 133
  not-completed basins → `data/dry-basins/*.edn` + `_hitlist.json` (136
  advance-typed targets). Reviewed PASS (task #11).

## Finding F1 — the :checkpoint-only band is unreliable (task #12 audit)

Independent audit of the hitlist's top band: **0 of 4 `:checkpoint-only`
candidates were clean ready-to-close.** Full table + evidence:
`futon6/holes/dry-basin-hitlist-audit.md` (committed fa60be3). Summary:

- M-explore-aiqa — **already CLOSED** (Status line 11, 2026-06-04); the miner
  put a closed mission in the dry-basin set.
- M-war-machine-frontend-upgrade1 — **Open** (2/8 phases); the `:via` is a
  conditional close-criterion, not a verdict (false positive).
- M-bounded-in-flight-state — **genuinely open** (deliberate 2026-05-04
  reopen); its `:via` "exit criteria met" predates the reopen. NB its unmet
  exit (mana-earning unwired: `Block:` footers land but session `:earned 0`)
  IS the all-zero Session-AIF balances on the WM panel — a real `:ratified-car`
  mis-filed as `:checkpoint-only`.
- M-single-locus — **partial** (core done; siblings outstanding via GH #64);
  `:via` is a constraint note, not a close statement.

**Root cause:** advance-typing reads close-*language* anywhere in the body,
not the canonical `Status:` line, the `CLOSED` marker, or reopen markers.

## Loss-surface backlog (the improvement passes)

Carried from `_summary.json` (triple) + the audit (dry-basin). Ranked
DP-style in task #14; the miner-quality items born here:

1. **Status-aware advance-typing** (from F1): read `Status:`; detect
   `CLOSED`; honor the latest reopen marker. Fixes the whole `:checkpoint-only`
   band's reliability AND keeps already-closed missions out of the dry-basin
   set (they belong in neither corpus).
2. **Reconstruction pass** for the 58 `:missing-derive` triples (predate the
   derive gate) → push `:unminable` toward `:reconstructed-thin` via the
   HEAD/IDENTIFY/MAP fallback, honestly tiered.
3. **`:zero-pattern-cites` (38)** and **`:unverifiable-artifacts` (53/60)** —
   triage whether detector gap or genuinely absent.

## Cross-references

- Born: `futon3c/holes/flights/F-wm-piloted-2026-06-12.md` (sortie 11 work order).
- Consumers: `futon6/holes/policy-landscape-drainage.md` (drainage, 3 traps),
  the substrate metric (claude-3, paused — `scripts/substrate_metric_cascade_adapter.py`
  untracked), `holes/anatomy-of-a-futonic-mission.tex` §8.
- Schema: `golden-graphs/SCHEMA.md`. Upgrade path: `futon5a/holes/dark-tower-x-triples-synergies.md`.
