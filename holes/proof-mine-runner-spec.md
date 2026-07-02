# PROOF-MINE runner spec (the GPU discharge-evidence mining runner)

**Date:** 2026-07-02 · **Owner:** Joe + claude-11 (design owned here per Joe: "I'd trust you to design
the runner so that it actually tells us useful things") · **Status:** SPEC (design; the runner build
follows claude-3's CPU pilot — see §Rungs).
**Runner BUILT (2026-07-02, claude-1, direct — codex out of quota):** `futon6/scripts/proof_mine.py`
(D3/D4/D5/D6/D10) · `proof_mine_dossier.py` (dossier grain + D10 budget) · `check_proof_mine_gates.py`
(D2) · `linode-proof-mine.sh` (RUNG=smoke|gold|full) · `land_proof_mine.bb` (D7, dry-run default) ·
`proof_mine_manifest.py` (D8) · `tests/proof_mine_test.py` (9 tests). Stub-validated end-to-end; gates
PASS on the 10 gold missions; gold-eval aborts nonzero on the D5 bands. NO box commissioned, NO GPU
sweep, NO :7071 writes — those stay Joe's gated steps. `pair_unverified=true` on records until the
E-have-want pairs corpus is located on disk (the pilot's ⚠pair).
**For:** growing substrate-2's PROOF layer at scale — the deferred M-fold-ansatz batch-recovery, recast
to LAND: per-mission discharge evidence feeding [[E-C-vector-live]] §11 (`:discharged-by` toward the
~70 uncovered mission-directed c-entries), R16-EXEC-REACH (executor rule candidates), R14 variance
(contract v0.22: closure = γ ≠ 1.0 in a live trace), and E-wiring-diagram-corpus (impl-#3 supply).
**Siblings this learns from (read them; they are the working prior art):**
`linode-meme-mine.sh` / `linode-goals-and-holes-mine.sh` (tunnel-first harness),
`meme-mine-runner-spec.md` (the spec form + the smoke-gate-with-fail-bands added after a run shipped
~50% bad), `c_mine_joint.py` (per-item resilience + checkpointing after the null-crash total-loss),
`check_fold_embed_gates.py` (gates-as-code that caught the empty-cascades null-ablation before spend),
`mission_dossier.py` (per-mission input assembly), `futon0/README-linode.md` (StackScript 2142757 box path).

## What one unit of work is

For ONE mission: assemble the **dossier** (CPU, local — `mission_dossier.py` grain: mission doc +
commits citing it + code endpoints via `edits`/`calls` + live XTDB counts + its c-entries and their
`:outcome-ref`s), then ONE LLM pass (mark4-70b vLLM) that emits, **gold-primed by the A-next corpus
as few-shot**, a graded record:

```edn
{:mission        "<repo>-d/mission/<stem>"          ; CANONICAL, resolved BEFORE emission (D6)
 :discharges     [{:target <c-entry-name | sorry-ref>
                   :discharged-by <sha | method-ref | nil>
                   :grade :discharged | :open | :unverified | :research   ; A-next honesty grades
                   :witness "<verbatim span from the dossier>"}]           ; cite or :unsupported
 :endpoints      [...]                               ; the mission's sorry interface, A-next shape
 :rule-candidates [{:pattern <id> :box <verb> :warrant "<span>"}]}         ; executor-reach material
```

`:open` on IDENTIFY-stage missions is EXPECTED OUTPUT, not failure — the run's accounting reports the
grade distribution explicitly (the old "(b) would mostly add :open sorries" worry becomes a measured
number, not a reason not to look).

## Design decisions (each traceable to a scar or a working pattern)

- **D1 — Tunnel-first, sync NOTHING.** The box serves vLLM only; dossiers, gold corpus, and all writes
  stay on dev (`ssh -L 8000:... ` with `ServerAliveInterval=30` — the tunnel-keepalive lesson; kill
  tunnels by exact pid, never `pkill -f`). Mirrors the two mine shells verbatim.
- **D2 — Gates-as-code BEFORE spend.** `check_proof_mine_gates.py` (local, no torch, no GPU): every
  dossier non-empty (doc found, ≥1 commit or an explicit `:no-code-trail` flag, c-entries attached);
  every gold few-shot parses; every mission id resolves canonically. The fold-embed gate script caught
  a null ablation that three arms would have "run" silently — same move here.
- **D3 — Per-item resilience + checkpoint-append + resume.** Per-mission try/except (a null field must
  cost ONE mission, not the run — the `meme_mine_joint` 1448-ask total-loss); results APPEND to
  `proof-mine.jsonl` as they complete (never end-of-loop write); `--resume` skips missions already in
  the artifact. Checkpoint counter by NEW records, not modulo (the "Y2C" fix).
- **D4 — In-flight observability.** Every 10 missions, write `proof-mine-status.json` (done/total,
  grade distribution, grounding rate, mean latency, ETA) — tailable from dev mid-run (Rob's
  loss-logging preference; also the abort trigger's input).
- **D5 — GOLD-ANCHORED: the run carries its own yardstick.** Phase order is fixed: the 10 A-next gold
  missions are re-mined BLIND first, scored against the sealed `*-EMPIRICAL.edn` (endpoint recall/precision,
  discharge-grade agreement, witness-span validity). **Quantified abort band:** endpoint precision < 0.5
  OR grade agreement < 0.6 OR verbatim-witness rate < 0.7 ⇒ STOP, fix the prompt, no full sweep. This is
  the anti-cockup core: the run cannot produce 200 missions of plausible junk because it must first
  reproduce 10 missions of known truth.
- **D6 — Canonical vocabulary AT EMISSION.** Every mission/c-entry/method ref passes the mission-index +
  token bridge (futon2 `2fd9022` rules) before it is written to the artifact; unresolvable refs go to
  `proof-mine-quarantine.jsonl` with the raw span — NEVER minted (islands = 0; the archivist-gate spirit;
  the used-var-bundle lesson: unmatchable ids are worse than no ids).
- **D7 — Landing is a separate, gated, CPU step.** The box writes NOTHING to :7071. Landing reuses
  `promote_c_entries.bb`'s write path (run-write! pipeline, x-penholder api, dry-run default), with
  author≠reviewer: dry-run report → claude-11/claude-10 review → `--execute`. Rule-candidates land as
  fold_engine rule-table PRs, not silent edits.
- **D8 — Capture before decommission.** `manifest.json` (+ sha256 of every artifact) written and
  rsynced to dev BEFORE the box is deleted; box is from-dev mode (holds nothing unique); powered-off
  still bills ⇒ "stop" means DELETE, after the manifest gate.
- **D9 — Smoke ladder with QUANTIFIED fail bands** (the ~50%-bad lesson: stub mode does not exercise
  the prompt). Rungs: (0) claude-3's CPU pilot = prompt validation on 10–20 uncovered missions (in
  flight — its precision numbers are an input to this spec's go/no-go); (1) `--backend stub --limit 3`
  plumbing; (2) `--limit 3 --backend openai` REAL smoke — bands: ≥2/3 missions produce ≥1 graded
  discharge with a verbatim witness; grade distribution not degenerate (>90% one grade = prompt fail);
  (3) the gold-10 blind eval (D5 bands); (4) full sweep (~200 missions).
- **D10 — Cost + abort accounting up front.** Dossier budget ≤ 12k tokens (truncate by section priority:
  doc HEAD/status → cited-commit subjects → endpoint lists; log truncations — no silent caps), output
  ≤ 2k ⇒ ~3M tokens ≈ hours on the 4×RTX4000 box, not days. The status file's grade/grounding trends
  are the mid-run abort inputs; wall-clock hard stop 6h (past it, something is wrong — capture and stop).

## Run mechanics (mirrors the siblings)

Box: `g2-gpu-rtx4000a4-s` via StackScript `2142757` (README-linode). Serve mark4-70b
(`linode-4gpu-setup.sh`). One shell, `scripts/linode-proof-mine.sh`: wait-for-vLLM gate → gates-as-code
(D2) → the rung the operator asked for (env knob `RUNG=smoke|gold|full`) → owner-review rubric printed
at the end. Env knobs: `PORT MODEL REPO PYTHON LIMIT RUNG OUT`. All FATALs carry their remedy inline
(the sibling shells' discipline).

## Owner-review rubric (gate substance, not a PASS)

- **Grade split**: `:open`-heavy on IDENTIFY missions is healthy; `:discharged`-heavy overall = the
  prompt is credulous (check witnesses); >90% any single grade = degenerate.
- **Witness validity**: spot-check 10 random witnesses ARE verbatim dossier spans (the SFC2b discipline).
- **Quarantine rate**: a few % unresolvable is healthy; >20% = the vocabulary bridge or the dossier
  assembly is broken — fix before landing anything.
- **Join delta**: after landing, re-run the §11 coverage query — the point of the run is
  119/189 → materially higher, and the uncovered-70 shrinking. Report the delta, not the raw count.
- **R14 hook**: count rule-candidates accepted into the executor; after the next enacting ticks, check
  γ's perf-history for its first non-zero sample (the contract-v0.22 closure).

## Interlocks

claude-3's CPU pilot (dispatched 2026-07-02) is **rung 0** — its prompt + precision numbers feed D5/D9
directly; this spec goes to build only after the pilot reports and Joe commissions the box. The GFN
"Upgrades" shelf is untouched by this run (this is mining, not generation) — but D6's canonical refs are
exactly what makes any future impl-#3 corpus consumable.
