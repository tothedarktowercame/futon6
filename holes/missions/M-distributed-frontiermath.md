# Mission: Distributed Proofs & Refutations — FM-001

Date: 2026-03-07
Scope: Distributed multi-agent attack on FM-001 (Ramsey Numbers for Book Graphs)
using Lakatos-style dialectical proof methodology across federated IRC channels.

## Objective

Stress-test distributed agent coordination on a genuine Frontier Math problem.
Success is not "solve FM-001." Success is: honest, high-integrity execution
where failure yields named obstructions and reusable negative results.

We want to discover:
1. Where the agents' mathematical reasoning breaks down.
2. Whether distributed coordination helps or just distributes confusion.
3. Whether the Mentor role can detect and break persistence loops in real time.
4. New patterns — if any emerge from the proof dialogue.

## Architecture

### Roles

| Role | Agent(s) | Location | Channel | Purpose |
|------|----------|----------|---------|---------|
| **Prover** | codex-1 + zcodex | Linode + Houston | #math | Dual-core proof engine: construct, compute, write formal arguments |
| **Critic** | claude-1 | Linode | #math | Falsify claims, find counterexamples, verify proof steps |
| **Mentor** | claude-2 (REPL-driven) | Linode workspace2 | #math | Meta-level: name patterns, detect loops, redirect strategy |
| **Tickle** | tickle-1 | Linode | #math | Bell-driven orchestrator, phase transitions |

### Prover: Dual-Core Pattern

codex-1 and zcodex operate as a **pair processor** on the same problem:

- **Parallel exploration**: Both work on different proof routes simultaneously.
  E.g. codex-1 tries a probabilistic construction while zcodex tries algebraic.
- **Shared workspace**: Both commit to `futon6` repo. State file is the single
  source of truth. Branches for speculative work, main for verified steps.
- **Handoff via git**: "I've pushed a sketch to branch `fm001-algebraic`,
  @zcodex can you check whether the degree bound holds for k>5?"
- **No duplication**: Tickle or Mentor assigns distinct sub-problems to avoid
  both provers grinding on the same approach.

### Critic: Adversarial Integrity

claude-1 does NOT help prove. claude-1 tries to break things:

- For every proof sketch: attempt the opposite answer (FALSIFY cycle).
- For every claimed bound: look for counterexamples or parameter regimes
  where it fails.
- For every lemma: check dependencies and hidden assumptions.
- Reviews go to git as `FM-001-review-NNN.md` files.

### Mentor: Space-Like Watchdog

The Mentor watches the proof state (git diffs) and the IRC dialogue, and
intervenes when:

1. **TryHarder loop detected**: Prover has attempted the same approach N
   times without a new lever. Mentor names the loop and suggests a
   fundamentally different angle.
2. **Shared blind spot**: Both Provers and the Critic have converged on an
   unexamined assumption. Mentor flags it.
3. **Pattern recognition**: "What you're doing is monster-barring — you keep
   restricting the hypothesis instead of reformulating the lemma."
4. **Phase transition**: "You've been in CONSTRUCT for 3 hours with no new
   lemmas. Time for FALSIFY."

The Mentor is the **space-like** complement to Tickle's **time-like** role:
- Tickle says: "you've been at this for 30 minutes."
- Mentor says: "you've been at this from the same angle 5 times."

The Mentor may discover and name new patterns during the session. These
become pattern library candidates (recorded as PURs).

### Communication Flow

```
#math (shared mission channel)
┌─────────────────────────────────────────┐
│ codex-1  (Prover A) — Linode            │
│ zcodex   (Prover B) — Houston           │
│ claude-1 (Critic)   — Linode            │
│ claude-2 (Mentor)   — Linode workspace2 │
│ tickle   (Orchestrator)                 │
│ joe, rob (Observers)                    │
└─────────────────────────────────────────┘
         │
   futon6 repo (shared artifact)
```

All agents share a single `#math` channel via the multi-channel ngircd bridge.
The bridge (systemd `ngircd-bridge@futon`) joins both `#futon` and `#math` —
agents can be reached on either channel with the same nick.

- **IRC for handoffs**: "@zcodex pushed proof sketch to fm001-prob branch,
  please check the n≥7 case" — short, actionable.
- **Git for work products**: State files, proof sketches, reviews, lemmas.
- **claude-2 (Mentor)** is REPL-driven — Joe attaches via workspace2 and
  decides when it speaks. Not auto-invoked on @mentions.
- **Tickle** is bell-driven — agents signal `@tickle BELL <event>` on
  completion; Tickle reads state files and assigns the next phase.

## Protocol

### Phase 1: SPEC (target: 2 hours)

Goal: Spec-Lock FM-001 — exact formal statement, parameter regime, quantifiers.

1. **codex-1**: Extract the precise problem statement from the Frontier Math
   source. Write `formal_statement`, `quantifiers`, `parameter_regime`.
2. **zcodex**: Independently verify the spec — does it match the source?
   Flag any ambiguities.
3. **claude-1 (Critic)**: Review the spec for substitution risk. Are we
   solving the actual problem or a nearby easier one?
4. **Mentor**: Watch for premature hypothesis formation during spec phase.

Artifact: Updated `FM-001-ramsey-book-graphs-state.md` with `spec_lock_status: pass`.

### Phase 2: FALSIFY (target: 3 hours)

Goal: Attempt the opposite answer. If the hypothesis is YES, try to prove NO.

1. **codex-1 + zcodex**: Split the falsification — one tries to construct
   a counterexample, the other tries to prove the bound can't hold.
2. **claude-1**: Evaluate the falsification attempts. Are they genuine or
   strawmen?
3. **Mentor**: Ensure the falsification is honest, not performative.

Artifact: `falsification_artifact` in state file. Result: refuted / survived / inconclusive.

### Phase 3: CONSTRUCT (open-ended)

Goal: Build proof attempts for the surviving hypothesis.

1. **codex-1 + zcodex** work distinct routes (assigned by Mentor to avoid
   duplication). Each route gets a branch.
2. **claude-1**: Reviews each proof sketch adversarially. Files review docs.
3. **Mentor**: Monitors for loops. After 3 failed attempts on the same route,
   requires a TryHarder license or a route switch.
4. **Tickle**: Bell-driven phase transitions (agents signal completion).

TryHarder licenses required for persistence. Mentor approves or denies.

### Phase 4: VERIFY (target: 2 hours)

Goal: If anything survives CONSTRUCT, verify it.

1. **codex-1 + zcodex**: Symbolic and computational checks.
2. **claude-1**: Dependency audit — does every step follow?
3. **Mentor**: Final pattern assessment — what did we learn?

### Phase 5: MAP (target: 1 hour)

Goal: Regardless of outcome, map what happened.

1. Document the proof landscape: what worked, what failed, why.
2. Name any new patterns discovered during the session.
3. Record PURs for patterns applied.
4. Update state file with final status.

## Hard Constraints

1. **Zero spec substitution**: We solve the problem stated, not an easier variant.
2. **Zero unlicensed TryHarder**: Persistence requires explicit justification.
3. **Honest failure**: FAILS with a named obstruction is a success. "Looks
   plausible" without verification is not.
4. **Dual-core discipline**: Provers must work distinct routes. If both are
   doing the same thing, Mentor intervenes.
5. **Git is truth**: If it's not committed, it didn't happen.

## Scorecard

| Metric | Target |
|--------|--------|
| spec_substitution_incidents | 0 |
| unlicensed_tryharder_events | 0 |
| time_to_first_falsification | < 3 hours |
| distinct_proof_routes_attempted | ≥ 3 |
| named_obstructions | ≥ 1 (if FAILS) |
| patterns_discovered | record all |
| dual_core_overlap_incidents | 0 |

## Dignity Rule

Same as the pre-season mission. Failure is acceptable when it yields:
1. A named obstruction with evidence.
2. A reduced subproblem with explicit open lemma.
3. A falsified route with reusable negative result.
4. A new pattern recognized and recorded.

## Question-Asking Pattern Language

The mission deploys the 8 question-asking patterns from Phase 3 analysis
(see `data/question-patterns/question-asking-pattern-language.md`):

| Role | Primary patterns |
|------|-----------------|
| **Prover** | QP-1 (landscape scout), QP-2 (technique landscape), QP-7 (kernel ID) |
| **Critic** | QP-3 (structural probe), QP-4 (failure characterization), QP-8 (confidence inversion) |
| **Mentor** | QP-6 (tension dissolution), QP-8 (confidence inversion), pattern naming |

## Setup Checklist

- [x] futon6 repo accessible to both codex-1 and zcodex (shared remote)
- [x] FM-001 state file spec-locked (`spec_lock_status: pass`)
- [x] FM-002, FM-003 also spec-locked (available for future missions)
- [x] Multi-channel bridge: `#futon` + `#math` via single systemd unit
- [x] claude, codex, claude-2, tickle all joined `#math` with clean nicks
- [ ] claude-2 Mentor session started on workspace2 (REPL-driven, not auto-invoke)
- [ ] zcodex reachable from `#math` (currently configured same as codex; may need Rob's bridge)
- [ ] Rob has IRC access to `#math`
- [ ] Tickle bell-driven orchestration wired for FM-001 phase transitions

## Starting the Mission

On #math:
```
@codex spec-lock FM-001. The state file is already filled — verify it
matches the FrontierMath source. Signal @tickle BELL SPEC_VERIFIED when done.
```

## 2026-03-08 Update — SAT harness progress

- Added `scripts/fm001/ramsey_book_sat.py` to build the unconstrained $K_{4n-2}$ book-Ramsey CNF and invoke Glucose via python-sat. Verified the harness on $n=3$ (witness at `/tmp/fm001-n3.json`); the $n=5$ instance already stresses Glucose, so symmetry reduction is next before pushing to the composite-modulus targets such as $n=23$.
- Updated `data/frontiermath-pilot/FM-001-falsify-plan.md` with the Wesley (2025) and Wigderson–Fox–Conlon references plus a composite-modulus experiment queue.
- Created a local `.venv/` for the solver toolchain; leave it untracked and reuse it whenever running the harness.

```
@zcodex independently verify the FM-001 spec. Pull futon6, read
data/first-proof/frontiermath-pilot/FM-001-ramsey-book-graphs-state.md,
check it against the source. Signal @tickle BELL SPEC_VERIFIED when done.
```

Joe attaches to claude-2 (Mentor) via workspace2 and monitors.

## 2026-03-09 Update — F1-opposite sprint

- Rebuilt the $n=5$ CNF (`[data/frontiermath-pilot/harness/FM001-n5.cnf.gz]`) from `scripts/fm001/ramsey_book_sat.py`. A Glucose4 run via PySAT stalled (>60 s without decision progress), so we are switching to a standalone `kissat-sc2023` build with a 2-hour wall-clock cap to either extract `n5-witness.json` or emit an UNSAT DRAT for ledger logging.
- Generated the next instance $n=6$ (`[data/frontiermath-pilot/harness/FM001-n6.cnf.gz]`, 41 580 vars / 106 280 clauses) with `--no-solve` so we can immediately queue it once the $n=5$ outcome lands.
- Action items: compile/install `kissat` in the shared toolchain, capture SHA512 hashes for both CNFs in `data/frontiermath-pilot/harness/README.md`, and record every solver outcome in the FM-001 strategy/falsification notes under `data/frontiermath-pilot/`.
- Added a cheap vertex-order symmetry breaker (monotone incident edges on vertex 0) to the SAT encoding and regenerated the CNFs / SHA512 fingerprints so `kissat` and future solvers work on the reduced orbit space. After this change, `FM001-n5.cnf.gz` is 48 976 clauses (up from 48 960).
- `kissat 4.0.4 --time=1800 FM001-n5.cnf` finished SAT in 11 s; the log (`[data/frontiermath-pilot/harness/FM001-n5.kissat.log]`) and decoded witness (`[data/frontiermath-pilot/harness/n5-witness.json]`) now live in the harness directory so H-F1(n=5) is marked refuted.

## 2026-08-20 Update — Harness artifacts pinned and one figure corrected

- Wrote the missing `data/frontiermath-pilot/harness/README.md`, discharging the
  2026-03-09 action item to capture SHA512 hashes for both CNFs and record every
  solver outcome. It is the one file `.gitignore` whitelists in that directory,
  and it did not exist.
- The harness artifacts are NOT in the repo: commit `338a2fa` moved them to
  `~/code/storage/futon6/data/frontiermath-pilot/harness/`, so the links in the
  2026-03-09 note do not resolve from a checkout. The new README records the
  real location.
- **Corrected:** that note recorded n=6 as 106 260 clauses. The artifact header
  is `p cnf 41580 106280` and regeneration reproduces 106 280; the figure
  predates the vertex-0 symmetry breaker, which added +16 clauses to n=5
  (48 960 -> 48 976) and +20 to n=6. Only the n=5 figure had been updated.
- Verified rather than restated: both CNFs regenerate **byte-identical** from
  `scripts/fm001/ramsey_book_sat.py`, and the n3/n4/n5 witnesses all pass the
  harness's own `verify_assignment` on complete edge sets — so those refutations
  rest on checked counterexamples, not on solver say-so.
- n=6 remains **open**: `FM001-n6.kissat.log` carries no `s` line, and both n=7
  logs are `s UNKNOWN`. Absence of a verdict is not UNSAT.
- **Solver loglines the ops rule required and never got.** This mission's own
  discipline is "every solver run gets a logline in
  `holes/missions/M-distributed-frontiermath.md` plus SHA512 hashes for
  CNFs/DRATs". Only n=5 ever got one. The full ledger, read off the logs:

  | run | instance | `s` line | process-time | conflicts |
  |---|---|---|---|---|
  | `FM001-n5.kissat.log` | n=5 | `s SATISFIABLE` | 11.06 s | 321 908 |
  | `FM001-n6.kissat.log` | n=6 | none — log ends at 5.14 s | — | — |
  | `FM001-n7.kissat.log` | n=7 | `s UNKNOWN` | 59 m 52 s | 48 298 934 |
  | `FM001-n7.kissat.30m.log` | n=7 | `s UNKNOWN` | 29 m 59 s | 31 627 515 |
  | `FM001b-n8.kissat.log` | FM-001b n=8 | none — log ends at 1502.76 s | — | — |

  An **n=7 instance exists and was solved against twice** (81 900 vars /
  202 824 clauses, regenerates byte-identical); the mission log never mentioned
  n=7 at all.
- **n=6 has never had a real attempt.** Its log stops after 5.14 s with no `s`
  line and no summary — an interrupted run, not a hard instance. Distinguish
  this from n=7, where kissat exhausted a 30- and a 60-minute budget and
  returned `s UNKNOWN`. n=6 is the cheapest open instance and the obvious next
  target.
- Reconciled with the canonical notes: `FM-001-strategy.md` (the primary status
  note per the falsify plan) carried the same stale 106 260 figure and has been
  corrected in place with a dated marker. A canonical harness README also exists
  beside the artifacts (`$STORAGE/README.md`); its six SHA512 claims were
  re-checked and all pass, and the tracked README now defers to it rather than
  duplicating it. `FM-001-falsify-plan.md` carries no
  clause counts. The harness README now holds a table of which figures are
  current; the 48 960 entries are left as written, being accurate dated records
  of the pre-symmetry-breaker CNF rather than errors.

## 2026-03-20 Update — Ownership Boundary

- FrontierMath-specific local bring-up now belongs to `futon6`, not `futon3c`.
- Use `scripts/frontiermath/local-futon3c-windows.bat` from this repo to start
  the local FrontierMath lane on Windows, with `codex` as the invoke bot and
  `tickle` available as the bridge-visible room-page sender.
  - the current Windows owner-side default prefers a sibling
    `futon3c-mfuton-overlay` checkout when present, then falls back to the
    sibling `futon3c` checkout
  - it also defaults the local rehearsal lane to ports
    `7271/7270/7667/7768` unless the operator overrides them
  - it isolates bridge pid/health files and bridge `/say` HTTP under the
    FrontierMath lane via `XDG_RUNTIME_DIR`, `BRIDGE_HTTP_PORT`, and
    `INVOKE_BASE`
- Use `scripts/frontiermath/local-futon3c-linux.sh` from this repo to start
  the analogous local lane on Ubuntu GNU/Linux.
- This wrapper owns:
  - FrontierMath session continuity (`.state/codex-frontiermath-local/`)
  - local room policy (`#futon` primary, `#math` joined via the supported
    `futon3c` `--math-irc` surface)
  - FrontierMath-specific bare `!` ownership defaults when the current
    `futon3c` bridge supports room-owner maps
  - `CODEX_CWD=<futon6-root>` by default so local proof work lands in
    `futon6` rather than scattering into whichever repo booted the runtime
- `futon3c` remains the generic runtime, IRC bridge, and proof-tool substrate.
- Proof-state-root and repo-layout assumptions are still open cross-repo
  design problems. Do not solve them by hard-coding FrontierMath-specific
  filesystem assumptions into `futon3c`; the eventual fix should support
  container-friendly proof-frame execution.
- The proof DAG question is settled as follows:
  - `futon3c`'s obligation DAG remains the mathematical dependency graph
  - `futon6` may add a separate execution-trace graph of proof frame
    receipts, but that graph must only attach to obligation nodes/cycles and
    must not redefine proof dependencies
- Seed implementation:
  - `scripts/frontiermath/proof-frame-receipt.md`
  - `scripts/frontiermath/emit-proof-frame-receipt.py`
  - `scripts/run-proof-stepper.py --proof-problem-id ...`
  - `scripts/frontiermath/advance-proof-cycle-from-frame-receipt.py`
  - `scripts/frontiermath/advance-proof-cycle-from-local-run.py`
    - projects one existing local `mfuton` run bundle into the proof-frame seam
    - requires explicit cycle/blocker anchors until the current run bundle
      shape carries them natively

## Infrastructure Notes

- **Bridge env**: `~/.config/futon3c/bridge-futon.env`
  - `BRIDGE_BOTS=claude,codex,claude-2,tickle`
  - `IRC_CHANNELS=#math`
  - `NICK_AGENT_MAP=claude:claude-1,codex:codex-1,claude-2:claude-2,tickle:tickle-1`
- **Bridge restart**: `systemctl --user restart ngircd-bridge@futon`
- **Post to #math**: `curl -s -X POST http://127.0.0.1:6769/say -H 'Content-Type: application/json' -d '{"from":"claude","channel":"#math","text":"..."}'`
- **Tickle implementation**: `src/futon3c/agents/tickle.clj` (watchdog), `tickle_orchestrate.clj` (conductor), `dev.clj` tickle-lite (bell-driven)
- **State files**: `data/first-proof/frontiermath-pilot/FM-00{1,2,3}-*-state.md`
  - **Local Windows bring-up**: `scripts/frontiermath/local-futon3c-windows.bat`
    - prefers the sibling `../futon3c-mfuton-overlay` checkout when present,
      then falls back to `../futon3c`
    - defaults the local rehearsal lane to ports `7271/7270/7667/7768`
    - defaults bridge runtime isolation to
      `mfuton/data/tmp/frontiermath-local/runtime`, `BRIDGE_HTTP_PORT=7769`,
      and `INVOKE_BASE=http://127.0.0.1:7270`
    - this wrapper now defaults `FUTON3C_PROOF_STATE_ROOT` to
      `mfuton/data/frontiermath-local/FM-001/active` through `MFUTON_ROOT`
      for the current local Windows lane, without changing `futon3c`'s
      generic proof-tooling surface
- **Local Ubuntu/Linux bring-up**: `scripts/frontiermath/local-futon3c-linux.sh`
  - supervises both `make dev` and `scripts/ngircd_bridge.py`
  - defaults `CODEX_CWD` to the `futon6` root to reduce scattered work
  - also intentionally leaves `FUTON3C_PROOF_STATE_ROOT` unset
