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
| **Prover** | codex-1 + zcodex | Linode + Houston | #futon / #zabuton | Dual-core proof engine: construct, compute, write formal arguments |
| **Critic** | claude-1 | Linode | #futon | Falsify claims, find counterexamples, verify proof steps |
| **Mentor** | claude-2 (new session) | Linode | #futon | Meta-level: name patterns, detect loops, redirect strategy |
| **Tickle** | tickle-1 | Linode | #futon | Timekeeper, phase nudges, relay between channels |

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
#zabuton (Rob)              #futon (Joe)
┌──────────────┐            ┌──────────────────────────┐
│ zcodex       │◄──git───►  │ codex-1                  │
│  (Prover B)  │  pushes    │  (Prover A)              │
│              │            │ claude-1 (Critic)         │
│              │◄──tickle──►│ claude-2 (Mentor)         │
│              │  relays    │ tickle-1 (Timekeeper)     │
└──────────────┘            └──────────────────────────┘
                    │
            futon6 repo (shared artifact)
```

- **IRC for handoffs**: "@zcodex pushed proof sketch to fm001-prob branch,
  please check the n≥7 case" — short, actionable.
- **Git for work products**: State files, proof sketches, reviews, lemmas.
- **Tickle relays** between channels when cross-team coordination is needed.

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
4. **Tickle**: Phase nudges every 60 minutes.

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

## Setup Checklist

- [ ] futon6 repo accessible to both codex-1 and zcodex (shared remote)
- [ ] FM-001 state file at `data/first-proof/frontiermath-pilot/FM-001-ramsey-book-graphs-state.md`
- [ ] #futon bridge running (claude-1, codex-1, tickle-1)
- [ ] #zabuton bridge running (zcodex)
- [ ] Mentor session (claude-2) registered and monitoring
- [ ] Rob has IRC access to #zabuton and can observe #futon
- [ ] Joe has IRC access to both channels

## Starting the Mission

On #futon:
```
@claude spec-lock FM-001. Extract the exact problem statement from
the Frontier Math source and fill in the state file. Do not hypothesize.
```

On #zabuton:
```
@zcodex independently verify the FM-001 spec that codex-1 is writing.
Pull futon6, read the state file, check it against the source.
```

The Mentor watches and waits.
