# FrontierMath Pre-Season Mission

Date: 2026-02-20  
Scope: Exercise First Proof 2 protocol on public open problems before next challenge batch.

## Objective

Stress-test the readiness protocol on hard open problems where failure is acceptable, but process quality is non-negotiable.

Success condition for this mission is not "solve everything."  
Success condition is: high-integrity execution under uncertainty.

## Pilot Set

Run a 3-problem pilot from `https://epoch.ai/frontiermath/open-problems`:

1. Ramsey Numbers for Book Graphs
2. A Ramsey-style Problem on Hypergraphs
3. Large Steiner Systems

## Hard Constraints

For each problem:

1. Pass `Spec-Lock` before proof drafting.
2. Run one full opposite-answer (`FALSIFY`) cycle.
3. Use TryHarder only with a signed license (target, new lever, witness, kill condition).
4. End with one status only:
   - `CLOSES`, `NARROWS`, `FAILS`, or `OPEN-LEMMA-LOCALIZED`.

## Mission Scorecard

Track:

1. `spec_substitution_incidents` (target: 0)
2. `no_unlicensed_tryharder_events` (target: 0)
3. `time_to_first_falsification` per problem
4. `closure_grade_count`
5. `named_obstruction_count`

## Dignity Rule

Failure is acceptable when it yields one of:

1. A named obstruction with evidence.
2. A reduced subproblem with explicit open lemma.
3. A falsified route with reusable negative result.

Unacceptable failure:

1. Spec drift.
2. Repeated TryHarder loops without new levers.
3. "Looks plausible" closure claims without gate completion.

## Run Plan (48-hour pilot)

1. Day 1 AM: Spec-Lock + FALSIFY on all 3 problems.
2. Day 1 PM: CONSTRUCT on top two highest-leverage routes.
3. Day 2 AM: VERIFY + dependency audit.
4. Day 2 PM: publish scorecard + go/no-go recommendation for larger run.

## Artifacts

Use:

1. `data/first-proof/frontiermath-pilot/problem-state-template.md`
2. `data/first-proof/frontiermath-pilot/tryharder-license-template.md`
3. Problem state files under `data/first-proof/frontiermath-pilot/`

