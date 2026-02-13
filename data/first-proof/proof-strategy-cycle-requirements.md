# Proof Strategy And Cycle Requirements (P4/P6)

Date: 2026-02-13
Scope: Problem 4 and Problem 6 proof-development workflow.
Purpose: Convert the AIF+ audit into enforceable requirements for an improved proof strategy and execution cycle.

---

## 1) Strategy Requirements (SR)

### SR-1 Boundary Lock (MANDATORY)
- Requirement:
  - Each problem must maintain a single canonical theorem statement and a single canonical closure criterion.
  - Any wording change must be tracked as a versioned diff, not silently replaced.
- Pass condition:
  - Canonical statement appears once in the problem solution file.
  - `status` language matches the canonical closure criterion.

### SR-2 Explicit Gap Ledger (MANDATORY)
- Requirement:
  - Every unresolved point must be a named lemma/bridge with status in `{proved, partial, open, false}`.
  - "Gap" text without a named item is forbidden.
- Pass condition:
  - No untagged "open/gap/remains" text in final writeups.
  - All blocking items appear in a dedicated ledger section.

### SR-3 Dependency DAG (MANDATORY)
- Requirement:
  - Each ledger item must list `depends_on` and `unlocks`.
  - Proof claims cannot be marked closed if upstream dependencies are open.
- Pass condition:
  - Dependency links are present and acyclic for blocking claims.

### SR-4 Counterexample-First Testing (MANDATORY)
- Requirement:
  - Any new conjectural lemma must include a falsification attempt before escalation to proof draft.
- Pass condition:
  - Cycle record contains one "counterexample attempt" artifact and outcome.

### SR-5 Proof/Evidence Separation (MANDATORY)
- Requirement:
  - Empirical verification can prioritize routes but cannot close theorem-level claims.
  - Closure labels must distinguish `proved` from `numerically verified`.
- Pass condition:
  - No theorem marked "proved" with only empirical support.

### SR-6 Route Selection By Blocker Impact (MANDATORY)
- Requirement:
  - The next route must target the highest-impact open blocker in the dependency DAG.
- Pass condition:
  - Cycle record identifies the chosen blocker and why it dominates alternatives.

### SR-7 Artifact Minimum (MANDATORY)
- Requirement:
  - Each cycle must emit all three:
    1. theorem text delta (or explicit no-delta),
    2. verifier output (script + result),
    3. short rationale linking result to ledger status change.
- Pass condition:
  - Missing any one artifact invalidates the cycle.

### SR-8 Honesty Invariant (MANDATORY)
- Requirement:
  - Failed routes must remain documented with exact failure point.
  - No deletion of failed-path evidence unless superseded by a direct correction note.
- Pass condition:
  - Every `false`/failed item has preserved evidence and citation.

---

## 2) Cycle Requirements (CR)

Each proof cycle must execute in this order and produce the listed outputs.

### CR-1 Observe
- Input:
  - current gap ledger + dependency DAG
- Output:
  - one selected blocking target ID

### CR-2 Propose
- Input:
  - selected blocker
- Output:
  - one route hypothesis, explicit stop conditions, expected failure modes

### CR-3 Execute
- Input:
  - route hypothesis
- Output:
  - one concrete lemma attempt/reduction/proof fragment (single focus)

### CR-4 Validate
- Input:
  - produced fragment
- Output:
  - symbolic and/or numerical validation artifact tied to that fragment

### CR-5 Classify
- Input:
  - validation result
- Output:
  - status update in `{proved, partial, open, false}` with reason

### CR-6 Integrate
- Input:
  - classification
- Output:
  - updates to solution text, ledger, and dependency links

### CR-7 Commit
- Input:
  - integrated artifacts
- Output:
  - one commit with cycle ID and blocker ID in message

### CR-8 Gate Review
- Input:
  - cycle outputs
- Output:
  - pass/fail determination for cycle validity (invalid cycles cannot advance status)

---

## 3) Gate Checklist (G5-G0 Mapping)

### G5 Task Specification Gate
- Must include:
  - canonical theorem statement,
  - canonical closure criterion,
  - blocker ID selected for this cycle.

### G4 Capability/Assignment Gate
- Must include:
  - named agent/owner for the cycle,
  - stated method family (e.g., A-route, E/F route, interlacing route).

### G3 Pattern Reference Gate
- Must include:
  - explicit reference to prior pattern or rationale for "new pattern".

### G2 Execution Gate
- Must include:
  - one focused lemma attempt and proof/derivation artifact.

### G1 Validation Gate
- Must include:
  - concrete verifier output and interpretation.

### G0 Evidence Durability Gate
- Must include:
  - durable file paths for all artifacts,
  - commit hash for the cycle.

---

## 4) Status Policy

Use only these labels:
- `proved`: theorem-level closure under stated assumptions, with proof artifact.
- `partial`: nontrivial progress that closes a subcase or reduction only.
- `open`: unresolved blocker.
- `false`: disproved claim/lemma with counterexample.
- `numerically verified`: empirical support only (never substitutes for `proved`).

Forbidden:
- ambiguous labels such as "basically solved", "should work", or "almost closed" in the formal status table.

---

## 5) Required Cycle Record Template

Each cycle must append a record matching this schema:

```text
cycle_id:
problem_id: P4|P6
blocker_id:
hypothesis:
stop_conditions:
execution_artifact_paths:
validation_artifact_paths:
result_status: proved|partial|open|false|numerically verified
status_change:
failure_point: (required if partial/open/false)
next_blocker:
commit_hash:
```

---

## 6) Acceptance Criteria For Release Candidate

A problem may be marked release-candidate only if all hold:
1. No unnamed open gaps.
2. All blockers are represented in the dependency DAG.
3. Every `proved` claim has a proof artifact and validation artifact.
4. Any remaining unresolved items are explicitly out-of-scope and non-blocking.
5. Status table is mechanically derivable from the ledger labels.

---

## 7) Immediate Adoption Plan

1. Add a ledger block to `problem4-solution.md` and `problem6-solution.md` using the status policy above.
2. Add cycle records to a durable log file (one entry per commit).
3. Use this requirements file as the gate reference for future proof cycles.
