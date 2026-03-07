# FM-003 Problem State

## Metadata

- `problem_id`: FM-003
- `title`: Large Steiner Systems
- `source_url`: https://epoch.ai/frontiermath/open-problems
- `owner`: Joe + Codex
- `started_utc`: 2026-02-20T00:00:00Z

## Spec-Lock

- `formal_statement`: |
    An (n,q,r)-Steiner system S is a collection of q-subsets of [n] = {1,...,n}
    such that every r-subset of [n] is contained in exactly 1 element of S.

    **Problem**: Construct an (n,q,r)-Steiner system with n > q > r > 5, r < 10,
    and n < 200.

    No known examples exist with r > 5 despite theoretical existence proofs (Keevash 2014).
- `quantifiers`: "Existential — find any single (n,q,r) triple satisfying the constraints and construct the system."
- `parameter_regime`: "n < 200, r in {6,7,8,9}, q in (r, n)."
- `output_format`: "Multiline string. Line 1: '#n,q,r'. Subsequent lines: elements of each q-subset separated by whitespace."
- `forbidden_substitutions`: "Do not replace with generic design-theory existence question. Do not weaken r > 5 to r >= 5. Do not claim existence without explicit construction."
- `spec_lock_status`: `pass`

## Current Hypothesis

- `answer_hypothesis`: `unknown`
- `confidence`: `low`

## Risk Flags

- `spec_risk`: low (spec-locked)
- `no_bias_risk`: medium
- `domain_depth_risk`: high
- `verification_risk`: high

## Mode

- `current_mode`: `SPEC`
- `mode_transition_reason`: "Initial state; statement not normalized."

## Opposite-Answer Cycle

- `opposite_answer_attempted`: `no`
- `falsification_artifact`: pending
- `result`: pending

## Open Lemmas

1. pending
2. pending
3. pending

## Verification

- `symbolic_checks`: pending
- `computational_checks`: pending
- `adversarial_checks`: pending
- `dependency_audit`: pending

## TryHarder Log

1. none yet

## Final Status

- `status`: `NARROWS`
- `final_note`: "Initialized; awaiting Spec-Lock."
- `next_action`: "Begin FALSIFY phase — attempt the opposite answer."

