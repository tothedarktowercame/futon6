# FM-001 Problem State

## Metadata

- `problem_id`: FM-001
- `title`: Ramsey Numbers for Book Graphs
- `source_url`: https://epoch.ai/frontiermath/open-problems
- `owner`: Joe + Codex
- `started_utc`: 2026-02-20T00:00:00Z

## Spec-Lock

- `formal_statement`: |
    Let B_n denote a triangular book graph (n triangular "pages" sharing a common edge).
    The upper bound R(B_{n-1}, B_n) <= 4n - 1 was established in 1978.

    **Full Problem**: Develop an algorithm accepting n as input that produces a graph
    on 4n - 2 vertices such that the graph contains no copy of B_{n-1} and the
    complement contains no copy of B_n.

    **Warm-up (n=25)**: Construct a graph on 98 vertices avoiding B_24 whose
    complement avoids B_25.

    **Single (n=50)**: Construct a graph on 198 vertices avoiding B_49 whose
    complement avoids B_50.
- `quantifiers`: "For all n (full problem). Fixed n=25 (warm-up), n=50 (single)."
- `parameter_regime`: "n <= 100, must complete within 10 minutes on standard hardware."
- `output_format`: "Python function `solution(n: int) -> str` returning adjacency string (binary sequence listing edges in column-major order, zero-indexed)."
- `forbidden_substitutions`: "Do not replace with related non-book-graph Ramsey variant. Do not weaken to asymptotic bounds. Do not substitute R(B_n, B_n) for the off-diagonal R(B_{n-1}, B_n)."
- `spec_lock_status`: `pass`

## Current Hypothesis

- `answer_hypothesis`: `unknown`
- `confidence`: `low`

## Risk Flags

- `spec_risk`: low (spec-locked)
- `no_bias_risk`: medium
- `domain_depth_risk`: medium
- `verification_risk`: medium

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

