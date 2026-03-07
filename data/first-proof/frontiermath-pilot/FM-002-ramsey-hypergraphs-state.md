# FM-002 Problem State

## Metadata

- `problem_id`: FM-002
- `title`: A Ramsey-style Problem on Hypergraphs
- `source_url`: https://epoch.ai/frontiermath/open-problems
- `owner`: Joe + Codex
- `started_utc`: 2026-02-20T00:00:00Z

## Spec-Lock

- `formal_statement`: |
    A hypergraph (V, H) contains a **partition of size n** if there exist D ⊆ V
    and P ⊆ H such that |D| = n and every member of D is contained in exactly
    one member of P.

    H(n) = greatest k such that there exists a hypergraph (V, H) with |V| = k,
    no isolated vertices, and no partitions of size > n.

    **Warm-up**: Construct a hypergraph with |V| >= 64, |H| <= 20, no partitions
    of size > 20, no isolated vertices.

    **Single**: Same but |V| >= 66.

    **Full Problem**: Improve the known lower bound H(n) >= k_n where k_1 = 1
    and k_n = floor(n/2) + k_{floor(n/2)} + k_{floor((n+1)/2)}.
    Demonstrate H(n) >= c * k_n for some c > 1, effective by n=15.
- `quantifiers`: "For all n (full problem). Fixed n=20 (warm-up/single)."
- `parameter_regime`: "n <= 100, must complete within 10 minutes on standard hardware."
- `output_format`: "Python function `solution(n: int) -> str` producing hypergraphs as edge strings, e.g. '{1,2,3},{2,4},{3,4,5}'."
- `forbidden_substitutions`: "Do not replace with easier graph-only Ramsey variant. Do not weaken the partition definition."
- `spec_lock_status`: `pass`

## Current Hypothesis

- `answer_hypothesis`: `unknown`
- `confidence`: `low`

## Risk Flags

- `spec_risk`: low (spec-locked)
- `no_bias_risk`: medium
- `domain_depth_risk`: high
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

