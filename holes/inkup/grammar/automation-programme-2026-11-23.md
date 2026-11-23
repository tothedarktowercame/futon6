# InkUp automation programme: reader anticipation and publication spin

Date: 2026-11-23

Status: executable programme v0; the stages and gates are machine-readable in
[`automation-programme-v0.tsv`](automation-programme-v0.tsv). This programme is
an independent generative/audit lane. It does not change Claude-7's residual
classes, accepted-head process, or readiness claims.

## Objective

Build and compare two transformations that share a supported content plan but
optimize for different purposes:

1. **Reader-question telling:** anticipate questions a prelim student is likely
   to ask and expose the reasoning relations needed to answer them.
2. **Publication spin:** take a supported informal proof and organize its
   scopes, emphasis, and compression using patterns calibrated against
   professional mathematical writing.

The first is primarily a content-planning problem. The second is primarily a
scope and realization problem. They must not be evaluated by one blended
"quality" score.

## Pipeline

```text
prelim statement ── vocabulary/concept retrieval ── candidate reader questions
       │                                                    │
       └── Blueprint + reference solution + formal proof ───┤ support check
                                                            ▼
                                             content presentation plan
                                                            │
                                      reader profile ────────┤
                                                            ▼
                                               reader-oriented telling
                                                            │
                                  mined professional scopes ─┤
                                                            ▼
                                             publication presentation plan
                                                            │
                                                            ▼
                                               publication-style telling
```

The support check is the decisive boundary. Lexical or conceptual overlap may
retrieve a useful question, but it does not make the question relevant to the
current proof. A question enters the telling only when it can be aligned to a
supported node or relation. Otherwise it is retained as an unsupported or
contested diagnostic.

## Lane A: anticipate reader questions

For each prelim problem:

1. Extract mathematical vocabulary, named methods, objects, transformations,
   and the Blueprint's strategic moves.
2. Search Math.SE/MathOverflow using both surface vocabulary and normalized
   mathematical concepts. Retain thread, speaker role, passage, and retrieval
   reason.
3. Map candidate passages to the question functions in
   `question-functions-v0.tsv`. Do not discard the original wording.
4. Ask whether the reference solution, Blueprint, formal proof, or secondary
   reasoning graph supports the requested relation.
5. For a specified reader profile, choose whether to anticipate, defer, or omit
   the question. An anticipated answer becomes a presentation-plan obligation
   with explicit evidence.
6. Generate the proof and a sidecar that says which questions were anticipated,
   where they were answered, and how much ink they consumed.

The first retrieval experiment should use a96J06. It already has a checked
Blueprint alignment and the question "why rescale?" represented as a secondary
reasoning graph. That supplies one positive control: retrieval should rank
questions about normalization, domain conversion, or applicability of Schwarz's
lemma above merely co-occurring questions about polynomials or disks.

## Lane B: publication spin

Publication work starts from the supported informal proof, not directly from
the Lean term:

1. Annotate passages with formal role, communicative function, argument
   relation, and placement scope.
2. Retrieve structurally similar scope configurations from the manually audited
   CT and RH/GRH professional samples. Retrieval is by scope pattern and proof
   position, not necessarily by vocabulary.
3. Select a spin: which supported nodes to show, compress, expand, omit, or nest;
   where to place overview, motivation, auxiliary construction, and closure;
   and what exposition budget to allocate.
4. Realize new prose while preserving the content plan's mandatory nodes and
   dependencies.
5. Compare the two plans before judging the prose. A fluent publication telling
   that loses a dependency is a failed transformation.

Mark7/8 mining proposes scope descriptions and exemplars; it does not itself
certify good style. Initial calibration uses the audited CT fixture and a small
manual sample from the retained 100-paper RH/GRH run. Broader mined material is
admitted only after anchor precision and duplication are measured.

## Artifacts per problem

Each run should produce:

- `question-candidates.jsonl`: retrieved wording and provenance;
- `question-alignments.jsonl`: function, requested relation, support anchors,
  review state, and disposition;
- `content-plan.json`: mandatory proof nodes plus anticipated questions;
- `reader-telling.md` and its node/ink sidecar;
- `publication-annotations.jsonl`: passage scopes and graph anchors;
- `publication-plan.json`: the professional spin;
- `publication-telling.md` and its node/ink sidecar;
- `comparison.tsv`: coverage and ink deltas by question function,
  communicative function, formal role, and scope.

Rejected and contested candidates remain append-only evidence. Generated prose
is never the sole record of why a passage exists.

## Evaluation and stopping rules

The reader lane reports retrieval precision, question-function agreement,
supported/contested/unsupported questions, anticipated-question coverage,
unsupported generated claims, and ink by question function.

The publication lane reports annotation precision, agreement on scope labels,
mandatory-node preservation, lost dependencies, scope transitions, and ink by
communicative function. Human preference can be collected only after these
structural checks; preference does not repair an unsupported claim.

A first pilot is complete enough to expand when three prelim proofs satisfy all
of the following: every anticipated answer has a support anchor; independent
review finds no invented mathematical claim; both tellings account for all
mandatory nodes; and at least one audience-dependent realization difference is
explained by the plans rather than post-hoc prose commentary.

## Initial automated work queue

1. **a96J06 positive control:** implement retrieval and alignment around "why
   rescale?", then generate the two sidecars without regenerating prose.
2. **Two contrast problems:** choose one elementary proof with little strategic
   machinery and one proof with a case split or auxiliary construction.
3. **Professional retrieval baseline:** index audited Mark7/8 passages by
   communicative function, argument relation, placement, and local scope shape.
4. **Paired generation:** produce student/prelim and publication profiles from
   the same supported content plan.
5. **Falsification review:** deliberately retrieve high-vocabulary-overlap but
   irrelevant questions and structurally similar but mathematically unrelated
   professional passages; verify that the two gates reject them.

These jobs are read-heavy and can run alongside residual cleanup. Their only
shared dependency is an immutable accepted proof/render identifier when an APM
artifact is used; they do not need to occupy Claude-7's integration lane.
