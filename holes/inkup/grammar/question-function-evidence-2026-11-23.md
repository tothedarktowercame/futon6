# Question functions as an InkUp calibration axis

Date: 2026-11-23

This note adds a **question-function axis** to the positive grammar of proof
presentation. It is deliberately distinct from the prose calibration corpus:

- professional CT and RH/GRH papers calibrate how mature exposition is written;
- Math.SE and MathOverflow supply questions that readers actually articulate;
- proof-reading activity supplies process questions which are often absent from
  both the formal proof and the finished paper.

The first machine-readable inventory is
[`question-functions-v0.tsv`](question-functions-v0.tsv). Each row identifies a
question form, the missing relation it requests, and the presentation response
that could discharge it. A question therefore becomes a typed diagnostic over a
secondary reasoning graph, rather than an instruction to generate an
unconstrained explanatory sentence.

## Evidence checked in this slice

`scripts/iatc_mose_scan.py` was rerun over the four retained sample files in
`futon5/data/stackexchange-samples/`: 200 threads, comprising 200 questions, 313
answers and 1,224 comments. The run regenerated
`/tmp/iatc-mose-examples.json`; it was used as transient inspection output and is
not a corpus artifact.

The sample confirms that dialogue vocabulary cannot be recognized reliably by
the arXiv-oriented cue list alone. Strict cues found agreement in 2.0% of
threads, challenge in 3.0%, and retraction in 0.5%; MO/SE-native phrasings found
29.0%, 7.5%, and 5.5%, respectively. The observed wording must therefore travel
with provenance, and any normalized question function is an interpretation of
it rather than a replacement for it.

The larger earlier survey remains useful for discovery, not as a validated
distribution. The recorded source files contain 95,321 MathOverflow situations
and 805,200 Math.SE situations. Its keyword classes include existence,
connection, surprise, generalization, classification, counterexample, failure,
computability, conjecture, sharpness, analogy, and obstruction questions. The
survey's own caveats remain binding: keyword classification was crude, its
`situation` fields were LLM-generated from real questions, parse coverage was
low, and no trajectory-level validation was performed.

## Consequence for the grammar

Question functions and presentation actions are orthogonal:

| Missing relation | Typical question | Possible presentation action |
|---|---|---|
| `strategy-enables-goal` | Why rescale? | expand the strategy node; show the obstacle and unlocked method |
| `claim-suffices-for-goal` | Why is this enough? | show the reduction, then compress its routine discharge |
| `claim-depends-on-scope` | Where did this assumption enter? | nest the claim beneath its owning assumption |
| `result-extends-under-variation` | Does this generalize? | show the hypothesis boundary and alternative case |
| `local-steps-implement-strategy` | What is the main idea? | compress the steps under a named overview |

Thus `show`, `compress`, `expand`, `omit`, and `nest` remain realization
choices. The question function says *which relation a reader lacks*; audience
and telling profile determine how much ink to spend answering it.

## Calibration protocol

1. Mine question candidates while retaining source text, site, thread, speaker
   role and local dialogue context.
2. Map a candidate to a question function only when the requested relation can
   be named; retain contested mappings rather than forcing one label.
3. Check whether the formal or secondary reasoning graph contains that
   relation. Absence is a planning diagnostic, not permission to invent it.
4. Compare the chosen presentation action with professional-paper scopes.
5. Measure coverage separately: question evidence, graph support, and rendered
   ink must never be collapsed into one confidence number.

This gives the earlier Q&A material a precise role. It expands the grammar's
space of reader needs, while Mark 7/8 and professional-paper data remain the
stronger evidence for scope, placement, and mature exposition.
