# a96J06 — Blueprint lineage and exposition pilot

Status: first human-reviewed alignment; no presentation-quality claim

This pilot asks whether one mathematical proof can be followed from an informal
plan through formal implementation, mined IATC structure, and several intended
presentations.  It does not treat the formal proof as exposition and does not
treat model-produced mining output as authoritative.

## Result

`alignment.json` gives the proof a seven-node spine:

1. state the radius-expansion goal;
2. open the contradiction for a fixed radius;
3. normalize the disk;
4. establish the equality-case hypotheses;
5. apply Schwarz rigidity;
6. pass from identity on the disk to polynomial and coefficient identity;
7. close the contradiction and restore the universal radius.

Each node records argument roles, expository roles, reader obligations, and
anchors into the existing `proof-plan.edn`, informal solution, compiled Lean
proof, and Mark8-contract S3 graph.  The source hashes make drift visible.

The Mark8 graph is useful but not clean ground truth.  In particular, its node
`:n13` glosses “Every higher-order coefficient must vanish” while its text and
quote span repeat the rotation formula.  The alignment retains that anchor as
`rejected`, making the mining error part of the calibration evidence.

## Three controlled profiles

- **Expert reference** keeps the strategy hinge and coefficient bridge but
  compresses the routine analytic side conditions.
- **Teaching** shows the whole spine and spends its explanation budget on why
  scaling permits Schwarz rigidity and why derivative equality matters.
- **Formal audit** exposes every side condition and the Lean-only bridge from
  equality on an infinite set to polynomial equality.

These are presentation plans, not three claims that the proof has already been
rewritten well.  `realizations.md` contains the first candidate prose produced
from each profile, followed by six concrete reader tasks.  The node labels are
evaluation markup that makes omissions and misplaced details countable.

`generative-ink.tsv` makes the positive presentation choices countable.  For
each profile it assigns every Blueprint node an action (`show`, `compress`, or
`expand`) and a percentage of the profile's exposition budget.  These
percentages describe where explanation is spent, not how much of the proof is
correct.  They therefore remain separate from PaperPrinter residual ink.

## Validation

Run:

```sh
python3 holes/inkup/pilots/a96J06/validate.py
```

The validator checks source hashes, node and profile references, dependency
acyclicity, reader obligations, source names, and review states using only the
Python standard library.
