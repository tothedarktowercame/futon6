# Handoff: Question-Asking as Reverse Morphogenesis

## Source

This handoff comes from a futon3c session where we implemented the evidence
landscape (M-forum-refactor Parts I-IV) and then connected the implementation
to new axioms extending the futonic logic vocabulary.

The new axioms are formalized in:
`~/code/futon3/library/futon-theory/reverse-morphogenesis.flexiarg`

The core claim: **asking a good question is reverse morphogenesis**.

## The (← 象 香) Relation

From the futonic vocabulary:
- **象** (xiàng): a stabilized form, configuration, structured whole
- **香** (xiāng): embodied salience, a pre-individual "difference field" that
  signals without full formalization
- **←**: speculative history operator — reads backwards from stabilized form to
  the constraints that made it stable

**← applied to question-asking in mathematics:**

Given a mathematical form (象) and a sense of what understanding would be
valuable (香), the **question** is the constraint that reverse morphogenesis
infers: what would I need to know/prove/construct such that this form yields
that understanding?

```
(reverse-morphogenesis mathematical-object understanding)
  → question (the inferred constraint)
```

## Three Failure Modes of Bad Questions

1. **Wrong 象**: the form is not well-specified
   - Asking about "infinity" without saying which infinity
   - Asking about "convergence" without specifying the topology

2. **Wrong 香**: the salience signal is not grounded
   - Asking "why is this important?" without a context that makes importance
     detectable
   - Asking "what's interesting about this?" without a decomposition regime
     (部) that makes "interesting" actionable

3. **Wrong ←**: the constraint inference is broken
   - Asking a question whose answer would not yield the intended understanding
   - The form and the salience are both fine, but the question connects them
     wrongly

## Connection to the Flexiformal Pathway

The EXPLORE → ASSIMILATE → CANALIZE pathway maps onto ←:

| Phase | ← operation | Question quality |
|-------|-------------|------------------|
| EXPLORE | Detect 香 (sense what matters) | Pre-question: "something is interesting here" |
| ASSIMILATE | Apply ← (infer constraints from form + salience) | Question formulated: "what makes X stable under Y?" |
| CANALIZE | Stabilize 象 (constraints become structural) | Question answered: understanding is now structural, not effortful |

## Proposed Work for Futon6

### Task 1: Classify P0 patterns by ← structure

The 25 math-informal patterns in `data/pattern-tags.edn` already tag
PlanetMath entries. For each pattern, identify:

- What is the **象** (the mathematical form the pattern operates on)?
- What is the **香** (the salience signal the pattern detects)?
- What is the **←** reading (what constraint does the pattern infer)?

This produces a table mapping patterns to their ← structure, which tests
whether the axiom actually captures what the patterns do.

```
:in (READ-ONLY):
  data/pattern-tags.edn
  ~/code/futon3/library/futon-theory/reverse-morphogenesis.flexiarg
  ~/code/futon3/library/futon-theory/futonic-logic.flexiarg

:out (CREATE):
  data/pattern-reverse-morphogenesis.edn  — pattern → {象, 香, ←} mapping
```

### Task 2: Question quality assessment on physics.SE

The 114K physics.SE QA pairs (P7) provide a corpus of real questions. Sample
50 questions across quality levels (high-voted, low-voted, closed) and
classify each by failure mode:

- Well-formed ←: 象, 香, and constraint inference all correct
- Wrong 象: form under-specified
- Wrong 香: salience not grounded
- Wrong ←: question doesn't connect form to understanding

This tests whether the three failure modes are exhaustive and whether they
predict question quality (vote count, closure status).

```
:in (READ-ONLY):
  data/se-physics.json
  ~/code/futon3/library/futon-theory/reverse-morphogenesis.flexiarg

:out (CREATE):
  data/question-quality-sample.edn  — 50 annotated questions with ← analysis
```

### Task 3: Circle ← Group as worked example

The axiom includes a mathematical example: circle ← group. The group axioms
(closure, composition, identity, inverse) are the constraint-schema that ←
infers when you ask "what makes circular return stable?"

Write this up as a futon6 dictionary entry that demonstrates the non-Platonic
reading: the dictionary entry is not "what is a group?" but "what constraints
make circular return stable, and why does that matter?"

This tests whether the ← framing produces better dictionary entries than the
standard definitional approach.

```
:in (READ-ONLY):
  ~/code/futon3/library/futon-theory/reverse-morphogenesis.flexiarg

:out (CREATE):
  resources/entries/circle-group-example.md  — worked example of ← reading
```

## References

- `~/code/futon3/library/futon-theory/futonic-logic.flexiarg` — base vocabulary
- `~/code/futon3/library/futon-theory/reverse-morphogenesis.flexiarg` — new axioms
- `~/code/futon3c/src/futon3c/evidence/threads.clj` — `thread-patterns` as ← in code
- `~/code/futon3c/src/futon3c/social/validate.clj` — `validate-outcome` as ← assessment
- Corneli (2014) Table 24 — five dimensions as 部 for reading ← across mathematical practice
