# Positive grammar of proof presentation — evidence slice v0

Status: candidate factorization grounded in one Mark7 professional-paper
fixture and one APM alternate-telling pilot; not yet a quality standard.

## Finding

`show`, `compress`, `expand`, and `omit` are **realization actions**.  Mark7's
scope vocabulary mostly describes different things: what a passage is doing,
where that function is scoped, and how claims are related.  The positive
grammar should therefore be factored rather than enlarged into one flat list.

| Axis | Question answered | Initial values |
|---|---|---|
| realization | How much of this node is told? | show, compress, expand, omit |
| communicative function | What does this passage do for a reader? | connection, characterization, auxiliary construction, difficulty assessment, generalisation, open-problem status |
| argument relation | How does this claim support another? | implies, iff, by-cases |
| placement | Over what textual/proof region does it operate? | section lead, inflight passage, in-proof, leaf section |
| formal role | What proof-plan obligation anchors it? | goal, assumption, normalization, side-condition, lemma application, identity, contradiction, closure |
| profile | For whom and for what purpose is it told? | audience, purpose, mandatory nodes, exposition budget |
| evidence | Why may the annotation be trusted? | source anchor, producer, review state, accepted/rejected |

This factorization permits, for example, the same formal lemma application to
be `expand` + `characterization` + `in-proof` for a student, but `compress` +
`characterization` + `in-proof` for an expert.  Changing the spin changes the
realization and budget without pretending that the mathematical function has
changed.

## Mark7 evidence: 0708.2185

The checked fixture contains 3,552 spans and 85 proof-graph edges.  Its S4
layer contains 163 expository spans:

| S4 kind | Count |
|---|---:|
| connection | 112 |
| universal-property/characterizes | 12 |
| auxiliary-construction | 4 |
| carved/section-lead | 4 |
| open-problem/status | 3 |
| difficulty-assessment | 3 |
| generalisation | 2 |
| carved/inflight | 21 |
| carved/in-proof | 1 |
| carved/leaf-section | 1 |

The 85 S3 proof-graph relations split into `iff` (45), `implies` (30), and
`by-cases` (10).  These relations should remain distinct from expository
function: an implication can be introduced as a connection, compressed as a
routine step, or expanded as the proof's strategic hinge.

Counts are regenerated from
`futon3c/web/scopes/fixtures/mark7-0708.2185.json` by `inventory_mark7.py` and
frozen with the fixture hash in `mark7-0708.2185-inventory.json`.

## Provisional bridge to a96J06

These are hypotheses to test, not asserted equivalences.

| Blueprint node | Existing APM role | Mark7-informed annotation to test |
|---|---|---|
| B0 | overview/goal | placement: section-lead; function: connection to the non-linearity hypothesis |
| B2 | motivation/normalization/bridge | function: connection; explain why rescaling makes the rigidity result applicable |
| B3 | side-condition/calculation | function: auxiliary-construction when the normalized map is treated as the helper object |
| B4 | lemma application/identity | function: characterization; relation: implication from equality-case hypotheses to identity |
| B5 | identity/contradiction bridge | function: connection; relation: implication from disk identity to polynomial identity to coefficient equality |
| B6 | closure | placement: in-proof; relation: implication closing the contradiction and restoring the quantified goal |

## Next falsifier

Apply the factored annotations independently to several APM reference
solutions and several Mark7 papers.  Reject or split a category when annotators
cannot agree on its question, when it merely renames a formal role, or when it
cannot predict a different realization for two reader profiles.  Preserve
contested readings rather than forcing consensus.
