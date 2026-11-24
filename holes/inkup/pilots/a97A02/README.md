# a97A02: exemplar of the InkUp style built from a reader dialogue

Status: one worked example, written and read by model seats and reviewed by claude-12
(2026-11-24). No claim that the style is validated.

The problem: for f in L^p([0,1]), 1 ≤ p < ∞, the sliding average f_h is continuous and
|f_h(t)| ≤ (2h)^(-1/p) ‖f‖_p.

## How it was made

1. **Reader dialogue** (futon6 `holes/M-reader-clear.md`). A reader compared the
   PaperPrinter render with the informal solution and asked four questions; a seat holding
   the Lean answered; the reader restated.
2. **Content plan** (`content-plan.json`, `alignment.json`, five graphs under
   `holes/inkup/reasoning/a97A02-*.json`). Nine nodes following the Lean proof. Five
   questions: the reader's four, restated as questions about the mathematics, plus one
   that nobody asked (why this constant). Per-reader decisions for each. Lengths predicted
   before any telling was written.
3. **Four tellings** from the plan: teaching, expert reference, formal audit, publication,
   each with a sidecar tying paragraphs to nodes and question answers to quoted passages.
4. **Two readers** given only the problem statement and the teaching telling.
5. **Revision** of the teaching telling from what the readers said.

Steps 2 to 4 were done by Zai seats, each step by a different seat; review and step 5 by
claude-12.

## What the questions did to the text

| Reader's question | Where the teaching telling answers it |
|---|---|
| What does the integral mean for t near an endpoint? | B1: restricted measure is the zero extension; near an endpoint the effective window is shorter. |
| How does continuity follow from f in L^p? | B2 then B4: integrable first, then the primitive F is continuous. |
| The informal proof's symmetric-difference measure 2\|s−t\| is wrong for \|s−t\| > 2h | B4: gives 2 min(\|s−t\|, 2h), says the estimate works for p > 1, and that its p = 1 fallback already proves every case. |
| Is t ∈ [0,1] needed in the bound? | B7: no; the estimate holds for every real t. |
| (added) Why (2h)^(-1/p)? | B6: it is the conjugate-exponent norm of the normalised window indicator. |

The informal solution in apm-lean answers none of the first, third or fourth, and has the
third wrong. The Lean proof takes a different route to continuity from the informal one;
the tellings follow the Lean route and use the informal route only as the contrast.

## What the two readers said

Both marked all nine paragraphs justified, said they could write the proof without the
text, and said the text explains why the proof goes as it does, pointing to B4, B6 and B1.

Both raised the same point: the first version of B4 said the primitive is continuous
"because the restricted Lebesgue measure has no atoms". That is the hypothesis of the Lean
lemma, not the reason a student would give; a grader expects absolute continuity of the
integral by name. One also asked whether the primitive route is preferred only for p = 1.
B4 and B5 were rewritten accordingly (the formal-audit telling keeps the atomless
phrasing, where it belongs). The cold reader then re-read the two rewritten paragraphs: both questions answered, the
argument for p > 1 and p = 1 checked by hand, the new endpoint sentence in B5 judged
correct and needed, nothing made worse.

One of the two readers had asked the original four questions and so was not reading cold.
The readers' files are in the storage area under
`futon6/inkup/a97A02/20261124T-exemplar-read/`.

## Lengths: the pilot's rule did not hold

Rule from the three-problem pilot: words = nodes × base + questions × increment
(teaching 24/23, expert 16/21, audit 22/21, publication 14/14).

| Telling | Nodes | Questions | Predicted | Words | Words with each formula counted as one |
|---|---|---|---|---|---|
| teaching (revised) | 9 | 5 | 331 | 718 | 512 |
| formal audit | 9 | 5 | 303 | 506 | 382 |
| publication | 7 + 2 merged | 4 in passing | 126 | 165 | 112 |
| expert reference | 6 | 0 | 96 | 133 | 91 |

Part of the excess is counting: these tellings write `\( f \)` with spaces, so one formula
counts as several words, where the pilot's wrote `\(p\)`. With each formula counted once,
the two tellings that answer no questions separately come out close to the rule (91 against
96, 112 against 126), and the two that answer five questions are still well over (512
against 331, 382 against 303). On this problem a question answered in the teaching
telling cost about 60 words, not 23. The publication telling was trimmed once, from 191
to 165 words, to meet that profile's 168-word limit; nothing else was fitted to a length.

## Limits

One problem. Writers and readers from one model family. The plan's "why this route"
question came from the Lean file's own comment and the reader's erratum, not from a
reader who had seen only the statement.
