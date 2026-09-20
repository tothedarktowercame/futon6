# The accretion curve measures its own parameter

Joe, 2026-09-20, on finding the 40-cue cap: *"why would we disable one of the key
things this script is supposed to achieve, which is modelling the entire
vocabulary in mathematical domains?"*

Three separate defects. Fixing any one alone does not make the curve readable.

## 1. The cap manufactured the convergence the docstring claims to observe

`accretion_curves.py` says the curve rises *"until the move vocabulary saturates
= convergence"*. `cluster_cues` defaulted to `top=40`. Recomputed over the
probe's 121 graphs:

| n | cues@40 | score@40 | cues uncapped | score uncapped |
|---|---|---|---|---|
| 10 | 21 | 0.1840 | 21 | 0.1840 |
| 30 | 40 ← binds | 0.2410 | 81 | 0.2820 |
| 100 | 40 | 0.2760 | 202 | 0.3610 |
| 121 | 40 | **0.2760** | 251 | **0.4010** |

The cap binds from n≈30. Every later point tracks a constant, so the reported
plateau is the parameter. The vocabulary never saturated: it reaches 251 and is
still growing.

**This is why the 0.114→0.275 rise must not be cited as accretion evidence, and
why "cues saturating at 40" is not evidence of anything.** Both were stated to
the mfuton side on 2026-09-20 and corrected the same day.

## 2. The metric responds to cue count, not to the corpus

Corpus held **fixed** at 121 graphs, varying only the cap:

| cap | 10 | 20 | 40 | 80 | 160 | 251 |
|---|---|---|---|---|---|---|
| score | 0.194 | 0.235 | 0.276 | 0.326 | 0.384 | 0.401 |

Corpus size never changes and the score still climbs. More cues match more words
in the same fixed windows, and those windows are the ones the cues were harvested
from. So **raising the cap does not fix the curve** — it moves the number up for
the same bad reason. Only a paper-disjoint split with the cue count held constant
can separate accretion from cue inflation. That harness is `accretion_heldout.py`.

## 3. The tokenizer read LaTeX command names as mathematical language

`re.findall(r"[a-z]{5,}", phrase)` over source text harvests `colon` from
`\colon`, `tilde` from `\tilde`, `rightarrow` from `\rightarrow`. Commands now
get stripped before tokenising.

The cap was the de-facto defence against this, and it was a bad trade: 4 command
names in 176 cues, suppressed by discarding 132 real ones. Ranks 41-80 — the
first thing the cap threw away — are `colimit, adjunction, exactness,
compactness, homotopy, factorization, composition, acyclic, natural`.

## What the clean re-extraction changed

Harvesting the same 53 proofs of 0806.1324 from the corrupted probe vs the
2026-09-20 clean run:

| | probe | clean |
|---|---|---|
| cues harvested | 120 | **176** |
| LaTeX-artifact cues after the fix | 5 | **0** |

The five survivors in the probe are all serialisation damage, including a case
not previously catalogued: `$\"mathcal{T}$`, where the JSON escape alphabet chose
`\"` for `\mathcal` and left a stray quote in the text.

And the vocabulary changes character. The probe contributed `invertibility,
acyclicity, surjectivity, smallness, orthogonality` — nominalisations from the
model's own retyped prose, since `:text` held the model's wording. The clean run
contributes `apply, denote, sends, characterization, equivalent, multiplicative`,
harvested from verbatim source. **The corpus was partly modelling the model's
voice reflected back at it.**

## Status

`top` now defaults to `None` (uncapped) and the tokenizer strips commands. The
curve is still not evidence: defect 2 is a measurement design problem, not a bug,
and stands until the held-out evaluation reports.
