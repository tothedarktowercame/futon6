# Leiden alignment — the proof-checking work as a quality/reviewability instrument (and where it goes)

*Positioning note (claude-loop, 2026-06-17, with Joe). Re `data/leidendeclaration.html` — the
**Leiden Declaration on Artificial Intelligence and Mathematics** (2 June 2026, endorsed by the IMU). The
point is NOT to defend "AI for maths"; it is that our work **serves the mathematical-quality values the
Declaration champions**, author-agnostic, and is precisely the kind of reviewing support it asks for.*

## Separate the two buckets

The Declaration mixes two very different things; respond to them oppositely.

- **Governance / politics** — attribution economics, research autonomy, hiring/funding incentives, copyright,
  press-release science, environmental & ethical concerns. Policy positions; **not our tool's domain**, and
  we should not pretend otherwise.
- **Mathematical quality & reviewability** — correctness, transparency, independent verifiability, proper
  references, *understanding of why*, and **support for the needs of reviewing**. These are real, would be
  real with no AI in the picture, and **our work is an ally of them, not a threat.**

The frame that keeps this honest: **the tool is author-agnostic.** A gap-detector does not care whether a
human or a model wrote a thin step — and the thin steps a student notices are in human-written published
maths too; that is *where it is most useful*. We are not building an "AI maths" artefact the Declaration
should fear; we are building **a reviewability/quality instrument for the mathematical commons** — exactly
the "support the needs of reviewing" recommendation.

## Concern → capability (the quality bucket)

| Leiden quality concern (verbatim-ish) | What the discursive-core work offers |
|---|---|
| "plausible but unreliable (or incorrect) arguments difficult to distinguish from correct proofs … review under increasing pressure" | the **gap map** — rung-3 thin steps · R2d undefined terms · SFC2b ungrounded symbols — surfaces *exactly where* an argument is thin/ungrounded, focusing reviewer attention |
| "give precise and complete references"; models "do not properly cite the human works they synthesize" | the **import-descent** (R2d + `mark3_thread_tapestry`) is an attribution map: what a proof depends on, which cited paper grounds it; undefined terms = missing references, flagged |
| "support the needs of reviewing — make it easier for your peers to review your work" | the **per-paper certificate / guide** (read-first · start-here · open-questions · why-valuable) *is* reviewer orientation |
| "transparency and independent verifiability … no proprietary knowledge or equipment required" | deterministic-where-possible + auditable-questions-on-the-residue; classical + bounded LLM; the gap map makes verification **tractable**, not magical |
| "proofs … impart understanding of why their conclusions are true" (their stated core value) | technique-grounding (rung-3) is about the *why* — it flags where the why is missing (the heuristic-leaf / undischarged HOWEVER) |

**What we explicitly do NOT address:** the governance bucket. And we should not overclaim the quality bucket
either — the generator is thin, the checking infra is partial; this is a *direction* that serves these
values, demonstrated on a handful of papers, not a finished product.

## Where it goes — the same machinery, read forward (gaps → ArSE → answers → new papers)

The discursive core says **a sorry is a question.** That gives the work two directions over one substrate:

- **Backward = review (defensive, build first).** The question is a *flag*: "this step is thin — check
  here." This is the Leiden-aligned reviewability instrument above, validatable now on existing papers.
- **Forward = research (generative, the horizon).** The *same* question is a *frontier*: a gap recurring
  across papers, ranked by the certificate's value-signals (centrality / recurrence / connections), is an
  **open problem**. Phrase it via the question-pattern menu → **ArSE** → pursue → **answer**. An answer
  *persists as data*: a new fill / pattern / **result** that re-enters the corpus, which the tool re-reads.

So the gap is a **Janus object** — a quality-flag and a research-opportunity at once — and the review tool is
the research engine *read backward*: **building one builds the other.** Threads that already point here:
reverse-morphogenesis ("asking a good question is reverse-morphogenesis" — the gap→question step *is* it);
**conjectures** (author-declared gaps) are a ready-made research agenda; the pattern-seeding loop (an answered
gap that is a valid novel technique mints a reusable pattern) is the meta-level of "answer → new paper".

A quiet rebuttal to Leiden's *autonomy* fear sits in the forward arc: the Declaration worries research gets
prioritized "because of amenability to automated mathematics rather than expert judgment of deeper
significance." But this loop takes its direction from **the literature's own gaps**, ranked by
recurrence/centrality/significance — the questions come *from the mathematics*, not from what a model finds
easy. (Holds only if the value-signals genuinely track significance — hard — but the *direction* is
corpus-driven, not automation-shaped.)

## Sequencing + the honest line

1. **Build the review tool first** — it is near-term, validatable, and lands squarely on the IMU-endorsed
   "support the needs of reviewing." Positioning: *a transparency/reviewability instrument for the
   mathematical commons*, author-agnostic — not "AI maths".
2. **The research engine is the same substrate read forward** — no rebuild; the gaps→ArSE→answers→new-papers
   loop reuses the certificate, the scopes, the question-pattern menu.
3. **Don't overclaim, and don't touch the governance bucket** — the credibility of (1) depends on staying on
   the quality side and being honest about how early it is.

This sits with the commons / Krowne-tetrahedron line: a tool *for* the mathematical commons' own quality
standards, which is a far stronger and truer position than "AI for maths".
