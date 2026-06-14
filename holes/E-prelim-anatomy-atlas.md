# E-prelim-anatomy-atlas — apply the CT anatomy markup to the 489 prelim problems

**Excursion (bounded, single-owner). Spun out 2026-06-14 to keep the main
session on CT. Owner: a Codex agent. Bell claude-1 back with results + shas.**

## Goal
Apply the futon6 structure-markup capability (`scripts/dp_paper_view.py` +
`scripts/check_invariants.py`) to the **489 UT-Austin prelim problems** (the
prelim-tutor corpus) → a "Prelim Anatomy Atlas": per-problem anatomy markup + a
coverage atlas surfacing the best-covered exemplars. This is a mechanical
*application* of the existing, validated CT-mining kernel to a new corpus — a
clean Codex handoff.

## Scope / steps
1. **Locate the corpus** — the 489 prelim problems (grep `~/code` for the
   prelim-tutor data; see the user-memory "UT Austin Prelim Tutor". If it's not
   in eprint-tarball form, adapt the reader — the detector only needs text).
2. Run the DP markup over the 489 (reuse `dp_paper_view`; the prelim text is
   shorter/cleaner than arXiv, so expect high coverage). If the input shape
   differs from arXiv eprints, write a thin adapter, do NOT fork the detector.
3. Measure with `check_invariants` (per-problem + aggregate); surface the
   top-coverage exemplars as the "atlas."
4. Emit `data/showcases/prelim-atlas/` + write results back into this file.

## Acceptance
489 problems marked; grounding/wf measured; top exemplars surfaced; no detector
fork (adapter only). data/ is gitignored — commit scripts/adapters + this note.

## Constraints
Never restart the futon3c JVM. Co-Authored-By: Claude Fable 5
<noreply@anthropic.com>. Bell claude-1 back with {problems marked, coverage,
top exemplars} + shas.
