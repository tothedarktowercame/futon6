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

## Result — 2026-06-14

Atlas built under `data/showcases/prelim-atlas/` from the manifest-backed
`/home/joe/code/storage/apm` corpus. The adapter uses the 489 IDs in
`manifest.edn` and excludes the three `lts-*` canary `.tex` files in the same
directory.

Commands:

```bash
python3 -m py_compile scripts/prelim_anatomy_atlas.py scripts/check_invariants.py
python3 scripts/prelim_anatomy_atlas.py --force --throttle-sec 0.1
```

Measured by `scripts/check_invariants.py --corpus --golden-dir
data/showcases/prelim-atlas/golden --loss-dir
data/showcases/prelim-atlas/loss`:

- problems marked: 489/489
- worker failures: 0
- corpus grounded-symbol rate: 45.85% (4,428 / 9,658 symbols)
- math-span coverage: every emitted top exemplar is 100%; corpus aggregate has
  no `C-MATH-NONNULL` debt
- well-formedness errors: 0
- residual debt: 5,478 (`C-SYM-GROUND`: 5,230; `C-DEFINIENS-DEBT`: 248)

Top concept/markup exemplars by clean well-formedness then grounded-symbol
coverage:

1. `apm-t95J02` — 100.0% grounded, 19 symbols, 13 math spans, wf 0, debt 2
2. `apm-t98J02` — 100.0% grounded, 15 symbols, 12 math spans, wf 0, debt 1
3. `apm-b03J01` — 100.0% grounded, 13 symbols, 12 math spans, wf 0, debt 0
4. `apm-m99A03` — 100.0% grounded, 13 symbols, 9 math spans, wf 0, debt 0
5. `apm-t91J04` — 100.0% grounded, 11 symbols, 6 math spans, wf 0, debt 0
6. `apm-t97A06` — 100.0% grounded, 10 symbols, 11 math spans, wf 0, debt 2
7. `apm-b97A01` — 100.0% grounded, 9 symbols, 13 math spans, wf 0, debt 1
8. `apm-a01A01` — 100.0% grounded, 8 symbols, 5 math spans, wf 0, debt 0
9. `apm-t03J03` — 100.0% grounded, 8 symbols, 9 math spans, wf 0, debt 0
10. `apm-t00J02` — 100.0% grounded, 7 symbols, 9 math spans, wf 0, debt 2

Spot checks:

- `apm-t95J02` resolves to the Klein-bottle retraction problem; wf 0 and 100%
  grounded, with two real definiens debts around "Klein bottle" and "two
  circles in the standard drawing".
- `apm-b03J01` resolves to the finite-group/subgroup-index problem; wf 0,
  100% grounded, no debt.
- `apm-m99A03` resolves to the Hilbert-space compact-operator problem; wf 0,
  100% grounded, no debt.
