# M-reader-clear — a reading gate on zero-residual-ink PaperPrinter records

**Status:** DRAFT (2026-11-24, claude-12, at Joe's request). Nothing is dispatched.
Awaiting claude-7's view on the sample and the routing, then Joe's go.

**Type:** mission (quality assurance). **Raised:** 2026-11-24 by Joe, after the InkUp
pilot: "If we had a sample of prelim papers that had been marked 'clear' by claude-7,
this reading approach could add another level of quality assurance."

## The question this answers

The PaperPrinter cleanup lab (futon4 `holes/labs/E-paperprinter-cleanup/endgame`)
measures residual ink: the share of printed proof text marked by a known defect class.
A problem with zero residual ink has no text any class extractor flags. Does that mean a
competent reader can follow it?

The pilot says not necessarily. a96J06 has zero residual ink at the accepted head
`7db1eb81`, and a reader who had the hand-written proof beside it could justify 10 of its
23 printed steps. The places the readers stopped mostly print something well-formed: "by
(3), (5), and (6)" where the content is one unnamed Mathlib theorem; "Suppose w = 0" with
the contradiction missing; "by simplification" for a monotonicity argument. The ink
measure cannot see these, because nothing is garbled. (The lab's own intake note for the
pilot findings says the same.)

**So the question is:** of the problems with zero residual ink, what fraction can a reader
justify step by step from the print alone, and for the steps they cannot, whose defect is
it: the printer's, the Lean source's, the statement's, or the reader's?

## Population and sample

At head `7db1eb81`: 468 problem modules, 8,778 records. 312 modules have no ink span in
any declaration (counted from `residual-ink-annotations` and `residual-ink-overlay`; the
count is to be confirmed by claude-7). All 312 have an `informal-solution.md` in apm-lean.
Their median size is 5 declarations and about 1,100 printed proof words.

- **Sample:** 30 of the 312, drawn once with a recorded seed, stratified by subject letter
  and by printed length (thirds). a96J06 is excluded as already read.
- **Frozen:** the sample list is committed before any reading. It is not used to tune
  printer repairs, so that it can count toward the lab's readiness gate 6 (a discovery
  pass not used to tune the repair).
- **Head:** every record is read at one named accepted head. A later head means a new
  reading, not an update.

## Procedure per problem

Three passes, as in the pilot (protocol and file formats:
`holes/inkup/` pilot records and the wrap-up note in the storage area,
`futon6/inkup/PILOT-WRAPUP-2026-11-24.md`).

1. **Read.** Two reader seats, independently, neither having seen the Lean. Each gets the
   problem statement, the informal solution, and the print. Each marks every printed step
   justified or not, and for every step not justified posts one question to ArSE with the
   verbatim locus, its own guess, and what would settle it.
2. **Answer.** One answering seat that holds the Lean answers each question in the same
   ArSE thread (`/arse/reply`), cites the Lean line or Mathlib name, runs a compile
   experiment where a hypothesis is in doubt, and assigns one diagnosis.
3. **Restate.** Each reader restates the answer in its own words and marks the thread
   settled, settled with residue, or not settled. Residues get one further turn.

**Diagnoses** (one per question): printer drops content; printer wording; Lean proof
choice; statement carries an unused hypothesis; statement lacks a needed hypothesis;
informal proof skips or is loose; reader gap.

## What is measured

- **Step-justified rate** per problem and per reader: steps justified before any answer,
  over printed steps.
- **Reader-clear:** a problem is reader-clear when both readers justify every step before
  any answer. The headline number is reader-clear problems over 30.
- **Questions by diagnosis**, and by the lab's defect class for the printer ones.
- **Agreement between the two readers** on which steps are unjustified.
- **Settlement:** threads settled after one answer, after two, not settled.

## Controls

Without these the rates cannot be interpreted.

- **Planted defect.** In 5 further problems, one step of the print is removed or its
  cited reason replaced by a wrong one, by a seat that is not a reader. Measures what
  readers miss.
- **Known-good.** 5 short records that claude-12 and claude-7 have both read and consider
  fully justified. Measures false alarms.
- **Verbatim loci.** A script checks every quoted locus against the text; a reading with
  non-matching loci is returned to the seat. (Two of three pilot seats paraphrased.)
- **Seat assignment.** Readers are assigned to problems by the seeded draw, not by
  availability, and each reader model family reads at least 10 problems, so seat and
  problem effects can be separated.

## Where findings go

| Diagnosis | Destination |
|---|---|
| printer drops content, printer wording | cleanup lab intake, as typed class entries with locus, in the form of `INTAKE-inkup-2026-11-24.md` |
| statement unused or missing hypothesis, uncalled lemma | APM statement-repair pool, each with its compile experiment |
| Lean proof choice that could be simplified | note to the problem's owner; no edit while a render batch is in trial |
| informal proof skips or is loose | informal corpus errata list |
| reader gap | recorded, no action |

This mission changes no printer code and no Lean source. It reads and routes.

## Cost

Five seat jobs per problem (two reads, one answer, two restatements), each under an hour
in the pilot. 40 problems with controls is about 200 jobs. Dispatch at most three Zai
seats at once and stagger Codex seats by 30 seconds; at that rate the sample takes several
days of background running. A first tranche of 10 problems (50 jobs) is reviewed before the
rest is released.

## Exit

- The 30-problem sample and both control sets are read, answered and restated.
- A report gives the reader-clear rate with the control rates beside it, questions by
  diagnosis and class, reader agreement, and the routed findings with their destinations'
  acknowledgements.
- Joe decides, from that report, whether reader-clear becomes a standing gate for accepted
  heads, a periodic sample, or stops here.

## Not in scope

Generating tellings or content plans (the other half of the InkUp pilot); reading records
that still carry residual ink; reading at more than one head.

## Open points

1. The zero-ink count (312) and the definition of "no ink in any declaration" are mine,
   from the head's annotation files. claude-7 to confirm or correct.
2. Whether the lab wants the sample drawn from the accepted head or from the head after
   its current batch is accepted.
3. A reader who has the informal proof never asks why the proof takes its main approach.
   A statement-only reading would cover that; it is not part of this mission.
