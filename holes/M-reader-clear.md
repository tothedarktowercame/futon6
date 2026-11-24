# M-reader-clear — a reading gate on zero-residual-ink PaperPrinter records

**Status:** OPEN (Joe's go, 2026-11-24: "It is OK with me, codex and Zai are both
relatively fresh, probably good to use a mix of those two agent classes"). Tooling is
being built; the sample draw waits for the cleanup lab's next accepted head.

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

## The argument

**IF:** a problem's PaperPrinter record is to be offered to a reader as an account of
its proof,
**HOWEVER:** the lab's ink measure marks only defect classes that have an extractor, so
zero residual ink is a lower bound on defects and says nothing about steps that print
well-formed text with the reason missing or wrong,
**THEN:** have independent readers try to justify every printed step of a frozen sample
of zero-ink problems, and route each failure by diagnosis,
**BECAUSE:** in the pilot this reading found such steps in a zero-ink problem (a96J06, 13
of 23 steps), and also found statement and Lean-source defects that no printer measure
could.

## Population and sample

At head `7db1eb81`: 468 problem modules, 8,778 records. 312 modules have no ink span in
any declaration (289 annotated records in 156 modules; confirmed by claude-7 from
`residual-ink-annotations`). All 312 have an `informal-solution.md` in apm-lean. Their
median size is 5 declarations and about 1,100 printed proof words. These figures will be
recounted at the head the sample is drawn from.

"Zero residual ink" means none of the known, measurable defect classes; it does not mean
read clean. Sentences that assert without a reason are not inked (the lab measured 149
such sentences the N1 extractor does not count), so readers should be expected to find
them.

- **Exclusion:** records with a non-empty `proof_error` have no ink because they have no
  proof text. A module containing such a record is dropped before the draw.

- **Sample:** 30 of the 312, drawn once with a recorded seed, stratified by subject letter
  and by printed length (thirds). a96J06 is excluded as already read.
- **Frozen:** the sample list, seed and head sha are committed before any reading.
- **Head:** the sample is drawn from the lab's next accepted head (the batch in trial on
  2026-11-24 changes several hundred records, so `7db1eb81` would be partly stale).
  Renders are read from the stored head directory, and every finding carries the head
  sha. Heads will keep moving; a reading stays valid as a reading of its head, and the
  lab can say afterwards which findings later heads removed.
- **Gate 6:** this sample does not count toward the lab's readiness gate 6 (claude-7's
  reading; the gate is claude-10's and Joe's). It is drawn before the last repair, from
  zero-ink modules only, and its findings feed repairs. It is discovery evidence and a
  rehearsal of the protocol. A qualifying pass would be the same protocol on a fresh
  sample drawn after the final head, across all modules, reusing no problem from this one.

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
  cited reason replaced by a wrong one, by a seat that is not a reader. The planting is
  done in copies of the stored render text, never in a printer worktree. Measures what
  readers miss.
- **Known-good.** 5 short records that claude-12 and claude-7 have both read and consider
  fully justified. Measures false alarms.
- **Verbatim loci.** A script checks every quoted locus against the text; a reading with
  non-matching loci is returned to the seat. (Two of three pilot seats paraphrased.)
- **Seat assignment.** Each problem is read by one Codex seat and one Zai seat (Joe,
  2026-11-24: a mix of the two classes), assigned by the seeded draw and not by
  availability, so that reader class and problem effects can be separated. The answering
  seat is Codex, since it builds Lean.

## Where findings go

| Diagnosis | Destination |
|---|---|
| printer drops content, printer wording | cleanup lab, as one TSV row per finding (below) |
| statement unused or missing hypothesis, uncalled lemma | APM statement-repair pool, each with its compile experiment |
| Lean proof choice that could be simplified | note to the problem's owner; no edit while a render batch is in trial |
| informal proof skips or is loose | informal corpus errata list |
| reader gap | recorded, no action |

**Findings table for the lab** (claude-7's form, for an expected 300 to 600 findings):
one TSV row per finding with head sha, module, declaration, the quoted printed line or
step ordinal, the reader verdict, the diagnosis type (printer, Lean proof, statement,
reader error), a proposed class (an existing id from the lab's `classes.tsv` or
`NEW:<short name>`), and one sentence of evidence; plus a per-class count table. Both are
delivered in the storage area under `futon6/inkup/`. Nothing is written into the lab's
directory; the lab writes its own intake note and class rows from the table.

This mission changes no printer code and no Lean source. It reads and routes.

**Wording in reports.** "Justified" is a reading result, not a correctness claim. Until
the lab's gates hold, reports keep its line: rendering works; informalisation validation
is in progress.

## Cost

Five seat jobs per problem (two reads, one answer, two restatements), each under an hour
in the pilot. 40 problems with controls is about 200 jobs. Dispatch at most three Zai
seats at once and stagger Codex seats by 30 seconds; at that rate the sample takes several
days of background running. A first tranche of 10 problems (50 jobs) is reviewed before the
rest is released.

**Seats and machine.** Reader and answering seats are not drawn from those the lab is
using (on 2026-11-24: codex-2, 12, 13, 15, 21, 26, kimi-1, zai-6; check again at
dispatch). Reading stored renders costs nothing; an answering seat that builds Lean runs
one `nice` process at a time while the lab's census is running.

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

1. The lab's next accepted head: claude-7 will bell claude-12 when it is accepted. The
   draw waits for that.
2. A reader who has the informal proof never asks why the proof takes its main approach.
   A statement-only reading would cover that; it is not part of this mission.
3. Whether a gate-6 pass by this protocol is wanted later is for Joe and claude-10.
