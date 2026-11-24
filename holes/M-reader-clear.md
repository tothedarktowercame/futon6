# M-reader-clear — a reading gate on zero-residual-ink PaperPrinter records

**Status:** OPEN (Joe's go, 2026-11-24: "It is OK with me, codex and Zai are both
relatively fresh, probably good to use a mix of those two agent classes"). Sample frozen
2026-11-24 at head `b38755b7`, seed 20261124: 332 eligible zero-ink problems of 468, 30
sampled and 5 for planted controls (`holes/reader-clear/sample.head-b38755b7.json`,
release order in `release.head-b38755b7.json`). Tranche 1 (10 problems) was read, answered
and restated on 2026-11-24: none is reader-clear (review below). Both control sets were
read on 2026-11-24 (controls review below). Tranche 2 (the 17 proved problems of the
remaining 20; 3 are unproved) was read, answered and restated on 2026-11-24: none of the 27
proved problems is reader-clear (tranche 2 review below). Left: file the statement and
owner notes; recheck the eligible population for `sorry`.

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

## Tranche 1 review (2026-11-24, claude-12)

Ten problems, 1,214 printed steps, 50 seat jobs plus 5 follow-up jobs. Findings table,
class counts and a notes file are in the storage area under
`futon6/inkup/reader-clear/` (`findings.head-b38755b7.tsv`, 171 rows).

| problem | steps | Codex reader, unjustified | Zai reader, unjustified | both | either |
|---|---|---|---|---|---|
| a98J02 | 71 | codex-19: 1 | zai-2: 2 | 0 | 3 |
| m03J04 | 82 | codex-20: 29 | zai-4: 8 | 6 | 31 |
| t91J05 | 313 | codex-3: 10 | zai-14: 34 | 2 | 42 |
| b03J04 | 29 | codex-18: 7 | zai-2: 6 | 5 | 8 |
| t94A05 | 6 | codex-19: 3 | zai-5: 3 | 3 | 3 |
| a96J04 | 154 | codex-3: 6 | zai-14: 8 | 1 | 13 |
| m96J05 | 117 | codex-20: 37 | zai-4: 12 | 12 | 37 |
| a96A06 | 52 | codex-19: 1 | zai-5: 0 | 0 | 1 |
| m03J02 | 326 | codex-18: 37 | zai-2: 13 | 12 | 38 |
| aunk04 | 64 | codex-20: 15 | zai-5: 4 | 4 | 15 |

- **Reader-clear:** 0 of 10. 191 steps (16%) were unjustified by at least one reader, 45
  (4%) by both.
- **Questions by diagnosis (171):** printer 155, Lean proof 6, informal 5, statement
  already repaired 4, statement 1, reader gap 0. Largest printer classes:
  P-missing-subproof 37, P-unnamed-library-result 35, P-lost-row 20,
  P-implementation-vocabulary 17, lost instance hypotheses (new) 12,
  P-misattributed-reason 12.
- **Settlement:** 149 settled after one answer, 21 with a residue (each given one further
  answer), 1 objection.
- **Findings outside the table** (from follow-up turns; in the notes file): proofs that
  cite a library theorem where the problem asks for an argument (t94A05, m03J02), unused
  declarations (t91J05, a98J02, aunk04), "Lemma k" numbers that do not identify a
  declaration (six problems), one statement that requires more than the problem (m03J02,
  equation on all of R), one narrowing (t91J05).

What the tranche showed about the protocol:

1. **Reader agreement depends on the seat.** With codex-18 or codex-20 as the Codex
   reader the Zai reader's unjustified set lies almost wholly inside the Codex one. With
   codex-3 the two sets barely overlap; codex-3's first return on t91J05 was generated by
   a script and was sent back. A count of "unjustified by either" therefore mixes problem
   and seat.
2. **No question was diagnosed as a reader gap,** and the stricter reader was borne out
   by the answers each time. Without the controls this cannot be told apart from an
   answering seat that never blames the reader.
3. **Checks added during the tranche:** distinct notes and questions among unjustified
   steps (catches scripted returns); restate packets say that the reader's answer rows
   exist; answer packets ask for existing lab class ids. The kit builder now removes Lean
   code blocks from the informal version (a96J04 was read with a stale one).
4. **Zero step questions is not zero questions.** zai-5 justified every step of a96A06
   and still asked two questions about the statement and a quoted theorem.

**Decision.** The controls are run before the remaining 20 problems, since the tranche
result (0 of 10, no reader gaps) is the kind the controls exist to interpret.

## Controls review (2026-11-24, claude-12)

Kits in storage under `reader-clear/head-b38755b7/kits-c/` (a neutral name; readers were
not told these were controls); planting record in `ledger/planted.jsonl` beside it. Each
record was read by one Codex and one Zai seat. A defect counts as *caught* only when the
step is marked not-justified and a question on it names the defect.

**Known-good (five records supplied by claude-7, read and accepted by claude-12; not
checked against Lean).** a92J05, m93J03, m03J05 D99, a96A04, m98A06: every step justified
in all 10 readings. No false alarms. These were single-declaration excerpts; the first
three kits did not say so and each reader asked where the main proof was. Those questions
are not counted.

**Defects (9 records, 18 readings).**

| record | defect | Codex reader | Zai reader |
|---|---|---|---|
| m03J05 D121 | natural: wrong reason ("algebra" for a substitution) | passed | passed, wrong label noted |
| m96A02 D9 | natural: wrong reason ("by adding to both sides") | passed | passed, wrong label noted |
| b01J01 D8.S5 | planted: wrong reason | passed | passed |
| a96J07 D2.S9 | planted: wrong reason | passed | passed, wrong label noted |
| m02J04 D16.S4 | planted: wrong reason | passed | passed, wrong label noted |
| t96A08 D19 | natural: calculation stops one line early | marked, other question asked | caught |
| t94J08 D10 | planted: one-line consequence removed | passed | passed, gap noted |
| t02A02 D10 | planted: equality removed that later lines cite | caught | caught |
| (same kits) | steps marked outside the altered declaration | 55 | 36 |

- **Wrong reason on a true line:** caught in 0 of 10 readings; mentioned in a step note in
  4 (all Zai).
- **Missing line or step:** caught in 3 of 6 readings; marked for another reason in 1;
  noted and passed in 1; passed in 1.
- The three wrong-reason plants by codex-3 only changed a lemma number, which the
  lemma-numbering defect makes undetectable; claude-12 replaced them with named wrong
  reasons. Both versions are in the planting record.

**Reading of the result.** A reader marks a step when they cannot reconstruct why it
holds. A step whose printed reason is wrong, or which lacks a line the reader can supply,
is passed; the reader packet told them to ("justified if you can supply the reason").
Zai seats often record the flaw in the note; Codex seats usually do not. So tranche 1's
16% counts steps readers could not follow, and does not count print defects that a
competent reader repairs silently. "Reader-clear" by this protocol means followable, not
correctly printed.

**Change for the remaining 20.** The reader packet gains a verdict `wrong-as-printed`
(conclusion holds and the reader can supply the argument, but the printed reason is not
the reason, a needed line is absent, or a formula is misprinted), with a required note and
no question row. A line with no printed reason that the reader can supply stays
`justified`, as in tranche 1, so the not-justified rates remain comparable; the new
verdict is reported as a separate rate. It was not tested on the controls.

**Other observations.** codex-19 returned the 663-step m02J04 in under three minutes by
the job record, with no note on any justified row; its three marks are correct. Its
justified verdicts on that kit are weak evidence. Findings from control readings (a power
printed with base and exponent swapped in m02J04; statement and informal items) are in
storage, `inkup/reader-clear/notes-beyond-printer.head-b38755b7.md`.

## Tranche 2 notes (2026-11-24, claude-12)

- **Three sampled problems are unproved.** t99A04, t97A06 and t02A08 have a `sorry` in
  the Lean, printed as "asserted without proof". They carry zero residual ink, so the
  draw admitted them; eligibility did not test for `sorry`. t99A04 was read before this
  was noticed (both readers marked the one unproved step). They are reported as a
  separate group, "unproved", and get no further reading, answer or restate passes; the
  reader-clear rate is over the 27 proved problems. The eligible population of 332 has
  not been rechecked for `sorry`.
- **aunk05:** the answer job went to codex-18, one of its two readers, by mistake. Only
  zai-2's restatement is taken for that problem. Answer jobs go to a seat that reads
  nothing in the tranche (codex-3).
- **`wrong-as-printed` in use:** Codex seats apply it to whole missing arguments they
  can supply, not only to wrong labels (aunk05, b93A04: 15 of 18 asserting steps, no
  not-justified, no questions). Rates for tranche 2 are therefore given both for
  not-justified alone and for either mark.

## Tranche 2 review (2026-11-24, claude-12)

All 17 proved problems of tranche 2 were read by one Codex and one Zai seat, answered by a
Codex seat that read nothing in that problem (aunk05 excepted, see notes above), and
restated. With tranche 1 that is 27 proved problems, 2,650 steps, 54 readings.

**Result: no problem is reader-clear.** Every one of the 27 has at least one step marked
by at least one reader. Two tranche-2 problems (a99J05, m99J02) have no not-justified step
but 33 and 19 wrong-as-printed ones.

| | Tranche 1 (10) | Tranche 2 (17) | All 27 |
|---|---|---|---|
| Steps | 1,214 | 1,436 | 2,650 |
| Not-justified by either reader | 191 (15.7%) | 31 (2.2%) | 222 (8.4%) |
| Either mark by either reader | 191 (15.7%) | 278 (19.4%) | 469 (17.7%) |
| Marked by both readers | 45 (3.7%) | 88 (6.1%) | 133 (5.0%) |
| Steps both readers call setup | 251 | 419 | 670 |

The drop in not-justified between tranches is the new verdict, not a better print: readers
who could supply the argument moved those steps to wrong-as-printed. The comparable figure
across tranches is "either mark": 16% then 19% of all steps (20% and 27% of steps not
called setup by both readers).

**The two seat classes read differently.** In tranche 2 the Codex readers marked 235 steps
wrong-as-printed, 2 not-justified, and asked 9 questions; the Zai readers marked 100
wrong-as-printed, 29 not-justified, and asked 60. Agreement on which steps are marked is
low (88 of 278). The readers also disagree on what is setup (a01A01: 62 against 122).

**Where the 69 tranche-2 questions were diagnosed:** printer 51, informal text 10, Lean
proof 4, statement 3, reader gap 1 (a03J01: the reader thought A < ∞ unnecessary; a
counterexample and a failed recompile show it is needed). Restates: 37 settled, 30 settled
with residue, 1 not settled (a98J01 Q4, answered afterwards by the owner), 1 not taken.

**Answer quality.** Three answers by the answering seats were wrong or misfiled and were
corrected by the owner on the ArSE thread: m99J02 (an answer filed under the wrong
question, with an answer from another problem posted in its place), a98J01 ("coarser"
topology where the source refines), a01A01 (d = 0 called the one-dimensional case). The
owner's earlier statement that lemma numbers restart per declaration was wrong for a97A02
and is withdrawn. The answer pass needs the owner's reading; row counts are not enough.

**Findings outside the print** (details in the notes file in storage):

- Statement narrower than the problem: a98J01 (Borel measurable on ℝ where the problem says
  Lebesgue measurable on [a,b]; header says no statement repairs).
- Formal proof does not carry the argument the problem asks for: t97J07 part (c) (no
  Lefschetz step linking the two computations), m01J01 part (c) (closed graph cited from
  the library, the open-mapping assumption unused), aunk05 and t95A03 (single library
  calls), t03J06 (parametrisation integrals; that they make up the surface is not stated).
- Unused hypothesis: a97A02 (t ∈ [0,1], recompiled without it), aunk05 (connectedness).
- Informal errata: a97A02, a98J01, a03J01, a01A01, t96A03, t95A03, a99J05.

**What "clear" means after this.** Zero residual ink says the printer rendered every
construct. It does not say a reader can follow the result: in this sample roughly one
step in five is marked, most often because hypotheses carried by instances are not
printed, the printed reason is a tactic name, or a cited lemma cannot be identified. And
by the controls, an unmarked step is weak evidence, since readers repaired planted wrong
reasons silently in 10 of 10 readings.

**Delivered** to the cleanup lab in storage: findings table (240 rows), a new marks table
(604 marked steps with notes), and a delivery note listing the recurring defects.

**Not done:** the three unproved problems were not read further; the eligible population
of 332 has not been rechecked for `sorry`; statement and owner notes are written but not
yet filed with the APM statement-repair pool or the proof owners; the reader packet gives
a checker path that Zai seats could not resolve from their start directory.

## After delivery (2026-11-24, claude-12)

**claude-7's reply to the tranche 2 delivery.** Nothing in it was unusable. Three points:

- "Lemma n" has two numbering regimes in the printer (`paper_lemmas.py` on the batch
  branch): `module_lemma_numbers` numbers the module's theorems in source order (the
  a97A02 reading), and `page_numbering_scope` numbers only the theorems rendered on one
  declaration's page (the a99J05 and a01A01 readings). Which regime each record ran under
  is not yet settled. The lab takes it in as a defect either way: a reader of one record
  cannot resolve the number without the name.
- Dropped instance hypotheses go in as one class of 7 problems, ranked with false-formula
  defects, with t95A03 and m01J01 flagged as false as printed.
- Some marked steps may already be changing on the batch branch (swapped power, proofs
  ending on "It suffices to show", removed closers, `X = X` calculation rows). None is
  accepted yet. The lab will bell at the next accepted head with the changed records in
  the 27 modules, and will take in both tranches together. t99A04 stays out of counts.

**Statement and proof notes filed.** apm-lean
`holes/labs/statement-registry/reader-clear-1124.md` (commit `d294575aef`). Compile
experiments for five statement points from the control sets (m03J05, m98A06, t96A08,
m02J04, b93A04) were sent to a Codex seat; results go in
`statement-experiments.head-b38755b7.md` in the storage area and are folded into that note.

**`sorry` in the population.** The draw filtered on residual ink only and did not store
the eligible list of 332, so the eligible set cannot be rechecked exactly. 3 of the 30
sampled modules were unproved. Across apm-lean, 26 of 480 `Main.lean` files contain the
word `sorry` (a text count, not a compile). A later draw should store the eligible list
and exclude modules with `sorry`.

## Errata register (opened 2026-11-24, claude-12)

Joe (2026-11-24): the errata found by the reading "need to be recorded and fixed", and
this file is where they are recorded for now. Each row is a defect in an apm-lean
statement, Lean proof or informal solution, not in the print (print defects went to the
cleanup lab; see "Where findings go"). Status is `open` until a commit fixes it; then the
row gets the commit and the date. Evidence and line numbers:
apm-lean `holes/labs/statement-registry/reader-clear-1124.md` and
`statement-experiments.head-b38755b7.md` in the storage area. Nothing here has been
regraded. Fixes to `Main.lean` files wait until claude-7's lab is not reading them at a
pinned head.

### A. Informal solutions that say something false or unsupported

| Id | Problem | Defect | Fix | Status |
|---|---|---|---|---|
| A1 | t91J05 | "the number of sheets is finite since the base is compact and the cover connected" is false (ℝ → S¹). | Take finiteness from the hypothesis that the cover is finite. | fixed `063ef70827` 2026-11-24 |
| A2 | t94A05 | "an open subset of ℝⁿ is path connected" is false. | Use a ball inside the chart image. | fixed `c8b02586e2` 2026-11-24 |
| A3 | b03J04 | A root of unity from an unspecified extension is multiplied by α in K with no common field named. | Fix a splitting field containing both. | fixed `334df3aa46` 2026-11-24 |
| A4 | a92J05 | Closing remark (a single dominating term exists for any polynomial on any circle) is false: z² + 3z + 3 on \|z\| = 1. | Delete the remark or restrict it. | fixed `31e03b2b60` 2026-11-24 |
| A5 | a96J04 | `informal-solution.md` embeds a stale Lean statement with closed intervals; `Main.lean` has open ones. | Replace the embedded statement. | fixed `d55bad50a1` 2026-11-24 |
| A6 | a98J01 | The separation constant is a minimum over compact sets that may be empty or fewer than two. | Treat those cases separately. | fixed `3654c35d2c` 2026-11-24 |
| A7 | a97A02 | The symmetric difference has measure 2·min(\|s−t\|, 2h), not 2\|s−t\|. | State the min, or an inequality (≤ 2\|s−t\|), which is all the proof uses. | fixed `56ef022aca` 2026-11-24 |
| A8 | a03J01 | "potentially uncountable collection" said of a sequence of sets. | Say countable. | fixed `ca7e158be5` 2026-11-24 |
| A9 | t95A03 | "B₂ ⊆ B₁" is not given by the cover and not needed. | Remove it. | fixed `eb7d632d63` 2026-11-24 |
| A10 | m93J03 | Uniqueness is obtained only inside the neighbourhood given by the implicit function theorem. | Check against the Lean statement first, then say which uniqueness is proved. | fixed `8fa9d21ec0` 2026-11-24 |
| A11 | m02J04 | Finds a stationary catenary without proving it minimal (the Lean proof does prove it). | Add the minimality argument, following the Lean proof. | claim corrected `67db5a86c8` 2026-11-24; minimality argument not written out |

Group A was edited on apm-lean master on 2026-11-24 after claude-7 confirmed that its lab
does not read `informal-solution.md`. Three rows turned out differently from the fix
first written down:

- A10 (m93J03): the Lean file already records that the printed global uniqueness is
  false (its second conjunct gives two solutions for every ε). The informal solution now
  claims only local uniqueness and says so.
- A11 (m02J04): the informal solution now says it shows stationarity only and cites
  `apm_m02J04_global_area_minimal`. The Lean minimality proof is a case analysis of about
  1000 lines; an informal version of it has not been written, so the row stays open in
  that respect.
- A5 (a96J04): the sketch still writes [a,b] where the problem and the Lean statement
  have [0,1]; a sentence now says so.

None of the edits has been read by a second reader.

### B. Lean statements that differ from the problem

| Id | Problem | Defect | Fix | Status |
|---|---|---|---|---|
| B1 | a98J01 | Assumes `Measurable f` (Borel) for f : ℝ → ℝ; the problem is Lusin's theorem for Lebesgue measurable f on [a,b]. Header says "Statement repairs: None". | Restate for Lebesgue measurable f on [a,b] and prove it; at the least, list the difference in the header. | open |
| B2 | t96A08 | Assumes piecewise-smooth reachability of every point (`hreachable`) where the problem has a connected manifold; the codomain is changed from M to ℝ. Neither is in the header. | Derive reachability from connectedness; document the codomain repair. | open |
| B3 | t91J05 | Main theorem assumes `[ConnectedSpace X]`; the problem's finite covers may be disconnected. Removing the instance fails to compile. | Prove the disconnected case or list the restriction as a repair. | open |
| B4 | m03J05 | `g 0 = 0` is unused; `Monotone G` is unused in the same conjunct; a nonincreasing G is not covered. | Drop the unused hypotheses (a copy without `g 0 = 0` compiles); decide the reading of "monotone" and say so in the header. | fixed `0f9365a92d` 2026-11-24 |
| B5 | b93A04 | States the join of the two fixing subgroups; the problem writes the product set H₁H₂. The header says so. | Add the 19-line set-equality lemma via `Subgroup.mul_normal` (compiles in a scratch copy). | fixed `03c0b5f982` 2026-11-24 |
| B6 | m03J02 | Imposes the equation at every real t; the problem poses it for t > 0. The file comment gives a reason. | Not judged; read and decide. | open |
| B7 | a97A02 | t ∈ [0,1] is unused in the estimate. | Drop it (a generalised copy compiles). | fixed `bc3b69d043` 2026-11-24 |
| B8 | aunk05 | The hypothesis that Ω is connected is bound and unused. | Drop it (a copy without it compiles). | fixed `e53f57413f` 2026-11-24 |

### C. Lean proofs that do not carry out what the problem asks

| Id | Problem | Defect | Status |
|---|---|---|---|
| C1 | t97J07 | Part (c): χ(ℂP²) = 3 and the fixed-point contributions summing to 3 are proved separately; no Lefschetz–Hopf statement links them. The line contribution's factor 2 is hard-coded. | open |
| C2 | t94A05 | The problem says "Do not quote any theorems you do not prove"; the proof is two library calls. | open |
| C3 | m03J02 | Part (a) says to use the contraction mapping theorem; existence and uniqueness are cited from Mathlib's Picard–Lindelöf and ODE uniqueness results. | open |
| C4 | m01J01 | Both parts are single library calls; in the implication the open-mapping assumption is bound as `_` and unused. | open |
| C5 | b03J04 | Irreducibility of X^p − β is one call to `X_pow_sub_C_irreducible_iff_of_prime`. | open |
| C6 | aunk05 | Two library calls. | open |
| C7 | t03J06 | That the two parametrised pieces make up the surface with the stated orientation is a comment, not a statement. | open |
| C8 | a98J01 | The proof is `Measurable.exists_continuous` plus regularity, not the Lusin argument. Goes with B1. | open |

C2 and C3 contradict an explicit instruction in the problem. C4, C5, C6 and C8 are proofs
by citation where the problem is the cited theorem; whether that counts as proved is a
grading question, not decided here.

### D. Unused declarations

| Id | Problem | Declaration | Status |
|---|---|---|---|
| D1 | t91J05 | Declaration 19 (Euler-characteristic lemma) is cited by nothing. | fixed `bb9c97b4b3` 2026-11-24 |
| D2 | a98J02 | Declaration 6; deleting it compiles. | fixed `c78d5ff309` 2026-11-24 |
| D3 | aunk04 | `apm_aunk04_rouche_bound`; the file still imports `ConstructionTargets.Rouche`. | fixed `5e194e6b1f` 2026-11-24 |
| D4 | m99J02 | `Continuous q` is used only for integrability. | no edit: the hypothesis is used |

### Lean edits made on apm-lean master (2026-11-24)

Seven `Main.lean` files were edited, each compiled once afterwards
(`lake env lean` on the file, exit 0, no `sorry` warning). claude-7 cleared all seven
beforehand: its renders, census and accepted head `ee47b461` read a pinned worktree and a
fixed library dump, so these edits appear in the paper-proof pages only when it rebuilds
both from master, which it will do as a separate step with its own census. It asked for
this list so that it can attribute the changes then.

| Problem | apm-lean commit | Change | Effect on the print at rebuild |
|---|---|---|---|
| m03J05 | `0f9365a92d` | `g 0 = 0` and `Monotone G` removed from the differentiability clause of `apm_m03J05_bridge_2` and `apm_m03j05`; header notes it | statement changes |
| b93A04 | `03c0b5f982` | new theorem `apm_b93A04_fixingSubgroup_inf_carrier_eq_mul` appended; nothing existing changed | one more declaration |
| a97A02 | `bc3b69d043` | estimate stated for every real t | statement changes |
| aunk05 | `e53f57413f` | `IsConnected Ω` removed | statement changes |
| t91J05 | `bb9c97b4b3` | `apm_t91J05_eulerCharacteristic_odd_genus` deleted (also noted in `proof-outline.md`) | later "Lemma n" numbers shift |
| a98J02 | `c78d5ff309` | `apm_a98J02_tendsto_integral_deriv_sq_symmetric` deleted | later "Lemma n" numbers shift |
| aunk04 | `5e194e6b1f` | `apm_aunk04_rouche_bound` and the `ConstructionTargets.Rouche` import deleted | one declaration fewer |

In m03J05 the removal of `Monotone G` goes beyond the scratch experiment, which removed
only `g 0 = 0`; the edited file compiles. D4 (m99J02) needs no edit: `Continuous q` is
used, for integrability. None of the changed statements has been regraded.

### E. Unproved, though the record shows zero residual ink

t99A04, t97A06 and t02A08 contain `sorry`. They are not errata in a proof, but each is
listed here so that it is not counted as proved from its record.

### Order of repair

1. Group A: text edits to `informal-solution.md`, one commit per problem.
2. B4, B5, B7, B8 and group D: small Lean edits already shown to compile in copies.
3. B1, B2, B3, B6: new Lean work, one problem per handoff.
4. Group C: after a decision on whether a proof by citation of the theorem asked for
   counts.

## Not in scope

Generating tellings or content plans (the other half of the InkUp pilot); reading records
that still carry residual ink; reading at more than one head.

## Open points

1. The lab's next accepted head: claude-7 will bell claude-12 when it is accepted. The
   draw waits for that.
2. A reader who has the informal proof never asks why the proof takes its main approach.
   A statement-only reading would cover that; it is not part of this mission.
3. Whether a gate-6 pass by this protocol is wanted later is for Joe and claude-10.
4. The errata register above: group A and the small Lean edits (B4, B5, B7, B8, D1-D3)
   were made on 2026-11-24. Open: B1, B2, B3, B6 (new Lean work), group C (8 rows, waits
   on whether citing the theorem asked for counts), group E (3 unproved). A11 is fixed in
   part. claude-7 asks to be told before any further `Main.lean` is edited.
