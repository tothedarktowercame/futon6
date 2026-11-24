From {OWNER}. A reading task on one mathematics problem for mission M-reader-clear. About 45 minutes. Commit nothing; write only in {DIR}.

ROLE. You are reader {READER} for problem {P}. You have not seen the Lean proof and must not look for it.

READ ONLY these files in {DIR}: problem.md, version-I.informal.md (a hand-written proof), version-P.paperprinter.md (a machine-printed account of the formal proof), steps.jsonl (the printed proof cut into numbered steps). Do not open any other file about this problem, any Lean source, another reader's files, or ArSE threads. Do not search the web.

TASK. For every row of steps.jsonl, decide whether you can justify that printed step from what is printed before it, the statement, and standard mathematics. The informal proof is there to tell you what the argument is meant to be; a step is not justified merely because the informal proof says something similar.

WRITE in {DIR}:

1. steps.{READER}.jsonl, one row per row of steps.jsonl, same order:
   {"id": "<step id>", "verdict": "justified" | "wrong-as-printed" | "not-justified" | "setup", "note": "<at most one sentence, optional>"}
   Use "setup" only for sentences that introduce variables or hypotheses and assert nothing.
   Use "wrong-as-printed" when the step's conclusion holds and you can supply the argument, but the print gets it wrong: the printed reason is not the reason, a line the argument needs is absent, or a formula is misprinted. A note saying what is wrong is then required; no question row is needed.

2. questions.{READER}.jsonl, one row for each step marked not-justified (several adjacent steps with one cause may share a row; list them):
   {"id": "Q<n>", "steps": ["<step id>", ...], "locus_P": "<a passage copied exactly from version-P.paperprinter.md, at least 12 characters>", "locus_I": "<the corresponding passage copied exactly from version-I.informal.md, or empty>", "kind": "why-true" | "what-needed" | "what-if" | "mismatch" | "unclear-wording", "contrast": "I-explains-P-silent" | "P-explicit-I-skips" | "both-silent" | "both-clear-but-different" | "not-applicable", "question": "<the question>", "my_guess": "<your best guess at the answer>", "would_settle": "<what kind of answer would settle it>"}
   Copy loci character for character, including LaTeX; do not add quotation marks or change delimiters. A checker rejects loci that are not substrings of the text.

3. Also record, as extra rows of questions.{READER}.jsonl with "steps": [], anything in the statement or the informal proof that looks wrong, unused or missing (a hypothesis never used, a condition that seems needed but is not stated).

4. reading-notes.{READER}.md: five to ten lines: which version you found clearer and where; whether you could now state the proof without the texts.

READ EVERY STEP YOURSELF. Do not generate verdicts, notes or questions with a script or a rule such as "no 'by' clause means not-justified". A line with no printed reason is justified if you can supply the reason from earlier lines, the informal version or standard mathematics; say that reason in the note. A line that only fixes a witness or opens a case is setup. Each not-justified note and each question is about that line's mathematics; identical notes or questions across steps will be returned.

CHECK before replying, and fix what it reports:
  (run from the futon6 checkout, which sits beside futon3c)
  python3 scripts/reader-clear-check-loci.py --print {DIR}/version-P.paperprinter.md --informal {DIR}/version-I.informal.md {DIR}/questions.{READER}.jsonl

REPLY with: counts of justified / wrong-as-printed / not-justified / setup, number of questions by kind, the checker's summary line, and nothing from outside the four files.
