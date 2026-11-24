From {OWNER}. An answering task for mission M-reader-clear, problem {P}. About 60 minutes. Commit nothing; write only in {DIR}.

ROLE. You hold the Lean proof. Two readers who have not seen it marked the printed steps they could not justify and asked questions. Answer every question from the Lean.

READ: {DIR}/problem.md, version-I.informal.md, version-P.paperprinter.md, steps.jsonl, questions.{READER_A}.jsonl, questions.{READER_B}.jsonl; the Lean source at apm-lean problems/{P}/lean/ ; Mathlib as needed.

WRITE {DIR}/answers.jsonl, one row per question of either reader:
  {"reader": "<reader>", "id": "<question id>", "verdict_on_guess": "right" | "partly" | "wrong" | "no-guess", "answer": "<the answer, two to five sentences>", "support": [{"kind": "lean" | "mathlib" | "experiment" | "none", "ref": "<file:line or declaration name>", "says": "<what it shows>"}], "diagnosis": "printer-drops-content" | "printer-wording" | "lean-proof-choice" | "statement-unused-hypothesis" | "statement-missing-hypothesis" | "informal-skips-or-loose" | "reader-gap", "proposed_class": "<an id from the cleanup lab's classes.tsv, or NEW:<short name>; empty unless the diagnosis is a printer one>", "proposed_print": "<for printer diagnoses, the sentence the print should have had>", "confidence": "high" | "medium" | "low"}

Rules:
- Every answer cites at least one Lean line or Mathlib name, or says support kind "none" and why.
- Where a question doubts a hypothesis (unused, or needed but missing), test it: copy the Lean file to /tmp, change the hypothesis, and compile with `cd apm-lean && nice lake env lean /tmp/<file>.lean`. One compile at a time. Record the result as support kind "experiment". Never edit the Lean source in place.
- Post each answer to ArSE as well. The first answer on a thread goes to POST /api/alpha/arse/answer; if the thread already has an answer use POST /api/alpha/arse/reply with role "answer". Thread ids are in the readers' files when they posted; if a question has none, skip ArSE for it.
- If both readers asked the same thing, answer both rows and say so in the answer.

REPLY with: number of answers, counts by diagnosis, each experiment and its result, and any question you could not answer from the Lean.
