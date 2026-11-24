From {OWNER}. Second pass of your reading for mission M-reader-clear, problem {P}. About 30 minutes. Commit nothing; write only in {DIR}.

READ: your own files from the first pass, and the rows of {DIR}/answers.jsonl whose "reader" is {READER}. Still do not open the Lean source or the other reader's files.

WRITE {DIR}/round2.{READER}.jsonl, one row per question of yours:
  {"id": "<question id>", "status": "settled" | "settled-with-residue" | "not-settled" | "objection", "restatement": "<the answer in your own words, one to three sentences>", "check": "<any small check you did by hand or by computation, or empty>", "residue_or_objection": "<what remains, or empty>", "would_the_print_have_prevented_it": "yes" | "partly" | "no"}

"would_the_print_have_prevented_it" asks: had the print contained the answer's proposed sentence, would you have marked the step justified?

For each row with a residue or objection, write the follow-up question in the same row as "followup": "<question>".

REPLY with counts by status, and each residue or objection in full.
