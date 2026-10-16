"""The S1 mark kinds that a proof can prove.

S1 canonicalises an author's environment names (thm -> theorem, lemme ->
lemma; dp_paper_view._ENV_CANON) and keeps any other name as it stands. The
stages that pair a proof with the statement before it (S3's candidates, S6's
paper object) each kept their own list of four (theorem, lemma, proposition,
corollary). A problem set states problems: the APM corpus has 501 env/problem
marks and 39 env/theorem, so S3 showed the model no statement for any of its
501 proofs and S6 refused 499 of 501 papers as "proofs attached to no
statement" (apm-501-xi-r1, 2026-10-16). One list, here, for all of them.
"""

PROVABLE = frozenset({"theorem", "lemma", "proposition", "corollary", "claim", "conjecture",
                      "problem", "question", "exercise"})
STATEMENT_KINDS = frozenset(f"env/{name}" for name in PROVABLE)
