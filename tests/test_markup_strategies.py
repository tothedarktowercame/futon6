"""Strategies: identity by name, with the rule that chose each binding recorded."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import markup_strategies as ms

TEXT = "\n".join([
    r"\begin{document}",                                                        # 1
    r"\section{Rigid objects}",                                                 # 2
    r"Hoshino, Kato and Miyachi look at $t$-structures induced by $S$.",        # 3
    "",                                                                         # 4
    r"\df{An object $S$ of $\T$ is called \textit{$\F$-nice} if $\Hom(S,S)=0$.}",  # 5
    "",                                                                         # 6
    r"\prop{Let $S$ be a compact object of $\T$. Then $S$ is $\B$-nice.}",       # 7
    r"\prf Since $S$ is compact, $\Hom(S,S)=0$ and $S$ is $\B$-nice. \eprf",     # 8
    "",                                                                         # 9
    r"Every $\F$-nice object is nice, for all $i$ with $\Sigma^{i}S$ defined.",  # 10
    r"\end{document}",                                                          # 11
])
STARTS = [0]
for _line in TEXT.split("\n"):
    STARTS.append(STARTS[-1] + len(_line) + 1)


def env(kind, first, last):
    return {"kind": f"env/{kind}", "start": STARTS[first - 1], "end": STARTS[last] - 1}


def sym(name, line, nth=1, grounded=None):
    at = -1
    for _ in range(nth):
        at = TEXT.index(name, at + 1 if at >= 0 else STARTS[line - 1])
    mark = {"kind": "symbol-grounded" if grounded else "symbol", "start": at, "end": at + len(name)}
    if grounded:
        mark["fields"] = [["bound", grounded]]
    return mark


LET = TEXT.index(r"Let $S$ be a compact object")
MARKS = [env("definition", 5, 5), env("proposition", 7, 7), env("proof", 8, 8),
         {"kind": "bind/let", "start": LET, "end": LET + 34, "fields": [["symbol", "S"], ["type", "a compact object of $\\T$"]]},
         {"kind": "concept", "start": TEXT.index("object $S$") , "end": TEXT.index("object $S$") + 6,
          "fields": [["source", "lexicon"], ["grounded", "lexicon:object"]]}]


def edge(marks, name):
    return next(e for e in ms.symbol_hypergraph(TEXT, marks) if e["name"] == name)


def test_an_apposition_binds_but_a_term_prefix_does_not():
    sites = ms.binding_sites(TEXT, MARKS)
    S = [b for b in sites if b["name"] == "S"]
    assert any(b["source"] == "apposition" and b["type"].startswith("object of") for b in S)
    assert not [b for b in sites if b["name"] == "t"]        # "look at $t$-structures" is a term, not a binding


def test_a_use_in_an_environment_takes_that_environment_s_binding():
    marks = MARKS + [sym("S", 5, nth=2)]                      # the S inside \Hom(S,S) of the definition
    use = edge(marks, "S")["occurrences"][-1]
    binder = edge(marks, "S")["binders"][use["binder"]]
    assert use["rule"] == "in-environment" and binder["line"] == 5


def test_a_use_in_a_proof_takes_the_statement_it_proves():
    marks = MARKS + [sym("S", 8, nth=2)]
    e = edge(marks, "S")
    use = e["occurrences"][-1]
    assert use["rule"] == "proved-statement" and e["binders"][use["binder"]]["line"] == 7


def test_a_use_with_no_earlier_binding_is_unbound_not_guessed():
    marks = MARKS + [sym("S", 3)]                             # before any binding site
    use = edge(marks, "S")["occurrences"][0]
    assert use["rule"] == "unbound" and use["binder"] is None


def test_s1_s_grounding_is_compared_with_the_binding_not_replaced():
    agrees = MARKS + [sym("S", 8, nth=2, grounded="compact object of $\\T$")]
    differs = MARKS + [sym("S", 8, nth=2, grounded="a nonzero morphism")]
    assert edge(agrees, "S")["occurrences"][-1]["s1-agrees"] is True
    assert edge(differs, "S")["occurrences"][-1]["s1-agrees"] is False
    assert edge(differs, "S")["occurrences"][-1]["s1-grounding"] == "a nonzero morphism"


def test_a_use_of_a_parametrised_term_records_the_parameter_it_instantiates():
    terms = {t["term"]: t for t in ms.defined_terms(TEXT, MARKS)}
    uses = terms["$\\F$-nice"]["uses"]
    # The last use is the bare head word ("is nice"), which instantiates no parameter.
    assert [(TEXT[u["start"]:u["end"]], u.get("parameter")) for u in uses] == [
        ("$\\F$-nice", "$\\F$"), ("$\\B$-nice", "$\\B$"), ("$\\B$-nice", "$\\B$"),
        ("$\\F$-nice", "$\\F$"), ("nice", None)]
    assert re.fullmatch(ms.term_pattern("$\\F$-nice")[0], "$\\mathcal{B}$-nices")


def test_a_windows_bindings_name_the_symbols_it_uses_with_the_rule_that_chose_them():
    import mark3_extract_candidates as mec
    strat = ms.strategies(TEXT, MARKS + [sym("S", 8, nth=2)])
    rows = mec.window_bindings(strat, 8, 8)                  # the proof line
    assert any(r.startswith("$S$") and "proved-statement" in r and "L7" in r for r in rows)
    assert mec.window_terms(strat, 10, 10) == ["$\\F$-nice — defined at L5"]


def test_a_macro_the_paper_defines_for_an_object_is_a_symbol_too():
    at = TEXT.index(r"\T$ is called")
    marks = MARKS + [{"kind": "classified", "start": at, "end": at + 2,
                      "tip": r"\T · author-defined · ID · paper.tex:45"}]
    assert any(e["name"] == "\\T" for e in ms.symbol_hypergraph(TEXT, marks))


def test_a_symbol_inside_the_name_of_a_defined_term_is_not_a_loose_variable():
    # The \F of "$\F$-nice" is part of the term's name where the term is used.
    use = TEXT.index(r"$\F$-nice object is nice")
    inside = {"kind": "symbol", "start": use + 1, "end": use + 3}
    loose = {"kind": "symbol", "start": TEXT.index(r"$\Sigma^{i}S$") + 1, "end": TEXT.index(r"$\Sigma^{i}S$") + 7}
    names = {e["name"] for e in ms.symbol_hypergraph(TEXT, MARKS + [inside, loose])}
    assert "\\F" not in names and "\\Sigma" in names


PROSE = "\n".join([
    r"\begin{document}",
    r"A category $\C$ is called \emph{small} if its objects form a set.",
    r"A functor is called a {\em localization functor} if it is idempotent.",
    r"We shall call such a map \textit{cartesian\/} when it lifts.",
    r"Every small category has a localization functor; a cartesian map is small.",
    r"\end{document}",
])


def test_terms_defined_in_prose_are_found_without_a_definition_environment():
    # 0806.1324 has no definition environment at all and names 25 terms this way.
    terms = {t["term"]: t for t in ms.defined_terms(PROSE, [])}
    assert set(terms) == {"small", "localization functor", "cartesian"}
    assert [PROSE[u["start"]:u["end"]] for u in terms["small"]["uses"]] == ["small", "small", "small"]
    assert terms["localization functor"]["definition"].startswith("A functor is called")


def test_an_emphasis_that_is_not_a_term_is_not_taken_for_one():
    import re as _re
    grab = lambda text: ms.emphasised(_re.search(ms.EMPH, text))
    assert grab("{\\em small\nsets}") == "small sets"          # a line break is not part of the term
    assert grab(r"\emph{map\/}") == "map"                      # nor an italic correction
    assert grab(r"\emph{1}") == ""                             # nor is a bare numeral a term
    assert grab(r"\emph{lment tordant (tensorielle}") == "lment tordant"
