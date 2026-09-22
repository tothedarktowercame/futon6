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
