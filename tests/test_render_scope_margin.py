"""Margin notes judge each node's quote against its own gloss, and mock the fixes."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import render_scope_margin as margin

SPANS = [{"kind": "env/proposition", "start": 0, "end": 60, "text": "Let T be a triangulated category and suppose it has coproducts"},
         {"kind": "let-binder", "start": 4, "end": 40, "text": "T be a triangulated category"},
         {"kind": "quant/universal", "start": 61, "end": 120, "text": "for all objects X there exists a precover"}]


def node(i, gloss, spans, kind="claim"):
    return {"id": f"n{i}", "kind": kind, "gloss": gloss, "text": " ".join(SPANS[int(s[1:]) - 1]["text"] for s in spans),
            "quote-spans": spans, "source": {"lines": [1, 1]}}


def graph(nodes):
    return {"passage/id": "p:proof0:L1-9", "source": {"lines": [1, 9]}, "nodes": nodes, "edges": []}


def test_a_node_citing_the_next_clause_in_the_list_is_re_anchored_to_the_one_it_describes():
    g = graph([node(1, "T is a triangulated category with coproducts", ["s1"]),
               node(2, "For all objects X there exists a precover", ["s2"])])      # cites s2, means s3
    n1, n2 = margin.proof_note(g, {"spans": SPANS}, [0])["nodes"]
    assert (n1["verdict"], n1["sequential"]) == ("agrees", True)
    assert n2["verdict"] == "re-anchor" and n2["proposal"]["span"] == "s3"
    assert n2["mock"]["verdict"] == "re-anchored"


def test_one_shared_word_is_not_evidence_and_formulae_are_not_judged():
    g = graph([node(1, "Definition of a triangulated structure", ["s3"], kind="ref"),
               node(2, "Hom(S,ΣM)=0", ["s1"])])
    ref, formula = margin.proof_note(g, {"spans": SPANS}, [0])["nodes"]
    assert ref["verdict"] == "unclear" and ref["proposal"] is None    # only "triangulated" is shared
    assert formula["verdict"] == "uncheckable" and formula["mock"]["verdict"] == "kept"


def test_mock_scope_rules():
    defs = {"connection": "", "connection/example-source": "", "connection/literature-gap": "",
            "connection/literature-gap/terminology-origin": ""}
    base = {"kind": "connection", "bare-parent": True, "flags": []}
    assert margin.mock_scope({**base, "fill": "the terminology of Iyama and Yoshino"}, defs)["suggest"] == \
        "connection/literature-gap/terminology-origin"
    assert margin.mock_scope({**base, "fill": "the example of a chain DGA"}, defs)["suggest"] == "connection/example-source"
    assert margin.mock_scope({**base, "fill": "its suspension functor"}, defs)["verdict"] == "held"
    assert margin.mock_scope({**base, "fill": "x", "flags": ["unanchored"]}, defs)["verdict"] == "rejected"
    assert margin.mock_scope({**base, "fill": None}, defs)["verdict"] == "held"


def test_an_s1_kind_without_a_definition_stops_the_build():
    assert margin.mark_kind("env/lemma")[0] == "region"
    assert margin.mark_kind("symbol-grounded")[0] == "term"
    try:
        margin.mark_kind("brand-new-kind")
    except ValueError as e:
        assert "brand-new-kind" in str(e)
    else:
        raise AssertionError("an undefined kind was accepted")


def test_defined_terms_classify_each_use_by_what_s1_marked():
    text = ("\\begin{document}\n"
            "A pair is a \\textit{co-$t$-structure} if it has a \\textit{heart}.\n"
            "Every co-$t$-structure here has a heart, and the heart is abelian; so is a homotopy colimit.\n")
    env_end = text.index(".\n") + 1
    at = text.index("co-$t$-structure here")
    s = text.index("structure", at)
    h = text.index("heart,")
    marks = [{"kind": "env/definition", "start": text.index("A pair"), "end": env_end},
             {"kind": "concept", "start": s, "end": s + 9, "fields": [["source", "lexicon"], ["grounded", "lexicon:structure"]]},
             {"kind": "concept", "start": h, "end": h + 5, "fields": [["source", "defined-in-paper"]]}]
    starts = [0] + [i + 1 for i, c in enumerate(text) if c == "\n"]
    terms = {t["term"]: t for t in margin.defined_terms(text, marks, starts)}
    assert set(terms) == {"co-$t$-structure", "heart"}          # "homotopy colimit" is used, not defined
    cot = [(text[a:b], st, how) for a, b, st, how in terms["co-$t$-structure"]["occurrences"]]
    assert ("co-$t$-structure", "tagged generically", "lexicon:structure") in cot
    assert ("co-$t$-structure", "untagged", "") in cot            # the defining use itself
    heart = [st for _, _, st, _ in terms["heart"]["occurrences"]]
    assert heart.count("tagged as defined") == 1 and heart.count("untagged") == 2


def test_a_leading_parameter_matches_any_symbol_but_t_is_literal():
    import re
    assert re.fullmatch(margin.term_pattern("$\\F$-preenvelope"), "$\\mathcal{B}$-preenvelopes")
    assert not re.fullmatch(margin.term_pattern("$t$-structure"), "$n$-structure")
