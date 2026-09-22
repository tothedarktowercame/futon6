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



def test_the_quote_check_is_the_one_the_run_measures():
    import iatc_quote_check
    offered = [dict(sp, id=f"u{i}") for i, sp in enumerate(SPANS, 1)]
    node = {"id": "n2", "gloss": "For all objects X there exists a precover",
            "text": offered[0]["text"], "quote-spans": ["u1"]}
    row = iatc_quote_check.node_check(node, offered)
    assert row["verdict"] == "re-anchor" and row["proposal"]["span"] == "u3"
    assert iatc_quote_check.sequential(1, ["u1"], offered) is True
    assert iatc_quote_check.sequential(2, ["u1"], offered) is False
    row["sequential"] = False                                  # check_graph adds this per node
    assert iatc_quote_check.tally([{"nodes": [row]}])["checkable"] == 1


def test_a_page_pinned_light_must_restore_the_converter_s_own_colours():
    import typeset_preview
    pinned = "@media (prefers-color-scheme: dark) { html { background: #fffff8; } " + typeset_preview.DARK_PIN + " }"
    typeset_preview.check_dark_mode_pin(pinned)                      # complete: no complaint
    typeset_preview.check_dark_mode_pin("<html>no dark block here</html>")
    try:
        typeset_preview.check_dark_mode_pin("@media (prefers-color-scheme: dark) { html { background: #fffff8; } }")
    except SystemExit as e:
        assert "invisible in dark mode" in str(e)
    else:
        raise AssertionError("a light-pinned page with inverted diagram colours was accepted")


def test_a_refused_proof_is_read_from_the_run_s_own_accounting(tmp_path):
    # 0806.1324's lemma B.7 was read by S3 and its graph rejected for circularity;
    # the margin was blank beside it, which reads as "nobody tried" (Joe, 2026-09-22).
    import json as _json
    run = tmp_path
    (run / "accounting/S3/S3-a001").mkdir(parents=True)
    (run / "accounting/S3/S3-a001/S3.loop.json").write_text(_json.dumps({"items": [
        {"id": "p:__p60", "paper": "p", "status": "rejected", "attempts": [{"attempt": 1}],
         "reason": "contract: the steps derive node 5 -> node 6 -> node 5"},
        {"id": "p:__p59", "paper": "p", "status": "accepted", "reason": ""},
        {"id": "q:__p1", "paper": "q", "status": "rejected", "reason": "another paper"}]}))
    (run / "accounting/S4/S4-a001").mkdir(parents=True)
    (run / "accounting/S4/S4-a001/S4.select.json").write_text(_json.dumps({"items": [
        {"id": "p:p-inflight-0002:L259-265", "status": "deferred", "reason": "cap 30 per paper"},
        {"id": "p:p-leaf-0001:L292-305", "status": "accepted", "reason": ""}]}))
    got = margin.stage_outcomes(run, "p")
    assert list(got["proofs"]) == ["p:__p60"]                       # accepted and other papers left out
    assert got["proofs"]["p:__p60"]["attempts"] == 1
    assert got["regions"] == [{"id": "p-inflight-0002", "lines": [259, 265],
                               "status": "deferred", "why": "cap 30 per paper"}]
