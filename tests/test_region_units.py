"""S4 cites sentence units and quotes from them, so a scope has an extent."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import expository_json
import region_units

TEXT = ("\\section{Introduction}\\label{intro}\n"
        "A category is accessible if it is $\\lambda$-accessible for some $\\lambda$. "
        "We recall this from \\cite{AR}, cf. the discussion there. "
        "Write $\\Hom(S,\\Sigma^{i}S)=0$ for $0<i<n.$ Then the result follows.\n")
STARTS = [0] + [i + 1 for i, c in enumerate(TEXT) if c == "\n"]
UNITS = region_units.units_for(TEXT, STARTS, 1, 3)


def test_sentences_split_outside_math_and_not_on_abbreviations():
    texts = [u["text"] for u in UNITS]
    assert any(t.startswith("A category is accessible") for t in texts)   # the heading is not glued on
    assert any("cf. the discussion there." in t for t in texts)          # cf. is not a sentence end
    assert any("$0<i<n.$" in t for t in texts)                            # the period is inside math
    assert not any(t.startswith("\\section") for t in texts)              # markup-only: no prose to read


def test_a_unit_id_names_its_line_and_its_offsets_are_the_papers():
    u = UNITS[0]
    assert u["id"].startswith(f"L{u['line']}-")
    assert TEXT[u["start"]:u["end"]] == u["text"]


def test_a_paraphrased_fill_is_recorded_not_refused():
    """A fill the model wrote rather than quoted is a finding, not a contract breach.

    It was refused, and since anything problems() returns rejects the whole
    passage, one paraphrased scope discarded every scope beside it. The two are
    told apart by :fill-span, which to_edn writes only for the quoted one.
    """
    kinds = {"connection": "structure"}
    quoted = {"scopes": [{"kind": "connection", "units": [UNITS[0]["id"]],
                          "fill": "accessible if it is $\\lambda$-accessible", "held_reason": ""}]}
    made_up = {"scopes": [{"kind": "connection", "units": [UNITS[0]["id"]],
                           "fill": "the theory of accessible categories", "held_reason": ""}]}
    assert expository_json.problems(quoted, 1, 3, kinds, UNITS) == []
    assert expository_json.problems(made_up, 1, 3, kinds, UNITS) == []

    cand = {"paper-id": "p", "passage-id": "p:r:L1-3", "window-lines": [1, 3], "units": UNITS}
    assert ":fill-span" in expository_json.to_edn(quoted, cand, kinds, "m")
    assert ":fill-span" not in expository_json.to_edn(made_up, cand, kinds, "m")


def test_a_scope_citing_no_unit_of_this_region_is_still_refused():
    """Structural, not a judgement of the words: the model cited nothing citable."""
    kinds = {"connection": "structure"}
    doc = {"scopes": [{"kind": "connection", "units": ["L99-dead"],
                       "fill": "accessible if it is $\\lambda$-accessible", "held_reason": ""}]}
    assert "cites no unit of this region" in expository_json.problems(doc, 1, 3, kinds, UNITS)[0]


def test_a_quote_may_cross_a_sentence_edge():
    """The carver splits on the full stop; what a scope is about need not.

    Two consecutive sentences quoted word for word read as the model's own
    words, because each cited unit was searched alone.
    """
    import re
    a, b = UNITS[0], UNITS[1]
    fill = (a["text"][-24:] + " " + b["text"][:20]).strip()
    at = region_units.locate_in_units(fill, [a, b])
    assert at is not None
    # The span covers real source, so between the two sentences it holds whatever
    # the paper holds -- a newline, two spaces -- not the single space joining them.
    assert re.sub(r"\s+", " ", TEXT[at[0]:at[1]]).strip() == re.sub(r"\s+", " ", fill).strip()
    assert at[0] >= a["start"] and at[1] <= b["end"]

    far = dict(b, start=b["start"] + 9000, end=b["end"] + 9000)
    assert region_units.locate_in_units(fill, [a, far]) is None


def test_the_edn_carries_the_scope_s_extent_and_the_fill_s_own_span():
    kinds = {"connection": "structure"}
    doc = {"scopes": [{"kind": "connection", "units": [UNITS[0]["id"]],
                       "fill": "accessible if it is $\\lambda$-accessible", "held_reason": ""}]}
    cand = {"paper-id": "p", "passage-id": "p:r:L1-3", "window-lines": [1, 3], "units": UNITS}
    edn = expository_json.to_edn(doc, cand, kinds, "m")
    assert ':units ["' + UNITS[0]["id"] + '"]' in edn
    at = region_units.locate_in_units(doc["scopes"][0]["fill"], UNITS)
    assert f":fill-span [{at[0]} {at[1]}]" in edn
    assert TEXT[at[0]:at[1]] == "accessible if it is $\\lambda$-accessible"


def test_a_candidate_without_units_is_refused_under_the_contract():
    import run_contract
    assert run_contract.missing_expository_inputs({"units": []}) == ["units"]
    assert run_contract.missing_expository_inputs({"units": UNITS}) == []
