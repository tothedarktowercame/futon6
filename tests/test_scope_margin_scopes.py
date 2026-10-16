import pytest

from scripts.scope_margin_scopes import scopes_data


def fixture():
    source = "mark one\nproof premise conclusion\nquoted scope\nline scope\n"
    starts = [0]
    for index, char in enumerate(source):
        if char == "\n":
            starts.append(index + 1)
    payload = {
        "summary": {"run": "tiny-run", "nodes": 2, "edges": 1, "scopes": 2,
                    "proofs": 1, "carving": {"now": {"regions": 2}}},
        "starts": starts,
        "kinds": [["concept", "term", "concept meaning"],
                  ["sentence", "clause", "sentence meaning"]],
        "marks": [[0, 4, 0, 1, "grounded"], [5, 8, 1, 0, "ungrounded"]],
        "notes": [
            {"type": "proof", "id": "proof0", "lo": 2, "hi": 2, "nodes": [
                {"id": "n1", "kind": "claim", "gloss": "premise", "verdict": "agrees",
                 "at": [[9, 16]]},
                {"id": "n2", "kind": "claim", "gloss": "conclusion", "verdict": "agrees",
                 "at": [[17, 35]]},
            ], "edges": [{"relation": "implies", "premises": ["n1"],
                           "conclusion": "n2", "warrant": "claim"}]},
            {"type": "refused", "id": "proof1", "lo": 1, "hi": 1,
             "status": "rejected", "why": "bad graph"},
            {"type": "region", "id": "region0", "scopes": [
                {"scope": "s1", "kind": "connection", "span": [34, 46],
                 "lines": [3, 3], "fill": "quoted scope", "flags": [],
                 "mock": {"verdict": "kept"}},
                {"scope": "s2", "kind": "connection", "span": None,
                 "lines": [4, 4], "fill": "line scope", "flags": ["bare-noun"],
                 "mock": {"verdict": "held"}},
            ]},
        ],
        "carved": [{"id": "c1", "type": "in-proof", "section": "Proof",
                    "lines": [2, 2]},
                   {"id": "c2", "type": "section-lead", "section": "Lead",
                    "lines": [3, 3]}],
        "glossary": {"S1 mark kinds": {"concept": "concept meaning",
                                         "sentence": "sentence meaning"},
                     "S3 node kinds": {"claim": "claim meaning"},
                     "S3 warrants": {"claim": "stated"},
                     "S4 scope kinds": {"connection": "connection meaning"},
                     "This page": {"in-proof": "inside proof",
                                   "section-lead": "section opening"}},
    }
    return payload, source


def test_scopes_data_maps_all_three_layers():
    payload, source = fixture()
    data = scopes_data(payload, source)
    assert len(data["spans"]) == 10
    assert data["spans"][1]["weight"] == 0.5
    assert data["spans"][1]["layer"] == "s1-clause"
    assert [layer["id"] for layer in data["layers"]] == ["s1-term", "s1-clause", "s3", "s4"]
    assert data["edges"] == [{"from": "proof0-n1", "to": "proof0-n2",
                              "relation": "implies", "label": "claim"}]
    line_scope = next(span for span in data["spans"] if span["id"] == "region0-s2")
    assert [line_scope["start"], line_scope["end"]] == [payload["starts"][3],
                                                         payload["starts"][4]]
    assert line_scope["weight"] == 0.6
    quoted_scope = next(span for span in data["spans"] if span["id"] == "region0-s1")
    assert [quoted_scope["start"], quoted_scope["end"]] == [34, 46]
    assert quoted_scope["weight"] == 1.0
    assert source[34:46] == "quoted scope"
    proof_body = next(span for span in data["spans"] if span["id"] == "proof0-body")
    assert [proof_body["start"], proof_body["end"], proof_body["weight"]] == [9, 34, 0.3]
    refused = next(span for span in data["spans"] if span["id"] == "proof1-refused")
    assert refused["weight"] == 0.12
    carved = {span["id"]: span for span in data["spans"] if span["id"].startswith("carved-")}
    assert carved["carved-c1"]["weight"] == 0.14
    assert carved["carved-c2"]["weight"] == 0.3
    assert data["legend"]["carved/in-proof"] == "inside proof"
    assert data["legend"]["carved/section-lead"] == "section opening"
    used = {span["kind"] for span in data["spans"]}
    used |= {edge["relation"] for edge in data["edges"]}
    used |= {edge["label"] for edge in data["edges"]}
    assert used <= data["legend"].keys()


def test_scopes_data_rejects_mark_past_source():
    payload, source = fixture()
    payload["marks"][0][1] = len(source) + 1
    with pytest.raises(ValueError, match="outside source length"):
        scopes_data(payload, source)


def test_scopes_data_rejects_undefined_carved_type():
    payload, source = fixture()
    payload["carved"][0]["type"] = "mystery"
    with pytest.raises(ValueError, match="has no defined type"):
        scopes_data(payload, source)
