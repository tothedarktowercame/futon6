import pytest

from scripts.scope_margin_scopes import scopes_data


def fixture():
    source = "mark one\nproof premise conclusion\nquoted scope\nline scope\n"
    starts = [0]
    for index, char in enumerate(source):
        if char == "\n":
            starts.append(index + 1)
    payload = {
        "summary": {"run": "tiny-run", "nodes": 2, "edges": 1, "scopes": 2},
        "starts": starts,
        "kinds": [["concept", "term", "concept meaning"]],
        "marks": [[0, 4, 0, 1, "grounded"], [5, 8, 0, 0, "ungrounded"]],
        "notes": [
            {"type": "proof", "id": "proof0", "nodes": [
                {"id": "n1", "kind": "claim", "gloss": "premise", "verdict": "agrees",
                 "at": [[9, 16]]},
                {"id": "n2", "kind": "claim", "gloss": "conclusion", "verdict": "agrees",
                 "at": [[17, 35]]},
            ], "edges": [{"relation": "implies", "premises": ["n1"],
                           "conclusion": "n2", "warrant": "claim"}]},
            {"type": "region", "id": "region0", "scopes": [
                {"scope": "s1", "kind": "connection", "span": [34, 46],
                 "lines": [3, 3], "fill": "quoted scope", "flags": [],
                 "mock": {"verdict": "kept"}},
                {"scope": "s2", "kind": "connection", "span": None,
                 "lines": [4, 4], "fill": "line scope", "flags": ["bare-noun"],
                 "mock": {"verdict": "held"}},
            ]},
        ],
        "glossary": {"S1 mark kinds": {"concept": "concept meaning"},
                     "S3 node kinds": {"claim": "claim meaning"},
                     "S3 warrants": {"claim": "stated"},
                     "S4 scope kinds": {"connection": "connection meaning"}},
    }
    return payload, source


def test_scopes_data_maps_all_three_layers():
    payload, source = fixture()
    data = scopes_data(payload, source)
    assert len(data["spans"]) == 6
    assert data["spans"][1]["weight"] == 0.5
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
    used = {span["kind"] for span in data["spans"]}
    used |= {edge["relation"] for edge in data["edges"]}
    used |= {edge["label"] for edge in data["edges"]}
    assert used <= data["legend"].keys()


def test_scopes_data_rejects_mark_past_source():
    payload, source = fixture()
    payload["marks"][0][1] = len(source) + 1
    with pytest.raises(ValueError, match="outside source length"):
        scopes_data(payload, source)
