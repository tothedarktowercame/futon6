"""Skolem audit: each failure class fires on a synthetic tree, and the
two-channel discrimination (real violation vs detector blindness) holds."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import mission_scope_bindings
from mission_scope_bindings import analyze_tree, attributed_code_files, content_grade


def scope(scope_id, binder, phase, title, ends, position=0, end=0):
    return {
        "scope-id": scope_id,
        "binder-type": binder,
        "ends": [
            {"role": "entity", "ident": "M-test"},
            {"role": "environment", "name": title, "phase": phase},
            {"role": "heading", "level": 2, "title": title},
            *ends,
        ],
        "hx/content": {"match": title, "position": position, "end": end},
    }


# Layout: MAP region [0,200) binds two files + a mission; DERIVE region
# [200,400) uses one file (in ends) and mentions the mission (text only —
# detector blind there); the second file appears nowhere in the body.
TEXT = (
    "MAP cites alpha.clj and beta.clj and M-elsewhere here."
    + " " * 145
    + "DERIVE builds on alpha.clj and follows M-elsewhere."
    + " " * 148
)

TREE = {
    "mission": "M-test",
    "scope-hyperedges": [
        scope(
            "M-test:scope-000",
            "eightfold-phase",
            "map",
            "MAP",
            [
                {"role": "source", "kind": "file", "ref": "src/alpha.clj"},
                {"role": "source", "kind": "file", "ref": "src/beta.clj"},
                {"role": "mission", "ident": "M-elsewhere", "relation": "relates-to"},
            ],
            position=0,
            end=200,
        ),
        scope(
            "M-test:scope-001",
            "eightfold-phase",
            "derive",
            "DERIVE",
            [
                {"role": "source", "kind": "file", "ref": "src/alpha.clj"},
                {"role": "pattern", "ident": "structure/never-introduced"},
            ],
            position=200,
            end=400,
        ),
        scope("M-test:scope-002", "map-item", "map", "Empty map item", [], position=0, end=10),
        scope(
            "M-test:scope-003",
            "loose-section",
            "loose",
            "Concept-only section",
            [{"role": "concept", "term": "agents"}],
            position=0,
            end=10,
        ),
    ],
}


def test_content_grades():
    grades = [content_grade(s) for s in TREE["scope-hyperedges"]]
    assert grades == ["bound", "bound", "vacuous", "concept-only"]


def test_skolem_classes():
    r = analyze_tree(TREE, TEXT)
    assert r["spine"] is True

    assert [v["scope-id"] for v in r["vacuous"]] == ["M-test:scope-002"]
    assert r["concept-only"] == 1

    # beta.clj: bound in MAP, absent from the body in BOTH channels — a
    # confirmed unused binding. M-elsewhere: absent from body ENDS but present
    # in body TEXT — detector blindness, not a confirmed violation.
    unused = {u["ident"]: u["confirmed"] for u in r["unused-bindings"]}
    assert unused == {"src/beta.clj": True, "M-elsewhere": False}
    verdicts = {u["ident"]: u["verdict"] for u in r["unused-bindings"]}
    assert verdicts == {"src/beta.clj": "confirmed-unused", "M-elsewhere": "doc-used"}

    # The pattern is used in DERIVE but never introduced anywhere — a
    # confirmed free variable.
    free = {f["ident"]: f["confirmed"] for f in r["free-variables"]}
    assert free == {"structure/never-introduced": True}

    assert r["bound-items"] == 3
    assert r["used-items"] == 1


def test_attributed_code_files_joins_commit_mission_to_edits():
    edges = [
        {
            "hx/type": "code/v05/commit→mission",
            "hx/endpoints": ["sha1", "M-test", "dir:sha1→M-test"],
        },
        {
            "hx/type": "code/v05/edits",
            "hx/endpoints": ["sha1", "futon6/src/beta.clj", "dir:sha1→futon6/src/beta.clj"],
        },
        {
            "hx/type": "code/v05/commit→mission",
            "hx/endpoints": ["sha2", "M-other", "dir:sha2→M-other"],
        },
        {
            "hx/type": "code/v05/edits",
            "hx/endpoints": ["sha2", "src/foreign.clj", "dir:sha2→src/foreign.clj"],
        },
    ]
    assert attributed_code_files("M-test", edges) == {"futon6/src/beta.clj"}


def test_bound_file_can_be_code_discharged_by_attributed_commit():
    edges = [
        {
            "hx/type": "code/v05/commit→mission",
            "hx/endpoints": ["sha1", "M-test", "dir:sha1→M-test"],
            "hx/props": {"relation/provenance": "trailer"},
        },
        {
            "hx/type": "code/v05/edits",
            "hx/endpoints": ["sha1", "futon6/src/beta.clj", "dir:sha1→futon6/src/beta.clj"],
        },
    ]
    r = analyze_tree(TREE, TEXT, code_edges=edges)
    verdicts = {u["ident"]: u["verdict"] for u in r["unused-bindings"]}
    confirmed = {u["ident"]: u["confirmed"] for u in r["unused-bindings"]}

    assert verdicts["src/beta.clj"] == "code-discharged"
    assert confirmed["src/beta.clj"] is False
    assert verdicts["M-elsewhere"] == "doc-used"


def test_unattributed_or_other_mission_edits_do_not_discharge_binding():
    edges = [
        {
            "hx/type": "code/v05/commit→mission",
            "hx/endpoints": ["sha1", "M-other", "dir:sha1→M-other"],
        },
        {
            "hx/type": "code/v05/edits",
            "hx/endpoints": ["sha1", "src/beta.clj", "dir:sha1→src/beta.clj"],
        },
        {
            "hx/type": "code/v05/edits",
            "hx/endpoints": ["sha-unattributed", "src/beta.clj", "dir:sha-unattributed→src/beta.clj"],
        },
    ]
    r = analyze_tree(TREE, TEXT, code_edges=edges)
    verdicts = {u["ident"]: u["verdict"] for u in r["unused-bindings"]}
    assert verdicts["src/beta.clj"] == "confirmed-unused"


def test_fetch_code_edges_uses_store_boundary_with_mocked_urlopen(monkeypatch):
    seen_urls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"hyperedges": [{"hx/type": "code/v05/edits"}]}).encode()

    def fake_urlopen(req, timeout):
        seen_urls.append(req.full_url)
        assert timeout == 5
        return Response()

    monkeypatch.setattr(mission_scope_bindings, "urlopen", fake_urlopen)

    edges = mission_scope_bindings.fetch_code_edges("http://store.test")

    assert edges == [{"hx/type": "code/v05/edits"}, {"hx/type": "code/v05/edits"}]
    assert len(seen_urls) == 2
    assert "code%2Fv05%2Fcommit%E2%86%92mission" in seen_urls[0]
    assert "code%2Fv05%2Fedits" in seen_urls[1]
