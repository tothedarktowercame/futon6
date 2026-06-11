"""Skolem audit: each failure class fires on a synthetic tree, and the
two-channel discrimination (real violation vs detector blindness) holds."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from mission_scope_bindings import analyze_tree, content_grade


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

    # The pattern is used in DERIVE but never introduced anywhere — a
    # confirmed free variable.
    free = {f["ident"]: f["confirmed"] for f in r["free-variables"]}
    assert free == {"structure/never-introduced": True}

    assert r["bound-items"] == 3
    assert r["used-items"] == 1
