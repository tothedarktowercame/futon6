"""Fixtures-only tests for pattern_roads (no live data, no substrate)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pattern_roads as pr


def _scope(sid, mission, ident, state="linked", ref=None):
    return {"scope/id": sid, "mission": mission, "ident": ident,
            "ref": ref, "state": state}


def test_shared_patterns_give_one_row_per_pair_pattern():
    scopes = [
        _scope("M-a/pattern/P", "M-a", "P"), _scope("M-b/pattern/P", "M-b", "P"),
        _scope("M-c/pattern/P", "M-c", "P"),
        _scope("M-a/pattern/Q", "M-a", "Q"), _scope("M-b/pattern/Q", "M-b", "Q"),
    ]
    positioned = {"M-a", "M-b", "M-c"}
    dating = {s["scope/id"]: {"dated": "2026-01-01T00:00:00+00:00",
                              "date_source": "blame"} for s in scopes}
    t = pr.build_roads(scopes, positioned, dating)
    rows = t["roads"]
    assert len(rows) == 4  # 3 P-rows (a-b, a-c, b-c) + 1 Q-row (a-b)
    pq = sorted(r["pattern"] for r in rows if r["a"] == "M-a" and r["b"] == "M-b")
    assert pq == ["P", "Q"]  # the pair sharing both gets 2 rows
    assert t["patterns"]["P"]["n_missions"] == 3


def test_non_linked_scope_is_tallied_not_roaded():
    scopes = [
        _scope("M-a/pattern/P", "M-a", "P"),
        _scope("M-b/pattern/P", "M-b", "P", state="detached"),
    ]
    t = pr.build_roads(scopes, {"M-a", "M-b"}, {})
    assert t["roads"] == []
    assert t["excluded_states"] == {"detached": 1}
    assert t["patterns"]["P"]["missions"] == ["M-a"]


def test_off_map_mission_lands_in_off_map_missions():
    scopes = [
        _scope("M-a/pattern/P", "M-a", "P"),
        _scope("M-x/pattern/P", "M-x", "P"),
    ]
    t = pr.build_roads(scopes, {"M-a"}, {})
    p = t["patterns"]["P"]
    assert p["missions"] == ["M-a"]
    assert p["off_map_missions"] == ["M-x"]
    assert t["roads"] == []


def test_earliest_dated_scope_wins_and_weakest_source_propagates():
    scopes = [
        _scope("M-a/pattern/P", "M-a", "P"),
        _scope("M-a/pattern/P-dup", "M-a", "P"),
        _scope("M-b/pattern/P", "M-b", "P"),
    ]
    dating = {
        "M-a/pattern/P": {"dated": "2026-05-01T00:00:00+00:00",
                          "date_source": "mission-fallback"},
        "M-a/pattern/P-dup": {"dated": "2026-01-01T00:00:00+00:00",
                              "date_source": "blame"},
        "M-b/pattern/P": {"dated": "2026-03-01T00:00:00+00:00",
                          "date_source": "blame"},
    }
    t = pr.build_roads(scopes, {"M-a", "M-b"}, dating)
    (r,) = t["roads"]
    assert r["a_scope"] == "M-a/pattern/P-dup"  # earliest-dated end
    assert r["constructed"] == "2026-03-01T00:00:00+00:00"  # later of the two
    assert r["constructed_source"] == "blame"


def test_pair_diffs():
    roads = [{"a": "M-a", "b": "M-b", "pattern": "P",
              "a_scope": "x", "b_scope": "y",
              "constructed": None, "constructed_source": "undated"}]
    text = [["M-a", "M-c", 3], ["M-b", "M-a", 0]]
    d = pr.pair_diffs(roads, text)
    assert d["text_only_pairs"] == [["M-a", "M-c"]]
    assert d["scope_only_pairs"] == []


def test_bare_and_namespaced_idents_are_one_pattern():
    ref = "futon3/library/agent/evidence-over-assertion.flexiarg"
    scopes = [
        {"scope/id": "M-a/pattern/x", "mission": "M-a", "ident": "agent/evidence-over-assertion",
         "ref": ref, "state": "linked"},
        {"scope/id": "M-b/pattern/x", "mission": "M-b", "ident": "evidence-over-assertion",
         "ref": ref, "state": "linked"},
    ]
    table = pr.build_roads(scopes, {"M-a", "M-b"}, {})
    assert list(table["patterns"]) == ["agent/evidence-over-assertion"]
    assert table["patterns"]["agent/evidence-over-assertion"]["idents"] == [
        "agent/evidence-over-assertion", "evidence-over-assertion"]
    assert [(r["a"], r["b"]) for r in table["roads"]] == [("M-a", "M-b")]
