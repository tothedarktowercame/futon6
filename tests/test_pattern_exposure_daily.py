"""Fixtures-only tests for pattern_exposure_daily (fake fetch, no live JVM)."""
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pattern_exposure_daily as ped

KEYS = {"agent/evidence-over-assertion", "orchestration/consent-gate"}
BYNAME = {"evidence-over-assertion": "agent/evidence-over-assertion",
          "consent-gate": "orchestration/consent-gate"}


def _entry(eid, day, ids):
    body = '{"event" "context-retrieval", "results" [' + ",".join(
        '{"id" "%s", "score" 1.0}' % i for i in ids) + "]}"
    return {"evidence/id": eid, "evidence/at": f"{day}T12:00:00Z",
            "evidence/body": body}


def test_result_id_parsing_edn_and_json():
    e_json = {"evidence/body": '{"results": [{"id": "a/b", "score": 0.9}]}'}
    e_edn = {"evidence/body": '{:results [{:id "a/b" :score 0.9}]}'}
    e_bad = {"evidence/body": "{not parseable"}
    assert ped.retrieval_result_ids(e_json) == ["a/b"]
    assert ped.retrieval_result_ids(e_edn) == ["a/b"]
    assert ped.retrieval_result_ids(e_bad) == []


def test_bare_and_namespaced_ids_share_the_flexiarg_key():
    day = "2026-11-24"
    state = ped.empty_state()
    ped.absorb_entries(state, [
        _entry("e1", day, ["agent/evidence-over-assertion"]),
        _entry("e2", day, ["evidence-over-assertion"]),  # bare, same flexiarg
        _entry("e3", day, ["no-such-pattern-id"]),
    ], KEYS, BYNAME)
    exp = state["exposure"]
    assert exp["agent/evidence-over-assertion"][day] == 2
    assert state["unmatched"] == {"no-such-pattern-id": 1}
    assert state["turns"][day] == 3


def test_failed_day_lands_in_gaps_not_zero():
    def fetch(since, before, limit):
        if since.startswith("2026-11-24"):
            raise TimeoutError("dead JVM")
        return [_entry("e1", "2026-11-23", ["consent-gate"])]
    state = ped.run(fetch, KEYS, BYNAME,
                    start_day=date(2026, 11, 24), earliest=date(2026, 11, 23))
    table = ped.render(state)
    assert table["gaps"] == ["2026-11-24"]
    assert "2026-11-24" not in table["turns"]  # never rendered as a dark day
    assert table["turns"] == {"2026-11-23": 1}
    assert table["exposure"]["orchestration/consent-gate"] == {"2026-11-23": 1}


def test_day_hitting_limit_is_split_not_accepted():
    day = date(2026, 11, 24)
    calls = []

    def fetch(since, before, limit):
        calls.append((since, before))
        if "T00:00:00Z" in since and "T00:00:00Z" in before:
            return [_entry(f"full-{i}", day.isoformat(), [])
                    for i in range(limit)]  # exactly limit: suspect
        return []  # hour pages
    state = ped.empty_state()
    entries = ped.fetch_day(day, fetch=fetch, limit=5)
    assert len(calls) == 1 + 24  # day page, then 24 hour pages
    assert entries == []  # hour pages empty: the day page was a lie, not data


def test_run_crosses_an_outage_and_stops_after_a_long_empty_run():
    # log: 11-24 and 11-10 have turns; 11-11..11-23 empty (an outage);
    # nothing before 11-10. The walk must cross the outage (the bad case:
    # stopping at 11-23 loses 11-10) and record the outage days as zero.
    def fetch(since, before, limit):
        for d in ("2026-11-24", "2026-11-10"):
            if since.startswith(d):
                return [_entry("e-" + d, d, [])]
        return []
    state = ped.run(fetch, KEYS, BYNAME, start_day=date(2026, 11, 24),
                    earliest=date(2026, 1, 1))
    assert state["turns"]["2026-11-24"] == 1
    assert state["turns"]["2026-11-10"] == 1
    assert state["turns"]["2026-11-15"] == 0        # inside the log: known zero
    assert "2026-11-09" not in state["turns"]       # before the log: not recorded
    assert state["days_queried"] == 15 + ped.EMPTY_RUN_STOP
