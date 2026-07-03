"""Tests for the PROOF-MINE runner (proof-mine-runner-spec.md). Fixtures + monkeypatch only —
no network, no GPU, no live XTDB. Proves the load-bearing invariants: D10 budget truncation,
D6 canonical bridge + quarantine, D3 null-normalization / skip-and-continue / resume-append,
and the D5 abort bands."""
import json
import os
import sys

_SCRIPTS = os.path.join(os.path.dirname(__file__), os.pardir, "scripts")
if os.path.abspath(_SCRIPTS) not in sys.path:
    sys.path.insert(0, os.path.abspath(_SCRIPTS))

import proof_mine_dossier as pmd
import proof_mine as pm
import proof_mine_land_prep as plp


# ---------------------------------------------------------------- D10: dossier budget
def _fixture_repo(tmp_path, stem, body_paras=200):
    doc = tmp_path / "myrepo" / "holes" / "missions" / ("M-%s.md" % stem)
    doc.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join("paragraph %d — lorem ipsum dolor sit amet consectetur." % i
                     for i in range(body_paras))
    doc.write_text("# M-%s\n\n**Status:** IDENTIFY\n\n%s\n" % (stem, body))
    return doc


def test_dossier_budget_truncates_and_logs(tmp_path):
    _fixture_repo(tmp_path, "budgetmission")
    d = pmd.assemble("M-budgetmission", repos_root=str(tmp_path), budget_tokens=200)
    assert d["doc_found"]
    assert d["mission"] == "myrepo-d/mission/budgetmission"
    assert d["truncations"], "a tiny budget MUST log truncations — no silent caps (D10)"
    assert pmd.est_tokens(d["text"]) <= 260, "text should respect the ~200-token budget"


def test_dossier_missing_doc_is_flagged_not_raised(tmp_path):
    d = pmd.assemble("M-nope-not-here", repos_root=str(tmp_path))
    assert d["doc_found"] is False
    assert d["no_code_trail"] is True
    assert any("NOT FOUND" in n for n in d["notes"])


def test_dossier_status_line_extracted(tmp_path):
    _fixture_repo(tmp_path, "statusmission", body_paras=3)
    d = pmd.assemble("M-statusmission", repos_root=str(tmp_path))
    assert "IDENTIFY" in d["status"]


# ---------------------------------------------------------------- D6: canonical bridge + quarantine
def test_resolve_ref_bridges_and_quarantines():
    idx = {"good": "myrepo-d/mission/good"}
    # a known stem resolves to canonical
    ref, ok = pm.resolve_ref("M-good", idx)
    assert ok and ref == "myrepo-d/mission/good"
    # an unknown mission-shaped ref is flagged for quarantine (never minted)
    ref, ok = pm.resolve_ref("M-unknownmission", idx)
    assert ok is False
    # a non-mission ref (sorry/sha/method) passes through intact
    ref, ok = pm.resolve_ref("sorry/some-hole", idx)
    assert ok is True and ref == "sorry/some-hole"


def test_build_record_normalizes_nulls_and_quarantines_bad_target():
    idx = {"good": "myrepo-d/mission/good"}
    dossier = {"mission": "myrepo-d/mission/good", "stem": "good",
               "text": "== DOC ==\nthe verbatim witness span lives here",
               "no_code_trail": True, "truncations": []}
    raw = {
        "mission": "myrepo-d/mission/good",
        "discharges": [
            {"target": "sorry/good-hole", "discharged_by": "abc1234",
             "grade": "discharged", "witness": "the verbatim witness span"},
            {"target": "M-ghostmission", "grade": "open", "witness": None},   # unresolvable → quarantine
        ],
        "endpoints": None,          # the 2026-06-25 null-field lesson
        "rule_candidates": None,
    }
    rec, quarantine = pm.build_record(dossier["mission"], raw, dossier, idx)
    assert rec["endpoints"] == [], "null endpoints must normalize to [] (D3), not crash"
    assert rec["rule_candidates"] == []
    targets = [d["target"] for d in rec["discharges"]]
    assert "sorry/good-hole" in targets
    assert "M-ghostmission" not in targets, "unresolvable target must NOT be minted"
    assert any(q["raw"] == "M-ghostmission" for q in quarantine), "it must land in quarantine"
    # witness verbatim detection
    d0 = rec["discharges"][0]
    assert d0["witness_verbatim"] is True
    assert rec["pair_unverified"] is True   # the pilot's ⚠pair, until the pairs corpus is on disk


# ---------------------------------------------------------------- D3: resilience / resume / append
def _canned_assemble(stem, **kwargs):
    if stem == "good":
        return {"doc_found": True, "mission": "myrepo-d/mission/good", "stem": "good",
                "text": "== DOC ==\nfirst line span", "no_code_trail": True, "truncations": []}
    return {"doc_found": False, "mission": None, "stem": stem, "notes": ["NOT FOUND"],
            "no_code_trail": True, "text": "", "truncations": []}


def test_sweep_bad_mission_costs_one_and_resume_skips(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "assemble", _canned_assemble)
    monkeypatch.setattr(pm, "_MISSION_INDEX", {"good": "myrepo-d/mission/good"})
    out = str(tmp_path / "pm-out")

    pm.run_sweep(["good", "this-one-has-no-doc"], backend="stub", model="x",
                 out_dir=out, resume=False)
    lines = [l for l in open(os.path.join(out, "proof-mine.jsonl"))]
    assert len(lines) == 1, "the bad mission is skipped, the good one lands — one bad item ≠ total loss"
    rec = json.loads(lines[0])
    assert rec["stem"] == "good"
    assert rec["endpoints"] == []                      # stub emitted null; normalization held
    assert os.path.exists(os.path.join(out, "proof-mine-status.json"))

    # resume: the good mission is already in the artifact → not re-appended
    pm.run_sweep(["good", "this-one-has-no-doc"], backend="stub", model="x",
                 out_dir=out, resume=True)
    lines2 = [l for l in open(os.path.join(out, "proof-mine.jsonl"))]
    assert len(lines2) == 1, "--resume must skip already-mined missions (D3)"


def test_load_done_and_append(tmp_path):
    p = str(tmp_path / "a.jsonl")
    pm.append_jsonl(p, {"stem": "m1"})
    pm.append_jsonl(p, {"stem": "m2"})
    assert pm.load_done(p) == {"m1", "m2"}


# ---------------------------------------------------------------- witness snap-to-span + endpoints
def test_snap_witness_recovers_real_span():
    dossier = "== DOC ==\nThe durable clock record persists across teardown\nand reconstitutes on boot."
    # a paraphrase that overlaps a real span → recovers the verbatim span
    got = pm.snap_witness("the durable clock record persists across teardown reliably", dossier)
    assert got is not None and "durable clock record persists across teardown" in got
    # already-verbatim (mod whitespace) → returned as-is
    assert pm.snap_witness("reconstitutes on boot", dossier) is not None
    # unrelated text → not grounded → None (caller marks :unsupported)
    assert pm.snap_witness("something entirely unrelated to any dossier content", dossier) is None


def test_build_record_snaps_or_unsupports_witness():
    idx = {"good": "myrepo-d/mission/good"}
    dossier = {"mission": "myrepo-d/mission/good", "stem": "good",
               "text": "== DOC ==\nthe order-independent retract repair landed in 84d4d04",
               "no_code_trail": True, "truncations": []}
    raw = {"mission": "myrepo-d/mission/good", "endpoints": [], "rule_candidates": [],
           "discharges": [
               {"target": "sorry/a", "grade": "discharged",
                "witness": "the order independent retract repair landed"},   # paraphrase → snapped
               {"target": "sorry/b", "grade": "open",
                "witness": "a totally invented claim not in the dossier"}]}    # → :unsupported
    rec, _q = pm.build_record(dossier["mission"], raw, dossier, idx)
    d0, d1 = rec["discharges"]
    assert d0["witness_verbatim"] is True and d0["witness"] in dossier["text"]
    assert d1["witness_verbatim"] is False and d1["witness"] == ":unsupported"


def test_line_cite_witness_extraction():
    dossier = "L-map test\nsecond line has the real evidence here\nthird line unrelated"
    # single line ref
    assert pm._extract_line_witness("L2", dossier) == "second line has the real evidence here"
    # range picks contiguous lines
    got = pm._extract_line_witness("L1-L2", dossier)
    assert "L-map test" in got and "real evidence" in got
    # out-of-range / no ref → None (caller falls back to snap or :unsupported)
    assert pm._extract_line_witness("L99", dossier) is None
    assert pm._extract_line_witness("just prose", dossier) is None


def test_build_record_prefers_line_cite():
    idx = {"good": "myrepo-d/mission/good"}
    dossier = {"mission": "myrepo-d/mission/good", "stem": "good",
               "text": "L0 header\nthe repair landed in commit 84d4d04 on the clock path\ntail",
               "no_code_trail": True, "truncations": []}
    raw = {"mission": "myrepo-d/mission/good", "endpoints": [], "rule_candidates": [],
           "discharges": [{"target": "sorry/a", "grade": "discharged", "witness": "L2"}]}
    rec, _q = pm.build_record(dossier["mission"], raw, dossier, idx)
    d0 = rec["discharges"][0]
    assert d0["witness_verbatim"] is True
    assert "the repair landed in commit 84d4d04" in d0["witness"]


def test_ep_match_is_fair_but_not_promiscuous():
    gold = ["clock/clocked-on hyperedge (the durable clock record)", "agent nodes (agent:<id>)"]
    assert pm._ep_match("clock/clocked-on hyperedge", gold) is True       # shares ref-shaped token
    assert pm._ep_match("held/on-mission hyperedge", gold) is False       # no distinctive overlap
    assert pm._ep_match("the and for with", gold) is False                # stopwords only → no match


# ---------------------------------------------------------------- D5: gold scoring + abort bands
# ---------------------------------------------------------------- D7: landing prep (cleaning)
def test_clean_sha_and_repo_of():
    assert plp.clean_sha("472f93e (capability-star-map arc + FutonZero §22)") == "472f93e"
    assert plp.clean_sha("06dac23 Complete M-transport-adapters") == "06dac23"
    assert plp.clean_sha("REAL: futon3a/src/futon/peripheral/pattern_author.clj") is None
    assert plp.clean_sha("NULL") is None
    assert plp.repo_of("futon3c-d/mission/autoclock-in") == "futon3c"
    assert plp.repo_of("not-canonical") is None


def test_prep_validates_and_rejects(tmp_path, monkeypatch):
    # only 84d4d04 "exists in git"; everything else must be rejected, never landed
    monkeypatch.setattr(plp, "sha_exists", lambda repo, sha, cache: sha == "84d4d04")
    recs = [
        {"mission": "futon3c-d/mission/good", "discharges": [
            {"target": "sorry/a", "discharged_by": "84d4d04 the repair", "grade": "discharged",
             "witness_verbatim": True, "witness": "the repair landed"}]},                 # LANDS
        {"mission": "futon3c-d/mission/good", "discharges": [
            {"target": "the whole mission prose", "discharged_by": "SUPERSEDED by x",
             "grade": "discharged", "witness_verbatim": True, "witness": "w"}]},            # no-sha
        {"mission": "futon3c-d/mission/good", "discharges": [
            {"target": "sorry/c", "discharged_by": "deadbee the ghost commit", "grade": "discharged",
             "witness_verbatim": True, "witness": "w"}]},                                   # sha-not-in-git
        {"mission": "futon3c-d/mission/good", "discharges": [
            {"target": "sorry/d", "discharged_by": "84d4d04", "grade": "discharged",
             "witness_verbatim": False, "witness": "paraphrase"}]},                         # witness-not-verbatim
        {"mission": "not-canonical", "discharges": [
            {"target": "x", "discharged_by": "84d4d04", "grade": "discharged",
             "witness_verbatim": True, "witness": "w"}]},                                   # mission-not-canonical
        {"mission": "futon3c-d/mission/good", "discharges": [
            {"target": "sorry/e", "discharged_by": None, "grade": "open",
             "witness_verbatim": True, "witness": "w"}]},                                   # not :discharged
    ]
    d = tmp_path / "pm"; d.mkdir()
    with open(d / "proof-mine.jsonl", "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    landing, rejects = plp.prep(str(d))
    assert len(landing) == 1 and landing[0]["sha"] == "84d4d04"
    assert landing[0]["source"] == "proof-mine"
    reasons = {r["reason"] for r in rejects}
    assert reasons == {"no-sha-in-discharged_by", "sha-not-in-git", "witness-not-verbatim",
                       "mission-not-canonical"}


def test_gold_bands_pass_and_fail():
    ok, reasons = pm.gold_bands({"endpoint_precision": 0.8, "grade_agreement": 0.7, "witness_rate": 0.9})
    assert ok and not reasons
    ok, reasons = pm.gold_bands({"endpoint_precision": 0.4, "grade_agreement": 0.7, "witness_rate": 0.9})
    assert not ok and any("precision" in r for r in reasons)
    ok, reasons = pm.gold_bands({"endpoint_precision": 0.8, "grade_agreement": 0.5, "witness_rate": 0.5})
    assert not ok and len(reasons) == 2


def test_score_gold_matches_endpoints_grades_witness():
    record = {"endpoints": ["code/v05/foo"],
              "discharges": [{"grade": "open", "witness": "hello world"}]}
    gold = {"endpoints": ["code/v05/foo"], "grades": ["open"], "discharged_by": []}
    scores = pm.score_gold(record, gold, dossier_text="say hello world today")
    assert scores["endpoint_precision"] == 1.0
    assert scores["endpoint_recall"] == 1.0
    assert scores["grade_agreement"] == 1.0
    assert scores["witness_rate"] == 1.0

    # a witness that is NOT a verbatim dossier span drops the witness rate
    record2 = {"endpoints": [], "discharges": [{"grade": "open", "witness": "not in the text"}]}
    scores2 = pm.score_gold(record2, gold, dossier_text="say hello world today")
    assert scores2["witness_rate"] == 0.0
