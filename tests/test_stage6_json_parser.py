"""Regression tests for Stage 6 JSON-object parsing robustness."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _load_superpod_job():
    root = Path(__file__).parent.parent
    sys.path.insert(0, str(root / "scripts"))
    return importlib.import_module("superpod-job")


def test_stage6_parser_accepts_json_with_prose_prefix():
    mod = _load_superpod_job()
    raw = (
        "Here is your result:\\n\\n"
        "```json\\n"
        "{"
        "\"xiang_form\":\"f\","
        "\"xiang_salience\":\"s\","
        "\"arrow_constraint\":\"a\","
        "\"quality\":{\"form\":\"good\",\"salience\":\"weak\",\"arrow\":\"broken\"},"
        "\"situation_S\":\"sit\","
        "\"roundtrip_check\":\"ok\""
        "}\\n```"
    )
    parsed = mod._parse_json_object_response(raw)
    assert parsed["xiang_form"] == "f"
    assert parsed["quality"]["form"] == "good"
    assert "parse_error" not in parsed


def test_stage6_parser_repairs_trailing_comma_and_unclosed_object():
    mod = _load_superpod_job()
    raw = (
        "{"
        "\"xiang_form\":\"f\","
        "\"xiang_salience\":\"s\","
        "\"arrow_constraint\":\"a\","
        "\"quality\":{\"form\":\"good\",\"salience\":\"weak\",\"arrow\":\"broken\",},"
        "\"situation_S\":\"sit\","
        "\"roundtrip_check\":\"ok\""
    )  # intentionally missing final "}"
    parsed = mod._parse_json_object_response(raw)
    assert parsed["roundtrip_check"] == "ok"
    assert parsed["quality"]["arrow"] == "broken"
    assert "parse_error" not in parsed


def test_stage6_parser_accepts_python_literal_dict():
    mod = _load_superpod_job()
    raw = (
        "{'xiang_form':'f','xiang_salience':'s','arrow_constraint':'a',"
        "'quality':{'form':'good','salience':'weak','arrow':'broken'},"
        "'situation_S':'sit','roundtrip_check':'ok'}"
    )
    parsed = mod._parse_json_object_response(raw)
    assert parsed["situation_S"] == "sit"
    assert "parse_error" not in parsed


def test_stage6_parser_reports_no_json():
    mod = _load_superpod_job()
    parsed = mod._parse_json_object_response("No object here at all.")
    assert parsed.get("parse_error") == "no JSON object found"


def test_stage6_result_marks_parse_failures_explicitly():
    mod = _load_superpod_job()
    result = mod._build_stage6_result("arxiv-1", 7, "No object here at all.")
    assert result["status"] == "failed"
    assert result["reason"] == "stage6-parse-error"
    assert result["analysis"]["parse_error"] == "no JSON object found"
    assert result["schema_version"] == "reverse-morphogenesis-v2"


def test_stage6_result_marks_distinct_slots_ok():
    mod = _load_superpod_job()
    raw = (
        "{"
        "\"xiang_form\":\"spectral sequence\","
        "\"xiang_salience\":\"understand why the filtration converges\","
        "\"arrow_constraint\":\"identify the exact vanishing hypothesis needed\","
        "\"quality\":{\"form\":\"good\",\"salience\":\"good\",\"arrow\":\"good\"},"
        "\"situation_S\":\"A researcher has computed early pages and needs a convergence criterion.\","
        "\"roundtrip_check\":\"Yes, the situation naturally leads to that question.\""
        "}"
    )
    result = mod._build_stage6_result("arxiv-2", 8, raw)
    assert result["status"] == "ok"
    assert result["reason"] is None
    assert result["collapsed"] is False
    assert result["slot_distinctness"]["status"] == "distinct"


def test_stage6_result_flags_slot_collapse_for_clarification():
    mod = _load_superpod_job()
    repeated = "understand the significance of the S-Prime Element Principle"
    raw = (
        "{"
        "\"xiang_form\":\"prime element principle\","
        f"\"xiang_salience\":\"{repeated}\","
        f"\"arrow_constraint\":\"{repeated}\","
        "\"quality\":{\"form\":\"good\",\"salience\":\"good\",\"arrow\":\"good\"},"
        f"\"situation_S\":\"{repeated}\","
        "\"roundtrip_check\":\"The source only supports one axis of explanation.\""
        "}"
    )
    result = mod._build_stage6_result("arxiv-3", 9, raw)
    assert result["status"] == "clarification"
    assert result["reason"] == "slot-collapse"
    assert result["collapsed"] is True
    assert result["slot_distinctness"]["collapsed_pairs"]


def test_stage6_result_reports_missing_required_keys():
    mod = _load_superpod_job()
    raw = (
        "{"
        "\"xiang_form\":\"f\","
        "\"quality\":{\"form\":\"good\",\"salience\":\"good\",\"arrow\":\"good\"},"
        "\"situation_S\":\"sit\","
        "\"roundtrip_check\":\"ok\""
        "}"
    )
    result = mod._build_stage6_result("arxiv-4", 10, raw)
    assert result["status"] == "failed"
    assert result["reason"] == "stage6-missing-required-keys"
    assert "xiang_salience" in result["missing_keys"]
