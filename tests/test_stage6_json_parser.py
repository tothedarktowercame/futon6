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
