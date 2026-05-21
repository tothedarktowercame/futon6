"""Tests for the nLab dictionary seed loader."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import edn_format


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "seed-dictionary-from-nlab.py"
FIXTURE_PAGES = REPO_ROOT / "tests" / "fixtures" / "nlab-mini" / "pages"
FIXTURE_PM_SEED = REPO_ROOT / "tests" / "fixtures" / "pm-mini-seed.edn"


def load_script_module():
    spec = importlib.util.spec_from_file_location("seed_dictionary_from_nlab", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def edn_to_plain(value):
    if isinstance(value, edn_format.ImmutableDict):
        out = {}
        for key, inner in value.items():
            key_name = key.name if hasattr(key, "name") else str(key)
            out[key_name] = edn_to_plain(inner)
        return out
    if isinstance(value, (edn_format.ImmutableList, list, tuple)):
        return [edn_to_plain(item) for item in value]
    if hasattr(value, "name"):
        return value.name
    return value


def test_extracts_multiple_definition_envs():
    module = load_script_module()
    content = (FIXTURE_PAGES / "1" / "content.md").read_text(encoding="utf-8")
    envs = module.extract_definition_envs(content)
    defs = module.definition_records("1", envs, "nlab:1", "2026-05-20T00:00:00Z")
    assert len(defs) == 2
    assert "ringed space" in defs[0]["def/text"].lower()


def test_page_without_definition_is_canonical_no_definition():
    module = load_script_module()
    entry = module.nlab_page_to_entry(
        "2",
        "functor category",
        (FIXTURE_PAGES / "2" / "content.md").read_text(encoding="utf-8"),
        extracted_at_iso="2026-05-20T00:00:00Z",
    )
    assert entry["term/status"].name == "canonical-no-definition"
    assert entry["term/definitions"] == []


def test_collision_report_counts_pm_overlap():
    module = load_script_module()
    pm_by_id, pm_by_lower, pm_count = module.load_pm_indices(FIXTURE_PM_SEED)
    entries = [
        module.nlab_page_to_entry(
            "1",
            "ringed space",
            (FIXTURE_PAGES / "1" / "content.md").read_text(encoding="utf-8"),
            extracted_at_iso="2026-05-20T00:00:00Z",
        ),
        module.nlab_page_to_entry(
            "2",
            "functor category",
            (FIXTURE_PAGES / "2" / "content.md").read_text(encoding="utf-8"),
            extracted_at_iso="2026-05-20T00:00:00Z",
        ),
    ]
    report = module.collision_report(entries, pm_by_id, pm_by_lower, pm_count)
    assert report["collision-counts"]["term-id"] == 1
    assert report["collision-counts"]["term-lower"] == 1
    assert report["collision-counts"]["any"] == 1
    assert report["new-term-estimate"] == 1


def test_end_to_end_run_is_idempotent(tmp_path: Path):
    module = load_script_module()
    out_dir = tmp_path / "out"
    argv = [
        "--pages-dir", str(FIXTURE_PAGES),
        "--pm-seed", str(FIXTURE_PM_SEED),
        "--out-dir", str(out_dir),
        "--schema-path", str(REPO_ROOT / "holes" / "excursions" / "dictionary-schema.edn"),
        "--timestamp", "2026-05-20T00:00:00Z",
    ]
    module.main(argv)
    first_entries = (out_dir / "entries-nlab-seed.edn").read_text(encoding="utf-8")
    module.main(argv)
    second_entries = (out_dir / "entries-nlab-seed.edn").read_text(encoding="utf-8")
    assert first_entries == second_entries

    parsed = edn_to_plain(edn_format.loads(first_entries))
    assert parsed["dictionary/entry-count"] == 2
    assert len(parsed["dictionary/entries"]) == 2

    stats = json.loads((out_dir / "nlab-run-stats.json").read_text(encoding="utf-8"))
    assert stats["succeeded_entries"] == 2
    assert stats["collision_counts"]["any"] == 1
    assert stats["new_term_estimate"] == 1

    collision_stats = json.loads((out_dir / "nlab-collision-stats.json").read_text(encoding="utf-8"))
    assert collision_stats["collision-counts"]["any"] == 1
    assert collision_stats["pm-entry-count"] == 2
