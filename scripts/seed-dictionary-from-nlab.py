#!/usr/bin/env python3
"""Seed an OED-shape dictionary from the local nLab page corpus."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator, Optional

import edn_format


DEFAULT_NLAB_PAGES = Path.home() / "code" / "nlab-content" / "pages"
DEFAULT_PM_SEED = Path.home() / "code" / "futon6" / "data" / "dictionary" / "entries-pm-seed.edn"
DEFAULT_OUT_DIR = Path.home() / "code" / "futon6" / "data" / "dictionary"
DEFAULT_SCHEMA_PATH = Path.home() / "code" / "futon6" / "holes" / "excursions" / "dictionary-schema.edn"
PROGRESS_EVERY = 500


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PM_SEED = load_module("seed_dictionary_from_pm", Path(__file__).with_name("seed-dictionary-from-pm.py"))
NLAB_WIRING = load_module("nlab_wiring_seed", Path(__file__).with_name("nlab-wiring.py"))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages-dir", type=Path, default=DEFAULT_NLAB_PAGES)
    parser.add_argument("--pm-seed", type=Path, default=DEFAULT_PM_SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--schema-path", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--timestamp", help="Stable UTC timestamp for deterministic outputs, e.g. 2026-05-20T00:00:00Z")
    return parser.parse_args(argv)


def strip_nlab_markup(text: str) -> str:
    text = re.sub(r"\[\[!include.*?\]\]", " ", text)
    text = re.sub(r"\[\[([^\]|]+?)(?:\|([^\]]+))?\]\]", lambda m: m.group(2) or m.group(1), text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return PM_SEED.collapse_whitespace(PM_SEED.latex_to_text(text))


def nlab_page_to_term_id(name: str) -> str:
    slug = re.sub(r"[+/]", " ", name)
    slug = re.sub(r"[^A-Za-z0-9]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug)
    return slug.strip("-").lower()


def extract_definition_envs(content_md: str) -> list[dict]:
    envs = NLAB_WIRING.parse_environments(content_md)
    return [env for env in envs if env.get("env_type") == "env/definition" and env.get("text")]


def definition_records(page_id: str, definition_envs: list[dict], source_id: str, extracted_at_iso: str) -> list[dict]:
    records = []
    for idx, env in enumerate(definition_envs, start=1):
        text = strip_nlab_markup(env["text"])
        if not text:
            continue
        records.append({
            "def/id": f"nlab-{page_id}-d{idx}",
            "def/text": text,
            "def/extracted-from": source_id,
            "def/source-context": PM_SEED.collapse_whitespace(env["text"])[:4000],
            "def/extraction-method": PM_SEED.kw("nlab-seed"),
            "def/extracted-at": PM_SEED.inst_value(extracted_at_iso),
            "def/confidence": 1.0,
            "def/status": PM_SEED.kw("canonical"),
        })
    return records


def usage_example(name: str, content_md: str, source_id: str) -> dict:
    preview = strip_nlab_markup(content_md)[:1000]
    return {
        "example/paper": source_id,
        "example/role": PM_SEED.kw("canonical-source"),
        "example/context": preview or name,
        "example/seen-at": None,
    }


def nlab_page_to_entry(page_id: str, name: str, content_md: str, *, extracted_at_iso: str) -> dict:
    term_id = nlab_page_to_term_id(name)
    source_id = f"nlab:{page_id}"
    definitions = definition_records(page_id, extract_definition_envs(content_md), source_id, extracted_at_iso)
    entry = {
        "term/id": term_id,
        "term/headword": PM_SEED.normalize_display_headword(name),
        "term/lower": PM_SEED.normalize_lookup_term(name),
        "term/part": PM_SEED.kw("noun"),
        "term/aliases": [],
        "term/etymology": {
            "first-source": source_id,
            "first-source-date": None,
            "first-extractor": PM_SEED.kw("nlab-seed-loader/v1"),
            "note": "nLab canonical page entry; seeded from local nlab-content clone.",
        },
        "term/definitions": definitions,
        "term/usage-examples": [usage_example(name, content_md, source_id)] if definitions else [],
        "term/canon-source": PM_SEED.kw("nlab-seed"),
        "term/first-seen": None,
        "term/last-seen": None,
        "term/occurrence-count": 1,
        "term/cross-refs": [],
        "term/review-notes": [f"Seeded from nLab {extracted_at_iso[:10]}."],
        "term/graduated-at": PM_SEED.inst_value(extracted_at_iso),
        "term/source-metadata": {
            "nlab-page-id": page_id,
            "pages-dir": str(DEFAULT_NLAB_PAGES),
        },
    }
    if definitions:
        entry["term/status"] = PM_SEED.kw("canonical")
    else:
        entry["term/status"] = PM_SEED.kw("canonical-no-definition")
        entry["term/review-notes"].append("nLab page had no extractable definition environment.")
    return entry


def iter_nlab_seed_entries(pages_dir: Path, *, extracted_at_iso: str) -> Iterator[dict]:
    for page_id, name, content_md in NLAB_WIRING.iter_nlab_pages(pages_dir):
        if PM_SEED.is_numeric_id_headword(name):
            continue
        yield nlab_page_to_entry(page_id, name, content_md, extracted_at_iso=extracted_at_iso)


def load_pm_indices(pm_seed_path: Path) -> tuple[dict[str, dict], dict[str, list[dict]], int]:
    raw = edn_format.loads(pm_seed_path.read_text(encoding="utf-8"))
    entries = raw[edn_format.Keyword("dictionary/entries")]
    by_id: dict[str, dict] = {}
    by_lower: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        term_id = entry[edn_format.Keyword("term/id")]
        term_lower = entry[edn_format.Keyword("term/lower")]
        headword = entry[edn_format.Keyword("term/headword")]
        row = {"term/id": term_id, "term/lower": term_lower, "term/headword": headword}
        by_id[term_id] = row
        by_lower[term_lower].append(row)
    return by_id, by_lower, len(entries)


def collision_report(entries: list[dict], pm_by_id: dict[str, dict], pm_by_lower: dict[str, list[dict]], pm_entry_count: int) -> dict:
    id_hits = 0
    lower_hits = 0
    any_hits = 0
    lower_only_hits = 0
    examples = []
    for entry in entries:
        term_id = entry["term/id"]
        term_lower = entry["term/lower"]
        id_collision = term_id in pm_by_id
        lower_collision = term_lower in pm_by_lower
        if id_collision:
            id_hits += 1
        if lower_collision:
            lower_hits += 1
        if id_collision or lower_collision:
            any_hits += 1
            if lower_collision and not id_collision:
                lower_only_hits += 1
            if len(examples) < 200:
                examples.append({
                    "term/id": term_id,
                    "term/lower": term_lower,
                    "term/headword": entry["term/headword"],
                    "collision-kinds": [
                        kind for kind, present in (
                            ("term-id", id_collision),
                            ("term-lower", lower_collision),
                        ) if present
                    ],
                    "pm-matches": sorted({
                        pm["term/id"]
                        for pm in ([pm_by_id[term_id]] if id_collision else [])
                        + pm_by_lower.get(term_lower, [])
                    }),
                })
    return {
        "pm-entry-count": pm_entry_count,
        "nlab-entry-count": len(entries),
        "collision-counts": {
            "term-id": id_hits,
            "term-lower": lower_hits,
            "any": any_hits,
            "term-lower-only": lower_only_hits,
        },
        "new-term-estimate": len(entries) - any_hits,
        "collision-examples": examples,
    }


def build_entries_document(entries: list[dict], *, timestamp_iso: str, pages_dir: Path) -> dict:
    return {
        "dictionary/version": "0.1-nlab-seed",
        "dictionary/created": PM_SEED.inst_value(timestamp_iso),
        "dictionary/created-by": PM_SEED.kw("nlab-seed-loader/v1"),
        "dictionary/source-root": str(pages_dir),
        "dictionary/entry-count": len(entries),
        "dictionary/entries": entries,
    }


def run_pipeline(args: argparse.Namespace) -> dict:
    started = time.time()
    timestamp_iso = args.timestamp or PM_SEED.iso_utc_now()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pm_by_id, pm_by_lower, pm_entry_count = load_pm_indices(args.pm_seed)

    entries: list[dict] = []
    processed = 0
    skipped = Counter()
    for entry in iter_nlab_seed_entries(args.pages_dir, extracted_at_iso=timestamp_iso):
        processed += 1
        if processed % PROGRESS_EVERY == 0:
            print(f"Processed {processed} nLab pages...", flush=True)
        entries.append(entry)

    entries.sort(key=lambda entry: entry["term/id"])
    collision_stats = collision_report(entries, pm_by_id, pm_by_lower, pm_entry_count)
    entries_doc = build_entries_document(entries, timestamp_iso=timestamp_iso, pages_dir=args.pages_dir)

    entries_path = args.out_dir / "entries-nlab-seed.edn"
    audit_path = args.out_dir / "nlab-audit-sample.json"
    collision_path = args.out_dir / "nlab-collision-stats.json"
    stats_path = args.out_dir / "nlab-run-stats.json"

    PM_SEED.write_edn_file(entries_path, entries_doc)
    PM_SEED.validate_edn_round_trip(entries_path)
    audit_sample = PM_SEED.audit_sample(entries)
    audit_path.write_text(json.dumps(PM_SEED.json_ready(audit_sample), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    collision_path.write_text(json.dumps(PM_SEED.json_ready(collision_stats), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    elapsed_seconds = round(time.time() - started, 3)
    status_counts = Counter(entry["term/status"].name for entry in entries)
    stats = {
        "pages_dir": str(args.pages_dir),
        "pm_seed": str(args.pm_seed),
        "schema_path": str(args.schema_path),
        "timestamp": timestamp_iso,
        "processed_pages": processed,
        "succeeded_entries": len(entries),
        "status_counts": dict(sorted(status_counts.items())),
        "collision_counts": collision_stats["collision-counts"],
        "new_term_estimate": collision_stats["new-term-estimate"],
        "audit_sample_size": len(audit_sample),
        "skipped": {
            "total": sum(skipped.values()),
            "by_reason": dict(sorted(skipped.items())),
        },
        "elapsed_seconds": elapsed_seconds,
    }
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(entries)} nLab entries with {collision_stats['collision-counts']['any']} PM collisions "
        f"in {elapsed_seconds:.3f}s.",
        flush=True,
    )
    return {
        "entries_path": entries_path,
        "audit_path": audit_path,
        "collision_path": collision_path,
        "stats_path": stats_path,
        "stats": stats,
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if not args.pages_dir.exists():
        raise SystemExit(f"nLab pages directory not found: {args.pages_dir}")
    if not args.pm_seed.exists():
        raise SystemExit(f"PM seed file not found: {args.pm_seed}")
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
