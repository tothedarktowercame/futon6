#!/usr/bin/env python3
"""Make-like runner for the WARP concept spine.

The runner intentionally shells out to the existing stage scripts.  It does not
own or rebuild downstream SFC-D3 artifacts such as data/warp/concept-index.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
WARP = ROOT / "data" / "warp"
EPRINTS = Path("/home/joe/code/storage/futon6/data/arxiv-math-ct-eprints")
ANATOMY = Path("/home/joe/code/storage/futon6/data/ct-anatomy-v0")
GOLDEN = ROOT / "data" / "showcases" / "ct-anatomy" / "golden"
BACKGROUND = ROOT / "data" / "background-corpus-index.json"
MANIFEST = WARP / "warp-manifest.json"

GUARDED_OUTPUTS = {
    WARP / "concept-index.json",
    ROOT / "holes" / "excursions" / "sfc-concept-index.md",
}
GUARDED_SCRIPTS = {"sfc_concept_index.py"}


@dataclass(frozen=True)
class Stage:
    stage_id: str
    script: str
    inputs: tuple[Path, ...]
    outputs: tuple[Path, ...]
    command: tuple[str, ...] | None = None
    overlay: bool = False
    audit_only: bool = False
    notes: str = ""

    @property
    def runnable(self) -> bool:
        return self.command is not None and not self.audit_only


def p(rel: str) -> Path:
    return ROOT / rel


SPINE_STAGES: tuple[Stage, ...] = (
    Stage(
        "S1a",
        "warp_concordance.py",
        (EPRINTS, ANATOMY, GOLDEN),
        (p("data/warp/concordance.json"),),
        ("scripts/warp_concordance.py",),
    ),
    Stage(
        "S1b",
        "warp_bib.py",
        (EPRINTS,),
        (p("data/warp/bib-index.json"), p("data/warp/bib")),
        ("scripts/warp_bib.py",),
    ),
    Stage(
        "S1c",
        "warp_citations.py",
        (EPRINTS, p("data/warp/bib-index.json"), p("data/warp/bib")),
        (p("data/warp/citations.json"),),
        ("scripts/warp_citations.py",),
    ),
    Stage(
        "S2",
        "warp_defined_pass.py",
        (EPRINTS,),
        (p("data/warp/defined-index.json"),),
        ("scripts/warp_defined_pass.py",),
    ),
    Stage(
        "S3",
        "warp_hitlist.py",
        (p("data/warp/concordance.json"), p("data/warp/defined-index.json")),
        (p("data/warp/hitlist.json"),),
        ("scripts/warp_hitlist.py",),
    ),
    Stage(
        "S4a",
        "warp_def_snippets.py",
        (p("data/warp/hitlist.json"), EPRINTS),
        (p("data/warp/def-snippets.json"),),
        ("scripts/warp_def_snippets.py",),
        notes="Script has no dry-run CLI and hard-codes live hitlist/eprints.",
    ),
    Stage(
        "S4b",
        "warp_concept_usage.py",
        (p("data/warp/hitlist.json"), EPRINTS),
        (p("data/warp/concept-usage.json"),),
        ("scripts/warp_concept_usage.py",),
    ),
    Stage(
        "S5",
        "warp_concept_graph.py",
        (p("data/warp/hitlist.json"), p("data/warp/def-snippets.json")),
        (p("data/warp/concept-graph.json"),),
        ("scripts/warp_concept_graph.py",),
        notes="Must precede S4c because embeddings consume concept-graph.json.",
    ),
    Stage(
        "S4c",
        "warp_concept_embed.py",
        (
            p("data/warp/hitlist.json"),
            p("data/warp/def-snippets.json"),
            p("data/warp/concept-graph.json"),
        ),
        (p("data/warp/concept-embed.npy"), p("data/warp/concept-carpet-pos.json")),
        ("scripts/warp_concept_embed.py",),
    ),
    Stage(
        "S6a",
        "mark3_thread_tapestry.py",
        (
            GOLDEN,
            p("data/concept-encyclopedia/ct"),
            p("data/warp/cite-resolution"),
        ),
        (p("tmp/mark3-threads/ct-threads.json"),),
        ("scripts/mark3_thread_tapestry.py",),
        notes="WARP-ORCH-3 will promote this to a named data/warp artifact.",
    ),
    Stage(
        "S6b",
        "build_concept_encyclopedia.py",
        (
            p("data/term-prior-ct.json"),
            BACKGROUND,
            p("data/warp/def-snippets.json"),
            p("data/warp/concept-graph.json"),
            p("data/warp/defined-index.json"),
        ),
        (p("data/concept-encyclopedia-ct.json"), p("data/concept-encyclopedia/ct")),
        ("scripts/build_concept_encyclopedia.py",),
    ),
)

OVERLAY_STAGES: tuple[Stage, ...] = (
    Stage(
        "O1",
        "warp_or_curvature.py",
        (p("data/warp/citations.json"),),
        (p("data/warp/or-curvature.json"),),
        ("scripts/warp_or_curvature.py",),
        overlay=True,
    ),
    Stage(
        "O2",
        "warp_salingaros.py",
        (GOLDEN,),
        (p("data/warp/aliveness.json"),),
        ("scripts/warp_salingaros.py",),
        overlay=True,
    ),
    Stage(
        "O3",
        "warp_paper_landscape.py",
        (
            p("data/warp/hitlist.json"),
            p("data/warp/concept-embed.npy"),
            p("data/warp/defined-index.json"),
            p("data/warp/concordance.json"),
            p("data/warp/concept-usage.json"),
            p("data/warp/or-curvature.json"),
            p("data/warp/aliveness.json"),
        ),
        (p("data/warp/paper-landscape.json"), p("data/warp/paper-landscape.html")),
        ("scripts/warp_paper_landscape.py",),
        overlay=True,
    ),
    Stage(
        "O4",
        "warp_greatest_hits.py",
        (
            Path("/tmp/gh200.txt"),
            GOLDEN,
            p("data/warp/concept-usage.json"),
            p("data/warp/hitlist.json"),
            p("data/warp/concept-embed.npy"),
            p("data/warp/citations.json"),
        ),
        (p("data/warp/greatest-hits.html"),),
        ("scripts/warp_greatest_hits.py",),
        overlay=True,
    ),
    Stage(
        "O5",
        "warp_debt_report.py",
        (
            p("data/warp/concordance.json"),
            p("data/mathlib/training.jsonl"),
            p("data/nlab/Pages"),
            p("data/planetmath/planetmath.jsonl"),
        ),
        (p("data/warp/corpus-debt.json"),),
        ("scripts/warp_debt_report.py",),
        overlay=True,
    ),
)

AUDIT_ONLY_STAGES: tuple[Stage, ...] = (
    Stage(
        "consumer",
        "build_term_prior.py",
        (GOLDEN,),
        (p("data/term-prior-ct.json"),),
        None,
        audit_only=True,
    ),
    Stage(
        "consumer",
        "sfc_concept_coverage.py",
        (
            p("data/warp/concept-usage.json"),
            p("data/warp/def-snippets.json"),
            p("data/warp/defined-index.json"),
            p("data/concept-encyclopedia-ct.json"),
            p("data/warp/concept-graph.json"),
        ),
        (p("holes/excursions/sfc-concept-coverage.md"),),
        None,
        audit_only=True,
    ),
    Stage(
        "SFC-D3",
        "sfc_concept_index.py",
        (
            p("data/warp/concept-usage.json"),
            p("data/warp/def-snippets.json"),
            p("data/warp/defined-index.json"),
            p("data/concept-encyclopedia-ct.json"),
        ),
        (p("data/warp/concept-index.json"), p("holes/excursions/sfc-concept-index.md")),
        None,
        audit_only=True,
        notes="Canonical downstream artifact; read-only in WARP-ORCH-2.",
    ),
)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def walk_files(path: Path) -> Iterable[Path]:
    if not path.exists():
        return
    if path.is_file():
        yield path
        return
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            yield Path(root) / name


def max_mtime(path: Path) -> float | None:
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_mtime
    newest = path.stat().st_mtime
    for file_path in walk_files(path):
        newest = max(newest, file_path.stat().st_mtime)
    return newest


def min_output_mtime(outputs: Iterable[Path]) -> float | None:
    mtimes: list[float] = []
    for output in outputs:
        mtime = max_mtime(output)
        if mtime is None:
            return None
        mtimes.append(mtime)
    return min(mtimes) if mtimes else None


def outputs_present(outputs: Iterable[Path]) -> bool:
    return all(output.exists() for output in outputs)


def is_fresh(stage: Stage) -> bool:
    output_mtime = min_output_mtime(stage.outputs)
    if output_mtime is None:
        return False
    input_mtimes = [mtime for mtime in (max_mtime(path) for path in stage.inputs) if mtime is not None]
    return bool(input_mtimes) and output_mtime >= max(input_mtimes)


def input_hash(inputs: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(inputs, key=lambda item: display_path(item)):
        digest.update(display_path(path).encode())
        if not path.exists():
            digest.update(b"\0missing")
            continue
        for file_path in walk_files(path):
            rel = display_path(file_path)
            stat = file_path.stat()
            digest.update(rel.encode())
            digest.update(str(stat.st_size).encode())
            digest.update(str(int(stat.st_mtime_ns)).encode())
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open() as fh:
        return json.load(fh)


def count_rows(stage: Stage) -> dict[str, Any]:
    script = stage.script
    try:
        if script == "warp_concordance.py":
            data = load_json(stage.outputs[0])
            return {"terms": len(data.get("terms", data))}
        if script == "warp_bib.py":
            bib_index = load_json(stage.outputs[0])
            if isinstance(bib_index, dict) and "papers" in bib_index:
                return {
                    "papers": len(bib_index.get("papers", [])),
                    "bibitems": bib_index.get("stats", {}).get("bibitems"),
                }
            return {"papers": len(bib_index), "bibitems": None}
        if script == "warp_citations.py":
            data = load_json(stage.outputs[0])
            return {"edges": len(data.get("edges", [])), "cited_by": len(data.get("cited_by", {}))}
        if script == "warp_defined_pass.py":
            data = load_json(stage.outputs[0])
            return {"concept_to_papers": len(data.get("concept_to_papers", data))}
        if script == "warp_hitlist.py":
            data = load_json(stage.outputs[0])
            return {"hitlist": len(data.get("hitlist", [])), "frontier": len(data.get("frontier", []))}
        if script == "warp_def_snippets.py":
            data = load_json(stage.outputs[0])
            return {
                "snippets": len(data.get("snippets", data if isinstance(data, list) else [])),
                "papers_scanned": data.get("papers_scanned"),
            }
        if script == "warp_concept_usage.py":
            data = load_json(stage.outputs[0])
            return {
                "paper_concepts": len(data.get("paper_concepts", data)),
                "papers_scanned": data.get("papers_scanned"),
            }
        if script == "warp_concept_graph.py":
            data = load_json(stage.outputs[0])
            if "n_nodes" in data or "n_edges" in data:
                return {
                    "n_nodes": data.get("n_nodes"),
                    "n_edges": data.get("n_edges"),
                    "authority": len(data.get("authority", [])),
                }
            return {
                "n_nodes": len(data.get("nodes", [])),
                "n_edges": len(data.get("edges", [])),
                "authority": len(data.get("authority", [])),
            }
        if script == "warp_concept_embed.py":
            positions = load_json(stage.outputs[1])
            return {"positions": len(positions), "npy_bytes": stage.outputs[0].stat().st_size}
        if script == "mark3_thread_tapestry.py":
            if not stage.outputs[0].exists():
                return {}
            data = load_json(stage.outputs[0])
            return {
                "concepts_with_threads": len(data.get("concepts", data)),
                "n_papers": len(data.get("papers", [])),
            }
        if script == "build_concept_encyclopedia.py":
            data = load_json(stage.outputs[0])
            return {"entries": len(data.get("entries", data))}
        if script == "build_term_prior.py":
            data = load_json(stage.outputs[0])
            return {"terms": len(data.get("df", data))}
        if script == "sfc_concept_coverage.py":
            return {"report_bytes": stage.outputs[0].stat().st_size}
        if script == "sfc_concept_index.py":
            data = load_json(stage.outputs[0])
            values = data.values() if isinstance(data, dict) else []
            return {
                "concepts": len(data),
                "genuine": sum(1 for item in values if isinstance(item, dict) and item.get("genuine")),
                "defined": sum(1 for item in values if isinstance(item, dict) and item.get("defined")),
            }
        if script == "warp_or_curvature.py":
            return {"keys": len(load_json(stage.outputs[0]))}
        if script == "warp_salingaros.py":
            return {"keys": len(load_json(stage.outputs[0]))}
        if script == "warp_paper_landscape.py":
            data = load_json(stage.outputs[0])
            return {"papers": len(data.get("papers", data if isinstance(data, list) else []))}
        if script == "warp_greatest_hits.py":
            return {"html_bytes": stage.outputs[0].stat().st_size}
        if script == "warp_debt_report.py":
            return {"keys": len(load_json(stage.outputs[0]))}
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
        return {}
    return {}


def stage_status(stage: Stage) -> str:
    if not outputs_present(stage.outputs):
        return "missing"
    return "fresh" if is_fresh(stage) else "stale"


def validate_guards(stages: Iterable[Stage]) -> None:
    violations: list[str] = []
    for stage in stages:
        script_name = Path(stage.script).name
        if stage.runnable and script_name in GUARDED_SCRIPTS:
            violations.append(f"{stage.stage_id} runs guarded script {script_name}")
        if stage.runnable:
            for output in stage.outputs:
                if output.resolve() in {path.resolve() for path in GUARDED_OUTPUTS}:
                    violations.append(f"{stage.stage_id} writes guarded output {display_path(output)}")
    if violations:
        raise SystemExit("Refusing guarded WARP run:\n" + "\n".join(violations))


def selected_stages(include_overlays: bool = False, audit: bool = False) -> tuple[Stage, ...]:
    stages: tuple[Stage, ...] = SPINE_STAGES
    if include_overlays:
        stages += OVERLAY_STAGES
    if audit:
        stages += AUDIT_ONLY_STAGES
    return stages


def audit_rows(stages: Iterable[Stage]) -> list[dict[str, Any]]:
    rows = []
    for stage in stages:
        rows.append(
            {
                "stage": stage.stage_id,
                "script": stage.script,
                "present": outputs_present(stage.outputs),
                "freshness": stage_status(stage),
                "output": ", ".join(display_path(output) for output in stage.outputs),
                "rows": count_rows(stage),
                "notes": stage.notes,
            }
        )
    return rows


def print_audit(rows: list[dict[str, Any]]) -> None:
    print("| Stage | Script | Present | Freshness | Output | Rows / keys |")
    print("|---|---|---:|---|---|---:|")
    for row in rows:
        counts = ", ".join(f"{key}={value}" for key, value in row["rows"].items() if value is not None)
        print(
            f"| {row['stage']} | `{row['script']}` | "
            f"{'yes' if row['present'] else 'no'} | {row['freshness']} | "
            f"`{row['output']}` | `{counts}` |"
        )


def manifest_entry(stage: Stage, status: str) -> dict[str, Any]:
    return {
        "script": stage.script,
        "inputs": [display_path(path) for path in stage.inputs],
        "input-hash": input_hash(stage.inputs),
        "output": [display_path(path) for path in stage.outputs],
        "built-at": datetime.now(timezone.utc).isoformat(),
        "rows": count_rows(stage),
        "status": status,
        "freshness": stage_status(stage),
        "notes": stage.notes,
    }


def run_stage(stage: Stage, dry_run: bool = False) -> str:
    if not stage.runnable:
        return "audit-only"
    if is_fresh(stage):
        return "skipped"
    if dry_run:
        return "would-run"
    assert stage.command is not None
    command = [sys.executable, *stage.command]
    subprocess.run(command, cwd=ROOT, check=True)
    return "built"


def run(stages: Iterable[Stage], dry_run: bool = False, manifest_path: Path = MANIFEST) -> dict[str, Any]:
    stage_list = tuple(stages)
    validate_guards(stage_list)
    records: dict[str, Any] = {}
    for stage in stage_list:
        status = run_stage(stage, dry_run=dry_run)
        records[stage.stage_id] = manifest_entry(stage, status)
    if not dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlays", action="store_true", help="include inspection-layer overlay stages")
    parser.add_argument("--audit", action="store_true", help="read-only liveness audit; writes no manifest")
    parser.add_argument("--dry-run", action="store_true", help="show manifest statuses without running stages")
    parser.add_argument("--manifest", type=Path, default=MANIFEST, help="manifest path")
    args = parser.parse_args(argv)

    stages = selected_stages(include_overlays=args.overlays, audit=args.audit)
    validate_guards(stages)
    if args.audit:
        print_audit(audit_rows(stages))
        return 0

    records = run(stages, dry_run=args.dry_run, manifest_path=args.manifest)
    built = sum(1 for entry in records.values() if entry["status"] == "built")
    skipped = sum(1 for entry in records.values() if entry["status"] == "skipped")
    would_run = sum(1 for entry in records.values() if entry["status"] == "would-run")
    print(
        f"warp_run: built={built} skipped={skipped} would-run={would_run} "
        f"manifest={display_path(args.manifest)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
