#!/usr/bin/env python3
"""Executable CAS check registry.

CAS selection emits static check labels keyed by matched proof patterns.  This
module turns those labels into executable registry entries while reusing the
existing rung-2 semantic checker for R2a/R2b/R2c/R2d.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import edn_format

ROOT = Path(__file__).resolve().parents[1]


def _load_cas_select() -> Any:
    spec = importlib.util.spec_from_file_location("cas_select", ROOT / "scripts" / "cas_select.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cas_select = _load_cas_select()


def keyword_name(value: Any) -> str:
    text = str(value)
    return text[1:] if text.startswith(":") else text


def edn_to_plain(value: Any) -> Any:
    if isinstance(value, edn_format.Keyword):
        return ":" + keyword_name(value)
    if isinstance(value, Mapping) or hasattr(value, "items"):
        return {keyword_name(k): edn_to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)) or (
        not isinstance(value, (str, bytes)) and hasattr(value, "__iter__")
    ):
        return [edn_to_plain(v) for v in value]
    return value


def strip_keyword(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(":"):
        return value[1:]
    return value


def normalize_check_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["check"] = strip_keyword(out.get("check"))
    out["status"] = strip_keyword(out.get("status"))
    return out


@dataclass(frozen=True)
class CheckContext:
    graph_path: Path | None = None
    semcheck_report: dict[str, Any] | None = None


Predicate = Callable[[dict[str, Any]], bool]
Runner = Callable[[CheckContext, dict[str, Any], str], dict[str, Any]]


@dataclass(frozen=True)
class RegistryEntry:
    label: str
    check: str
    source: str
    predicate: Predicate
    run: Runner


def fires_label(label: str) -> Predicate:
    return lambda match: label in (match.get("fires") or [])


def na_result(label: str, check: str, match: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "check": check,
        "dispatch-label": label,
        "status": "na",
        "pass": True,
        "rate": None,
        "reasons": [reason],
        "per-item": [],
        "step": match.get("step"),
        "pattern": match.get("pattern"),
    }


def load_semcheck_report(graph_path: Path) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix=".edn", delete=False) as f:
        out_path = Path(f.name)
    try:
        cmd = ["bb", "scripts/iatc_semcheck.bb", "--out", str(out_path), str(graph_path)]
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
        text = re.sub(r":([A-Za-z0-9_./?=-]+)'", r":\1-prime", out_path.read_text())
        return edn_to_plain(edn_format.loads(text))
    finally:
        out_path.unlink(missing_ok=True)


def semcheck_report(context: CheckContext) -> dict[str, Any] | None:
    if context.semcheck_report is not None:
        return context.semcheck_report
    if context.graph_path is None:
        return None
    return load_semcheck_report(context.graph_path)


def semcheck_result(context: CheckContext, check_name: str) -> dict[str, Any] | None:
    report = semcheck_report(context)
    if not report:
        return None
    graphs = report.get("graphs") or []
    if not graphs:
        return None
    for row in graphs[0].get("checks") or []:
        normalized = normalize_check_row(row)
        if normalized.get("check") == check_name:
            return normalized
    return None


def run_semcheck_check(check_name: str) -> Runner:
    def runner(context: CheckContext, match: dict[str, Any], label: str) -> dict[str, Any]:
        row = semcheck_result(context, check_name)
        if row is None:
            return na_result(
                label,
                check_name,
                match,
                "N/A: no graph path or semcheck report supplied for executable rung-2 check",
            )
        out = dict(row)
        out["dispatch-label"] = label
        out["step"] = match.get("step")
        out["pattern"] = match.get("pattern")
        return out

    return runner


def run_unbuilt_stub(check_name: str) -> Runner:
    def runner(context: CheckContext, match: dict[str, Any], label: str) -> dict[str, Any]:
        return na_result(
            label,
            check_name,
            match,
            f"N/A: {label} is registered but its proof-shape checker is not built yet",
        )

    return runner


REGISTRY: tuple[RegistryEntry, ...] = (
    RegistryEntry(
        label="R2a-anchor",
        check="anchor-faithfulness",
        source="scripts/iatc_semcheck.bb",
        predicate=fires_label("R2a-anchor"),
        run=run_semcheck_check("anchor-faithfulness"),
    ),
    RegistryEntry(
        label="R2b-closure",
        check="closure",
        source="scripts/iatc_semcheck.bb",
        predicate=fires_label("R2b-closure"),
        run=run_semcheck_check("closure"),
    ),
    RegistryEntry(
        label="R2b-disjointness",
        check="R2b-disjointness",
        source="cas_checks.stub",
        predicate=fires_label("R2b-disjointness"),
        run=run_unbuilt_stub("R2b-disjointness"),
    ),
    RegistryEntry(
        label="R2c-warrant",
        check="warrant-resolution",
        source="scripts/iatc_semcheck.bb",
        predicate=fires_label("R2c-warrant"),
        run=run_semcheck_check("warrant-resolution"),
    ),
    RegistryEntry(
        label="R2d-concept-coverage",
        check="concept-coverage",
        source="scripts/iatc_semcheck.bb",
        predicate=fires_label("R2d-concept-coverage"),
        run=run_semcheck_check("concept-coverage"),
    ),
    RegistryEntry(
        label="decomposition-exhaustive",
        check="decomposition-exhaustive",
        source="cas_checks.stub",
        predicate=fires_label("decomposition-exhaustive"),
        run=run_unbuilt_stub("decomposition-exhaustive"),
    ),
    RegistryEntry(
        label="forall-eps-structure",
        check="forall-eps-structure",
        source="cas_checks.stub",
        predicate=fires_label("forall-eps-structure"),
        run=run_unbuilt_stub("forall-eps-structure"),
    ),
    RegistryEntry(
        label="well-defined-on-quotient",
        check="well-defined-on-quotient",
        source="cas_checks.stub",
        predicate=fires_label("well-defined-on-quotient"),
        run=run_unbuilt_stub("well-defined-on-quotient"),
    ),
    RegistryEntry(
        label="cases-exhaustive",
        check="cases-exhaustive",
        source="cas_checks.stub",
        predicate=fires_label("cases-exhaustive"),
        run=run_unbuilt_stub("cases-exhaustive"),
    ),
)

REGISTRY_BY_LABEL = {entry.label: entry for entry in REGISTRY}


def static_fires_for_pattern(pattern: str) -> list[str]:
    return list(cas_select.CHECK_MENU.get(pattern, []))


def check_rows(select_output: dict[str, Any]) -> list[dict[str, Any]]:
    rows = select_output.get("checks")
    if rows is not None:
        return list(rows)
    return [
        {"step": match.get("step"), "pattern": match.get("pattern"), "fires": static_fires_for_pattern(match["pattern"])}
        for match in select_output.get("matches") or []
        if match.get("pattern")
    ]


def select_registry_entries(select_output: dict[str, Any]) -> list[dict[str, Any]]:
    selected = []
    for row in check_rows(select_output):
        for label in row.get("fires") or []:
            entry = REGISTRY_BY_LABEL.get(label)
            if entry is None:
                selected.append(
                    {
                        "step": row.get("step"),
                        "pattern": row.get("pattern"),
                        "label": label,
                        "check": "unknown",
                        "source": "cas_checks.missing",
                        "registered": False,
                    }
                )
            elif entry.predicate(row):
                selected.append(
                    {
                        "step": row.get("step"),
                        "pattern": row.get("pattern"),
                        "label": entry.label,
                        "check": entry.check,
                        "source": entry.source,
                        "registered": True,
                    }
                )
    return selected


def run_selected_checks(
    select_output: dict[str, Any],
    *,
    graph_path: Path | None = None,
    semcheck_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = CheckContext(graph_path=graph_path, semcheck_report=semcheck_report)
    cached_report = semcheck_report
    results = []
    for row in check_rows(select_output):
        for label in row.get("fires") or []:
            entry = REGISTRY_BY_LABEL.get(label)
            if entry is None:
                results.append(
                    na_result(label, "unknown", row, f"N/A: {label} has no registry entry")
                )
                continue
            if cached_report is None and graph_path is not None and entry.source == "scripts/iatc_semcheck.bb":
                cached_report = load_semcheck_report(graph_path)
                context = CheckContext(graph_path=graph_path, semcheck_report=cached_report)
            results.append(entry.run(context, row, label))
    statuses = [row.get("status") for row in results]
    return {
        "check": "cas-check-registry",
        "status": "na" if not results else ("pass" if all(row.get("pass") for row in results) else "fail"),
        "pass": all(row.get("pass") for row in results),
        "rate": None,
        "reasons": [] if results else ["N/A: no matched patterns fired registered checks"],
        "per-item": results,
        "selected": select_registry_entries(select_output),
        "summary": {
            "pass": statuses.count("pass"),
            "fail": statuses.count("fail"),
            "na": statuses.count("na"),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("select_json", type=Path, help="CAS select JSON payload")
    parser.add_argument("--graph", type=Path, default=None, help="IATC graph EDN to run rung-2 checks")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = json.loads(args.select_json.read_text())
    result = run_selected_checks(payload, graph_path=args.graph)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
