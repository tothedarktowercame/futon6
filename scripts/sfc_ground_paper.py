#!/usr/bin/env python3
"""Per-paper SFC2b symbol-grounding driver.

Bridges the per-formula `sfc_symbol_grounding.ground` API to the enriched IATC
candidate artifact used by CAS-CERT.  The deterministic path is the `stub`
backend; `openai` delegates symbol proposals to the configured LLM while keeping
the same verbatim-evidence check in `sfc_symbol_grounding`.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import sfc_symbol_grounding as sfc

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "data" / "symbol-grounding" / "loop-run-70b"


def load_candidate(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def inline_math(text: str) -> list[str]:
    out = []
    for match in re.finditer(r"\$(.+?)\$", text, re.S):
        formula = " ".join(match.group(1).split())
        if formula:
            out.append(formula)
    return out


def candidate_formula(candidate: dict[str, Any]) -> str:
    for key in ("formula", "math"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    formulas = candidate.get("formulas")
    if isinstance(formulas, list):
        joined = ", ".join(str(f).strip() for f in formulas if str(f).strip())
        if joined:
            return joined
    fragments = inline_math(str(candidate.get("source-window") or ""))
    if fragments:
        return ", ".join(fragments)
    binder_symbols = []
    for line in candidate.get("binder-context") or []:
        match = re.search(r"definiendum #\d+:\s*\$([^$]+)\$", line)
        if match:
            binder_symbols.append(match.group(1).strip())
    return ", ".join(binder_symbols)


def merge_groundings(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    merged: dict[str, dict[str, Any]] = {}
    for result in results:
        for row in result.get("groundings") or []:
            symbol = str(row.get("symbol", ""))
            if not symbol or symbol in merged:
                continue
            merged[symbol] = dict(row)
    counts = Counter(row.get("status") for row in merged.values())
    summary = {
        "symbols": len(merged),
        "grounded": counts["grounded"],
        "undefined_in_context": counts["undefined-in-context"],
        "unsupported": counts["unsupported"],
    }
    return list(merged.values()), summary


def ground_candidate(
    candidate_path: Path,
    *,
    backend: str = "stub",
    model: str = "mark4-70b",
) -> dict[str, Any]:
    candidate = load_candidate(candidate_path)
    context = str(candidate.get("source-window") or "")
    formula = candidate_formula(candidate)
    result = sfc.ground(formula, context, backend, model)
    groundings, summary = merge_groundings([result])
    return {
        "schema": "sfc-symbol-grounding/v0",
        "paper_id": candidate.get("paper-id") or candidate.get("paper_id") or candidate_path.stem.split(".")[0],
        "candidate": str(candidate_path),
        "backend": backend,
        "formula": formula,
        "structure": result.get("structure"),
        "groundings": groundings,
        "summary": summary,
    }


def write_doc(doc: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--backend", choices=["stub", "openai"], default="stub")
    parser.add_argument("--model", default="mark4-70b")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    doc = ground_candidate(args.candidate, backend=args.backend, model=args.model)
    out_path = args.out or (args.out_dir / f"{doc['paper_id']}.symbols.json")
    write_doc(doc, out_path)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
