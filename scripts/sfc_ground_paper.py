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
STOP_WORDS = {
    "and",
    "or",
    "share",
    "the",
    "with",
}


def load_candidate(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def inline_math(text: str) -> list[str]:
    spans = []
    patterns = [
        r"\$\$(.+?)\$\$",
        r"\\\[(.+?)\\\]",
        r"\\\((.+?)\\\)",
        r"(?<!\\)(?<!\$)\$(?!\$)([^$]+?)(?<!\\)\$(?!\$)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.S):
            spans.append((match.start(), match.group(1)))
    out = []
    for _, raw in sorted(spans, key=lambda row: row[0]):
        formula = " ".join(raw.split())
        if formula:
            out.append(formula)
    return out


def binder_definienda(candidate: dict[str, Any]) -> list[str]:
    out = []
    for line in candidate.get("binder-context") or []:
        match = re.search(r"definiendum #\d+:\s*\$([^$]+)\$", line)
        if match:
            formula = " ".join(match.group(1).split())
            if formula:
                out.append(formula)
    return out


def candidate_formulas(candidate: dict[str, Any]) -> list[str]:
    for key in ("formula", "math"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return [value.strip()]
    formulas = candidate.get("formulas")
    if isinstance(formulas, list):
        out = [str(f).strip() for f in formulas if str(f).strip()]
        if out:
            return out
    binder = binder_definienda(candidate)
    if binder:
        return binder
    fragments = inline_math(str(candidate.get("source-window") or ""))
    if fragments:
        return fragments
    return []


def candidate_formula(candidate: dict[str, Any]) -> str:
    return ", ".join(candidate_formulas(candidate))


def formula_variants(text: str) -> set[str]:
    return {text, text.replace("\\\\", "\\")}


def symbol_variants(symbol: str) -> set[str]:
    stripped = symbol.strip()
    return {stripped, stripped.replace("\\\\", "\\")}


def atom_in_formula(symbol: str, formula: str) -> bool:
    for sym in symbol_variants(symbol):
        if not sym:
            continue
        for source in formula_variants(formula):
            if sym.startswith("\\"):
                if sym in source:
                    return True
                continue
            if re.fullmatch(r"[A-Za-z]", sym):
                if re.search(rf"(?<![A-Za-z]){re.escape(sym)}(?![A-Za-z])", source):
                    return True
                continue
            if re.search(rf"(?<![A-Za-z]){re.escape(sym)}(?![A-Za-z])", source):
                return True
    return False


def symbol_filter_reason(symbol: str, formula: str) -> str | None:
    normalized = symbol.strip().replace("\\\\", "\\")
    lower = normalized.lower()
    if lower in STOP_WORDS:
        return "drop prose stop-word"
    if not atom_in_formula(normalized, formula):
        return "drop parser artifact not present as a TeX atom"
    if re.fullmatch(r"[a-z]{2,}", normalized):
        return "drop pure lowercase word-like token"
    return None


def filter_groundings(result: dict[str, Any], formula: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    kept = []
    dropped = []
    for row in result.get("groundings") or []:
        symbol = str(row.get("symbol", ""))
        reason = symbol_filter_reason(symbol, formula)
        if reason:
            dropped.append({"symbol": symbol, "formula": formula, "reason": reason})
        else:
            kept.append(row)
    return kept, dropped


def merge_groundings(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
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
    formulas = candidate_formulas(candidate)
    results = [sfc.ground(formula, context, backend, model) for formula in formulas]
    kept = []
    dropped = []
    for formula, result in zip(formulas, results):
        rows, drops = filter_groundings(result, formula)
        kept.extend(rows)
        dropped.extend(drops)
    groundings, summary = merge_groundings(kept)
    return {
        "schema": "sfc-symbol-grounding/v0",
        "paper_id": candidate.get("paper-id") or candidate.get("paper_id") or candidate_path.stem.split(".")[0],
        "candidate": str(candidate_path),
        "backend": backend,
        "formula": ", ".join(formulas),
        "formulas": formulas,
        "structures": [result.get("structure") for result in results],
        "groundings": groundings,
        "dropped_symbols": dropped,
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
