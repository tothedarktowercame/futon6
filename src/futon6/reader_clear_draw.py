"""Deterministically draw the stratified reader-clear mission sample.

Render sizes use whitespace-split words in every row's ``proof`` field. Length
thirds use nearest-rank tercile cut points over the eligible module sizes. All
random choices consume ``random.Random(seed)`` values over sorted populations.
"""
from __future__ import annotations

import gzip
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "reader-clear.sample/v0"


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error


def problem_id(module: str) -> str:
    parts = module.split(".")
    if len(parts) < 2 or not parts[1]:
        raise ValueError(f"module does not contain a problem id: {module!r}")
    return parts[1]


def load_modules(head_dir: Path) -> dict[str, dict[str, Any]]:
    modules: dict[str, dict[str, Any]] = {}
    render_files = sorted(head_dir.glob("*.jsonl"))
    if not render_files:
        raise ValueError(f"no render JSONL files in {head_dir}")
    for path in render_files:
        for row in _jsonl(path):
            module = row.get("module")
            if not isinstance(module, str) or not module:
                raise ValueError(f"render row in {path} lacks module")
            item = modules.setdefault(module, {"records": 0, "words": 0,
                                               "proof_error": False})
            item["records"] += 1
            item["words"] += len((row.get("proof") or "").split())
            item["proof_error"] = item["proof_error"] or bool(row.get("proof_error"))
    return modules


def load_inked_modules(head_dir: Path) -> set[str]:
    path = head_dir / "residual-ink-annotations.jsonl.gz"
    if not path.exists():
        raise ValueError(f"missing ink annotations: {path}")
    return {
        row["module"]
        for row in _jsonl(path)
        if isinstance(row.get("module"), str) and row.get("spans")
    }


def overlay_modules(head_dir: Path) -> set[str] | None:
    path = head_dir / "residual-ink-overlay.jsonl.gz"
    if not path.exists():
        return None
    found: set[str] = set()
    for row in _jsonl(path):
        module = (row.get("record") or {}).get("module")
        if isinstance(module, str) and module:
            found.add(module)
    return found


def tercile_cut_points(sizes: list[int]) -> tuple[int, int]:
    if not sizes:
        raise ValueError("cannot stratify an empty eligible population")
    ordered = sorted(sizes)
    return (ordered[math.ceil(len(ordered) / 3) - 1],
            ordered[math.ceil(2 * len(ordered) / 3) - 1])


def length_third(words: int, cuts: tuple[int, int]) -> str:
    if words <= cuts[0]:
        return "short"
    if words <= cuts[1]:
        return "middle"
    return "long"


def largest_remainder(populations: dict[str, int], n: int) -> dict[str, int]:
    total = sum(populations.values())
    if n < 0 or n > total:
        raise ValueError(f"sample size {n} exceeds eligible population {total}")
    exact = {key: n * value / total for key, value in populations.items()}
    allocated = {key: math.floor(value) for key, value in exact.items()}
    remaining = n - sum(allocated.values())
    order = sorted(populations, key=lambda key: (-(exact[key] - allocated[key]), key))
    for key in order[:remaining]:
        allocated[key] += 1
    return allocated


def _random_order(items: Iterable[str], rng: random.Random) -> list[str]:
    return [item for _, item in sorted((rng.random(), item) for item in sorted(items))]


def build_draw(head_dir: Path, apm_problems: Path, seed: int, *, n: int = 30,
               controls: int = 5, exclude: Iterable[str] = ()) -> dict[str, Any]:
    modules = load_modules(head_dir)
    inked = load_inked_modules(head_dir)
    overlay = overlay_modules(head_dir)
    excluded_ids = set(exclude)

    zero_ink = {module: data for module, data in modules.items() if module not in inked}
    proof_errors = {module for module, data in zero_ink.items() if data["proof_error"]}
    after_errors = {module: data for module, data in zero_ink.items()
                    if module not in proof_errors}
    explicitly_excluded = {module for module in after_errors
                           if problem_id(module) in excluded_ids}
    after_excludes = {module: data for module, data in after_errors.items()
                      if module not in explicitly_excluded}
    missing_solutions = {
        module for module in after_excludes
        if not (apm_problems / problem_id(module) / "informal-solution.md").is_file()
    }
    eligible = {module: data for module, data in after_excludes.items()
                if module not in missing_solutions}

    if n + controls > len(eligible):
        raise ValueError(f"sample plus controls ({n + controls}) exceeds eligible population "
                         f"({len(eligible)})")

    cuts = tercile_cut_points([data["words"] for data in eligible.values()])
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for module, data in eligible.items():
        pid = problem_id(module)
        stratum = f"{pid[0]}:{length_third(data['words'], cuts)}"
        data["stratum"] = stratum
        by_stratum[stratum].append(module)

    populations = {key: len(value) for key, value in sorted(by_stratum.items())}
    allocations = largest_remainder(populations, n)
    rng = random.Random(seed)
    sampled: list[str] = []
    for stratum in sorted(by_stratum):
        sampled.extend(_random_order(by_stratum[stratum], rng)[:allocations[stratum]])
    sampled_set = set(sampled)
    control_modules = _random_order(set(eligible) - sampled_set, rng)[:controls]

    def record(module: str, role: str) -> dict[str, Any]:
        data = eligible[module]
        return {"problem_id": problem_id(module), "module": module,
                "records": data["records"], "words": data["words"],
                "stratum": data["stratum"], "role": role}

    return {
        "schema": SCHEMA,
        "head": head_dir.name,
        "seed": seed,
        "counts": {
            "modules": len(modules),
            "inked": len(set(modules) & inked),
            "dropped_proof_error": len(proof_errors),
            "dropped_excluded": len(explicitly_excluded),
            "dropped_missing_informal_solution": len(missing_solutions),
            "eligible": len(eligible),
            "overlay_modules_agree": overlay is None or overlay == inked,
        },
        "length_tercile_cut_points": {"short_max": cuts[0], "middle_max": cuts[1]},
        "strata": [
            {"stratum": key, "population": populations[key],
             "allocated": allocations[key]}
            for key in sorted(populations)
        ],
        "drawn": ([record(module, "sample") for module in sampled]
                  + [record(module, "planted-control") for module in control_modules]),
    }


def write_draw(output: Path, draw: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(draw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
