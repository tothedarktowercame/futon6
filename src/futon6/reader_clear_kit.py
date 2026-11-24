"""Build the mechanically derived text kit for each reader-clear seat.

Proof step counts count sentence boundaries at ``. ``, ``.\n``, and the end
of non-empty text, except while inside ``\\(...\\)`` or ``\\[...\\]`` math.
This intentionally simple rule is deterministic and only supplies a rate
denominator.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "reader-clear.kit/v0"


def replace_leanrefs(text: str) -> str:
    """Replace balanced ``\\leanref{display}{target}`` forms with display."""
    marker = r"\leanref"
    output: list[str] = []
    cursor = 0
    while True:
        start = text.find(marker, cursor)
        if start < 0:
            output.append(text[cursor:])
            return "".join(output)
        output.append(text[cursor:start])
        position = start + len(marker)
        try:
            display, position = _braced(text, position)
            _, position = _braced(text, position)
        except ValueError:
            output.append(marker)
            cursor = start + len(marker)
            continue
        output.append(replace_leanrefs(display))
        cursor = position


def _braced(text: str, position: int) -> tuple[str, int]:
    if position >= len(text) or text[position] != "{":
        raise ValueError("expected opening brace")
    depth = 1
    index = position + 1
    while index < len(text):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[position + 1:index], index + 1
        index += 1
    raise ValueError("unclosed brace")


def step_count(proof: str) -> int:
    """Count the sentence boundaries described in the module docstring."""
    count = 0
    math_end: str | None = None
    last_boundary = 0
    index = 0
    while index < len(proof):
        pair = proof[index:index + 2]
        if math_end is None and pair in (r"\(", r"\["):
            math_end = r"\)" if pair == r"\(" else r"\]"
            index += 2
            continue
        if math_end is not None and pair == math_end:
            math_end = None
            index += 2
            continue
        if math_end is None and proof[index] == "." and index + 1 < len(proof):
            if proof[index + 1] in " \n":
                count += 1
                last_boundary = index + 2
                index += 2
                continue
        index += 1
    if proof[last_boundary:].strip():
        count += 1
    return count


def prepare_informal(text: str) -> tuple[str, bool]:
    """Drop leading HTML comments and content beginning at a Stage 3 heading."""
    lines = text.splitlines(keepends=True)
    while lines and lines[0].lstrip().startswith("<!--"):
        while lines:
            line = lines.pop(0)
            if "-->" in line:
                break
    for index, line in enumerate(lines):
        heading = line.lstrip("#* ").lower()
        if heading.startswith("stage 3"):
            return "".join(lines[:index]), True
    return "".join(lines), False


def _render_rows(head_dir: Path) -> Iterable[dict[str, Any]]:
    paths = sorted(head_dir.glob("*.jsonl"))
    if not paths:
        raise ValueError(f"no render JSONL files in {head_dir}")
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"{path}:{number}: invalid JSON: {error}") from error


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _print_text(problem_id: str, rows: list[dict[str, Any]]) -> str:
    parts = [f"# {problem_id}: machine-printed proof (PaperPrinter render of the formal proof)"]
    for number, row in enumerate(rows, 1):
        body = replace_leanrefs(str(row.get("body") or ""))
        proof = replace_leanrefs(str(row.get("proof") or ""))
        parts.append(f"## Declaration {number}: {row.get('declaration', '')}\n\n"
                     f"**Statement.** {body}\n\n**Proof.**\n\n{proof}")
    return "\n\n".join(parts) + "\n"


def build_kits(head_dir: Path, apm_problems: Path, sample_path: Path,
               out_dir: Path) -> list[dict[str, Any]]:
    """Build all kits described by *sample_path* and return their metadata."""
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    if sample.get("schema") != "reader-clear.sample/v0":
        raise ValueError(f"unsupported sample schema: {sample.get('schema')!r}")
    rows_by_module: dict[str, list[dict[str, Any]]] = {}
    for row in _render_rows(head_dir):
        module = row.get("module")
        if isinstance(module, str):
            rows_by_module.setdefault(module, []).append(row)

    results = []
    for drawn in sample.get("drawn", []):
        problem_id = drawn["problem_id"]
        module = drawn["module"]
        rows = rows_by_module.get(module, [])
        if not rows:
            raise ValueError(f"no render rows for drawn module {module}")
        source_dir = apm_problems / problem_id
        problem_source = source_dir / "problem.md"
        informal_source = source_dir / "informal-solution.md"
        if not problem_source.is_file():
            raise ValueError(f"missing problem.md for {problem_id}: {problem_source}")
        if not informal_source.is_file():
            raise ValueError(f"missing informal-solution.md for {problem_id}: {informal_source}")

        kit_dir = out_dir / problem_id
        kit_dir.mkdir(parents=True, exist_ok=True)
        problem_path = kit_dir / "problem.md"
        informal_path = kit_dir / "version-I.informal.md"
        print_path = kit_dir / "version-P.paperprinter.md"
        problem_path.write_bytes(problem_source.read_bytes())
        informal, was_cut = prepare_informal(informal_source.read_text(encoding="utf-8"))
        informal_path.write_text(informal, encoding="utf-8")
        print_text = _print_text(problem_id, rows)
        print_path.write_text(print_text, encoding="utf-8")
        declarations = [
            {"declaration": row.get("declaration", ""),
             "step_count": step_count(str(row.get("proof") or ""))}
            for row in rows
        ]
        metadata = {
            "schema": SCHEMA,
            "head": head_dir.name,
            "problem_id": problem_id,
            "module": module,
            "declarations": declarations,
            "total_words": len(print_text.split()),
            "informal_cut": was_cut,
            "sha256": {
                "problem.md": _sha256(problem_path),
                "version-I.informal.md": _sha256(informal_path),
                "version-P.paperprinter.md": _sha256(print_path),
            },
        }
        (kit_dir / "kit.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        results.append(metadata)
    return results
