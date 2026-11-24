"""Verify that reader-clear loci are verbatim passages from supplied texts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_WHITESPACE = re.compile(r"\s+")
_ELLIPSIS = re.compile(r"(?:\.\.\.|…)")
MIN_LOCUS_LENGTH = 12


def normalize(text: str) -> str:
    """Collapse whitespace and strip ends; deliberately change nothing else."""
    return _WHITESPACE.sub(" ", text).strip()


@dataclass(frozen=True)
class Failure:
    file: str
    id: str
    field: str
    reason: str
    locus: str

    def as_dict(self) -> dict[str, str]:
        return {
            "file": self.file,
            "id": self.id,
            "field": self.field,
            "reason": self.reason,
            "locus": self.locus[:60],
        }


@dataclass(frozen=True)
class CheckResult:
    rows: int
    failures: tuple[Failure, ...]

    @property
    def failure_count(self) -> int:
        return len(self.failures)

    def as_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "failures": self.failure_count,
            "failure_rows": [failure.as_dict() for failure in self.failures],
        }


def _ordered_match(locus: str, text: str) -> bool:
    parts = [part.strip() for part in _ELLIPSIS.split(locus)]
    if any(not part for part in parts):
        return False
    cursor = 0
    for part in parts:
        found = text.find(part, cursor)
        if found < 0:
            return False
        cursor = found + len(part)
    return True


def _locus_failure(
    *, source_file: Path, row_id: str, field: str, value: object, text: str | None
) -> Failure | None:
    if not isinstance(value, str):
        return Failure(str(source_file), row_id, field, "locus is not a string", str(value))
    locus = normalize(value)
    if not locus:
        return None
    if len(locus) < MIN_LOCUS_LENGTH:
        return Failure(str(source_file), row_id, field, "locus is shorter than 12 characters", locus)
    if text is None:
        return Failure(str(source_file), row_id, field, "no informal text was supplied", locus)
    if not _ordered_match(locus, text):
        return Failure(str(source_file), row_id, field, "locus not found verbatim", locus)
    return None


def check_readings(
    print_path: Path,
    informal_path: Path | None,
    reading_paths: Iterable[Path],
) -> CheckResult:
    """Check JSONL rows, returning at most one failure for each failing row."""
    print_text = normalize(print_path.read_text(encoding="utf-8"))
    informal_text = (
        normalize(informal_path.read_text(encoding="utf-8")) if informal_path else None
    )
    failures: list[Failure] = []
    row_count = 0
    for reading_path in reading_paths:
        with reading_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row_count += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    failures.append(
                        Failure(str(reading_path), "<missing>", "id", "invalid JSON", line.strip())
                    )
                    continue
                if not isinstance(row, dict):
                    failures.append(
                        Failure(
                            str(reading_path), "<missing>", "id", "row is not an object", line.strip()
                        )
                    )
                    continue
                row_id = row.get("id")
                if not isinstance(row_id, str) or not row_id:
                    failures.append(
                        Failure(
                            str(reading_path),
                            "<missing>",
                            "id",
                            f"missing id at line {line_number}",
                            "",
                        )
                    )
                    continue
                checks = (("locus_P", print_text), ("locus_I", informal_text))
                for field, text in checks:
                    value = row.get(field)
                    if value in (None, ""):
                        continue
                    failure = _locus_failure(
                        source_file=reading_path,
                        row_id=row_id,
                        field=field,
                        value=value,
                        text=text,
                    )
                    if failure:
                        failures.append(failure)
                        break
    return CheckResult(row_count, tuple(failures))


def render_text(result: CheckResult) -> str:
    lines = [
        "\t".join(
            [failure.file, failure.id, failure.field, failure.reason, failure.locus[:60]]
        )
        for failure in result.failures
    ]
    lines.append(f"{result.rows} rows, {result.failure_count} failures")
    return "\n".join(lines)
