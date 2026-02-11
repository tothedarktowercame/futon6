#!/usr/bin/env python3
"""Run Stage 6 reverse-morphogenesis prompts through Codex CLI.

Reads JSONL prompt records, executes each prompt via `codex exec`, parses the
JSON response, and writes normalized JSONL results.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


DEFAULT_INPUT = (
    Path(__file__).resolve().parent.parent
    / "se-data"
    / "math-processed"
    / "moist-prompts"
    / "stage6-reverse-morphogenesis.jsonl"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "se-data"
    / "math-processed"
    / "moist-prompts"
    / "stage6-results-codex.jsonl"
)


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "xiang_form": {"type": "string"},
        "xiang_salience": {"type": "string"},
        "arrow_constraint": {"type": "string"},
        "quality": {
            "type": "object",
            "properties": {
                "form": {"type": "string", "enum": ["good", "weak", "broken"]},
                "salience": {"type": "string", "enum": ["good", "weak", "broken"]},
                "arrow": {"type": "string", "enum": ["good", "weak", "broken"]},
            },
            "required": ["form", "salience", "arrow"],
            "additionalProperties": False,
        },
        "situation_S": {"type": "string"},
        "roundtrip_check": {"type": "string"},
    },
    "required": [
        "xiang_form",
        "xiang_salience",
        "arrow_constraint",
        "quality",
        "situation_S",
        "roundtrip_check",
    ],
    "additionalProperties": False,
}


def build_instruction(prompt_text: str) -> str:
    return (
        "You must answer exactly as one JSON object matching the required schema. "
        "Do not wrap JSON in markdown fences. Do not add extra commentary.\n\n"
        + prompt_text
    )


def run_codex_once(
    codex_bin: str,
    model: str,
    cwd: Path,
    schema_path: Path,
    prompt_text: str,
) -> tuple[int, str, str]:
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as out_f:
        out_path = Path(out_f.name)

    cmd = [
        codex_bin,
        "exec",
        "--cd",
        str(cwd),
        "--sandbox",
        "workspace-write",
        "--model",
        model,
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(out_path),
        "-",
    ]
    proc = subprocess.run(
        cmd,
        input=build_instruction(prompt_text),
        text=True,
        capture_output=True,
    )
    try:
        response_text = out_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        response_text = ""
    out_path.unlink(missing_ok=True)
    return proc.returncode, response_text, proc.stderr.strip()


def normalize_response(raw_response: str) -> tuple[dict | None, bool]:
    try:
        parsed = json.loads(raw_response)
    except Exception:
        return None, True
    if not isinstance(parsed, dict):
        return None, True
    return parsed, False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--model", default="gpt-5.3-codex")
    ap.add_argument("--codex-bin", default="codex")
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Working directory passed to codex exec --cd",
    )
    args = ap.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)

    counts = {
        "form": Counter({"good": 0, "weak": 0, "broken": 0}),
        "salience": Counter({"good": 0, "weak": 0, "broken": 0}),
        "arrow": Counter({"good": 0, "weak": 0, "broken": 0}),
    }
    parse_errors = 0
    processed = 0

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as sf:
        json.dump(RESPONSE_SCHEMA, sf, ensure_ascii=True, indent=2)
        schema_path = Path(sf.name)

    try:
        with args.input.open("r", encoding="utf-8") as fin, args.output.open(
            "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                if processed >= args.limit:
                    break
                record = json.loads(line)
                entity_id = record.get("entity_id")
                question_id = record.get("question_id")
                prompt = record.get("prompt", "")

                rc, raw_response, stderr_text = run_codex_once(
                    codex_bin=args.codex_bin,
                    model=args.model,
                    cwd=args.repo_root,
                    schema_path=schema_path,
                    prompt_text=prompt,
                )

                out = {"entity_id": entity_id, "question_id": question_id}

                parsed_obj, bad = normalize_response(raw_response)
                if rc == 0 and not bad and parsed_obj is not None:
                    out.update(parsed_obj)
                    quality = parsed_obj.get("quality")
                    if isinstance(quality, dict):
                        for dim in ("form", "salience", "arrow"):
                            v = quality.get(dim)
                            if isinstance(v, str):
                                v = v.strip().lower()
                                if v in ("good", "weak", "broken"):
                                    counts[dim][v] += 1
                else:
                    parse_errors += 1
                    raw_parts = []
                    if raw_response:
                        raw_parts.append(raw_response)
                    if rc != 0:
                        raw_parts.append(f"[codex_exit_code={rc}]")
                    if stderr_text:
                        raw_parts.append(f"[stderr]\n{stderr_text}")
                    out["raw"] = "\n".join(raw_parts).strip()
                    out["parse_error"] = True

                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                processed += 1
                print(
                    f"[{processed:02d}/{args.limit}] entity_id={entity_id} "
                    f"question_id={question_id} parse_error={out.get('parse_error', False)}"
                )
                sys.stdout.flush()
    finally:
        schema_path.unlink(missing_ok=True)

    valid_json = processed - parse_errors
    print("---SUMMARY---")
    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"model={args.model}")
    print(f"processed={processed}")
    print(f"valid_json={valid_json}")
    print(f"parse_errors={parse_errors}")
    print("象: " + ", ".join(f"{k}={counts['form'][k]}" for k in ("good", "weak", "broken")))
    print(
        "香: "
        + ", ".join(f"{k}={counts['salience'][k]}" for k in ("good", "weak", "broken"))
    )
    print("←: " + ", ".join(f"{k}={counts['arrow'][k]}" for k in ("good", "weak", "broken")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
