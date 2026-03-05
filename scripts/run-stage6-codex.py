#!/usr/bin/env python3
"""Run Stage 6 reverse-morphogenesis prompts through Codex CLI.

Reads JSONL prompt records, executes each prompt via `codex exec` with an
output schema, and writes normalized JSONL results.

Hardening features:
- resumable runs (append mode, skip already processed records)
- retry on transient parse/CLI failures
- deterministic random sampling for moist runs
- optional success-rate gate for CI-like checks
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable


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


def _record_key(record: dict) -> tuple[str, str]:
    entity_id = str(record.get("entity_id", ""))
    question_id = str(record.get("question_id", ""))
    return entity_id, question_id


def load_processed_keys(output_path: Path) -> set[tuple[str, str]]:
    processed: set[tuple[str, str]] = set()
    if not output_path.exists():
        return processed
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            processed.add(_record_key(rec))
    return processed


def iter_input_records(
    input_path: Path,
    limit: int,
    shuffle: bool,
    seed: int,
    skip_keys: set[tuple[str, str]],
) -> Iterable[dict]:
    if limit <= 0:
        return []

    if not shuffle:
        out = []
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                if len(out) >= limit:
                    break
                rec = json.loads(line)
                if _record_key(rec) in skip_keys:
                    continue
                out.append(rec)
        return out

    rng = random.Random(seed)
    reservoir: list[dict] = []
    seen = 0
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if _record_key(rec) in skip_keys:
                continue
            seen += 1
            if len(reservoir) < limit:
                reservoir.append(rec)
                continue
            j = rng.randrange(seen)
            if j < limit:
                reservoir[j] = rec
    return reservoir


def run_with_retries(
    codex_bin: str,
    model: str,
    cwd: Path,
    schema_path: Path,
    prompt_text: str,
    max_retries: int,
) -> tuple[int, str, str, dict | None, int]:
    attempts = 0
    last_rc = 1
    last_raw = ""
    last_stderr = ""
    last_parsed: dict | None = None

    total_attempts = max(1, max_retries + 1)
    for _ in range(total_attempts):
        attempts += 1
        rc, raw_response, stderr_text = run_codex_once(
            codex_bin=codex_bin,
            model=model,
            cwd=cwd,
            schema_path=schema_path,
            prompt_text=prompt_text,
        )
        parsed_obj, bad = normalize_response(raw_response)
        last_rc = rc
        last_raw = raw_response
        last_stderr = stderr_text
        last_parsed = parsed_obj
        if rc == 0 and not bad and parsed_obj is not None:
            return rc, raw_response, stderr_text, parsed_obj, attempts

    return last_rc, last_raw, last_stderr, last_parsed, attempts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--model", default="gpt-5.3-codex")
    ap.add_argument("--codex-bin", default="codex")
    ap.add_argument(
        "--shuffle",
        action="store_true",
        help="Sample records uniformly at random instead of taking first N",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --shuffle is set",
    )
    ap.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Retries after first failed attempt (default=1 => up to 2 attempts)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Append to output and skip already-processed entity/question keys",
    )
    ap.add_argument(
        "--target-success-rate",
        type=float,
        default=0.90,
        help="Target valid-json ratio used by --enforce-target",
    )
    ap.add_argument(
        "--enforce-target",
        action="store_true",
        help="Exit non-zero if success rate falls below --target-success-rate",
    )
    ap.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write run summary JSON",
    )
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
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    skip_keys: set[tuple[str, str]] = set()
    if args.resume:
        skip_keys = load_processed_keys(args.output)

    selected = list(
        iter_input_records(
            input_path=args.input,
            limit=args.limit,
            shuffle=args.shuffle,
            seed=args.seed,
            skip_keys=skip_keys,
        )
    )
    if not selected:
        print("No records selected (input exhausted or all selected records already processed).")
        return 0

    counts = {
        "form": Counter({"good": 0, "weak": 0, "broken": 0}),
        "salience": Counter({"good": 0, "weak": 0, "broken": 0}),
        "arrow": Counter({"good": 0, "weak": 0, "broken": 0}),
    }
    parse_errors = 0
    processed = 0
    total_attempts = 0

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as sf:
        json.dump(RESPONSE_SCHEMA, sf, ensure_ascii=True, indent=2)
        schema_path = Path(sf.name)

    try:
        open_mode = "a" if args.resume and args.output.exists() else "w"
        with args.output.open(open_mode, encoding="utf-8") as fout:
            for record in selected:
                entity_id = record.get("entity_id")
                question_id = record.get("question_id")
                prompt = record.get("prompt", "")

                rc, raw_response, stderr_text, parsed_obj, attempts = run_with_retries(
                    codex_bin=args.codex_bin,
                    model=args.model,
                    cwd=args.repo_root,
                    schema_path=schema_path,
                    prompt_text=prompt,
                    max_retries=args.max_retries,
                )
                total_attempts += attempts

                out = {"entity_id": entity_id, "question_id": question_id}

                if rc == 0 and parsed_obj is not None:
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
                out["attempts"] = attempts

                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                processed += 1
                print(
                    f"[{processed:03d}/{len(selected)}] entity_id={entity_id} "
                    f"question_id={question_id} parse_error={out.get('parse_error', False)} "
                    f"attempts={attempts}"
                )
                sys.stdout.flush()
    finally:
        schema_path.unlink(missing_ok=True)

    valid_json = processed - parse_errors
    success_rate = (valid_json / processed) if processed else 0.0
    print("---SUMMARY---")
    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"model={args.model}")
    print(f"resume={args.resume}")
    print(f"shuffle={args.shuffle}")
    print(f"seed={args.seed}")
    print(f"max_retries={args.max_retries}")
    print(f"processed={processed}")
    print(f"valid_json={valid_json}")
    print(f"parse_errors={parse_errors}")
    print(f"success_rate={success_rate:.4f}")
    print(f"avg_attempts={(total_attempts / processed) if processed else 0.0:.3f}")
    print("象: " + ", ".join(f"{k}={counts['form'][k]}" for k in ("good", "weak", "broken")))
    print(
        "香: "
        + ", ".join(f"{k}={counts['salience'][k]}" for k in ("good", "weak", "broken"))
    )
    print("←: " + ", ".join(f"{k}={counts['arrow'][k]}" for k in ("good", "weak", "broken")))

    if args.summary_json is not None:
        summary = {
            "input": str(args.input),
            "output": str(args.output),
            "model": args.model,
            "resume": bool(args.resume),
            "shuffle": bool(args.shuffle),
            "seed": int(args.seed),
            "max_retries": int(args.max_retries),
            "processed": int(processed),
            "valid_json": int(valid_json),
            "parse_errors": int(parse_errors),
            "success_rate": float(success_rate),
            "avg_attempts": float((total_attempts / processed) if processed else 0.0),
            "quality_counts": {
                dim: {k: int(v) for k, v in counts[dim].items()}
                for dim in ("form", "salience", "arrow")
            },
            "target_success_rate": float(args.target_success_rate),
            "target_met": bool(success_rate >= args.target_success_rate),
        }
        args.summary_json.write_text(
            json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"summary_json={args.summary_json}")

    if args.enforce_target and success_rate < args.target_success_rate:
        print(
            f"ERROR: success_rate={success_rate:.4f} below target={args.target_success_rate:.4f}",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
