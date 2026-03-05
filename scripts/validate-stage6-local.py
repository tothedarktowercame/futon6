#!/usr/bin/env python3
"""Local validation harness for Stage 6 reverse-morphogenesis prompts.

Samples prompts from the JSONL file and submits them to Claude or Codex
(via your subscription CLIs), compares prompt variants, and scores results.

Usage examples:

  # Run 20 random prompts through Claude:
  python scripts/validate-stage6-local.py --backend claude --limit 20

  # Run specific entity IDs through Codex:
  python scripts/validate-stage6-local.py --backend codex --entity-ids se-math-1,se-math-5,se-math-8

  # Compare two prompt variants on the same 30 samples:
  python scripts/validate-stage6-local.py --backend claude --limit 30 --variant both

  # Dry run: dump prompts to stdout for manual testing:
  python scripts/validate-stage6-local.py --backend dry-run --limit 3

  # Run with vLLM local server (for superpod prep):
  python scripts/validate-stage6-local.py --backend vllm --vllm-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Schema and constants
# ---------------------------------------------------------------------------

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
        "xiang_form", "xiang_salience", "arrow_constraint",
        "quality", "situation_S", "roundtrip_check",
    ],
    "additionalProperties": False,
}

EXPECTED_KEYS = set(RESPONSE_SCHEMA["required"])

# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------

def _original_prompt(prompt_text: str) -> str:
    """The original superpod prompt, as-is from the JSONL."""
    return prompt_text


def _tightened_prompt(prompt_text: str) -> str:
    """Tightened variant: stronger JSON-only instruction, schema example moved
    to the end, explicit token-budget hint for local models."""
    # Extract the Q&A section (everything after "Now analyze this Q&A pair")
    marker = "Now analyze this Q&A pair"
    idx = prompt_text.find(marker)
    if idx == -1:
        # Fallback: return with prefix
        return (
            "Return ONLY a JSON object. No prose, no markdown, no explanation.\n\n"
            + prompt_text
        )

    qa_section = prompt_text[idx:]

    return f"""You are a mathematics education researcher. Analyze the Q&A pair below.

Return ONLY a valid JSON object with these exact keys (no other text):
- "xiang_form": the mathematical object/structure (string, ≤40 words)
- "xiang_salience": what understanding is sought (string, ≤40 words)
- "arrow_constraint": what you'd need to know/prove for form→understanding (string, ≤40 words)
- "quality": {{"form": "good"|"weak"|"broken", "salience": "good"|"weak"|"broken", "arrow": "good"|"weak"|"broken"}}
- "situation_S": a concrete situation from which this question naturally arises (string, ≤60 words)
- "roundtrip_check": does the situation produce this question? (string, ≤30 words)

Quality ratings:
- 象 form: Is the mathematical object well-specified?
- 香 salience: Does the questioner know WHY they want to know?
- ← arrow: Does the question connect form to understanding?

{qa_section}"""


PROMPT_VARIANTS = {
    "original": _original_prompt,
    "tightened": _tightened_prompt,
}

# ---------------------------------------------------------------------------
# JSON parser (from superpod-job.py — proven at scale)
# ---------------------------------------------------------------------------

def _parse_json_object_response(text):
    """Extract a JSON object from LLM response text with progressive tolerance."""

    def _score_obj(obj):
        if not isinstance(obj, dict):
            return 0
        return sum(1 for k in EXPECTED_KEYS if k in obj)

    def _normalize_quotes(s):
        return (s.replace("\u201c", '"').replace("\u201d", '"')
                 .replace("\u2018", "'").replace("\u2019", "'"))

    def _strip_fences(s):
        s = s.strip()
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
            s = re.sub(r"\s*```$", "", s)
        return s.strip()

    def _cleanup_json(s):
        s = _strip_fences(_normalize_quotes(s))
        s = re.sub(r",\s*([}\]])", r"\1", s)
        return s

    def _balanced_span(s, start_idx):
        depth = 0
        in_string = False
        escape = False
        for i in range(start_idx, len(s)):
            ch = s[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
        return None

    def _try_parse_obj(candidate):
        cleaned = _cleanup_json(candidate)
        if not cleaned:
            return None
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        balance = cleaned.count("{") - cleaned.count("}")
        if balance > 0:
            healed = cleaned + ("}" * balance)
            healed = re.sub(r",\s*([}\]])", r"\1", healed)
            try:
                obj = json.loads(healed)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
        try:
            obj = ast.literal_eval(cleaned)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        return None

    text = str(text or "")
    start = text.find("{")
    if start == -1:
        return {"raw": text, "parse_error": "no JSON object found"}

    best_obj = None
    best_score = -1
    best_len = -1
    for m in re.finditer(r"\{", text):
        s = m.start()
        end = _balanced_span(text, s)
        if end is None:
            continue
        candidate = text[s:end + 1]
        parsed = _try_parse_obj(candidate)
        if parsed is not None:
            score = _score_obj(parsed)
            clen = len(candidate)
            if score > best_score or (score == best_score and clen > best_len):
                best_obj = parsed
                best_score = score
                best_len = clen

    tail = text[start:]
    parsed = _try_parse_obj(tail)
    if parsed is not None:
        tail_score = _score_obj(parsed)
        if tail_score > best_score or (tail_score == best_score and len(tail) > best_len):
            return parsed
    if best_obj is not None:
        return best_obj

    if tail.count("{") > tail.count("}"):
        return {"raw": tail, "parse_error": "unclosed JSON object"}
    return {"raw": tail, "parse_error": "invalid JSON"}


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def _clean_env() -> dict:
    """Return env dict without CLAUDECODE (avoids nested-session block)."""
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    return env


def run_claude(prompt: str, model: str = "claude-sonnet-4-20250514") -> str:
    """Submit prompt via `claude` CLI (Claude Max subscription)."""
    proc = subprocess.run(
        ["claude", "-p", "--model", model, "--output-format", "text"],
        input=prompt,
        text=True,
        capture_output=True,
        timeout=120,
        env=_clean_env(),
    )
    if proc.returncode != 0:
        return f"[claude_exit_code={proc.returncode}]\n{proc.stderr.strip()}"
    return proc.stdout.strip()


def run_codex(prompt: str, model: str = "gpt-5.3-codex",
              schema_path: Path | None = None) -> str:
    """Submit prompt via `codex exec` (Codex Pro subscription)."""
    cmd = ["codex", "exec", "--sandbox", "workspace-write", "--model", model]
    if schema_path:
        cmd.extend(["--output-schema", str(schema_path)])

    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as out_f:
        out_path = Path(out_f.name)

    cmd.extend(["--output-last-message", str(out_path), "-"])

    proc = subprocess.run(
        cmd,
        input=(
            "You must answer exactly as one JSON object matching the required "
            "schema. Do not wrap JSON in markdown fences. Do not add extra "
            "commentary.\n\n" + prompt
        ),
        text=True,
        capture_output=True,
        timeout=120,
        env=_clean_env(),
    )
    try:
        response = out_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        response = ""
    out_path.unlink(missing_ok=True)

    if proc.returncode != 0 and not response:
        return f"[codex_exit_code={proc.returncode}]\n{proc.stderr.strip()}"
    return response


def run_vllm(prompt: str, model: str = "meta-llama/Llama-3-8B-Instruct",
             base_url: str = "http://localhost:8000") -> str:
    """Submit prompt to a local vLLM server with JSON schema constraint."""
    import urllib.request
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.3,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "stage6", "schema": RESPONSE_SCHEMA},
        },
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[vllm_error] {e}"


def run_dry(prompt: str) -> str:
    """Dry run: print prompt, return empty."""
    print("=" * 72)
    print(prompt[:2000])
    if len(prompt) > 2000:
        print(f"... ({len(prompt) - 2000} more chars)")
    print("=" * 72)
    return ""


BACKENDS = {
    "claude": run_claude,
    "codex": run_codex,
    "vllm": run_vllm,
    "dry-run": run_dry,
}

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_result(parsed: dict) -> dict:
    """Score a parsed result for completeness and quality."""
    scores = {}

    # Parse success
    scores["parsed"] = "parse_error" not in parsed

    if not scores["parsed"]:
        scores["error_type"] = parsed.get("parse_error", "unknown")
        return scores

    # Key completeness
    present = EXPECTED_KEYS & set(parsed.keys())
    scores["keys_present"] = len(present)
    scores["keys_total"] = len(EXPECTED_KEYS)
    scores["complete"] = present == EXPECTED_KEYS

    # Quality enum validity
    quality = parsed.get("quality", {})
    if isinstance(quality, dict):
        for dim in ("form", "salience", "arrow"):
            v = quality.get(dim, "")
            scores[f"quality_{dim}"] = v if v in ("good", "weak", "broken") else "INVALID"
    else:
        scores["quality_invalid"] = True

    # Content quality heuristics
    for field in ("xiang_form", "xiang_salience", "arrow_constraint",
                  "situation_S", "roundtrip_check"):
        val = parsed.get(field, "")
        if isinstance(val, str):
            wc = len(val.split())
            scores[f"{field}_words"] = wc
            # Flag suspiciously short or long
            if wc < 3:
                scores[f"{field}_flag"] = "too_short"
            elif field == "situation_S" and wc > 100:
                scores[f"{field}_flag"] = "too_long"

    return scores


def print_summary(results: list[dict], variant_name: str):
    """Print aggregate summary for a set of results."""
    total = len(results)
    if total == 0:
        print(f"\n[{variant_name}] No results.")
        return

    parsed_ok = sum(1 for r in results if r["scores"]["parsed"])
    complete = sum(1 for r in results if r["scores"].get("complete", False))

    print(f"\n{'=' * 60}")
    print(f"  Variant: {variant_name}  |  N={total}")
    print(f"{'=' * 60}")
    print(f"  Parse success: {parsed_ok}/{total} ({100*parsed_ok/total:.0f}%)")
    print(f"  Complete (6/6 keys): {complete}/{total} ({100*complete/total:.0f}%)")

    if parsed_ok == 0:
        # Show error breakdown
        errors = Counter(r["scores"].get("error_type", "?") for r in results
                        if not r["scores"]["parsed"])
        for err, cnt in errors.most_common():
            print(f"  Error: {err}: {cnt}")
        return

    # Quality distribution
    for dim in ("form", "salience", "arrow"):
        counts = Counter(r["scores"].get(f"quality_{dim}", "?")
                        for r in results if r["scores"]["parsed"])
        dist = ", ".join(f"{k}={v}" for k, v in
                        sorted(counts.items(), key=lambda x: -x[1]))
        label = {"form": "象", "salience": "香", "arrow": "←"}[dim]
        print(f"  {label} quality: {dist}")

    # Word count stats for situation_S
    s_words = [r["scores"].get("situation_S_words", 0)
               for r in results if r["scores"]["parsed"]]
    if s_words:
        print(f"  situation_S words: median={sorted(s_words)[len(s_words)//2]}, "
              f"min={min(s_words)}, max={max(s_words)}")

    # Flags
    flagged = sum(1 for r in results if r["scores"]["parsed"]
                  and any(k.endswith("_flag") for k in r["scores"]))
    if flagged:
        print(f"  Flagged (too short/long): {flagged}/{parsed_ok}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input", type=Path,
        default=Path(__file__).resolve().parent.parent
        / "se-data" / "math-processed" / "moist-prompts"
        / "stage6-reverse-morphogenesis.jsonl",
        help="JSONL prompt file",
    )
    ap.add_argument(
        "--output", type=Path, default=None,
        help="Output JSONL (default: data/stage6-validation/<backend>-<variant>-<timestamp>.jsonl)",
    )
    ap.add_argument("--backend", choices=list(BACKENDS.keys()), default="claude")
    ap.add_argument("--limit", type=int, default=10, help="Number of prompts to sample")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    ap.add_argument(
        "--entity-ids", type=str, default=None,
        help="Comma-separated entity IDs to test (overrides --limit)",
    )
    ap.add_argument(
        "--variant", choices=["original", "tightened", "both"], default="both",
        help="Which prompt variant(s) to test",
    )
    ap.add_argument("--claude-model", default="claude-sonnet-4-20250514")
    ap.add_argument("--codex-model", default="gpt-5.3-codex")
    ap.add_argument("--vllm-url", default="http://localhost:8000")
    ap.add_argument("--vllm-model", default="meta-llama/Llama-3-8B-Instruct")
    ap.add_argument(
        "--with-schema", action="store_true",
        help="Pass JSON schema to codex backend (tests constrained decoding)",
    )
    args = ap.parse_args()

    # --- Load and sample prompts ---
    print(f"Loading prompts from {args.input}...")
    records = []
    target_ids = None
    if args.entity_ids:
        target_ids = set(args.entity_ids.split(","))

    with args.input.open("r", encoding="utf-8") as f:
        if target_ids:
            for line in f:
                rec = json.loads(line)
                if rec.get("entity_id") in target_ids:
                    records.append(rec)
                    target_ids.discard(rec["entity_id"])
                    if not target_ids:
                        break
            if target_ids:
                print(f"Warning: entity IDs not found: {target_ids}", file=sys.stderr)
        else:
            # Reservoir sample
            rng = random.Random(args.seed)
            seen = 0
            for line in f:
                rec = json.loads(line)
                seen += 1
                if len(records) < args.limit:
                    records.append(rec)
                else:
                    j = rng.randrange(seen)
                    if j < args.limit:
                        records[j] = rec

    print(f"Selected {len(records)} prompts.")

    if not records:
        return 0

    # --- Set up output ---
    ts = time.strftime("%Y%m%d-%H%M%S")
    if args.output is None:
        outdir = Path(__file__).resolve().parent.parent / "data" / "stage6-validation"
        outdir.mkdir(parents=True, exist_ok=True)
        args.output = outdir / f"{args.backend}-{args.variant}-{ts}.jsonl"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # --- Prepare schema file for codex ---
    schema_path = None
    if args.with_schema and args.backend == "codex":
        sf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(RESPONSE_SCHEMA, sf, indent=2)
        sf.close()
        schema_path = Path(sf.name)

    # --- Determine variants ---
    if args.variant == "both":
        variants = ["original", "tightened"]
    else:
        variants = [args.variant]

    # --- Run ---
    all_results: dict[str, list[dict]] = {v: [] for v in variants}

    for i, rec in enumerate(records):
        entity_id = rec.get("entity_id", f"idx-{i}")
        base_prompt = rec["prompt"]

        for variant_name in variants:
            prompt = PROMPT_VARIANTS[variant_name](base_prompt)

            print(f"[{i+1}/{len(records)}] {entity_id} variant={variant_name} ...",
                  end=" ", flush=True)

            t0 = time.time()
            if args.backend == "claude":
                raw = run_claude(prompt, model=args.claude_model)
            elif args.backend == "codex":
                raw = run_codex(prompt, model=args.codex_model,
                               schema_path=schema_path)
            elif args.backend == "vllm":
                raw = run_vllm(prompt, model=args.vllm_model,
                              base_url=args.vllm_url)
            else:
                raw = run_dry(prompt)
            elapsed = time.time() - t0

            # Parse
            if raw and not raw.startswith("["):
                parsed = _parse_json_object_response(raw)
            else:
                parsed = {"raw": raw, "parse_error": "empty or error response"}

            scores = score_result(parsed)

            status = "OK" if scores["parsed"] else f"FAIL({scores.get('error_type', '?')})"
            print(f"{status} ({elapsed:.1f}s)")

            result = {
                "entity_id": entity_id,
                "question_id": rec.get("question_id"),
                "variant": variant_name,
                "backend": args.backend,
                "parsed": scores["parsed"],
                "analysis": parsed if scores["parsed"] else None,
                "raw": raw[:2000] if not scores["parsed"] else None,
                "scores": scores,
                "elapsed_s": round(elapsed, 2),
            }
            all_results[variant_name].append(result)

    # --- Clean up ---
    if schema_path:
        schema_path.unlink(missing_ok=True)

    # --- Write results ---
    with args.output.open("w", encoding="utf-8") as fout:
        for variant_name in variants:
            for r in all_results[variant_name]:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nResults written to {args.output}")

    # --- Print summaries ---
    for variant_name in variants:
        print_summary(all_results[variant_name], variant_name)

    # --- Side-by-side comparison if both variants ---
    if len(variants) == 2 and len(records) > 0:
        print(f"\n{'=' * 60}")
        print("  HEAD-TO-HEAD COMPARISON")
        print(f"{'=' * 60}")
        v1, v2 = variants
        r1 = {r["entity_id"]: r for r in all_results[v1]}
        r2 = {r["entity_id"]: r for r in all_results[v2]}
        wins = {v1: 0, v2: 0, "tie": 0}
        for eid in r1:
            s1 = r1[eid]["scores"]
            s2 = r2.get(eid, {}).get("scores", {})
            p1, p2 = s1.get("parsed", False), s2.get("parsed", False)
            c1 = s1.get("keys_present", 0) if p1 else 0
            c2 = s2.get("keys_present", 0) if p2 else 0
            if c1 > c2:
                wins[v1] += 1
            elif c2 > c1:
                wins[v2] += 1
            else:
                wins["tie"] += 1
        for k, v in wins.items():
            print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
