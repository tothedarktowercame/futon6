#!/usr/bin/env python3
"""mark3 expository model loop.

Per expository candidate: build a vocab-grounded prompt, call a stub/openai
backend, extract EDN, self-gate with expository_argcheck.bb, retry on failure,
and emit only gated graphs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
ARGCHECK = REPO / "scripts" / "expository_argcheck.bb"
VOCAB = REPO / "holes" / "excursions" / "expository-superpod-vocab.edn"
ALIGNMENT = REPO / "holes" / "excursions" / "E-iatc-expository-alignment.md"
CANDIDATE_SCHEMA = "expo-candidate/v1"
MAX_ATTEMPTS = 3

SYSTEM = """You classify and fill arXiv expository-region scopes as EDN.
Rules:
- Output exactly one EDN map.
- Use only :kind values from expository-superpod-vocab.edn :scopes.
- Do not use any :out-of-scope-arxiv kind.
- Every scope has :source {:lines [a b]} inside the candidate window.
- Every scope has either :slot-fill with source-anchored text or :held {:reason "..."}.
- This is the informal expository layer, not the formal IATC proof-DAG layer."""


def extract_edn(response: str) -> str | None:
    if "```" in response:
        segment = response.split("```", 2)
        if len(segment) >= 2:
            response = segment[1].split("\n", 1)[-1] if segment[1].lower().startswith("edn") else segment[1]
    start = response.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(response)):
        ch = response[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return response[start : idx + 1]
    return None


def vocab_excerpt() -> str:
    return VOCAB.read_text(encoding="utf-8")


def exemplar_excerpt() -> str:
    text = ALIGNMENT.read_text(encoding="utf-8")
    start = text.find("## 3b.")
    end = text.find("## 4.", start)
    if start < 0:
        return ""
    return text[start : end if end > start else start + 3500]


def render_enrichment(candidate: dict[str, Any]) -> str:
    rows = candidate.get("enrichment") or []
    if not rows:
        return "(no deterministic anatomy detected in this region)"
    return "\n".join(f"L{r['line']} ({r['kind']}) {r['tip']}" for r in rows)


def build_prompt(candidate: dict[str, Any]) -> str:
    return f"""{SYSTEM}

# Finalized vocabulary
{vocab_excerpt()}

# Exemplar bank excerpt
{exemplar_excerpt()}

# Candidate
paper-id: {candidate['paper-id']}
passage-id: {candidate['passage-id']}
window-lines: {candidate['window-lines']}
region-type: {candidate.get('region-type')}

deterministic anatomy in this region:
{render_enrichment(candidate)}

source-window:
{candidate['source-window']}

EDN graph:"""


def first_vocab_kind() -> str:
    match = re.search(r":kind\s+(:[^\s\]}]+)", VOCAB.read_text(encoding="utf-8"))
    return match.group(1) if match else ":rationale/telos"


def edn_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def call_stub(prompt: str, candidate: dict[str, Any], attempt: int) -> str:
    lo, hi = candidate["window-lines"]
    kind = first_vocab_kind()
    snippet = " ".join(str(candidate.get("source-window", "")).split())[:120]
    if not snippet:
        snippet = "source text not recoverable"
    return (
        '{:paper/id "' + candidate["paper-id"] + '"\n'
        ' :passage/id "' + candidate["passage-id"] + '"\n'
        f" :source {{:lines [{lo} {hi}] :kind :expository}}\n"
        f" :scopes [{{:id :s1 :kind {kind}\n"
        f"           :slot-fill {edn_string(snippet)}\n"
        f"           :source {{:lines [{lo} {hi}]}}}}]}}\n"
    )


def call_openai(prompt: str, candidate: dict[str, Any], attempt: int, model: str) -> str:
    import urllib.request

    base = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    key = os.environ.get("OPENAI_API_KEY", "x")
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2 if attempt == 0 else 0.5,
            "max_tokens": 2048,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.loads(response.read())["choices"][0]["message"]["content"]


def gate_one(path: Path) -> tuple[bool, str]:
    check = subprocess.run(["bb", str(ARGCHECK), str(path)], capture_output=True, text=True)
    if check.returncode != 0:
        return False, (check.stdout + check.stderr).strip()[-800:]
    return True, "ok"


def candidate_check(edn: str, candidate: dict[str, Any]) -> tuple[bool, str]:
    paper_match = re.search(r':paper/id\s+"([^"]+)"', edn)
    if paper_match and paper_match.group(1) != candidate["paper-id"]:
        return False, f"faithfulness: :paper/id {paper_match.group(1)!r} != candidate {candidate['paper-id']!r}"
    lo, hi = candidate["window-lines"]
    outside = []
    for a, b in re.findall(r":lines\s*\[\s*(\d+)\s+(\d+)\s*\]", edn):
        a_i, b_i = int(a), int(b)
        if a_i < lo or b_i > hi:
            outside.append([a_i, b_i])
    if outside:
        return False, f"faithfulness: source span outside candidate window [{lo} {hi}], e.g. {outside[0]}"
    return True, "ok"


def require_enriched(candidate_paths: list[Path]) -> bool:
    stale = []
    for path in candidate_paths:
        try:
            candidate = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            stale.append((path.name, f"unreadable: {exc}"))
            continue
        vocab_path = candidate.get("vocab-path")
        vocab_ok = isinstance(vocab_path, str) and (REPO / vocab_path).exists()
        if candidate.get("schema") != CANDIDATE_SCHEMA or not vocab_ok:
            stale.append(
                (
                    path.name,
                    f"schema={candidate.get('schema')!r}, vocab-path={vocab_path!r}, vocab-ok={vocab_ok}",
                )
            )
    if stale:
        print(
            f"FATAL: {len(stale)}/{len(candidate_paths)} candidate(s) fail the expository precondition. "
            "Refusing to call the model stage.",
            file=sys.stderr,
        )
        for name, why in stale[:10]:
            print(f"  - {name}: {why}", file=sys.stderr)
        print(
            "Expected schema 'expo-candidate/v1' and a repo-local vocab-path. "
            "Re-extract: python3 scripts/mark3_extract_expository_candidates.py --out <candidates-dir>",
            file=sys.stderr,
        )
        return False
    return True


def safe_output_name(candidate: dict[str, Any]) -> str:
    passage = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate["passage-id"])
    return f"{passage}.edn"


def run(args: argparse.Namespace) -> int:
    candidate_paths = sorted(Path(args.candidates).glob("*.candidate.json"))
    if not candidate_paths:
        print("no candidates found", file=sys.stderr)
        return 2
    if not require_enriched(candidate_paths):
        return 2

    outdir = Path(args.out)
    attempts = outdir / ".attempts"
    outdir.mkdir(parents=True, exist_ok=True)
    attempts.mkdir(exist_ok=True)
    results = []
    bypaper = {}  # paper-id -> [total, passed], for the S4 expository-coverage emit
    for candidate_path in candidate_paths:
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        pid = candidate.get("paper-id") or str(candidate["passage-id"]).split(":")[0]
        prompt = build_prompt(candidate)
        status, last_error = "fail", ""
        for attempt in range(MAX_ATTEMPTS):
            attempt_prompt = (
                prompt
                if attempt == 0
                else prompt + f"\n\n# previous attempt failed:\n{last_error}\n# fix it and emit only EDN."
            )
            response = (
                call_stub(attempt_prompt, candidate, attempt)
                if args.backend == "stub"
                else call_openai(attempt_prompt, candidate, attempt, args.model)
            )
            edn = extract_edn(response)
            if not edn:
                last_error = "no EDN map found in response"
                continue
            attempt_path = attempts / f"{candidate_path.stem}.attempt{attempt}.edn"
            attempt_path.write_text(edn, encoding="utf-8")
            ok, err = candidate_check(edn, candidate)
            if ok:
                ok, err = gate_one(attempt_path)
            if ok:
                (outdir / safe_output_name(candidate)).write_text(edn, encoding="utf-8")
                status, last_error = "pass", f"attempt {attempt}"
                break
            last_error = err
        results.append((candidate["passage-id"], status, last_error))
        rec = bypaper.setdefault(pid, [0, 0])
        rec[0] += 1
        rec[1] += 1 if status == "pass" else 0
        print(f"  {candidate['passage-id']}: {status} ({last_error[:100]})")

    passed = sum(1 for _, status, _ in results if status == "pass")
    print(f"\nexpository-loop: {passed}/{len(results)} graphs gated PASS")
    if getattr(args, "run_dir", None):  # S4 inline metric: per-paper expository-coverage
        try:
            import sys as _sys
            _sys.path.insert(0, str(REPO / "scripts"))
            import metric_harness as mh
            for pid, (tot, ok) in bypaper.items():
                mh.emit_record(args.run_dir, run_id=args.run_id, corpus_id=args.corpus_id,
                               paper_id=pid, stage="S4", metric="expository-coverage",
                               axis="completeness", value=round(ok / max(1, tot), 4), computable=True)
        except Exception as ee:
            print(f"  (S4 metric emit skipped: {ee})")
    return 0 if passed == len(results) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default=str(REPO / "data" / "expository-candidates"))
    parser.add_argument("--out", default=str(REPO / "data" / "expository-scope-graphs" / "loop-run"))
    parser.add_argument("--backend", choices=["stub", "openai"], default="stub")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--run-dir", help="if set, emit S4 expository-coverage MetricRecords here")
    parser.add_argument("--run-id", default="adhoc")
    parser.add_argument("--corpus-id", default="adhoc")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
