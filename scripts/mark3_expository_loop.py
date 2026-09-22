#!/usr/bin/env python3
"""S4 expository model loop: one schema-constrained model call per selected region.

Per expository candidate: prompt with the finalized vocabulary and the numbered
region -> the model returns JSON under the expository_json schema -> code checks
what the schema cannot and writes the EDN scope graph -> expository_argcheck.bb ->
accept. The model never writes EDN; nothing is repaired or retried to fix a format.
Each item gets one call per stage invocation at temperature 0; re-invoking a failed
stage retries only the items that were not accepted.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import expository_json  # noqa: E402
import run_contract  # noqa: E402
import stage_accounting as accounting  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ARGCHECK = REPO / "scripts" / "expository_argcheck.bb"
VOCAB = expository_json.VOCAB
ALIGNMENT = REPO / "holes" / "excursions" / "E-iatc-expository-alignment.md"
CANDIDATE_SCHEMA = "expo-candidate/v2"
MAX_TOKENS = int(os.environ.get("FUTON6_EXPOSITORY_MAX_TOKENS", "2048"))

SYSTEM = """You classify one expository region of a published mathematics paper.

Return JSON with a list "scopes". Each scope is one thing the prose is doing, typed
by a "kind" from the vocabulary below (its :hole says what to fill). For each scope:
- "units": the ids of the sentence units this scope reads, from the list below. An id
  names the unit's line; there is no order to follow, so read the unit before citing it;
- "fill": the words from THOSE UNITS that fill the kind's hole. Copy them from the
  source; a fill that is not in the units it cites is rejected, OR
- "held_reason": why the hole cannot be filled from this text (then "fill" is "").
Exactly one of "fill" and "held_reason" is non-empty. Hold rather than invent.
This is the informal expository layer, not the formal proof layer."""


def exemplar_excerpt() -> str:
    text = ALIGNMENT.read_text(encoding="utf-8")
    start = text.find("## 3b.")
    end = text.find("## 4.", start)
    if start < 0:
        return ""
    return text[start: end if end > start else start + 3500]


def render_enrichment(candidate: dict[str, Any]) -> str:
    rows = candidate.get("enrichment") or []
    if not rows:
        return "(no deterministic anatomy detected in this region)"
    return "\n".join(f"L{r['line']} ({r['kind']}) {r['tip']}" for r in rows)


def numbered_window(candidate: dict[str, Any]) -> str:
    """Source with ABSOLUTE line numbers, so anchors are read, not counted (H21)."""
    lo = (candidate.get("window-lines") or [1, 1])[0]
    body = str(candidate.get("source-window", ""))
    return "\n".join(f"{lo + i:5d} | {ln}" for i, ln in enumerate(body.split("\n")))


def render_units(candidate: dict[str, Any]) -> str:
    """The region's sentence units, under the ids "units" takes."""
    units = candidate.get("units") or []
    if not units:
        return "(this candidate carries no units)"
    return "\n".join(f"  {u['id']}  {' '.join(str(u['text']).split())}" for u in units)


def build_prompt(candidate: dict[str, Any]) -> str:
    lo, hi = candidate["window-lines"]
    return f"""{SYSTEM}

# Vocabulary (use only the :scopes kinds; the :out-of-scope-arxiv kinds are excluded)
{VOCAB.read_text(encoding="utf-8")}

# What these categories look like
{exemplar_excerpt()}

# Region ({candidate.get('region-type')}), lines {lo}-{hi}
Deterministic anatomy in this region:
{render_enrichment(candidate)}

Units of this region. "units" takes ids from THIS list, and a fill must be words
from the units it cites:
{render_units(candidate)}

Source (ABSOLUTE line numbers on the left):
{numbered_window(candidate)}"""


class ModelCallError(Exception):
    """The endpoint could not produce a judgeable answer (HTTP error, truncation)."""


def call_stub(prompt: str, candidate: dict[str, Any], kinds: dict[str, str]) -> str:
    lo, hi = candidate["window-lines"]
    units = candidate.get("units") or []
    if units:
        # Quote the unit, as the contract requires of a real answer.
        first = units[0]
        return json.dumps({"scopes": [{"kind": sorted(kinds)[0], "units": [first["id"]],
                                       "fill": " ".join(str(first["text"]).split())[:120],
                                       "held_reason": ""}]})
    snippet = " ".join(str(candidate.get("source-window", "")).split())[:120] or "source text"
    return json.dumps({"scopes": [{"kind": sorted(kinds)[0], "first_line": lo, "last_line": hi,
                                   "fill": snippet, "held_reason": ""}]})


def call_openai(prompt: str, candidate: dict[str, Any], kinds: dict[str, str], model: str) -> str:
    import urllib.error
    import urllib.request
    base = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    key = os.environ.get("OPENAI_API_KEY", "x")
    lo, hi = candidate["window-lines"]
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "expository_region", "strict": True, "schema": expository_json.schema(lo, hi, kinds, candidate.get("units") or ())}},
    }).encode()
    req = urllib.request.Request(f"{base}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=int(os.environ.get("FUTON6_LLM_TIMEOUT", "300"))) as response:
            choice = json.loads(response.read())["choices"][0]
    except urllib.error.HTTPError as e:
        raise ModelCallError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")
    except urllib.error.URLError as e:
        raise ModelCallError(str(e.reason))
    if choice.get("finish_reason") == "length":
        raise ModelCallError(f"output truncated at max_tokens={MAX_TOKENS}")
    return choice["message"]["content"]


def gate_one(path: Path) -> tuple[bool, str]:
    check = subprocess.run(["bb", str(ARGCHECK), str(path)], capture_output=True, text=True)
    if check.returncode != 0:
        # The informative part, not the file path the gate echoes first (H18).
        lines = [ln.strip() for ln in (check.stdout + check.stderr).splitlines()
                 if ln.strip() and not ln.strip().startswith(("FAIL ", "expository-argcheck:"))]
        return False, " | ".join(lines)[-800:]
    return True, "ok"


def require_candidates(candidate_paths: list[Path]) -> bool:
    stale = []
    for path in candidate_paths:
        try:
            candidate = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            stale.append((path.name, f"unreadable: {exc}"))
            continue
        vocab_path = candidate.get("vocab-path")
        vocab_ok = isinstance(vocab_path, str) and (REPO / vocab_path).exists()
        missing = run_contract.missing_expository_inputs(candidate)
        if candidate.get("schema") != CANDIDATE_SCHEMA or not vocab_ok or missing:
            why = f"schema={candidate.get('schema')!r}, vocab-path={vocab_path!r}"
            if missing:
                why += f", lacks {', '.join(missing)} required by {run_contract.contract_id()}"
            stale.append((path.name, why))
    if stale:
        print(f"FATAL: {len(stale)}/{len(candidate_paths)} candidate(s) fail the expository precondition "
              f"({CANDIDATE_SCHEMA} with a repo-local vocab-path). Re-extract: "
              "python3 scripts/mark3_extract_expository_candidates.py --out <candidates-dir>", file=sys.stderr)
        for name, why in stale[:10]:
            print(f"  - {name}: {why}", file=sys.stderr)
        return False
    return True


def safe_output_name(candidate: dict[str, Any]) -> str:
    passage = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate["passage-id"])
    return f"{passage}.edn"


def attempt_one(candidate, args, kinds, attempts: Path) -> tuple[str, str, dict]:
    lo, hi = candidate["window-lines"]
    stem = safe_output_name(candidate)[:-len(".edn")]
    record: dict = {"attempt": 0}
    try:
        raw = (call_stub(build_prompt(candidate), candidate, kinds) if args.backend == "stub"
               else call_openai(build_prompt(candidate), candidate, kinds, args.model))
    except ModelCallError as e:
        record["result"] = str(e)[:300]
        return "errored", str(e), record
    response = attempts / f"{stem}.response.json"
    response.write_text(raw, encoding="utf-8")
    record["response"] = accounting.relative(response)
    try:
        doc = json.loads(raw)
    except ValueError as e:
        why = f"endpoint returned non-JSON despite the schema ({e}); check serving conformance"
        record["result"] = why
        return "errored", why, record
    found = expository_json.problems(doc, lo, hi, kinds, candidate.get("units") or ())
    if found:
        why = "contract: " + "; ".join(found[:6])
        record["result"] = why[:500]
        return "rejected", why, record
    graph = attempts / f"{stem}.edn"
    graph.write_text(expository_json.to_edn(doc, candidate, kinds, args.model), encoding="utf-8")
    record["graph"] = accounting.relative(graph)
    ok, why = gate_one(graph)
    if not ok:
        record["result"] = why[:500]
        return "rejected", why, record
    record["result"] = "accepted"
    return "accepted", "", record


def run(args: argparse.Namespace) -> int:
    candidate_paths = sorted(Path(args.candidates).glob("*.candidate.json"))
    if not candidate_paths:
        print("no candidates found", file=sys.stderr)
        return 2
    if not require_candidates(candidate_paths):
        return 2
    kinds = expository_json.vocabulary()
    outdir = Path(args.out)
    # Attempts are scoped by run and invocation (H37): a retry adds to the history.
    invocation = os.environ.get(accounting.INVOCATION_ENV) or "standalone"
    run_tag = os.environ.get("RUN_ID") or getattr(args, "run_id", None) or "unscoped"
    attempts = outdir / ".attempts" / run_tag / invocation
    if invocation != "standalone" and attempts.exists():
        print(f"attempt history already exists for invocation {invocation}", file=sys.stderr)
        return 2
    attempts.mkdir(parents=True, exist_ok=True)
    loaded = [json.loads(p.read_text(encoding="utf-8")) for p in candidate_paths]
    ledger = accounting.Accounting("S4", "loop", [c["passage-id"] for c in loaded])
    counts = {"accepted": 0, "rejected": 0, "errored": 0, "carried": 0}
    bypaper = {}  # paper-id -> [total, accepted], for the S4 expository-coverage emit
    for candidate in loaded:
        pid = candidate.get("paper-id") or str(candidate["passage-id"]).split(":")[0]
        item = candidate["passage-id"]
        final = outdir / safe_output_name(candidate)
        rec = bypaper.setdefault(pid, [0, 0])
        rec[0] += 1
        if final.exists():
            # Keep an earlier acceptance from this run rather than resampling it;
            # an unexplained final is stale output and cannot count as accepted.
            carried, why = accounting.carried_acceptance(outdir, item, final)
            if carried is None:
                counts["errored"] += 1
                ledger.record(item, "errored", why, paper=pid, artifacts=[accounting.relative(final)])
                print(f"  {item}: ERROR ({why})")
                continue
            counts["accepted"] += 1
            counts["carried"] += 1
            rec[1] += 1
            ledger.record(item, "accepted", paper=pid, outputs=[item], artifacts=[accounting.relative(final)],
                          attempts=[{"carried-from": carried.get("invocation"), "path": carried.get("path")}])
            print(f"  {item}: accepted (carried from {carried.get('invocation')})")
            continue
        status, why, record = attempt_one(candidate, args, kinds, attempts)
        counts[status] += 1
        if status == "accepted":
            accounting.publish_accepted(outdir, item, final, (attempts / final.name).read_bytes(),
                                        {"path": record["graph"]})
            rec[1] += 1
            ledger.record(item, "accepted", paper=pid, outputs=[item], attempts=[record],
                          artifacts=[accounting.relative(final)])
        else:
            ledger.record(item, status, why, paper=pid, attempts=[record])
        print(f"  {item}: {status}" + (f" ({why[:160]})" if why else ""))

    print(f"\nexpository-loop: accepted {counts['accepted']} (carried {counts['carried']}) · "
          f"rejected {counts['rejected']} · errored {counts['errored']} of {len(loaded)}")
    if getattr(args, "run_dir", None):
        import metric_harness as mh
        # This is accepted/SELECTED, and the selector caps each paper long before
        # the loop sees a region. In the 0919b run that made every one of the
        # twelve records 1.0 while 1,859 of 2,139 discovered regions had been
        # deferred: a metric named "coverage" on axis "completeness" that could
        # not report incompleteness, because the deferred regions were removed
        # from its denominator upstream. It is an acceptance rate; it now says so.
        regions = Path(args.candidates) / "regions"
        for pid, (tot, ok) in bypaper.items():
            mh.emit_record(args.run_dir, run_id=args.run_id, corpus_id=args.corpus_id,
                           paper_id=pid, stage="S4", metric="expository-acceptance",
                           axis="quality", value=round(ok / max(1, tot), 4), computable=True)
            # Real coverage needs the denominator the selector discarded. Emit it
            # only when the discovered regions are still on disk; never guess it.
            found = len(list(regions.glob(f"{pid}*.json"))) if regions.is_dir() else 0
            mh.emit_record(args.run_dir, run_id=args.run_id, corpus_id=args.corpus_id,
                           paper_id=pid, stage="S4", metric="expository-coverage",
                           axis="completeness",
                           value=round(ok / found, 4) if found else None,
                           computable=bool(found))
    # Refused regions are recorded per item and weighed by the runner against the
    # run's floor; the loop itself ran, so it exits 0. See mark3_iatc_loop.py.
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default=str(REPO / "data" / "expository-candidates"))
    parser.add_argument("--out", default=str(REPO / "data" / "expository-scope-graphs" / "loop-run"))
    parser.add_argument("--backend", choices=["stub", "openai"], default="stub")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--run-dir", help="if set, emit S4 expository-coverage MetricRecords here")
    parser.add_argument("--run-id", default="adhoc")
    parser.add_argument("--corpus-id", default="adhoc")
    args = parser.parse_args()
    if args.run_dir and "adhoc" in (args.run_id, args.corpus_id):
        parser.error("--run-dir requires explicit --run-id and --corpus-id (records would be tagged adhoc)")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
