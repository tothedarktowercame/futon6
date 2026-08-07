#!/usr/bin/env python3
"""CAS-SEL-3 topology extractor.

Given already-segmented proof steps, retrieve candidate math-informal patterns,
verify one match per step with either a deterministic oracle stub or an OpenAI-
compatible endpoint, then assemble the proof topology, wiring, residual sorries,
induce queue, and static check menu.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
FUTON3 = Path(os.environ.get("FUTON3_ROOT", "/home/joe/code/futon3"))
DEFAULT_INDEX = FUTON3 / "resources" / "sigils" / "patterns-index.tsv"
DEFAULT_LIBRARY = FUTON3 / "library" / "math-informal"
DEFAULT_FIXTURES = REPO / "tests" / "fixtures" / "cas-select"

CHECK_MENU = {
    "construct-an-explicit-witness": [],
    "construct-auxiliary-object": [],
    "count-over-a-decomposition": ["decomposition-exhaustive"],
    "epsilon-of-room": ["forall-eps-structure"],
    "estimate-by-bounding": [],
    "induction-and-well-ordering": ["R2c-warrant"],
    "local-to-global": ["R2b-closure"],
    "quotient-by-irrelevance": ["well-defined-on-quotient"],
    "reduce-to-known-result": ["R2c-warrant"],
    "separate-into-independent-pieces": ["R2b-disjointness"],
    "split-into-cases": ["cases-exhaustive"],
    "unfold-the-definition": [],
}

SYSTEM = (
    "You classify one informal proof step against a short list of reasoning patterns. "
    "Return JSON only: {\"pattern\":\"name\"|null,\"slot\":string|null,\"confidence\":0..1}."
)


@dataclass(frozen=True)
class Pattern:
    name: str
    title: str
    hotwords: tuple[str, ...]
    conclusion: str
    however: str


def norm_pattern_name(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("math-informal/"):
        return raw.split("/", 1)[1]
    if "/" in raw:
        return raw.rsplit("/", 1)[1]
    return raw


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9-]*", text.lower()))


def read_index(path: Path = DEFAULT_INDEX) -> dict[str, tuple[str, ...]]:
    rows: dict[str, tuple[str, ...]] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 5:
            continue
        name = norm_pattern_name(cols[0])
        if not cols[0].startswith("math-informal/"):
            continue
        rows[name] = tuple(w.strip().lower() for w in cols[4].split(",") if w.strip())
    return rows


def _field_block(text: str, marker: str) -> str:
    pat = re.compile(rf"^\s*\+\s+{re.escape(marker)}:\s*$", re.M)
    m = pat.search(text)
    if not m:
        return ""
    rest = text[m.end() :].splitlines()
    out: list[str] = []
    for line in rest:
        if re.match(r"^\s*\+\s+[A-Z][A-Z-]*:", line):
            break
        if re.match(r"^\s*next\[", line):
            break
        out.append(line.strip())
    return " ".join(x for x in out if x).strip()


def parse_flexiarg(path: Path, hotwords: tuple[str, ...]) -> Pattern:
    text = path.read_text()
    name = path.stem
    title = name
    keyword_words: list[str] = []
    for line in text.splitlines():
        if line.startswith("@flexiarg"):
            name = norm_pattern_name(line.split(None, 1)[1])
        elif line.startswith("@title"):
            title = line.split(None, 1)[1].strip()
        elif line.startswith("@keywords"):
            keyword_words.extend(w.strip().lower() for w in line.split(None, 1)[1].split(",") if w.strip())
    conclusion = _field_block(text, "THEN") or _field_block(text, "conclusion")
    however = _field_block(text, "HOWEVER")
    merged_hotwords = tuple(sorted(set(hotwords) | set(keyword_words) | tokenize(title)))
    return Pattern(name=name, title=title, hotwords=merged_hotwords, conclusion=conclusion, however=however)


def load_patterns(
    *,
    index_path: Path = DEFAULT_INDEX,
    library_dir: Path = DEFAULT_LIBRARY,
    allowed: set[str] | None = None,
) -> dict[str, Pattern]:
    index = read_index(index_path)
    patterns: dict[str, Pattern] = {}
    for path in sorted(library_dir.glob("*.flexiarg")):
        name = path.stem
        if allowed is not None and name not in allowed:
            continue
        hotwords = tuple(
            sorted(set(index.get(name, ()) + tuple(tokenize(path.stem.replace("-", " ")))))
        )
        patterns[name] = parse_flexiarg(path, hotwords)
    return patterns


def retrieve(step_text: str, patterns: dict[str, Pattern], k: int = 4) -> list[dict[str, Any]]:
    """Tier 0: deterministic hotword retrieval, model-free."""
    toks = tokenize(step_text)
    scored = []
    for pattern in patterns.values():
        hot = set(pattern.hotwords) | tokenize(pattern.title)
        hits = toks & hot
        if not hits:
            continue
        score = len(hits) + (len(hits) / max(1, len(hot)))
        scored.append({"pattern": pattern.name, "score": round(score, 6), "hits": sorted(hits)})
    scored.sort(key=lambda r: (-r["score"], r["pattern"]))
    return scored[:k]


def load_steps(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_oracle(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    return {row["step"]: row for row in data["matches"]}


def call_stub(step: dict[str, Any], candidates: list[dict[str, Any]], oracle: dict[str, Any]) -> dict[str, Any]:
    row = oracle.get(step["id"])
    if not row:
        return {"pattern": None, "slot": None, "confidence": 0.0}
    candidate_names = {c["pattern"] for c in candidates}
    if row["pattern"] not in candidate_names:
        return {"pattern": None, "slot": None, "confidence": 0.0}
    return {
        "pattern": row["pattern"],
        "slot": row.get("slot"),
        "confidence": float(row.get("confidence", 1.0)),
        "tier1": "verified",
        "declares_sorry": bool(row.get("declares_sorry", True)),
    }


def build_prompt(step: dict[str, Any], candidates: list[dict[str, Any]], patterns: dict[str, Pattern]) -> str:
    rows = []
    for cand in candidates:
        p = patterns[cand["pattern"]]
        rows.append(f"- {p.name}: {p.title}. THEN: {p.conclusion[:280]}")
    return (
        f"{SYSTEM}\n\nSTEP {step['id']}:\n{step['text']}\n\n"
        "CANDIDATE PATTERNS:\n" + "\n".join(rows) + "\n\n"
        "Which candidate, if any, does the step instantiate? Include the slot fill."
    )


def call_openai(prompt: str, model: str) -> dict[str, Any]:
    import json as _json

    base = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    key = os.environ.get("OPENAI_API_KEY", "x")
    body = _json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            # The verdict is one small JSON object, but no cap was sent, so the
            # model could generate to the 65536-token context. Observed on Zone:
            # ~15 min per call and two steps exceeding the 1800s timeout, which
            # put the 98-graph pass on a ~9-day trajectory. The prompt is ~85
            # tokens; the cost was entirely unbounded decode.
            "max_tokens": int(os.environ.get("FUTON6_LLM_MAX_TOKENS", "256")),
        }
    ).encode()
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    # Hardcoded 300s assumes GPU throughput; the local CPU endpoint needs longer.
    # Same defect as H3, which fixed the other three LLM callers and missed this one.
    with urllib.request.urlopen(
            req, timeout=int(os.environ.get("FUTON6_LLM_TIMEOUT", "300"))) as r:
        txt = _json.loads(r.read())["choices"][0]["message"]["content"]
    return _parse_verdict(txt)


NO_MATCH = {"pattern": None, "slot": None, "confidence": 0.0}


def _parse_verdict(txt: str) -> dict[str, Any]:
    r"""Recover a verdict from model prose, and never raise.

    A single malformed response killed a 136-call run outright on 2026-08-07: the
    greedy `\{.*\}` grabbed a JSON object the model had written with a trailing
    comma, `json.loads` raised, and the exception unwound through `select_proof`
    to `main`, discarding every verdict computed up to that point. One bad
    generation out of a hundred-odd should cost one verdict, not the run.

    Two lessons applied: the greedy match spans from the first `{` to the LAST
    `}`, so any prose containing two objects yields garbage — prefer the last
    well-formed object. And an unparseable verdict is a legitimate outcome
    ("no match"), not an error condition.
    """
    for cand in reversed(re.findall(r"\{[^{}]*\}", txt or "", re.S)):
        for attempt in (cand, re.sub(r",\s*([}\]])", r"\1", cand)):
            try:
                v = json.loads(attempt)
            except ValueError:
                continue
            if isinstance(v, dict):
                return v
    return dict(NO_MATCH)


def verify(
    step: dict[str, Any],
    candidates: list[dict[str, Any]],
    patterns: dict[str, Pattern],
    *,
    backend: str,
    oracle: dict[str, Any] | None = None,
    model: str = "mark4-70b",
    confidence_floor: float = 0.0,
) -> dict[str, Any]:
    if backend == "stub":
        raw = call_stub(step, candidates, oracle or {})
    else:
        raw = call_openai(build_prompt(step, candidates, patterns), model)
    pattern = raw.get("pattern")
    confidence = float(raw.get("confidence", raw.get("score", 0.0)) or 0.0)
    if not pattern or pattern == "NONE" or pattern not in patterns or confidence < confidence_floor:
        return {"pattern": None, "slot": None, "confidence": confidence, "tier1": "none"}
    return {
        "pattern": pattern,
        "slot": raw.get("slot"),
        "confidence": confidence,
        "tier1": "verified",
        "declares_sorry": bool(raw.get("declares_sorry", True)),
    }


def assemble(
    paper_id: str,
    matches: list[dict[str, Any]],
    induce_queue: list[dict[str, Any]],
    patterns: dict[str, Pattern],
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    topology = [m["pattern"] for m in matches]
    wiring = [
        {"step": m["step"], "conclusion": patterns[m["pattern"]].conclusion}
        for m in matches
    ]
    sorry = [
        {
            "step": m["step"],
            "pattern": m["pattern"],
            "obligation": patterns[m["pattern"]].however,
            "kind": "declared",
        }
        for m in matches
        if m.get("declares_sorry", True)
    ]
    sorry.extend({"step": row["step"], "kind": "thin"} for row in induce_queue)
    checks = [
        {"step": m["step"], "pattern": m["pattern"], "fires": CHECK_MENU.get(m["pattern"], [])}
        for m in matches
    ]
    return {
        "paper_id": paper_id,
        "topology": topology,
        "matches": matches,
        "wiring": wiring,
        "sorry": sorry,
        "induce_queue": induce_queue,
        "checks": checks,
        "errors": errors or [],
    }


def select_proof(
    steps_doc: dict[str, Any],
    patterns: dict[str, Pattern],
    *,
    backend: str = "stub",
    oracle: dict[str, Any] | None = None,
    model: str = "mark4-70b",
    k: int = 4,
    confidence_floor: float = 0.0,
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    induce_queue: list[dict[str, Any]] = []
    for step in steps_doc["steps"]:
        candidates = retrieve(step["text"], patterns, k=k)
        if backend == "stub" and oracle:
            # PERFECT-RETRIEVAL SIMULATION (stub/tests only): inject the oracle pattern into
            # the candidate set when Tier-0 hotword retrieval misses it, so the stub tests can
            # exercise verify + assemble + the Tier-2 trigger logic IN ISOLATION from Tier-0
            # recall. This does NOT run on the openai backend. Tier-0's true recall is measured
            # separately (test_tier0_retrieval_recall_is_honest); it is ~16/22@k4, ceiling ~19/22
            # at full pool — 3 steps have zero lexical overlap and need a semantic retriever.
            oracle_pattern = oracle.get(step["id"], {}).get("pattern")
            if oracle_pattern in patterns and oracle_pattern not in {c["pattern"] for c in candidates}:
                candidates = [*candidates, {"pattern": oracle_pattern, "score": 1.0, "hits": ["oracle"]}]
        # One step's verification must never cost the run. The 2026-08-07 abort
        # unwound a single bad generation all the way to main() and discarded
        # every verdict already computed; a timeout or a 503 from the endpoint
        # would have done the same. A failed step is recorded as unverified and
        # counted, so the loss is visible in the payload rather than silent.
        try:
            verdict = verify(
                step,
                candidates,
                patterns,
                backend=backend,
                oracle=oracle,
                model=model,
                confidence_floor=confidence_floor,
            )
        except Exception as e:                                  # noqa: BLE001
            errors.append({"step": step["id"], "error": f"{type(e).__name__}: {e}"})
            verdict = {"pattern": None, "slot": None, "confidence": 0.0, "tier1": "error"}
        if verdict["pattern"]:
            matches.append(
                {
                    "step": step["id"],
                    "pattern": verdict["pattern"],
                    "slot": verdict.get("slot"),
                    "score": verdict.get("confidence", 0.0),
                    "tier1": verdict.get("tier1", "verified"),
                    "declares_sorry": bool(verdict.get("declares_sorry", True)),
                }
            )
        else:
            induce_queue.append(
                {
                    "step": step["id"],
                    "candidates": [c["pattern"] for c in candidates],
                    "reason": "no candidate verified",
                }
            )
    return assemble(steps_doc["paper_id"], matches, induce_queue, patterns, errors)


def evaluate(results: dict[str, dict[str, Any]], fixture_dir: Path = DEFAULT_FIXTURES) -> dict[str, Any]:
    total = 0
    correct = 0
    per_proof = {}
    for paper_id, result in sorted(results.items()):
        oracle = load_oracle(fixture_dir / f"{paper_id}.oracle.json")
        expected = {step: row["pattern"] for step, row in oracle.items()}
        actual = {m["step"]: m["pattern"] for m in result["matches"]}
        proof_total = len(expected)
        proof_correct = sum(1 for step, pattern in expected.items() if actual.get(step) == pattern)
        total += proof_total
        correct += proof_correct
        per_proof[paper_id] = {"correct": proof_correct, "total": proof_total, "rate": proof_correct / proof_total}
    return {"correct": correct, "total": total, "rate": correct / total if total else 1.0, "per_proof": per_proof}


def run_fixture_dir(args: argparse.Namespace) -> dict[str, Any]:
    allowed = None
    if args.exclude_patterns:
        excluded = set(args.exclude_patterns.split(","))
        all_names = {p.stem for p in DEFAULT_LIBRARY.glob("*.flexiarg")}
        allowed = all_names - excluded
    patterns = load_patterns(index_path=args.index, library_dir=args.library, allowed=allowed)
    results = {}
    for steps_path in sorted(args.fixtures.glob("*.steps.json")):
        steps_doc = load_steps(steps_path)
        oracle = load_oracle(args.fixtures / f"{steps_doc['paper_id']}.oracle.json")
        results[steps_doc["paper_id"]] = select_proof(
            steps_doc,
            patterns,
            backend=args.backend,
            oracle=oracle if args.backend == "stub" else None,
            model=args.model,
            k=args.k,
            confidence_floor=args.confidence_floor,
        )
    return {"results": results, "evaluation": evaluate(results, args.fixtures)}


def _allowed_patterns(args: argparse.Namespace) -> set[str] | None:
    if not args.exclude_patterns:
        return None
    excluded = set(args.exclude_patterns.split(","))
    all_names = {p.stem for p in args.library.glob("*.flexiarg")}
    return all_names - excluded


def run_steps_paths(args: argparse.Namespace, paths: list[Path]) -> dict[str, Any]:
    """Select over many papers, writing each one's result as it lands.

    Without `--checkpoint` this is a wager on the last call succeeding: the
    payload goes to stdout only at the end, so an abort at paper 97 of 98
    discards 96 papers of completed LLM work. That is what happened on
    2026-08-07. It also means a long run gives no progress signal at all —
    an empty output file looks identical whether the run is healthy, wedged,
    or dead, which is not a distinction to leave to guesswork on a booked
    window.

    With `--checkpoint`, each paper is appended as one JSON line as soon as it
    completes, and a restart skips the papers already present. Resume is by
    paper id rather than by position, so it survives the input set changing.
    """
    patterns = load_patterns(index_path=args.index, library_dir=args.library, allowed=_allowed_patterns(args))
    results: dict[str, Any] = {}
    ckpt: Path | None = getattr(args, "checkpoint", None)

    if ckpt and ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue                      # a torn final line from a hard kill
            if isinstance(row, dict) and "paper_id" in row:
                results[row["paper_id"]] = row
        if results:
            print(f"resuming: {len(results)} paper(s) already in {ckpt}", file=sys.stderr)

    todo = [q for q in sorted(paths) if load_steps(q)["paper_id"] not in results]
    for i, steps_path in enumerate(todo, 1):
        steps_doc = load_steps(steps_path)
        row = select_proof(
            steps_doc,
            patterns,
            backend=args.backend,
            oracle=None,
            model=args.model,
            k=args.k,
            confidence_floor=args.confidence_floor,
        )
        results[steps_doc["paper_id"]] = row
        if ckpt:
            with ckpt.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                os.fsync(fh.fileno())         # survive a kill, not just an exit
        print(f"  [{i}/{len(todo)}] {steps_doc['paper_id']}: "
              f"{len(row.get('matches', []))} match(es), {len(row.get('errors', []))} error(s)",
              file=sys.stderr, flush=True)
    return {"results": results}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    ap.add_argument("--steps", type=Path, help="Run one produced .steps.json file")
    ap.add_argument("--steps-dir", type=Path, help="Run all *.steps.json files in a produced steps directory")
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    ap.add_argument("--backend", choices=["stub", "openai"], default="stub")
    ap.add_argument("--model", default="mark4-70b")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--confidence-floor", type=float, default=0.0)
    ap.add_argument("--checkpoint", type=Path,
                    help="append each paper's result here as JSON lines; resume skips those already present")
    ap.add_argument("--exclude-patterns", default="")
    args = ap.parse_args(argv)
    if args.steps and args.steps_dir:
        ap.error("--steps and --steps-dir are mutually exclusive")
    if args.steps:
        payload = run_steps_paths(args, [args.steps])
    elif args.steps_dir:
        payload = run_steps_paths(args, list(args.steps_dir.glob("*.steps.json")))
    else:
        payload = run_fixture_dir(args)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
