#!/usr/bin/env python3
"""mark3 IATC model loop — the GPU/LLM half of the validated reconstruction path.

Per candidate (from mark3_extract_candidates.py):
  build few-shot prompt (seed graphs + the source window) -> call LLM ->
  parse EDN -> self-gate (iatc_argcheck.bb AND substance_gate.py) ->
  retry with the gate errors fed back -> emit on PASS. Finally re-run the
  substance gate over the whole batch (cross-item template/warrant checks).

Backends:
  --backend stub    : no GPU; returns varied seed graphs to validate the plumbing
                      (prompt build -> EDN parse -> gates -> emit/retry).
  --backend openai  : OpenAI-compatible HTTP (works against a vLLM server on the
                      Linode's GPU). Reads OPENAI_BASE_URL + OPENAI_API_KEY; --model.

This is the owner-authored harness; the Linode supplies only the model endpoint.
Gate = checker PASS + substance PASS; final acceptance is owner review.

Usage (local plumbing check):
  python scripts/mark3_iatc_loop.py --candidates data/iatc-candidates \
      --out data/iatc-argument-graphs/loop-run --backend stub
Usage (on the Linode):
  OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=x \
  python scripts/mark3_iatc_loop.py --candidates data/iatc-candidates \
      --out data/iatc-argument-graphs/loop-run --backend openai --model <hf-id>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Valid IATC argument-graph seeds only. NB: holes/golden-graphs/ is mostly the
# "anatomy"/GrCalc string-diagram format (8/9 FAIL iatc_argcheck) — wrong schema
# for few-shot here; use the checker's golden fixtures + the accepted pilot.
SEED_DIRS = [REPO / "holes" / "iatc-argcheck" / "fixtures" / "golden",
             REPO / "data" / "iatc-argument-graphs" / "gh200"]
ARGCHECK = REPO / "scripts" / "iatc_argcheck.bb"
SUBSTANCE = REPO / "scripts" / "substance_gate.py"
MAX_ATTEMPTS = 3

SYSTEM = """You reconstruct the warranted argument DAG of a single mathematical \
proof passage as an IATC graph in EDN. Rules:
- Standoff: every node/edge carries :source {:lines [a b]} into the given window. \
Do not invent line numbers outside the window.
- Recover the REAL premises, intermediate claims, objects, and conclusion of THIS \
passage — not a fixed template. Graphs vary in size with the argument.
- :nodes have :kind :object|:claim|:ref and :text (a faithful short gloss of the \
source claim). :edges are :kind :infer with :relation, :premise, :conclusion, and \
either a real :warrant {:kind :claim/:citation ...} or :warrant {:kind :missing-warrant ...}.
- A :missing-warrant's :wanted must NAME the SPECIFIC elided justification for THAT \
step (what fact/lemma/computation the prose skipped), e.g. \
:dimension-shift-through-short-exact-sequence. Never a generic bucket.
- Cited justifications ("by [3]", "according to Thm 1.4") are :warrant {:kind :citation ...}, \
not holes.
- Output ONLY the EDN map. No prose."""


def load_seeds(n: int = 3) -> str:
    out, seen = [], 0
    for d in SEED_DIRS:
        for f in sorted(d.glob("*.edn")):
            if "canon-links" in f.name:
                continue
            out.append(f"% example ({f.name})\n{f.read_text().strip()}")
            seen += 1
            if seen >= n:
                return "\n\n".join(out)
    return "\n\n".join(out)


def build_prompt(cand: dict, seeds: str) -> str:
    binders = "\n".join(cand.get("binder-context", [])) or "(none)"
    return f"""{SYSTEM}

# Few-shot examples (the target form):
{seeds}

# Now reconstruct the graph for this passage.
paper-id: {cand['paper-id']}
window-lines: {cand['window-lines']}
binder-context (variable typings established earlier):
{binders}

source-window:
{cand['source-window']}

EDN graph:"""


# --- backends ---

def call_stub(prompt: str, cand: dict, attempt: int) -> str:
    """No-GPU plumbing stub: return a real, varied seed graph (cycles by paper)."""
    seeds = []
    for d in SEED_DIRS:
        seeds += sorted(f for f in d.glob("*.edn") if "canon-links" not in f.name)
    idx = (abs(hash(cand["paper-id"])) + attempt) % len(seeds)
    return seeds[idx].read_text()


def call_openai(prompt: str, cand: dict, attempt: int, model: str) -> str:
    import urllib.request
    base = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    key = os.environ.get("OPENAI_API_KEY", "x")
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2 if attempt == 0 else 0.5,
        "max_tokens": 2048,
    }).encode()
    req = urllib.request.Request(f"{base}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def extract_edn(resp: str) -> str | None:
    if "```" in resp:
        seg = resp.split("```", 2)
        if len(seg) >= 2:
            resp = seg[1].split("\n", 1)[-1] if seg[1].lower().startswith("edn") else seg[1]
    i = resp.find("{")
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(resp)):
        if resp[j] == "{":
            depth += 1
        elif resp[j] == "}":
            depth -= 1
            if depth == 0:
                return resp[i:j + 1]
    return None


def gate_one(path: Path) -> tuple[bool, str]:
    chk = subprocess.run(["bb", str(ARGCHECK), str(path)], capture_output=True, text=True)
    if chk.returncode != 0:
        return False, "checker: " + (chk.stdout + chk.stderr).strip()[-500:]
    sub = subprocess.run([sys.executable, str(SUBSTANCE), str(path), "--kind", "iatc"],
                         capture_output=True, text=True)
    if sub.returncode != 0:
        return False, "substance: " + (sub.stdout + sub.stderr).strip()[-500:]
    return True, "ok"


def candidate_check(edn: str, cand: dict) -> tuple[bool, str]:
    """Candidate-aware faithfulness: the graph must be about THIS paper and anchor
    only into the given window. A small model hallucinates the paper/passage id and
    cites :source lines outside the window it was shown (observed 2026-06-16)."""
    pid = cand["paper-id"]
    m = re.search(r':paper/id\s+"([^"]+)"', edn)
    if m and m.group(1) != pid:
        return False, f"faithfulness: :paper/id '{m.group(1)}' != candidate '{pid}'"
    lo, hi = cand["window-lines"]
    slack = 3
    out = []
    for a, b in re.findall(r':lines\s*\[\s*(\d+)\s+(\d+)\s*\]', edn):
        a, b = int(a), int(b)
        if a < lo - slack or b > hi + slack:
            out.append([a, b])
    if out:
        return False, (f"faithfulness: {len(out)} :source span(s) outside window "
                       f"[{lo} {hi}] (±{slack}), e.g. {out[0]}")
    return True, "ok"


def run(args) -> int:
    cands = sorted(Path(args.candidates).glob("*.candidate.json"))
    if not cands:
        print("no candidates found", file=sys.stderr)
        return 2
    seeds = load_seeds(args.shots)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = outdir / ".attempts"
    tmp.mkdir(exist_ok=True)
    results = []
    for cf in cands:
        cand = json.loads(cf.read_text())
        pid = cand["paper-id"]
        prompt = build_prompt(cand, seeds)
        status, last_err = "fail", ""
        for attempt in range(MAX_ATTEMPTS):
            p = prompt if attempt == 0 else prompt + f"\n\n# previous attempt failed the gate:\n{last_err}\n# fix it and re-emit ONLY the EDN."
            if args.backend == "stub":
                resp = call_stub(p, cand, attempt)
            else:
                resp = call_openai(p, cand, attempt, args.model)
            edn = extract_edn(resp)
            if not edn:
                last_err = "no EDN map found in response"
                continue
            ap = tmp / f"{pid}.attempt{attempt}.edn"
            ap.parent.mkdir(parents=True, exist_ok=True)
            ap.write_text(edn)
            ok, err = candidate_check(edn, cand)
            if ok:
                ok, err = gate_one(ap)
            if ok:
                outdir.mkdir(parents=True, exist_ok=True)
                (outdir / f"{pid}.edn").write_text(edn)
                status, last_err = "pass", f"attempt {attempt}"
                break
            last_err = err
        results.append((pid, status, last_err))
        print(f"  {pid}: {status} ({last_err[:80]})")

    # cross-item substance gate over the accepted batch
    print("\n=== batch substance gate (cross-item) ===")
    sub = subprocess.run([sys.executable, str(SUBSTANCE), str(outdir), "--kind", "iatc"],
                         capture_output=True, text=True)
    print(sub.stdout.strip()[-400:])
    n_pass = sum(1 for _, s, _ in results if s == "pass")
    print(f"\nloop: {n_pass}/{len(results)} graphs gated PASS · batch-substance "
          f"{'PASS' if sub.returncode == 0 else 'FAIL'}")
    print("Next: OWNER REVIEW — spot-check faithfulness against source at the anchors.")
    return 0 if (n_pass == len(results) and sub.returncode == 0) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(REPO / "data" / "iatc-candidates"))
    ap.add_argument("--out", default=str(REPO / "data" / "iatc-argument-graphs" / "loop-run"))
    ap.add_argument("--backend", choices=["stub", "openai"], default="stub")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--shots", type=int, default=3)
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
