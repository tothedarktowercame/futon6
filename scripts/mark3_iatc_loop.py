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
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Valid IATC argument-graph seeds only. NB: holes/golden-graphs/ is mostly the
# "anatomy"/GrCalc string-diagram format (8/9 FAIL iatc_argcheck) — wrong schema
# for few-shot here; use the checker's golden fixtures + the accepted pilot.
SEED_DIRS = [REPO / "holes" / "iatc-argcheck" / "fixtures" / "golden",
             REPO / "data" / "iatc-argument-graphs" / "gh200"]
ARGCHECK = REPO / "scripts" / "iatc_argcheck.bb"
SUBSTANCE = REPO / "scripts" / "substance_gate.py"
SEMCHECK = REPO / "scripts" / "iatc_semcheck.bb"
REPAIR = REPO / "scripts" / "iatc_repair.bb"
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
- CRITICAL — the validator REJECTS the graph unless ALL of these hold: \
(1) EVERY :edges entry carries :source {:lines [a b]} (not only :nodes). \
(2) The map includes a top-level :holes vector, and EVERY edge whose :warrant is \
{:kind :missing-warrant :wanted X} is mirrored by a matching {:kind :missing-warrant :wanted X} \
entry in :holes. \
(3) EVERY :ref node resolves via :label/:target/:citation, or is listed in :holes.
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


def render_enrichment(cand: dict) -> str:
    rows = cand.get("enrichment") or []
    if not rows:
        return "(no deterministic anatomy detected in this window)"
    return "\n".join(f"L{r['line']} ({r['kind']}) {r['tip']}" for r in rows)


def numbered_window(cand: dict) -> str:
    """Render the source window with ABSOLUTE line numbers.

    The window used to be handed over as bare text with only its bounds stated,
    so the model had to COUNT lines to produce :source {:lines [a b]} anchors —
    and it miscounted: measured over the e2e corpus, only 41% of node anchors
    covered the line their own text came from (median drift 3 lines, 72% within
    3). Numbering turns counting into reading (E-superpod-hardening H21).
    """
    lo = (cand.get("window-lines") or [1, 1])[0]
    body = str(cand.get("source-window", ""))
    return "\n".join(f"{lo + i:5d} | {ln}" for i, ln in enumerate(body.split("\n")))


def build_prompt(cand: dict, seeds: str) -> str:
    binders = "\n".join(cand.get("binder-context", [])) or "(none)"
    anatomy = render_enrichment(cand)
    return f"""{SYSTEM}

# Few-shot examples (the target form):
{seeds}

# Now reconstruct the graph for this passage.
paper-id: {cand['paper-id']}
window-lines: {cand['window-lines']}
binder-context (variable typings established earlier):
{binders}

deterministic anatomy detected IN this window (symbol typings, definitions,
quantifiers, proof-moves, citations — anchor to these; do not contradict them):
{anatomy}

source-window (ABSOLUTE line numbers on the left; use them verbatim in
every :source {{:lines [a b]}} — do not count lines yourself):
{numbered_window(cand)}

EDN graph:"""


# --- backends ---

def call_stub(prompt: str, cand: dict, attempt: int) -> str:
    """No-GPU plumbing stub: return a real, varied seed graph (cycles by paper)."""
    seeds = []
    for d in SEED_DIRS:
        seeds += sorted(f for f in d.glob("*.edn") if "canon-links" not in f.name)
    idx = (abs(hash(cand["paper-id"])) + attempt) % len(seeds)
    return seeds[idx].read_text()


class ModelCallError(Exception):
    """A vLLM/HTTP call failed (e.g. 400 context-overflow) — surfaced so the run
    loop can skip the paper instead of crashing the whole batch."""
    def __init__(self, code, detail):
        self.code = code
        super().__init__(f"HTTP {code}: {detail}")


def call_openai(prompt: str, cand: dict, attempt: int, model: str) -> str:
    import urllib.request
    import urllib.error
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
    try:
        with urllib.request.urlopen(
                req, timeout=int(os.environ.get("FUTON6_LLM_TIMEOUT", "300"))) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise ModelCallError(e.code, detail)
    except urllib.error.URLError as e:
        raise ModelCallError(0, str(e.reason))


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
    # `path` is the single explicit file we want gated (during the retry loop it lives
    # under .attempts/ as <pid>.attemptN.edn). The bb gates' default skips attempt-named
    # files — that exclusion is meant for *directory* scans, not an explicitly-named file.
    # Pass --include-attempts so the file we hand it is actually checked (else argcheck
    # reports "No .edn files found" and every paper fails the gate). See run_rung2 twin.
    chk = subprocess.run(["bb", str(ARGCHECK), "--include-attempts", str(path)], capture_output=True, text=True)
    if chk.returncode != 0:
        return False, "checker: " + (chk.stdout + chk.stderr).strip()[-500:]
    sub = subprocess.run([sys.executable, str(SUBSTANCE), str(path), "--kind", "iatc"],
                         capture_output=True, text=True)
    if sub.returncode != 0:
        return False, "substance: " + (sub.stdout + sub.stderr).strip()[-500:]
    return True, "ok"


def rung2_passed(report_path: Path) -> bool:
    text = report_path.read_text(encoding="utf-8")
    return ":pass true" in text and ":pass false" not in text


def run_rung2(graph_path: Path, report_path: Path, *, gate: bool) -> tuple[bool, str]:
    """Run rung-2 as a description-first sidecar.

    Soft mode records the profile/verdict without rejecting the graph. Hard mode
    passes `--gate` through to the checker so semantic failures force a retry.
    """
    # --include-attempts: in hard-gate mode graph_path is an attempt file (.attempts/…),
    # which semcheck's default would skip (dir-scan exclusion). Harmless for the soft-mode
    # final-graph path (not attempt-named). Twin of the gate_one fix.
    cmd = ["bb", str(SEMCHECK), "--include-attempts", "--out", str(report_path)]
    if gate:
        cmd.append("--gate")
    cmd.append(str(graph_path))
    sem = subprocess.run(cmd, capture_output=True, text=True)
    if not report_path.exists():
        return False, "rung2: no semcheck report emitted: " + (sem.stdout + sem.stderr).strip()[-500:]
    passed = rung2_passed(report_path)
    if gate and sem.returncode != 0:
        return False, "rung2: " + (sem.stdout + sem.stderr).strip()[-500:]
    return passed, "rung2-pass" if passed else "rung2-soft-fail"


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


CANDIDATE_SCHEMA = "iatc-candidate/v2-enriched"


def require_enriched(cands: list[Path]) -> bool:
    """Hard precondition gate: refuse to run the model stage on candidates that do
    not carry the inlined deterministic anatomy. Without this, the loop silently
    feeds the model raw source + binders only (the enrichment-bypass liability) and
    a whole GPU run is wasted before anyone notices."""
    stale = []
    for cf in cands:
        try:
            c = json.loads(cf.read_text())
        except Exception as e:
            stale.append((cf.name, f"unreadable: {e}"))
            continue
        if c.get("schema") != CANDIDATE_SCHEMA or "enrichment" not in c:
            stale.append((cf.name, f"schema={c.get('schema')!r}, enrichment={'enrichment' in c}"))
    if stale:
        print(f"FATAL: {len(stale)}/{len(cands)} candidate(s) are pre-enrichment — "
              f"the deterministic anatomy would never reach the model "
              f"(the silent-bypass liability). Refusing to run the model stage.",
              file=sys.stderr)
        for name, why in stale[:10]:
            print(f"  - {name}: {why}", file=sys.stderr)
        print(f"Expected schema '{CANDIDATE_SCHEMA}' with an 'enrichment' field. "
              f"Re-extract: python scripts/mark3_extract_candidates.py --out <candidates-dir>",
              file=sys.stderr)
        return False
    return True


def run(args) -> int:
    cands = sorted(Path(args.candidates).glob("*.candidate.json"))
    if not cands:
        print("no candidates found", file=sys.stderr)
        return 2
    if not require_enriched(cands):
        return 2
    seeds = load_seeds(args.shots)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = outdir / ".attempts"
    tmp.mkdir(exist_ok=True)
    results = []
    accepted_graphs = []
    # in-flight progress (Rob's ask: periodic snapshots, not just the final
    # summary — same pattern as superpod-job.py's stage-5 loss snapshots).
    n_total = len(cands)
    t0 = time.time()
    n_resumed = n_rung2_soft = 0
    # tolerate callers that build their own args Namespace (tests, mark4_iatc_concurrent)
    loss_log_interval = getattr(args, "loss_log_interval", 100)

    def loss_snapshot(i):
        done = i - n_resumed  # freshly processed this session
        rate = done / max(time.time() - t0, 1e-9) * 60
        n_pass = sum(1 for _, s, _ in results if s == "pass")
        eta_min = (n_total - i) / max(rate, 1e-9)
        print(f"  [{i}/{n_total}] loss snapshot: pass={n_pass} fail={i - n_pass} "
              f"resumed={n_resumed} rung2-soft-fail={n_rung2_soft} · "
              f"pass-rate={n_pass / max(i, 1):.2f} · {rate:.1f} proofs/min · "
              f"ETA {eta_min / 60:.1f}h", flush=True)

    for i, cf in enumerate(cands, 1):
        cand = json.loads(cf.read_text())
        pid = cand.get("proof-id", cand["paper-id"])  # unique per proof (all-proofs); falls back to paper-id
        final = outdir / f"{pid}.edn"
        if final.exists():                       # resume: skip papers already done
            results.append((pid, "pass", "(resumed: existing graph)"))
            n_resumed += 1
            print(f"  [{i}/{n_total}] {pid}: pass (resumed)", flush=True)
            if loss_log_interval and i % loss_log_interval == 0:
                loss_snapshot(i)
            continue
        prompt = build_prompt(cand, seeds)
        status, last_err = "fail", ""
        for attempt in range(MAX_ATTEMPTS):
            p = prompt if attempt == 0 else prompt + f"\n\n# previous attempt failed the gate:\n{last_err}\n# fix it and re-emit ONLY the EDN."
            if args.backend == "stub":
                resp = call_stub(p, cand, attempt)
            else:
                try:
                    resp = call_openai(p, cand, attempt, args.model)
                except ModelCallError as e:
                    last_err = str(e)
                    if e.code == 400:            # context-overflow/bad request — retry won't help; skip paper
                        break
                    continue
            edn = extract_edn(resp)
            if not edn:
                last_err = "no EDN map found in response"
                continue
            # LaTeX in :text is illegal EDN escaping (\\Phi, \\xi); repair before
            # gating so the bb reader does not reject an otherwise good graph (H18).
            try:
                import os as _os
                import sys as _sys
                _h = _os.path.dirname(_os.path.abspath(__file__))
                if _h not in _sys.path:
                    _sys.path.insert(0, _h)
                from edn_compat import repair_string_escapes as _rse
                edn = _rse(edn)
            except Exception:
                pass
            ap = tmp / f"{pid}.attempt{attempt}.edn"
            ap.parent.mkdir(parents=True, exist_ok=True)
            ap.write_text(edn)
            # mechanical canonicalization before gating: mirror missing-warrants
            # into :holes + back-fill edge :source from endpoint nodes (no LLM).
            subprocess.run(["bb", str(REPAIR), str(ap)], capture_output=True, text=True)
            edn = ap.read_text()
            ok, err = candidate_check(edn, cand)
            if ok:
                ok, err = gate_one(ap)
            if ok:
                final = outdir / f"{pid}.edn"
                rung2_report = outdir / f"{pid}.rung2.edn"
                if args.rung2_gate:
                    attempt_report = tmp / f"{pid}.attempt{attempt}.rung2.edn"
                    r2_ok, r2_msg = run_rung2(ap, attempt_report, gate=True)
                    if not r2_ok:
                        last_err = r2_msg
                        continue
                final.write_text(edn)
                r2_ok, r2_msg = run_rung2(final, rung2_report, gate=False)
                if not r2_ok:
                    n_rung2_soft += 1
                accepted_graphs.append(final)
                status = "pass"
                last_err = f"attempt {attempt}; {r2_msg}; report {rung2_report.name}"
                break
            last_err = err
        results.append((pid, status, last_err))
        print(f"  [{i}/{n_total}] {pid}: {status} ({last_err[:80]})", flush=True)
        if loss_log_interval and i % loss_log_interval == 0:
            loss_snapshot(i)

    # cross-item substance gate over the accepted batch
    print("\n=== batch substance gate (cross-item) ===")
    sub_paths = [str(p) for p in accepted_graphs] or [str(outdir)]
    sub = subprocess.run([sys.executable, str(SUBSTANCE), *sub_paths, "--kind", "iatc"],
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
    ap.add_argument("--rung2-gate", action="store_true",
                    help="Hard-gate rung-2 semantic failures; default records the profile/verdict only.")
    ap.add_argument("--loss-log-interval", type=int, default=100,
                    help="print an in-flight loss snapshot every N proofs (pass/fail/rate/ETA); "
                         "0 disables. Same pattern as superpod-job.py stage 5.")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
