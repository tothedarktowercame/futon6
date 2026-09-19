#!/usr/bin/env python3
"""S3 IATC model loop: one schema-constrained model call per S1-identified proof.

Per candidate (from mark3_extract_candidates.py --all-proofs):
  prompt with the statement and the numbered proof -> the model returns JSON under
  the iatc_json schema -> code checks what the schema cannot (references, step
  order, line ranges) -> code writes the EDN graph -> iatc_argcheck + substance
  gate -> rung-2 profile -> accept. Finally the substance gate runs over the batch.

The model never writes EDN, and nothing is repaired or retried to fix a format.
An output that breaks the contract is a rejected item with its reasons. Each
item gets one call per stage invocation, at temperature 0, so the result is a
measurement of the prompt, model and contract rather than of resampling luck;
re-invoking a failed stage retries only the items that were not accepted.

Backends:
  --backend stub    : no GPU; a small deterministic JSON document per candidate.
  --backend openai  : OpenAI-compatible HTTP with response_format json_schema.
                      Reads OPENAI_BASE_URL + OPENAI_API_KEY; --model.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iatc_json  # noqa: E402
import stage_accounting as accounting  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ARGCHECK = REPO / "scripts" / "iatc_argcheck.bb"
SUBSTANCE = REPO / "scripts" / "substance_gate.py"
SEMCHECK = REPO / "scripts" / "iatc_semcheck.bb"
CANDIDATE_SCHEMA = "iatc-candidate/v3-proof"
MAX_TOKENS = int(os.environ.get("FUTON6_IATC_MAX_TOKENS", "8192"))

NODES_TASK = """You read ONE mathematical proof and list what its argument is made of.

Return JSON with a list "nodes". Each node is one thing the proof uses or
establishes: kind "claim" (an assertion), "object" (a mathematical object it
introduces or constructs), "definition", or "ref" (a result it points to — put the
label or citation in "citation", e.g. "Theorem~\\ref{main}" or "[AR, 2.36]";
otherwise leave it ""). "text" is a faithful short gloss of the source.
"first_line"/"last_line" are the ABSOLUTE line numbers printed on the left.

List them in the order the proof introduces them, hypotheses first and the final
conclusion last. Include every intermediate claim the argument passes through; the
number of nodes follows the proof."""

STEPS_TASK = """Now give the argument over the nodes you listed, as JSON with an
object "derivations".

Its keys are the numbers of the nodes the proof DERIVES; a node the proof simply
assumes, introduces or cites has no entry. Each key's value is a list holding that
node's derivation — two entries only if the proof really derives it twice, by
separate routes.

A derivation gives the "premises" it follows from (numbers of OTHER nodes),
"relation" for how, and the warrant for why. Choose the warrant kind by what the
TEXT does, not by what you can supply:
- "stated" only when the proof itself gives the reason, in the text;
- "citation" when it points to a result (give it in "warrant");
- "missing" when the text asserts the step without saying why — including
  "clearly", "it is easy to see", "a routine computation", or nothing at all.
  Then "warrant" names the specific fact the proof elided (e.g. "dimension shift
  through a short exact sequence"), never a generic word.
Published proofs elide steps constantly; recording that honestly is the point of
this layer. "first_line"/"last_line" locate the derivation.

Checked by code; an output that breaks this is rejected:
- The derivations must not go in a circle: if node A is used to derive node B, then
  B must not, directly or through other nodes, be used to derive A. Write an
  equivalence as ONE derivation with relation "iff".
- Every line lies in the given source."""


def render_enrichment(cand: dict) -> str:
    rows = cand.get("enrichment") or []
    if not rows:
        return "(no deterministic anatomy detected in this proof)"
    return "\n".join(f"L{r['line']} ({r['kind']}) {r['tip']}" for r in rows)


def numbered_window(cand: dict) -> str:
    """Source with ABSOLUTE line numbers, so anchors are read, not counted (H21)."""
    lo = (cand.get("window-lines") or [1, 1])[0]
    body = str(cand.get("source-window", ""))
    return "\n".join(f"{lo + i:5d} | {ln}" for i, ln in enumerate(body.split("\n")))


def build_prompt(cand: dict, task: str, nodes: list | None = None) -> str:
    binders = "\n".join(cand.get("binder-context", [])) or "(none)"
    proved = cand.get("proved")
    statement = (f"The proof establishes this {proved['kind']} (lines {proved['lines'][0]}-{proved['lines'][1]}):\n"
                 f"{proved['text']}" if proved else "No preceding statement was identified for this proof.")
    lo, hi = cand["window-lines"]
    listing = ""
    if nodes is not None:
        rows = "\n".join(f"  {i}. ({n['kind']}) {n['text']}" + (f"  [{n['citation']}]" if n.get("citation") else "")
                          for i, n in enumerate(nodes, 1))
        listing = f"\nThe nodes you listed for this proof:\n{rows}\n"
    return f"""{task}

{statement}

Variable typings established earlier in the paper:
{binders}

Deterministic anatomy detected in this source (symbol typings, definitions,
quantifiers, citations — consistent with the text; do not contradict them):
{render_enrichment(cand)}

Source, lines {lo}-{hi} (ABSOLUTE line numbers on the left):
{numbered_window(cand)}
{listing}"""


class ModelCallError(Exception):
    """The endpoint could not produce a judgeable answer (HTTP error, truncation)."""

    def __init__(self, code, detail):
        self.code = code
        super().__init__(f"HTTP {code}: {detail}" if code else detail)


def call_stub(prompt: str, cand: dict, schema: dict) -> str:
    """No-GPU plumbing: a minimal valid answer for whichever phase is asked."""
    lo, hi = cand["proof-lines"]
    if "nodes" in schema["properties"]:
        return json.dumps({"nodes": [{"kind": "claim", "text": "hypotheses of the statement", "citation": "",
                                      "first_line": lo, "last_line": lo},
                                     {"kind": "claim", "text": "conclusion of the statement", "citation": "",
                                      "first_line": hi, "last_line": hi}]})
    return json.dumps({"derivations": {"2": [{"relation": "implies", "premises": [1],
                                              "warrant_kind": "missing",
                                              "warrant": f"argument of {cand['proof-id']}",
                                              "first_line": lo, "last_line": hi}]}})


def call_openai(prompt: str, cand: dict, model: str, schema: dict) -> str:
    import urllib.error
    import urllib.request
    base = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    key = os.environ.get("OPENAI_API_KEY", "x")
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "iatc_proof", "strict": True, "schema": schema}},
    }).encode()
    req = urllib.request.Request(f"{base}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(
                req, timeout=int(os.environ.get("FUTON6_LLM_TIMEOUT", "600"))) as r:
            choice = json.loads(r.read())["choices"][0]
    except urllib.error.HTTPError as e:
        raise ModelCallError(e.code, e.read().decode("utf-8", "replace")[:300])
    except urllib.error.URLError as e:
        raise ModelCallError(0, str(e.reason))
    if choice.get("finish_reason") == "length":
        # A truncated document is not the model's answer; nothing is salvaged from it.
        raise ModelCallError(0, f"output truncated at max_tokens={MAX_TOKENS}")
    return choice["message"]["content"]


def gate_one(path: Path) -> tuple[bool, str]:
    # --include-attempts: the explicit file lives under .attempts/, which the bb
    # gates otherwise skip in directory scans.
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
    """Rung-2 semantic profile; with gate=True a failing profile rejects the graph."""
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


def require_candidates(cands: list[Path]) -> bool:
    """Refuse candidates from any other extraction contract (e.g. proof-move groups)."""
    stale = []
    for cf in cands:
        try:
            c = json.loads(cf.read_text())
        except ValueError as e:
            stale.append((cf.name, f"unreadable: {e}"))
            continue
        if c.get("schema") != CANDIDATE_SCHEMA or not c.get("proof-lines"):
            stale.append((cf.name, f"schema={c.get('schema')!r}"))
    if stale:
        print(f"FATAL: {len(stale)}/{len(cands)} candidate(s) are not S1 proof candidates "
              f"({CANDIDATE_SCHEMA}). Re-extract: python scripts/mark3_extract_candidates.py "
              "--all-proofs --out <candidates-dir>", file=sys.stderr)
        for name, why in stale[:10]:
            print(f"  - {name}: {why}", file=sys.stderr)
        return False
    return True


def attempt_one(cand: dict, args, tmp: Path) -> tuple[str, str, dict]:
    """(status, reason, attempt record) for one model call on one proof."""
    pid = cand["proof-id"]
    lo, hi = cand["window-lines"]
    record: dict = {"attempt": 0}
    doc: dict = {}
    for phase, task in (("nodes", NODES_TASK), ("steps", STEPS_TASK)):
        schema = (iatc_json.nodes_schema(lo, hi) if phase == "nodes"
                  else iatc_json.steps_schema(lo, hi, len(doc.get("nodes", []))))
        prompt = build_prompt(cand, task, doc.get("nodes") if phase == "steps" else None)
        try:
            raw = (call_stub(prompt, cand, schema) if args.backend == "stub"
                   else call_openai(prompt, cand, args.model, schema))
        except ModelCallError as e:
            record["result"] = f"{phase}: {e}"[:300]
            return "errored", f"{phase}: {e}", record
        raw_path = tmp / f"{pid}.{phase}.json"
        raw_path.write_text(raw)
        record[f"{phase}-response"] = accounting.relative(raw_path)
        try:
            part = json.loads(raw)
        except ValueError as e:
            why = f"{phase}: endpoint returned non-JSON despite the schema ({e}); check serving conformance"
            record["result"] = why
            return "errored", why, record
        doc.update(part)
        if phase == "steps":
            doc["steps"] = iatc_json.steps_of(doc)
        if phase == "nodes" and len(doc.get("nodes") or []) < 2:
            why = f"contract: {len(doc.get('nodes') or [])} node(s); a proof has at least two"
            record["result"] = why
            return "rejected", why, record
    found = iatc_json.problems(doc, lo, hi)
    if found:
        why = "contract: " + "; ".join(found[:6])
        record["result"] = why[:500]
        return "rejected", why, record
    graph = tmp / f"{pid}.edn"
    graph.write_text(iatc_json.to_edn(doc, cand, args.model))
    record["graph"] = accounting.relative(graph)
    ok, why = gate_one(graph)
    if ok and args.rung2_gate:
        ok, why = run_rung2(graph, tmp / f"{pid}.rung2.edn", gate=True)
    if not ok:
        record["result"] = why[:500]
        return "rejected", why, record
    record["result"] = "accepted"
    return "accepted", "", record


def run(args) -> int:
    cands = sorted(Path(args.candidates).glob("*.candidate.json"))
    if not cands:
        print("no candidates found", file=sys.stderr)
        return 2
    if not require_candidates(cands):
        return 2
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    # Attempts are scoped by run and runner invocation (H37), so a retried stage
    # adds to the history instead of overwriting the previous try's evidence.
    run_tag = os.environ.get("RUN_ID") or getattr(args, "run_id", None) or "unscoped"
    invocation = os.environ.get(accounting.INVOCATION_ENV) or "standalone"
    tmp = outdir / ".attempts" / run_tag / invocation
    if invocation != "standalone" and tmp.exists():
        print(f"attempt history already exists for invocation {invocation}", file=sys.stderr)
        return 2
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "RUN").write_text(f"run_id={run_tag}\ninvocation={invocation}\ncandidates={len(cands)}\n"
                             f"contract={iatc_json.GENERATOR}\nmodel={args.model}\n")
    loaded = [json.loads(cf_path.read_text()) for cf_path in cands]
    ledger = accounting.Accounting("S3", "loop", [c["proof-id"] for c in loaded])
    counts = {"accepted": 0, "rejected": 0, "errored": 0, "carried": 0}
    accepted_graphs = []
    t0 = time.time()
    interval = getattr(args, "loss_log_interval", 100)
    total = len(loaded)
    done = 0
    # Accounting.checkpoint() rewrites one file per record; two threads doing that
    # at once would race on the same .partial. Serialise bookkeeping only — the
    # model call and rung-2 stay outside, which is the whole point of batching.
    books = threading.Lock()

    def finish(cand: dict, status: str, why: str, record: dict, *, carried=None):
        nonlocal done
        pid = cand["proof-id"]
        final = outdir / f"{pid}.edn"
        rung2_report = outdir / f"{pid}.rung2.edn"
        artifacts = [accounting.relative(p) for p in (final, rung2_report) if p.exists()]
        with books:
            counts[status] += 1
            if carried is not None:
                counts["carried"] += 1
                accepted_graphs.append(final)
                ledger.record(pid, "accepted", paper=cand["paper-id"], outputs=[pid],
                              artifacts=artifacts,
                              attempts=[{"carried-from": carried.get("invocation"),
                                         "path": carried.get("path")}])
                note = f"accepted (carried from {carried.get('invocation')})"
            elif status == "accepted":
                accepted_graphs.append(final)
                ledger.record(pid, "accepted", paper=cand["paper-id"], outputs=[pid],
                              attempts=[record], artifacts=artifacts)
                note = status
            elif status == "errored" and record is None:
                ledger.record(pid, "errored", why, paper=cand["paper-id"], artifacts=artifacts)
                note = f"ERROR ({why})"
            else:
                ledger.record(pid, status, why, paper=cand["paper-id"], attempts=[record])
                note = status + (f" ({why[:160]})" if why else "")
            done += 1
            print(f"  [{done}/{total}] {pid}: {note}", flush=True)
            if interval and done % interval == 0:
                rate = done / max(time.time() - t0, 1e-9) * 60
                print(f"  [{done}/{total}] accepted={counts['accepted']} rejected={counts['rejected']} "
                      f"errored={counts['errored']} · {rate:.1f} proofs/min", flush=True)

    # A prior acceptance is settled by the filesystem, not the model: resolve those
    # first so the pool only ever holds real work. This is what makes resume cheap.
    pending = []
    for cand in loaded:
        final = outdir / f"{cand['proof-id']}.edn"
        if not final.exists():
            pending.append(cand)
            continue
        carried, why = accounting.carried_acceptance(outdir, cand["proof-id"], final)
        if carried is None:
            finish(cand, "errored", why, None)
        else:
            finish(cand, "accepted", "", {}, carried=carried)

    def work(cand: dict):
        pid = cand["proof-id"]
        status, why, record = attempt_one(cand, args, tmp)
        if status == "accepted":
            final = outdir / f"{pid}.edn"
            accounting.publish_accepted(outdir, pid, final, (tmp / f"{pid}.edn").read_bytes(),
                                        {"path": record["graph"]})
            _, record["rung2"] = run_rung2(final, outdir / f"{pid}.rung2.edn", gate=False)
        finish(cand, status, why, record)

    workers = max(1, int(getattr(args, "concurrency", 1) or 1))
    if workers == 1 or len(pending) <= 1:
        for cand in pending:
            work(cand)
    else:
        print(f"== {len(pending)} proof(s) at concurrency {workers} ==", flush=True)
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            for future in cf.as_completed([pool.submit(work, c) for c in pending]):
                future.result()       # re-raise in the caller; a dead pool is not a pass
    accepted_graphs.sort()            # submission order must not reach the gate

    # Cross-item substance gate (template collapse, warrant reuse) over accepted graphs.
    print("\n=== batch substance gate (cross-item) ===")
    sub_paths = [str(p) for p in accepted_graphs] or [str(outdir)]
    sub = subprocess.run([sys.executable, str(SUBSTANCE), *sub_paths, "--kind", "iatc"],
                         capture_output=True, text=True)
    print(sub.stdout.strip()[-400:])
    print(f"\nloop: accepted {counts['accepted']} (carried {counts['carried']}) · rejected {counts['rejected']} · "
          f"errored {counts['errored']} of {len(loaded)} · batch-substance {'PASS' if sub.returncode == 0 else 'FAIL'}")
    # Exit status reports whether the LOOP could do its work, not whether every
    # proof survived the contract. Refused proofs are recorded per item; how many
    # refusals a run tolerates is the runner's decision, taken against the floor in
    # the run manifest. Conflating the two stopped a 124-paper window at S3 because
    # three proofs came back circular. The batch substance gate stays fatal: it is a
    # property of the whole batch (template collapse, reused warrants), not an item.
    return 0 if sub.returncode == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(REPO / "data" / "iatc-candidates"))
    ap.add_argument("--out", default=str(REPO / "data" / "iatc-argument-graphs" / "loop-run"))
    ap.add_argument("--backend", choices=["stub", "openai"], default="stub")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--rung2-gate", action="store_true",
                    help="Reject graphs whose rung-2 semantic profile fails; default records it only.")
    ap.add_argument("--concurrency", type=int,
                    default=int(os.environ.get("FUTON6_CONCURRENCY")
                                or os.environ.get("CONCURRENCY") or 1),
                    help="proofs in flight at once; defaults to FUTON6_CONCURRENCY "
                         "(set from the detected hardware by futon6_config.scale)")
    ap.add_argument("--loss-log-interval", type=int, default=100,
                    help="print running accepted/rejected/errored counts every N proofs; 0 disables")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
