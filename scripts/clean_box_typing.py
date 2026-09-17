#!/usr/bin/env python3
"""S4 box-typing driver: IATC graph -> CLean, with LLaMA typing via the served 70B.

Closes the S4 automation gap. For each IATC argument-graph:
  1. build the mechanical CLean skeleton + the box-typing prompt (iatc_to_clean)
  2. query the served model at temperature 0 under a JSON schema whose keys are
     exactly this graph's box ids and whose values are clean-method-vocab methods
  3. check the typing (every box, only vocabulary methods); no re-prompting: an
     answer that breaks the contract rejects the graph with its reason
  4. derive the macro from the box methods (clean_macro_fix) and apply the typing
     -> typed *.clean.edn, written by code
Failures are logged, never silently dropped (the typed proof is simply absent and
named in the failure list). After the batch, run clean_argcheck + clean_vocab_gate.

The vLLM call path is exercised on the box; --stub validates the plumbing locally
(deterministic in-vocab typing, no model).

Usage (on the Linode host, after S3):
  futon6/.venv/bin/python scripts/clean_box_typing.py \
      --graphs data/iatc-argument-graphs/<run> --out holes/clean-ct \
      --endpoint http://localhost:8000/v1/chat/completions \
      --model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
  # local plumbing test:
  futon6/.venv/bin/python scripts/clean_box_typing.py --graphs data/iatc-argument-graphs/gh200 --out /tmp/ct --stub
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import edn_format as edn  # noqa: E402
import iatc_to_clean as itc  # noqa: E402
import stage_accounting as accounting  # noqa: E402


def load_vocab():
    d = dict(edn.loads(open(os.path.join(ROOT, "holes/clean/clean-method-vocab.edn")).read()))
    def names(m):
        return {str(k).lstrip(":") for k in dict(m).keys()}
    mv = next(v for k, v in d.items() if str(k).endswith("method-vocab"))
    sv = next(v for k, v in d.items() if str(k).endswith("macro-shapes"))
    return names(mv), names(sv)


def typing_schema(sk, methods):
    boxes = [b["id"] for b in sk["boxes"]]
    return {"type": "object", "additionalProperties": False, "required": boxes,
            "properties": {b: {"type": "string", "enum": sorted(methods)} for b in boxes}}


def valid(typing, sk, methods):
    if not isinstance(typing, dict):
        return False, "not a json object"
    extra = sorted(set(typing) - {b["id"] for b in sk["boxes"]})
    if extra:
        return False, f"keys that are not box ids: {extra[:5]}"
    for b in sk["boxes"]:
        mt = typing.get(b["id"])
        if mt not in methods:
            return False, f"box {b['id']} method {mt!r} not in vocab"
    return True, "ok"


class TypingCallError(Exception):
    """The endpoint could not produce a judgeable answer."""


def query_model(endpoint, model, prompt, sk, methods):
    max_tokens = max(600, 40 * len(sk["boxes"]))
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}],
                       "response_format": {"type": "json_schema", "json_schema": {
                           "name": "clean_typing", "strict": True, "schema": typing_schema(sk, methods)}}}).encode()
    req = urllib.request.Request(endpoint, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(
                req, timeout=int(os.environ.get("FUTON6_LLM_TIMEOUT", "120"))) as r:
            choice = json.loads(r.read())["choices"][0]
    except urllib.error.URLError as e:
        raise TypingCallError(f"query error: {e}")
    if choice.get("finish_reason") == "length":
        raise TypingCallError(f"output truncated at max_tokens={max_tokens}")
    try:
        return json.loads(choice["message"]["content"])
    except ValueError as e:
        raise TypingCallError(f"endpoint returned non-JSON despite the schema ({e}); check serving conformance")


def wait_for_server(endpoint):
    """Server-down (connection refused) is a server state, not a graph property:
    fast-failing wiped 87/98 graphs during a 35 s llama-server restart (2026-08-06).
    Wait (bounded) for /health before the one call on this graph."""
    import time
    for _ in range(30):
        try:
            urllib.request.urlopen(endpoint.rsplit("/v1", 1)[0] + "/health", timeout=5)
            return
        except Exception:
            time.sleep(10)


def stub_typing(sk):
    # deterministic, in-vocab: lets the plumbing run with no model
    return {b["id"]: "reduce-to-known-result" for b in sk["boxes"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--endpoint", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--model", default="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--run-dir", help="if set, emit S7 MetricRecords here (INSTANTIATE-GPU)")
    ap.add_argument("--run-id", default="adhoc")
    ap.add_argument("--corpus-id", default="adhoc")
    args = ap.parse_args()

    if args.run_dir and "adhoc" in (args.run_id, args.corpus_id):
        ap.error("--run-dir requires explicit --run-id and --corpus-id (records would be tagged adhoc)")
    methods, _macros = load_vocab()
    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
    typed, failed, rejected = [], [], []
    graphs_dir = os.path.join(ROOT, args.graphs)
    if args.stub:
        # Plumbing mode reads every final; it has no acceptance provenance to check.
        finals = [(os.path.basename(g)[:-4], g) for g in sorted(glob.glob(os.path.join(graphs_dir, "*.edn")))
                  if not g.endswith(".rung2.edn")]
        refused = []
    else:
        # Only S3 finals with verified acceptance provenance are typed; a stale or
        # foreign final is an errored item, never input to a CLean.
        accepted, refused = accounting.accepted_finals(graphs_dir)
        finals = [(item, str(path)) for item, path in accepted]
    ledger = accounting.Accounting("S7", "typing", [item for item, _ in finals])
    for name, why in refused:
        failed.append((name, why))
        print(f"  FAIL {name}: {why}")
    for pid, gf in finals:
        try:
            dropped = []
            nodes, edges = itc.load_graph(gf, skipped=dropped)
            if dropped:
                raise ValueError(f"infer edge(s) without :id/:conclusion cannot become boxes: {dropped[:5]}")
            sk0 = itc.build_skeleton(nodes, edges)
            prompt = itc.emit_prompt(pid, nodes, edges, sk0, ask_macro=False)
        except Exception as e:   # malformed graph shouldn't abort the whole batch
            rejected.append({"pid": pid, "reason": f"load error: {type(e).__name__}: {e}"})
            ledger.record(pid, "rejected", f"load error: {type(e).__name__}: {e}", paper=pid)
            print(f"  REJECT {pid}: load error — {e}")
            continue
        try:
            if args.stub:
                answer = stub_typing(sk0)
            else:
                wait_for_server(args.endpoint)
                answer = query_model(args.endpoint, args.model, prompt, sk0, methods)
        except TypingCallError as e:
            failed.append((pid, str(e)))
            ledger.record(pid, "errored", f"typing: {e}", paper=pid)
            print(f"  FAIL {pid}: {e}")
            continue
        ok, why = valid(answer, sk0, methods)
        if not ok:
            failed.append((pid, why))
            ledger.record(pid, "rejected", f"typing contract: {why}", paper=pid)
            print(f"  REJECT {pid}: typing contract: {why}")
            continue
        typing = dict(answer)
        # macro is DERIVED from the box methods, not the model's (the 70B over-tags one
        # default — mark5 D1/Diagnostic-2). Override before applying.
        from clean_macro_fix import derive_macro
        typing["_macro"] = derive_macro([typing[b["id"]] for b in sk0["boxes"] if b["id"] in typing])
        sk = itc.build_skeleton(nodes, edges, typing)
        vacuous = sum(1 for e in edges if all(p == e["conclusion"] for p in e["premise"]))
        outfile = os.path.join(ROOT, args.out, f"{pid}.clean.edn")
        open(outfile, "w").write(itc.render_edn(pid, sk, vacuous) + "\n")
        # gate each CLean individually; cyclic-equivalence proofs aren't DAG combs —
        # log + set aside (G-cyclic: never silently drop, never fail the whole batch)
        # Report the gate that ACTUALLY failed. This used to discard argcheck's
        # output to /dev/null and print one guessed reason -- "not a DAG comb
        # (e.g. cyclic-equivalence)" -- for every rejection. It was wrong for six
        # of the ten rejects in the 98-graph corpus, which failed G1 (unreadable
        # EDN, from a serialization bug) and had no cycle at all. A wrong reason
        # is worse than none: it was believed, and it cost an investigation.
        gate = subprocess.run(["bb", "scripts/clean_argcheck.bb", outfile],
                              cwd=ROOT, capture_output=True, text=True)
        if gate.returncode != 0:
            os.remove(outfile)
            why = " | ".join(
                ln.strip() for ln in (gate.stdout + gate.stderr).splitlines()
                if ln.strip() and ("FAIL" in ln or ln.strip().startswith(("G", "-", "["))))
            rejected.append({"pid": pid, "reason": why[:300] or f"exit {gate.returncode}"})
            # The gate output is the evidence (e.g. G7); the rejected CLean is not kept
            # as an output, so record the reason where accounting can see it.
            ledger.record(pid, "rejected", "clean_argcheck: " + (why[:300] or f"exit {gate.returncode}"), paper=pid)
            print(f"  REJECT {pid}: {why[:200] or 'argcheck exit ' + str(gate.returncode)}")
            continue
        typed.append(pid)
        ledger.record(pid, "accepted", paper=pid, outputs=[pid], artifacts=[accounting.relative(outfile)])
        if args.run_dir:  # S4 inline metric emit (non-fatal — never abort the CLean)
            try:
                import metric_harness as mh
                txt = open(outfile).read()
                nbox = max(1, len(sk.get("boxes", [])))
                discharge = max(0, nbox - txt.count(":hole")) / nbox
                mh.emit_record(args.run_dir, run_id=args.run_id, corpus_id=args.corpus_id,
                               paper_id=pid, stage="S7", metric="clean-discharge-rate",
                               axis="completeness", value=round(discharge, 4), computable=True)
            except Exception as ee:
                print(f"    (S7 metric emit skipped: {ee})")

    # "(cyclic)" was a guess baked into the summary line as well as the per-item
    # one. Rejections are reported by their actual gate now, and grouped, so a
    # systematic cause is visible as a cluster instead of being read as a
    # property of the mathematics.
    print(f"\ntyped {len(typed)} / rejected {len(rejected)} (gate) / failed {len(failed)} (typing)"
          f"  -> {args.out}  ({'stub' if args.stub else args.model})")
    if rejected:
        by_reason = {}
        for r in rejected:
            by_reason.setdefault(r["reason"][:80], []).append(r["pid"])
        print("rejected by gate:")
        for reason, pids in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
            print(f"  [{len(pids)}] {reason}")
            print(f"      {', '.join(pids)}")
    if failed:
        print(f"typing-failed: {[p for p,_ in failed]}")
    # A CLean in the output directory that this invocation did not accept is stale
    # (e.g. its graph lost acceptance provenance); the vocab gate and S8 would read it.
    stale = sorted(os.path.basename(c)[:-len(".clean.edn")]
                   for c in glob.glob(os.path.join(ROOT, args.out, "*.clean.edn"))
                   if os.path.basename(c)[:-len(".clean.edn")] not in set(typed))
    if stale:
        print(f"stale CLeans not produced by this invocation: {stale[:5]}")
        failed.extend((s_, "stale CLean output") for s_ in stale)
    # S7 postcondition gates over the accepted CLeans
    rc2 = os.system(f"cd {ROOT} && bb scripts/clean_vocab_gate.bb {args.out} >/dev/null 2>&1")
    print(f"[gate] clean_vocab_gate over accepted: {'PASS' if rc2==0 else 'FAIL'}")
    # Success means every graph typed and passed clean_argcheck. A gate rejection
    # (G1-G8, including a G7 cycle) used to exit 0 as "cleanly rejected", which put
    # a passing S7 ledger row over a CLean corpus with proofs missing.
    sys.exit(0 if (not failed and not rejected and rc2 == 0) else 1)


if __name__ == "__main__":
    main()
