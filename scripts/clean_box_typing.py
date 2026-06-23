#!/usr/bin/env python3
"""S4 box-typing driver: IATC graph -> CLean, with LLaMA typing via the served 70B.

Closes the S4 automation gap. For each IATC argument-graph:
  1. build the mechanical CLean skeleton + the box-typing prompt (iatc_to_clean)
  2. query the served vLLM (OpenAI-compatible /v1/chat/completions, temperature 0)
  3. parse the JSON typing {box-id: method, "_macro": macro}
  4. VALIDATE in-loop against clean-method-vocab.edn (the G-method-vocab gate);
     on parse-fail or off-vocab, re-prompt with a correction, up to --max-retries
  5. apply the typing -> typed *.clean.edn
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
import re
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import edn_format as edn  # noqa: E402
import iatc_to_clean as itc  # noqa: E402


def load_vocab():
    d = dict(edn.loads(open(os.path.join(ROOT, "holes/clean/clean-method-vocab.edn")).read()))
    def names(m):
        return {str(k).lstrip(":") for k in dict(m).keys()}
    mv = next(v for k, v in d.items() if str(k).endswith("method-vocab"))
    sv = next(v for k, v in d.items() if str(k).endswith("macro-shapes"))
    return names(mv), names(sv)


def extract_json(text):
    # strip ```json fences, grab the first {...}
    text = re.sub(r"```(?:json)?", "", text)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def valid(typing, sk, methods, macros):
    if not isinstance(typing, dict):
        return False, "not a json object"
    if typing.get("_macro") not in macros:
        return False, f"_macro {typing.get('_macro')!r} not in vocab"
    for b in sk["boxes"]:
        mt = typing.get(b["id"])
        if mt not in methods:
            return False, f"box {b['id']} method {mt!r} not in vocab"
    return True, "ok"


def query_model(endpoint, model, prompt):
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": 600,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(endpoint, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def stub_typing(sk):
    # deterministic, in-vocab: lets the plumbing run with no model
    return {**{b["id"]: "reduce-to-known-result" for b in sk["boxes"]},
            "_macro": "construct-exploit-discharge"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--endpoint", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--model", default="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--run-dir", help="if set, emit S4 MetricRecords here (INSTANTIATE-GPU)")
    ap.add_argument("--run-id", default="adhoc")
    ap.add_argument("--corpus-id", default="adhoc")
    args = ap.parse_args()

    methods, macros = load_vocab()
    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
    typed, failed, rejected = [], [], []
    # skip sidecar report files (e.g. <pid>.rung2.edn) the IATC loop writes alongside
    # the proof graphs — they aren't argument graphs.
    for gf in sorted(g for g in glob.glob(os.path.join(ROOT, args.graphs, "*.edn"))
                     if not g.endswith(".rung2.edn")):
        pid = os.path.basename(gf)[:-4]
        try:
            nodes, edges = itc.load_graph(gf)
            sk0 = itc.build_skeleton(nodes, edges)
            prompt = itc.emit_prompt(pid, nodes, edges, sk0)
        except Exception as e:   # malformed graph shouldn't abort the whole batch
            failed.append((pid, f"load error: {type(e).__name__}: {e}"))
            print(f"  FAIL {pid}: load error — {e}")
            continue
        typing, why = None, "no attempt"
        for attempt in range(args.max_retries + 1):
            try:
                t = stub_typing(sk0) if args.stub else extract_json(
                    query_model(args.endpoint, args.model, prompt))
                ok, why = valid(t, sk0, methods, macros) if t is not None else (False, "unparseable")
                if ok:
                    typing = t
                    break
                prompt += (f"\n\nYour previous answer was invalid ({why}). Return ONLY "
                           f"valid JSON using ONLY the listed tags.")
            except Exception as e:  # network / endpoint
                why = f"query error: {e}"
        if typing is None:
            failed.append((pid, why))
            print(f"  FAIL {pid}: {why}")
            continue
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
        if os.system(f"cd {ROOT} && bb scripts/clean_argcheck.bb {outfile} >/dev/null 2>&1") != 0:
            os.remove(outfile)
            rejected.append(pid)
            print(f"  REJECT {pid}: not a DAG comb (e.g. cyclic-equivalence) — logged")
            continue
        typed.append(pid)
        if args.run_dir:  # S4 inline metric emit (non-fatal — never abort the CLean)
            try:
                import metric_harness as mh
                txt = open(outfile).read()
                nbox = max(1, len(sk.get("boxes", [])))
                discharge = max(0, nbox - txt.count(":hole")) / nbox
                mh.emit_record(args.run_dir, run_id=args.run_id, corpus_id=args.corpus_id,
                               paper_id=pid, stage="S4", metric="clean-discharge-rate",
                               axis="completeness", value=round(discharge, 4), computable=True)
            except Exception as ee:
                print(f"    (S4 metric emit skipped: {ee})")

    print(f"\ntyped {len(typed)} / rejected {len(rejected)} (cyclic) / failed {len(failed)} (typing)"
          f"  -> {args.out}  ({'stub' if args.stub else args.model})")
    if rejected:
        print(f"rejected (cyclic, logged): {rejected}")
    if failed:
        print(f"typing-failed: {[p for p,_ in failed]}")
    # S4 postcondition gates over the accepted CLeans
    rc2 = os.system(f"cd {ROOT} && bb scripts/clean_vocab_gate.bb {args.out} >/dev/null 2>&1")
    print(f"[gate] clean_vocab_gate over accepted: {'PASS' if rc2==0 else 'FAIL'}")
    # success = all graphs either typed or cleanly rejected; no typing failures, vocab clean
    sys.exit(0 if (not failed and rc2 == 0) else 1)


if __name__ == "__main__":
    main()
