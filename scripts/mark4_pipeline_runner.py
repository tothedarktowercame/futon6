#!/usr/bin/env python3
"""mark4 — proper GPU-validation pipeline runner.

Declares the dp-anatomy -> IATC stages in DEPENDENCY ORDER, VALIDATES the order
(every stage consumes only artifacts produced by an EARLIER stage; the single GPU
stage — the 70B IATC LLM — runs LAST, after all CPU enrichment), and proves the
enrichment is real by running the CPU anatomy stage locally.

The flaw it fixes: today's run executed IATC (the last stage) FIRST, on raw source
+ a few binder hints — skipping the grounding / scopes / expository / CPU-inference
enrichment that IATC is designed to read.

    mark4_pipeline_runner.py --plan            # print + validate the ordered DAG
    mark4_pipeline_runner.py --evidence 0905.0595   # prove stage-1 enrichment exists
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

# Ordered pipeline. consumes = artifacts from EARLIER stages; produces = for LATER.
# device='gpu' only for the IATC LLM call — everything upstream is CPU & cheap.
STAGES = [
    {"id": "1.anatomy", "device": "cpu", "script": "dp_paper_view.build / render_gh200.py",
     "consumes": ["paper.tex"],
     "produces": ["marks: scopes, binders, grounding(sym->type), expository, CPU-inferences"]},
    {"id": "2.candidates", "device": "cpu", "script": "mark3_extract_candidates.py",
     "consumes": ["marks"],
     "produces": ["candidate: passage + binder-context + (FIX: + grounding/scopes/expository)"]},
    {"id": "3.iatc", "device": "GPU", "script": "mark3_iatc_loop.py  [vLLM 70B-AWQ TP=4]",
     "consumes": ["candidate"],
     "produces": ["argument-graph (.edn)"]},
    {"id": "4.repair+gate", "device": "cpu", "script": "iatc_repair.bb + iatc_argcheck.bb + substance_gate.py",
     "consumes": ["argument-graph (.edn)"],
     "produces": ["gated-graph"]},
    {"id": "5.render", "device": "cpu", "script": "build_iatc_goldens.py",
     "consumes": ["gated-graph", "marks"],
     "produces": ["side-by-side demo (CPU marks vs GPU marks)"]},
]


def validate() -> bool:
    print("=== mark4 pipeline — ordered stage DAG ===\n")
    produced_by = {}            # artifact -> first stage index that produces it
    ok = True
    for i, s in enumerate(STAGES):
        dev = "GPU" if s["device"] == "GPU" else "cpu"
        print(f"  [{s['id']:14}] ({dev})  {s['script']}")
        for c in s["consumes"]:
            if c == "paper.tex":
                print(f"        consumes {c:36} ✓ external input")
            elif c in produced_by:
                print(f"        consumes {c:36} ✓ from stage {STAGES[produced_by[c]]['id']}")
            else:
                print(f"        consumes {c:36} ✗ NOT produced by an earlier stage")
                ok = False
        for p in s["produces"]:
            produced_by.setdefault(p.split(":")[0].strip() if ":" in p else p, i)
            produced_by.setdefault(p, i)
    gpu = [i for i, s in enumerate(STAGES) if s["device"] == "GPU"]
    print()
    if len(gpu) == 1:
        gi = gpu[0]
        upstream_cpu = all(STAGES[j]["device"] == "cpu" for j in range(gi))
        print(f"  GPU stage = '{STAGES[gi]['id']}' at position {gi+1}/{len(STAGES)}")
        print(f"  all {gi} upstream stages are CPU enrichment that runs BEFORE it: "
              f"{'✓ yes' if upstream_cpu else '✗ NO'}")
        ok = ok and upstream_cpu
    print(f"\n  ORDER VALID: {'✓ YES — IATC consumes enrichment produced before it' if ok else '✗ NO'}")
    print("\n  Contrast with today's (wrong) run: stage 3 (IATC) ran FIRST, consuming raw "
          "source + binders\n  only — stages 1's grounding/scopes/expository never reached it.")
    return ok


def evidence(pid: str) -> int:
    """Run stage 1 (CPU anatomy) and show the enrichment that IATC SHOULD consume —
    vs what today's candidate actually carried."""
    import dp_paper_view as dpv
    import json
    from collections import Counter
    d = dpv.build(pid, with_ca=True, with_binders=True, with_scopes=True, with_xref=True)
    kinds = Counter(m.get("kind") for m in d["marks"])
    grounded = sum(v for k, v in kinds.items() if k and ("grounded" in k or k in ("definiens", "concept")))
    print(f"=== stage 1 (CPU anatomy) on {pid} — the enrichment IATC should read ===")
    print(f"  total marks: {len(d['marks'])}  ({len(d['text'])} chars)")
    for k, v in kinds.most_common(12):
        print(f"    {k:24} {v}")
    print(f"  grounded/concept marks: {grounded}")
    cand = REPO / "data" / "iatc-candidates-dpdemo" / f"{pid}.candidate.json"
    if cand.exists():
        c = json.loads(cand.read_text())
        print(f"\n  what TODAY's candidate carried to IATC: keys={list(c.keys())}")
        print(f"    binder-context: {len(c.get('binder-context',[]))} entries; source-window: RAW LaTeX")
        print(f"  -> {len(d['marks'])} enrichment marks produced, but only "
              f"{len(c.get('binder-context',[]))} binder hints reached the 70B. That is the gap.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--evidence", metavar="PID")
    a = ap.parse_args(argv)
    if a.evidence:
        return evidence(a.evidence)
    validate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
