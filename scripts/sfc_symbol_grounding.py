#!/usr/bin/env python3
"""SFC2b — LLM symbol grounding for the definition `:structure` (futon6).

The H-SFC2b layer of E-structure-first-concepts: fill the `:grounding :hole` slots
that the deterministic SFC2a transducer leaves, by binding each abstract symbol to
its *per-paper domain meaning* read from the surrounding prose. This is
M-symbol-grounding applied at the definition layer:

  - PER-PAPER + DEFEASIBLE: a binding is a claim about THIS context, not a global fact;
  - CHECKABLE: a `:grounded` binding must cite a verbatim evidence span in the context
    (else it is `:unsupported` and rejected) — and a symbol the context never
    introduces is honestly `:undefined-in-context`, never guessed.

Mirrors the pre-superpod IATC/expository loops: OpenAI-compatible client
(`OPENAI_BASE_URL`/`OPENAI_API_KEY`, `--model`), with a `stub` backend so the
plumbing + the evidence-check validate with no GPU, and an `openai` backend ready
for a served model.

  sfc_symbol_grounding.py --formula <tex> --context <file|-> [--backend stub|openai] [--model M]
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
SFC_DEF = REPO / "scripts" / "sfc_def_structure.bb"

SYSTEM = (
    "You ground the abstract symbols of ONE mathematical definition to their meaning "
    "AS GIVEN IN THIS PAPER's surrounding text. For each ungrounded symbol output its "
    "domain binding using ONLY what the context states or introduces. If the context "
    "does not introduce the symbol, mark it :undefined-in-context — DO NOT GUESS. "
    "Cite the exact verbatim substring of the context that introduces the symbol. "
    'Reply with JSON only: {"groundings":[{"symbol":S,"binding":"...",'
    '"evidence":"exact substring","status":"grounded"|"undefined-in-context"}]}'
)


def structure_and_holes(formula: str) -> tuple[str, list[str]]:
    """Run the deterministic SFC2a transducer; return (:structure edn, [symbols])."""
    out = subprocess.run(["bb", str(SFC_DEF), "-"], input=formula,
                         capture_output=True, text=True)
    edn = out.stdout
    syms = re.findall(r':symbol\s+"([^"]+)"', edn)
    # de-dup, preserve order
    seen, ordered = set(), []
    for s in syms:
        if s not in seen:
            seen.add(s); ordered.append(s)
    struct = re.search(r":structure\s+(\(.*?\))\s*,?\s*:ungrounded", edn, re.S)
    return (struct.group(1).strip() if struct else edn.strip()), ordered


def build_prompt(symbols: list[str], context: str) -> str:
    return (f"{SYSTEM}\n\nCONTEXT (the paper text around the definition):\n"
            f"\"\"\"\n{context}\n\"\"\"\n\nUNGROUNDED SYMBOLS: {', '.join(symbols)}\n"
            "Ground each, citing verbatim evidence; JSON only.")


def call_stub(symbols: list[str], context: str) -> dict:
    """No-GPU plumbing stub: context-inspecting (NOT canned per example). A symbol is
    grounded iff it occurs in the context (evidence = a window around it); otherwise
    undefined-in-context. Exercises the evidence-check honestly."""
    out = []
    for s in symbols:
        i = context.find(s)
        if i >= 0:
            lo, hi = max(0, i - 24), min(len(context), i + len(s) + 24)
            out.append({"symbol": s, "binding": f"(stub) introduced in context near '{s}'",
                        "evidence": context[lo:hi].strip(), "status": "grounded"})
        else:
            out.append({"symbol": s, "binding": "", "evidence": "",
                        "status": "undefined-in-context"})
    return {"groundings": out}


def call_openai(prompt: str, model: str) -> dict:
    import urllib.request
    base = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    key = os.environ.get("OPENAI_API_KEY", "x")
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0}).encode()
    req = urllib.request.Request(f"{base}/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        txt = json.loads(r.read())["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", txt, re.S)
    return json.loads(m.group(0)) if m else {"groundings": []}


def check(groundings: list[dict], context: str) -> list[dict]:
    """Defeasible evidence-check: a 'grounded' binding's evidence must appear verbatim
    in the context, else it is downgraded to 'unsupported' (rejected)."""
    checked = []
    for g in groundings:
        st = g.get("status")
        if st == "grounded":
            ev = (g.get("evidence") or "").strip()
            if not ev or ev not in context:
                g = {**g, "status": "unsupported"}
        checked.append(g)
    return checked


def ground(formula: str, context: str, backend: str, model: str) -> dict:
    struct, symbols = structure_and_holes(formula)
    if backend == "openai":
        raw = call_openai(build_prompt(symbols, context), model).get("groundings", [])
        # keep only requested symbols, in order
        by = {g.get("symbol"): g for g in raw}
        raw = [by.get(s, {"symbol": s, "status": "undefined-in-context"}) for s in symbols]
    else:
        raw = call_stub(symbols, context)["groundings"]
    g = check(raw, context)
    summary = {"symbols": len(symbols),
               "grounded": sum(1 for x in g if x["status"] == "grounded"),
               "undefined_in_context": sum(1 for x in g if x["status"] == "undefined-in-context"),
               "unsupported": sum(1 for x in g if x["status"] == "unsupported")}
    return {"schema": "sfc-symbol-grounding/v0", "backend": backend,
            "structure": struct, "groundings": g, "summary": summary}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--formula", required=True)
    ap.add_argument("--context", required=True, help="file path, or - for stdin")
    ap.add_argument("--backend", choices=["stub", "openai"], default="stub")
    ap.add_argument("--model", default="mark4-70b")
    a = ap.parse_args(argv)
    context = sys.stdin.read() if a.context == "-" else Path(a.context).read_text()
    print(json.dumps(ground(a.formula, context, a.backend, a.model), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
