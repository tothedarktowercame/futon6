#!/usr/bin/env python3
"""code_scope_grain_pilot.py — first-probe for M-differentiable-code's `:scope` grain.

Owner: claude-6 (futon5 M-differentiable-code, E2). Runs in the futon6
M-differentiable-math timeline (claude-2) — escrow-clean, OUT of campaign.

Question (Joe, 2026-06-01): for a code embedding, don't embed whole files (too
coarse — the 115k-line conditioning problem) nor bare functions (too fine /
context-free); embed an OVERLAY — a `:scope` node that carries local context.
Does a scope-grain text window give MORE concern-coherent nearest-neighbours
than a bare-function window? Mirrors claude-2's math BGE first-probe.

Method: parse futon6/src/futon6/*.py with `ast`; for each function/method/class
build two text variants:
  - bare  : "name(signature)"                         (context-free)
  - scope : "module <m>: <module-doc>\n<qualname>(<sig>)\n<docstring>"  (overlay)
Embed both with all-MiniLM-L6-v2 (local cache), cosine kNN, and compare
same-module-in-top-k rate (a cheap coherence proxy — the module is the known
concern label). Higher = the grain groups by concern.
"""
from __future__ import annotations
import ast, json, os, sys, pathlib
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

SRC = pathlib.Path("/home/joe/code/futon6/src/futon6")
OUT = pathlib.Path("/home/joe/code/futon6/resources/differentiable-math/code-scope-probe")
TOPK = 5


def first_lines(s, n):
    if not s:
        return ""
    return " ".join(l.strip() for l in s.strip().splitlines()[:n] if l.strip())


def sig_of(node):
    try:
        a = node.args
        names = [ar.arg for ar in (a.posonlyargs + a.args)]
        if a.vararg:
            names.append("*" + a.vararg.arg)
        names += [ar.arg for ar in a.kwonlyargs]
        if a.kwarg:
            names.append("**" + a.kwarg.arg)
        return ", ".join(names)
    except Exception:
        return ""


def extract():
    nodes = []
    for path in sorted(SRC.glob("*.py")):
        mod = path.stem
        if mod == "__init__":
            continue
        try:
            tree = ast.parse(path.read_text())
        except Exception as e:
            print(f"  skip {mod}: {e}", file=sys.stderr)
            continue
        moddoc = first_lines(ast.get_docstring(tree), 1)
        # qualified-name walk (one level of class nesting is enough here)
        def walk(body, prefix=""):
            for n in body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qual = prefix + n.name
                    doc = first_lines(ast.get_docstring(n), 2)
                    nodes.append(dict(
                        id=f"{mod}:{qual}", module=mod, name=qual,
                        bare=f"{n.name}({sig_of(n)})",
                        ctx=f"{qual}({sig_of(n)})\n{doc}",
                        scope=f"module {mod}: {moddoc}\n{qual}({sig_of(n)})\n{doc}",
                    ))
                elif isinstance(n, ast.ClassDef):
                    cdoc = first_lines(ast.get_docstring(n), 1)
                    nodes.append(dict(
                        id=f"{mod}:{n.name}", module=mod, name=n.name,
                        bare=f"class {n.name}",
                        ctx=f"class {n.name}\n{cdoc}",
                        scope=f"module {mod}: {moddoc}\nclass {n.name}\n{cdoc}",
                    ))
                    walk(n.body, prefix=n.name + ".")
        walk(tree.body)
    return nodes


def topk_same_module_rate(vecs, modules, k):
    sims = vecs @ vecs.T
    np.fill_diagonal(sims, -1.0)
    nn = np.argsort(-sims, axis=1)[:, :k]
    same = 0
    total = 0
    for i in range(len(modules)):
        for j in nn[i]:
            same += int(modules[i] == modules[j])
            total += 1
    return same / total, sims, nn


def main():
    nodes = extract()
    modules = [n["module"] for n in nodes]
    print(f"extracted {len(nodes)} scope nodes from {len(set(modules))} modules")

    from sentence_transformers import SentenceTransformer
    model_name = sys.argv[1] if len(sys.argv) > 1 else "sentence-transformers/all-MiniLM-L6-v2"
    print(f"model = {model_name}")
    model = SentenceTransformer(model_name)

    results = {}
    for variant in ("bare", "ctx", "scope"):
        texts = [n[variant] for n in nodes]
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        vecs = np.asarray(vecs, dtype=np.float32)
        rate, sims, nn = topk_same_module_rate(vecs, modules, TOPK)
        # O4(c) discrimination: off-diagonal cosine spread (collapse => all near 1.0)
        iu = np.triu_indices(len(nodes), k=1)
        off = sims[iu]
        pct = np.percentile(off, [50, 90, 99])
        frac_collapsed = float((off > 0.95).mean())
        results[variant] = dict(same_module_top5=rate, sims=sims, nn=nn,
                                cos_med=float(pct[0]), cos_p90=float(pct[1]),
                                cos_p99=float(pct[2]), frac_gt_0p95=frac_collapsed)
        print(f"[{variant:5s}] same-module-top{TOPK}={rate:.3f}  "
              f"cos median/p90/p99 = {pct[0]:.3f}/{pct[1]:.3f}/{pct[2]:.3f}  "
              f"frac>0.95 = {frac_collapsed:.4f}")

    # qualitative: a few probes from distinct concerns, scope variant
    sims = results["scope"]["sims"]
    nn = results["scope"]["nn"]
    probes = ["latex_sexp", "faiss_index", "symbol_grounding", "stackexchange",
              "theorem_extraction"]
    print("\n--- scope-grain nearest neighbours (qualitative) ---")
    for pm in probes:
        idxs = [i for i, n in enumerate(nodes) if n["module"] == pm]
        if not idxs:
            continue
        i = idxs[0]
        print(f"\n[{nodes[i]['id']}]  ->")
        for j in nn[i]:
            print(f"    {sims[i, j]:.3f}  {nodes[j]['id']}")

    # cross-module bridges under scope grain (the 'ameliorate here' analogue):
    s = results["scope"]["sims"].copy()
    cross_best = []
    for i in range(len(nodes)):
        order = np.argsort(-s[i])
        for j in order:
            if j != i and modules[i] != modules[j]:
                cross_best.append((s[i, j], nodes[i]["id"], nodes[j]["id"]))
                break
    cross_best.sort(reverse=True)
    print("\n--- strongest cross-module scope neighbours (semantic bridges) ---")
    for sim, a, b in cross_best[:8]:
        print(f"    {sim:.3f}  {a}  <->  {b}")

    OUT.mkdir(parents=True, exist_ok=True)
    summary = dict(
        n_nodes=len(nodes), n_modules=len(set(modules)),
        same_module_top5_bare=results["bare"]["same_module_top5"],
        same_module_top5_ctx=results["ctx"]["same_module_top5"],
        same_module_top5_scope=results["scope"]["same_module_top5"],
        top_cross_module=[[float(s), a, b] for s, a, b in cross_best[:20]],
    )
    (OUT / "code-scope-grain-result.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT/'code-scope-grain-result.json'}")


if __name__ == "__main__":
    main()
