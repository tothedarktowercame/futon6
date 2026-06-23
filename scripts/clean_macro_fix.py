#!/usr/bin/env python3
"""Macro cleanup (mark5 D1/Diagnostic-2): the 70B over-tags the macro (defaults
construct-exploit-discharge regardless of method). Derive the macro DETERMINISTICALLY
from the box method-composition instead, against the grown macro vocab (+transport-
symmetry, +reduce-to-known, +local-to-global-glue). Re-checks macro-entropy on the 102
mark5 CLeans and emits an old-vs-new side-by-side HTML.

  futon6/.venv/bin/python scripts/clean_macro_fix.py
"""
import glob
import json
import math
import os
import re
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANS = os.path.join(ROOT, "data/mark5-ct100-run/holes/clean-ct200")
EMB = os.path.join(ROOT, "data/mark5-ct100-run/data/showcases/clean-ct200-demo/clean-embed.json")
OUT = os.path.join(ROOT, "data/showcases/macro-fix-comparison.html")

# dominant-method -> macro-shape (overrides for the structural methods first)
DOM_MAP = {
    "construct-auxiliary-object": "construct-exploit-discharge",
    "reduce-to-known-result": "reduce-to-known",
    "transport-along-symmetry": "transport-symmetry",
    "compute-invariant": "count-invariant-obstruct",
    "count-by-decomposition": "count-invariant-obstruct",
    "local-to-global": "local-to-global-glue",
    "cover-and-estimate": "cover-estimate",
    "estimate-by-bounding": "cover-estimate",
    "quotient-by-irrelevance": "construct-exploit-discharge",
}


def derive_macro(methods):
    """Macro from method composition. Structural methods (contradiction/induction) win;
    else the dominant method's shape."""
    s = set(methods)
    if "argue-by-contradiction" in s:
        return "contradiction-reduce"
    if "induct-up-a-tower" in s:
        return "induct-tower"
    if not methods:
        return "construct-exploit-discharge"
    dom = Counter(methods).most_common(1)[0][0]
    return DOM_MAP.get(dom, "construct-exploit-discharge")


def entropy(macros):
    c = Counter(macros)
    n = len(macros)
    H = -sum((v / n) * math.log2(v / n) for v in c.values())
    return (H / math.log2(len(c))) if len(c) > 1 else 0.0, dict(c)


def load():
    emb = json.load(open(EMB))
    old = {p: m for p, m in zip(emb["ids"], emb["macros"])}
    rows = []
    for f in sorted(glob.glob(os.path.join(CLEANS, "*.clean.edn"))):
        pid = os.path.basename(f)[:-len(".clean.edn")]
        boxes = re.findall(r"\{:id :[a-z0-9-]+ :method :([a-z0-9-]+)\s+:text \"([^\"]{0,80})", open(f).read())
        meths = [m for m, _ in boxes]
        if not meths:
            continue
        rows.append({"pid": pid, "methods": meths, "snippet": boxes[0][1] if boxes else "",
                     "old": old.get(pid, "?"), "new": derive_macro(meths)})
    return rows


def write_html(rows):
    old_e, old_d = entropy([r["old"] for r in rows])
    new_e, new_d = entropy([r["new"] for r in rows])
    changed = [r for r in rows if r["new"] != r["old"]]
    same = [r for r in rows if r["new"] == r["old"]]
    sel = changed[:14] + same[:4]
    css = ("body{font:14px/1.5 Georgia,serif;margin:24px;background:#f7f5ef;color:#1a1a1a}"
           "table{border-collapse:collapse;width:100%;margin-top:12px}"
           "td,th{border:1px solid #ddd6c8;padding:6px 9px;vertical-align:top;font-size:12.5px}"
           "th{background:#efe9dd;text-align:left}.old{color:#a11;background:#fbe9e9;font-weight:600}"
           ".new{color:#1d7a3a;background:#e7f6ec;font-weight:600}.m{font:11px ui-monospace,monospace;color:#444}"
           ".s{color:#555;font-style:italic}code{background:#ece8dd;padding:1px 4px;border-radius:3px}")
    h = [f"<!doctype html><meta charset=utf-8><title>Macro fix — old vs new</title><style>{css}</style>",
         "<h1>Macro cleanup — 70B over-tag (old) vs method-derived (new)</h1>",
         f"<p>macro-entropy(norm): <span class=old>old {old_e:.2f}</span> → "
         f"<span class=new>new {new_e:.2f}</span> (floor 0.5). "
         f"{len(changed)}/{len(rows)} proofs re-macro'd.</p>",
         f"<p class=m>old dist {old_d}<br>new dist {new_d}</p>",
         "<table><tr><th>paper</th><th>box methods</th><th>OLD macro (70B)</th>"
         "<th>NEW macro (derived)</th><th>first box</th></tr>"]
    for r in sel:
        mc = Counter(r["methods"])
        meths = " · ".join(f"{m}×{n}" for m, n in mc.most_common())
        h.append(f"<tr><td>{r['pid']}</td><td class=m>{meths}</td>"
                 f"<td class=old>{r['old']}</td><td class=new>{r['new']}</td>"
                 f"<td class=s>{r['snippet']}</td></tr>")
    h.append("</table>")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(h))
    return old_e, new_e, old_d, new_d, len(changed)


if __name__ == "__main__":
    rows = load()
    oe, ne, od, nd, nch = write_html(rows)
    print(f"proofs: {len(rows)}")
    print(f"OLD macro-entropy(norm): {oe:.2f}   dist={od}")
    print(f"NEW macro-entropy(norm): {ne:.2f}   dist={nd}")
    print(f"re-macro'd: {nch}/{len(rows)} ; floor 0.5 -> {'PASS' if ne >= 0.5 else 'still below'}")
    print(f"wrote {os.path.relpath(OUT, ROOT)}")
