#!/usr/bin/env python3
"""paper_binding_gradient.py — the Skolem instrument ported to papers.

Per paper: what fraction of grounded math symbols fall inside (a) a
binder span (bind/*, constrain/*: STRICT), (b) only an environment span
(env-tex/* or env/*: WEAK), (c) neither (FLOATING)? The binding gradient
that yesterday measured nLab 18.3% / writeups 74% / proofs 6.0% floating,
now computable per paper — the evaluation metric for the NLP lane.
"""
import bisect
import json
import sys
from pathlib import Path


def spans_of(scopes, prefixes):
    out = []
    for s in scopes:
        t = str(s.get("hx/type", ""))
        if t.startswith(prefixes):
            c = s.get("hx/content") or {}
            p, e = c.get("position"), c.get("end")
            if p is not None and e is not None and e > p:
                out.append((p, e))
    out.sort()
    return out


def covered(pos, spans):
    i = bisect.bisect_right(spans, (pos, float("inf"))) - 1
    while i >= 0:
        p, e = spans[i]
        if p <= pos < e:
            return True
        if e <= pos and p <= pos:
            return False
        i -= 1
        if i >= 0 and spans[i][1] < pos - 500_000:
            break
    return False


def gradient(scopes):
    binders = spans_of(scopes, ("bind/", "constrain/", "quant/", "assume/"))
    envs = spans_of(scopes, ("env-tex/", "env/"))
    strict = weak = floating = 0
    for s in scopes:
        t = str(s.get("hx/type", ""))
        # population: fine-grain math constructs (shared by the Feb run and
        # the fresh suite); envelopes are spans, not point-constructs
        if t.startswith("math/") and t != "math/envelope":
            c = s.get("hx/content") or {}
            pos = c.get("position")
            if pos is None:
                continue
            if covered(pos, binders):
                strict += 1
            elif covered(pos, envs):
                weak += 1
            else:
                floating += 1
    total = strict + weak + floating
    return {"symbols": total, "strict": strict, "weak": weak,
            "floating": floating,
            "floating_frac": round(floating / total, 3) if total else None}


if __name__ == "__main__":
    for path in sys.argv[1:]:
        rec = json.loads(Path(path).read_text())
        g = gradient(rec.get("scopes", []))
        print(f"{rec.get('entity_id', path)}: {g}")
