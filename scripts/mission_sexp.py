#!/usr/bin/env python3
# mission_sexp.py — render a mission's nested scope-tree as ersatz-Clojure sexps
# (Joe, 2026-06-08): the lifecycle VERBS are the top forms; a filled phase is an
# evaluated form, an UNFILLED phase a bare form = a hole. Sub-scopes nest; concepts
# are the bound symbols. A mission becomes a partially-written program.
import re, sys
sys.path.insert(0, 'scripts')
from mission_fold import load_sip, load_tree, build, top_sip

PHASES = ["head","identify","map","derive","argue","verify","instantiate","document"]

def slug(t, n=3):
    t = re.sub(r'[^a-z0-9]+','-', t.lower()).strip('-')
    parts = [p for p in t.split('-') if p and p not in ('the','a','of','for','to','as','and','mission')]
    return '-'.join(parts[:n]) or 'scope'

def canon(title):
    u = title.strip().upper()
    return next((v for v in PHASES if u.startswith(v.upper())), None)

def dedup_children(nodes, ids):
    kids = {}
    for nid in ids:
        for c in nodes[nid]['children']:
            kids.setdefault(nodes[c]['title'].strip().lower(), []).append(c)
    return sorted(kids.values(), key=lambda x: -sum(nodes[i]['sub_mass'] for i in x))

def render(nodes, sip, ids, ind):
    own = [f for nid in ids for f in nodes[nid]['fillers']]
    body = ' '.join(slug(c,2) for c in top_sip(own, sip, 6))
    name = slug(nodes[ids[0]]['title'])
    kid_forms = [render(nodes, sip, k, ind+'  ') for k in dedup_children(nodes, ids)]
    if kid_forms:
        head = f"{ind}({name}{(' '+body) if body else ''}"
        return head + "\n" + "\n".join(kid_forms) + ")"
    return f"{ind}({name}{(' '+body) if body else ''})"

def mission_sexp(stem):
    tree,_=load_tree(stem); nodes,roots=build(tree,sip)
    present={}; asides=[]
    for r in roots:
        for c in nodes[r]['children']:
            v = canon(nodes[c]['title']) if nodes[c]['binder']=='eightfold-phase' else None
            if v: present.setdefault(v,[]).append(c)
            elif nodes[c]['binder']=='loose-section': asides.append(c)
    out=[f"({slug(stem,4)}"]
    for v in PHASES:
        if v in present:
            forms=[render(nodes,sip,k,'    ') for k in dedup_children(nodes, present[v])]
            out.append(f"  ({v}\n" + "\n".join(forms) + ")" if forms else f"  ({v})")
        else:
            out.append(f"  ({v})  ; hole")
    for a in asides:
        forms=[render(nodes,sip,k,'    ') for k in dedup_children(nodes,[a])]
        if forms:
            out.append(f"  (aside {slug(nodes[a]['title'],4)}\n" + "\n".join(forms) + ")")
    return "\n".join(out)+")"

sip=load_sip()
for stem in (sys.argv[1:] or ["M-war-machine","M-agency-forum"]):
    print(f"\n;; ===== {stem} =====")
    print(mission_sexp(stem))
