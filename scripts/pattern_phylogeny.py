#!/usr/bin/env python3
# pattern_phylogeny.py — the PATTERN dual of the mission phylogeny (Joe, 2026-06-08).
# Patterns as the policy layer (Alexander's cascades / pattern-languages), latent in flexiargs filed
# in flat sublibraries. Descent = cross-reference (X cites primitive Y => X descends from Y);
# generativity = in-cite-degree (primitives = the trunk); co-application = HGT roads; COLOUR = sublibrary
# (so cross-cutting cascades show as colour-mixed clusters). Radial tree, trunk = the deep primitives.
import re, glob, math, hashlib, json
from pathlib import Path
from collections import Counter, defaultdict
ROOT=Path("/home/joe/code"); OUT=ROOT/"futon6/data/pattern-phylogeny.html"; EDGES=ROOT/"futon6/data/pattern-phylogeny-edges.json"
fx={}
for f in glob.glob(str(ROOT/'futon*/library/**/*.flexiarg'),recursive=True):
    pr=Path(f).parts
    if 'library' in pr: fx[Path(f).stem]={'lib':pr[pr.index('library')+1],'text':Path(f).read_text(errors='ignore')}
P=[b for b in fx if b.count('-')>=2 and len(b)>=12]
Ps=set(P)
# cross-reference (descent) + in-cite generativity
cites=defaultdict(set); indeg=Counter()
for x in P:
    t=fx[x]['text']
    for y in P:
        if y!=x and y in t: cites[x].add(y); indeg[y]+=1
parent={}; children=defaultdict(list); roots=[]
for x in P:
    cand=max((y for y in cites[x]), key=lambda y:(indeg[y],y), default=None)
    if cand and indeg[cand]>indeg.get(x,0): parent[x]=cand; children[cand].append(x)
    else: parent[x]=None; roots.append(x)
VR='__r__'; children[VR]=sorted(roots,key=lambda x:-indeg[x])
depth={}; leaves={}
def dfs(s,d,seen):
    depth[s]=d; ch=[c for c in children.get(s,[]) if c not in seen]
    if not ch: leaves[s]=1; return 1
    leaves[s]=sum(dfs(c,d+1,seen|{s}) for c in ch); return leaves[s]
dfs(VR,0,set())
CX=CY=1300; RING=145; pos={}
def place(s,a0,a1):
    a=(a0+a1)/2; pos[s]=(CX+depth[s]*RING*math.cos(a),CY+depth[s]*RING*math.sin(a))
    ch=children.get(s,[]); tot=sum(leaves[c] for c in ch) or 1; cur=a0
    for c in ch:
        w=(a1-a0)*leaves[c]/tot; place(c,cur,cur+w); cur+=w
place(VR,-math.pi/2,1.5*math.pi)
def libcol(lib):
    h=int(hashlib.md5(lib.encode()).hexdigest(),16)%360
    return f"hsl({h},62%,58%)"
# co-application (HGT) from missions
paths={p.stem:p for p in ROOT.glob('futon*/holes/**/M-*.md')}
ap={s:{b for b in Ps if b in p.read_text(errors='ignore')} for s,p in paths.items()}
co=Counter()
for ps in ap.values():
    ps=sorted(ps)
    for i in range(len(ps)):
        for j in range(i+1,len(ps)): co[(ps[i],ps[j])]+=1
hgt=[f'<line x1="{pos[a][0]:.0f}" y1="{pos[a][1]:.0f}" x2="{pos[b][0]:.0f}" y2="{pos[b][1]:.0f}" stroke="#c8b58a" stroke-width="{0.3+0.12*min(w,6):.1f}" opacity="{0.03+0.02*min(w,5):.2f}"/>' for (a,b),w in co.items() if a in pos and b in pos]
desc=[f'<line x1="{pos[parent[x]][0]:.0f}" y1="{pos[parent[x]][1]:.0f}" x2="{pos[x][0]:.0f}" y2="{pos[x][1]:.0f}" stroke="#566" stroke-width="0.8" opacity="0.3"/>' for x in P if parent[x] and parent[x] in pos]
nodes=[f'<circle cx="{pos[x][0]:.0f}" cy="{pos[x][1]:.0f}" r="{3+1.8*math.sqrt(indeg[x]):.1f}" fill="{libcol(fx[x]["lib"])}" stroke="#0008" stroke-width="0.4"><title>{x}  [{fx[x]["lib"]}]  cited-by={indeg[x]} co-app-deg={sum(1 for k in co if x in k)}</title></circle>' for x in sorted(P,key=lambda x:indeg[x])]
labels=[f'<text x="{pos[x][0]+6:.0f}" y="{pos[x][1]:.0f}" fill="#eee" font-size="13">{x} <tspan fill="#888" font-size="10">[{fx[x]["lib"]}]</tspan></text>' for x,_ in indeg.most_common(12) if x in pos]
doc=f"""<!doctype html><meta charset=utf-8><title>Pattern phylogeny</title>
<style>body{{margin:0;background:#0a0c11;color:#cdd3df;font:13px sans-serif}}header{{padding:12px 20px}}h1{{font-size:16px;margin:0 0 4px}}p{{margin:0;color:#8b95a7;font-size:12px}}</style>
<header><h1>Pattern phylogeny — the latent policy layer ({len(P)} patterns)</h1>
<p>Descent = cross-reference (toward the primitive trunk); node size = in-cite (primitives biggest); colour =
sublibrary; faint gold = co-application roads. <b>Colour-MIXED clusters = pattern-languages the flat sublibraries split apart.</b>
Trunk primitives: argue-empirically-not-persuasively, stop-the-line, evidence-over-assertion. Zoom in.</p></header>
<svg width="2600" height="2600" viewBox="0 0 2600 2600"><g>{''.join(hgt)}</g><g>{''.join(desc)}</g><g>{''.join(nodes)}</g><g>{''.join(labels)}</g></svg>"""
edge_doc={
    "patterns": sorted(P),
    "descent": [[x,y] for x in sorted(cites) for y in sorted(cites[x])],
    "co_app": [[a,b,w] for (a,b),w in sorted(co.items())],
}
OUT.write_text(doc)
EDGES.write_text(json.dumps(edge_doc, indent=2, sort_keys=True) + "\n")
print(f"wrote {OUT}")
print(f"wrote {EDGES}")
print(f"{len(P)} patterns, {sum(len(v) for v in cites.values())} cascade edges, {len(roots)} roots, max depth {max(depth.values())}")
print("trunk (most-cited primitives):", [(y,indeg[y]) for y,_ in indeg.most_common(6)])
