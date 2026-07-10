#!/usr/bin/env python3
"""E-fold-embed-pipeline Stage A+B → training bundle (laptop/CPU).
Emits to futon6/data/fold-embed/: nodes.jsonl · edges.jsonl · pairs.jsonl · manifest.json
Consumed on the 4-GPU Linode by train_fold_embed.py. Reproducible from substrate-2 + git-sha citations.
"""
import json,re,os,glob,urllib.request,urllib.parse,time,collections
OUT="/home/joe/code/futon6/data/fold-embed"; os.makedirs(OUT,exist_ok=True)
enc=urllib.parse.quote; t0=time.time()
def get(u): return urllib.request.urlopen(u,timeout=120).read().decode()
def pull(t,lim=300000,cache=None):
    if cache and os.path.exists(cache): return open(cache).read()
    b=get(f"http://localhost:7071/api/alpha/hyperedges?type={enc(t)}&limit={lim}")
    if cache: open(cache,"w").write(b)
    return b
def eps(body):  # endpoints lists (dir: dropped)
    return [[x for x in re.findall(r'"([^"]+)"',ep) if not x.startswith("dir:")]
            for ep in re.findall(r':hx/endpoints \[([^\]]*)\]',body)]
def nsof(v):
    x=v.split("/",1)[-1] if re.match(r'^[a-z0-9]+-d/',v) else v
    return x.rsplit("/",1)[0] if "/" in x else x

# ---- graph endpoint (patterns/missions) ----
g=json.loads(get("http://localhost:7070/api/alpha/cascade-real/graph"))
# ---- code graph ----
calls=[q for q in eps(pull("code/v05/calls",cache="/tmp/_calls.edn")) if len(q)>=2]
contains=[q for q in eps(pull("code/v05/contains",cache="/tmp/_contains.edn")) if len(q)>=2]
edits=[q for q in eps(pull("code/v05/edits",cache="/tmp/_edits.edn")) if len(q)>=2]
print(f"pulled calls {len(calls)} contains {len(contains)} edits {len(edits)} ({time.time()-t0:.0f}s)")
# ---- A.2 linkage: mission-doc -> cited commits -> vars ----
cb=pull("code/v05/commit",cache="/tmp/_commit.edn"); shas=set(re.findall(r'code/v05/commit:([0-9a-f]+)',cb)); short={s[:7]:s for s in shas}
c2v=collections.defaultdict(set)
for sha,var in edits: c2v[sha].add(var)
doc2v={}
for d in glob.glob("/home/joe/code/futon*/holes/**/*.md",recursive=True):
    leaf=os.path.basename(d)[:-3]
    try: txt=open(d,errors="ignore").read()
    except: continue
    vs=set()
    for h in set(re.findall(r'\b([0-9a-f]{7,40})\b',txt)):
        f=short.get(h[:7])
        if f: vs|=c2v.get(f,set())
    if vs: doc2v[leaf]=vs
print(f"linked missions (doc→code): {len(doc2v)}")
# ---- nodes ----
nodes={}  # id -> {type,text}
def add(i,t,txt): 
    if i not in nodes: nodes[i]={"type":t,"text":txt}
for a,b in calls: add(a,"var",a.split("/")[-1]); add(b,"var",b.split("/")[-1])
for a,b in contains: add(a,"namespace",a.replace("/"," ")); add(b,"var",b.split("/")[-1])  # FIX 2026-07-01: emit ns nodes so `contains` edges resolve (were 100% dangling -> filtered out)
for e in g["patterns"]["edges"]:
    add(e["pattern"],"pattern",e["pattern"].replace("/"," ")); add(e["mission"],"mission",e["mission"].split("/")[-1].replace("-"," "))
for m in doc2v: nodes.setdefault("mission-doc:"+m,{"type":"mission","text":m.replace("-"," ")})
# ---- edges ----
edges=[]
for a,b in calls: edges.append([a,b,"calls"])
for a,b in contains: edges.append([a,b,"contains"])
for e in g["patterns"]["edges"]: edges.append([e["mission"],e["pattern"],"uses-pattern"])
# mission→var linkage edges (A.2) — key structural signal
link=0
for m,vs in doc2v.items():
    mid="mission-doc:"+m
    for v in vs:
        if v in nodes: edges.append([mid,v,"touches"]); link+=1
# ---- pairs (bulk train) : mission cascade -> used-external-var endpoints + hard negs ----
patt_by_m=collections.defaultdict(list)
for e in g["patterns"]["edges"]: patt_by_m[e["mission"]].append(e["pattern"])
callmap=collections.defaultdict(list)
for a,b in calls: callmap[a].append(b)
allvars=[i for i,d in nodes.items() if d["type"]=="var"]
deg=collections.Counter(v for a,b in calls for v in (a,b))
pairs=[]; rng=__import__("random"); rng.seed(7)
# --- mission-id reconciliation (FIX 2026-07-01) ---
# Cascade patterns are keyed by the CANONICAL pattern-graph id (<repo>-d/mission/<suffix>), but the
# doc-linkage is keyed by the doc-leaf (M-/E-/C-<suffix>). The old `patt_by_m.get(m)` used the leaf key
# -> 0/110 matched -> EVERY cascade was empty -> casc_vec falls back to the global mean -> all three
# ablation arms produce an identical, mission-independent ranking (a null experiment). Reconcile
# leaf->canonical, and only emit pairs that HAVE a cascade: a cascade-less pair trains the scorer against
# the global mean = a popularity prior, the exact failure this experiment exists to beat. The corpus is
# therefore bounded by {missions with a mined cascade} ∩ {missions with >=3 code endpoints} (small today).
# (Checked by library/data-mining/{gates-as-code,smoke-before-the-paid-run} via check_fold_embed_gates.py.)
canon_by_suffix={e["mission"].rsplit("/mission/",1)[-1]:e["mission"]
                 for e in g["patterns"]["edges"] if "/mission/" in e["mission"]}
def norm_leaf(x): return re.sub(r'^[A-Z]{1,4}-','',x).split('.')[0]  # strip M-/E-/C- prefix + sub-doc suffix
cand=[]
for m in sorted(doc2v):
    canon=canon_by_suffix.get(norm_leaf(m)); casc=patt_by_m.get(canon,[]) if canon else []
    if not casc: continue                                    # RESTRICT: no cascade -> not a valid query
    vs=doc2v[m]; own_ns={nsof(v) for v in vs}
    used=collections.Counter()
    for v in vs:
        for cal in callmap.get(v,()):
            if nsof(cal) not in own_ns and cal in nodes: used[cal]+=1
    pos=[v for v,_ in used.most_common(60)]
    if len(pos)<3: continue
    negpool=[v for v,_ in deg.most_common(400) if v not in set(pos) and v in nodes]  # popular near-miss negs
    hard=rng.sample(negpool,min(len(pos)*3,len(negpool)))
    cand.append({"mission":canon,"leaf":m,"cascade":casc,"pos":pos,"hard_neg":hard})
# split over the FILTERED corpus so val/test are non-empty (small corpus -> ~1/5 each, by mission)
for i,p in enumerate(cand):
    p["split"]="test" if i%5==0 else ("val" if i%5==1 else "train"); pairs.append(p)
# ---- write ----
with open(f"{OUT}/nodes.jsonl","w") as f:
    for i,d in nodes.items(): f.write(json.dumps({"id":i,**d})+"\n")
with open(f"{OUT}/edges.jsonl","w") as f:
    for e in edges: f.write(json.dumps(e)+"\n")
with open(f"{OUT}/pairs.jsonl","w") as f:
    for p in pairs: f.write(json.dumps(p)+"\n")
man={"nodes":len(nodes),"edges":len(edges),"mission_link_edges":link,"pairs":len(pairs),
     "splits":collections.Counter(p["split"] for p in pairs),
     "node_types":collections.Counter(d["type"] for d in nodes.values()),
     "edge_rels":collections.Counter(e[2] for e in edges),"built_s":round(time.time()-t0,1)}
json.dump(man,open(f"{OUT}/manifest.json","w"),indent=2,default=int)
print(json.dumps(man,default=int,indent=2))
