#!/usr/bin/env python3
"""Offline, same-budget comparison of archived and size-scaled S4 selection."""
from __future__ import annotations
import argparse, hashlib, json, math, statistics
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA="futon6/mark8-s4-policy-comparison/v1"; ACCT="futon6-stage-accounting/v1"
def enc(x): return (json.dumps(x,indent=2,sort_keys=True)+"\n").encode()
def digest(x): return hashlib.sha256(enc(x)).hexdigest()
def finite(x):
 if isinstance(x,float) and not math.isfinite(x): raise ValueError("nonfinite input")
 if isinstance(x,dict):
  for v in x.values(): finite(v)
 if isinstance(x,list):
  for v in x: finite(v)
def unique(rows,label):
 out={}
 for x in rows:
  if not isinstance(x,dict) or not isinstance(x.get("id"),str) or not x["id"]: raise ValueError(f"invalid {label} item")
  if x["id"] in out: raise ValueError(f"duplicate {label} id: {x['id']}")
  out[x["id"]]=x
 return out
def semantic(doc): return digest({**doc,"items":sorted(doc["items"],key=lambda x:x["id"])})
def load(path,producer):
 d=json.loads(path.read_bytes()); finite(d)
 if d.get("schema")!=ACCT or d.get("stage")!="S4" or d.get("producer")!=producer: raise ValueError(f"refused {producer} schema/stage/producer")
 if not isinstance(d.get("items"),list): raise ValueError(f"{producer} items must be a list")
 return d,unique(d["items"],producer)
def allocate(counts,budget):
 total=sum(counts.values()); spend=min(total,budget)
 weights={p:min(120,max(12,round(6*math.sqrt(n)))) for p,n in counts.items()}
 remaining=dict(counts); quota={p:0 for p in counts}
 while sum(quota.values())<spend:
  active=[p for p in counts if remaining[p]>0]
  left=spend-sum(quota.values()); sw=sum(weights[p] for p in active)
  shares={p:left*weights[p]/sw for p in active}
  grants={p:min(remaining[p],int(math.floor(shares[p]))) for p in active}
  if not any(grants.values()):
   p=min(active,key=lambda p:(-(shares[p]-math.floor(shares[p])),p)); grants[p]=1
  for p,n in grants.items(): quota[p]+=n; remaining[p]-=n
 return quota,weights,spend
def corr(xs,ys):
 if len(xs)<2 or statistics.pstdev(xs)==0 or statistics.pstdev(ys)==0:return None
 return sum((x-statistics.mean(xs))*(y-statistics.mean(ys)) for x,y in zip(xs,ys))/(len(xs)*statistics.pstdev(xs)*statistics.pstdev(ys))
def summary(values): return {"min":min(values),"median":statistics.median(values),"max":max(values)}
def compare(extract_path,select_path,manifest_path):
 manifest_path=manifest_path.resolve(); run=manifest_path.parent
 expected={"extract":run/"accounting/S4/S4-a001/S4.extract.json","select":run/"accounting/S4/S4-a001/S4.select.json"}
 for name,path in (("extract",extract_path.resolve()),("select",select_path.resolve())):
  if path!=expected[name].resolve(): raise ValueError(f"{name} path mismatch or escape")
 extract,ep=load(extract_path,"extract"); select,sp=load(select_path,"select")
 manifest=json.loads(manifest_path.read_bytes()); finite(manifest)
 papers=manifest.get("papers"); selection=manifest.get("selection")
 if not isinstance(papers,list) or len(papers)!=len(set(papers)) or not all(isinstance(x,str) and x for x in papers): raise ValueError("invalid manifest papers")
 if not isinstance(selection,dict): raise ValueError("missing archived selection policy")
 universe={}; bypaper=defaultdict(list)
 for paper,row in ep.items():
  if row.get("paper")!=paper or row.get("status")!="accepted" or not isinstance(row.get("outputs"),list): raise ValueError("inconsistent extract paper/status/outputs")
  for ident in row["outputs"]:
   if not isinstance(ident,str) or ident in universe: raise ValueError(f"duplicate/invalid extracted id: {ident!r}")
   universe[ident]=paper; bypaper[paper].append(ident)
 if set(ep)!=set(papers): raise ValueError("extract/manifest paper mismatch")
 baseline=set()
 for ident,row in sp.items():
  if ident not in universe: raise ValueError(f"foreign selection id: {ident}")
  if row.get("paper")!=universe[ident]: raise ValueError(f"inconsistent paper id: {ident}")
  if row.get("status") not in {"accepted","deferred"}: raise ValueError(f"invalid selection status: {ident}")
  if row["status"]=="accepted": baseline.add(ident)
 if set(sp)!=set(universe): raise ValueError("selection does not exactly account extracted universe")
 counts={p:len(bypaper[p]) for p in sorted(bypaper)}; quota,weights,spend=allocate(counts,len(baseline))
 counter={x for p in sorted(bypaper) for x in sorted(bypaper[p])[:quota[p]]}
 rows=[]
 for p in sorted(bypaper):
  b=sum(x in baseline for x in bypaper[p]); c=quota[p]
  rows.append({"paper-id":p,"regions":counts[p],"scaled-weight":weights[p],"baseline-selected":b,"counterfactual-selected":c,"delta":c-b})
 def policy_stats(key):
  vals=[r[key] for r in rows]; regions=[r["regions"] for r in rows]
  return {"coverage":summary([v/n for v,n in zip(vals,regions)]),"selected-count":summary(vals),"region-count-correlation":corr(regions,vals)}
 sources=[]
 for label,path,doc in (("extract",extract_path,extract),("select",select_path,select)):
  sources.append({"role":label,"path":str(path.resolve().relative_to(run)),"semantic-sha256":semantic(doc)})
 sources.append({"role":"manifest","path":"run-manifest.json","sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest()})
 name=selection.get("expository-selection") or "archived-observed-selection"
 return {"schema":SCHEMA,"sources":sources,"archived-policy":{"name":name,"evidence":{"expository-cap":selection.get("expository-cap"),"expository-cap-rule":selection.get("expository-cap-rule")}},
  "counterfactual-policy":{"name":"size-scaled-same-budget/v1","weight-formula":"clamp(round(6 * sqrt(region-count)), 12, 120)","global-apportionment":"iterative Hamilton-style proportional allocation by weight, capacity constrained","candidate-tie-break":"exact candidate ID ascending","remainder-tie-break":"fractional remainder descending, exact paper ID ascending"},
  "global":{"eligible":len(universe),"budget":len(baseline),"unspent":len(baseline)-spend,"baseline":{"selected":len(baseline),"unselected":len(universe)-len(baseline)},"counterfactual":{"selected":len(counter),"unselected":len(universe)-len(counter)},"conserved":len(counter)+(len(baseline)-spend)==len(baseline)},
  "selections":{"baseline":sorted(baseline),"counterfactual":sorted(counter)},
  "per-paper":rows,"analytics":{"baseline":policy_stats("baseline-selected"),"counterfactual":policy_stats("counterfactual-selected")}}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--extract",type=Path,required=True);ap.add_argument("--select",type=Path,required=True);ap.add_argument("--manifest",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args();a.out.write_bytes(enc(compare(a.extract,a.select,a.manifest)));return 0
if __name__=="__main__":raise SystemExit(main())
