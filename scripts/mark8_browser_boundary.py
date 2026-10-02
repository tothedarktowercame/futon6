#!/usr/bin/env python3
"""Freeze an explicit, read-only structural-browser boundary for a Mark7 run."""
from __future__ import annotations
import argparse,hashlib,json,math
from collections import Counter
from pathlib import Path
from collections.abc import Mapping,Sequence
from edn_format import Keyword,loads

SCHEMA="futon6/mark8-browser-boundary/v1"; ACCT="futon6-stage-accounting/v1"; MAX_FILES=20000; MAX_BYTES=300_000_000; MAX_OUTPUT_BYTES=15_000_000
def enc(x):return (json.dumps(x,indent=2,sort_keys=True)+"\n").encode()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def k(name):return Keyword(name)
def finite(x):
 if isinstance(x,float) and not math.isfinite(x):raise ValueError("nonfinite JSON")
 if isinstance(x,dict):
  for v in x.values():finite(v)
 if isinstance(x,list):
  for v in x:finite(v)
def inside(run,path,label):
 if path.is_absolute(): raise ValueError(f"absolute {label} path refused")
 if ".." in path.parts:raise ValueError(f"{label} path escape refused")
 target=(run/path).resolve()
 try:target.relative_to(run.resolve())
 except ValueError:raise ValueError(f"{label} symlink escape refused")
 if not target.is_file() or target.stat().st_size==0:raise ValueError(f"missing/empty/non-regular {label}: {path}")
 return target
def accounting(run,path,stage):
 p=inside(run,path,"accounting");d=json.loads(p.read_bytes());finite(d)
 if d.get("schema")!=ACCT or d.get("stage")!=stage or d.get("producer")!="loop":raise ValueError("bad accounting schema/stage/producer")
 if d.get("invocation")!=f"{stage}-a001":raise ValueError("bad accounting invocation")
 seen={}
 for row in d.get("items",[]):
  ident=row.get("id")
  if not isinstance(ident,str) or not ident or ident in seen:raise ValueError("duplicate/invalid accounting item")
  seen[ident]=row
 expected=d.get("expected");counts=d.get("counts")
 if not isinstance(expected,list) or len(expected)!=len(set(expected)) or not all(isinstance(x,str) and x for x in expected) or set(expected)!=set(seen):raise ValueError("accounting expected identities contradict items")
 vocab={"accepted","rejected","errored","deferred","expected","unaccounted"}
 if not isinstance(counts,dict) or set(counts)!=vocab or not all(isinstance(v,int) and not isinstance(v,bool) and v>=0 for v in counts.values()):raise ValueError("invalid accounting counts")
 statuses=Counter(x.get("status") for x in seen.values())
 if any(x not in {"accepted","rejected","errored","deferred"} for x in statuses):raise ValueError("invalid accounting status")
 actual={x:statuses[x] for x in ("accepted","rejected","errored","deferred")};actual.update(expected=len(expected),unaccounted=0)
 if counts!=actual:raise ValueError("accounting counts contradict items/expected")
 return p,d,seen
def add_file(files,seen,run,rel,stage,role,paper,item):
 if rel in seen:raise ValueError(f"duplicate published path: {rel}")
 p=inside(run,Path(rel),role);seen.add(rel)
 files.append({"path":Path(rel).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p),"producer-stage":stage,"role":role,"paper-id":paper,"item-id":item})
 return p
def producer_keyword(value,label):
 if not isinstance(value,Keyword) or not str(value)[1:]:raise ValueError(f"invalid producer-native {label}")
 return str(value)
def build(run,s3_path,s4_path,manifest_path):
 run=run.resolve();mp=inside(run,manifest_path,"manifest");manifest=json.loads(mp.read_bytes());finite(manifest)
 if manifest.get("schema-version")!=1 or isinstance(manifest.get("schema-version"),bool):raise ValueError("bad manifest schema-version")
 s3p,s3,s3rows=accounting(run,s3_path,"S3");s4p,s4,s4rows=accounting(run,s4_path,"S4")
 files=[];seen=set();entities=[];relations=[]
 for ident,row in sorted(s3rows.items()):
  if row.get("status")!="accepted":continue
  paper=row.get("paper"); arts=row.get("artifacts")
  if not isinstance(paper,str) or not isinstance(arts,list):raise ValueError("inconsistent S3 item")
  primary=[x for x in arts if isinstance(x,str) and x.endswith(".edn") and not x.endswith(".rung2.edn")]
  if len(primary)!=1:raise ValueError("accepted S3 item lacks unique primary graph")
  gp=add_file(files,seen,run,primary[0],"S3","accepted-graph",paper,ident);g=loads(gp.read_text())
  passage=g.get(k("passage/id"))
  if g.get(k("paper/id"))!=paper or not isinstance(passage,str) or not passage or not isinstance(g.get(k("nodes")),Sequence):raise ValueError("invalid graph schema or identity")
  nodes=[]
  for node in g[k("nodes")]:
   raw=node.get(k("id"));nid=producer_keyword(raw,"node id")
   if not nid or nid in nodes:raise ValueError("duplicate/empty graph node id")
   nodes.append(nid)
  sp=f"artifacts/steps/{ident}.steps.json";step_path=add_file(files,seen,run,sp,"S3","derived-steps",paper,ident);sd=json.loads(step_path.read_bytes());finite(sd)
  if sd.get("paper_id")!=paper or not isinstance(sd.get("steps"),list):raise ValueError("invalid steps schema or identity")
  steps=[]
  for step in sd["steps"]:
   sid=step.get("id")
   if not isinstance(sid,str) or not sid or sid in steps:raise ValueError("duplicate/empty step id")
   steps.append(sid)
  warrants=set()
  for edge in g.get(k("edges"),[]):
   if k("warrant") not in edge:continue
   warrant=edge[k("warrant")]
   if not isinstance(warrant,Mapping) or not warrant:raise ValueError("invalid producer-native warrant map")
   warrants.add(producer_keyword(warrant.get(k("kind")),"warrant kind"))
  warrants=sorted(warrants)
  entities.append({"id":f"item:{ident}","kind":"accepted-proof","item-id":ident,"paper-id":paper,"passage-id":passage,"node-ids":nodes,"step-ids":steps,"warrant-kinds":warrants})
  relations.append({"from":f"paper:{paper}","to":f"item:{ident}","kind":"has-accepted-proof"})
 for ident,row in sorted(s4rows.items()):
  if row.get("status")!="accepted":continue
  paper=row.get("paper");arts=row.get("artifacts")
  if not isinstance(arts,list) or len(arts)!=1:raise ValueError("accepted S4 item lacks unique artifact")
  ep=add_file(files,seen,run,arts[0],"S4","accepted-expository",paper,ident);d=loads(ep.read_text())
  if d.get(k("paper/id"))!=paper or d.get(k("passage/id"))!=ident or not isinstance(d.get(k("scopes")),Sequence):raise ValueError("invalid expository schema or identity")
  entities.append({"id":f"item:{ident}","kind":"accepted-expository","item-id":ident,"paper-id":paper,"scope-count":len(d[k("scopes")])});relations.append({"from":f"paper:{paper}","to":f"item:{ident}","kind":"has-accepted-expository"})
 papers=sorted({x["paper-id"] for x in entities});entities.extend({"id":f"paper:{p}","kind":"paper","paper-id":p} for p in papers)
 ids=[x["id"] for x in entities]
 if len(ids)!=len(set(ids)):raise ValueError("duplicate entity identity")
 known=set(ids)
 if any(r["from"] not in known or r["to"] not in known for r in relations):raise ValueError("dangling relationship endpoint")
 total=sum(x["bytes"] for x in files)
 if len(files)>MAX_FILES or total>MAX_BYTES:raise ValueError("browser boundary size/count budget exceeded")
 sources=[{"path":Path(x).as_posix(),"sha256":sha(inside(run,Path(x),"source"))} for x in (manifest_path,s3_path,s4_path)]
 result={"schema":SCHEMA,"sources":sources,"files":files,"entities":sorted(entities,key=lambda x:x["id"]),"relationships":sorted(relations,key=lambda x:(x["from"],x["kind"],x["to"])),"skipped-roles":[{"role":"rung3","reason":"paper aggregate has no exact accepted-item identity"},{"role":"paper-graphs","reason":"paper aggregate, outside accepted-item boundary"},{"role":"embeddings","reason":"no exact accepted-item source identity"},{"role":"citations","reason":"unpinned citation substrate absent"}],"summary":{"files":len(files),"bytes":total,"entities":len(entities),"relationships":len(relations),"papers":len(papers),"max-files":MAX_FILES,"max-bytes":MAX_BYTES,"max-output-bytes":MAX_OUTPUT_BYTES}}
 if len(enc(result))>MAX_OUTPUT_BYTES:raise ValueError("encoded browser boundary output budget exceeded")
 return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--run-root",type=Path,required=True);ap.add_argument("--s3-accounting",type=Path,required=True);ap.add_argument("--s4-accounting",type=Path,required=True);ap.add_argument("--manifest",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args();a.out.write_bytes(enc(build(a.run_root,a.s3_accounting,a.s4_accounting,a.manifest)));return 0
if __name__=="__main__":raise SystemExit(main())
