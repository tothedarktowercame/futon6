#!/usr/bin/env python3
"""E-fold-embed-pipeline Stage C+D — runs on the 4-GPU Linode (torch + sentence-transformers).
Ablation: text-only (BGE) vs struct-only (GNN) vs hybrid; endpoint-recovery vs popularity/BGE/random.
Authored on laptop (torch absent locally) — reviewed, run on the box after `rsync futon6`.
  python scripts/fold_embed/train_fold_embed.py --data data/fold-embed --mode hybrid --epochs 40
"""
import json,argparse,os,math,random,collections
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

def load(d):
    nodes=[json.loads(l) for l in open(f"{d}/nodes.jsonl")]
    edges=[json.loads(l) for l in open(f"{d}/edges.jsonl")]
    pairs=[json.loads(l) for l in open(f"{d}/pairs.jsonl")]
    return nodes,edges,pairs

def bge_features(nodes, model_name, dev):
    from sentence_transformers import SentenceTransformer
    m=SentenceTransformer(model_name,device=dev)
    txt=[n["text"] or n["id"] for n in nodes]
    X=m.encode(txt,batch_size=512,show_progress_bar=True,convert_to_numpy=True,normalize_embeddings=True)
    return torch.tensor(X,dtype=torch.float32)

class SAGE(nn.Module):
    def __init__(self,din,dh,dout,layers=2):
        super().__init__(); self.proj=nn.Linear(din,dh); dims=[dh]*layers+[dout]
        self.ls=nn.ModuleList([nn.Linear(2*dims[i],dims[i+1]) for i in range(layers)])
    def forward(self,x,ei,N):
        h=F.relu(self.proj(x)); src,dst=ei
        for l in self.ls:
            agg=torch.zeros(N,h.size(1),device=h.device)
            deg=torch.zeros(N,1,device=h.device).index_add_(0,dst,torch.ones(dst.size(0),1,device=h.device)).clamp_min(1)
            agg=agg.index_add_(0,dst,h[src])/deg
            h=F.relu(l(torch.cat([h,agg],1))); h=F.normalize(h,dim=1)
        return h

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="data/fold-embed"); ap.add_argument("--mode",default="hybrid",choices=["text","struct","hybrid"])
    ap.add_argument("--model",default="BAAI/bge-small-en-v1.5"); ap.add_argument("--epochs",type=int,default=40)
    ap.add_argument("--dh",type=int,default=256); ap.add_argument("--dout",type=int,default=128); ap.add_argument("--lr",type=float,default=1e-3)
    a=ap.parse_args(); dev="cuda" if torch.cuda.is_available() else "cpu"; print("device",dev,"mode",a.mode)
    nodes,edges,pairs=load(a.data); N=len(nodes); idx={n["id"]:i for i,n in enumerate(nodes)}
    is_var=torch.tensor([n["type"]=="var" for n in nodes])
    # undirected edge_index (structural)
    e=[[idx[s],idx[t]] for s,t,_ in edges if s in idx and t in idx]; e+=[[b,aa] for aa,b in e]
    ei=torch.tensor(e,dtype=torch.long).t().contiguous().to(dev)
    deg=torch.zeros(N).index_add_(0,ei[1].cpu(),torch.ones(ei.size(1)))  # popularity baseline
    # features
    if a.mode=="struct": X=torch.randn(N,384)*0.1
    else: X=bge_features(nodes,a.model,dev)
    X=X.to(dev); din=X.size(1)
    net=SAGE(din,a.dh,a.dout).to(dev) if a.mode!="text" else None
    def embed():
        if a.mode=="text": return F.normalize(X,dim=1)
        return net(X,ei,N)
    def casc_vec(H,pats):
        ii=[idx[p] for p in pats if p in idx]
        return H[ii].mean(0) if ii else H.mean(0)
    tr=[p for p in pairs if p["split"]=="train"]; te=[p for p in pairs if p["split"]=="test"]
    if net:
        opt=torch.optim.Adam(net.parameters(),lr=a.lr)
        for ep in range(a.epochs):
            net.train(); H=embed(); loss=0; random.shuffle(tr)
            for p in tr:
                pos=[idx[v] for v in p["pos"] if v in idx]; neg=[idx[v] for v in p["hard_neg"] if v in idx]
                if not pos or not neg: continue
                c=casc_vec(H,p["cascade"]); sp=H[pos]@c; sn=H[neg]@c
                loss=loss+F.margin_ranking_loss(sp.unsqueeze(1).expand(-1,len(sn)).reshape(-1),
                     sn.unsqueeze(0).expand(len(sp),-1).reshape(-1),
                     torch.ones(len(sp)*len(sn),device=dev),margin=0.2)
            opt.zero_grad(); loss.backward(); opt.step()
            if ep%10==0: print(f"ep{ep} loss {float(loss):.3f}")
    # ---- eval: endpoint-recovery on held-out test missions ----
    net and net.eval()
    with torch.no_grad(): H=embed()
    varmask=is_var.to(dev); vperm=torch.where(varmask)[0]
    def recall(scorer):
        rr=[]
        for p in te:
            pos={idx[v] for v in p["pos"] if v in idx}
            if not pos: continue
            s=scorer(p)[vperm]; order=vperm[torch.argsort(-s)]
            ranks=[(order==pi).nonzero().item() for pi in pos if pi in set(vperm.tolist())]
            if ranks: rr.append((np.mean([r<20 for r in ranks]),np.mean([1/(r+1) for r in ranks])))
        r=np.array(rr); return (r[:,0].mean(),r[:,1].mean()) if len(r) else (0,0)
    emb_s=lambda p: H@casc_vec(H,p["cascade"])
    pop_s=lambda p: deg.to(dev)
    rng=np.random.default_rng(0); rnd_s=lambda p: torch.tensor(rng.standard_normal(N),dtype=torch.float32,device=dev)
    res={"mode":a.mode}
    for nm,sc in [(a.mode,emb_s),("popularity",pop_s),("random",rnd_s)]:
        rec,mrr=recall(sc); res[nm]={"recall@20":round(float(rec),3),"MRR":round(float(mrr),3)}
    print(json.dumps(res,indent=2)); json.dump(res,open(f"{a.data}/scorecard-{a.mode}.json","w"),indent=2)
    if net: torch.save(H.cpu(),f"{a.data}/embeddings-{a.mode}.pt")
if __name__=="__main__": main()
