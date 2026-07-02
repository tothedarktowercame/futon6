#!/usr/bin/env python3
"""E-fold-embed-pipeline G.1 — seed the fold-GFN LOCALLY on the A-next gold corpus (n=10, CPU).
Verdict: does the TB sampler concentrate on gold endpoint sets on honest data?
Run: /home/joe/code/gflownet/.venv/bin/python gfn_seed_v0.py [--selfcheck] [--steps 1200] [--reduced]
"""
import sys, os, re, json, glob, argparse, collections
sys.path.insert(0, "/home/joe/code/gflownet")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SEED = 20260702; BETA = 6.0
LAB = "/home/joe/code/futon2/holes/labs"
GOLD10 = ["autoclock-in","invariant-queue-unstuck","a-sorry-enterprise","agency-rebuild","f6-ingest",
          "pattern-ingest","patterns-done-right","single-entry-point","state-snapshot-witness","stepper-calibration"]

def load_corpus():
    # CANONICAL loader = claude-11's proper bb->EDN parse (my regex truncated autoclock-in's want-ref
    # at the "]" inside "{missions [...]}"; cross-agent review caught it). Reuse the cached corpus.
    GC = "/home/joe/code/futon6/data/fold-embed-gfn/gold-corpus.json"
    d = json.load(open(GC))
    return {m: list(dict.fromkeys(d[m]["refs"])) for m in GOLD10}

def build_pool(miss, reduced_for=None):
    if reduced_for is None:
        pool = list(dict.fromkeys([r for rs in miss.values() for r in rs]))
    else:  # fallback rung: own gold + 2 sibling missions
        sibs = [m for m in GOLD10 if m != reduced_for][:2]
        pool = list(dict.fromkeys(miss[reduced_for] + [r for s in sibs for r in miss[s]]))
    return pool

def selfcheck(miss):
    pool = build_pool(miss)
    shared = collections.Counter(r for rs in miss.values() for r in rs)
    multi = {r:c for r,c in shared.items() if c > 1}
    print(f"[selfcheck] missions {len(miss)}  pool {len(pool)}")
    for m in GOLD10: print(f"   {m:24} gold={len(miss[m])}")
    print(f"   reward range: exp(BETA*0)={1.0}  ..  exp(BETA*1)={__import__('math').exp(BETA):.1f}  (ratio {__import__('math').exp(BETA):.0f}x, >=100x OK)")
    print(f"   multi-mission shared refs: {len(multi)}  e.g. {list(multi.items())[:3]}")
    assert all(len(miss[m])>=3 for m in GOLD10), "a gold set <3"
    assert len(pool) >= 40, "pool too small"
    print("[selfcheck] PASS")

def run(miss, steps, reduced=False):
    import sorry_proxy
    from gflownet.utils.common import gflownet_from_config
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    import torch, math
    def make_config(n_options, k):
        GlobalHydra.instance().clear()
        initialize_config_dir(config_dir="/home/joe/code/gflownet/config", version_base="1.1")
        return compose(config_name="tests", overrides=[
            "env=choices", f"env.n_options={n_options}", f"env.max_selection={k}",
            "env.with_replacement=False", "env.can_select_fewer_than_max=False",
            "proxy=uniform", "proxy._target_=sorry_proxy.SorryProxy",
            "gflownet=trajectorybalance", f"gflownet.optimizer.n_train_steps={steps}",
            "policy.forward.n_hid=64","policy.forward.n_layers=2",
            "evaluator.first_it=False","evaluator.period=-1","logger.do.online=False",
            f"seed={SEED}","device=cpu"])
    def probe(gfn, gold_idx, K=128):
        batch,_ = gfn.sample_batch(n_forward=K, train=False)
        sels = [tuple(sorted(sorry_proxy.selection_from_proxy_state(gfn.env.states2proxy([st])[0])))
                for st in batch.get_terminating_states()]
        exact = sum(1 for s in sels if set(s)==gold_idx)/max(len(sels),1)
        cov = sum(len(set(s)&gold_idx)/max(len(gold_idx),1) for s in sels)/max(len(sels),1)
        distinct = len(set(sels))/max(len(sels),1)
        return {"P_exact":round(exact,3),"mean_cov":round(cov,3),"distinct_frac":round(distinct,3),"K":len(sels)}
    verdicts={}
    for m in GOLD10:
        pool = build_pool(miss, reduced_for=m if reduced else None)
        pi={r:i for i,r in enumerate(pool)}; gold_idx=set(pi[r] for r in miss[m]); k=len(gold_idx)
        sorry_proxy.GOLD_IDX=gold_idx; sorry_proxy.BETA=BETA
        cfg=make_config(len(pool),k); gfn=gflownet_from_config(cfg)
        pre=probe(gfn,gold_idx)
        gfn.train()
        post=probe(gfn,gold_idx)
        verdicts[m]={"pool":len(pool),"k":k,"untrained":pre,"trained":post,
                     "cov_lift":round(post["mean_cov"]-pre["mean_cov"],3)}
        print(f"  {m:24} pool={len(pool)} k={k}  cov {pre['mean_cov']}->{post['mean_cov']}  Pexact {pre['P_exact']}->{post['P_exact']}  distinct {pre['distinct_frac']}->{post['distinct_frac']}", flush=True)
    return verdicts

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--selfcheck",action="store_true")
    ap.add_argument("--steps",type=int,default=1200); ap.add_argument("--reduced",action="store_true")
    a=ap.parse_args(); miss=load_corpus()
    if a.selfcheck: selfcheck(miss); sys.exit(0)
    selfcheck(miss)
    v=run(miss,a.steps,a.reduced)
    import math
    out={"seed":SEED,"beta":BETA,"reward_range":[1.0,round(math.exp(BETA),1)],"steps":a.steps,
         "reduced_pool":a.reduced,"n_missions":len(v),
         "mean_cov_lift":round(sum(x["cov_lift"] for x in v.values())/len(v),3),
         "n_with_positive_lift":sum(1 for x in v.values() if x["cov_lift"]>0.05),
         "verdicts":v,"repro":"gflownet/.venv/bin/python futon6/scripts/fold_embed/gfn_seed_v0.py --steps %d%s"%(a.steps," --reduced" if a.reduced else "")}
    od="/home/joe/code/futon6/data/fold-embed-gfn"; os.makedirs(od,exist_ok=True)
    rung=("reduced" if a.reduced else "full"); json.dump(out,open(f"{od}/gfn-seed-verdicts-{rung}.json","w"),indent=2)
    print("\nSUMMARY:",json.dumps({k:out[k] for k in ["mean_cov_lift","n_with_positive_lift","reward_range"]}))
