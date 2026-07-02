"""SorryProxy — steep 0-sorry-discharge reward for the fold-GFN seed (E-fold-embed-pipeline, G.1).
reward(selection) = exp(BETA * coverage), coverage = |selected ∩ gold| / |gold|.
Module globals GOLD_IDX / BETA set by gfn_seed_v0.py before gflownet_from_config."""
import torch, math
from gflownet.proxy.base import Proxy

GOLD_IDX = set()
BETA = 6.0

def selection_from_proxy_state(state):
    # exact copy of the working s4 cascade_proxy decoder
    out = []
    for k, v in state.items():
        if isinstance(k, int) and state.get("_dones", [])[k]:
            idx = int(v[0]) - 1
            if idx >= 0:
                out.append(idx)
    return out

def coverage(sel):
    if not GOLD_IDX: return 0.0
    return len(set(sel) & GOLD_IDX) / len(GOLD_IDX)

class SorryProxy(Proxy):
    def __init__(self, **kw):
        kw.setdefault("reward_min", 1e-6); kw.setdefault("do_clip_rewards", True)
        super().__init__(**kw)
    def setup(self, env=None): pass
    def __call__(self, states):
        vals = [math.exp(BETA * coverage(selection_from_proxy_state(st))) for st in states]
        return torch.tensor(vals, dtype=self.float, device=self.device)
