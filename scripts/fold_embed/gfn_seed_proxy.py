"""Gold-coverage proxy for the GFN seed — the steep 0-sorry reward adapter.

Selection = a set of endpoint-pool indices; coverage = fraction of the current
mission's GOLD endpoint set selected; reward = eps + exp(beta * coverage)
(gfn_gold_loader.reward_for_coverage — range ~403x, vs the flat 2.6x that
failed to concentrate TB in contest v1). GOLD is module state set by the
trainer per mission, the s4 cascade_proxy.MOVES pattern.
"""
import torch

from gflownet.proxy.base import Proxy
from gfn_gold_loader import reward_for_coverage

# set by the trainer before each mission's agent build
GOLD: set = set()


def selection_from_proxy_state(state: dict) -> list:
    """Decode the Choices/SetFix composite proxy format (int-keyed substates,
    tensor([opt_index+1])) into pool indices. Ported from s4/cascade_proxy.py."""
    out = []
    for k, v in state.items():
        if isinstance(k, int) and state.get("_dones", [])[k]:
            idx = int(v[0]) - 1
            if idx >= 0:
                out.append(idx)
    return out


class FoldGoldProxy(Proxy):
    def __init__(self, **kwargs):
        kwargs.setdefault("reward_min", 1e-6)
        kwargs.setdefault("do_clip_rewards", True)
        super().__init__(**kwargs)

    def setup(self, env=None):
        pass

    def __call__(self, states) -> torch.Tensor:
        vals = []
        for st in states:
            sel = set(selection_from_proxy_state(st))
            c = len(sel & GOLD) / len(GOLD) if GOLD else 0.0
            vals.append(reward_for_coverage(c))
        return torch.tensor(vals, dtype=self.float, device=self.device)
