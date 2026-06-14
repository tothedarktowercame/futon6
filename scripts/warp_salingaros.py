#!/usr/bin/env python3
"""Salingaros aliveness per paper (#2) — from a paper's OWN scope structure.

Salingaros' "life" = organized complexity: many interrelated units, nested across
SCALES, coherently CONNECTED. We read each DP-marked paper's anatomy (golden
marks) and compute a proxy from three structural channels:
  - hierarchy : nesting depth of scopes (units within units = scale structure)
  - multiplicity: how many structural scopes (organized parts)
  - coherence : fraction of symbols actually grounded to a binder (connections;
                ungrounded symbols are dead/disconnected ornament)
  - scale-span: log range of scope sizes (genuine multi-scale, not one scale)

  L = coherence * log(1+nscopes) * (1+depth) * (1+scale_span)   (then 0..1 scaled)

Covers the DP-marked papers (golden set) — a second, independent overlay
alongside the OR-curvature terrain. Aliveness is a STRUCTURAL reading (life of
the exposition), orthogonal to the epistemic tension and the citation curvature.

    warp_salingaros.py -> data/warp/aliveness.json {paper: L}
"""
import json
import math
from pathlib import Path

GOLD = Path("/home/joe/code/futon6/data/showcases/ct-anatomy/golden")
OUT = Path("/home/joe/code/futon6/data/warp/aliveness.json")
SCOPE_KINDS = {"let-binder"}


def is_scope(m):
    return m.get("layer") == "scope" or (m.get("layer") == "dp" and m.get("kind") in SCOPE_KINDS)


def nesting_depth(scopes):
    """max number of scopes strictly containing a scope."""
    best = 0
    for a in scopes:
        d = sum(1 for b in scopes if b is not a and b["start"] <= a["start"]
                and b["end"] >= a["end"] and (b["end"] - b["start"]) > (a["end"] - a["start"]))
        best = max(best, d)
    return best


def aliveness(marks):
    scopes = [m for m in marks if is_scope(m) and m.get("end", 0) > m.get("start", 0)]
    if not scopes:
        return None
    sizes = [m["end"] - m["start"] for m in scopes]
    depth = nesting_depth(scopes) if len(scopes) <= 600 else 0   # cap O(n^2) on huge papers
    scale_span = math.log((max(sizes) + 1) / (min(sizes) + 1) + 1)
    sg = sum(1 for m in marks if m.get("kind") == "symbol-grounded")
    su = sum(1 for m in marks if m.get("kind") == "symbol")
    coherence = sg / (sg + su) if (sg + su) else 0.0
    return coherence * math.log(1 + len(scopes)) * (1 + depth) * (1 + scale_span)


def main():
    raw = {}
    for f in GOLD.glob("fable-*-dp-emacs.json"):
        pid = f.name[len("fable-"):-len("-dp-emacs.json")]
        try:
            L = aliveness(json.load(open(f))["marks"])
        except Exception:
            L = None
        if L is not None:
            raw[pid] = L
    if not raw:
        print("no golden papers found"); return 0
    vals = sorted(raw.values())
    lo, hi = vals[len(vals) // 20], vals[-max(1, len(vals) // 20)]
    out = {p: round(min(1.0, max(0.0, (v - lo) / (hi - lo + 1e-9))), 4) for p, v in raw.items()}
    OUT.write_text(json.dumps({"schema": "aliveness-v1", "n_papers": len(out),
                               "note": "Salingaros structural-life proxy, DP-marked papers",
                               "paper_aliveness": out}))
    top = sorted(out.items(), key=lambda kv: -kv[1])[:6]
    print(f"aliveness for {len(out)} DP-marked papers; top: "
          + ", ".join(f"{p}={v:.2f}" for p, v in top))


if __name__ == "__main__":
    raise SystemExit(main())
