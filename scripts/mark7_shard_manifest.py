#!/usr/bin/env python3
"""Partition a manifest into N strided shards for data-parallel runs across the Superpod's
GPUs. Strided (ids[k::N]) so each shard gets a representative slice (the citation-ranked
backbone is spread across shards, not piled in shard 0).

Each shard's id-list feeds `linode_stepper.py --ids <shard> --from S1 --to S1` (etc.) with
its own CUDA_VISIBLE_DEVICES (via mfuton-superpod-gpu-policy.sh). Corpus-wide stages
(S2 substrate, S5 comprehension, S8-S12) run once over the full manifest after the relevant
shard phase merges — see handoff-superpod-all.sh's Block-1/Block-2 split.

  futon6/.venv/bin/python scripts/mark7_shard_manifest.py --ids holes/math-ct-full.ids.txt --num-shards 8
"""
import argparse
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="holes/math-ct-full.ids.txt")
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--out", default="holes/shards")
    a = ap.parse_args()
    src = a.ids if os.path.isabs(a.ids) else os.path.join(ROOT, a.ids)
    ids = [l.strip() for l in open(src) if l.strip()]
    outdir = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(outdir, exist_ok=True)
    for k in range(a.num_shards):
        shard = ids[k::a.num_shards]
        p = os.path.join(outdir, f"mark7-shard-{k}.ids.txt")
        open(p, "w").write("\n".join(shard) + "\n")
        print(f"  shard {k}: {len(shard):5d} papers -> {os.path.relpath(p, ROOT)}")
    print(f"{len(ids)} papers -> {a.num_shards} strided shards in {os.path.relpath(outdir, ROOT)}")


if __name__ == "__main__":
    main()
