#!/usr/bin/env python3
"""Build a reproducible math.CT run manifest for the Linode stepper (S1 input).

No canonical "200" set existed (prior runs were ~5000 mark2 or tiny IATC sets), and
the pre-mortem flagged keyword-selected pools as biased — so we draw an UNBIASED,
deterministic sample of primary math.CT papers from the arXiv manifest, spread
across time (every k-th by creation date), and force-include the papers we already
have IATC graphs for (the "warm subset" — instant S4->S8 + comparability).

Manifest is just arXiv ids (+ metadata); S1 fetches the .tex on the Linode host.

Usage:
  futon6/.venv/bin/python scripts/build_ct_manifest.py [--n 200] [--since 2007-01-01] \
      [--db /home/joe/code/storage/arxiv-manifest/arxiv_manifest.sqlite] \
      [--out holes/math-ct-200.manifest.json]
"""
import argparse
import glob
import json
import os
import sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def warm_ids():
    """arXiv ids we already have IATC graphs for (finals only)."""
    ids = set()
    for f in glob.glob(os.path.join(ROOT, "data/iatc-argument-graphs/**/*.edn"), recursive=True):
        if "/.attempts/" in f or "/by-pid/" in f:
            continue
        stem = os.path.basename(f)[:-4]
        ids.add(stem.replace("__", "/"))   # math__0204218 -> math/0204218
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/home/joe/code/storage/arxiv-manifest/arxiv_manifest.sqlite")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--since", default="2007-01-01")
    ap.add_argument("--out", default="holes/math-ct-200.manifest.json")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    rows = con.execute(
        "select arxiv_id, title, primary_category, created, eprint_url from papers "
        "where primary_category='math.CT' and created >= ? order by created, arxiv_id",
        (args.since,)).fetchall()
    pool = [{"arxiv_id": r[0], "title": (r[1] or "").strip(), "created": r[3]} for r in rows]
    eu_by_id = {r[0]: (r[4] or f"https://arxiv.org/e-print/{r[0]}") for r in rows}
    pool_ids = {p["arxiv_id"] for p in pool}

    warm = sorted(warm_ids() & pool_ids)        # warm papers that are in the math.CT pool
    warm_set = set(warm)
    rest = [p for p in pool if p["arxiv_id"] not in warm_set]

    # deterministic even spread across creation time for the non-warm remainder
    take = max(0, args.n - len(warm))
    step = max(1, len(rest) // take) if take else 1
    spread = rest[::step][:take]

    chosen = []
    for p in pool:
        if p["arxiv_id"] in warm_set or p in spread:
            chosen.append({**p, "warm": p["arxiv_id"] in warm_set})
    # keep deterministic order (by created)
    chosen.sort(key=lambda p: (p["created"], p["arxiv_id"]))

    out = {
        "manifest": "math-ct-stepper",
        "selection": {"primary_category": "math.CT", "since": args.since,
                      "draw": "even-spread-by-created + warm-subset", "target_n": args.n},
        "pool_size": len(pool), "n": len(chosen), "n_warm": len(warm),
        "papers": chosen,
    }
    outp = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(out, open(outp, "w"), indent=1)
    # sibling id-list (one arxiv_id per line) — the input render_gh200.py --list and
    # fetch-arxiv-eprints.py consume on the Linode host (S1).
    # canonical pipeline id is the SAFE form (old-style "math/NNNN" -> "math__NNNN"):
    # slashes break every id->filename construction (marks, candidates, graphs). The
    # real slashed id survives only in the fetch URL. ~2.5% of CT, much more at scale.
    def safe(i):
        return i.replace("/", "__")
    listp = outp.rsplit(".manifest.json", 1)[0] + ".ids.txt"
    open(listp, "w").write("\n".join(safe(p["arxiv_id"]) for p in chosen) + "\n")
    # fetch JSONL ({id, eprint_url}) — fetch-arxiv-eprints.py input (S1); id is the
    # safe form (so the stored eprint filename matches), eprint_url keeps the real id.
    fjsonl = outp.rsplit(".manifest.json", 1)[0] + ".fetch.jsonl"
    with open(fjsonl, "w") as fh:
        for p in chosen:
            fh.write(json.dumps({"id": safe(p["arxiv_id"]),
                                 "eprint_url": eu_by_id[p["arxiv_id"]]}) + "\n")

    print(f"math.CT pool (primary, since {args.since}): {len(pool)}")
    print(f"manifest: {len(chosen)} papers ({len(warm)} warm / already-IATC'd), "
          f"spread {chosen[0]['created'][:7]}..{chosen[-1]['created'][:7]}")
    print(f"wrote {args.out}")
    print("warm subset:", warm[:8], "..." if len(warm) > 8 else "")


if __name__ == "__main__":
    main()
