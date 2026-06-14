#!/usr/bin/env python3
"""Cross-MSC pre-superpod validation (Joe): does the classical pipeline generalize
OFF math.CT? Samples papers across diverse MSC classes from the local mark2 inbox
(date-sorted, all-of-math) and runs the detector + checker per class -> a per-MSC
coverage/wf table. This is the demo for Rob (who has all papers for the full run).

    warp_crossmsc_demo.py [batch.tar.gz] [K-per-class]
"""
import io
import json
import re
import sys
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dp_paper_view as dpv
import check_invariants as ci

BATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "code/storage/mark2/inbox/batch-007.tar.gz"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 5
TARGETS = ["math.AG", "math.NT", "math.CO", "math.AP", "math.PR",
           "math.DG", "math.RT", "math.GT", "math.QA", "math.LO"]


def base_of(fname):                       # eprints/<id>v<ver>.tar.gz -> <id>
    return re.sub(r"v\d+\.tar\.gz$", "", fname.split("/")[-1])


def main():
    # base_id (no version) -> primary category
    cat = {}
    with tarfile.open(BATCH) as t:
        jf = t.extractfile([m for m in t.getmembers() if m.name.endswith(".jsonl")][0])
        for line in io.TextIOWrapper(jf, encoding="utf-8", errors="replace"):
            try:
                d = json.loads(line)
                cat[d["base_id"]] = d.get("primary_category")
            except Exception:
                pass
    # eprint members keyed by base
    with tarfile.open(BATCH) as t:
        epmem = {base_of(m.name): m.name for m in t.getmembers()
                 if "/eprints/" in m.name and m.name.endswith(".tar.gz")}

    def catof(base):                      # match eprint base to a jsonl category
        for cand in (base, base.replace("__", "/"), "math/" + base):
            if cand in cat:
                return cat[cand]
        return None

    by_cat = defaultdict(list)
    for base, member in epmem.items():
        c = catof(base)
        if c in TARGETS:
            by_cat[c].append((base, member))

    tmp = Path(tempfile.mkdtemp(prefix="crossmsc-"))
    dpv.EPRINTS = tmp                     # point the reader at our temp eprint dir
    rows = []
    with tarfile.open(BATCH) as t:
        for c in TARGETS:
            done = 0
            for base, member in by_cat.get(c, []):
                if done >= K:
                    break
                pid = base.replace("/", "__")
                try:
                    data = t.extractfile(member).read()
                    (tmp / f"{pid}.tar.gz").write_bytes(data)
                    built = dpv.build(pid, with_binders=True, with_scopes=True,
                                      with_ca=True, with_xref=True)
                    rep = ci.check_paper(pid, {"text": built["text"], "marks": built["marks"]})
                    cov = rep["coverage"]
                    rows.append((c, pid, cov["symbol_grounded"], cov["symbol_tagged"],
                                 cov["math_coverage"], cov["wellformed_errors"], cov["symbols"]))
                    done += 1
                except Exception as ex:
                    rows.append((c, base, None, None, None, None, str(ex)[:40]))
    # per-class aggregate
    print(f"=== cross-MSC validation ({BATCH.name}, K={K}/class) ===")
    print(f"{'class':9} {'n':>2} {'grounded':>9} {'tagged':>7} {'math':>6} {'wf-err':>7}")
    agg = defaultdict(list)
    for c, pid, g, tg, mc, wf, sy in rows:
        if g is not None:
            agg[c].append((g, tg, mc, wf, sy))
    import statistics as st
    for c in TARGETS:
        a = agg.get(c, [])
        if a:
            print(f"{c:9} {len(a):>2} {st.fmean(x[0] for x in a):>8.0%} "
                  f"{st.fmean(x[1] for x in a):>6.0%} {st.fmean(x[2] for x in a):>5.0%} "
                  f"{sum(x[3] for x in a):>7}")
        else:
            print(f"{c:9}  0   (no papers sampled / all errored)")
    okct = sum(len(a) for a in agg.values())
    print(f"\nsampled+marked {okct} papers across {len(agg)} non-CT MSC classes")
    json.dump({"rows": [list(r) for r in rows]}, open("/tmp/crossmsc-result.json", "w"))


if __name__ == "__main__":
    raise SystemExit(main())
