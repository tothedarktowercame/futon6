#!/usr/bin/env python3
"""ct_fresh_extract.py — the cleanup pass for the ages.

Re-extract scope anatomy for ALL math.CT papers locally: patched eprint
loader (PR-50 fallthrough — recovers the ~39% gzipped-single-TeX class),
TODAY's nlab-wiring detector suite, stage-5 merge parity
(scope + math + math_ast + comment). One JSON per paper under
storage/mark2/ct-fresh-scopes/. CPU multiprocessing; resumable (skips
existing outputs).
"""
import importlib
import importlib.util
import json
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

FUTON6 = Path("/home/joe/code/futon6")
LOADER_TREE = Path("/tmp/futon6-sbs")
EPRINTS = Path("/home/joe/code/storage/futon6/data/arxiv-math-ct-eprints")
INDEX = Path("/home/joe/code/storage/futon6/data/arxiv-math-ct-file-index.jsonl")
OUT = Path("/home/joe/code/storage/mark2/ct-fresh-scopes")
OUT.mkdir(exist_ok=True)

_sj = None
_nw = None


def _init():
    global _sj, _nw
    sys.path.insert(0, str(LOADER_TREE / "scripts"))
    sys.path.insert(0, str(FUTON6 / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "superpod_job", str(LOADER_TREE / "scripts" / "superpod-job.py"))
    _sj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_sj)
    _nw = importlib.import_module("nlab-wiring")


def extract(rec):
    pid = rec["id"]
    safe = rec["safe_id"]
    out = OUT / (safe + ".json")
    if out.exists():
        return ("skip", pid, 0)
    entity_id = "arxiv-" + pid
    try:
        text, meta = _sj._load_eprint_text_for_entity(EPRINTS, entity_id)
        scopes = []
        if text:
            for det in ("detect_scopes", "detect_math_scopes",
                        "detect_math_scopes_ast", "detect_comments"):
                fn = getattr(_nw, det, None)
                if fn:
                    try:
                        scopes.extend(fn(entity_id, text) or [])
                    except Exception as exc:
                        scopes.append({"hx/type": "detector/error",
                                       "detector": det, "error": str(exc)[:200]})
        out.write_text(json.dumps(
            {"entity_id": entity_id, "eprint_status": meta.get("status"),
             "detector": "nlab-wiring@2026-06-12 (scope+math+math_ast+comment)",
             "count": len(scopes),
             "type_counts": dict(Counter(
                 s.get("hx/type", "?") for s in scopes).most_common()),
             "scopes": scopes}))
        return ("ok", pid, len(scopes))
    except Exception as exc:
        out.write_text(json.dumps({"entity_id": entity_id,
                                   "error": str(exc)[:300], "count": -1}))
        return ("err", pid, -1)


def main():
    recs = [json.loads(l) for l in open(INDEX)]
    print(f"{len(recs)} papers; output {OUT}", flush=True)
    t0 = time.time()
    done = ok = err = skip = 0
    with Pool(processes=10, initializer=_init) as pool:
        for status, pid, n in pool.imap_unordered(extract, recs, chunksize=8):
            done += 1
            ok += status == "ok"
            err += status == "err"
            skip += status == "skip"
            if done % 250 == 0:
                rate = done / (time.time() - t0)
                eta = (len(recs) - done) / rate / 60
                print(f"  {done}/{len(recs)} ok={ok} err={err} skip={skip} "
                      f"({rate:.1f}/s, eta {eta:.0f}m)", flush=True)
    print(f"DONE {done} ok={ok} err={err} skip={skip} in {(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
