#!/usr/bin/env python3
"""ct_nlp_side_by_side.py — February's detector vs today's, same TeX.

The CT handoff's scopes were produced by nlab-wiring.detect_scopes as of
2026-02-20; yesterday's prep work upgraded the detector lane months past
that. This harness re-runs TODAY's detectors on the SAME papers' local
eprints and renders the disagreement (combining-methods-as-diagnostic).

Sample: stratified ~40 across the corpus census strata
(high / mid / small / flat / zero-scope).
Output: data/ct-nlp-sbs.json (+ the showcase comparison page builder reads it).
"""
import importlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

FUTON6 = Path("/home/joe/code/futon6")          # today's detectors
LOADER_TREE = Path("/tmp/futon6-sbs")            # PR-50 eprint loader
EPRINTS = Path("/home/joe/code/storage/futon6/data/arxiv-math-ct-eprints")
SLICES = Path("/home/joe/code/storage/mark2/ct-handoff/ct-anatomy-slices")
OUT = FUTON6 / "data" / "ct-nlp-sbs.json"

sys.path.insert(0, str(LOADER_TREE / "scripts"))
sys.path.insert(0, str(FUTON6 / "scripts"))

spec = importlib.util.spec_from_file_location(
    "superpod_job", str(LOADER_TREE / "scripts" / "superpod-job.py"))
sj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sj)
nw = importlib.import_module("nlab-wiring")

count_re = re.compile(rb'"count": (\d+)')


def slice_count(path: Path) -> int:
    tail = path.read_bytes()[-200:]
    m = count_re.search(tail)
    return int(m.group(1)) if m else -1


def stratified_sample(n_per=8):
    counts = {}
    for f in SLICES.glob("*.json"):
        if f.name == "_meta.json":
            continue
        counts[f.stem] = slice_count(f)
    by = sorted(counts.items(), key=lambda kv: -kv[1])
    high = [k for k, v in by[:n_per]]
    mid = [k for k, v in by if 100 <= v <= 1000][:n_per]
    small = [k for k, v in by if 1 <= v < 20][:n_per]
    zero = [k for k, v in by if v == 0][:n_per * 2]
    flat = ["arxiv-math_0506470"]
    return {"high": high, "mid": mid, "small": small, "zero": zero, "flat": flat}


def eid_to_arxiv(eid: str) -> str:
    return eid.removeprefix("arxiv-").replace("_", "/")


def main():
    strata = stratified_sample()
    results = []
    for stratum, eids in strata.items():
        for eid_file in eids:
            eid = "arxiv-" + eid_to_arxiv(eid_file).replace("arxiv-", "") \
                if not eid_file.startswith("arxiv-") else eid_file
            arxiv_eid = eid_file.replace("_", "/", 1) if "_" in eid_file and "/" not in eid_file else eid_file
            entity_id = arxiv_eid if arxiv_eid.startswith("arxiv-") else "arxiv-" + arxiv_eid
            text, meta = sj._load_eprint_text_for_entity(EPRINTS, entity_id)
            row = {"stratum": stratum, "paper": eid_file,
                   "eprint_status": meta.get("status")}
            if text:
                try:
                    fresh = nw.detect_scopes(entity_id, text)
                except Exception as exc:
                    row["fresh_error"] = str(exc)[:200]
                    fresh = []
                row["fresh_total"] = len(fresh)
                row["fresh_types"] = dict(Counter(
                    (s.get("hx/type") or s.get("type", "?")) for s in fresh).most_common(10))
            else:
                row["fresh_total"] = 0
                row["fresh_types"] = {}
            sl = SLICES / (eid_file + ".json")
            if sl.exists():
                rec = json.loads(sl.read_text())
                feb = rec.get("scopes", [])
                row["feb_total"] = len(feb)
                row["feb_types"] = dict(Counter(s.get("hx/type", "?") for s in feb).most_common(10))
            results.append(row)
            print(f"[{stratum}] {eid_file}: feb={row.get('feb_total')} "
                  f"fresh={row.get('fresh_total')} eprint={row.get('eprint_status')}",
                  flush=True)
    OUT.write_text(json.dumps({"results": results}, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
