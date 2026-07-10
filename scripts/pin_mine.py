#!/usr/bin/env python3
"""pin_mine — canonicalise + checksum the mining run (M-post-mining-ingest DERIVE, half (a)).

The mining run (the State apparatus: M-operational-vocabulary + M-goals-and-holes) produced a
one-off, expensive, multi-artifact snapshot whose canonical members are marked only by CONVENTION
(the unsuffixed `*.openai.json`), with no checksum/version travelling with the file (MAP finding
#4; the mtime-trap is live).  This writes a TRACKED registry that PINS each canonical product:
provenance + sha256 + bytes + version + class + (for data products) the node-type it claims.  A
consumer can then read one pinned source instead of guessing which sibling JSON is canonical.

  futon6/.venv/bin/python scripts/pin_mine.py           # regenerate data/mine-registry.edn
  futon6/.venv/bin/python scripts/pin_mine.py --check    # verify on-disk canon matches the pin

`--check` recomputes every sha256 and exits non-zero if a canonical product drifted from the
committed registry — the consistency guard the consumer contract (DERIVE half (b)) builds on.
This half is :7071-free; the substrate-2 `mine/meme` ingest is a separate, greenlit step.
"""
import hashlib, json, os, sys, time, datetime, collections

F6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY = os.path.join(F6, "data", "mine-registry.edn")
MINE_VERSION = "2026-06-25-v1"  # the snapshot this pin freezes (the canonical mine's mine-date)

# The de-facto canonical products (the unsuffixed live set, per MAP §A) + how to govern each.
#   class :data    -> pin the bytes (checksum); it claims substrate node-ids
#   class :derived  -> regenerable from canonical data; pin the recipe-metadata, not just bytes
#   class :fixture  -> never read as data (none canonical here)
PRODUCTS = [
    {"id": "joint-memes",    "path": "data/meme-mine/joint-memes.openai.json",
     "class": "data", "node-type": "meme", "node-id-pattern": "meme:ask-*"},
    {"id": "resolved-memes", "path": "data/meme-mine/resolved-memes.openai.json",
     "class": "data", "node-type": "meme", "node-id-pattern": "meme:ask-*"},
    {"id": "concept-index",  "path": "data/meme-mine/concept-index.json",
     "class": "derived", "node-type": "concept",
     "note": "keys are mission/* & pattern/* strings -> ingest as mine/concept-about REFERENCE "
             "edges, NOT TYPE claims (would collide with O3 mission: typing). Deferred from car 1."},
    {"id": "action-cert",    "path": "data/meme-mine/action-cert.json", "class": "data"},
    {"id": "c-entries",      "path": "data/c-vector/c-entries.openai.json",
     "class": "data", "node-type": "c-entry", "node-id-pattern": "ask-*"},
    {"id": "move-basins",    "path": "data/c-vector/move-basins.json",
     "class": "derived",
     "note": "MiniLM move classifier (build/reach/steer) — NOT node rows; pin the recipe."},
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(p):
    """Pin one product: real bytes/sha/mtime + truthful provenance pulled FROM the artifact."""
    full = os.path.join(F6, p["path"])
    if not os.path.exists(full):
        return {**p, "missing?": True}
    st = os.stat(full)
    rec = {**p, "canonical?": True, "sha256": sha256(full), "bytes": st.st_size,
           "mtime": datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
           # Exact model id does NOT travel in-artifact (MAP #4); record the family by convention.
           "model": "all-MiniLM-L6-v2" if p["id"] == "move-basins"
                    else ("openai" if ".openai." in p["path"] else "unrecorded"),
           "gate-status": "unverified"}  # honest: this script checksums, it does not gate-check
    try:
        d = json.load(open(full))
    except Exception:
        return rec
    if isinstance(d, list):
        rec["count"] = len(d)
        sessions = sorted({(e.get("provenance") or {}).get("session")
                           for e in d if isinstance(e, dict)} - {None})
        if sessions:
            rec["provenance/sessions"] = sessions  # REAL coverage from each entry's provenance
    elif isinstance(d, dict):
        if p["id"] == "move-basins":  # capture the real recipe metadata
            rec["recipe"] = {"model": d.get("model"), "classes": d.get("classes"),
                             "val_accuracy": d.get("val_accuracy")}
        else:
            rec["count"] = len(d)
    return rec


def to_edn(v, indent=0):
    pad = "  " * indent
    if isinstance(v, dict):
        items = "\n".join(f"{pad}  {to_edn(k)} {to_edn(val, indent + 1)}" for k, val in v.items())
        return "{\n" + items + "\n" + pad + "}"
    if isinstance(v, list):
        return "[" + " ".join(to_edn(x, indent) for x in v) + "]"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return ":" + v if v in ("data", "derived", "fixture", "unverified") else json.dumps(v)
    if v is None:
        return "nil"
    return str(v)


def build():
    return {"mine/version": MINE_VERSION,
            "mine/pinned-at-ms": int(time.time() * 1000),
            "mine/generator": "scripts/pin_mine.py",
            "mine/note": "Canonical pin for the M-operational-vocabulary + M-goals-and-holes "
                         "mining run. The unsuffixed *.openai.json set is canonical; this freezes "
                         "it by checksum so consumers read one pinned source (M-post-mining-ingest "
                         "DERIVE (a)). The mine/meme substrate-2 ingest (b) is a separate step.",
            "mine/products": [describe(p) for p in PRODUCTS]}


def main():
    if "--check" in sys.argv:
        if not os.path.exists(REGISTRY):
            print("no registry — run pin_mine.py first"); sys.exit(2)
        # Compare just the per-product sha256 (the durable identity) against fresh recompute.
        old = open(REGISTRY).read()
        drift = []
        for p in PRODUCTS:
            full = os.path.join(F6, p["path"])
            if not os.path.exists(full):
                continue
            want = sha256(full)
            if f'"sha256" "{want}"' not in old:
                drift.append(p["id"])
        if drift:
            print("DRIFT — canonical product(s) changed since pin:", ", ".join(drift)); sys.exit(1)
        print("OK — all canonical products match the pin"); sys.exit(0)
    reg = build()
    with open(REGISTRY, "w") as f:
        f.write(to_edn(reg) + "\n")
    n = sum(1 for p in reg["mine/products"] if not p.get("missing?"))
    nodes = sum(p.get("count", 0) for p in reg["mine/products"] if p.get("node-type") == "meme")
    print(f"pinned {n} products -> {REGISTRY}")
    print(f"  version {reg['mine/version']} · {nodes} meme nodes claimable as meme:ask-* (car 1)")
    for p in reg["mine/products"]:
        tag = "MISSING" if p.get("missing?") else p["sha256"][:12]
        print(f"  {tag}  {p['id']:15} {p.get('node-type', p['class']):8} n={p.get('count', '-')}")


if __name__ == "__main__":
    main()
