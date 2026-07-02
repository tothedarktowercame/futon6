#!/usr/bin/env python3
"""D8 — capture before decommission. Write manifest.json (+ sha256 of every artifact) for the
PROOF-MINE run so the box can be DELETED (powered-off still bills) with nothing lost. The box is
from-dev mode (holds nothing unique); rsync $OUT back to dev BEFORE `linode-cli linodes delete`.

  python scripts/proof_mine_manifest.py --out data/proof-mine
"""
import argparse, hashlib, json, os


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.expanduser("~"), "code/futon6/data/proof-mine"))
    a = ap.parse_args()
    if not os.path.isdir(a.out):
        print("no artifact dir %s — nothing to manifest" % a.out)
        return
    entries = []
    for name in sorted(os.listdir(a.out)):
        p = os.path.join(a.out, name)
        if not os.path.isfile(p) or name == "manifest.json":
            continue
        entries.append({"file": name, "bytes": os.path.getsize(p), "sha256": sha256(p)})
    manifest = {"artifact_dir": a.out, "n_files": len(entries), "files": entries}
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    print("manifest.json: %d artifacts" % len(entries))
    for e in entries:
        print("  %s  %8d B  %s" % (e["sha256"][:12], e["bytes"], e["file"]))
    print("rsync '%s' to dev, verify these sha256, THEN `linode-cli linodes delete <id>` (D8)." % a.out)


if __name__ == "__main__":
    main()
