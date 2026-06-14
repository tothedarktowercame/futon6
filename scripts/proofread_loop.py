#!/usr/bin/env python3
"""The proofread learning loop: defect tag → minted defect-record → fix
discharges it, surfaced AROUND POINT in the paper-anatomy buffer.

Joe's critique (2026-06-13): "if this is a learning loop, the defects should
lead directly to new fixes, and I don't see anything being minted around
point." This closes it. A proofread tag is a typed defect (a sorry: have =
the defect, want = correct detection); the fix is its discharge (a commit).
Both are minted to a per-paper record the viewer surfaces as overlays, and
promoted to the meme.db BHK arrow store (the durable substrate).

    proofread_loop.py mint   --paper P --position N --span S --verdict V --note "..."
    proofread_loop.py discharge --id D --fix "<commit/desc>"
    proofread_loop.py list   --paper P
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import time
from pathlib import Path

DEFECTS_DIR = Path("/home/joe/code/futon6/data/proofread-defects")


def _path(paper: str) -> Path:
    return DEFECTS_DIR / f"{paper}.edn"


def _read(paper: str) -> list[dict]:
    p = _path(paper)
    if not p.exists():
        return []
    # EDN here is a list of flat maps with string/number values — read as a
    # python literal after light translation (keywords→strings).
    txt = p.read_text()
    try:
        return json.loads(txt) if txt.lstrip().startswith("[") and '"' in txt else _edn_to_py(txt)
    except Exception:
        return _edn_to_py(txt)


def _edn_to_py(txt: str) -> list[dict]:
    # minimal: we always WRITE json-compatible edn (see _write), so reading
    # back as json works; this is the fallback for hand-edits.
    import re
    txt = re.sub(r"(?m):([a-zA-Z][\w-]*)", r'"\1"', txt)
    try:
        return ast.literal_eval(txt.replace("{", "{").replace("nil", "None"))
    except Exception:
        return []


def _write(paper: str, defects: list[dict]) -> None:
    DEFECTS_DIR.mkdir(parents=True, exist_ok=True)
    # EDN that is also valid JSON (keys quoted) — the viewer reads it as JSON.
    _path(paper).write_text(json.dumps(defects, indent=1))


def _did(paper: str, position: int, span: str) -> str:
    h = hashlib.sha1(f"{paper}|{position}|{span[:40]}".encode()).hexdigest()[:8]
    return f"defect-{paper}-{h}"


def mint(paper, position, span, verdict, note, at):
    defects = _read(paper)
    did = _did(paper, position, span)
    if any(d["id"] == did for d in defects):
        print(f"already minted: {did}")
        return did
    defects.append({
        "id": did, "paper": paper, "position": int(position),
        "span": span[:200], "verdict": verdict, "note": note or "",
        "status": "open", "fix": None, "minted-at": at,
        # the sorry shape: have → want
        "have": f"detector defect ({verdict}) at {paper}:{position}",
        "want": "correct detection (fix discharges this)",
    })
    _write(paper, defects)
    print(f"minted {did} (open) — surfaced around position {position} in {paper}")
    return did


def discharge(did, fix, at):
    # find which paper holds it
    for p in DEFECTS_DIR.glob("*.edn"):
        paper = p.stem
        defects = _read(paper)
        for d in defects:
            if d["id"] == did:
                d["status"] = "fixed"
                d["fix"] = fix
                d["fixed-at"] = at
                _write(paper, defects)
                print(f"discharged {did} → fixed by {fix}")
                return
    print(f"defect not found: {did}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mint")
    m.add_argument("--paper", required=True); m.add_argument("--position", required=True)
    m.add_argument("--span", required=True); m.add_argument("--verdict", default="incomplete")
    m.add_argument("--note", default="")
    d = sub.add_parser("discharge")
    d.add_argument("--id", required=True); d.add_argument("--fix", required=True)
    ls = sub.add_parser("list"); ls.add_argument("--paper", required=True)
    a = ap.parse_args(argv)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if a.cmd == "mint":
        mint(a.paper, a.position, a.span, a.verdict, a.note, now)
    elif a.cmd == "discharge":
        discharge(a.id, a.fix, now)
    elif a.cmd == "list":
        for x in _read(a.paper):
            print(f"  {x['status']:6} {x['id']}  @{x['position']}  {x['verdict']}"
                  f"  {('→ '+x['fix']) if x['fix'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
