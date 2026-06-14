#!/usr/bin/env python3
"""Paleo-topography layer for the drainage-basin policy landscape.

THE CHARTER (holes/policy-landscape-drainage.md + holes/E-drainage-basin-policy-
landscape.md): reconstruct the underlying terrain/history BEFORE overlaying flow.
Trap 3 (non-stationarity): cascades reshape the terrain they drain, so a realized
route drained PAST terrain — scoring it against CURRENT topography compares
incommensurables. PALEO-TOPOGRAPHY is therefore a PREREQUISITE, not a refinement:
pin each basin's terrain to its drainage time, reconstructed from git history.

This layer builds, per basin (= one mined cascade in data/mission-triples/*.edn):
  - basin identity: the typed hole it closed (:have -> :want).
  - realized route: the ordered pattern-cite chain that actually drained it.
  - drainage pin: the git commit + timestamp at which the basin drained (the
    closing commit of its mission source, plus any closing-artifact SHAs) — the
    commit-pin that makes the terrain a PALEO surface, not the live one.
  - terrain: the substrate metric (Ollivier-Ricci curvature over the cascade
    graph) — "what is near, what is steep" — computed by REUSING the existing
    cascade adapter (substrate_metric_cascade_adapter), itself a commit-pinned
    snapshot read (terrain honesty), never live HEAD.
  - drainage outcome: the retrodictive witness — fraction of closing artifacts
    that verifiably exist (the "it drained" fact).

Output (data/paleo-topography.json) is ORDERED BY DRAINAGE TIME: the literal
paleo-stratigraphy, oldest terrain first. The drainage/flow layer
(drainage_flow.py) reads THIS, never the live terrain.

Usage: python3 scripts/paleo_topography.py [--corpus DIR] [--out FILE]
Reads only; writes one JSON artifact. No JVM, no network.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

FUTON6 = Path("/home/joe/code/futon6")
sys.path.insert(0, "/home/joe/code/futon3c/scripts")
import edn_format  # noqa: E402
import substrate_metric_cascade_adapter as adapter  # noqa: E402  (curvature: reused)

GET, KW = adapter.get, adapter.kw


def git_drainage_pin(source: str) -> dict:
    """The basin's drainage moment, reconstructed from git: the LAST commit that
    touched its mission source (the closing commit) — date, sha, subject. This is
    the temporal signal the retrospective-reconstruction lane established."""
    if not source:
        return {"at": None, "sha": None, "subject": None, "rule": "no-source"}
    p = Path(source)
    cwd = str(p.parent if p.parent.exists() else FUTON6)
    try:
        out = subprocess.run(
            ["git", "-C", cwd, "log", "-1", "--format=%aI%x09%h%x09%s", "--", source],
            capture_output=True, text=True, timeout=20)
        line = out.stdout.strip()
        if not line:
            return {"at": None, "sha": None, "subject": None, "rule": "untracked"}
        at, sha, *subj = line.split("\t")
        return {"at": at, "sha": sha, "subject": ("\t".join(subj))[:80],
                "rule": "closing-commit"}
    except Exception as ex:  # noqa: BLE001 — a git failure is data, report it
        return {"at": None, "sha": None, "subject": None, "rule": f"error:{ex}"}


def realized_route(triple) -> list[dict]:
    """The ordered pattern-cite chain that actually carried the discharge."""
    cascade = GET(triple, "cascade") or {}
    cites = GET(cascade, "pattern-cites") or []
    route = []
    for c in cites:
        route.append({
            "ident": KW(GET(c, "ident")) if GET(c, "ident") else None,
            "order": GET(c, "order"),
            "ref": KW(GET(c, "ref")) if GET(c, "ref") else None,
        })
    route.sort(key=lambda r: (r["order"] is None, r["order"]))
    return route


def drainage_outcome(triple) -> dict:
    """Retrodictive witness: of the closing artifacts, how many verifiably exist.
    'It drained' is this fact (corpus-wide 1052/1320 ~ 0.80)."""
    val = GET(triple, "validation") or {}
    arts = GET(val, "artifacts") or []
    total = len(arts)
    exists = sum(1 for a in arts if GET(a, "exists?") is True)
    return {"artifacts": total, "verified": exists,
            "verified_frac": round(exists / total, 4) if total else None}


def basin_record(path: Path) -> dict:
    triple = edn_format.loads(path.read_text())
    hole = GET(triple, "hole") or {}
    source = KW(GET(triple, "source")) if GET(triple, "source") else None
    # terrain: reuse the adapter's curvature (commit-pinned snapshot, not HEAD)
    proj = adapter.project_cascade(triple)
    cur = adapter.mission_curvature(proj)
    terrain = {
        "n_nodes": cur["n_nodes"], "n_edges": cur["n_edges"],
        "n_components": cur["n_components"],
        "satiety_full_frac": round(cur["satiety_full_frac"], 4),
        "kappa": cur["kappa"],  # curvature quantiles: the elevation/steepness
    }
    pin = git_drainage_pin(source)
    route = realized_route(triple)
    return {
        "basin": path.stem,
        "mission": KW(GET(triple, "mission")) if GET(triple, "mission") else path.stem,
        "source": source,
        "hole": {
            "have": GET(hole, "have"),
            "want": GET(hole, "want"),
            "confidence": KW(GET(hole, "confidence")) if GET(hole, "confidence") else None,
            "id": GET(hole, "id"),
        },
        "drainage_pin": pin,           # WHEN it drained (git) -> the paleo stratum
        "terrain": terrain,            # the curvature topography at that time
        "realized_route": route,       # WHAT drained it (the channel chain)
        "route_len": len(route),
        "outcome": drainage_outcome(triple),  # THAT it drained (retrodictive)
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(FUTON6 / "data" / "mission-triples"))
    ap.add_argument("--out", default=str(FUTON6 / "data" / "paleo-topography.json"))
    args = ap.parse_args()

    basins, errors = [], []
    for p in sorted(Path(args.corpus).glob("*.edn")):
        try:
            basins.append(basin_record(p))
        except Exception as ex:  # noqa: BLE001
            errors.append({"basin": p.stem, "error": str(ex)})

    # ORDER BY DRAINAGE TIME — the paleo-stratigraphy (oldest terrain first).
    # Undated basins (untracked) sort last; they have no pinned stratum.
    basins.sort(key=lambda b: (b["drainage_pin"]["at"] is None,
                               b["drainage_pin"]["at"] or ""))

    dated = [b for b in basins if b["drainage_pin"]["at"]]
    routed = [b for b in basins if b["route_len"] > 0]
    report = {
        "schema": "paleo-topography-v1",
        "charter": "holes/policy-landscape-drainage.md (trap 3: paleo first)",
        "terrain_metric": "Ollivier-Ricci curvature over the cascade graph "
                          "(commit-pinned snapshot via substrate_metric_cascade_adapter)",
        "paleo_note": "each basin's terrain is pinned to its DRAINAGE COMMIT "
                     "(git closing commit of the mission source), reconstructed "
                     "from git history — NOT live HEAD. Basins are ordered by "
                     "drainage time = the stratigraphy.",
        "n_basins": len(basins),
        "n_drainage_dated": len(dated),
        "n_with_realized_route": len(routed),
        "n_errors": len(errors),
        "drainage_span": {
            "earliest": dated[0]["drainage_pin"]["at"] if dated else None,
            "latest": dated[-1]["drainage_pin"]["at"] if dated else None,
        },
        "errors": errors,
        "basins": basins,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"paleo-topography: {len(basins)} basins "
          f"({len(dated)} drainage-dated, {len(routed)} with realized route, "
          f"{len(errors)} errors)")
    if dated:
        print(f"  stratigraphy span: {report['drainage_span']['earliest']} "
              f"-> {report['drainage_span']['latest']}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
