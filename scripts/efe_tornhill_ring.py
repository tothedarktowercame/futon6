#!/usr/bin/env python3
# efe_tornhill_ring.py — the EFE field page's pink code ring, reading the file-level
# Tornhill report (M-the-perfect-crime packet 1b, E-kimi-task-44, 2026-09-26).
# Standard library only; no file reads at import.
from __future__ import annotations

import json
import math
from pathlib import Path

import futon6_config as config

REPORT_DIR = config.path("FUTON6_TORNHILL_REPORT",
                         Path.home() / ".local/share/futon-audits/tornhill")


def load_report(dir_or_path):
    """Load the newest tornhill-YYYY-MM-DD.json at/below dir_or_path; None when absent."""
    p = Path(dir_or_path)
    if p.is_dir():
        cands = sorted(p for p in p.glob("tornhill-*.json")
                       if not p.name.startswith("tornhill-chat-"))
        if not cands:
            return None
        p = cands[-1]
    if not p.is_file():
        return None
    report = json.load(open(p))
    report["_filename"] = p.name
    return report


def load_chat(report_path):
    """The tornhill-chat file sitting beside the report, when it exists."""
    p = Path(report_path)
    if p.is_dir():
        cands = sorted(p.glob("tornhill-chat-*.json"))
        if not cands:
            return None
        p = cands[-1]
    else:
        p = p.with_name(p.name.replace("tornhill-", "tornhill-chat-", 1))
    if p.is_file():
        return json.load(open(p))
    return None


def index(report, chat=None):
    """{(repo, path): file_row}; rows gain a 'seats' set when chat is given."""
    idx = {}
    for repo, rdata in (report.get("repos") or {}).items():
        for row in rdata.get("files") or []:
            idx[(repo, row["path"])] = dict(row, repo=repo)
    if chat:
        for row in chat.get("files") or []:
            key = (row.get("repo"), row.get("path"))
            if key in idx and row.get("seats"):
                idx[key]["seats"] = set(row["seats"])
    return idx


def mission_ring(mission_row, idx):
    """Classify one mission-activity row against the Tornhill index.

    state ∈ no-report | no-link | unmeasured | measured."""
    if idx is None:
        return {"state": "no-report"}
    code = (mission_row or {}).get("code") or {}
    files = code.get("files") or []
    if not files:
        return {"state": "no-link"}
    hits = []
    for repo, relpath in files:
        row = idx.get((repo, relpath))
        if row is not None:
            hits.append(row)
    if not hits:
        return {"state": "unmeasured", "n_files": len(files)}
    seat_union = set()
    have_seats = False
    for row in hits:
        if row.get("seats"):
            have_seats = True
            seat_union |= set(row["seats"])
    top = []
    for row in sorted(hits, key=lambda r: -(r.get("hotspot") or 0))[:3]:
        top.append({
            "path": f'{row.get("repo", "?")}/{row["path"]}',
            "revs": row.get("revs") or 0,
            "hotspot": row.get("hotspot") or 0,
            "trend": "new" if row.get("born_in_window") else row.get("trend_ratio"),
        })
    return {
        "state": "measured",
        "revs": sum(r.get("revs") or 0 for r in hits),
        "hotspot": sum(r.get("hotspot") or 0 for r in hits),
        "top": top,
        "n_files_in_report": len(hits),
        "n_files": len(files),
        "seats": len(seat_union) if have_seats else None,
    }


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def ring_svg(x, y, r, mission, ring, report_meta, hotspot_max):
    """SVG string for one district's code ring, in the state ring reports."""
    state = ring["state"]
    if state == "no-report":
        return ""
    if state == "no-link":
        return (f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r:.1f}" fill="none" stroke="#3a4252" '
                f'stroke-width="0.6" stroke-dasharray="1,4" opacity="0.4">'
                f'<title>{_esc(mission)} · code ring: NO mission→code link '
                f'(no code.files on this mission row — coverage gap, stated not hidden)</title></circle>')
    meta = report_meta or {}
    src = f'source: Tornhill report {_esc(meta.get("filename", "?"))} generated {_esc(meta.get("generated", "?"))} (window {meta.get("window_days", "?")}d)'
    if state == "unmeasured":
        return (f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r:.1f}" fill="none" stroke="#ec4899" '
                f'stroke-width="0.8" stroke-dasharray="3,3" opacity="0.6">'
                f'<title>{_esc(mission)} · code ring: {ring["n_files"]} linked code file(s), none changed '
                f'in the report\'s window (or none are code) · {src}</title></circle>')
    # measured
    w = 0.8 + 3.0 * (math.log1p(ring["hotspot"]) / math.log1p(hotspot_max)) if hotspot_max else 0.8
    lines = []
    for t in ring["top"]:
        trend = t["trend"]
        trend_s = "new" if trend == "new" else (f"trend ×{trend:.2f}" if isinstance(trend, (int, float)) else "trend not sampled: the report samples each repo's top 10 hotspots")
        lines.append(f'{t["path"]} (revs {t["revs"]}, hotspot {t["hotspot"]}, {trend_s})')
    seats = f' · seats {ring["seats"]}' if ring.get("seats") is not None else ""
    title = (f'{_esc(mission)} · CODE ring (Tornhill report): revs {ring["revs"]}, hotspot {ring["hotspot"]} · '
             f'{ring["n_files_in_report"]} of {ring["n_files"]} files in the report{seats} · '
             f'top: {"; ".join(_esc(l) for l in lines)} · {src}')
    return (f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r:.1f}" fill="none" stroke="#ec4899" '
            f'stroke-width="{w:.1f}" opacity="0.9"><title>{title}</title></circle>')
