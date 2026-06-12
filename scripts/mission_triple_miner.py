#!/usr/bin/env python3
"""Mine retrospective mission triples from completed mission markdown.

The v0 miner is deterministic and deliberately lossy: it records what it can
recover from mission prose, plus explicit loss where old missions predate the
derive-gate/cascade/wiring conventions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CODE_ROOT = Path("/home/joe/code")
ROOT = CODE_ROOT / "futon6"
OUT_DIR = ROOT / "data" / "mission-triples"
SCOPE_DETECT = ROOT / "scripts" / "mission_scope_detect.py"

HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.M)
CHECKPOINT_RE = re.compile(
    r"(?ims)^\s*(?:#{1,6}\s*)?(?:[-*]\s*)?Checkpoint\s+(\d+)\b"
    r"\s*(?:\(([^)]*)\))?\s*:?\s*(.*?)"
    r"(?=^\s*(?:#{1,6}\s*)?(?:[-*]\s*)?Checkpoint\s+\d+\b|\Z)"
)
STATUS_RE = re.compile(r"(?im)^\s*(?:\*\*)?Status(?:\*\*)?\s*:?\s*(.+)$")
PATH_RE = re.compile(r"(?<![A-Za-z0-9_./-])(?:~?/)?(?:[A-Za-z0-9_.{}-]+/)+[A-Za-z0-9_.{}-]+\.(?:clj|cljs|cljc|py|edn|bb|el|json|md|html|css|ts|js|cert|svg|tex|sty)(?![A-Za-z0-9_.-])")
SHA_RE = re.compile(r"\b[0-9a-f]{7,40}\b")


@dataclass(frozen=True)
class Section:
    level: int
    title: str
    start: int
    content_start: int
    end: int
    text: str


def load_scope_detect():
    spec = importlib.util.spec_from_file_location("mission_scope_detect", SCOPE_DETECT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCOPE_DETECT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


SCOPE = load_scope_detect()
PATTERN_CANDIDATE_RE = SCOPE.PATTERN_CANDIDATE_RE


def split_sections(text: str) -> list[Section]:
    matches = list(HEADER_RE.finditer(text))
    sections: list[Section] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = re.sub(r"`([^`]+)`", r"\1", m.group(2)).strip().strip("*")
        sections.append(Section(len(m.group(1)), title, m.start(), m.end(), end, text[m.end():end]))
    return sections


def section_slug(title: str) -> str:
    title = re.sub(r"^\d+[a-z]?[.)-]?\s*", "", title.strip().lower())
    return re.sub(r"[^a-z0-9]+", "-", title).strip("-")


def section_for(sections: list[Section], offset: int) -> Section | None:
    candidates = [s for s in sections if s.start <= offset < s.end]
    return max(candidates, key=lambda s: s.start) if candidates else None


def sentence_window(text: str, start: int, end: int, radius: int = 240) -> str:
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    line_lo = text.rfind("\n", 0, start)
    line_hi = text.find("\n", end)
    if line_lo != -1:
        lo = max(lo, line_lo + 1)
    if line_hi != -1:
        hi = min(hi, line_hi)
    return re.sub(r"\s+", " ", text[lo:hi]).strip()


def discover_missions(root: Path = CODE_ROOT) -> list[Path]:
    out: dict[str, Path] = {}

    def score(path: Path) -> tuple[int, int, str]:
        repo = path.relative_to(root).parts[0]
        penalty = 0
        if any(tag in repo for tag in ("desktop-save", "health-main", "arguing-worlds")):
            penalty += 10
        return (penalty, len(path.parts), str(path))

    for repo in sorted(root.glob("futon*")):
        if not repo.is_dir() or ".state" in repo.parts:
            continue
        for pat in ("holes/M-*.md", "holes/missions/M-*.md"):
            for path in repo.glob(pat):
                if ".state" not in path.parts and path.is_file():
                    old = out.get(path.stem)
                    if old is None or score(path) < score(old):
                        out[path.stem] = path
    return sorted(out.values())


def status_classifier(text: str) -> dict[str, Any]:
    status_line = None
    for m in STATUS_RE.finditer(text):
        status_line = m.group(1).strip()
        break
    evidence = status_line or ""
    low = evidence.lower()
    completed = False
    reason = "no status line"
    if evidence:
        has_hard_close = re.search(r"\b(?:closed|delivered|done)\b|:done", low)
        has_complete = re.search(r"\b(?:complete|completed)\b", low)
        if has_hard_close or has_complete:
            if "mission-close is the operator" in low or "mission close is the operator" in low:
                reason = "explicit mission-close still operator-gated"
            elif "not yet" in low or "pending" in low or "in progress" in low:
                reason = "status contains open/pending qualifier"
            elif has_hard_close:
                completed = True
                reason = "status hard closure token"
            elif re.search(r"\b(?:phase|head|identify|map|derive|argue|verify|instantiate|stage|part|exit|gate|checkpoint)\s+[a-z0-9-]*\s*(?:complete|done|met|pass)", low):
                reason = "phase/checkpoint completion only"
            else:
                completed = True
                reason = "status closure token"
        elif "archived" in low:
            completed = True
            reason = "archived status"
        else:
            reason = "status not closed"
    return {"completed?": completed, "evidence": evidence[:500], "classifier": reason}


def section_by_phase(sections: list[Section], phase: str) -> Section | None:
    phase = phase.lower()
    for sec in sections:
        slug = section_slug(sec.title)
        words = slug.split("-")
        if words and words[0] == phase:
            return sec
        if phase in words[:3]:
            return sec
    return None


def phase_region(text: str, sections: list[Section], phase: str) -> tuple[str, str | None]:
    sec = section_by_phase(sections, phase)
    if not sec:
        return "", None
    end = len(text)
    for other in sections:
        if other.start > sec.start and other.level <= sec.level:
            end = other.start
            break
    return text[sec.content_start:end], sec.title


def titled_region(text: str, sections: list[Section], title_re: str) -> tuple[str, str | None]:
    pat = re.compile(title_re, re.I)
    for sec in sections:
        if pat.search(sec.title):
            end = len(text)
            for other in sections:
                if other.start > sec.start and other.level <= sec.level:
                    end = other.start
                    break
            return text[sec.content_start:end], sec.title
    return "", None


def clean_endpoint(s: str) -> str:
    s = re.sub(r"[`*_]", "", s)
    s = s.replace("→", "->")
    s = re.sub(r"\s+", "-", s.strip().lower())
    s = re.sub(r"[^a-z0-9:/._-]+", "", s).strip("-")
    return s[:160] or "unmined"


def first_substantial_paragraph(text: str) -> str | None:
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", para).strip()
        if len(para) >= 80 and not para.startswith("|"):
            return para[:700]
    return None


def last_substantial_paragraph(text: str) -> str | None:
    paras = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", para).strip()
        if len(para) >= 80 and not para.startswith("|"):
            paras.append(para[:700])
    return paras[-1] if paras else None


def mine_hole(text: str, sections: list[Section]) -> dict[str, Any]:
    derive_text, derive_title = phase_region(text, sections, "derive")
    fallback_regions = []
    for p in ("head", "identify", "map"):
        region, _title = phase_region(text, sections, p)
        if region:
            fallback_regions.append((p, region))
    source = derive_text or "\n".join(region for _phase, region in fallback_regions)
    explicit = re.search(
        r"(?:typed hole minted:\s*)?`?(?P<id>arr-[A-Za-z0-9-]+)`?.{0,260}?"
        r"have\s*[=`:]?\s*`?(?P<have>[A-Za-z0-9_.:/-]+)`?\s*(?:→|->|to)\s*want\s*[=`:]?\s*`?(?P<want>[A-Za-z0-9_.:/-]+)`?",
        source,
        re.I | re.S,
    )
    if explicit:
        via = sentence_window(source, explicit.start(), explicit.end(), 360)
        return {
            "confidence": kw("authored"),
            "id": explicit.group("id"),
            "have": explicit.group("have"),
            "want": explicit.group("want"),
            "source-section": derive_title or "fallback",
            "via": via,
            "rule": "sortie-11/hole-from-derive-gate",
        }

    hw = re.search(r"What we have:?\s*(?P<have>.+?)(?:\n\n|What we lack:)", source, re.I | re.S)
    wl = re.search(r"What we lack:?\s*(?P<want>.+?)(?:\n\n|There is|The mission|$)", source, re.I | re.S)
    if hw and wl:
        have = clean_endpoint(hw.group("have")[:180])
        want = clean_endpoint(wl.group("want")[:180])
        return {
            "confidence": kw("reconstructed-thin"),
            "id": None,
            "have": have,
            "want": want,
            "source-section": derive_title or "HEAD/IDENTIFY/MAP fallback",
            "via": re.sub(r"\s+", " ", (hw.group(0) + " " + wl.group(0))[:700]).strip(),
            "rule": "sortie-11/fallback-have-want",
        }

    fallback_by_phase = {phase: region for phase, region in fallback_regions}
    have_quote = first_substantial_paragraph(fallback_by_phase.get("head", "")) or first_substantial_paragraph(fallback_by_phase.get("identify", ""))
    want_quote = last_substantial_paragraph(fallback_by_phase.get("map", "")) or last_substantial_paragraph(fallback_by_phase.get("identify", ""))
    if have_quote and want_quote:
        return {
            "confidence": kw("reconstructed-thin"),
            "id": None,
            "have": clean_endpoint(have_quote[:180]),
            "want": clean_endpoint(want_quote[:180]),
            "source-section": "HEAD/IDENTIFY/MAP fallback",
            "via": f"HAVE quote: {have_quote} WANT quote: {want_quote}",
            "rule": "sortie-11/fallback-chain-quoted",
        }

    head = sections[0].text if sections else text[:500]
    return {
        "confidence": kw("unminable"),
        "id": None,
        "have": None,
        "want": None,
        "source-section": derive_title or "HEAD/IDENTIFY/MAP fallback",
        "reason": "no derive-gate endpoints and no What we have/lack pair",
        "via": re.sub(r"\s+", " ", head[:500]).strip(),
        "rule": "sortie-11/no-fabrication",
    }


def kw(name: str) -> str:
    return ":" + name


def mine_patterns(path: Path, text: str, sections: list[Section], pattern_index: dict[str, str], hole: dict[str, Any]) -> dict[str, Any]:
    pattern_text, pattern_title = titled_region(text, sections, r"pattern cross-reference|pattern cross references")
    search_text = pattern_text or text
    base_offset = text.find(pattern_text) if pattern_text else 0
    cites = []
    seen_at = set()
    for m in PATTERN_CANDIDATE_RE.finditer(search_text):
        ident = m.group(1)
        if ident not in pattern_index:
            continue
        abs_start = base_offset + m.start()
        abs_end = base_offset + m.end()
        key = (ident, abs_start)
        if key in seen_at:
            continue
        seen_at.add(key)
        sec = section_for(sections, abs_start)
        cites.append({
            "order": len(cites),
            "ident": ident,
            "ref": pattern_index[ident],
            "offset": abs_start,
            "section": sec.title if sec else None,
            "phase": section_slug(sec.title).split("-")[0] if sec else None,
            "via": sentence_window(text, abs_start, abs_end, 280),
            "rule": "sortie-11/pattern-cite-site",
        })
    nodes = [
        {
            "id": kw("problem"),
            "role": kw("scope"),
            "satiety": {"hungry-for": kw("payoff")} if hole.get("confidence") != kw("unminable") else {"hungry-for": kw("parse")},
            "form": f"{hole.get('have')} -> {hole.get('want')}" if hole.get("have") else "unmined mission hole",
            "via": hole.get("via", ""),
            "rule": "sortie-11/cascade-problem-node",
        }
    ]
    hyperedges = []
    prev = kw("problem")
    for cite in cites:
        node_id = kw(f"pattern-{cite['order']:02d}")
        nodes.append({
            "id": node_id,
            "role": kw("concept"),
            "satiety": kw("full"),
            "form": cite["ident"],
            "ref": cite["ref"],
            "via": cite["via"],
            "rule": "sortie-11/cascade-pattern-node",
        })
        hyperedges.append({
            "kind": kw("differentiates"),
            "ends": [{"role": kw("context"), "node": prev}, {"role": kw("pattern"), "node": node_id}],
            "via": cite["via"],
            "rule": "sortie-11/cite-order-cascade",
        })
        prev = node_id
    if cites:
        hyperedges.append({
            "kind": kw("states"),
            "ends": [{"role": kw("pattern"), "node": prev}, {"role": kw("problem"), "node": kw("problem")}],
            "via": "terminal cite in authored pattern-cite order closes the v0 cascade skeleton",
            "rule": "sortie-11/cascade-discharge-skeleton",
        })
    return {
        "region": {"source": str(path), "label": pattern_title or "whole-mission-pattern-cites"},
        "pattern-cites": cites,
        "nodes": nodes,
        "hyperedges": hyperedges,
    }


def mine_wiring(text: str) -> dict[str, Any]:
    checkpoints = []
    nodes = []
    edges = []
    prev = None
    for m in CHECKPOINT_RE.finditer(text):
        n = int(m.group(1))
        body = re.sub(r"\s+", " ", m.group(0)).strip()
        node_id = kw(f"ckpt-{n:02d}")
        checkpoints.append({"number": n, "offset": m.start(), "via": body[:1200]})
        nodes.append({
            "id": node_id,
            "role": kw("application"),
            "satiety": kw("full"),
            "form": f"Checkpoint {n}",
            "witness": body[:1200],
            "via": body[:1200],
            "rule": "sortie-11/checkpoint-as-witnessed-application",
        })
        if prev is not None:
            edges.append({
                "kind": kw("composes"),
                "ends": [{"role": kw("from"), "node": prev}, {"role": kw("to"), "node": node_id}],
                "via": f"Checkpoint {n - 1} -> Checkpoint {n} in authored checkpoint order",
                "rule": "sortie-10/wiring-composition",
            })
        prev = node_id
    artifacts = extract_artifacts(text)
    return {
        "checkpoints": checkpoints,
        "closing-artifacts": artifacts,
        "nodes": nodes,
        "hyperedges": edges,
    }


def extract_artifacts(text: str) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for m in PATH_RE.finditer(text):
        value = m.group(0).strip("`.,);")
        if value not in seen:
            seen.add(value)
            out.append({"kind": kw("file"), "named": value, "offset": m.start(), "via": sentence_window(text, m.start(), m.end(), 180)})
    for m in SHA_RE.finditer(text):
        value = m.group(0)
        if value not in seen:
            seen.add(value)
            out.append({"kind": kw("commit"), "named": value, "offset": m.start(), "via": sentence_window(text, m.start(), m.end(), 120)})
    return out


def repo_roots() -> list[Path]:
    return [p for p in sorted(CODE_ROOT.glob("futon*")) if (p / ".git").exists()]


def artifact_exists(artifact: dict[str, Any], roots: list[Path]) -> bool:
    value = artifact["named"]
    if artifact["kind"] == kw("file"):
        candidates = []
        if value.startswith("~/"):
            candidates.append(Path.home() / value[2:])
        elif value.startswith("/"):
            candidates.append(Path(value))
        else:
            candidates.append(CODE_ROOT / value)
            for root in roots:
                candidates.append(root / value)
        if any(p.exists() for p in candidates):
            return True
        if not value.startswith(("/", "~/")):
            suffix = Path(value).as_posix()
            for root in roots:
                for dirname in ("src", "scripts", "resources"):
                    base = root / dirname
                    if not base.exists():
                        continue
                    try:
                        if any(base.glob(f"**/{suffix}")):
                            return True
                    except OSError:
                        continue
            return False
    if artifact["kind"] == kw("commit"):
        for root in roots:
            try:
                subprocess.run(
                    ["git", "-C", str(root), "cat-file", "-e", f"{value}^{{commit}}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=1.5,
                    check=True,
                )
                return True
            except (subprocess.SubprocessError, OSError):
                continue
    return False


def validation(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    roots = repo_roots()
    rows = []
    for art in artifacts:
        rows.append({
            "named": art["named"],
            "kind": art["kind"],
            "exists?": artifact_exists(art, roots),
            "via": art.get("via", ""),
            "rule": "sortie-11/retrospective-witness",
        })
    return {"artifacts": rows}


def loss(hole: dict[str, Any], cascade: dict[str, Any], validation_row: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    if hole.get("confidence") == kw("unminable"):
        rows.append({"kind": kw("missing-derive"), "detail": hole.get("reason"), "rule": "sortie-11/per-mission-loss"})
    elif hole.get("confidence") == kw("reconstructed-thin"):
        rows.append({"kind": kw("thin-derive"), "detail": "hole reconstructed from fallback chain", "rule": "sortie-11/per-mission-loss"})
    if not cascade["pattern-cites"]:
        rows.append({"kind": kw("zero-pattern-cites"), "detail": "no library pattern cite found", "rule": "sortie-11/per-mission-loss"})
    missing = [r["named"] for r in validation_row["artifacts"] if not r["exists?"]]
    if missing:
        rows.append({"kind": kw("unverifiable-artifacts"), "detail": missing[:40], "count": len(missing), "rule": "sortie-11/per-mission-loss"})
    return rows


def analyze_mission(path: Path, pattern_index: dict[str, str], force: bool, out_dir: Path, full_run: bool) -> dict[str, Any] | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    status = status_classifier(text)
    if full_run and not status["completed?"]:
        return {"mission": path.stem, "path": str(path), "skipped": True, "status": status}
    out_path = out_dir / f"{path.stem}.edn"
    if out_path.exists() and not force:
        return {"mission": path.stem, "path": str(path), "skipped-existing": True, "status": status}
    sections = split_sections(text)
    hole = mine_hole(text, sections)
    cascade = mine_patterns(path, text, sections, pattern_index, hole)
    wiring = mine_wiring(text)
    val = validation(wiring["closing-artifacts"])
    losses = loss(hole, cascade, val)
    row = {
        "mission": path.stem,
        "source": str(path),
        "status": status,
        "hole": hole,
        "cascade": cascade,
        "wiring": wiring,
        "validation": val,
        "loss": losses,
        "arrow-candidate": {
            "have": hole.get("have"),
            "want": hole.get("want"),
            "format": f"{hole.get('have')} -> {hole.get('want')}" if hole.get("have") and hole.get("want") else None,
            "escrow": "do-not-write-meme.db",
        },
    }
    out_path.write_text(to_edn(row) + "\n", encoding="utf-8")
    return {
        "mission": path.stem,
        "path": str(path),
        "output": str(out_path),
        "status": status,
        "hole-tier": hole.get("confidence"),
        "patterns": len(cascade["pattern-cites"]),
        "checkpoints": len(wiring["checkpoints"]),
        "artifacts": len(val["artifacts"]),
        "artifact-exists": sum(1 for r in val["artifacts"] if r["exists?"]),
        "loss": [r["kind"] for r in losses],
    }


def edn_key(k: str) -> str:
    return ":" + k.replace("_", "-").replace("?", "?")


def edn_str(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def to_edn(obj: Any, indent: int = 0) -> str:
    sp = " " * indent
    if isinstance(obj, str):
        if obj.startswith(":") and re.match(r"^:[A-Za-z0-9*+!_?$%&=<>./-]+$", obj):
            return obj
        return edn_str(obj)
    if obj is None:
        return "nil"
    if obj is True:
        return "true"
    if obj is False:
        return "false"
    if isinstance(obj, (int, float)):
        return repr(obj)
    if isinstance(obj, list):
        if not obj:
            return "[]"
        return "[" + "\n".join((" " * (indent + 1)) + to_edn(x, indent + 1) for x in obj) + "]"
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        parts = []
        for k, v in obj.items():
            key = edn_key(k) if isinstance(k, str) and not k.startswith(":") else str(k)
            parts.append(f"{' ' * (indent + 1)}{key} {to_edn(v, indent + 1)}")
        return "{" + "\n".join(parts) + "}"
    raise TypeError(type(obj))


def summarize(results: list[dict[str, Any] | None]) -> dict[str, Any]:
    present = [r for r in results if r]
    processed = [r for r in present if not r.get("skipped") and not r.get("skipped-existing")]
    skipped = [r for r in present if r.get("skipped")]
    tiers = Counter(r.get("hole-tier") for r in processed)
    loss_hist = Counter(kind for r in processed for kind in r.get("loss", []))
    return {
        "missions-seen": len(present),
        "processed": len(processed),
        "skipped-not-completed": len(skipped),
        "skipped-existing": sum(1 for r in present if r.get("skipped-existing")),
        "tier-distribution": dict(sorted(tiers.items())),
        "loss-histogram": dict(sorted(loss_hist.items())),
        "patterns": sum(r.get("patterns", 0) for r in processed),
        "checkpoints": sum(r.get("checkpoints", 0) for r in processed),
        "artifacts": sum(r.get("artifacts", 0) for r in processed),
        "artifact-exists": sum(r.get("artifact-exists", 0) for r in processed),
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mission-path", action="append", type=Path, default=[], help="Analyze one mission path regardless of completed classifier.")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    patterns = SCOPE.load_pattern_index()
    pattern_index = {p["name"]: p["ref"] for p in patterns}
    if args.mission_path:
        missions = [p if p.is_absolute() else Path.cwd() / p for p in args.mission_path]
        full_run = False
    else:
        missions = discover_missions()
        full_run = True
    if args.limit:
        missions = missions[: args.limit]
    results = []
    for path in missions:
        result = analyze_mission(path, pattern_index, args.force, args.out_dir, full_run)
        results.append(result)
        if result and not result.get("skipped") and not result.get("skipped-existing"):
            print(f"{result['mission']}: tier={result['hole-tier']} patterns={result['patterns']} checkpoints={result['checkpoints']} loss={result['loss']}", file=sys.stderr)
    summary = summarize(results)
    summary["out-dir"] = str(args.out_dir)
    summary["mode"] = "single" if args.mission_path else "completed-corpus"
    (args.out_dir / "_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "results": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
