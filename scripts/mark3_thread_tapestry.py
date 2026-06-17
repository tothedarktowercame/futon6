#!/usr/bin/env python3
"""mark4 — concept threads woven through the tapestry.

Beyond H12's "usage rose over time": this traces a concept's concrete TRAJECTORY
across papers (the spec's *thread* — per-concept phylogeny, "git-blame for a
concept") and types each activation by HOW the concept reached that paper, woven
over the import/export *tapestry* (H11's citation graph):

  - definition          : the thread origin (concept appears in a definiendum/definiens mark)
  - cited-activation    : a later paper uses it AND cites a prior user  -> explicit import
                          (an H11 edge). Lightning along a wire.
  - uncited-activation  : next paper to use it with NO citation to any prior user
                          -> the implicit concept-import. Lightning through the air.
  - redefinition        : a later paper that re-defines it (definiendum after first def)

`bursts` are years where many papers activate at once (the "lightning strike").
`cited_ratio` distinguishes citation-propagated concepts (well-attributed) from
folklore/assumed-background ones (mostly uncited) — the invisible edge the spec chases.

Substrate reused: mark3_diachronic (dated term extraction, same term set as the
prior) + mark3_xdoc_graph (H11 citation edges). The tapestry is H11's graph; a
cited-activation IS an H11 edge, an uncited-activation is a thread strand with none.

Usage:
  python scripts/mark3_thread_tapestry.py --concept "monoidal category"
  python scripts/mark3_thread_tapestry.py            # summary over encyclopedia concepts
  python scripts/mark3_thread_tapestry.py --self-test
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "data" / "showcases" / "ct-anatomy" / "golden"
CONCEPT_DIR = ROOT / "data" / "concept-encyclopedia" / "ct"
CITES = ROOT / "data" / "warp" / "cite-resolution"
DEFAULT_OUT = ROOT / "data" / "warp" / "concept-phylogeny.json"
DEF_MARK_KINDS = {"definiendum", "definiens"}


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_dia = _load("mark3_diachronic", "scripts/mark3_diachronic.py")
_xdoc = _load("mark3_xdoc_graph", "scripts/mark3_xdoc_graph.py")


def paper_date(pid: str):
    try:
        return _dia.parse_arxiv_month(pid)
    except Exception:
        return None


def scan_occurrences(golden_dir: Path, candidates: set[str], paper_filter: set[str] | None = None):
    """concept -> [{paper, date, is_def}]; uses the same extractor as the prior.
    A paper *defines* a concept if the concept surface appears inside a
    definiendum/definiens mark span in that paper."""
    word_re, _, _ = _dia.load_term_extractor()
    by_len: dict[int, set[str]] = defaultdict(set)
    for candidate in candidates:
        words = candidate.split()
        if words:
            by_len[len(words)].add(candidate)
    occ: dict[str, list] = defaultdict(list)
    papers = 0
    for path in sorted(golden_dir.glob("fable-*-dp-emacs.json")):
        pid = _dia.paper_id_from_path(path)
        if paper_filter is not None and pid not in paper_filter:
            continue
        date = paper_date(pid)
        if not date:
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        text = data.get("text", "")
        if not text:
            continue
        papers += 1
        words = word_re.findall(text.lower())
        present: set[str] = set()
        for n, wanted in by_len.items():
            if n > len(words):
                continue
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i + n])
                if phrase in wanted:
                    present.add(phrase)
        if not present:
            continue
        # definienda surfaces in this paper (text inside def-marks)
        def_text = " ".join(
            text[int(m["start"]):int(m["end"])].lower()
            for m in data.get("marks", [])
            if m.get("kind") in DEF_MARK_KINDS and "start" in m and "end" in m
        )
        for c in present:
            occ[c].append({"paper": pid, "date": date, "is_def": c in def_text})
    for c in occ:
        occ[c].sort(key=lambda r: (r["date"], r["paper"]))
    return occ, papers


def detect_bursts(dates: list[str]):
    by_year: dict[str, int] = defaultdict(int)
    for d in dates:
        by_year[d[:4]] += 1
    if len(by_year) < 2:
        return []
    counts = list(by_year.values())
    mean = statistics.mean(counts)
    sd = statistics.pstdev(counts) or 1.0
    return [{"year": y, "n_papers": n} for y, n in sorted(by_year.items())
            if n >= max(3, mean + sd)]


def build_thread(concept: str, occ: list, cites: dict):
    """Type each activation; cited iff the paper cites a prior user of the concept."""
    trajectory, prior_users = [], []
    n_cited = n_uncited = n_redef = 0
    for i, o in enumerate(occ):
        cited_ids = set(cites.get(o["paper"], []))   # paper -> [cited corpus-id, ...]
        cite_hit = next((p for p in prior_users if p in cited_ids), None)
        if i == 0:
            kind = "definition" if o["is_def"] else "first-use"
        elif o["is_def"]:
            kind = "redefinition"; n_redef += 1
        elif cite_hit:
            kind = "cited-activation"; n_cited += 1
        else:
            kind = "uncited-activation"; n_uncited += 1
        trajectory.append({"paper": o["paper"], "date": o["date"], "type": kind,
                           "cites-prior-user": cite_hit})
        prior_users.append(o["paper"])
    activations = n_cited + n_uncited
    return {
        "concept": concept,
        "n_papers": len(occ),
        "first_seen": occ[0]["date"] if occ else None,
        "trajectory": trajectory,
        "bursts": detect_bursts([o["date"] for o in occ]),
        "stats": {"cited_activations": n_cited, "uncited_activations": n_uncited,
                  "redefinitions": n_redef,
                  "cited_ratio": (n_cited / activations) if activations else None},
    }


def worked_example(threads: dict[str, dict]) -> dict | None:
    for concept in sorted(threads):
        trajectory = threads[concept].get("trajectory", [])
        for step in trajectory:
            if step.get("type") == "cited-activation" and step.get("cites-prior-user"):
                return {
                    "concept": concept,
                    "paper": step["paper"],
                    "via": "cited-activation",
                    "from_paper": step["cites-prior-user"],
                    "sentence": (
                        f"concept {concept!r} reached paper {step['paper']} "
                        f"via cited-activation from paper {step['cites-prior-user']}."
                    ),
                }
    return None


def artifact_payload(*, papers: int, candidates: int, threads: dict[str, dict]) -> dict:
    activation_counts = defaultdict(int)
    for thread in threads.values():
        for step in thread.get("trajectory", []):
            activation_counts[step.get("type", "unknown")] += 1
    return {
        "schema": "futon6/warp/concept-phylogeny/v1",
        "artifact": "data/warp/concept-phylogeny.json",
        "description": (
            "Per-concept citation-descent phylogeny: definition, cited-activation, "
            "uncited-activation, and redefinition events woven over the resolved "
            "citation graph."
        ),
        "cas_sel": {
            "role": "genealogical-select descent input",
            "relation": "a paper inherits its imports'/citations' concept patterns",
        },
        "r2d": {
            "role": "R2d-3 coupling candidate",
            "relation": "citation-descent edges can be checked against proof/warrant inheritance",
        },
        "summary": {
            "papers": papers,
            "concepts_with_threads": len(threads),
            "candidate_concepts": candidates,
            "activation_counts": dict(sorted(activation_counts.items())),
            "data_limit_note": (
                "Local CT sample; cited/uncited split sharpens at arXiv scale "
                "where citation overlap and concept reuse are dense."
            ),
        },
        "worked_example": worked_example(threads),
        "threads": threads,
    }


def run(args) -> dict:
    candidates = _dia.load_candidate_terms(args.concept_dir)
    raw_cites = _xdoc.load_cites(args.cites)
    citation_neighborhood = set(raw_cites)
    citation_neighborhood.update(
        r["corpus_id"]
        for recs in raw_cites.values()
        for r in recs
        if r.get("corpus_id")
    )
    occ, papers = scan_occurrences(args.golden_dir, candidates, paper_filter=citation_neighborhood)
    # normalize cite values to plain target-id lists
    cites = {p: [r["corpus_id"] for r in recs] for p, recs in raw_cites.items()}
    threads = {c: build_thread(c, occ[c], cites) for c in occ}
    if args.concept:
        key = args.concept.lower()
        out = threads.get(key) or {"concept": key, "error": "no occurrences"}
        print(json.dumps(out, indent=2))
        return out
    ranked = sorted(threads.values(),
                    key=lambda t: (len(t["bursts"]), t["n_papers"]), reverse=True)
    payload = artifact_payload(papers=papers, candidates=len(candidates), threads=threads)
    payload["summary"]["citation_neighborhood_papers"] = len(citation_neighborhood)
    summary = {
        "meta": payload["summary"],
        "top_threads": [
            {"concept": t["concept"], "n_papers": t["n_papers"], "first_seen": t["first_seen"],
             "bursts": t["bursts"], **t["stats"]}
            for t in ranked[:args.top_n]],
        "worked_example": payload["worked_example"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def self_test() -> int:
    import tempfile
    d = Path(tempfile.mkdtemp())
    g, c, conc = d / "g", d / "c", d / "ct"
    for p in (g, c, conc):
        p.mkdir()
    (conc / "snake-lemma.edn").write_text('{:concept/id :snake-lemma :name "snake lemma"}')
    # A (2000): defines it; B (2001): uses + cites A; C (2002): uses, no cite
    (g / "fable-0001.0001-dp-emacs.json").write_text(json.dumps(
        {"text": "snake lemma is defined here", "marks": [
            {"kind": "definiendum", "start": 0, "end": 11}]}))   # span == "snake lemma"
    (g / "fable-0101.0001-dp-emacs.json").write_text(json.dumps(
        {"text": "by the snake lemma we proceed", "marks": []}))
    (g / "fable-0201.0001-dp-emacs.json").write_text(json.dumps(
        {"text": "the snake lemma also applies", "marks": []}))
    (c / "0101.0001.cite-resolution.json").write_text(json.dumps(
        {"paper-id": "0101.0001", "records": [
            {"cite/marker": "[1]", "resolved-corpus-id": "0001.0001", "confidence": 0.9}]}))
    (c / "0201.0001.cite-resolution.json").write_text(json.dumps(
        {"paper-id": "0201.0001", "records": []}))
    args = argparse.Namespace(golden_dir=g, concept_dir=conc, cites=c, concept=None,
                              out=d / "out.json", top_n=10)
    run(args)
    occ, _ = scan_occurrences(g, {"snake lemma"})
    cites = {"0101.0001": ["0001.0001"]}
    th = build_thread("snake lemma", occ["snake lemma"], cites)
    types = [s["type"] for s in th["trajectory"]]
    ok = types == ["definition", "cited-activation", "uncited-activation"]
    print("SELF-TEST", "PASS" if ok else f"FAIL {types}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="mark4 concept thread / tapestry builder")
    ap.add_argument("--golden-dir", type=Path, default=GOLDEN)
    ap.add_argument("--concept-dir", type=Path, default=CONCEPT_DIR)
    ap.add_argument("--cites", type=Path, default=CITES)
    ap.add_argument("--concept", default=None, help="trace one concept's thread")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--top-n", type=int, default=15)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    return self_test() if a.self_test else (run(a) and 0) or 0


if __name__ == "__main__":
    raise SystemExit(main())
