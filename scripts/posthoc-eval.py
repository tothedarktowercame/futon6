#!/usr/bin/env python3
"""Post-hoc evaluation harness for superpod outputs.

Combines internal quality checks with externally relevant "readiness" probes.

Designed to run after a superpod job directory is produced.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


DEFINITIONAL_SOURCES = {
    "is-called",
    "called-as",
    "defined-as",
    "definition-of",
    "definition-block-subject",
    "latex-emph",
}

NOISY_TERMS = {
    "weak", "strong", "which", "define", "condition", "morphisms",
    "acknowledgements", "cells", "result", "example", "theorem",
}


def _now_utc():
    return datetime.now(timezone.utc).isoformat()


def _safe_median(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(statistics.median(vals))


def _safe_mean(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _iter_json_array_lines(path: Path):
    """Iterate objects from compact JSON-array files written by superpod."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s in {"[", "]", ","}:
                continue
            if s.startswith(","):
                s = s[1:].strip()
            if s.endswith(","):
                s = s[:-1].strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                continue


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                continue


def _jaccard(a, b):
    if not a and not b:
        return 0.0
    i = len(a & b)
    u = len(a | b)
    return i / u if u else 0.0


def analyze_hypergraphs(hg_path: Path, max_graphs: int = 0):
    if not hg_path.exists():
        return {
            "exists": False,
            "error": f"{hg_path} not found",
        }

    n_graphs = 0
    n_nodes = 0
    n_edges = 0
    empty_graphs = 0
    node_type_counts = Counter()
    edge_type_counts = Counter()
    paper_terms = defaultdict(set)        # thread_id -> terms
    term_df = Counter()                   # term -> papers
    paper_scope_types = defaultdict(set)  # thread_id -> scope subtypes
    paper_primary_cat = {}

    for hg in _iter_json_array_lines(hg_path):
        n_graphs += 1
        if max_graphs and n_graphs > max_graphs:
            break

        tid = str(hg.get("thread_id", f"graph-{n_graphs}"))
        nodes = hg.get("nodes", []) or []
        edges = hg.get("edges", []) or []

        n_nodes += len(nodes)
        n_edges += len(edges)
        if not nodes or not edges:
            empty_graphs += 1

        term_by_node_id = {}
        post_tags = []

        for node in nodes:
            ntype = node.get("type", "?")
            node_type_counts[ntype] += 1
            nid = node.get("id")
            subtype = node.get("subtype", "")
            if ntype == "term" and nid:
                term_by_node_id[nid] = subtype
            elif ntype == "scope":
                if subtype:
                    paper_scope_types[tid].add(subtype)
            elif ntype == "post":
                attrs = node.get("attrs", {}) or {}
                tags = attrs.get("tags") or []
                if isinstance(tags, list) and tags:
                    post_tags = tags

        if post_tags:
            paper_primary_cat[tid] = str(post_tags[0])

        for edge in edges:
            etype = edge.get("type", "?")
            edge_type_counts[etype] += 1
            if etype == "mention":
                ends = edge.get("ends") or []
                if len(ends) >= 2:
                    term_id = ends[1]
                    term = term_by_node_id.get(term_id)
                    if term:
                        paper_terms[tid].add(term)

    for tid, terms in paper_terms.items():
        for t in terms:
            term_df[t] += 1

    return {
        "exists": True,
        "n_graphs": n_graphs,
        "total_nodes": n_nodes,
        "total_edges": n_edges,
        "avg_nodes": (n_nodes / n_graphs) if n_graphs else 0.0,
        "avg_edges": (n_edges / n_graphs) if n_graphs else 0.0,
        "empty_graphs": empty_graphs,
        "empty_rate": (empty_graphs / n_graphs) if n_graphs else 0.0,
        "node_type_counts": dict(node_type_counts),
        "edge_type_counts": dict(edge_type_counts),
        "papers_with_terms": sum(1 for t in paper_terms.values() if t),
        "papers_with_scopes": sum(1 for s in paper_scope_types.values() if s),
        "paper_terms": paper_terms,
        "term_df": term_df,
        "paper_scope_types": paper_scope_types,
        "paper_primary_cat": paper_primary_cat,
    }


def analyze_candidate_terms(path: Path):
    if not path.exists():
        return {"exists": False, "error": f"{path} not found", "rows": []}

    rows = list(_iter_jsonl(path))
    one_word = 0
    noisy = 0
    for r in rows:
        term = str(r.get("term_lower", "")).strip().lower()
        if not term:
            continue
        if len(term.split()) == 1:
            one_word += 1
        if term in NOISY_TERMS:
            noisy += 1

    definitional = [
        r for r in rows
        if any(src in DEFINITIONAL_SOURCES for src in (r.get("sources") or {}).keys())
    ]
    ready = [
        r for r in definitional
        if int(r.get("entity_count", 0)) >= 2
    ]

    return {
        "exists": True,
        "count": len(rows),
        "one_word_ratio": (one_word / len(rows)) if rows else 0.0,
        "noisy_ratio": (noisy / len(rows)) if rows else 0.0,
        "definitional_count": len(definitional),
        "ready_count": len(ready),
        "top_ready_terms": [
            {
                "term": r.get("term_lower"),
                "entity_count": r.get("entity_count"),
                "candidate_count": r.get("candidate_count"),
                "sources": r.get("sources"),
            }
            for r in sorted(
                ready,
                key=lambda x: (-float(x.get("score", 0.0)), -int(x.get("entity_count", 0)))
            )[:20]
        ],
        "rows": rows,
    }


def analyze_mit_hits(path: Path):
    if not path.exists():
        return {"exists": False, "error": f"{path} not found"}
    rows = list(_iter_jsonl(path))
    labels = Counter(r.get("mit_label", "unknown") for r in rows)
    confs = [float(r.get("mit_confidence", 0.0)) for r in rows]
    likely = [r for r in rows if r.get("mit_label") == "likely-distinctor"]
    likely_entities = len({r.get("entity_id") for r in likely})
    return {
        "exists": True,
        "count": len(rows),
        "label_counts": dict(labels),
        "avg_confidence": _safe_mean(confs),
        "likely_distinctor_count": len(likely),
        "likely_distinctor_entities": likely_entities,
        "top_likely": [
            {
                "entity_id": r.get("entity_id"),
                "pair": r.get("pair"),
                "scope_type": r.get("scope_type"),
                "confidence": r.get("mit_confidence"),
                "rationale": r.get("mit_rationale"),
            }
            for r in sorted(
                likely, key=lambda x: -float(x.get("mit_confidence", 0.0))
            )[:20]
        ],
    }


def _load_arxiv_texts(metadata_jsonl: Path, paper_ids: set[str]):
    """Return map of entity_id ('arxiv-...') -> lowercased title+abstract."""
    if not metadata_jsonl or not metadata_jsonl.exists():
        return {}
    wanted = set(paper_ids)
    out = {}
    for row in _iter_jsonl(metadata_jsonl):
        aid = str(row.get("id", ""))
        if not aid:
            continue
        eid = f"arxiv-{aid}"
        if eid not in wanted:
            continue
        title = str(row.get("title", ""))
        abstract = str(row.get("abstract", ""))
        out[eid] = (title + " " + abstract).lower()
    return out


def external_term_query_probe(hg, metadata_jsonl: Path | None, max_terms: int = 150):
    """Externally relevant probe: concept-to-paper retrieval utility."""
    term_df = hg.get("term_df") or Counter()
    paper_terms = hg.get("paper_terms") or {}
    if not term_df or not paper_terms:
        return {"enabled": False, "reason": "no term graph available"}

    papers = set(paper_terms.keys())
    raw_text = _load_arxiv_texts(metadata_jsonl, papers) if metadata_jsonl else {}
    if not raw_text:
        return {"enabled": False, "reason": "metadata texts unavailable for baseline"}

    term_to_papers = defaultdict(set)
    for pid, terms in paper_terms.items():
        for term in terms:
            term_to_papers[term].add(pid)

    probe_terms = [
        t for t, df in term_df.items()
        if 2 <= int(df) <= 200 and len(t.split()) <= 4 and len(t) >= 4
    ]
    probe_terms.sort(key=lambda t: (-int(term_df[t]), t))
    probe_terms = probe_terms[:max_terms]

    proc_hits = []
    raw_hits = []
    lifts = []
    improved = 0
    novel = 0
    detailed = []
    showcase = []

    for t in probe_terms:
        pcount = int(term_df[t])
        raw_papers = {pid for pid, txt in raw_text.items() if t in txt}
        proc_papers = term_to_papers.get(t, set())
        rcount = len(raw_papers)
        proc_hits.append(pcount)
        raw_hits.append(rcount)
        if pcount > rcount:
            improved += 1
        if rcount == 0 and pcount > 0:
            novel += 1
        if rcount > 0:
            lifts.append(pcount / rcount)
        proc_only = sorted(proc_papers - raw_papers)
        if proc_only and len(showcase) < 20:
            showcase.append({
                "term": t,
                "processed_hits": pcount,
                "raw_exact_hits": rcount,
                "processed_only_papers": proc_only[:5],
            })
        detailed.append({
            "term": t,
            "processed_hits": pcount,
            "raw_exact_hits": rcount,
        })

    return {
        "enabled": True,
        "terms_probed": len(probe_terms),
        "processed_hits_median": _safe_median(proc_hits),
        "raw_exact_hits_median": _safe_median(raw_hits),
        "median_lift_over_raw_exact": _safe_median(lifts),
        "terms_improved_over_raw_exact": improved,
        "terms_novel_over_raw_exact": novel,
        "top_terms": detailed[:25],
        "showcase_processed_only_terms": showcase,
    }


def external_scope_neighbor_probe(hg, sample_size: int = 800, seed: int = 13):
    """Externally relevant probe: scope-signature nearest-neighbor relevance."""
    scope_sets = hg.get("paper_scope_types") or {}
    primary = hg.get("paper_primary_cat") or {}
    eligible = [
        pid for pid, scopes in scope_sets.items()
        if scopes and pid in primary and primary[pid]
    ]
    if len(eligible) < 20:
        return {"enabled": False, "reason": "insufficient scoped papers with category labels"}

    rnd = random.Random(seed)
    if len(eligible) > sample_size:
        eligible = rnd.sample(eligible, sample_size)

    same_cat_nn = 0
    same_cat_rand = 0
    total = 0
    sample = list(eligible)
    nearest_examples = []

    for i, pid in enumerate(sample):
        scopes_i = scope_sets[pid]
        cat_i = primary[pid]
        best_j = None
        best_score = -1.0
        for qid in sample:
            if qid == pid:
                continue
            s = _jaccard(scopes_i, scope_sets[qid])
            if s > best_score:
                best_score = s
                best_j = qid
        if best_j is None:
            continue
        total += 1
        if len(nearest_examples) < 25:
            nearest_examples.append({
                "paper_id": pid,
                "category": cat_i,
                "neighbor_id": best_j,
                "neighbor_category": primary.get(best_j),
                "scope_jaccard": round(best_score, 4),
            })
        if primary.get(best_j) == cat_i:
            same_cat_nn += 1
        # random baseline
        others = [x for x in sample if x != pid]
        rj = rnd.choice(others)
        if primary.get(rj) == cat_i:
            same_cat_rand += 1

    nn_rate = (same_cat_nn / total) if total else 0.0
    rand_rate = (same_cat_rand / total) if total else 0.0
    return {
        "enabled": True,
        "n_evaluated": total,
        "same_category_at1_scope_nn": nn_rate,
        "same_category_random_baseline": rand_rate,
        "lift_over_random": (nn_rate / rand_rate) if rand_rate > 0 else None,
        "nearest_neighbor_examples": nearest_examples,
    }


def _gate(name, status, detail):
    return {"name": name, "status": status, "detail": detail}


def build_internal_gates(manifest, hg, cand, mit):
    gates = []

    if manifest:
        gates.append(_gate("manifest_present", "pass", "manifest.json loaded"))
    else:
        gates.append(_gate("manifest_present", "fail", "manifest.json missing/unreadable"))
        return gates

    scope_cov = float((manifest.get("stage5_stats") or {}).get("scope_coverage", 0.0))
    if scope_cov >= 0.05:
        gates.append(_gate("scope_coverage", "pass", f"scope_coverage={scope_cov:.3f}"))
    elif scope_cov > 0:
        gates.append(_gate("scope_coverage", "warn", f"scope_coverage={scope_cov:.3f} (low)"))
    else:
        gates.append(_gate("scope_coverage", "fail", "scope_coverage=0"))

    if hg.get("exists"):
        if hg.get("n_graphs", 0) > 0:
            gates.append(_gate("hypergraphs_exist", "pass", f"{hg['n_graphs']} graphs"))
        else:
            gates.append(_gate("hypergraphs_exist", "fail", "hypergraphs file empty"))
    else:
        gates.append(_gate("hypergraphs_exist", "fail", hg.get("error", "missing hypergraphs.json")))

    edges = hg.get("edge_type_counts", {})
    required = ["mention", "scope", "surface"]
    missing = [e for e in required if edges.get(e, 0) == 0]
    if missing:
        gates.append(_gate("edge_type_coverage", "warn", f"missing edge types: {missing}"))
    else:
        gates.append(_gate("edge_type_coverage", "pass", f"edge types present: {required}"))

    empty_rate = float(hg.get("empty_rate", 1.0))
    if empty_rate <= 0.05:
        gates.append(_gate("empty_hypergraph_rate", "pass", f"empty_rate={empty_rate:.3f}"))
    elif empty_rate <= 0.20:
        gates.append(_gate("empty_hypergraph_rate", "warn", f"empty_rate={empty_rate:.3f}"))
    else:
        gates.append(_gate("empty_hypergraph_rate", "fail", f"empty_rate={empty_rate:.3f}"))

    train_metrics = ((manifest.get("stage9b_stats") or {}).get("train_metrics") or {})
    if train_metrics:
        acc1 = train_metrics.get("val_acc1_final")
        if acc1 is None:
            gates.append(_gate("gnn_validation_metric", "warn", "train_metrics present but val_acc1 missing"))
        else:
            gates.append(_gate("gnn_validation_metric", "pass", f"val_acc1={acc1:.3f}, val_acc5={train_metrics.get('val_acc5_final', 0):.3f}"))
    else:
        gates.append(_gate("gnn_validation_metric", "warn", "no stage9b train_metrics (graph embed skipped?)"))

    if cand.get("exists"):
        ready = int(cand.get("ready_count", 0))
        status = "pass" if ready >= 20 else ("warn" if ready > 0 else "warn")
        gates.append(_gate("open_world_term_readiness", status, f"ready_terms={ready}, total_candidates={cand.get('count',0)}"))
    else:
        gates.append(_gate("open_world_term_readiness", "warn", "candidate-new-terms.jsonl not found"))

    if mit.get("exists"):
        likely = int(mit.get("likely_distinctor_count", 0))
        status = "pass" if likely > 0 else "warn"
        gates.append(_gate("mit_distinctor_signal", status, f"likely_distinctor={likely}, total_hits={mit.get('count',0)}"))
    else:
        gates.append(_gate("mit_distinctor_signal", "warn", "distinctor-mit-hits.jsonl not found"))

    return gates


def render_markdown(report: dict) -> str:
    lines = []
    lines.append("# Post-hoc Evaluation Report")
    lines.append("")
    lines.append(f"- Generated: `{report.get('generated_utc','')}`")
    lines.append(f"- Output dir: `{report.get('outdir','')}`")
    lines.append("")

    lines.append("## Internal Gates")
    lines.append("")
    for g in report.get("internal_gates", []):
        lines.append(f"- `{g['status'].upper()}` {g['name']}: {g['detail']}")
    lines.append("")

    hg = report.get("internal", {}).get("hypergraphs", {})
    if hg.get("exists"):
        lines.append("## Hypergraph Summary")
        lines.append("")
        lines.append(f"- graphs: `{hg.get('n_graphs',0)}`")
        lines.append(f"- avg_nodes: `{round(hg.get('avg_nodes',0.0),3)}`")
        lines.append(f"- avg_edges: `{round(hg.get('avg_edges',0.0),3)}`")
        lines.append(f"- edge_type_counts: `{hg.get('edge_type_counts',{})}`")
        lines.append("")

    ext = report.get("external", {})
    lines.append("## External Relevance Probes")
    lines.append("")
    term_probe = ext.get("term_query_probe", {})
    lines.append(f"- Term-query utility enabled: `{term_probe.get('enabled', False)}`")
    if term_probe.get("enabled"):
        lines.append(
            f"  terms_probed={term_probe.get('terms_probed',0)}, "
            f"median_lift={term_probe.get('median_lift_over_raw_exact')}, "
            f"improved={term_probe.get('terms_improved_over_raw_exact',0)}, "
            f"novel={term_probe.get('terms_novel_over_raw_exact',0)}"
        )
        for row in term_probe.get("showcase_processed_only_terms", [])[:10]:
            lines.append(
                f"  showcase term `{row.get('term')}`: "
                f"processed={row.get('processed_hits')} raw={row.get('raw_exact_hits')} "
                f"processed_only_papers={row.get('processed_only_papers')}"
            )
    else:
        lines.append(f"  reason={term_probe.get('reason','n/a')}")

    scope_probe = ext.get("scope_neighbor_probe", {})
    lines.append(f"- Scope-neighbor relevance enabled: `{scope_probe.get('enabled', False)}`")
    if scope_probe.get("enabled"):
        lines.append(
            f"  same_cat@1={round(scope_probe.get('same_category_at1_scope_nn',0.0),4)}, "
            f"random={round(scope_probe.get('same_category_random_baseline',0.0),4)}, "
            f"lift={scope_probe.get('lift_over_random')}"
        )
        for row in scope_probe.get("nearest_neighbor_examples", [])[:10]:
            lines.append(
                f"  nn `{row.get('paper_id')}` ({row.get('category')}) -> "
                f"`{row.get('neighbor_id')}` ({row.get('neighbor_category')}) "
                f"jaccard={row.get('scope_jaccard')}"
            )
    else:
        lines.append(f"  reason={scope_probe.get('reason','n/a')}")
    lines.append("")

    cand = report.get("internal", {}).get("candidate_terms", {})
    if cand.get("exists"):
        lines.append("## Glossary Readiness")
        lines.append("")
        lines.append(
            f"- candidates={cand.get('count',0)}, "
            f"definitional={cand.get('definitional_count',0)}, "
            f"ready={cand.get('ready_count',0)}"
        )
        for row in cand.get("top_ready_terms", [])[:15]:
            lines.append(
                f"- `{row.get('term')}` "
                f"(entity_count={row.get('entity_count')}, candidate_count={row.get('candidate_count')})"
            )
        lines.append("")

    mit = report.get("internal", {}).get("mit_hits", {})
    if mit.get("exists"):
        lines.append("## MIT Triage Readiness")
        lines.append("")
        lines.append(
            f"- total_hits={mit.get('count',0)}, "
            f"likely_distinctor={mit.get('likely_distinctor_count',0)}, "
            f"entities={mit.get('likely_distinctor_entities',0)}"
        )
        lines.append("")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Post-hoc evaluation harness for superpod outputs")
    parser.add_argument("outdir", help="Superpod output directory")
    parser.add_argument("--metadata-jsonl", default=None,
                        help="Optional raw metadata JSONL for external baseline (defaults to manifest arxiv_jsonl)")
    parser.add_argument("--candidate-terms", default=None,
                        help="Path to candidate-new-terms.jsonl (default: <outdir>/candidate-new-terms.jsonl)")
    parser.add_argument("--mit-hits", default=None,
                        help="Path to distinctor-mit-hits.jsonl (default: <outdir>/distinctor-mit-hits.jsonl)")
    parser.add_argument("--report-json", default=None,
                        help="Output JSON path (default: <outdir>/posthoc-eval.json)")
    parser.add_argument("--report-md", default=None,
                        help="Output markdown path (default: <outdir>/posthoc-eval.md)")
    parser.add_argument("--max-hypergraphs", type=int, default=0,
                        help="Optional cap on hypergraphs loaded (0 = all)")
    parser.add_argument("--scope-neighbor-sample", type=int, default=800,
                        help="Sample size for scope-neighbor relevance probe")
    parser.add_argument("--term-probe-max", type=int, default=150,
                        help="Max terms for term-query probe")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    manifest_path = outdir / "manifest.json"
    manifest = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = None

    hg_path = outdir / "hypergraphs.json"
    cand_path = Path(args.candidate_terms) if args.candidate_terms else (outdir / "candidate-new-terms.jsonl")
    mit_path = Path(args.mit_hits) if args.mit_hits else (outdir / "distinctor-mit-hits.jsonl")

    metadata_path = None
    if args.metadata_jsonl:
        metadata_path = Path(args.metadata_jsonl)
    elif manifest and manifest.get("arxiv_jsonl"):
        metadata_path = Path(manifest["arxiv_jsonl"])

    hg = analyze_hypergraphs(hg_path, max_graphs=args.max_hypergraphs)
    cand = analyze_candidate_terms(cand_path)
    mit = analyze_mit_hits(mit_path)

    internal = {
        "manifest_present": manifest is not None,
        "manifest_stage5_stats": (manifest or {}).get("stage5_stats"),
        "manifest_stage9a_stats": (manifest or {}).get("stage9a_stats"),
        "manifest_stage9b_stats": (manifest or {}).get("stage9b_stats"),
        "hypergraphs": {k: v for k, v in hg.items() if k not in {"paper_terms", "term_df", "paper_scope_types", "paper_primary_cat"}},
        "candidate_terms": {k: v for k, v in cand.items() if k != "rows"},
        "mit_hits": mit,
    }

    ext_term = external_term_query_probe(hg, metadata_path, max_terms=args.term_probe_max)
    ext_scope = external_scope_neighbor_probe(hg, sample_size=args.scope_neighbor_sample)
    external = {
        "term_query_probe": ext_term,
        "scope_neighbor_probe": ext_scope,
    }

    gates = build_internal_gates(manifest, hg, cand, mit)
    status_counts = Counter(g["status"] for g in gates)
    overall = "pass"
    if status_counts.get("fail", 0) > 0:
        overall = "fail"
    elif status_counts.get("warn", 0) > 0:
        overall = "warn"

    report = {
        "generated_utc": _now_utc(),
        "outdir": str(outdir),
        "overall_status": overall,
        "gate_status_counts": dict(status_counts),
        "internal_gates": gates,
        "internal": internal,
        "external": external,
    }

    out_json = Path(args.report_json) if args.report_json else (outdir / "posthoc-eval.json")
    out_md = Path(args.report_md) if args.report_md else (outdir / "posthoc-eval.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    print(f"[posthoc] wrote {out_json}")
    print(f"[posthoc] wrote {out_md}")
    print(f"[posthoc] overall={overall} gates={dict(status_counts)}")
    if ext_term.get("enabled"):
        print(f"[posthoc] term_query_probe: median_lift={ext_term.get('median_lift_over_raw_exact')} "
              f"improved={ext_term.get('terms_improved_over_raw_exact')}/{ext_term.get('terms_probed')}")
    if ext_scope.get("enabled"):
        print(f"[posthoc] scope_neighbor_probe: same_cat@1={ext_scope.get('same_category_at1_scope_nn'):.4f} "
              f"vs random={ext_scope.get('same_category_random_baseline'):.4f}")


if __name__ == "__main__":
    main()
