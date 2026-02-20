#!/usr/bin/env python3
"""Build a small, concrete hypergraph showcase pack from local corpora.

Sources:
- data/stackexchange-samples/*.jsonl (thread-level examples)
- data/arxiv-math-ct-eprints/*.tar.gz + metadata JSONL (paper source snippet)

Outputs:
- data/showcases/hypergraph-showcase.json
- data/showcases/hypergraph-showcase.md
"""

from __future__ import annotations

import argparse
import importlib
import json
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys
import re



def _load_modules():
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "scripts"))
    sys.path.insert(0, str(root / "src"))
    assemble_wiring = importlib.import_module("assemble-wiring")
    nlab_wiring = importlib.import_module("nlab-wiring")
    from futon6.hypergraph import assemble as assemble_hypergraph
    return assemble_wiring, nlab_wiring, assemble_hypergraph



def _thread_for_wiring(sample_thread: dict) -> dict:
    q = sample_thread["question"]
    return {
        "id": q["id"],
        "body": q.get("body_text", ""),
        "title": q.get("title", ""),
        "tags": q.get("tags", []),
        "score": q.get("score", 0),
        "site": sample_thread.get("site", ""),
        "topic": sample_thread.get("topic", ""),
        "answers": [
            {
                "id": a["id"],
                "body": a.get("body_text", ""),
                "score": a.get("score", 0),
                "is_accepted": a.get("id") == q.get("accepted_answer_id"),
            }
            for a in sample_thread.get("answers", [])
        ],
        "comments": sample_thread.get("comments", {}),
    }



def _thread_for_hypergraph(sample_thread: dict) -> dict:
    q = sample_thread["question"]
    return {
        "question": {
            "id": q["id"],
            "title": q.get("title", ""),
            "score": q.get("score", 0),
            "tags": q.get("tags", []),
            "body_html": q.get("body_html", ""),
        },
        "answers": [
            {
                "id": a["id"],
                "score": a.get("score", 0),
                "body_html": a.get("body_html", ""),
            }
            for a in sample_thread.get("answers", [])
        ],
        "comments_q": [
            {
                "id": c["id"],
                "score": c.get("score", 0),
                "text": c.get("text", ""),
            }
            for c in sample_thread.get("comments", {}).get("question", [])
        ],
        "comments_a": {
            str(aid): [
                {
                    "id": c["id"],
                    "score": c.get("score", 0),
                    "text": c.get("text", ""),
                }
                for c in clist
            ]
            for aid, clist in sample_thread.get("comments", {}).get("answers", {}).items()
        },
    }



def _bridge_terms(hg: dict, qid: str, answer_ids: set[str]) -> list[str]:
    term_to_posts: dict[str, set[str]] = defaultdict(set)
    for e in hg.get("edges", []):
        if e.get("type") == "mention" and len(e.get("ends", [])) == 2:
            post_id, term_id = e["ends"]
            term_to_posts[term_id].add(post_id)

    bridge = []
    for term_id, posts in term_to_posts.items():
        if qid in posts and posts & answer_ids:
            bridge.append(term_id)
    bridge.sort()
    return bridge



def _scope_symbol_reuse(hg: dict) -> dict[str, int]:
    symbols: dict[str, set[str]] = defaultdict(set)
    for n in hg.get("nodes", []):
        if n.get("type") != "scope":
            continue
        for end in n.get("attrs", {}).get("ends", []):
            if end.get("role") == "symbol":
                sym = end.get("latex") or end.get("text")
                if sym:
                    symbols[sym].add(n.get("id"))
    return {sym: len(ids) for sym, ids in symbols.items() if len(ids) >= 2}



def _build_thread_candidates(
    sample_glob: str,
    ct_reference: Path,
    ner_kernel: Path,
):
    assemble_wiring, nlab_wiring, assemble_hypergraph = _load_modules()

    reference = json.loads(ct_reference.read_text())
    singles, multi_index, _ = nlab_wiring.load_ner_kernel(str(ner_kernel))

    candidates = []
    for path in sorted(Path().glob(sample_glob)):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                thread_w = _thread_for_wiring(item)
                wiring = assemble_wiring.build_thread_graph(
                    thread_w, reference, singles, multi_index)

                raw = _thread_for_hypergraph(item)
                hg = assemble_hypergraph(raw, {"nodes": wiring["nodes"], "edges": wiring["edges"]})

                qid = f"q-{item['question']['id']}"
                answer_ids = {f"a-{a['id']}" for a in item.get("answers", [])}
                bridge_terms = _bridge_terms(hg, qid, answer_ids)

                scope_types = Counter(
                    n.get("subtype", "?")
                    for n in hg.get("nodes", [])
                    if n.get("type") == "scope"
                )
                iatc = Counter(
                    e.get("attrs", {}).get("act", "?")
                    for e in hg.get("edges", [])
                    if e.get("type") == "iatc"
                )
                reuse = _scope_symbol_reuse(hg)

                candidates.append({
                    "source_file": path.name,
                    "thread_id": item.get("thread_id"),
                    "site": item.get("site"),
                    "topic": item.get("topic"),
                    "url": item["question"].get("url"),
                    "title": item["question"].get("title", ""),
                    "question_excerpt": item["question"].get("body_text", "")[:280],
                    "answer_excerpt": (item.get("answers") or [{}])[0].get("body_text", "")[:280],
                    "stats": {
                        "wiring_nodes": wiring["stats"].get("n_nodes"),
                        "wiring_edges": wiring["stats"].get("n_edges"),
                        "categorical_edges": wiring["stats"].get("n_categorical"),
                        "hypergraph_nodes": hg["meta"].get("n_nodes"),
                        "hypergraph_edges": hg["meta"].get("n_edges"),
                        "scope_nodes": hg["meta"].get("n_scopes"),
                        "expression_nodes": hg["meta"].get("n_expressions"),
                    },
                    "scope_type_top": scope_types.most_common(8),
                    "iatc_top": iatc.most_common(6),
                    "shared_terms_q_to_a": bridge_terms[:16],
                    "shared_term_count": len(bridge_terms),
                    "reused_scope_symbols": sorted(reuse.items(), key=lambda kv: (-kv[1], kv[0]))[:12],
                })
    return candidates



def _pick_showcases(candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []

    chosen = []
    used = set()

    def take_one(key_fn, pred=None):
        best = None
        for c in sorted(candidates, key=key_fn, reverse=True):
            if c["thread_id"] in used:
                continue
            if pred is not None and not pred(c):
                continue
            best = c
            break
        if best:
            used.add(best["thread_id"])
            chosen.append(best)

    # 1) Strong overall motif density.
    take_one(lambda c: (c["stats"]["scope_nodes"], c["shared_term_count"], c["stats"]["categorical_edges"]))

    # 2) Ensure one Math.SE category-theory thread.
    take_one(
        lambda c: (c["shared_term_count"], c["stats"]["scope_nodes"], c["stats"]["categorical_edges"]),
        pred=lambda c: c.get("source_file") == "math.stackexchange.com__category-theory.jsonl",
    )

    # 3) Ensure one MO category-theory thread.
    take_one(
        lambda c: (c["stats"]["categorical_edges"], c["stats"]["scope_nodes"], c["shared_term_count"]),
        pred=lambda c: c.get("source_file") == "mathoverflow.net__category-theory.jsonl",
    )

    # 4) Ensure one mathematical-physics thread.
    take_one(
        lambda c: (c["shared_term_count"], c["stats"]["scope_nodes"], c["stats"]["expression_nodes"]),
        pred=lambda c: "physics" in (c.get("topic") or ""),
    )

    # 5) Symbol reuse case from remaining candidates.
    take_one(lambda c: (len(c["reused_scope_symbols"]), c["stats"]["scope_nodes"], c["shared_term_count"]))

    return chosen



def _metadata_map(metadata_jsonl: Path) -> dict[str, dict]:
    m = {}
    if not metadata_jsonl.exists():
        return m
    with metadata_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            m[obj["id"]] = obj
    return m



def _scan_arxiv_tex_example(eprint_dir: Path, metadata_jsonl: Path, max_files: int = 600):
    _, nlab_wiring, _ = _load_modules()

    md = _metadata_map(metadata_jsonl)
    env_pat = re.compile(r"\\begin\{(theorem|lemma|proposition|corollary|proof|definition|defn)\}", re.I)
    binder_pat = re.compile(r"\\(forall|exists|sum|prod|int|coprod|bigcup|bigcap)\b")

    files = sorted(eprint_dir.glob("*.tar.gz"))
    checked = 0
    for p in files[:max_files]:
        checked += 1
        try:
            with tarfile.open(p, "r:gz") as tf:
                members = [
                    m for m in tf.getmembers()
                    if m.isfile() and m.name.lower().endswith(".tex") and m.size < 2_000_000
                ]
                for m in members[:10]:
                    fh = tf.extractfile(m)
                    if fh is None:
                        continue
                    text = fh.read().decode("utf-8", errors="ignore")
                    em = env_pat.search(text)
                    bm = binder_pat.search(text)
                    if not em or not bm:
                        continue

                    start = max(0, min(em.start(), bm.start()) - 220)
                    end = min(len(text), max(em.end(), bm.end()) + 260)
                    snippet = text[start:end]
                    snippet = " ".join(snippet.split())

                    arxiv_id = p.name.removesuffix(".tar.gz").replace("__", "/")
                    meta = md.get(arxiv_id, {})
                    scopes = nlab_wiring.detect_scopes(f"arxiv:{arxiv_id}", snippet)
                    scope_types = Counter(s.get("hx/type", "?") for s in scopes)

                    return {
                        "source": "arxiv-math-ct-eprint",
                        "arxiv_id": arxiv_id,
                        "title": meta.get("title", ""),
                        "url": meta.get("url", f"https://arxiv.org/abs/{arxiv_id}"),
                        "tar_file": p.name,
                        "tex_member": m.name,
                        "detected_env": em.group(1),
                        "detected_binder": bm.group(1),
                        "scope_type_top": scope_types.most_common(10),
                        "snippet": snippet[:700],
                        "checked_tar_files": checked,
                    }
        except Exception:
            continue

    return {
        "source": "arxiv-math-ct-eprint",
        "error": "no theorem+binder snippet found in scanned files",
        "checked_tar_files": checked,
    }



def _write_markdown(path: Path, payload: dict):
    lines = []
    lines.append("# Hypergraph Showcase Pack")
    lines.append("")
    lines.append(f"Generated: `{payload['generated_utc']}`")
    lines.append("")
    lines.append("## Why These Cases")
    lines.append("These examples highlight cross-scope reuse, cross-post wiring, and symbolic binders as first-class scope records feeding typed hypergraphs.")

    for i, case in enumerate(payload.get("thread_showcases", []), start=1):
        stats = case["stats"]
        lines.append("")
        lines.append(f"## Case {i}: {case['title']}")
        lines.append("")
        lines.append(f"- Source: `{case['source_file']}`")
        lines.append(f"- Thread: `{case['thread_id']}`")
        lines.append(f"- URL: {case['url']}")
        lines.append(f"- Stats: scopes={stats['scope_nodes']}, expressions={stats['expression_nodes']}, categorical={stats['categorical_edges']}, shared_q_to_a_terms={case['shared_term_count']}")
        if case.get("scope_type_top"):
            lines.append("- Top scope types: " + ", ".join(f"`{t}`:{n}" for t, n in case["scope_type_top"][:6]))
        if case.get("reused_scope_symbols"):
            lines.append("- Reused scope symbols: " + ", ".join(f"`{s}`x{n}" for s, n in case["reused_scope_symbols"][:6]))
        if case.get("shared_terms_q_to_a"):
            lines.append("- Shared term-node IDs (Q↔A bridge): " + ", ".join(f"`{t}`" for t in case["shared_terms_q_to_a"][:8]))
        lines.append("- Question excerpt:")
        lines.append("```text")
        lines.append(case.get("question_excerpt", ""))
        lines.append("```")

    arxiv = payload.get("arxiv_tex_showcase")
    if arxiv:
        lines.append("")
        lines.append("## ArXiv TeX Case")
        if arxiv.get("error"):
            lines.append(f"- Error: {arxiv['error']}")
            lines.append(f"- Scanned tar files: {arxiv.get('checked_tar_files', 0)}")
        else:
            lines.append(f"- arXiv ID: `{arxiv['arxiv_id']}`")
            lines.append(f"- Title: {arxiv.get('title','')}")
            lines.append(f"- URL: {arxiv.get('url','')}")
            lines.append(f"- TeX member: `{arxiv.get('tex_member','')}` in `{arxiv.get('tar_file','')}`")
            lines.append(f"- Signals: environment=`{arxiv.get('detected_env')}`, binder=`{arxiv.get('detected_binder')}`")
            if arxiv.get("scope_type_top"):
                lines.append("- Detected scope types: " + ", ".join(f"`{t}`:{n}" for t, n in arxiv["scope_type_top"][:8]))
            lines.append("- Snippet:")
            lines.append("```tex")
            lines.append(arxiv.get("snippet", ""))
            lines.append("```")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> int:
    parser = argparse.ArgumentParser(description="Build hypergraph showcase pack from local corpora")
    parser.add_argument("--sample-glob", default="data/stackexchange-samples/*.jsonl")
    parser.add_argument("--ct-reference", default="data/nlab-ct-reference.json")
    parser.add_argument("--ner-kernel", default="data/ner-kernel/terms.tsv")
    parser.add_argument("--arxiv-metadata", default="data/arxiv-math-ct-metadata.jsonl")
    parser.add_argument("--eprint-dir", default="data/arxiv-math-ct-eprints")
    parser.add_argument("--max-eprint-scan", type=int, default=600)
    parser.add_argument("--output-dir", default="data/showcases")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    candidates = _build_thread_candidates(
        sample_glob=args.sample_glob,
        ct_reference=Path(args.ct_reference),
        ner_kernel=Path(args.ner_kernel),
    )
    showcases = _pick_showcases(candidates)

    arxiv_case = _scan_arxiv_tex_example(
        eprint_dir=Path(args.eprint_dir),
        metadata_jsonl=Path(args.arxiv_metadata),
        max_files=args.max_eprint_scan,
    )

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "thread_candidates_total": len(candidates),
        "thread_showcases": showcases,
        "arxiv_tex_showcase": arxiv_case,
    }

    json_path = outdir / "hypergraph-showcase.json"
    md_path = outdir / "hypergraph-showcase.md"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)

    print(f"[showcase] wrote {json_path}")
    print(f"[showcase] wrote {md_path}")
    print(f"[showcase] thread candidates: {len(candidates)}")
    print(f"[showcase] selected cases: {len(showcases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
