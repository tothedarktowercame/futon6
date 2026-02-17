#!/usr/bin/env python3
"""Evaluate superpod run output: pipeline health + LWGM quality.

Run locally on the returned tarballs. Produces a JSON report and
prints a human-readable summary.

Usage:
    # Evaluate GPU run (has both text + structural embeddings):
    python scripts/evaluate-superpod-run.py math-processed-gpu/

    # Evaluate CPU run (stages 1-9a only, no embeddings):
    python scripts/evaluate-superpod-run.py math-processed/ --cpu-only

    # Compare CPU and GPU runs side by side:
    python scripts/evaluate-superpod-run.py math-processed-gpu/ \
        --compare math-processed/

    # Export cross-domain candidates for human review:
    python scripts/evaluate-superpod-run.py math-processed-gpu/ \
        --export-review cross-domain-candidates.json --n-review 50
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 1. Pipeline health
# ---------------------------------------------------------------------------

def check_pipeline_health(outdir: Path) -> dict:
    """Check manifest for completeness and stage-level health."""
    manifest_path = outdir / "manifest.json"
    if not manifest_path.exists():
        return {"ok": False, "error": "manifest.json not found"}

    manifest = json.loads(manifest_path.read_text())
    stages = manifest.get("stages_completed", [])

    report = {
        "ok": True,
        "stages_completed": stages,
        "n_stages": len(stages),
        "elapsed_seconds": manifest.get("elapsed_seconds"),
        "entity_count": manifest.get("entity_count"),
        "source": manifest.get("source"),
    }

    # Stage-level stats
    for key in ["stage5_stats", "stage7_stats", "stage8_stats",
                "stage9a_stats", "stage9b_stats", "stage10_stats"]:
        if manifest.get(key):
            report[key] = manifest[key]

    # Check critical invariants
    s7 = manifest.get("stage7_stats") or {}
    if not s7.get("ct_backed"):
        report["warnings"] = report.get("warnings", [])
        report["warnings"].append("stage7 not CT-backed")

    s8 = manifest.get("stage8_stats") or {}
    if s8.get("parse_rate", 1.0) < 0.80:
        report["warnings"] = report.get("warnings", [])
        report["warnings"].append(
            f"expression parse rate {s8['parse_rate']:.1%} < 80%")

    s9a = manifest.get("stage9a_stats") or {}
    if s9a:
        produced = s9a.get("hypergraphs_produced", 0)
        processed = s9a.get("threads_processed", 1)
        rate = produced / processed if processed else 0
        if rate < 0.95:
            report["warnings"] = report.get("warnings", [])
            report["warnings"].append(
                f"hypergraph assembly rate {rate:.1%} < 95%")

    return report


# ---------------------------------------------------------------------------
# 2. Embedding quality (degeneracy checks)
# ---------------------------------------------------------------------------

def check_embedding_quality(emb_path: Path, name: str) -> dict:
    """Check an embedding matrix for degeneracy."""
    if not emb_path.exists():
        return {"name": name, "ok": False, "error": f"{emb_path} not found"}

    emb = np.load(str(emb_path))
    n, d = emb.shape

    # Norm statistics
    norms = np.linalg.norm(emb, axis=1)

    # Isotropy: average pairwise cosine similarity (sample for speed)
    rng = np.random.default_rng(42)
    sample_size = min(1000, n)
    idx = rng.choice(n, sample_size, replace=False)
    sample = emb[idx]
    sample_normed = sample / (np.linalg.norm(sample, axis=1, keepdims=True) + 1e-8)
    cos_matrix = sample_normed @ sample_normed.T
    # Exclude diagonal
    mask = ~np.eye(sample_size, dtype=bool)
    avg_cos = cos_matrix[mask].mean()

    # Effective dimensionality (PCA explained variance)
    centered = emb - emb.mean(axis=0)
    # Use SVD on sample for speed
    sample_centered = centered[idx]
    _, s, _ = np.linalg.svd(sample_centered, full_matrices=False)
    variance_explained = (s ** 2) / (s ** 2).sum()
    cumvar = np.cumsum(variance_explained)
    eff_dim_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    eff_dim_95 = int(np.searchsorted(cumvar, 0.95)) + 1

    report = {
        "name": name,
        "ok": True,
        "shape": list(emb.shape),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "avg_pairwise_cosine": float(avg_cos),
        "effective_dim_90pct": eff_dim_90,
        "effective_dim_95pct": eff_dim_95,
    }

    # Degeneracy flags
    if avg_cos > 0.9:
        report["ok"] = False
        report["warning"] = "DEGENERATE: avg pairwise cosine > 0.9 (all vectors nearly identical)"
    elif avg_cos > 0.5:
        report["warning"] = f"HIGH avg pairwise cosine ({avg_cos:.3f}) — embeddings may be collapsing"

    if eff_dim_90 < 5:
        report["ok"] = False
        report["warning"] = report.get("warning", "") + f" DEGENERATE: 90% variance in {eff_dim_90} dims"

    return report


# ---------------------------------------------------------------------------
# 3. Structural vs text embedding comparison
# ---------------------------------------------------------------------------

def compare_embeddings(outdir: Path, n_sample: int = 500, k: int = 10) -> dict:
    """Compare LWGM structural neighbours vs text-embedding neighbours.

    For each sample thread, find k nearest neighbours under both embeddings
    and compare tag overlap (Jaccard similarity).
    """
    text_emb_path = outdir / "embeddings.npy"
    struct_emb_path = outdir / "hypergraph-embeddings.npy"
    entities_path = outdir / "entities.json"

    for p in [text_emb_path, struct_emb_path, entities_path]:
        if not p.exists():
            return {"ok": False, "error": f"{p} not found"}

    text_emb = np.load(str(text_emb_path))
    struct_emb = np.load(str(struct_emb_path))
    entities = json.loads(entities_path.read_text())

    # Both should have same number of rows (one per thread)
    n_text = text_emb.shape[0]
    n_struct = struct_emb.shape[0]

    # They might not be equal if some threads failed hypergraph assembly.
    # Use structural thread IDs to align.
    ids_path = outdir / "hypergraph-thread-ids.json"
    if ids_path.exists():
        struct_ids = json.loads(ids_path.read_text())
    else:
        struct_ids = list(range(n_struct))

    # Build tag lookup from entities
    tags_by_idx = {}
    for i, ent in enumerate(entities):
        tags_by_idx[i] = set(ent.get("tags", []))

    # Normalize both embedding matrices
    text_normed = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-8)
    struct_normed = struct_emb / (np.linalg.norm(struct_emb, axis=1, keepdims=True) + 1e-8)

    # Sample threads that have both embeddings
    rng = np.random.default_rng(42)
    sample_size = min(n_sample, n_struct)
    sample_idx = rng.choice(n_struct, sample_size, replace=False)

    text_jaccards = []
    struct_jaccards = []
    cross_domain_candidates = []

    for si in sample_idx:
        query_tags = tags_by_idx.get(si, set())
        if not query_tags:
            continue

        # Text embedding neighbours
        text_sims = text_normed[si] @ text_normed.T
        text_top = np.argsort(-text_sims)[1:k+1]
        text_tag_overlap = [
            _jaccard(query_tags, tags_by_idx.get(j, set()))
            for j in text_top
        ]
        text_jaccards.extend(text_tag_overlap)

        # Structural embedding neighbours
        struct_sims = struct_normed[si] @ struct_normed.T
        struct_top = np.argsort(-struct_sims)[1:k+1]
        struct_tag_overlap = [
            _jaccard(query_tags, tags_by_idx.get(j, set()))
            for j in struct_top
        ]
        struct_jaccards.extend(struct_tag_overlap)

        # Cross-domain candidates: high structural similarity, low tag overlap
        for rank, j in enumerate(struct_top[:5]):
            j_tags = tags_by_idx.get(j, set())
            sim = float(struct_sims[j])
            jac = _jaccard(query_tags, j_tags)
            if sim > 0.7 and jac < 0.1 and query_tags and j_tags:
                cross_domain_candidates.append({
                    "query_idx": int(si),
                    "query_tags": sorted(query_tags),
                    "neighbour_idx": int(j),
                    "neighbour_tags": sorted(j_tags),
                    "structural_similarity": round(sim, 4),
                    "tag_jaccard": round(jac, 4),
                    "rank": rank + 1,
                })

    text_mean = float(np.mean(text_jaccards)) if text_jaccards else 0
    struct_mean = float(np.mean(struct_jaccards)) if struct_jaccards else 0
    ratio = struct_mean / text_mean if text_mean > 0 else float('inf')

    report = {
        "ok": True,
        "n_sampled": sample_size,
        "k": k,
        "text_embedding_tag_jaccard_mean": round(text_mean, 4),
        "structural_embedding_tag_jaccard_mean": round(struct_mean, 4),
        "structural_to_text_ratio": round(ratio, 3),
        "n_cross_domain_candidates": len(cross_domain_candidates),
        "p11_criterion_2x": ratio >= 2.0,
    }

    # Sort cross-domain candidates by structural similarity
    cross_domain_candidates.sort(key=lambda x: -x["structural_similarity"])
    report["cross_domain_top_10"] = cross_domain_candidates[:10]

    return report, cross_domain_candidates


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# 4. Hypergraph topology checks
# ---------------------------------------------------------------------------

def check_hypergraph_quality(outdir: Path, n_sample: int = 200) -> dict:
    """Spot-check hypergraph structure for pathologies."""
    hg_path = outdir / "hypergraphs.json"
    if not hg_path.exists():
        return {"ok": False, "error": "hypergraphs.json not found"}

    with open(hg_path) as f:
        hypergraphs = json.load(f)

    n = len(hypergraphs)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n, min(n_sample, n), replace=False)

    node_counts = []
    edge_counts = []
    node_type_dist = {}
    edge_type_dist = {}
    empty_count = 0
    singleton_count = 0

    for i in sample_idx:
        hg = hypergraphs[i]
        nn = hg["meta"]["n_nodes"]
        ne = hg["meta"]["n_edges"]
        node_counts.append(nn)
        edge_counts.append(ne)

        if nn == 0:
            empty_count += 1
        if nn == 1:
            singleton_count += 1

        for node in hg.get("nodes", []):
            t = node.get("type", "unknown")
            node_type_dist[t] = node_type_dist.get(t, 0) + 1

        for edge in hg.get("edges", []):
            t = edge.get("type", "unknown")
            edge_type_dist[t] = edge_type_dist.get(t, 0) + 1

    report = {
        "ok": True,
        "n_total": n,
        "n_sampled": len(sample_idx),
        "node_count_mean": round(float(np.mean(node_counts)), 1),
        "node_count_median": round(float(np.median(node_counts)), 1),
        "node_count_p95": round(float(np.percentile(node_counts, 95)), 1),
        "edge_count_mean": round(float(np.mean(edge_counts)), 1),
        "edge_count_median": round(float(np.median(edge_counts)), 1),
        "edge_count_p95": round(float(np.percentile(edge_counts, 95)), 1),
        "empty_hypergraphs": empty_count,
        "singleton_hypergraphs": singleton_count,
        "node_type_distribution": node_type_dist,
        "edge_type_distribution": edge_type_dist,
    }

    if empty_count > len(sample_idx) * 0.05:
        report["ok"] = False
        report["warning"] = f"{empty_count}/{len(sample_idx)} empty hypergraphs (>5%)"

    return report


# ---------------------------------------------------------------------------
# 5. Human review export
# ---------------------------------------------------------------------------

def export_review_set(cross_domain_candidates: list, entities: list,
                      outpath: Path, n: int = 50) -> None:
    """Export cross-domain candidates for human review.

    Each record includes thread titles and tags for both threads,
    plus structural similarity score. Reviewer answers: "Are these
    structurally similar?" (yes/no/unsure).
    """
    # Index entities by position for title lookup
    title_by_idx = {}
    for i, ent in enumerate(entities):
        title_by_idx[i] = ent.get("title", f"Thread {i}")

    review_set = []
    for cand in cross_domain_candidates[:n]:
        review_set.append({
            "pair_id": len(review_set) + 1,
            "thread_a": {
                "idx": cand["query_idx"],
                "title": title_by_idx.get(cand["query_idx"], "?"),
                "tags": cand["query_tags"],
            },
            "thread_b": {
                "idx": cand["neighbour_idx"],
                "title": title_by_idx.get(cand["neighbour_idx"], "?"),
                "tags": cand["neighbour_tags"],
            },
            "structural_similarity": cand["structural_similarity"],
            "tag_jaccard": cand["tag_jaccard"],
            "judgement": None,  # To be filled: "yes" / "no" / "unsure"
            "notes": "",
        })

    outpath.write_text(json.dumps(review_set, indent=2, ensure_ascii=False))
    print(f"Exported {len(review_set)} pairs to {outpath}")
    print(f"Fill in 'judgement' field for each pair: yes / no / unsure")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate superpod run output")
    parser.add_argument("outdir", type=Path,
                        help="Output directory from superpod run")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Skip embedding comparisons (CPU run has no embeddings)")
    parser.add_argument("--compare", type=Path, default=None,
                        help="Second output directory for side-by-side comparison")
    parser.add_argument("--export-review", type=Path, default=None,
                        help="Export cross-domain candidates for human review")
    parser.add_argument("--n-review", type=int, default=50,
                        help="Number of pairs to export for review")
    parser.add_argument("--json-report", type=Path, default=None,
                        help="Write full report to JSON file")
    args = parser.parse_args()

    report = {"outdir": str(args.outdir)}

    # --- 1. Pipeline health ---
    print("=" * 60)
    print(f"EVALUATING: {args.outdir}")
    print("=" * 60)

    health = check_pipeline_health(args.outdir)
    report["pipeline_health"] = health
    print(f"\n1. Pipeline health:")
    print(f"   Stages completed: {health.get('stages_completed', [])}")
    print(f"   Entities: {health.get('entity_count', '?')}")
    print(f"   Elapsed: {health.get('elapsed_seconds', '?')}s")
    if health.get("stage8_stats"):
        s8 = health["stage8_stats"]
        print(f"   Expression parse rate: {s8.get('parse_rate', 0):.1%}")
    if health.get("warnings"):
        for w in health["warnings"]:
            print(f"   WARNING: {w}")

    # --- 2. Hypergraph quality ---
    hg_report = check_hypergraph_quality(args.outdir)
    report["hypergraph_quality"] = hg_report
    print(f"\n2. Hypergraph quality ({hg_report.get('n_total', 0)} total):")
    print(f"   Nodes: mean={hg_report.get('node_count_mean')}, "
          f"median={hg_report.get('node_count_median')}, "
          f"p95={hg_report.get('node_count_p95')}")
    print(f"   Edges: mean={hg_report.get('edge_count_mean')}, "
          f"median={hg_report.get('edge_count_median')}, "
          f"p95={hg_report.get('edge_count_p95')}")
    print(f"   Node types: {hg_report.get('node_type_distribution', {})}")
    print(f"   Edge types: {hg_report.get('edge_type_distribution', {})}")
    if hg_report.get("warning"):
        print(f"   WARNING: {hg_report['warning']}")

    # --- 3. Embedding quality ---
    if not args.cpu_only:
        for name, fname in [("text", "embeddings.npy"),
                            ("structural", "hypergraph-embeddings.npy")]:
            emb_path = args.outdir / fname
            emb_report = check_embedding_quality(emb_path, name)
            report[f"{name}_embedding_quality"] = emb_report
            print(f"\n3{'a' if name == 'text' else 'b'}. {name.title()} embedding quality:")
            if emb_report.get("ok") is False and "not found" in emb_report.get("error", ""):
                print(f"   SKIPPED ({fname} not found)")
                continue
            print(f"   Shape: {emb_report.get('shape')}")
            print(f"   Norm: mean={emb_report.get('norm_mean', 0):.3f}, "
                  f"std={emb_report.get('norm_std', 0):.3f}")
            print(f"   Avg pairwise cosine: {emb_report.get('avg_pairwise_cosine', 0):.4f}")
            print(f"   Effective dim (90%): {emb_report.get('effective_dim_90pct')}")
            print(f"   Effective dim (95%): {emb_report.get('effective_dim_95pct')}")
            if emb_report.get("warning"):
                print(f"   WARNING: {emb_report['warning']}")

        # --- 4. Structural vs text comparison ---
        text_path = args.outdir / "embeddings.npy"
        struct_path = args.outdir / "hypergraph-embeddings.npy"
        if text_path.exists() and struct_path.exists():
            comp_report, cross_domain = compare_embeddings(args.outdir)
            report["embedding_comparison"] = comp_report
            print(f"\n4. Structural vs text embedding comparison:")
            print(f"   Text neighbours avg tag Jaccard: "
                  f"{comp_report['text_embedding_tag_jaccard_mean']:.4f}")
            print(f"   Structural neighbours avg tag Jaccard: "
                  f"{comp_report['structural_embedding_tag_jaccard_mean']:.4f}")
            print(f"   Ratio (structural/text): "
                  f"{comp_report['structural_to_text_ratio']:.3f}")
            print(f"   P11 criterion (ratio >= 2.0): "
                  f"{'PASS' if comp_report['p11_criterion_2x'] else 'FAIL'}")
            print(f"   Cross-domain candidates found: "
                  f"{comp_report['n_cross_domain_candidates']}")

            if comp_report.get("cross_domain_top_10"):
                print(f"\n   Top cross-domain matches:")
                for cd in comp_report["cross_domain_top_10"][:5]:
                    print(f"     {cd['query_tags'][:3]} ↔ {cd['neighbour_tags'][:3]}  "
                          f"sim={cd['structural_similarity']:.3f}")

            # Export for human review if requested
            if args.export_review and cross_domain:
                entities = json.loads(
                    (args.outdir / "entities.json").read_text())
                export_review_set(cross_domain, entities,
                                  args.export_review, args.n_review)
        else:
            print(f"\n4. Embedding comparison: SKIPPED (need both text + structural)")

    # --- Summary ---
    all_ok = all(
        report.get(k, {}).get("ok", True)
        for k in ["pipeline_health", "hypergraph_quality",
                   "text_embedding_quality", "structural_embedding_quality"]
    )
    print(f"\n{'=' * 60}")
    print(f"OVERALL: {'PASS' if all_ok else 'ISSUES FOUND'}")
    print(f"{'=' * 60}")

    if args.json_report:
        args.json_report.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str))
        print(f"\nFull report written to {args.json_report}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
