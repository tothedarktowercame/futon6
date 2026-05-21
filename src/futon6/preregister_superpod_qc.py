"""Preregistered QC expectations for superpod runs."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median


DEFAULT_BASELINE_DIR = Path.home() / "code" / "storage" / "mark2" / "manifests"


def _as_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _quantile(sorted_vals, q):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    idx = (len(sorted_vals) - 1) * q
    lo = int(idx)
    hi = min(len(sorted_vals) - 1, lo + 1)
    frac = idx - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _range_summary(vals):
    vals = sorted(v for v in vals if isinstance(v, (int, float)))
    if not vals:
        return None
    return {
        "count": len(vals),
        "min": float(vals[0]),
        "p25": _quantile(vals, 0.25),
        "median": float(median(vals)),
        "p75": _quantile(vals, 0.75),
        "max": float(vals[-1]),
    }


def load_manifest(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_baseline_manifests(baseline_dir: Path, profile: str):
    rows = []
    if not baseline_dir.exists():
        return rows
    for path in sorted(baseline_dir.glob("*.json")):
        stem = path.stem
        is_mfuton = stem.startswith("mfuton-")
        if profile == "broad-arxiv" and is_mfuton:
            continue
        if profile == "mfuton" and not is_mfuton:
            continue
        rows.append(load_manifest(path))
    return rows


def summarize_baselines(manifests):
    scope_cov = []
    avg_nodes = []
    avg_edges = []
    claim_rates = []
    entity_counts = []
    readiness = []
    health_messages = []

    for manifest in manifests:
        entity_counts.append(manifest.get("entity_count"))
        readiness.append((manifest.get("readiness") or {}).get("status"))
        for issue in manifest.get("health_issues") or []:
            msg = issue.get("message")
            if msg:
                health_messages.append(msg)

        s5 = manifest.get("stage5_stats") or {}
        s9 = manifest.get("stage9a_stats") or {}
        gs = s9.get("geometry_stats") or {}

        value = _as_float(s5.get("scope_coverage"))
        if value is not None:
            scope_cov.append(value)
        value = _as_float(s9.get("avg_nodes"))
        if value is not None:
            avg_nodes.append(value)
        value = _as_float(s9.get("avg_edges"))
        if value is not None:
            avg_edges.append(value)
        papers = _as_float(gs.get("papers"))
        with_claims = _as_float(gs.get("with_claims"))
        if papers and with_claims is not None:
            claim_rates.append(with_claims / papers)

    return {
        "baseline_count": len(manifests),
        "entity_count_values": sorted({v for v in entity_counts if isinstance(v, int)}),
        "scope_coverage": _range_summary(scope_cov),
        "avg_nodes": _range_summary(avg_nodes),
        "avg_edges": _range_summary(avg_edges),
        "geometry_claim_rate": _range_summary(claim_rates),
        "readiness_counts": {k: readiness.count(k) for k in sorted(set(readiness)) if k is not None},
        "health_messages_seen": sorted(set(health_messages)),
    }


def _gate(name, status, message, **extra):
    payload = {"name": name, "status": status, "message": message}
    if extra:
        payload.update(extra)
    return payload


def summarize_structure_learning(manifest):
    """Build the structure-learning headline block surfaced at the top of QC.

    The numbers come from stage5_stats.structure_learning and Stage 5 comment
    counters. They are the answer to "what did this run actually learn?" —
    candidates discovered, how many cleared the promotion gate, how many
    fired against a prior seed JSON, and how much commented-out source the
    detector now ignores instead of treating as residual.
    """
    s5 = manifest.get("stage5_stats") or {}
    sl = s5.get("structure_learning") or {}
    candidates = sl.get("structure_seed_candidates") or []

    gated = 0
    classified = 0
    kind_breakdown: dict[str, int] = {}
    for c in candidates:
        pk = c.get("predicted_kind")
        if pk:
            classified += 1
            kind_breakdown[pk] = kind_breakdown.get(pk, 0) + 1
            if int(c.get("paper_count") or 0) >= 2:
                gated += 1

    loss = sl.get("loss") or {}
    audit = s5.get("audit_summary") or {}
    audit_agg = (audit or {}).get("aggregate") or {}
    return {
        "structure_learning_enabled": bool(sl.get("enabled")),
        "candidates_discovered": int(sl.get("candidates_written") or 0),
        "candidates_classified": classified,
        "candidates_kind_breakdown": dict(sorted(kind_breakdown.items())),
        "gated_for_promotion": gated,
        "seed_signatures_loaded": int(sl.get("seed_signatures_loaded") or 0),
        "seed_matches_applied": int(sl.get("seed_matches_applied") or 0),
        "entities_with_seed_matches": int(sl.get("entities_with_seed_matches") or 0),
        "free_floating_term_ratio": loss.get("free_floating_term_ratio"),
        "uncovered_sentences_with_known_terms": int(loss.get("uncovered_sentences_with_known_terms") or 0),
        "comment_scopes_total": int(s5.get("total_comments") or 0),
        "entities_with_comments": int(s5.get("entities_with_comments") or 0),
        "audit_sample_size": int((audit or {}).get("sample_size") or 0),
        "audit_inhabited_terms": int(audit_agg.get("inhabited") or 0),
        "audit_outer_terms": int(audit_agg.get("outer") or 0),
        "audit_straddled_terms": int(audit_agg.get("straddled") or 0),
        "audit_frontier_ratio": audit_agg.get("frontier_ratio"),
        "audit_depth_distribution": dict(audit_agg.get("depth_distribution") or {}),
        "audit_max_depth": int(audit_agg.get("max_depth") or 0),
    }


def evaluate_manifest(manifest, baseline_summary, profile: str):
    gates = []
    s5 = manifest.get("stage5_stats") or {}
    s9 = manifest.get("stage9a_stats") or {}
    open_ner = s5.get("open_ner") or {}
    gs = s9.get("geometry_stats") or {}
    stage_status = manifest.get("stage_status") or {}
    entity_count = manifest.get("entity_count")
    structure_learning = s5.get("structure_learning") or {}

    expected_entity_counts = baseline_summary.get("entity_count_values") or []
    if expected_entity_counts and entity_count in expected_entity_counts:
        gates.append(_gate(
            "batch_shape",
            "pass",
            f"entity_count={entity_count} matches {profile} history {expected_entity_counts}",
        ))
    else:
        gates.append(_gate(
            "batch_shape",
            "warn",
            f"entity_count={entity_count} is outside {profile} history {expected_entity_counts}",
        ))

    text_counts = s5.get("text_source_counts") or (stage_status.get("ner_scopes") or {}).get("text_source_counts") or {}
    if manifest.get("paper_eprint_dir"):
        eprint_hits = int(text_counts.get("eprint", 0))
        abstract_hits = int(text_counts.get("abstract", 0))
        paper_text_source = s9.get("paper_text_source")
        if eprint_hits > 0 and abstract_hits == 0:
            gates.append(_gate(
                "paper_text_provenance",
                "pass",
                f"Stage 5 paper text is eprint-backed (eprint={eprint_hits}, abstract={abstract_hits})",
            ))
        elif not text_counts and paper_text_source == "eprints":
            gates.append(_gate(
                "paper_text_provenance",
                "warn",
                "legacy manifest lacks Stage 5 provenance counters, but Stage 9a reports eprint-backed paper text",
            ))
        else:
            gates.append(_gate(
                "paper_text_provenance",
                "fail",
                f"Stage 5 paper text provenance is weak (eprint={eprint_hits}, abstract={abstract_hits})",
            ))

    scope_summary = baseline_summary.get("scope_coverage")
    scope_cov = _as_float(s5.get("scope_coverage"))
    if scope_summary and scope_cov is not None:
        floor = scope_summary["p25"]
        stretch = scope_summary["median"]
        if scope_cov >= stretch:
            status = "pass"
            msg = f"scope_coverage={scope_cov:.4f} meets/exceeds historical median {stretch:.4f}"
        elif scope_cov >= floor:
            status = "warn"
            msg = f"scope_coverage={scope_cov:.4f} is above p25 {floor:.4f} but below median {stretch:.4f}"
        else:
            status = "fail"
            msg = f"scope_coverage={scope_cov:.4f} is below historical p25 {floor:.4f}"
        gates.append(_gate("scope_coverage_prediction", status, msg))

    node_summary = baseline_summary.get("avg_nodes")
    avg_nodes = _as_float(s9.get("avg_nodes"))
    if node_summary and avg_nodes is not None:
        floor = node_summary["p25"]
        ceiling = node_summary["p75"]
        if floor <= avg_nodes <= ceiling:
            status = "pass"
            msg = f"avg_nodes={avg_nodes:.3f} is inside interquartile history [{floor:.3f}, {ceiling:.3f}]"
        elif node_summary["min"] <= avg_nodes <= node_summary["max"]:
            status = "warn"
            msg = f"avg_nodes={avg_nodes:.3f} is inside historical range but outside IQR"
        else:
            status = "fail"
            msg = f"avg_nodes={avg_nodes:.3f} is outside historical range [{node_summary['min']:.3f}, {node_summary['max']:.3f}]"
        gates.append(_gate("hypergraph_density_prediction", status, msg))

    claim_summary = baseline_summary.get("geometry_claim_rate")
    papers = _as_float(gs.get("papers"))
    with_claims = _as_float(gs.get("with_claims"))
    if claim_summary and papers and with_claims is not None:
        claim_rate = with_claims / papers
        floor = claim_summary["p25"]
        stretch = claim_summary["median"]
        if claim_rate >= stretch:
            status = "pass"
            msg = f"geometry claim-paper rate={claim_rate:.4f} meets/exceeds historical median {stretch:.4f}"
        elif claim_rate >= floor:
            status = "warn"
            msg = f"geometry claim-paper rate={claim_rate:.4f} is above p25 {floor:.4f} but below median {stretch:.4f}"
        else:
            status = "fail"
            msg = f"geometry claim-paper rate={claim_rate:.4f} is below historical p25 {floor:.4f}"
        gates.append(_gate("geometry_claim_rate_prediction", status, msg))

    if manifest.get("discover_terms"):
        learned = int(open_ner.get("learned_dictionary_written", 0))
        rhs_supported = int(open_ner.get("rhs_supported_terms", 0))
        new_terms = int(open_ner.get("new_terms_learned", 0))
        if learned > 0 and rhs_supported > 0:
            status = "pass"
            msg = (
                f"learned_dictionary_written={learned}, rhs_supported_terms={rhs_supported}, "
                f"new_terms_learned={new_terms}"
            )
        else:
            status = "warn"
            msg = (
                f"discover_terms was enabled but learned_dictionary_written={learned}, "
                f"rhs_supported_terms={rhs_supported}"
            )
        gates.append(_gate("term_learning_prediction", status, msg))

    # Structure-learning gates only fire when the run actually exercised the
    # structure-learning loop. The headline summary below reports the same
    # numbers in one place regardless of which gates ran.
    if structure_learning.get("enabled"):
        candidates_written = int(structure_learning.get("candidates_written") or 0)
        if candidates_written >= 1:
            gates.append(_gate(
                "structure_learning_capture",
                "pass",
                f"Stage 5 emitted {candidates_written} structure-seed candidate(s)",
            ))
        else:
            gates.append(_gate(
                "structure_learning_capture",
                "warn",
                "discover-structures was enabled but Stage 5 emitted 0 candidates",
            ))

        candidates = structure_learning.get("structure_seed_candidates") or []
        gated_count = sum(
            1 for c in candidates
            if c.get("predicted_kind") and int(c.get("paper_count") or 0) >= 2
        )
        if gated_count >= 1:
            gates.append(_gate(
                "gated_pattern_yield",
                "pass",
                f"{gated_count} candidate(s) cleared the promotion gate "
                f"(paper_count>=2 AND predicted_kind set)",
            ))
        else:
            gates.append(_gate(
                "gated_pattern_yield",
                "warn",
                f"no candidates cleared the promotion gate; {len(candidates)} candidates total. "
                f"Expected on small batches; check on larger ones.",
            ))

    seed_signatures_loaded = int(structure_learning.get("seed_signatures_loaded") or 0)
    if seed_signatures_loaded > 0:
        seed_matches = int(structure_learning.get("seed_matches_applied") or 0)
        entities_matched = int(structure_learning.get("entities_with_seed_matches") or 0)
        if seed_matches >= 1:
            gates.append(_gate(
                "structure_seed_replay",
                "pass",
                f"replay matcher fired on {seed_matches} record(s) across {entities_matched} entit(y/ies); "
                f"{seed_signatures_loaded} prior signature(s) loaded",
            ))
        else:
            gates.append(_gate(
                "structure_seed_replay",
                "warn",
                f"{seed_signatures_loaded} prior signature(s) loaded but no matches fired. "
                f"Either the corpus shifted or the matcher floor is too strict.",
            ))

    health_issues = manifest.get("health_issues") or []
    unusual = [
        issue for issue in health_issues
        if "Stage 9b" not in str(issue.get("stage")) or "val Acc@1" not in str(issue.get("message"))
    ]
    if unusual:
        gates.append(_gate(
            "health_issue_profile",
            "warn",
            f"run has non-baseline health issues: {len(unusual)}",
            issues=unusual,
        ))
    else:
        gates.append(_gate(
            "health_issue_profile",
            "pass",
            "health issues match the narrow historical warning profile or are absent",
        ))

    if not gates:
        overall = "warn"
    elif any(g["status"] == "fail" for g in gates):
        overall = "fail"
    elif any(g["status"] == "warn" for g in gates):
        overall = "warn"
    else:
        overall = "pass"

    return {
        "profile": profile,
        "overall": overall,
        "gates": gates,
    }


def build_report_from_manifest(manifest, baseline_dir: Path, profile: str):
    baselines = load_baseline_manifests(baseline_dir, profile)
    baseline_summary = summarize_baselines(baselines)
    evaluation = evaluate_manifest(manifest, baseline_summary, profile)
    return {
        "baseline_dir": str(baseline_dir),
        "profile": profile,
        "baseline_summary": baseline_summary,
        "evaluation": evaluation,
        "headline_summary": summarize_structure_learning(manifest),
    }


def build_report(manifest_path: Path, baseline_dir: Path, profile: str):
    manifest = load_manifest(manifest_path)
    report = build_report_from_manifest(manifest, baseline_dir, profile)
    report["manifest_path"] = str(manifest_path)
    return report


def infer_profile(*, site: str | None, arxiv_jsonl: str | None):
    text = " ".join(part for part in (site or "", arxiv_jsonl or "") if part).lower()
    if "mfuton" in text:
        return "mfuton"
    return "broad-arxiv"
