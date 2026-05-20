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


def evaluate_manifest(manifest, baseline_summary, profile: str):
    gates = []
    s5 = manifest.get("stage5_stats") or {}
    s9 = manifest.get("stage9a_stats") or {}
    open_ner = s5.get("open_ner") or {}
    gs = s9.get("geometry_stats") or {}
    stage_status = manifest.get("stage_status") or {}
    entity_count = manifest.get("entity_count")

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
