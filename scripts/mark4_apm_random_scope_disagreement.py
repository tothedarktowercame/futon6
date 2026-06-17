#!/usr/bin/env python3
"""Random-arm APM scope coverage + keyword/scope disagreement metric.

This complements ``mark4_apm_structure_coverage.py`` without changing its H9 gate.
It samples a fixed-seed, non-keyword-selected batch-007/008 eprint pool, extracts
scopes with the same nlab-wiring detector, then compares per-proof keyword
retrieval against per-eprint scope retrieval.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import io
import json
import random
import re
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCHES = [
    Path("/home/joe/code/storage/mark2/inbox/batch-007.tar.gz"),
    Path("/home/joe/code/storage/mark2/inbox/batch-008.tar.gz"),
]
DEFAULT_PROOF_SCOPES = Path("/home/joe/code/storage/apm/apm-proof-scopes.json")
DEFAULT_KEYWORD_SCOPES = Path("/home/joe/code/storage/apm/eprint-scopes.json")
DEFAULT_KEYWORD_HITS = ROOT / "data" / "mark4-batch-keyword-hits.json"
DEFAULT_TOP_TSV = ROOT / "data" / "mark4-retrieval-top200.tsv"
DEFAULT_RANDOM_SCOPES = Path(
    "/home/joe/code/storage/apm/mark4-random-eprint-scopes-seed20260617-n200.json"
)
DEFAULT_REPORT = Path(
    "/home/joe/code/storage/apm/mark4-random-scope-disagreement-seed20260617-n200.json"
)
DEFAULT_SEED = 20260617
DEFAULT_SAMPLE_SIZE = 200
DEFAULT_TAU = 0.05

MNUMBER = re.compile(r"\\mNumber\{[^}]*\}")
CTRL = re.compile(r"\\[a-zA-Z]+")
TOK = re.compile(r"[a-zA-Z]\w*")


def load_nlab_wiring():
    path = ROOT / "scripts" / "nlab-wiring.py"
    spec = importlib.util.spec_from_file_location("nlab_wiring", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def batch_name(path: Path) -> str:
    name = path.name
    return name[:-7] if name.endswith(".tar.gz") else path.stem


def iter_batch_records(batch_tar: Path):
    bname = batch_name(batch_tar)
    member_name = f"{bname}/{bname}.jsonl"
    with tarfile.open(batch_tar, "r:gz") as tf:
        member = tf.getmember(member_name)
        stream = tf.extractfile(member)
        if stream is None:
            return
        for raw in stream:
            if raw.strip():
                rec = json.loads(raw)
                rec["_batch_tar"] = str(batch_tar)
                rec["_batch_name"] = bname
                yield rec


def load_batch_records(batch_tars: list[Path]) -> list[dict]:
    records = []
    for batch_tar in batch_tars:
        records.extend(iter_batch_records(batch_tar))
    records.sort(key=lambda r: r["id"])
    return records


def sample_records(records: list[dict], seed: int, n: int) -> list[dict]:
    rng = random.Random(seed)
    if n > len(records):
        raise ValueError(f"sample size {n} > population {len(records)}")
    sample = rng.sample(records, n)
    return sorted(sample, key=lambda r: r["id"])


def eprint_member_name(record: dict) -> str:
    arxiv_id = str(record["id"]).replace("/", "__")
    return f"{record['_batch_name']}/eprints/{arxiv_id}.tar.gz"


def decode_source_blob(blob: bytes) -> str:
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as inner:
            chunks = []
            for member in sorted(inner.getmembers(), key=lambda m: m.name):
                lname = member.name.lower()
                if not member.isfile() or not lname.endswith((".tex", ".ltx", ".bbl")):
                    continue
                f = inner.extractfile(member)
                if f is not None:
                    chunks.append(f.read().decode("utf-8", errors="replace"))
            return "\n".join(chunks)
    except tarfile.TarError:
        pass
    try:
        return gzip.decompress(blob).decode("utf-8", errors="replace")
    except OSError:
        return blob.decode("utf-8", errors="replace")


def extract_eprint_text(tf: tarfile.TarFile, record: dict) -> str:
    try:
        member = tf.getmember(eprint_member_name(record))
    except KeyError:
        return ""
    f = tf.extractfile(member)
    if f is None:
        return ""
    return decode_source_blob(f.read())


def build_random_scopes(records: list[dict]) -> tuple[dict[str, list[dict]], list[str]]:
    nw = load_nlab_wiring()
    by_batch: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_batch[rec["_batch_tar"]].append(rec)
    scopes: dict[str, list[dict]] = {}
    missing_source = []
    for batch_tar, batch_records in sorted(by_batch.items()):
        with tarfile.open(batch_tar, "r:gz") as tf:
            for rec in batch_records:
                text = extract_eprint_text(tf, rec)
                if not text.strip():
                    scopes[rec["id"]] = []
                    missing_source.append(rec["id"])
                    continue
                scopes[rec["id"]] = nw.detect_scopes(rec["id"], text)
    return scopes, missing_source


def symbols(scope: dict) -> set[str]:
    out: set[str] = set()
    for end in scope.get("hx/ends", []):
        for key in ("latex", "text"):
            v = end.get(key)
            if not v:
                continue
            v = CTRL.sub(" ", MNUMBER.sub(" ", v))
            out |= set(TOK.findall(v.lower()))
    return out


def eprint_index(scopes: list[dict]) -> tuple[set[str], dict[str, set[str]]]:
    types = set()
    multi: dict[str, set[str]] = defaultdict(set)
    for scope in scopes:
        typ = scope.get("hx/type")
        if not typ:
            continue
        types.add(typ)
        multi[typ] |= {sym for sym in symbols(scope) if len(sym) >= 3}
    return types, multi


def proof_eprint_multichar_coverage(proof_scopes: list[dict], eprint_scopes: list[dict]) -> float:
    if not proof_scopes:
        return 0.0
    _types, multi = eprint_index(eprint_scopes)
    matched = 0
    for scope in proof_scopes:
        typ = scope.get("hx/type")
        p_syms = {sym for sym in symbols(scope) if len(sym) >= 3}
        if p_syms and p_syms & multi.get(typ, set()):
            matched += 1
    return matched / len(proof_scopes)


def load_top_ids(path: Path) -> set[str]:
    ids = set()
    with path.open() as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                ids.add(parts[1])
    return ids


def keyword_sets_by_proof(hits_path: Path, top_ids: set[str]) -> dict[str, set[str]]:
    hits = json.loads(hits_path.read_text())
    out: dict[str, set[str]] = defaultdict(set)
    for hit in hits:
        eid = hit["id"]
        if eid not in top_ids:
            continue
        for proof_id in hit.get("source_proofs", []):
            out[proof_id].add(eid)
    return out


def scope_sets_by_proof(
    proof_scopes: dict[str, list[dict]],
    eprint_scopes: dict[str, list[dict]],
    tau: float,
) -> tuple[dict[str, set[str]], dict[str, dict[str, float]]]:
    retrieved: dict[str, set[str]] = defaultdict(set)
    scores: dict[str, dict[str, float]] = {}
    for proof_id, scopes in proof_scopes.items():
        if not scopes:
            continue
        proof_scores = {}
        for eid, escopes in eprint_scopes.items():
            if not escopes:
                continue
            cov = proof_eprint_multichar_coverage(scopes, escopes)
            if cov:
                proof_scores[eid] = cov
            if cov >= tau:
                retrieved[proof_id].add(eid)
        scores[proof_id] = proof_scores
    return retrieved, scores


def disagreement_summary(
    keyword_sets: dict[str, set[str]],
    scope_sets: dict[str, set[str]],
    proof_ids: list[str],
) -> dict:
    rows = {}
    jaccards = []
    scope_not_keyword_rates = []
    keyword_not_scope_rates = []
    for proof_id in proof_ids:
        kws = keyword_sets.get(proof_id, set())
        scp = scope_sets.get(proof_id, set())
        union = kws | scp
        inter = kws & scp
        scope_not_keyword = scp - kws
        keyword_not_scope = kws - scp
        jaccard = len(inter) / len(union) if union else 1.0
        snk_rate = len(scope_not_keyword) / len(scp) if scp else 0.0
        kns_rate = len(keyword_not_scope) / len(kws) if kws else 0.0
        if union:
            jaccards.append(jaccard)
        if scp:
            scope_not_keyword_rates.append(snk_rate)
        if kws:
            keyword_not_scope_rates.append(kns_rate)
        rows[proof_id] = {
            "keyword_retrieved": len(kws),
            "scope_retrieved": len(scp),
            "intersection": len(inter),
            "scope_not_keyword": len(scope_not_keyword),
            "keyword_not_scope": len(keyword_not_scope),
            "jaccard": jaccard,
            "scope_not_keyword_rate": snk_rate,
            "keyword_not_scope_rate": kns_rate,
            "scope_not_keyword_ids": sorted(scope_not_keyword)[:25],
            "keyword_not_scope_ids": sorted(keyword_not_scope)[:25],
        }
    return {
        "proofs_compared": len(proof_ids),
        "mean_jaccard": statistics.mean(jaccards) if jaccards else 1.0,
        "mean_scope_not_keyword_rate": (
            statistics.mean(scope_not_keyword_rates) if scope_not_keyword_rates else 0.0
        ),
        "mean_keyword_not_scope_rate": (
            statistics.mean(keyword_not_scope_rates) if keyword_not_scope_rates else 0.0
        ),
        "proofs_with_scope_not_keyword": sum(
            1 for row in rows.values() if row["scope_not_keyword"] > 0
        ),
        "per_proof": rows,
    }


def pool_coverage(proof_scopes: dict[str, list[dict]], eprint_scopes: dict[str, list[dict]]) -> dict:
    etype_multi: dict[str, set[str]] = defaultdict(set)
    for scopes in eprint_scopes.values():
        for scope in scopes:
            typ = scope.get("hx/type")
            etype_multi[typ] |= {sym for sym in symbols(scope) if len(sym) >= 3}
    vals = []
    for scopes in proof_scopes.values():
        if not scopes:
            continue
        matched = 0
        for scope in scopes:
            typ = scope.get("hx/type")
            p_syms = {sym for sym in symbols(scope) if len(sym) >= 3}
            if p_syms & etype_multi.get(typ, set()):
                matched += 1
        vals.append(matched / len(scopes))
    return {
        "mean": statistics.mean(vals) if vals else 0.0,
        "median": statistics.median(vals) if vals else 0.0,
        "tail_ge80pct": sum(1 for val in vals if val >= 0.8),
    }


def verdict(keyword_cov: dict, random_cov: dict, disagreement: dict) -> str:
    delta = random_cov["mean"] - keyword_cov["mean"]
    snk = disagreement["mean_scope_not_keyword_rate"]
    if abs(delta) <= 0.03 and snk < 0.10:
        return "null-compatible: random scope coverage is close to keyword-pool coverage and scope-only retrieval is weak"
    if snk >= 0.25:
        return "signal: scope retrieval surfaces a substantial scope-not-keyword arm"
    return "mixed: some scope-not-keyword disagreement, but coverage/control delta is modest"


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=Path, action="append", dest="batches")
    ap.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    ap.add_argument("--proof-scopes", type=Path, default=DEFAULT_PROOF_SCOPES)
    ap.add_argument("--keyword-eprint-scopes", type=Path, default=DEFAULT_KEYWORD_SCOPES)
    ap.add_argument("--keyword-hits", type=Path, default=DEFAULT_KEYWORD_HITS)
    ap.add_argument("--top-tsv", type=Path, default=DEFAULT_TOP_TSV)
    ap.add_argument("--random-scopes-out", type=Path, default=DEFAULT_RANDOM_SCOPES)
    ap.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--reuse-random-scopes", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    batches = args.batches or DEFAULT_BATCHES
    records = load_batch_records(batches)
    sample = sample_records(records, args.seed, args.sample_size)
    if args.reuse_random_scopes and args.random_scopes_out.exists():
        random_scopes = json.loads(args.random_scopes_out.read_text())
        missing_source = []
    else:
        random_scopes, missing_source = build_random_scopes(sample)
        args.random_scopes_out.parent.mkdir(parents=True, exist_ok=True)
        args.random_scopes_out.write_text(json.dumps(random_scopes, indent=2, sort_keys=True))

    proof_scopes = json.loads(args.proof_scopes.read_text())
    keyword_scopes = json.loads(args.keyword_eprint_scopes.read_text())
    combined_scopes = dict(keyword_scopes)
    combined_scopes.update(random_scopes)
    top_ids = load_top_ids(args.top_tsv)
    keyword_sets = keyword_sets_by_proof(args.keyword_hits, top_ids)
    scope_sets, _scores = scope_sets_by_proof(proof_scopes, combined_scopes, args.tau)
    proof_ids = sorted(pid for pid, scopes in proof_scopes.items() if scopes)
    disagreement = disagreement_summary(keyword_sets, scope_sets, proof_ids)
    keyword_cov = pool_coverage(proof_scopes, keyword_scopes)
    random_cov = pool_coverage(proof_scopes, random_scopes)

    report = {
        "provenance": {
            "seed": args.seed,
            "sample_size": args.sample_size,
            "source_batches": [str(p) for p in batches],
            "population_size": len(records),
            "random_scope_path": str(args.random_scopes_out),
            "missing_source_count": len(missing_source),
            "missing_source_ids": missing_source,
            "sampled_ids": [rec["id"] for rec in sample],
        },
        "metric": {
            "scope_retrieved": "per proof/eprint type_multichar coverage >= tau",
            "tau": args.tau,
            "keyword_retrieved": "top-200 keyword-hit eprints whose source_proofs include the proof",
            "disagreement": (
                "scope_not_keyword = scope_retrieved - keyword_retrieved; "
                "keyword_not_scope = keyword_retrieved - scope_retrieved; "
                "Jaccard = |intersection| / |union|"
            ),
            "null_hypothesis": (
                "random-pool type_multichar coverage approximately equals keyword-pool "
                "coverage and scope_not_keyword rate is low, so scope matching tracks "
                "keyword selection rather than independent structure"
            ),
        },
        "coverage_control": {
            "keyword_pool_type_multichar": keyword_cov,
            "random_pool_type_multichar": random_cov,
        },
        "disagreement": disagreement,
    }
    report["verdict"] = verdict(keyword_cov, random_cov, disagreement)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({
        "provenance": {
            "seed": args.seed,
            "sample_size": args.sample_size,
            "population_size": len(records),
            "missing_source_count": len(missing_source),
        },
        "coverage_control": report["coverage_control"],
        "disagreement_summary": {
            k: v for k, v in disagreement.items() if k != "per_proof"
        },
        "verdict": report["verdict"],
        "report": str(args.report_out),
        "random_scopes": str(args.random_scopes_out),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
