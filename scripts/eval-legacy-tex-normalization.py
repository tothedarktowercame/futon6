#!/usr/bin/env python3
"""Evaluate legacy TeX normalization against baseline Stage 5d extraction.

Two input modes are supported:

1. `--entities-json ... --eprint-dir ...`
   Replay extraction over an entity list plus an external eprint cache.

2. `--batch-tar ...`
   Replay extraction over an input batch tarball containing
   `batch-*.jsonl` and `eprints/`.

The script reports baseline-vs-normalized claim/proof coverage and the
origin mix of normalized claim nodes.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from futon6.legacy_tex_normalize import normalize
from futon6.paper_hypergraph import extract_paper_hypergraph_classical


def _safe_arxiv_id(entity_id: str) -> str:
    if entity_id.startswith("arxiv-"):
        entity_id = entity_id[len("arxiv-"):]
    entity_id = entity_id.replace("/", "__")
    return re.sub(r"[^A-Za-z0-9._-]", "_", entity_id)


def _decode_plain_text(raw: bytes, max_chars: int) -> str:
    return raw.decode("utf-8", errors="ignore")[:max_chars]


def _looks_like_tex(text: str) -> bool:
    return "\\documentclass" in text or "\\begin{" in text or "\\newtheorem" in text or "$" in text


def _read_tex_members_from_bytes(raw: bytes, max_chars: int, max_members: int) -> str:
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tf:
            members = [m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(".tex")]
            members.sort(key=lambda m: m.size, reverse=True)
            chunks: list[str] = []
            total_chars = 0
            for member in members[:max_members]:
                remaining = max_chars - total_chars
                if remaining <= 0:
                    break
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                text = _decode_plain_text(fh.read(max(4096, remaining * 2)), remaining)
                if not text.strip():
                    continue
                chunks.append(text)
                total_chars += len(text)
            return "\n\n".join(chunks)[:max_chars]
    except (tarfile.TarError, OSError, EOFError):
        return ""


def _decode_embedded_member(name: str, raw: bytes, max_chars: int, max_members: int) -> str:
    lname = name.lower()
    if lname.endswith(".tex"):
        return _decode_plain_text(raw, max_chars)
    if lname.endswith(".tar.gz") or lname.endswith(".tar") or lname.endswith(".bin"):
        text = _read_tex_members_from_bytes(raw, max_chars=max_chars, max_members=max_members)
        if text:
            return text
        if lname.endswith(".tar.gz") or lname.endswith(".bin"):
            try:
                guess = _decode_plain_text(gzip.decompress(raw), max_chars)
                if _looks_like_tex(guess):
                    return guess
            except OSError:
                pass
        guess = _decode_plain_text(raw, max_chars)
        if _looks_like_tex(guess):
            return guess
    if lname.endswith(".gz"):
        try:
            guess = _decode_plain_text(gzip.decompress(raw), max_chars)
            if _looks_like_tex(guess):
                return guess
        except OSError:
            return ""
    return ""


def _load_text_from_external_dir(
    eprint_dir: Path,
    entity_id: str,
    max_chars: int,
    max_members: int,
) -> tuple[str | None, str]:
    sid = _safe_arxiv_id(entity_id)
    candidates = [p for p in eprint_dir.glob(f"{sid}*") if p.is_file()]
    if not candidates:
        return None, "missing"

    def _priority(path: Path) -> int:
        name = path.name.lower()
        if name.endswith(".tex"):
            return 0
        if name.endswith(".tar.gz"):
            return 1
        if name.endswith(".tar"):
            return 2
        if name.endswith(".bin"):
            return 3
        if name.endswith(".gz"):
            return 4
        return 9

    for path in sorted(candidates, key=_priority):
        raw = path.read_bytes()
        text = _decode_embedded_member(path.name, raw, max_chars=max_chars, max_members=max_members)
        if text:
            return text, "ok"
    return None, "unusable"


def _iter_entities_from_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a JSON list")
    return data


def _iter_entities_from_batch_tar(tf: tarfile.TarFile) -> tuple[list[dict], dict[str, tarfile.TarInfo]]:
    members = {m.name: m for m in tf.getmembers() if m.isfile()}
    jsonl_name = next((name for name in members if name.endswith(".jsonl") and "/batch-" in name), None)
    if not jsonl_name:
        raise ValueError("no batch jsonl found in bundle")
    lines = tf.extractfile(members[jsonl_name]).read().decode("utf-8", errors="ignore").splitlines()
    entities = [json.loads(line) for line in lines if line.strip()]
    return entities, members


def _load_text_from_batch_tar(
    tf: tarfile.TarFile,
    members: dict[str, tarfile.TarInfo],
    entity_id: str,
    max_chars: int,
    max_members: int,
) -> tuple[str | None, str]:
    sid = _safe_arxiv_id(entity_id)
    candidate_names = [
        name for name in members
        if "/eprints/" in name and Path(name).name.startswith(sid)
    ]
    if not candidate_names:
        return None, "missing"
    for name in sorted(candidate_names):
        fh = tf.extractfile(members[name])
        if fh is None:
            continue
        text = _decode_embedded_member(Path(name).name, fh.read(), max_chars=max_chars, max_members=max_members)
        if text:
            return text, "ok"
    return None, "unusable"


def _paper_id_from_entity(entity: dict) -> str:
    return (
        entity.get("entity/id")
        or entity.get("id")
        or entity.get("paper_id")
        or ""
    )


def _evaluate_entities(
    entities: list[dict],
    load_text,
    max_chars: int,
    max_members: int,
    max_entities: int | None,
) -> dict:
    metrics = {
        "entities_total": 0,
        "loaded_eprints": 0,
        "missing_or_unusable": 0,
        "base_with_claims": 0,
        "new_with_claims": 0,
        "base_with_proofs": 0,
        "new_with_proofs": 0,
        "base_claim_nodes": 0,
        "new_claim_nodes": 0,
        "new_claim_origin_native": 0,
        "new_claim_origin_alias": 0,
        "new_claim_origin_prose": 0,
        "sample_improved": [],
    }

    selected = entities[:max_entities] if max_entities else entities
    for entity in selected:
        entity_id = _paper_id_from_entity(entity)
        if not entity_id:
            continue
        metrics["entities_total"] += 1
        text, status = load_text(entity_id, max_chars, max_members)
        if not text:
            metrics["missing_or_unusable"] += 1
            continue
        metrics["loaded_eprints"] += 1

        base_hg = extract_paper_hypergraph_classical(text, paper_id=entity_id)
        base_claims = [n for n in base_hg["nodes"] if n["type"] == "claim"]
        base_proofs = [n for n in base_hg["nodes"] if n["type"] == "proof"]
        if base_claims:
            metrics["base_with_claims"] += 1
        if base_proofs:
            metrics["base_with_proofs"] += 1
        metrics["base_claim_nodes"] += len(base_claims)

        result = normalize(text, paper_id=entity_id)
        new_hg = extract_paper_hypergraph_classical(
            result.rewritten_text,
            paper_id=entity_id,
            block_annotations=result.block_annotations,
        )
        new_claims = [n for n in new_hg["nodes"] if n["type"] == "claim"]
        new_proofs = [n for n in new_hg["nodes"] if n["type"] == "proof"]
        if new_claims:
            metrics["new_with_claims"] += 1
        if new_proofs:
            metrics["new_with_proofs"] += 1
        metrics["new_claim_nodes"] += len(new_claims)
        for node in new_claims:
            origin = node.get("attrs", {}).get("block_origin", "native")
            if origin == "native":
                metrics["new_claim_origin_native"] += 1
            elif origin == "alias_expanded":
                metrics["new_claim_origin_alias"] += 1
            elif origin == "prose_synthesized":
                metrics["new_claim_origin_prose"] += 1
        if not base_claims and new_claims and len(metrics["sample_improved"]) < 20:
            metrics["sample_improved"].append(
                {
                    "paper_id": entity_id,
                    "claims": len(new_claims),
                    "proofs": len(new_proofs),
                    "rewrite_kinds": sorted(set(r.kind for r in result.rewrites)),
                }
            )

    metrics["delta_with_claims"] = metrics["new_with_claims"] - metrics["base_with_claims"]
    metrics["delta_with_proofs"] = metrics["new_with_proofs"] - metrics["base_with_proofs"]
    metrics["delta_claim_nodes"] = metrics["new_claim_nodes"] - metrics["base_claim_nodes"]
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--entities-json", type=Path, default=None,
                    help="JSON list of entities with entity/id fields")
    ap.add_argument("--eprint-dir", type=Path, default=None,
                    help="External eprint directory for --entities-json mode")
    ap.add_argument("--batch-tar", type=Path, default=None,
                    help="Batch tarball containing batch-*.jsonl and eprints/")
    ap.add_argument("--max-entities", type=int, default=None,
                    help="Optional limit for a bounded replay")
    ap.add_argument("--max-chars", type=int, default=240_000)
    ap.add_argument("--max-members", type=int, default=4)
    ap.add_argument("--json-out", type=Path, default=None,
                    help="Optional path to write JSON metrics")
    args = ap.parse_args()

    if bool(args.batch_tar) == bool(args.entities_json):
        ap.error("choose exactly one of --batch-tar or --entities-json")
    if args.entities_json and not args.eprint_dir:
        ap.error("--entities-json requires --eprint-dir")

    if args.batch_tar:
        with tarfile.open(args.batch_tar, "r:gz") as tf:
            entities, members = _iter_entities_from_batch_tar(tf)

            def load_text(entity_id: str, max_chars: int, max_members: int):
                return _load_text_from_batch_tar(
                    tf,
                    members,
                    entity_id,
                    max_chars=max_chars,
                    max_members=max_members,
                )

            result = {
                "mode": "batch-tar",
                "batch_tar": str(args.batch_tar),
                "metrics": _evaluate_entities(
                    entities,
                    load_text,
                    max_chars=args.max_chars,
                    max_members=args.max_members,
                    max_entities=args.max_entities,
                ),
            }
    else:
        entities = _iter_entities_from_json(args.entities_json)

        def load_text(entity_id: str, max_chars: int, max_members: int):
            return _load_text_from_external_dir(
                args.eprint_dir,
                entity_id,
                max_chars=max_chars,
                max_members=max_members,
            )

        result = {
            "mode": "entities-json",
            "entities_json": str(args.entities_json),
            "eprint_dir": str(args.eprint_dir),
            "metrics": _evaluate_entities(
                entities,
                load_text,
                max_chars=args.max_chars,
                max_members=args.max_members,
                max_entities=args.max_entities,
            ),
        }

    payload = json.dumps(result, indent=2)
    if args.json_out:
        args.json_out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
