#!/usr/bin/env python3
"""Build a deterministic, report-only recovery queue from frozen stage accounting."""
from __future__ import annotations

import argparse, hashlib, json, re
from collections import Counter
from pathlib import Path

SCHEMA = "futon6/mark8-recovery-queue/v1"
ACCOUNTING_SCHEMA = "futon6-stage-accounting/v1"
RULES = [
    {"class": "deterministic-input-defect", "pattern": r"^no clause (?:spans|units)\b", "disposition": "repair-input", "retry": False},
    {"class": "post-call-contract-failure", "pattern": r"^contract: .*derive node .* -> .* ->", "disposition": "repair-model-output-contract", "retry": False},
    {"class": "sanitize-reparse-candidate", "pattern": r"non-JSON.*Invalid control character", "disposition": "sanitize-and-reparse", "retry": False, "requires-response": True},
    {"class": "retryable-transport", "pattern": r"TimeoutError|\btimeout\b|timed out", "disposition": "retry-transport", "retry": True},
    {"class": "scoped-retry-candidate", "pattern": r"max_tokens|output truncated|truncation|maximum context length", "disposition": "retry-with-smaller-scope", "retry": True},
]
ALLOWED_STATUS = {"accepted", "deferred", "rejected", "errored"}

def encode(x): return (json.dumps(x, indent=2, sort_keys=True) + "\n").encode()
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def classify(reason, has_response):
    for rule in RULES:
        if re.search(rule["pattern"], reason, re.I) and (not rule.get("requires-response") or has_response):
            return rule["class"], rule["disposition"], rule["retry"]
    return "manual-review", "manual-review", False

def build(paths: list[Path]):
    seen, queue, audit, input_hashes = set(), [], Counter(), []
    sources = []
    for path in paths:
        path = path.resolve()
        run = path.parents[3]
        relative = path.relative_to(run).as_posix()
        sources.append((relative, path, run))
    for relative, path, run in sorted(sources, key=lambda row: row[0]):
        doc = json.loads(path.read_bytes())
        if doc.get("schema") != ACCOUNTING_SCHEMA or doc.get("stage") not in {"S3", "S4"} or doc.get("producer") != "loop":
            raise ValueError(f"{path}: refused accounting schema/stage/producer")
        stage = doc["stage"]
        items = doc.get("items")
        if not isinstance(items, list): raise ValueError(f"{path}: items must be a list")
        accounting_hash = hashlib.sha256(encode({**doc, "items": sorted(items, key=lambda x: str(x.get("id")))})).hexdigest()
        input_hashes.append({"path": relative, "semantic-sha256": accounting_hash})
        for item in items:
            ident, status = item.get("id"), item.get("status")
            if not isinstance(ident, str) or not ident or ident in seen: raise ValueError(f"duplicate/invalid item id: {ident!r}")
            seen.add(ident)
            if status not in ALLOWED_STATUS: raise ValueError(f"{ident}: invalid status {status!r}")
            reason, paper = item.get("reason"), item.get("paper")
            if not isinstance(reason, str) or not isinstance(paper, str) or not paper: raise ValueError(f"{ident}: invalid reason/paper")
            attempts, artifacts = item.get("attempts"), item.get("artifacts")
            if not isinstance(attempts, list) or not isinstance(artifacts, list): raise ValueError(f"{ident}: invalid attempts/artifacts")
            if not all(isinstance(x, str) and x for x in artifacts): raise ValueError(f"{ident}: invalid artifact reference")
            if not all(isinstance(a, dict) and isinstance(a.get("attempt"), int) and
                       not isinstance(a.get("attempt"), bool) and isinstance(a.get("result"), str)
                       for a in attempts): raise ValueError(f"{ident}: invalid attempt record")
            audit[("stage", stage)] += 1; audit[("status", status)] += 1
            if status not in {"rejected", "errored"}: continue
            responses = [a.get("response") for a in attempts if isinstance(a, dict) and isinstance(a.get("response"), str)]
            cls, disposition, retry = classify(reason, bool(responses))
            evidence = [{"kind": "accounting", "path": relative, "semantic-sha256": accounting_hash}]
            for kind, ref in sorted([("artifact", x) for x in artifacts] + [("response", x) for x in responses]):
                target = (run / ref).resolve()
                try: target.relative_to(run.resolve())
                except ValueError: raise ValueError(f"{ident}: evidence path escapes run: {ref}")
                if not target.is_file(): raise ValueError(f"{ident}: missing evidence {ref}")
                evidence.append({"kind": kind, "path": ref, "sha256": sha(target)})
            queue.append({"source-stage": stage, "item-id": ident, "paper-id": paper, "status": status,
                          "reason": reason, "reason-class": cls, "recovery-disposition": disposition,
                          "retry-eligible": retry, "evidence": evidence})
            audit[("reason", cls)] += 1; audit[("disposition", disposition)] += 1
    queue.sort(key=lambda x: (x["source-stage"], x["item-id"]))
    counts = {kind: dict(sorted((key, n) for (group, key), n in audit.items() if group == kind))
              for kind in ("stage", "status", "reason", "disposition")}
    frozen = [{k: v for k, v in rule.items()} for rule in RULES] + [{"class":"manual-review","pattern":"<unmatched>","disposition":"manual-review","retry":False}]
    return {"schema": SCHEMA, "classification-precedence": frozen, "inputs": input_hashes,
            "queue": queue, "counts": counts}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("accounting", nargs="+", type=Path); ap.add_argument("--out", required=True, type=Path); a=ap.parse_args()
    a.out.write_bytes(encode(build(a.accounting))); return 0
if __name__ == "__main__": raise SystemExit(main())
