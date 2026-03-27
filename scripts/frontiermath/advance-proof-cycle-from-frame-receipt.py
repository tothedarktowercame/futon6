#!/usr/bin/env python3
"""Advance a futon3c proof cycle execute phase from a futon6 proof frame receipt.

This adapter preserves the graph split:
- proof dependency lives in futon3c's obligation DAG
- execution trace lives in the futon6 frame receipt

It can either:
- print the execute-phase EDN payload
- print the full /eval form
- submit pb/cycle-advance! to a running futon3c /eval endpoint
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


class EdnKeyword(str):
    pass


def load_receipt(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def keyword_name(raw: str | None) -> EdnKeyword | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return EdnKeyword(text if text.startswith(":") else f":{text}")


def edn_str(obj) -> str:
    if isinstance(obj, EdnKeyword):
        return str(obj)
    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            key = k if str(k).startswith(":") else f":{k}"
            parts.append(f"{key} {edn_str(v)}")
        return "{" + " ".join(parts) + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + " ".join(edn_str(x) for x in obj) + "]"
    if isinstance(obj, str):
        escaped = obj.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if obj is None:
        return "nil"
    return str(obj)


def graph_ref_to_edn(ref: dict | None) -> dict | None:
    if not isinstance(ref, dict):
        return None
    ref_id = ref.get("ref/id")
    ref_type = keyword_name(ref.get("ref/type"))
    if not ref_id or not ref_type:
        return None
    out = {
        "ref/type": ref_type,
        "ref/id": str(ref_id),
    }
    label = ref.get("ref/label")
    if label:
        out["ref/label"] = str(label)
    return out


def adapt_boundary_kind(raw: str | None) -> EdnKeyword | None:
    kind = keyword_name(raw)
    if kind == ":frame":
        return EdnKeyword(":proof-step")
    return kind


def build_execute_payload(receipt: dict) -> dict:
    boundary = receipt.get("frame-boundary", {})
    raw_refs = boundary.get("boundary/graph-refs", []) or []
    graph_refs = [x for x in (graph_ref_to_edn(ref) for ref in raw_refs) if x]

    step_boundary = {}
    step_boundary["boundary/id"] = str(boundary.get("boundary/id"))
    step_boundary["boundary/kind"] = adapt_boundary_kind(boundary.get("boundary/kind"))
    for src_key in (
        "boundary/owner",
        "boundary/entrypoint",
        "boundary/workdir",
        "boundary/trace-id",
    ):
        val = boundary.get(src_key)
        if val:
            step_boundary[src_key] = str(val)
    artifacts = boundary.get("boundary/artifacts", []) or []
    if artifacts:
        step_boundary["boundary/artifacts"] = artifacts
    if graph_refs:
        step_boundary["boundary/graph-refs"] = graph_refs
    algorithm_ref = graph_ref_to_edn(boundary.get("boundary/algorithm-ref"))
    if algorithm_ref:
        step_boundary["boundary/algorithm-ref"] = algorithm_ref

    payload = {
        "artifacts": receipt.get("frame/artifacts", []) or artifacts,
        "step-boundary": step_boundary,
    }
    if graph_refs:
        payload["graph-refs"] = graph_refs

    notes = []
    notes.append(
        "Imported from futon6 proof frame receipt; obligation DAG remains authoritative for proof dependency."
    )
    if receipt.get("case-anchor"):
        notes.append(f"case-anchor={receipt['case-anchor']}")
    if receipt.get("frame/id"):
        notes.append(f"frame-id={receipt['frame/id']}")
    payload["notes"] = " ".join(notes)
    return payload


def build_eval_form(problem_id: str, cycle_id: str, payload: dict) -> str:
    return (
        "(do "
        "(require (quote [futon3c.proof.bridge :as pb])) "
        f"(pb/cycle-advance! {edn_str(problem_id)} {edn_str(cycle_id)} {edn_str(payload)}))"
    )


def read_admin_token(futon3c_root: Path | None) -> str:
    env_token = os.getenv("FUTON3C_ADMIN_TOKEN") or os.getenv("ADMIN_TOKEN")
    if env_token and env_token.strip():
        return env_token.strip()
    if futon3c_root is not None:
        token_file = futon3c_root / ".admintoken"
        if token_file.exists():
            token = token_file.read_text(encoding="utf-8").strip()
            if token:
                return token
    raise SystemExit(
        "No admin token found. Set FUTON3C_ADMIN_TOKEN or ADMIN_TOKEN, "
        "or create .admintoken in the futon3c root."
    )


def submit_eval(eval_url: str, token: str, form: str) -> str:
    req = urllib.request.Request(
        eval_url,
        data=form.encode("utf-8"),
        headers={"x-admin-token": token, "content-type": "text/plain; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"eval HTTP {exc.code}: {body}") from exc


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("receipt", type=Path)
    ap.add_argument("--problem-id")
    ap.add_argument("--cycle-id")
    ap.add_argument("--print-payload", action="store_true")
    ap.add_argument("--print-form", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--eval-url", default="http://127.0.0.1:6768/eval")
    ap.add_argument("--futon3c-root", type=Path)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    receipt = load_receipt(args.receipt)
    problem_id = args.problem_id or receipt.get("proof/problem-id")
    cycle_id = args.cycle_id or receipt.get("proof/cycle-id")
    if not problem_id:
        raise SystemExit("missing problem-id")
    if not cycle_id:
        raise SystemExit("missing cycle-id")

    payload = build_execute_payload(receipt)
    form = build_eval_form(str(problem_id), str(cycle_id), payload)

    if args.print_payload or (not args.print_form and not args.submit):
        print(edn_str(payload))
    if args.print_form:
        print(form)
    if args.submit:
        token = read_admin_token(args.futon3c_root)
        result = submit_eval(args.eval_url, token, form)
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
