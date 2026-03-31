#!/usr/bin/env python3
"""Emit a futon6-owned proof frame receipt.

The receipt is an execution-trace artifact. It attaches to the existing proof
obligation DAG but does not redefine mathematical dependency edges.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALGORITHM_REF = "futon3:eal/algorithms/create-container.md"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_ref(raw: str) -> dict[str, str]:
    if ":" not in raw:
        raise argparse.ArgumentTypeError(
            f"invalid ref '{raw}': expected TYPE:ID"
        )
    ref_type, ref_id = raw.split(":", 1)
    ref_type = ref_type.strip()
    ref_id = ref_id.strip()
    if not ref_type or not ref_id:
        raise argparse.ArgumentTypeError(
            f"invalid ref '{raw}': expected TYPE:ID"
        )
    return {"ref/type": ref_type, "ref/id": ref_id}


def norm_paths(values: list[str]) -> list[str]:
    out: list[str] = []
    for raw in values:
        p = Path(raw)
        if p.is_absolute():
            out.append(str(p))
        else:
            out.append(str((Path.cwd() / p).resolve()))
    return out


def default_output(problem_id: str, frame_id: str) -> Path:
    return REPO_ROOT / ".state" / "proof-frames" / problem_id / f"{frame_id}.json"


def load_workspace_metadata(path: str | None) -> dict | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_graph_refs(args: argparse.Namespace) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = [
        {"ref/type": "proof-problem", "ref/id": args.problem_id},
        {"ref/type": "proof-frame", "ref/id": args.frame_id},
    ]
    if args.cycle_id:
        refs.append({"ref/type": "proof-cycle", "ref/id": args.cycle_id})
    if args.blocker_id:
        refs.append({"ref/type": "proof-blocker", "ref/id": args.blocker_id})
    refs.extend(args.graph_ref)

    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []
    for ref in refs:
        key = (ref["ref/type"], ref["ref/id"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return deduped


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Emit a proof frame receipt under futon6/.state."
    )
    ap.add_argument("--problem-id", required=True)
    ap.add_argument("--frame-id", required=True)
    ap.add_argument("--frame-label")
    ap.add_argument("--cycle-id")
    ap.add_argument("--blocker-id")
    ap.add_argument(
        "--boundary-kind",
        choices=["container", "workspace", "frame"],
        default="frame",
    )
    ap.add_argument("--boundary-id")
    ap.add_argument("--trace-id")
    ap.add_argument("--entrypoint")
    ap.add_argument("--workdir")
    ap.add_argument("--owner", default="futon6")
    ap.add_argument("--algorithm-ref", default=DEFAULT_ALGORITHM_REF)
    ap.add_argument("--artifact", action="append", default=[])
    ap.add_argument("--graph-ref", type=parse_ref, action="append", default=[])
    ap.add_argument("--input", action="append", default=[])
    ap.add_argument("--readonly", action="append", default=[])
    ap.add_argument("--writable", action="append", default=[])
    ap.add_argument("--upstream-boundary", action="append", default=[])
    ap.add_argument("--case-anchor")
    ap.add_argument("--workspace-metadata")
    ap.add_argument("--output")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    output_path = Path(args.output) if args.output else default_output(
        args.problem_id, args.frame_id
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    workdir = args.workdir or str(Path.cwd().resolve())
    boundary_id = args.boundary_id or args.frame_id
    trace_id = args.trace_id or boundary_id
    workspace_metadata = load_workspace_metadata(args.workspace_metadata)
    artifacts = norm_paths(args.artifact)
    workspace_map = None
    if workspace_metadata:
        workspace_map = {
            "workspace/root": workspace_metadata.get("frame/workspace-root"),
            "workspace/module-root": workspace_metadata.get("frame/module-root"),
            "workspace/lean-root": workspace_metadata.get("frame/lean-root"),
            "workspace/shared-extension-root": workspace_metadata.get("frame/shared-extension-root"),
            "workspace/proof-plan": workspace_metadata.get("artifacts", {}).get("proof-plan"),
            "workspace/changelog": workspace_metadata.get("artifacts", {}).get("changelog"),
            "workspace/execute-notes": workspace_metadata.get("artifacts", {}).get("execute-notes"),
            "workspace/lean-main": workspace_metadata.get("artifacts", {}).get("lean-main"),
            "workspace/lean-scratch": workspace_metadata.get("artifacts", {}).get("lean-scratch"),
            "workspace/metadata": workspace_metadata.get("artifacts", {}).get("workspace-metadata"),
        }
        for key in (
            "proof-plan",
            "changelog",
            "execute-notes",
            "lean-main",
            "lean-scratch",
            "workspace-metadata",
        ):
            raw = workspace_metadata.get("artifacts", {}).get(key)
            if isinstance(raw, str) and raw.strip():
                artifacts.append(str(Path(raw).resolve()))
    artifacts = list(dict.fromkeys(artifacts))
    graph_refs = build_graph_refs(args)
    algorithm_ref = {"ref/type": "algorithm", "ref/id": args.algorithm_ref}

    receipt = {
        "receipt/schema": "proof-frame-receipt.v1",
        "receipt/generated-at": now_utc(),
        "receipt/owner-repo": "futon6",
        "proof/problem-id": args.problem_id,
        "proof/cycle-id": args.cycle_id,
        "proof/blocker-id": args.blocker_id,
        "proof/graph-role": "frame-trace-for-obligation-node",
        "frame/id": args.frame_id,
        "frame/label": args.frame_label or args.frame_id,
        "frame/upstream-boundaries": args.upstream_boundary,
        "frame-boundary": {
            "boundary/id": boundary_id,
            "boundary/kind": args.boundary_kind,
            "boundary/owner": args.owner,
            "boundary/algorithm-ref": algorithm_ref,
            "boundary/entrypoint": args.entrypoint,
            "boundary/workdir": workdir,
            "boundary/trace-id": trace_id,
            "boundary/artifacts": artifacts,
            "boundary/graph-refs": graph_refs,
        },
        "inputs": norm_paths(args.input),
        "state": {
            "readonly": norm_paths(args.readonly),
            "writable": norm_paths(args.writable),
        },
        "frame/artifacts": artifacts,
        "case-anchor": args.case_anchor,
    }
    if workspace_map:
        receipt["frame/workspace"] = workspace_map

    output_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
