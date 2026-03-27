#!/usr/bin/env python3
"""Project one mfuton local FrontierMath run bundle into the proof-frame seam.

This is the smallest honest owner-side hook after the local FM recovery:
- read one existing mfuton run bundle receipt
- emit a futon6 proof-frame receipt with explicit proof-graph anchors
- optionally hand that receipt to the existing futon3c cycle-advance adapter
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MFUTON_ROOT = (REPO_ROOT / ".." / ".." / "gh" / "mfuton").resolve()


def fail(msg: str) -> "NoReturn":
    raise SystemExit(msg)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_problem_id(run_dir: Path) -> str | None:
    if run_dir.parent.name == "runs":
        return run_dir.parent.parent.name
    return None


def resolve_ref(raw: str, *, mfuton_root: Path, run_dir: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()
    norm = raw.replace("\\", "/")
    if norm.startswith("mfuton/"):
        return (mfuton_root.parent / Path(norm)).resolve()
    return (run_dir / p).resolve()


def add_path(out: list[Path], seen: set[Path], path: Path | None) -> None:
    if path is None:
        return
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    if resolved in seen:
        return
    seen.add(resolved)
    out.append(resolved)


def build_artifacts(
    run_dir: Path,
    payload: dict,
    *,
    mfuton_root: Path,
) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()

    add_path(out, seen, run_dir)
    add_path(out, seen, run_dir / "receipt.json")

    turn_artifacts = payload.get("turn_artifacts", {}) or {}
    for raw in turn_artifacts.values():
        if isinstance(raw, str) and raw.strip():
            add_path(out, seen, resolve_ref(raw, mfuton_root=mfuton_root, run_dir=run_dir))

    root_check = payload.get("proof_runtime_root_check", {}) or {}
    evidence = root_check.get("evidence")
    if isinstance(evidence, str) and evidence.strip():
        add_path(out, seen, resolve_ref(evidence, mfuton_root=mfuton_root, run_dir=run_dir))

    runtime_bring_up = payload.get("runtime_bring_up", {}) or {}
    status_evidence = runtime_bring_up.get("status_evidence")
    if isinstance(status_evidence, str) and status_evidence.strip():
        add_path(
            out,
            seen,
            resolve_ref(status_evidence, mfuton_root=mfuton_root, run_dir=run_dir),
        )

    snapshots = payload.get("proof_state_snapshots", {}) or {}
    for key in ("before", "after"):
        raw = snapshots.get(key)
        if isinstance(raw, str) and raw.strip():
            add_path(out, seen, resolve_ref(raw, mfuton_root=mfuton_root, run_dir=run_dir))

    return out


def infer_writable_root(payload: dict, *, mfuton_root: Path, run_dir: Path) -> Path | None:
    root_check = payload.get("proof_runtime_root_check", {}) or {}
    configured = root_check.get("configured_root")
    if isinstance(configured, str) and configured.strip():
        return resolve_ref(configured, mfuton_root=mfuton_root, run_dir=run_dir)

    local_state = payload.get("local_proof_state_path", {}) or {}
    actual = local_state.get("actual_used")
    if isinstance(actual, str) and actual.strip():
        return resolve_ref(actual, mfuton_root=mfuton_root, run_dir=run_dir).parent

    return None


def run_checked(cmd: list[str], *, cwd: Path) -> str:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "command failed").strip()
        raise RuntimeError(detail)
    return (proc.stdout or "").strip()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, help="mfuton FM run bundle directory")
    ap.add_argument("--cycle-id", required=True)
    ap.add_argument("--blocker-id")
    ap.add_argument("--problem-id")
    ap.add_argument("--frame-id")
    ap.add_argument("--frame-label")
    ap.add_argument("--mfuton-root", type=Path, default=DEFAULT_MFUTON_ROOT)
    ap.add_argument("--output", type=Path, help="optional proof-frame receipt path")
    ap.add_argument("--emit-only", action="store_true")
    ap.add_argument("--print-payload", action="store_true")
    ap.add_argument("--print-form", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--eval-url", default="http://127.0.0.1:6768/eval")
    ap.add_argument("--futon3c-root", type=Path)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        fail(f"run bundle directory not found: {run_dir}")

    bundle_receipt = run_dir / "receipt.json"
    if not bundle_receipt.exists():
        fail(f"missing run bundle receipt: {bundle_receipt}")

    mfuton_root = args.mfuton_root.resolve()
    payload = load_json(bundle_receipt)

    run_id = str(payload.get("run_id") or run_dir.name)
    problem_id = args.problem_id or infer_problem_id(run_dir)
    if not problem_id:
        fail("missing problem-id and could not infer one from the run bundle path")

    frame_id = args.frame_id or run_id
    frame_label = args.frame_label or f"mfuton-local-frontiermath-run:{run_id}"
    writable_root = infer_writable_root(payload, mfuton_root=mfuton_root, run_dir=run_dir)
    artifacts = build_artifacts(run_dir, payload, mfuton_root=mfuton_root)

    emit_script = REPO_ROOT / "scripts" / "frontiermath" / "emit-proof-frame-receipt.py"
    advance_script = (
        REPO_ROOT
        / "scripts"
        / "frontiermath"
        / "advance-proof-cycle-from-frame-receipt.py"
    )

    emit_cmd = [
        sys.executable,
        str(emit_script),
        "--problem-id",
        problem_id,
        "--frame-id",
        frame_id,
        "--frame-label",
        frame_label,
        "--cycle-id",
        args.cycle_id,
        "--boundary-kind",
        "frame",
        "--entrypoint",
        "python scripts/frontiermath/advance-proof-cycle-from-local-run.py",
        "--workdir",
        str(mfuton_root),
        "--case-anchor",
        str(bundle_receipt),
        "--input",
        str(bundle_receipt),
    ]
    if args.blocker_id:
        emit_cmd.extend(["--blocker-id", args.blocker_id])
    if writable_root is not None:
        emit_cmd.extend(["--writable", str(writable_root)])
    for artifact in artifacts:
        emit_cmd.extend(["--artifact", str(artifact)])
    if args.output is not None:
        emit_cmd.extend(["--output", str(args.output.resolve())])

    receipt_path = Path(run_checked(emit_cmd, cwd=REPO_ROOT))
    if args.emit_only:
        print(receipt_path)
        return 0

    advance_cmd = [
        sys.executable,
        str(advance_script),
        str(receipt_path),
        "--problem-id",
        problem_id,
        "--cycle-id",
        args.cycle_id,
        "--eval-url",
        args.eval_url,
    ]
    if args.print_payload:
        advance_cmd.append("--print-payload")
    if args.print_form:
        advance_cmd.append("--print-form")
    if args.submit:
        advance_cmd.append("--submit")
    if args.futon3c_root is not None:
        advance_cmd.extend(["--futon3c-root", str(args.futon3c_root.resolve())])

    output = run_checked(advance_cmd, cwd=REPO_ROOT)
    if output:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
