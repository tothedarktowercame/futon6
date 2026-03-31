#!/usr/bin/env python3
"""Create one owner-side proof-frame workspace.

This script creates the bounded workspace that receipts alone do not provide:
- frame-local proof-plan / changelog / execute notes
- frame-local Lean scratch modules under the shared apm-lean project
- explicit metadata tying the workspace back to the frame id
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_APM_LEAN_ROOT = (REPO_ROOT.parent / "apm-lean").resolve()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def clean_segment(raw: str, *, prefix: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]", "_", raw.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = prefix
    if not text[0].isalpha():
        text = f"{prefix}_{text}"
    return text[0].upper() + text[1:]


def frame_root(problem_id: str, frame_id: str) -> Path:
    return REPO_ROOT / ".state" / "proof-frames" / problem_id / frame_id


def build_metadata(problem_id: str, frame_id: str, *, apm_lean_root: Path) -> dict:
    problem_segment = clean_segment(problem_id, prefix="P")
    frame_segment = clean_segment(frame_id, prefix="F")
    workspace = frame_root(problem_id, frame_id)
    lean_root = apm_lean_root / "ApmCanaries" / "Frames" / problem_segment / frame_segment
    local_root = apm_lean_root / "ApmCanaries" / "Local"
    return {
        "workspace/schema": "proof-frame-workspace.v1",
        "workspace/generated-at": now_utc(),
        "workspace/owner-repo": "futon6",
        "proof/problem-id": problem_id,
        "frame/id": frame_id,
        "frame/workspace-root": str(workspace),
        "frame/module-root": f"ApmCanaries.Frames.{problem_segment}.{frame_segment}",
        "frame/lean-root": str(lean_root),
        "frame/shared-extension-root": str(local_root),
        "artifacts": {
            "proof-plan": str(workspace / "proof-plan.edn"),
            "formal-alignment": str(workspace / "formal-alignment.edn"),
            "changelog": str(workspace / "changelog.edn"),
            "execute-notes": str(workspace / "execute.md"),
            "workspace-readme": str(workspace / "README.md"),
            "workspace-metadata": str(workspace / "workspace.json"),
            "lean-main": str(lean_root / "Main.lean"),
            "lean-scratch": str(lean_root / "Scratch.lean"),
        },
    }


def plan_template(problem_id: str, frame_id: str, module_root: str) -> str:
    return (
        "{:problem-id "
        + repr(problem_id)
        + "\n :frame-id "
        + repr(frame_id)
        + "\n :goal \"\"\n :terms []\n :strategy []\n"
        + " :checkpoints [{:stage 1 :status :pending}\n"
        + "               {:stage 2 :status :pending}\n"
        + "               {:stage 3 :status :pending}\n"
        + "               {:stage 4 :status :pending}]\n"
        + " :lean {:module-root "
        + repr(module_root)
        + "\n        :shared-extension-root \"ApmCanaries.Local\"}}\n"
    )


def changelog_template(problem_id: str, frame_id: str) -> str:
    return (
        "[{:kind :workspace-initialized\n"
        + "  :problem-id "
        + repr(problem_id)
        + "\n  :frame-id "
        + repr(frame_id)
        + "\n  :summary \"Created isolated proof frame workspace with plan/changelog/Lean scratch roots.\"}]\n"
    )


def formal_alignment_template(problem_id: str) -> str:
    return (
        "{:problem-id "
        + repr(problem_id)
        + "\n :main-claim {:informal-claim \"\"\n"
        + "             :formal-name \"\"\n"
        + "             :formal-target \"\"}\n"
        + " :alignments []}\n"
    )


def execute_template(metadata: dict) -> str:
    return f"""**Stage 1 — THE CLEAN PROOF**

[Fill in the authoritative reader-facing proof.]

**Stage 2 — LEMMA DEPENDENCY GRAPH**

[Record formal dependency, informal dependency, why-this-now, intended Lean target, and search terms.]

**PROOF-PLAN.EDN**

The machine-readable plan lives at:
{metadata["artifacts"]["proof-plan"]}

The formal/informal alignment artifact lives at:
{metadata["artifacts"]["formal-alignment"]}

**Stage 3 — LEAN FORMALIZATION**

Frame-local Lean files:
- {metadata["artifacts"]["lean-main"]}
- {metadata["artifacts"]["lean-scratch"]}

Shared extension root for promoted lemmas:
- {metadata["frame/shared-extension-root"]}

**Stage 4 — FORMAL-TO-INFORMAL REVISION**

[Back-port calibrated difficulty and local blockers into the reader-facing proof.]
"""


def readme_template(metadata: dict) -> str:
    return f"""# Proof Frame Workspace

Problem: `{metadata["proof/problem-id"]}`
Frame: `{metadata["frame/id"]}`

Workspace root:
- `{metadata["frame/workspace-root"]}`

Lean module root:
- `{metadata["frame/module-root"]}`

Frame-local Lean files:
- `{metadata["artifacts"]["lean-main"]}`
- `{metadata["artifacts"]["lean-scratch"]}`

Shared extension root:
- `{metadata["frame/shared-extension-root"]}`

Discipline:
- exploratory work lives in this frame workspace
- reusable lemmas are promoted explicitly into `ApmCanaries.Local`
- receipts record the frame; they do not replace workspace isolation
"""


def scratch_template(module_root: str) -> str:
    ns = module_root + ".Scratch"
    return f"""import Mathlib

noncomputable section

namespace {ns}

/- Frame-local scratch lemmas live here.
   Promote only stabilized material into ApmCanaries.Local. -/

end {ns}
"""


def main_template(module_root: str) -> str:
    scratch_module = module_root + ".Scratch"
    ns = module_root + ".Main"
    return f"""import Mathlib
import {scratch_module}

noncomputable section

namespace {ns}

/- Authoritative frame-local Lean development.
   Keep exploratory detours in Scratch.lean.
   Promote reusable lemmas explicitly into ApmCanaries.Local. -/

end {ns}
"""


def write_if_absent(path: Path, content: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem-id", required=True)
    ap.add_argument("--frame-id", required=True)
    ap.add_argument("--apm-lean-root", type=Path, default=DEFAULT_APM_LEAN_ROOT)
    ap.add_argument("--force", action="store_true",
                    help="overwrite metadata/templates when they already exist")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    metadata = build_metadata(args.problem_id, args.frame_id, apm_lean_root=args.apm_lean_root.resolve())
    workspace_root = Path(metadata["frame/workspace-root"])
    lean_root = Path(metadata["frame/lean-root"])

    workspace_root.mkdir(parents=True, exist_ok=True)
    lean_root.mkdir(parents=True, exist_ok=True)
    Path(metadata["frame/shared-extension-root"]).mkdir(parents=True, exist_ok=True)

    targets = {
        Path(metadata["artifacts"]["workspace-metadata"]): json.dumps(metadata, indent=2) + "\n",
        Path(metadata["artifacts"]["proof-plan"]): plan_template(
            args.problem_id, args.frame_id, metadata["frame/module-root"]
        ),
        Path(metadata["artifacts"]["formal-alignment"]): formal_alignment_template(args.problem_id),
        Path(metadata["artifacts"]["changelog"]): changelog_template(args.problem_id, args.frame_id),
        Path(metadata["artifacts"]["execute-notes"]): execute_template(metadata),
        Path(metadata["artifacts"]["workspace-readme"]): readme_template(metadata),
        Path(metadata["artifacts"]["lean-scratch"]): scratch_template(metadata["frame/module-root"]),
        Path(metadata["artifacts"]["lean-main"]): main_template(metadata["frame/module-root"]),
    }

    for path, content in targets.items():
        if args.force:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        else:
            write_if_absent(path, content)

    print(Path(metadata["artifacts"]["workspace-metadata"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
