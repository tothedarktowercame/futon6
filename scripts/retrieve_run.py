#!/usr/bin/env python3
"""Package a declared run prefix; verify inventory and replay after extraction."""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

import futon6_config as config
import run_manifest as manifest


def require_prefix(run_dir: Path, doc: dict, through: str):
    manifest.validate_records(run_dir, doc)
    manifest.require_artifacts(run_dir, doc, through)
    rows = [json.loads(line) for line in (run_dir / "phase-ledger.jsonl").read_text().splitlines() if line.strip()]
    passed = {row["stage"] for row in rows if row.get("gate") == "pass"}
    required = {f"S{i}" for i in range(1, int(through[1:]) + 1)}
    if not required <= passed:
        raise ValueError(f"prefix {through} has unpassed stages: {sorted(required - passed)}")


def verify(archive: Path, extract_to: Path | None = None) -> dict:
    if extract_to is not None and extract_to.exists():
        raise ValueError(f"extraction destination already exists: {extract_to}")
    with tempfile.TemporaryDirectory(prefix="mark7-verify-") as directory:
        temporary = Path(directory)
        with tarfile.open(archive, "r:gz") as bundle:
            members = bundle.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)) or any(not member.isfile() for member in members):
                raise ValueError("archive contains duplicate or non-regular members")
            if "inventory.json" not in names:
                raise ValueError("retrieval inventory absent")
            inventory = json.load(bundle.extractfile("inventory.json"))
            if inventory.get("schema-version") != 1:
                raise ValueError("unsupported retrieval inventory")
            if set(names) != set(inventory["files"]) | {"inventory.json"}:
                raise ValueError("archive membership differs from inventory")
            for member in members:
                if member.name == "inventory.json":
                    continue
                if not member.name.startswith("run/"):
                    raise ValueError(f"member outside run: {member.name}")
                target = manifest.contained(temporary, member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as output:
                    shutil.copyfileobj(bundle.extractfile(member), output)
                expected = inventory["files"][member.name]
                if target.stat().st_size != expected["bytes"] or manifest.digest(target) != expected["sha256"]:
                    raise ValueError(f"checksum/size mismatch: {member.name}")
        run_dir = temporary / "run"
        doc = manifest.load(run_dir)
        if (doc["run-id"], doc["corpus-id"]) != (inventory["run-id"], inventory["corpus-id"]):
            raise ValueError("retrieval identity differs from manifest")
        counts = {key: sum(name.startswith("run/" + relative + "/") for name in inventory["files"])
                  for key, relative in doc["artifacts"].items()}
        if counts != inventory["artifact-counts"]:
            raise ValueError("retrieval artifact counts differ from inventory")
        through = inventory["through"]
        if through not in [f"S{i}" for i in range(1, 13)]:
            raise ValueError("invalid prefix in retrieval inventory")
        require_prefix(run_dir, doc, through)
        replay = subprocess.run([*config.python_argv(), str(config.ROOT / "scripts/replay_e2e.py"),
                                 "--run-dir", str(run_dir), "--through", through],
                                capture_output=True, text=True)
        if replay.returncode:
            raise ValueError("retrieved copy failed replay:\n" + replay.stdout + replay.stderr)
        if extract_to is not None:
            extract_to.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(run_dir, extract_to)
        return {"run-id": doc["run-id"], "through": through,
                "files": len(inventory["files"]), "archive-sha256": manifest.digest(archive),
                "artifact-counts": inventory["artifact-counts"], "replay": "pass"}


def pack(run_dir: Path, output: Path, through: str) -> dict:
    run_dir, output = run_dir.resolve(), output.resolve()
    if output.is_relative_to(run_dir) or output.exists():
        raise ValueError("output must be a new archive outside the run directory")
    with manifest.lock(run_dir):
        doc = manifest.load(run_dir)
        require_prefix(run_dir, doc, through)
        paths = []
        for path in sorted(run_dir.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"symlink in run artifacts: {path}")
            if path.is_file() and path.name != ".run.lock":
                paths.append(path)
        files = {"run/" + str(path.relative_to(run_dir)): {"bytes": path.stat().st_size,
                 "sha256": manifest.digest(path)} for path in paths}
        inventory = {"schema-version": 1, "run-id": doc["run-id"], "corpus-id": doc["corpus-id"],
                     "through": through, "files": files,
                     "artifact-counts": {key: sum(path.is_relative_to(run_dir / rel) for path in paths)
                                         for key, rel in doc["artifacts"].items()}}
        output.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=output.parent, prefix=".retrieval-", suffix=".tgz")
        os.close(fd)
        temporary = Path(name)
        try:
            with tarfile.open(temporary, "w:gz") as archive:
                for path in paths:
                    archive.add(path, arcname="run/" + str(path.relative_to(run_dir)), recursive=False)
                data = (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode()
                entry = tarfile.TarInfo("inventory.json")
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))
            # Verify a newly extracted copy, not the source files used to create it.
            report = verify(temporary)
            temporary.replace(output)
            return report
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("pack")
    create.add_argument("--run-dir", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    create.add_argument("--through", choices=[f"S{i}" for i in range(1, 13)], default="S12")
    check = sub.add_parser("verify")
    check.add_argument("archive", type=Path)
    check.add_argument("--extract-to", type=Path)
    args = parser.parse_args()
    try:
        report = pack(args.run_dir, args.output, args.through) if args.action == "pack" else verify(args.archive, args.extract_to)
    except (OSError, ValueError, KeyError, tarfile.TarError) as exc:
        parser.exit(1, f"RETRIEVAL FAILED: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
