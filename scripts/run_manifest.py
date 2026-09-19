"""Immutable identity and portable artifact paths for a Mark7 run."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone

import futon6_config as config

NAME = "run-manifest.json"
ARTIFACTS = {key: "artifacts/" + key for key in (
    "marks", "loss", "candidates", "graphs", "expo-candidates", "expo",
    "steps", "rung3", "paper-graphs", "clean", "demo")}
EXPOSITORY_SELECTION = "even-spacing-in-source-order/v1"
ENV_KEYS = {key: "FUTON6_" + key.upper().replace("-", "_") for key in ARTIFACTS}
REQUIRED = {
    "marks": (1, "*.json"), "loss": (1, "dashboard.json"),
    "candidates": (3, "*.candidate.json"), "graphs": (3, "*.edn"),
    "expo-candidates": (4, "*.candidate.json"), "expo": (4, "*.edn"),
    "steps": (5, "*.steps.json"), "rung3": (5, "*.json"),
    "paper-graphs": (6, "*.B.json"), "clean": (7, "*.clean.edn"),
    "demo": (7, "clean-embed.json"),
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contained(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"unsafe run-relative path: {relative!r}")
    target = root / relative
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"path escapes run directory: {relative}")
    return target


@contextmanager
def lock(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"run is already in use: {run_dir}") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def identity(cli: str | None, variable: str, label: str) -> str:
    inherited = os.environ.get(variable)
    if cli and inherited and cli != inherited:
        raise ValueError(f"{label} disagrees with {variable}: {cli!r} != {inherited!r}")
    value = cli or inherited
    if not value or value == "adhoc" or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:@+-]*", value):
        raise ValueError(f"explicit non-adhoc {label} required (letters, digits, _ . : @ + -)")
    return value


def source_identity() -> dict:
    paths = [p for base in (config.ROOT / "scripts", config.ROOT / "src")
             for p in base.rglob("*") if p.is_file() and p.suffix in (".py", ".bb", ".clj", ".sh")]
    paths += [p for base in (config.ROOT / "holes", config.ROOT / "resources")
              for p in base.rglob("*.edn") if p.is_file() and
              any(token in p.name for token in ("vocab", "schema", "contract"))]
    paths += [config.ROOT / "holes" / name for name in
              ("superpod-dag-contract.md", "linode-stepper-contract.md")]
    entries = {str(p.relative_to(config.ROOT)): digest(p) for p in sorted(paths)}
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.ROOT,
                            capture_output=True, text=True)
    return {"git-head": result.stdout.strip() if result.returncode == 0 else None,
            "source-sha256": hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()}


def substrate_identity() -> dict:
    files = {name: config.ROOT / "data" / name for name in (
        "warp/concept-index.json", "warp/def-snippets.json", "warp/defined-index.json",
        "warp/concept-usage.json", "concept-encyclopedia-ct.json")}
    files["concept-authority"] = config.authority()
    sibling = config.sibling("futon3")
    files["patterns-index"] = sibling / "resources/sigils/patterns-index.tsv"
    for family in ("math-informal", "math-informal-CT"):
        patterns = sorted((sibling / "library" / family).glob("*.flexiarg"))
        if not patterns:
            raise ValueError(f"missing substrate pattern family: {family}")
        files.update({f"patterns/{family}/{p.name}": p for p in patterns})
    return {name: digest(path) for name, path in sorted(files.items())}


# A stage stops for items the contract refused only when the accepted share falls
# below this. It is not a quality target - every refusal is still recorded and
# excluded from the corpus - it is the line past which the run is no longer
# measuring the papers. Below 3/4 accepted, every collapse we have seen (a rule
# that refused valid mathematics, a stale prompt, an endpoint ignoring the schema)
# was systematic, and the rest of the window would have produced nothing usable.
# Validation corpora are small enough to read item by item: set 1.0 there.
DEFAULT_ITEM_FLOOR = 0.75


def declared_item_floor() -> float:
    """FUTON6_ITEM_FLOOR for a new run, pinned into its manifest at prepare()."""
    raw = os.environ.get("FUTON6_ITEM_FLOOR")
    if raw is None or raw == "":
        return DEFAULT_ITEM_FLOOR
    try:
        floor = float(raw)
    except ValueError:
        raise ValueError("FUTON6_ITEM_FLOOR must be a fraction between 0 and 1")
    if not 0.0 <= floor <= 1.0:
        raise ValueError("FUTON6_ITEM_FLOOR must be a fraction between 0 and 1")
    return floor


def item_floor(doc: dict) -> float:
    """The floor this run is judged by: whatever its manifest pinned.

    Manifests written before the floor existed are judged by the default rather
    than by the old refuse-everything rule, so a run halted by a handful of
    refusals can be resumed in place instead of restarted.
    """
    declared = (doc.get("acceptance") or {}).get("item-floor")
    return DEFAULT_ITEM_FLOOR if declared is None else float(declared)


def load(run_dir: Path) -> dict:
    doc = json.loads((run_dir / NAME).read_text())
    if not isinstance(doc, dict) or doc.get("schema-version") != 1 or doc.get("artifacts") != ARTIFACTS or doc.get("ids") != "corpus.ids.txt":
        raise ValueError("unsupported or malformed run manifest")
    if not all(isinstance(doc.get(key), str) and doc[key] for key in ("run-id", "corpus-id", "corpus-sha256")):
        raise ValueError("manifest has no run/corpus identity")
    for relative in (*doc["artifacts"].values(), doc["ids"], "logs"):
        contained(run_dir, relative)
    if digest(run_dir / doc["ids"]) != doc["corpus-sha256"]:
        raise ValueError("run's frozen corpus manifest changed")
    if [line.strip() for line in (run_dir / doc["ids"]).read_text().splitlines() if line.strip()] != doc.get("papers"):
        raise ValueError("manifest paper list disagrees with frozen corpus")
    return doc


def prepare(run_dir: Path, run_id: str, corpus_id: str, ids: Path) -> dict:
    """Caller holds lock. Never adopt unmanifested artifacts or mutate a resume identity."""
    raw = ids.read_bytes()
    papers = [line.strip() for line in raw.decode().splitlines() if line.strip()]
    if not papers or len(papers) != len(set(papers)):
        raise ValueError("corpus manifest must contain nonempty, unique paper IDs")
    cap = os.environ.get("FUTON6_EXPOSITORY_CAP_PER_PAPER", "0") or "0"
    if not cap.isdigit():
        raise ValueError("FUTON6_EXPOSITORY_CAP_PER_PAPER must be a nonnegative integer")
    cap = int(cap)
    floor = declared_item_floor()
    pinned = {"run-id": run_id, "corpus-id": corpus_id,
              "corpus-sha256": hashlib.sha256(raw).hexdigest(),
              "code": source_identity(), "substrate": substrate_identity(),
              "host-configuration": config.effective(),
              "model-revision": os.environ.get("FUTON6_MODEL_REVISION"),
              "selection": {"all-proofs": True,
                            "expository-cap": cap,
                            # Deferred regions are accounted as deferred, never as accepted.
                            "expository-selection": EXPOSITORY_SELECTION if cap else "all-regions"}}
    if (run_dir / NAME).exists():
        doc = load(run_dir)
        changed = [key for key, value in pinned.items() if doc.get(key) != value]
        if changed:
            raise ValueError("resume identity changed: " + ", ".join(changed) + "; start a new run directory")
        validate_records(run_dir, doc)
        return doc
    occupied = [p.name for p in run_dir.iterdir() if p.name not in (".run.lock", "host-config.jsonl")]
    if occupied:
        raise ValueError(f"refusing to adopt artifacts without a run manifest: {occupied[:5]}")
    doc = {"schema-version": 1, **pinned, "created-at": datetime.now(timezone.utc).isoformat(),
           "ids": "corpus.ids.txt", "papers": papers, "artifacts": ARTIFACTS,
           "acceptance": {"item-floor": floor}, "logs": ["logs/S7.command.log"]}
    (run_dir / doc["ids"]).write_bytes(raw)
    temporary = run_dir / (NAME + ".tmp")
    temporary.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    temporary.replace(run_dir / NAME)
    return doc


def environment(run_dir: Path, doc: dict) -> dict[str, str]:
    return {"RUN_ID": doc["run-id"], "CORPUS": doc["corpus-id"],
            "FUTON6_RUN_DIR": str(run_dir),
            **{ENV_KEYS[key]: str(contained(run_dir, value)) for key, value in doc["artifacts"].items()},
            "MARKS_DIR": str(contained(run_dir, doc["artifacts"]["marks"])),
            "EVAL_REPORT": str(run_dir / "eval-report.json"),
            "EVAL_SUMMARY": str(run_dir / "eval-summary.md")}


def validate_records(run_dir: Path, doc: dict):
    for name in ("phase-ledger.jsonl", "metrics.jsonl", "stage-attempts.jsonl"):
        path = run_dir / name
        if not path.exists():
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if (record.get("run_id"), record.get("corpus_id")) != (doc["run-id"], doc["corpus-id"]):
                raise ValueError(f"{name}:{number}: record belongs to another run/corpus")


def require_artifacts(run_dir: Path, doc: dict, through: str):
    stage = int(through[1:])
    for key, (needed, pattern) in REQUIRED.items():
        if stage >= needed:
            directory = contained(run_dir, doc["artifacts"][key])
            if not any(p.is_file() and p.stat().st_size for p in directory.glob(pattern)):
                raise ValueError(f"missing/empty {key} for {through}: {directory}/{pattern}")
    for name in ("phase-ledger.jsonl", "metrics.jsonl"):
        if not (run_dir / name).is_file() or not (run_dir / name).stat().st_size:
            raise ValueError(f"missing/empty {name}")
    for needed, names in (
        (9, ("pass3-holes.json", "hole-vocabulary.json")),
        (10, ("inference-lexicon.json",)),
        (11, ("structural-canon-defs.json", "structural-canon.json")),
        (12, ("accretion-curve.json",)),
    ):
        if stage >= needed:
            for name in names:
                path = run_dir / name
                if not path.is_file() or not path.stat().st_size:
                    raise ValueError(f"missing/empty {name} for {through}")
    if stage >= 8 and not any(p.is_file() and p.stat().st_size for p in
                            (run_dir / doc["artifacts"]["demo"] / "ingest").rglob("*.json")):
        raise ValueError("missing exported ingest artifacts")
