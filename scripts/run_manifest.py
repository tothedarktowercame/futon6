"""Immutable identity and portable artifact paths for a Mark7 run."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

import futon6_config as config
import run_contract

NAME = "run-manifest.json"
ARTIFACTS = {key: "artifacts/" + key for key in (
    "marks", "loss", "candidates", "graphs", "expo-candidates", "expo",
    "steps", "rung3", "paper-graphs", "clean", "demo")}
# v2: exposition first, then prose inside proofs if the cap leaves room (S3 reads
# proofs); v1 spaced every region evenly, which had no in-proof regions to spend on.
# v3: the cap itself scales with the number of regions the paper carved, so a note and
# a book are not given the same budget (mark3_extract_expository_candidates.scaled_cap).
EXPOSITORY_SELECTION = "exposition-first-scaled-cap/v3"

# A cap of 30 per paper was the same number for a six-page note and for a 40,000-line
# book: 0806.1324 carves 209 regions and S4 read 30, while 0708.2185 carves 27 and lost
# nothing (Joe, 2026-09-22). The cap scales with what the paper has to read. Regions,
# not lines: LaTeX line length varies so much that lines are a poor measure of content
# (0705.0102 has 71 regions in 690 lines, math/0310337 has 247 in 15,103), and regions
# are what is sampled. Sublinear, so one long paper cannot spend the whole window; the
# floor keeps a short paper worth reading. On the 12-paper run this reads 674 regions
# of 1,370 where the fixed 30 read 354.
SCALED = "scaled"
CAP_SCALE, CAP_FLOOR, CAP_CEILING = 6, 12, 120


def scaled_cap(region_count: int) -> int:
    """How many regions to read from a paper that carved `region_count` of them."""
    return max(CAP_FLOOR, min(CAP_CEILING, round(CAP_SCALE * math.sqrt(max(0, region_count)))))


def cap_for(setting, region_count: int) -> int:
    """The cap in force: the scaled rule, a pinned number, or 0 for every region."""
    if setting == SCALED:
        return scaled_cap(region_count)
    return int(setting or 0)


def cap_rule(setting) -> dict | None:
    return ({"scale": CAP_SCALE, "floor": CAP_FLOOR, "ceiling": CAP_CEILING, "of": "regions"}
            if setting == SCALED else None)
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


# Recorded in the manifest, excluded from the resume comparison.
#
# `pinned["host-configuration"]` is the WHOLE `config.effective()` dict, so every
# field in it gates a resume. Splitting the Slurm job id and node out of
# `hardware` into `hardware-placement` (futon6_config.VOLATILE_HARDWARE_FIELDS)
# did not help on its own: the values changed key and stayed inside the same
# pinned dict, so a resume in a new job was still refused with
#   resume identity changed: host-configuration
# On 2026-09-21 that blocked a run resuming at S10 after its allocation ended,
# with S1-S9 already ledgered and nothing about the run itself changed. This is
# the half that makes the split mean anything: placement is still recorded in
# full, and the comparison does not look at it. (june, 2026-09-21)
VOLATILE_CONFIG_KEYS = ("hardware-placement",)


def _identity_view(doc: dict) -> dict:
    """The pinned document as the resume check should see it: placement removed."""
    view = dict(doc)
    host = view.get("host-configuration")
    if isinstance(host, dict):
        view["host-configuration"] = {
            key: value for key, value in host.items() if key not in VOLATILE_CONFIG_KEYS
        }
    return view


# A CHANGED SOURCE TREE DOES NOT DISCARD COMPLETED STAGES (rob, 2026-09-23:
# "the code hash can be computed and reported, but it changing SHOULD NOT
# eliminate partial work").
#
# `code` is a SHA over every .py/.bb/.clj/.sh under scripts/ and src/ plus a few
# contract files. It was gating resume, which meant editing any line anywhere in
# the tree threw away every ledgered stage in every open run - a comment, an
# unrelated script, a fix to a stage that had not run yet. The stages that
# completed are on disk, accounted, and gate-checked; a later edit somewhere
# else does not make them untrue. Worse, it made the repair of a mid-run defect
# cost the whole run, which is exactly when repair is most needed: on 2026-09-21
# a one-line fix to this very file could not be applied without discarding S1-S9.
#
# The hash is still computed and still recorded in the manifest, and a change is
# reported on resume and appended to code-changes.jsonl in the run directory, so
# a run that spans an edit says so in its own record. Reporting it is the useful
# part; refusing on it was not.
#
# Set FUTON6_DISCARD_ON_CODE_CHANGE=1 to get the old behaviour where a changed
# tree refuses the resume. It defaults to off. Use it when the edit plausibly
# changes what the completed stages MEAN - a scoring rule, a gate threshold, a
# schema - and you would rather lose the partial work than mix two definitions
# in one run. That is a judgment about the specific edit, so it belongs to the
# operator, not to a hash that cannot tell the difference.
DISCARD_ON_CODE_CHANGE_ENV = "FUTON6_DISCARD_ON_CODE_CHANGE"
CODE_CHANGES_LOG = "code-changes.jsonl"
_TRUE = {"1", "true", "yes", "on"}


def discard_on_code_change() -> bool:
    """Whether a changed source tree should refuse the resume. Default: no."""
    return os.environ.get(DISCARD_ON_CODE_CHANGE_ENV, "").strip().lower() in _TRUE


def _record_code_change(run_dir: Path, was: dict | None, now: dict) -> None:
    """Append the code change to the run's own record, and say so on stderr.

    Appended to a sidecar rather than written into the manifest: the manifest is
    this run's immutable identity, and the point here is that the source tree is
    NOT part of that identity.
    """
    entry = {"at": datetime.now(timezone.utc).isoformat(), "was": was, "now": now}
    with (run_dir / CODE_CHANGES_LOG).open("a") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    def _short(value: str | None) -> str:
        return (value or "unknown")[:12]

    was_head, now_head = _short((was or {}).get("git-head")), _short(now.get("git-head"))
    # Uncommitted edits leave git-head identical and move only the source hash,
    # so reporting the head alone prints "abc -> abc" and explains nothing.
    if was_head == now_head:
        detail = (f"HEAD {now_head} unchanged, working tree edited "
                  f"(source {_short((was or {}).get('source-sha256'))} -> "
                  f"{_short(now.get('source-sha256'))})")
    else:
        detail = f"{was_head} -> {now_head}"
    print(
        f"NOTE: source tree changed since this run started ({detail}); "
        f"completed stages are kept and the change is recorded in {CODE_CHANGES_LOG}. "
        f"Set {DISCARD_ON_CODE_CHANGE_ENV}=1 to refuse instead.",
        file=sys.stderr, flush=True,
    )


def prepare(run_dir: Path, run_id: str, corpus_id: str, ids: Path) -> dict:
    """Caller holds lock. Never adopt unmanifested artifacts or mutate a resume identity."""
    raw = ids.read_bytes()
    papers = [line.strip() for line in raw.decode().splitlines() if line.strip()]
    if not papers or len(papers) != len(set(papers)):
        raise ValueError("corpus manifest must contain nonempty, unique paper IDs")
    cap = os.environ.get("FUTON6_EXPOSITORY_CAP_PER_PAPER", "0") or "0"
    if cap != SCALED:
        if not cap.isdigit():
            raise ValueError(f"FUTON6_EXPOSITORY_CAP_PER_PAPER must be '{SCALED}' or a nonnegative integer")
        cap = int(cap)
    floor = declared_item_floor()
    pinned = {"run-id": run_id, "corpus-id": corpus_id,
              "corpus-sha256": hashlib.sha256(raw).hexdigest(),
              "code": source_identity(), "substrate": substrate_identity(),
              # Two layers, recorded apart: the contract decides comparability,
              # the host configuration only explains speed.
              "run-contract": run_contract.active(),
              "host-configuration": config.effective(),
              "model-revision": os.environ.get("FUTON6_MODEL_REVISION"),
              "selection": {"all-proofs": True,
                            "expository-cap": cap,
                            # Deferred regions are accounted as deferred, never as accepted.
                            "expository-cap-rule": cap_rule(cap),
                            "expository-selection": EXPOSITORY_SELECTION if cap else "all-regions"}}
    if (run_dir / NAME).exists():
        doc = load(run_dir)
        recorded, current = _identity_view(doc), _identity_view(pinned)
        changed = [key for key, value in current.items() if recorded.get(key) != value]
        if "code" in changed and not discard_on_code_change():
            # Recorded and reported, never a reason to discard completed stages.
            changed.remove("code")
            _record_code_change(run_dir, recorded.get("code"), current["code"])
        if changed:
            detail = "; start a new run directory"
            if changed == ["code"]:
                detail = (f"; {DISCARD_ON_CODE_CHANGE_ENV} is set, so a changed source tree"
                          f" discards completed stages. Unset it to resume in place")
            raise ValueError("resume identity changed: " + ", ".join(changed) + detail)
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
