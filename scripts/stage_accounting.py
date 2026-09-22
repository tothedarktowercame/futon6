"""Per-item stage accounting shared by producers, the runner, and replay.

A stage command's exit status says whether the process finished; it does not say
which items were attempted, which were accepted, and why the others were not.
Each item-level producer therefore writes one accounting document per runner
invocation, checkpointed after every item so a crash still leaves evidence.

    <run>/accounting/<stage>/<invocation>/<stage>.<producer>.json

The runner decides acceptance from these documents. Two different things can be
wrong with one, and conflating them halted a 124-paper mining run over three
proofs the contract called circular:

  * the accounting is BROKEN - an expected item is unaccounted for, an item is
    recorded twice, output is claimed for an item nobody asked for, or an
    accepted item's artifact is missing. Then the evidence does not describe the
    corpus and nothing downstream can be trusted. This always stops the stage.
  * the contract REFUSED some items - they are rejected or errored. That is a
    finding about the data, which is what a mining run is for. It stops the stage
    only when the accepted share falls below the run's declared floor, because a
    collapse (mark7-val3 run 01 accepted 0 of 20) means the contract, the prompt
    or the endpoint is wrong and the remaining window would be spent on nothing.

Rejected and errored items are kept as evidence and are never converted into
success: they stay rejected in the accounting, in the attempt row and in the
ledger row, and every downstream stage expects only the items actually accepted.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "futon6-stage-accounting/v1"
STATUSES = ("accepted", "rejected", "errored", "deferred")
# Why an accounting does not show every item accepted: the record itself is
# unusable, or the contract refused items it did record.
BROKEN, REFUSED = "broken", "refused"
DIR_ENV = "FUTON6_ACCOUNTING_DIR"
# stage -> [(producer, inputs)]. Inputs name where the expected item ids come
# from: the frozen corpus, or the accepted outputs of an earlier producer (in this
# invocation for the same stage; in the ledgered invocation for another stage).
STAGES = {
    "S3": [("extract", "corpus"), ("loop", "S3.extract")],
    "S4": [("extract", "corpus"), ("select", "S4.extract"), ("loop", "S4.select")],
    "S6": [("assemble", "corpus")],
    "S7": [("typing", "S3.loop")],
}
INVOCATION_ENV = "FUTON6_STAGE_INVOCATION"


def filename(stage: str, producer: str) -> str:
    return f"{stage}.{producer}.json"


class Accounting:
    """Checkpointed accounting for one producer within one invocation.

    Without a destination (standalone use outside the runner) records are kept in
    memory only. A destination file that already exists belongs to another
    process, so the first write refuses rather than merging two histories.
    """

    def __init__(self, stage: str, producer: str, expected, directory: str | os.PathLike | None = None):
        self.stage, self.producer = stage, producer
        self.expected = list(dict.fromkeys(str(item) for item in expected))
        self.items: dict[str, dict] = {}
        directory = directory if directory is not None else os.environ.get(DIR_ENV)
        self.path = Path(directory) / filename(stage, producer) if directory else None
        if self.path is not None and self.path.exists():
            raise ValueError(f"accounting already written for this invocation: {self.path}")
        self.checkpoint()

    def record(self, item, status: str, reason: str = "", *, paper=None,
               artifacts=(), outputs=(), attempts=()):
        item = str(item)
        if status not in STATUSES:
            raise ValueError(f"unknown accounting status {status!r} for {item}")
        if item in self.items:
            raise ValueError(f"{self.stage}.{self.producer}: {item} accounted twice")
        if status in ("rejected", "errored", "deferred") and not reason:
            raise ValueError(f"{self.stage}.{self.producer}: {status} {item} needs a reason")
        self.items[item] = {"id": item, "status": status, "reason": reason,
                            "paper": paper, "artifacts": [str(a) for a in artifacts],
                            "outputs": [str(o) for o in outputs], "attempts": list(attempts)}
        self.checkpoint()

    def counts(self) -> dict[str, int]:
        counts = {status: 0 for status in STATUSES}
        for entry in self.items.values():
            counts[entry["status"]] += 1
        counts["expected"] = len(self.expected)
        counts["unaccounted"] = len(set(self.expected) - set(self.items))
        return counts

    def document(self) -> dict:
        return {"schema": SCHEMA, "stage": self.stage, "producer": self.producer,
                "invocation": os.environ.get(INVOCATION_ENV),
                "updated": datetime.now(timezone.utc).isoformat(),
                "expected": self.expected, "counts": self.counts(),
                "items": list(self.items.values())}

    def checkpoint(self):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_suffix(".partial")
        partial.write_text(json.dumps(self.document(), indent=1) + "\n")
        os.replace(partial, self.path)

    def failed(self) -> bool:
        return any(e["status"] in ("rejected", "errored") for e in self.items.values())


def load(directory: Path, stage: str, producer: str) -> dict:
    path = Path(directory) / filename(stage, producer)
    if not path.is_file():
        raise ValueError(f"missing accounting {path.name} in {directory}")
    doc = json.loads(path.read_text())
    if (doc.get("schema"), doc.get("stage"), doc.get("producer")) != (SCHEMA, stage, producer):
        raise ValueError(f"{path}: accounting schema/stage/producer mismatch")
    return doc


def accepted_outputs(doc: dict) -> list[str]:
    return [o for e in doc["items"] if e["status"] == "accepted" for o in e["outputs"]]


def findings(doc: dict, expected, *, run_dir: Path, allow_deferred: bool = False) -> list[tuple[str, str]]:
    """[(kind, message)] for one producer; kind is BROKEN or REFUSED (empty = all accepted)."""
    label = f"{doc['stage']}.{doc['producer']}"
    found: list[tuple[str, str]] = []
    expected = [str(e) for e in expected]
    # The same input set, in any order: producers enumerate files, upstream lists
    # outputs in paper order (e.g. p10 sorts before p2 by file name).
    if sorted(doc["expected"]) != sorted(set(expected)) or len(doc["expected"]) != len(set(doc["expected"])):
        found.append((BROKEN, f"{label}: declared inputs differ from the upstream inputs "
                              f"({len(doc['expected'])} declared, {len(set(expected))} upstream)"))
    ids = [e["id"] for e in doc["items"]]
    if len(ids) != len(set(ids)):
        found.append((BROKEN, f"{label}: duplicate item records"))
    missing = sorted(set(expected) - set(ids))
    extra = sorted(set(ids) - set(expected))
    if missing:
        found.append((BROKEN, f"{label}: {len(missing)} unaccounted item(s), e.g. {missing[:3]}"))
    if extra:
        found.append((BROKEN, f"{label}: {len(extra)} item(s) outside the inputs, e.g. {extra[:3]}"))
    for status in ("rejected", "errored") + (() if allow_deferred else ("deferred",)):
        bad = [e for e in doc["items"] if e["status"] == status]
        if bad:
            # Deferred outside a declared cap is a bookkeeping fault, not a finding:
            # nobody asked for those items to be skipped.
            kind = REFUSED if status in ("rejected", "errored") else BROKEN
            found.append((kind, f"{label}: {len(bad)} {status}, e.g. "
                          + "; ".join(f"{e['id']}: {e['reason'][:120]}" for e in bad[:3])))
    for entry in doc["items"]:
        if entry["status"] != "accepted":
            continue
        for artifact in entry["artifacts"]:
            path = Path(artifact)
            path = path if path.is_absolute() else Path(run_dir) / path
            if not path.is_file() or not path.stat().st_size:
                found.append((BROKEN, f"{label}: accepted {entry['id']} has missing/empty artifact {artifact}"))
    return found


def problems(doc: dict, expected, *, run_dir: Path, allow_deferred: bool = False) -> list[str]:
    """Every reason this producer did not accept all of its items (strict view)."""
    return [message for _, message in findings(doc, expected, run_dir=run_dir, allow_deferred=allow_deferred)]


def item_yield(doc: dict, allow_deferred: bool = False) -> tuple[int, int]:
    """(accepted, judged) - items the contract judged, excluding declared deferrals.

    Under a declared cap the deferred regions were never offered to the model, so
    counting them against the yield would read a 5-of-28 selection as an 82% loss.
    """
    judged = [e for e in doc["items"]
              if not (allow_deferred and e["status"] == "deferred")]
    return sum(1 for e in judged if e["status"] == "accepted"), len(judged)


def blocking(doc: dict, expected, *, run_dir: Path, allow_deferred: bool = False,
             item_floor: float = 1.0) -> tuple[list[str], list[str]]:
    """(reasons to stop, findings to record and continue past).

    Refusals are recorded and passed over while the accepted share holds at the
    run's floor; below it they become one blocking reason, because at that point
    the stage is evidence about the pipeline rather than about the corpus.
    """
    found = findings(doc, expected, run_dir=run_dir, allow_deferred=allow_deferred)
    stop = [message for kind, message in found if kind == BROKEN]
    noted = [message for kind, message in found if kind == REFUSED]
    accepted, judged = item_yield(doc, allow_deferred)
    if noted and judged and accepted < item_floor * judged:
        stop.append(f"{doc['stage']}.{doc['producer']}: accepted {accepted}/{judged} "
                    f"({accepted / judged:.1%}), below this run's floor of {item_floor:.0%}")
        stop += noted          # stopping for the yield means quoting what was refused
    return stop, noted


def relative(path, run_dir=None) -> str:
    """Artifact reference relative to the run when inside it (so copies replay)."""
    path = Path(path).resolve()
    run_dir = run_dir or os.environ.get("FUTON6_RUN_DIR")
    if run_dir:
        try:
            return str(path.relative_to(Path(run_dir).resolve()))
        except ValueError:
            pass
    return str(path)


# ---- acceptance provenance for model outputs kept across invocations ----
# A retried stage keeps graphs accepted by an earlier invocation of the same run
# (the manifest pins code, model and corpus). A final file only counts as
# accepted when its provenance record names the accepting attempt and its bytes
# still match; anything else is stale or foreign output and is refused.
def _provenance_path(outdir: Path, item: str) -> Path:
    return Path(outdir) / ".accepted" / (item.replace("/", "_") + ".json")


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_acceptance(outdir: Path, item: str, final: Path, attempt: dict):
    path = _provenance_path(outdir, item)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"item": item, "final": Path(final).name, "sha256": _sha256(final),
                                "invocation": os.environ.get(INVOCATION_ENV), **attempt}, indent=1) + "\n")


def publish_accepted(outdir: Path, item: str, final: Path, data: bytes, attempt: dict):
    """Write the acceptance record, then the final, so a crash cannot leave a final
    without provenance (which would block every retry of that item)."""
    import hashlib
    path = _provenance_path(outdir, item)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"item": item, "final": Path(final).name,
                                "sha256": hashlib.sha256(data).hexdigest(),
                                "invocation": os.environ.get(INVOCATION_ENV), **attempt}, indent=1) + "\n")
    partial = Path(final).with_name(Path(final).name + ".partial")
    partial.write_bytes(data)
    os.replace(partial, final)


def carried_acceptance(outdir: Path, item: str, final: Path) -> tuple[dict | None, str]:
    """(provenance, "") when an existing final is a verified earlier acceptance."""
    path = _provenance_path(outdir, item)
    if not path.is_file():
        return None, f"{Path(final).name} exists without acceptance provenance (stale or foreign output)"
    record = json.loads(path.read_text())
    if record.get("final") != Path(final).name or record.get("sha256") != _sha256(final):
        return None, f"{Path(final).name} differs from its recorded acceptance"
    return record, ""


def accepted_finals(outdir: Path, pattern: str = "*.edn") -> tuple[list[tuple[str, Path]], list[tuple[str, str]]]:
    """Consumer view of a producer directory: ([(item, final)], [(name, refusal)]).

    Only finals whose provenance record names them with matching bytes are
    returned; every other final is refused with its reason.
    """
    outdir = Path(outdir)
    by_final = {}
    for record_path in sorted((outdir / ".accepted").glob("*.json")):
        record = json.loads(record_path.read_text())
        by_final[record.get("final")] = record
    accepted, refused = [], []
    for final in sorted(outdir.glob(pattern)):
        if final.name.endswith(".rung2.edn"):
            continue
        record = by_final.get(final.name)
        if record is None:
            refused.append((final.name, "no acceptance provenance (stale or foreign output)"))
        elif record.get("sha256") != _sha256(final):
            refused.append((final.name, "differs from its recorded acceptance"))
        else:
            accepted.append((record["item"], final))
    return accepted, refused


# ---- stage-level verification shared by the runner and replay ----
def directory(run_dir: Path, stage: str, invocation: str) -> Path:
    return Path(run_dir) / "accounting" / stage / invocation


def ledgered_invocation(run_dir: Path, stage: str, corpus_id: str) -> str | None:
    path = Path(run_dir) / "phase-ledger.jsonl"
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("stage") == stage and row.get("corpus_id") == corpus_id and row.get("gate") == "pass":
            return row.get("invocation")
    return None


def stage_problems(run_dir: Path, stage: str, invocation: str, corpus_id: str):
    """(reasons to stop, per-producer counts, refusals to record) for one invocation.

    The floor comes from the run manifest, so the runner and a later replay of the
    retrieved copy judge the same accounting by the same rule.
    """
    import run_manifest
    doc = run_manifest.load(Path(run_dir))
    floor = run_manifest.item_floor(doc)
    loaded, counts, found, noted = {}, {}, [], []
    for producer, source in STAGES.get(stage, []):
        try:
            if source == "corpus":
                expected = doc["papers"]
            else:
                src_stage, src_producer = source.split(".")
                if src_stage == stage:
                    upstream = loaded[source]
                else:
                    upstream_invocation = ledgered_invocation(run_dir, src_stage, corpus_id)
                    if not upstream_invocation:
                        raise ValueError(f"{source}: no ledgered accounting for upstream stage")
                    upstream = load(directory(run_dir, src_stage, upstream_invocation), src_stage, src_producer)
                expected = accepted_outputs(upstream)
            current = load(directory(run_dir, stage, invocation), stage, producer)
        except (KeyError, ValueError, OSError) as exc:
            found.append(f"{stage}.{producer}: {exc}")
            break
        loaded[f"{stage}.{producer}"] = current
        counts[producer] = current["counts"]
        # The cap is a number, or "scaled" (a rule over each paper's region count):
        # either way a nonzero cap is what makes a deferred region legitimate.
        cap = doc["selection"]["expository-cap"]
        allow_deferred = (stage, producer) == ("S4", "select") and bool(cap) and cap != 0
        stop, refused = blocking(current, expected, run_dir=Path(run_dir),
                                 allow_deferred=allow_deferred, item_floor=floor)
        found += stop
        noted += refused
    return found, counts, noted
