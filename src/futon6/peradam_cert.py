"""Peradam certificate loader.

The peradam certificate is intentionally file-first: certificate records live
in ``data/peradams/`` and point at the fold-turn deposit plus structured fruit
and consent witnesses.  This loader follows the fold escrow precedent: invalid
records are refused loudly with the failing witness named, while other records
can still load.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import edn_format


DEFAULT_CERT_DIR = Path("/home/joe/code/futon6/data/peradams")
DEFAULT_FOLD_TURNS_DIR = Path("/home/joe/code/futon6/data/fold-turns")


@dataclass(frozen=True)
class Refusal:
    """A machine refusal for cause."""

    cause: str
    witness: str
    message: str
    file: str | None = None


class PeradamRefusal(Exception):
    """Raised when a certificate cannot typecheck as a certified peradam."""

    def __init__(self, refusal: Refusal):
        super().__init__(f"peradam REJECTED [{refusal.cause}] {refusal.message}")
        self.refusal = refusal


def _key(k: Any) -> str:
    name = k.name if hasattr(k, "name") else str(k)
    return name[1:] if name.startswith(":") else name


def plain(value: Any) -> Any:
    """Convert EDN values to Python natives, preserving namespaced key text."""

    if isinstance(value, edn_format.ImmutableDict):
        return {_key(k): plain(v) for k, v in value.items()}
    if isinstance(value, (edn_format.ImmutableList, list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, edn_format.Keyword):
        return value.name
    return value


def load_edn(path: Path) -> dict[str, Any]:
    try:
        value = edn_format.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exact parser errors vary
        _reject("unreadable-edn", "certificate", f"{path}: {exc}", path)
    value = plain(value)
    if not isinstance(value, dict):
        _reject("not-a-map", "certificate", f"{path}: expected an EDN map", path)
    return value


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _present_string(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _reject(cause: str, witness: str, message: str, file: Path | None = None) -> None:
    raise PeradamRefusal(Refusal(cause=cause, witness=witness, message=message, file=str(file) if file else None))


def _resolve_path(raw: str, *, base_dir: Path, repo_root: Path | None) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    local = (base_dir / p).resolve()
    if local.exists():
        return local
    if repo_root is not None:
        rooted = (repo_root / p).resolve()
        if rooted.exists():
            return rooted
    return local


def _id_of(identity: Any) -> str | None:
    if isinstance(identity, dict):
        for k in ("id", "agent", "name"):
            if _present_string(identity.get(k)):
                return identity[k]
    if _present_string(identity):
        return identity
    return None


def _deposit_path(cert: dict[str, Any], fold_turns_dir: Path, cert_dir: Path, repo_root: Path | None) -> Path:
    ref = cert.get("deposit-ref") or cert.get("deposit_ref")
    if _present_string(ref):
        if ref.endswith(".edn") or "/" in ref:
            return _resolve_path(ref, base_dir=cert_dir, repo_root=repo_root)
        return fold_turns_dir / f"{ref}.edn"
    deposit_id = cert.get("deposit-id") or cert.get("fold-turn/id")
    if _present_string(deposit_id):
        return fold_turns_dir / f"{deposit_id}.edn"
    _reject("missing-deposit-ref", "deposit-sha", "certificate needs :deposit-ref or :deposit-id")


def _verify_deposit(cert: dict[str, Any], cert_path: Path, fold_turns_dir: Path, repo_root: Path | None) -> tuple[Path, str, dict[str, Any]]:
    expected = cert.get("deposit-sha") or cert.get("deposit_sha")
    if not _present_string(expected):
        _reject("missing-deposit-sha", "deposit-sha", "certificate lacks :deposit-sha", cert_path)
    path = _deposit_path(cert, fold_turns_dir, cert_path.parent, repo_root)
    if not path.exists():
        _reject("missing-deposit", "deposit-sha", f"deposit file does not resolve: {path}", cert_path)
    actual = sha256_file(path)
    if actual != expected:
        _reject("deposit-sha-mismatch", "deposit-sha", f"stored {expected} vs actual {actual}", cert_path)
    deposit = load_edn(path)
    return path, actual, deposit


def _verify_seal(cert: dict[str, Any], cert_path: Path, repo_root: Path | None) -> tuple[Path, str]:
    seal = cert.get("seal-ref") or cert.get("seal_ref")
    if not isinstance(seal, dict):
        _reject("missing-seal", "seal-ref", "fruit witness missing: :seal-ref must be a structured map", cert_path)
    if seal.get("sealed?") is False or seal.get("sealed") is False or seal.get("status") == "unsealed":
        _reject("unsealed-key", "seal-ref", "fruit witness points at an unsealed key", cert_path)
    raw_path = seal.get("path") or seal.get("file")
    expected = seal.get("sha256") or seal.get("sha")
    if not (_present_string(raw_path) and _present_string(expected)):
        _reject("missing-seal", "seal-ref", ":seal-ref needs :path and :sha256", cert_path)
    path = _resolve_path(raw_path, base_dir=cert_path.parent, repo_root=repo_root)
    if not path.exists():
        _reject("missing-seal", "seal-ref", f"sealed file does not resolve: {path}", cert_path)
    actual = sha256_file(path)
    if actual != expected:
        _reject("seal-sha-mismatch", "seal-ref", f"stored {expected} vs actual {actual}", cert_path)
    return path, actual


def _verify_score(
    cert: dict[str, Any],
    cert_path: Path,
    *,
    deposit_sha: str,
    seal_sha: str,
    repo_root: Path | None,
) -> tuple[Path, str, dict[str, Any]]:
    score = cert.get("blind-score") or cert.get("blind_score")
    if not isinstance(score, dict):
        _reject("missing-blind-score", "blind-score", ":blind-score must be a structured score ref", cert_path)
    raw_path = score.get("path") or score.get("file")
    expected = score.get("sha256") or score.get("sha")
    if not (_present_string(raw_path) and _present_string(expected)):
        _reject("missing-blind-score", "blind-score", ":blind-score needs :path and :sha256", cert_path)
    path = _resolve_path(raw_path, base_dir=cert_path.parent, repo_root=repo_root)
    if not path.exists():
        _reject("missing-blind-score", "blind-score", f"score file does not resolve: {path}", cert_path)
    actual = sha256_file(path)
    if actual != expected:
        _reject("tampered-score", "blind-score", f"stored {expected} vs actual {actual}", cert_path)
    record = load_edn(path)
    if record.get("deposit-sha") != deposit_sha:
        _reject("score-deposit-mismatch", "blind-score", "score record does not bind the deposit sha", cert_path)
    if record.get("seal-sha") != seal_sha:
        _reject("score-seal-mismatch", "blind-score", "score record does not bind the seal sha", cert_path)
    if not _present_string(record.get("verdict")):
        _reject("bad-score-record", "blind-score", "score record needs a structured :verdict", cert_path)
    return path, actual, record


def _verify_identities(cert: dict[str, Any], score_record: dict[str, Any], cert_path: Path) -> tuple[str, str]:
    scorer = _id_of(cert.get("scorer-identity") or cert.get("scorer_identity")) or score_record.get("scorer-id")
    author = _id_of(cert.get("author-identity") or cert.get("author_identity")) or score_record.get("author-id")
    if not _present_string(scorer):
        _reject("missing-scorer-identity", "scorer-identity", "structured scorer identity is required", cert_path)
    if not _present_string(author):
        _reject("missing-author-identity", "author-identity", "structured author identity is required", cert_path)
    if scorer != score_record.get("scorer-id"):
        _reject("score-scorer-mismatch", "blind-score", "certificate scorer does not match score record scorer", cert_path)
    if scorer == author:
        _reject("scorer-is-author", "scorer-not-author", "no-self-certification invariant violated", cert_path)
    if cert.get("scorer-not-author?") is False or cert.get("scorer-not-author") is False:
        _reject("scorer-is-author", "scorer-not-author", "certificate does not assert scorer != author", cert_path)
    return scorer, author


def _verify_arming_or_mana(
    cert: dict[str, Any],
    cert_path: Path,
    *,
    deposit: dict[str, Any],
    repo_root: Path | None,
) -> dict[str, Any]:
    arming = cert.get("arming-ref") or cert.get("arming_ref")
    mana = cert.get("mana-ref") or cert.get("mana_ref")
    if not arming and not mana:
        _reject("missing-arming-or-mana", "arming-or-mana", "certificate needs :arming-ref or :mana-ref", cert_path)
    if isinstance(arming, dict):
        expected_id = arming.get("fold-turn/id") or arming.get("deposit-id")
        deposit_id = deposit.get("fold-turn/id")
        if _present_string(expected_id) and expected_id != deposit_id:
            _reject("arming-ref-mismatch", "arming-or-mana", "arming ref points at a different fold-turn", cert_path)
        if not isinstance(deposit.get("arming"), dict):
            _reject("missing-arming-or-mana", "arming-or-mana", "referenced deposit has no structured :arming", cert_path)
        return {"kind": "arming", "ref": arming}
    if isinstance(mana, dict):
        raw_path = mana.get("path") or mana.get("file")
        expected = mana.get("sha256") or mana.get("sha")
        if not (_present_string(raw_path) and _present_string(expected)):
            _reject("missing-arming-or-mana", "arming-or-mana", ":mana-ref needs :path and :sha256", cert_path)
        path = _resolve_path(raw_path, base_dir=cert_path.parent, repo_root=repo_root)
        if not path.exists():
            _reject("missing-arming-or-mana", "arming-or-mana", f"mana file does not resolve: {path}", cert_path)
        actual = sha256_file(path)
        if actual != expected:
            _reject("mana-sha-mismatch", "arming-or-mana", f"stored {expected} vs actual {actual}", cert_path)
        return {"kind": "mana", "ref": mana}
    _reject("missing-arming-or-mana", "arming-or-mana", ":arming-ref or :mana-ref must be structured", cert_path)


def validate_certificate(
    cert: dict[str, Any],
    cert_path: Path,
    *,
    fold_turns_dir: Path = DEFAULT_FOLD_TURNS_DIR,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Validate one peradam certificate or raise ``PeradamRefusal``."""

    if not isinstance(cert, dict):
        _reject("not-a-map", "certificate", "certificate must be a map", cert_path)
    cert_id = cert.get("peradam/id") or cert.get("id")
    if not _present_string(cert_id):
        _reject("missing-id", "certificate", "certificate needs :peradam/id", cert_path)
    deposit_path, deposit_sha, deposit = _verify_deposit(cert, cert_path, fold_turns_dir, repo_root)
    seal_path, seal_sha = _verify_seal(cert, cert_path, repo_root)
    score_path, score_sha, score_record = _verify_score(
        cert, cert_path, deposit_sha=deposit_sha, seal_sha=seal_sha, repo_root=repo_root
    )
    scorer, author = _verify_identities(cert, score_record, cert_path)
    consent = _verify_arming_or_mana(cert, cert_path, deposit=deposit, repo_root=repo_root)
    return {
        "peradam/id": cert_id,
        "status": "certified",
        "deposit-ref": str(deposit_path),
        "deposit-sha": deposit_sha,
        "seal-ref": str(seal_path),
        "seal-sha": seal_sha,
        "blind-score-ref": str(score_path),
        "blind-score-sha": score_sha,
        "verdict": score_record["verdict"],
        "scorer-id": scorer,
        "author-id": author,
        "scorer-not-author?": True,
        "consent": consent,
        "source-file": str(cert_path),
    }


def load_certificate(path: str | Path, *, fold_turns_dir: str | Path = DEFAULT_FOLD_TURNS_DIR, repo_root: str | Path | None = None) -> dict[str, Any]:
    cert_path = Path(path)
    cert = load_edn(cert_path)
    root = Path(repo_root) if repo_root is not None else None
    return validate_certificate(cert, cert_path, fold_turns_dir=Path(fold_turns_dir), repo_root=root)


def load_certificates(
    cert_dir: str | Path = DEFAULT_CERT_DIR,
    *,
    fold_turns_dir: str | Path = DEFAULT_FOLD_TURNS_DIR,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load a certificate store, preserving refused records for inspection."""

    cert_dir = Path(cert_dir)
    if not cert_dir.exists():
        return {"certified": [], "refused": []}
    root = Path(repo_root) if repo_root is not None else None
    out: dict[str, list[dict[str, Any]]] = {"certified": [], "refused": []}
    for path in sorted(cert_dir.glob("*.edn")):
        try:
            out["certified"].append(load_certificate(path, fold_turns_dir=fold_turns_dir, repo_root=root))
        except PeradamRefusal as exc:
            print(str(exc), file=sys.stderr)
            raw = path.read_text(encoding="utf-8", errors="replace")
            out["refused"].append({**exc.refusal.__dict__, "raw": raw})
    return out


def refusal_for_fold_turn(path: str | Path) -> dict[str, Any]:
    """Classify existing fold-turns that cannot yet issue certificates.

    This is deliberately not issuance.  It records why the current real corpus
    is below the structured certificate bar.
    """

    path = Path(path)
    deposit = load_edn(path)
    deposit_id = deposit.get("fold-turn/id") or path.stem
    if deposit_id == "ft-autoclock-in-001":
        cause = "unstructured-witnesses"
        msg = "ritual evidence exists only as prose; structured seal/score identity witnesses are not loader-checkable"
        witness = "blind-score"
    elif not isinstance(deposit.get("seal-ref"), dict):
        cause = "missing-seal"
        msg = "no structured fruit witness (:seal-ref) on the fold-turn"
        witness = "seal-ref"
    else:
        cause = "missing-certificate"
        msg = "fold-turn is not a certificate record; no peradam is issued"
        witness = "certificate"
    return {
        "fold-turn/id": deposit_id,
        "source-file": str(path),
        "status": "refused",
        "cause": cause,
        "witness": witness,
        "message": msg,
    }


def census_fold_turn_refusals(fold_turns_dir: str | Path = DEFAULT_FOLD_TURNS_DIR) -> list[dict[str, Any]]:
    return [refusal_for_fold_turn(p) for p in sorted(Path(fold_turns_dir).glob("ft-*.edn"))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Load/refuse peradam certificates without live writes.")
    ap.add_argument("--cert-dir", default=str(DEFAULT_CERT_DIR))
    ap.add_argument("--fold-turns-dir", default=str(DEFAULT_FOLD_TURNS_DIR))
    ap.add_argument("--repo-root", default="/home/joe/code/futon6")
    ap.add_argument("--census-fold-turns", action="store_true")
    args = ap.parse_args(argv)

    if args.census_fold_turns:
        print(json.dumps(census_fold_turn_refusals(args.fold_turns_dir), indent=2, sort_keys=True))
        return 0
    loaded = load_certificates(args.cert_dir, fold_turns_dir=args.fold_turns_dir, repo_root=args.repo_root)
    print(json.dumps(loaded, indent=2, sort_keys=True))
    return 1 if loaded["refused"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
