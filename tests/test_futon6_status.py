from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
from pathlib import Path


def _load_status():
    root = Path(__file__).parent.parent
    script = root / "scripts" / "futon6-status.py"
    loader = importlib.machinery.SourceFileLoader("futon6_status_script", str(script))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_find_local_mirrors_prefers_sha256_verified_match(tmp_path: Path):
    status = _load_status()
    storage = tmp_path / "storage"
    storage.mkdir()
    mirrored = storage / "superpod-math-processed.tar.gz"
    mirrored.write_bytes(b"same-data")

    remote = {
        "name": mirrored.name,
        "path": "/home/rob/superpod-math-processed.tar.gz",
        "size": mirrored.stat().st_size,
        "sha256": hashlib.sha256(b"same-data").hexdigest(),
    }

    mirrors = status.find_local_mirrors(remote, [storage], True)

    assert len(mirrors) == 1
    assert mirrors[0]["path"] == str(mirrored)
    assert mirrors[0]["evidence"] == ["size", "sha256"]


def test_candidate_delete_records_skips_non_matching_files(tmp_path: Path):
    status = _load_status()
    storage = tmp_path / "storage"
    storage.mkdir()
    local = storage / "results-009.tar.gz"
    local.write_bytes(b"local-data")

    remote = [{
        "name": local.name,
        "path": "/home/joe/mark2/outbox/results-009.tar.gz",
        "size": len(b"remote-data"),
        "sha256": hashlib.sha256(b"remote-data").hexdigest(),
    }]

    records = status.candidate_delete_records(remote, [storage], True)

    assert records == []


def test_build_delete_commands_separates_joe_and_sudo_paths():
    status = _load_status()
    candidates = {
        "mark2": [{
            "remote": {"path": "/home/joe/mark2/outbox/results-007.tar.gz"},
            "mirrors": [{"path": "/tmp/results-007.tar.gz", "evidence": ["size", "sha256"]}],
        }],
        "rob": [{
            "remote": {"path": "/home/rob/superpod-mo-processed.tar.gz"},
            "mirrors": [{"path": "/tmp/superpod-mo-processed.tar.gz", "evidence": ["size", "sha256"]}],
        }],
    }

    commands = status.build_delete_commands("linode-chicago", candidates)

    assert "ssh linode-chicago" in commands["joe"]
    assert "/home/joe/mark2/outbox/results-007.tar.gz" in commands["joe"]
    assert "ssh -t linode-chicago" in commands["sudo"]
    assert "/home/rob/superpod-mo-processed.tar.gz" in commands["sudo"]
