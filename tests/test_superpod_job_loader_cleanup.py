"""Regression tests for explicit DataLoader worker cleanup in superpod-job."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _load_superpod_job():
    root = Path(__file__).parent.parent
    sys.path.insert(0, str(root / "scripts"))
    return importlib.import_module("superpod-job")


class _FakeIterator:
    def __init__(self):
        self.shutdown_calls = 0

    def _shutdown_workers(self):
        self.shutdown_calls += 1


class _FakeLoader:
    def __init__(self, iterator):
        self._iterator = iterator


def test_shutdown_dataloader_workers_shuts_down_loader_iterator():
    mod = _load_superpod_job()
    iterator = _FakeIterator()
    loader = _FakeLoader(iterator)

    mod._shutdown_dataloader_workers(loader)

    assert iterator.shutdown_calls == 1
    assert loader._iterator is None


def test_shutdown_dataloader_workers_prefers_explicit_iterator():
    mod = _load_superpod_job()
    loader_iterator = _FakeIterator()
    explicit_iterator = _FakeIterator()
    loader = _FakeLoader(loader_iterator)

    mod._shutdown_dataloader_workers(loader, explicit_iterator)

    assert explicit_iterator.shutdown_calls == 1
    assert loader_iterator.shutdown_calls == 0
    assert loader._iterator is None
