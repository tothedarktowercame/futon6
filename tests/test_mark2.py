"""Tests for the mark2 batch coordinator."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def _load_mark2():
    root = Path(__file__).parent.parent
    script = root / "scripts" / "mark2"
    loader = importlib.machinery.SourceFileLoader("mark2_script", str(script))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_default_mark2_home_uses_env(monkeypatch, tmp_path: Path):
    mark2 = _load_mark2()
    monkeypatch.setenv("MARK2_HOME", str(tmp_path))

    assert mark2.default_mark2_home("/home/joe/mark2/mark2") == tmp_path


def test_default_mark2_home_uses_deployed_script_directory(monkeypatch):
    mark2 = _load_mark2()
    monkeypatch.delenv("MARK2_HOME", raising=False)

    assert mark2.default_mark2_home("/home/joe/mark2/mark2") == Path("/home/joe/mark2")


def test_default_mark2_home_repo_copy_keeps_user_home(monkeypatch, tmp_path: Path):
    mark2 = _load_mark2()
    monkeypatch.delenv("MARK2_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert mark2.default_mark2_home("/home/joe/code/futon6/scripts/mark2") == tmp_path / "mark2"


def test_pulled_missing_batch_reports_state_path(tmp_path: Path):
    root = Path(__file__).parent.parent
    env = {
        **os.environ,
        "MARK2_HOME": str(tmp_path / "shared-mark2"),
    }

    run = subprocess.run(
        [sys.executable, "scripts/mark2", "pulled", "2"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert run.returncode == 1
    assert f"batch 2 not found in state ({tmp_path / 'shared-mark2' / 'state.json'})" in run.stdout


def test_eprint_fetch_url_uses_export_host(monkeypatch):
    mark2 = _load_mark2()
    monkeypatch.setattr(mark2, "EPRINT_HOST", "export.arxiv.org")

    assert (
        mark2.eprint_fetch_url("https://arxiv.org/e-print/math/0001067v1")
        == "https://export.arxiv.org/e-print/math/0001067v1"
    )


def test_status_reports_batch_fill_config(tmp_path: Path):
    root = Path(__file__).parent.parent
    env = {
        **os.environ,
        "MARK2_HOME": str(tmp_path / "shared-mark2"),
    }

    run = subprocess.run(
        [sys.executable, "scripts/mark2", "status"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert run.returncode == 0
    assert (
        "config:    page_size=5,000 rate_limit=3.0s ready_target=6 "
        "auto_fill=on eprint_host=export.arxiv.org"
    ) in run.stdout


def test_fill_stops_when_ready_target_already_met(tmp_path: Path):
    root = Path(__file__).parent.parent
    home = tmp_path / "shared-mark2"
    home.mkdir()
    (home / "state.json").write_text(
        '{"batches":{"2":{"status":"inbox"}},"next_batch":3}',
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "MARK2_HOME": str(home),
    }

    run = subprocess.run(
        [sys.executable, "scripts/mark2", "fill", "--target-ready", "1", "--if-room"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert run.returncode == 0
    assert "ready target met: 1/1 batch(es) in inbox" in run.stdout


def test_pulled_can_skip_auto_fill(tmp_path: Path):
    root = Path(__file__).parent.parent
    home = tmp_path / "shared-mark2"
    home.mkdir()
    (home / "state.json").write_text(
        '{"batches":{"2":{"status":"inbox"}},"next_batch":3}',
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "MARK2_HOME": str(home),
        "MARK2_AUTO_FILL": "0",
    }

    run = subprocess.run(
        [sys.executable, "scripts/mark2", "pulled", "2"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert run.returncode == 0
    assert "batch 2 marked pulled" in run.stdout
    assert "auto-fill disabled" in run.stdout


def test_returned_cleans_stale_inbox_and_starts_auto_fill(tmp_path: Path, monkeypatch):
    mark2 = _load_mark2()
    home = tmp_path / "shared-mark2"
    home.mkdir()
    monkeypatch.setattr(mark2, "MARK2_HOME", home)

    (home / "state.json").write_text(
        '{"batches":{"2":{"status":"inbox"}},"next_batch":3}',
        encoding="utf-8",
    )
    (home / "inbox").mkdir()
    (home / "outbox").mkdir()
    (home / "inbox" / "batch-002.tar.gz").write_text("input", encoding="utf-8")
    (home / "outbox" / "results-002.tar.gz").write_text("output", encoding="utf-8")

    calls: list[str] = []
    monkeypatch.setattr(mark2, "start_background_fill", lambda reason: calls.append(reason))

    rc = mark2.cmd_returned(type("Args", (), {"batch_num": 2})())

    assert rc == 0
    assert not (home / "inbox" / "batch-002.tar.gz").exists()
    assert calls == ["batch 2 returned"]
    state = json.loads((home / "state.json").read_text(encoding="utf-8"))
    assert state["batches"]["2"]["status"] == "results-ready"
    assert "returned_at" in state["batches"]["2"]


def test_collected_starts_auto_fill_when_ready_target_is_low(tmp_path: Path, monkeypatch):
    mark2 = _load_mark2()
    home = tmp_path / "shared-mark2"
    home.mkdir()
    monkeypatch.setattr(mark2, "MARK2_HOME", home)

    (home / "state.json").write_text(
        '{"batches":{"2":{"status":"results-ready"}},"next_batch":3}',
        encoding="utf-8",
    )
    (home / "outbox").mkdir()
    (home / "outbox" / "results-002.tar.gz").write_text("output", encoding="utf-8")

    calls: list[str] = []
    monkeypatch.setattr(mark2, "start_background_fill", lambda reason: calls.append(reason))

    rc = mark2.cmd_collected(type("Args", (), {"batch_num": 2})())

    assert rc == 0
    assert not (home / "outbox" / "results-002.tar.gz").exists()
    assert calls == ["batch 2 collected"]
    state = json.loads((home / "state.json").read_text(encoding="utf-8"))
    assert state["batches"]["2"]["status"] == "done"
    assert "collected_at" in state["batches"]["2"]


def test_watchdog_alerts_when_lane_needs_fill_and_no_builder(tmp_path: Path, monkeypatch):
    mark2 = _load_mark2()
    home = tmp_path / "shared-mark2"
    home.mkdir()
    monkeypatch.setattr(mark2, "MARK2_HOME", home)

    (home / "state.json").write_text('{"batches":{},"next_batch":9}', encoding="utf-8")
    monkeypatch.setattr(mark2, "manifest_counts", lambda: (10, 20))
    monkeypatch.setattr(mark2, "active_mark2_builder", lambda: (None, None))
    monkeypatch.setattr(mark2, "read_build_lock", lambda: None)

    rc = mark2.cmd_watchdog(type("Args", (), {"email_to": None, "repeat_hours": 12.0})())

    assert rc == 1
    state = json.loads((home / "watchdog-state.json").read_text(encoding="utf-8"))
    assert state["last_status"] == "alert"
    assert "lane stalled" in state["last_alert_key"]


def test_watchdog_accepts_active_builder(tmp_path: Path, monkeypatch):
    mark2 = _load_mark2()
    home = tmp_path / "shared-mark2"
    home.mkdir()
    monkeypatch.setattr(mark2, "MARK2_HOME", home)

    (home / "state.json").write_text('{"batches":{},"next_batch":9}', encoding="utf-8")
    monkeypatch.setattr(mark2, "manifest_counts", lambda: (10, 20))
    monkeypatch.setattr(
        mark2,
        "active_mark2_builder",
        lambda: (1234, "python3 /home/joe/mark2/mark2 fill --if-room"),
    )
    monkeypatch.setattr(mark2, "read_build_lock", lambda: {"pid": "1234"})

    rc = mark2.cmd_watchdog(type("Args", (), {"email_to": None, "repeat_hours": 12.0})())

    assert rc == 0
    state = json.loads((home / "watchdog-state.json").read_text(encoding="utf-8"))
    assert state["last_status"] == "healthy"
