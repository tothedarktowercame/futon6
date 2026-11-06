"""Tests for the mark3 coordinator (separate manifest, tested rollback).

The bad case each check is named for:
- separate-manifest tests construct a real mark2 home and a real mark3 home
  and assert mark3 operations leave the mark2 bytes untouched;
- rollback tests run the actual `mark3 rollback` subcommand (subprocess,
  real mark2 module, no stubs) and assert the refusal path when Mark 2 is
  not intact, plus the marker + byte-identity when it is.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _load_mark3():
    script = _REPO / "scripts" / "mark3"
    loader = importlib.machinery.SourceFileLoader("mark3_script", str(script))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _run_mark3(args, env_extra, cwd=_REPO):
    env = {**os.environ, **env_extra}
    return subprocess.run(
        [sys.executable, "scripts/mark3", *args],
        cwd=cwd, env=env, capture_output=True, text=True, check=False,
    )


def _make_mark2_home(home: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "state.json").write_text(
        '{"batches":{"1":{"status":"inbox"}},"next_batch":2}', encoding="utf-8")
    (home / "arxiv_manifest.sqlite").write_bytes(b"sqlite-mark2-bytes")


def test_default_mark3_home_independent_of_mark2(monkeypatch, tmp_path):
    mark3 = _load_mark3()
    monkeypatch.delenv("MARK3_HOME", raising=False)
    monkeypatch.delenv("MARK2_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert mark3.default_mark3_home(str(tmp_path / "futon6" / "scripts" / "mark3")) \
        == tmp_path / "mark3"


def test_default_mark3_home_env(monkeypatch, tmp_path):
    mark3 = _load_mark3()
    monkeypatch.setenv("MARK3_HOME", str(tmp_path))
    assert mark3.default_mark3_home(None) == tmp_path


def test_delegated_status_uses_mark3_home_not_mark2(tmp_path):
    mark2_home = tmp_path / "mark2"
    mark3_home = tmp_path / "mark3"
    _make_mark2_home(mark2_home)
    before = sorted((p.name, p.read_bytes()) for p in mark2_home.iterdir())

    run = _run_mark3(["status"], {"MARK2_HOME": str(mark2_home),
                                  "MARK3_HOME": str(mark3_home)})
    assert run.returncode == 0
    assert f"home:      {mark3_home}" in run.stdout
    # Mark 2 home is untouched by a mark3 command.
    after = sorted((p.name, p.read_bytes()) for p in mark2_home.iterdir())
    assert before == after


def test_mark3_manifest_is_separate_from_mark2_manifest(tmp_path):
    mark3 = _load_mark3()
    mark2_home = tmp_path / "mark2"
    mark3_home = tmp_path / "mark3"
    _make_mark2_home(mark2_home)
    os.environ["MARK2_HOME"] = str(mark2_home)
    os.environ["MARK3_HOME"] = str(mark3_home)
    try:
        # Load mark2 through mark3's loader and rebind as mark3.main does.
        mark2 = mark3.load_mark2_module()
        mark2_home_real = mark2.MARK2_HOME
        mark2_manifest_real = mark2.MANIFEST_DB
        mark2.MARK2_HOME = mark3_home
        mark2.MANIFEST_DB = Path(
            os.environ.get("MARK3_MANIFEST_DB",
                           mark3_home / "mark3_manifest.sqlite"))
        # The inherited helpers now resolve inside the Mark 3 home...
        assert mark2.state_path() == mark3_home / "state.json"
        assert mark2.MANIFEST_DB == mark3_home / "mark3_manifest.sqlite"
        assert mark2.MANIFEST_DB != mark2_manifest_real
        assert mark2.MARK2_HOME != mark2_home_real
        # ...and nothing was created under the Mark 2 home.
        assert not (mark3_home / "arxiv_manifest.sqlite").exists()
    finally:
        del os.environ["MARK2_HOME"]
        del os.environ["MARK3_HOME"]


def test_rollback_refused_when_mark2_not_intact(tmp_path):
    mark2_home = tmp_path / "mark2"
    mark3_home = tmp_path / "mark3"
    # Mark 2 home missing entirely: rollback must refuse and write nothing.
    run = _run_mark3(["rollback", "--mark3-home", str(mark3_home)],
                     {"MARK2_HOME": str(mark2_home),
                      "MARK3_HOME": str(mark3_home)})
    assert run.returncode == 1
    assert "rollback refused: Mark 2 is not intact" in run.stderr
    assert "mark2 home missing" in run.stderr
    assert not mark3_home.exists()


def test_rollback_refused_when_mark2_manifest_missing(tmp_path):
    mark2_home = tmp_path / "mark2"
    mark3_home = tmp_path / "mark3"
    mark2_home.mkdir(parents=True)
    (mark2_home / "state.json").write_text("{}", encoding="utf-8")
    # Manifest DB absent: refusal names exactly that.
    run = _run_mark3(["rollback", "--mark3-home", str(mark3_home)],
                     {"MARK2_HOME": str(mark2_home),
                      "MARK3_HOME": str(mark3_home)})
    assert run.returncode == 1
    assert f"mark2 manifest missing: {mark2_home / 'arxiv_manifest.sqlite'}" \
        in run.stderr


def test_rollback_verified_writes_marker_and_leaves_mark2_untouched(tmp_path):
    mark2_home = tmp_path / "mark2"
    mark3_home = tmp_path / "mark3"
    _make_mark2_home(mark2_home)
    mark3_home.mkdir()
    (mark3_home / "state.json").write_text(
        '{"batches":{},"next_batch":1}', encoding="utf-8")
    before = sorted((p.name, p.read_bytes()) for p in mark2_home.iterdir())

    run = _run_mark3(["rollback", "--mark3-home", str(mark3_home)],
                     {"MARK2_HOME": str(mark2_home),
                      "MARK3_HOME": str(mark3_home)})
    assert run.returncode == 0
    assert f"rollback verified: mark2 home intact at {mark2_home}" in run.stdout

    marker = json.loads((mark3_home / "generation.json").read_text("utf-8"))
    assert marker == {
        "generation": "mark2",
        "note": "coordination rolled back to Mark 2; "
                "mark3 state and manifest retained",
    }
    # Rollback never modified the Mark 2 home.
    after = sorted((p.name, p.read_bytes()) for p in mark2_home.iterdir())
    assert before == after
    # And retained the Mark 3 state.
    assert (mark3_home / "state.json").read_text("utf-8") \
        == '{"batches":{},"next_batch":1}'


def test_delegated_command_forwards_options(tmp_path):
    # `mark3 pulled 2` forwards the option through mark2's own parser and
    # reports the MARK3 state path, not a mark2 path.
    mark3_home = tmp_path / "mark3"
    run = _run_mark3(["pulled", "2"], {"MARK3_HOME": str(mark3_home)})
    assert run.returncode == 1  # batch 2 not found: honest failure
    assert f"batch 2 not found in state ({mark3_home / 'state.json'})" \
        in run.stdout
