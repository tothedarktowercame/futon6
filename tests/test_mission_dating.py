"""Fixtures-only tests for mission_dating (no live data, no substrate)."""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mission_dating as md


def _git(repo, *args, env=None):
    import os
    e = dict(os.environ)
    e.update({
        "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@t",
    })
    e.update(env or {})
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, env=e, check=True)


def _commit(repo, date, msg):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", msg,
         env={"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date})


def _write(repo, rel, text):
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


@pytest.fixture()
def repo(tmp_path):
    repo = tmp_path / "futonX"
    repo.mkdir()
    _git(repo, "init")
    return repo


def test_admin_commit_does_not_move_last_work(repo):
    # one admin commit touching 6 mission docs, one work commit touching 1
    for i in range(3):
        _write(repo, f"holes/missions/M-a{i}.md", f"worked line {i}\n")
    _commit(repo, "2026-01-10T10:00:00+00:00", "work: three docs")
    for i in range(3, 6):
        _write(repo, f"holes/missions/M-a{i}.md", f"worked line {i}\n")
    _commit(repo, "2026-01-11T10:00:00+00:00", "work: three more docs")
    for i in range(6):
        p = repo / f"holes/missions/M-a{i}.md"
        p.write_text(p.read_text() + "admin sweep line\n")
    _commit(repo, "2026-09-01T10:00:00+00:00", "mission-accept sweep")
    # a real work commit on one doc, earlier than the sweep
    p = repo / "holes/missions/M-a0.md"
    p.write_text(p.read_text() + "real work\n")
    _commit(repo, "2026-03-01T10:00:00+00:00", "real work on M-a0")

    docs = {f"M-a{i}": repo / f"holes/missions/M-a{i}.md" for i in range(6)}
    table = md.build_table(docs, [])
    assert table["admin_commit_threshold"] == 5
    assert len(table["admin_commits"]) == 1
    admin = table["admin_commits"][0]
    assert admin["n_docs"] == 6
    assert admin["subject"] == "mission-accept sweep"
    m0 = table["missions"]["M-a0"]
    # last_work is the March work commit, not the September admin sweep
    assert m0["last_work"] == "2026-03-01T10:00:00+00:00"
    assert m0["last_admin"] == "2026-09-01T10:00:00+00:00"
    assert m0["n_work_commits"] == 2
    assert m0["n_admin_commits"] == 1
    m1 = table["missions"]["M-a1"]
    assert m1["last_work"] == "2026-01-10T10:00:00+00:00"
    assert m1["date_source"] == "git"


def test_scope_passage_not_found_falls_back(repo):
    _write(repo, "holes/missions/M-b.md", "alpha\nbeta passage here\ngamma\n")
    _commit(repo, "2026-02-02T10:00:00+00:00", "work")
    scopes = [{"scope/id": "M-b/section/nowhere", "mission": "M-b",
               "binder": "loose-section",
               "passage": "this passage is not in the doc"},
              {"scope/id": "M-b/section/beta", "mission": "M-b",
               "binder": "loose-section", "passage": "beta passage here"}]
    table = md.build_table({"M-b": repo / "holes/missions/M-b.md"}, scopes)
    s = table["scopes"]
    assert s["M-b/section/nowhere"]["date_source"] == "mission-fallback"
    assert s["M-b/section/nowhere"]["dated"] == "2026-02-02T10:00:00+00:00"
    assert s["M-b/section/beta"]["date_source"] == "blame"
    assert s["M-b/section/beta"]["dated"] == "2026-02-02T10:00:00+00:00"


def test_undated_mission_is_null_not_epoch(tmp_path):
    # a mission doc that exists on disk but has no git history
    doc = tmp_path / "elsewhere" / "holes" / "missions" / "M-c.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("no history\n")
    # needs a git repo for repo_root: init one around it but never commit
    repo = tmp_path / "elsewhere"
    _git(repo, "init")
    table = md.build_table({"M-c": doc},
                           [{"scope/id": "M-c/section/x", "mission": "M-c",
                             "binder": "loose-section", "passage": "no history"}])
    m = table["missions"]["M-c"]
    assert m["date_source"] == "undated"
    assert m["born"] is None and m["last_work"] is None
    assert m["n_work_commits"] == 0
    s = table["scopes"]["M-c/section/x"]
    assert s["date_source"] == "undated"
    assert s["dated"] is None


def test_admin_only_mission_is_dated_not_undated(repo):
    # a bulk import creates 6 docs and nothing touches them again
    for i in range(6):
        _write(repo, f"holes/missions/M-imp{i}.md", f"imported {i}\n")
    _commit(repo, "2026-02-18T10:00:00+00:00", "import missions")
    docs = {f"M-imp{i}": repo / f"holes/missions/M-imp{i}.md" for i in range(6)}
    table = md.build_table(docs, [{"scope/id": "M-imp0/section/missing",
                                   "mission": "M-imp0", "binder": "loose-section",
                                   "passage": "not in the doc"}])
    m = table["missions"]["M-imp0"]
    assert m["date_source"] == "git-admin-only"
    assert m["last_work"] is None
    assert m["last_admin"] == "2026-02-18T10:00:00+00:00"
    s = table["scopes"]["M-imp0/section/missing"]
    assert (s["date_source"], s["dated"]) == ("mission-fallback", "2026-02-18T10:00:00+00:00")


def test_canonical_doc_skips_run_copies(tmp_path):
    real = tmp_path / "futon2/holes/missions/M-u.md"
    copies = [tmp_path / "futon3c/holes/labs/wm-contract/runs/R1/authority/holes/missions/M-u.md",
              tmp_path / "futon2/holes/labs/wm-contract/runs/P/draft-missions/M-u.md"]
    assert md.canonical_doc(copies + [real]) == real
    assert md.canonical_doc([real] + copies) == real
