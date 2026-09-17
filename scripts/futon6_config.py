"""Shared host configuration for Mark7 producers, consumers and entry gates.

No configuration file is implicit. Environment overrides are resolved against
the checkout, so a child launched with another cwd sees the same resource.
Checkout-owned outputs always belong to this checkout, not FUTON_CODE_ROOT.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import sys
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]


def path(variable: str, default: Path) -> Path:
    value = os.environ.get(variable)
    result = Path(value).expanduser() if value else default
    if not result.is_absolute():
        result = ROOT / result
    return result.resolve()


def code_root() -> Path:
    return path("FUTON_CODE_ROOT", ROOT.parent)


def sibling(name: str) -> Path:
    return path(name.upper().replace("-", "_") + "_ROOT", code_root() / name)


def storage() -> Path:
    return path("FUTON6_STORAGE_ROOT", code_root() / "storage")


def eprints() -> Path:
    # Preflight and every reader share this discovery order. An explicit bad
    # override is returned as-is so preflight refuses it, never falls back.
    candidates = (storage() / "futon6/data/arxiv-math-ct-eprints",
                  Path.home() / "data/arxiv-math-ct-eprints",
                  ROOT / "data/arxiv-math-ct-eprints")
    if os.environ.get("FUTON6_EPRINTS"):
        return path("FUTON6_EPRINTS", candidates[0])
    for candidate in candidates:
        if candidate.is_dir() and next(candidate.iterdir(), None) is not None:
            return candidate.resolve()
    return candidates[0]


def anatomy() -> Path:
    return path("FUTON6_ANATOMY", storage() / "futon6/data/ct-anatomy-v0")


def authority() -> Path:
    return path("FUTON6_BACKGROUND_CORPUS_INDEX", ROOT / "data/background-corpus-index.json")


def python_argv() -> list[str]:
    """Interpreter plus flags; shell operators are never interpreted here."""
    command = os.environ.get("FUTON6_PYTHON_CMD")
    argv = shlex.split(command) if command is not None else [sys.executable, "-u"]
    if not argv:
        raise ValueError("FUTON6_PYTHON_CMD must name an interpreter")
    executable = os.path.expanduser(argv[0])
    if "/" in executable:
        candidate = Path(executable)
        # Do not resolve an executable symlink: venv Python needs its venv path.
        executable = os.path.abspath(candidate if candidate.is_absolute() else ROOT / candidate)
    else:
        executable = shutil.which(executable) or executable
    if not Path(executable).is_file() or not os.access(executable, os.X_OK):
        raise ValueError(f"Python interpreter is not executable: {executable}")
    return [executable, *argv[1:]]


def python_command() -> str:
    return shlex.join(python_argv())


def endpoint() -> str:
    return (os.environ.get("OPENAI_BASE_URL") or
            f"http://localhost:{os.environ.get('PORT', '8000')}/v1").rstrip("/")


def model() -> str:
    return os.environ.get("MODEL") or "mark4-70b"


def child_environment() -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "REPO": str(ROOT),  # the S3 shell wrapper must execute this checkout
        "FUTON_CODE_ROOT": str(code_root()),
        "FUTON6_STORAGE_ROOT": str(storage()),
        "FUTON6_EPRINTS": str(eprints()),
        "FUTON6_ANATOMY": str(anatomy()),
        "FUTON6_BACKGROUND_CORPUS_INDEX": str(authority()),
        "FUTON6_PYTHON_CMD": python_command(),
        "FUTON6_PYTHON": python_argv()[0],
        # Babashka reads the same argv without a second shell tokenizer.
        "FUTON6_PYTHON_ARGV_JSON": json.dumps(python_argv()),
        "OPENAI_BASE_URL": endpoint(),
        "MODEL": model(),
    })
    for name in ("futon3", "futon3c", "mathlib4", "planetmath", "nlab-content", "nnexus"):
        env[name.upper().replace("-", "_") + "_ROOT"] = str(sibling(name))
    for variable, default in (
        ("FUTON6_SCOPES", storage() / "mark2/ct-fresh-scopes"),
        ("FUTON6_NER_TERMS", storage() / "mark2/ct-handoff/output/ner-terms.json"),
    ):
        env[variable] = str(path(variable, default))
    return env


def effective() -> dict:
    url = urlsplit(endpoint())
    # Credential-bearing userinfo and queries must not enter the run record.
    public_endpoint = urlunsplit((url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, "", ""))
    return {
        "schema-version": 1,
        "checkout": str(ROOT),
        "code-root": str(code_root()),
        "storage-root": str(storage()),
        "eprints": str(eprints()),
        "anatomy": str(anatomy()),
        "concept-authority": str(authority()),
        "scopes": str(path("FUTON6_SCOPES", storage() / "mark2/ct-fresh-scopes")),
        "ner-terms": str(path("FUTON6_NER_TERMS", storage() / "mark2/ct-handoff/output/ner-terms.json")),
        "siblings": {name: str(sibling(name)) for name in
                     ("futon3", "futon3c", "mathlib4", "planetmath", "nlab-content", "nnexus")},
        "python-argv": python_argv(),
        "endpoint": public_endpoint,
        "model": model(),
    }


if __name__ == "__main__":
    if sys.argv[1:] == ["--python-argv0"]:
        for arg in python_argv():
            sys.stdout.buffer.write(arg.encode() + b"\0")
    else:
        print(json.dumps(effective(), indent=2, sort_keys=True))
