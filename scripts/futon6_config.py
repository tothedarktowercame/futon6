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
import subprocess
import sys
import urllib.error
import urllib.request
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


def marks() -> Path:
    """The manifest selects run-owned marks; standalone tools retain their default."""
    return path("FUTON6_MARKS", ROOT / "data/showcases/ct-anatomy/golden")


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
    # Children size themselves from the same decision the run record states,
    # so a shard cannot batch differently from what the manifest claims.
    sizing = scale()
    env.setdefault("FUTON6_SHARDS", str(sizing["shards"]))
    env.setdefault("FUTON6_CONCURRENCY", str(sizing["concurrency-per-shard"]))
    env.setdefault("CONCURRENCY", str(sizing["concurrency-per-shard"]))
    for name in ("futon3", "futon3c", "mathlib4", "planetmath", "nlab-content", "nnexus"):
        env[name.upper().replace("-", "_") + "_ROOT"] = str(sibling(name))
    for variable, default in (
        ("FUTON6_SCOPES", storage() / "mark2/ct-fresh-scopes"),
        ("FUTON6_NER_TERMS", storage() / "mark2/ct-handoff/output/ner-terms.json"),
    ):
        env[variable] = str(path(variable, default))
    return env


def _probe(argv: list[str], timeout: float = 5.0) -> str | None:
    """Never let inventory fail a run: an absent or slow tool is simply unknown."""
    if not shutil.which(argv[0]):
        return None
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def _requested_devices() -> tuple[list[str], str, str]:
    """Which GPUs this job may use, on whose authority, and in whose numbering.

    mfuton's surface reads Slurm's allocated GPU IDX from scontrol and
    deliberately does NOT trust CUDA_VISIBLE_DEVICES (ivan, 2026-09-19), so
    where that surface exists it outranks the environment variable. Its values
    are global/physical node indices; CUDA_VISIBLE_DEVICES is masked and
    process-relative. The two are different numbering schemes, so the record
    names which one it holds rather than silently mixing them.
    """
    home = os.environ.get("MFUTON_HOME")
    if home:
        policy = Path(home) / "agent_skills/development/superpod/current-job-gpus.sh"
        if policy.is_file():
            # Default output is JSON; --format ids is the comma-separated form.
            emitted = _probe(["bash", str(policy), "--format", "ids"])
            if emitted:
                listed = [d.strip() for d in emitted.replace("\n", ",").split(",") if d.strip()]
                if listed:
                    return listed, "mfuton current-job-gpus.sh --format ids", "global-physical"

    pinned = os.environ.get("CUDA_VISIBLE_DEVICES")
    if pinned is not None:
        listed = [d for d in pinned.split(",") if d != ""]
        return listed, "CUDA_VISIBLE_DEVICES", "process-visible"

    for variable in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS"):
        allocated = os.environ.get(variable)
        if allocated:
            return [d for d in allocated.split(",") if d != ""], variable, "global-physical"
    on_node = os.environ.get("SLURM_GPUS_ON_NODE")
    if on_node and on_node.isdigit():
        return [str(i) for i in range(int(on_node))], "SLURM_GPUS_ON_NODE", "count-only"

    listing = _probe(["nvidia-smi", "-L"])
    if listing:
        return [str(i) for i, _ in enumerate(listing.splitlines())], "nvidia-smi -L", "process-visible"
    return [], "none detected", "none"


# Where the run happened, not what it ran on. Recorded, never pinned: a resume is
# always a different job and may be a different node.
VOLATILE_HARDWARE_FIELDS = ("slurm-job", "node")


def hardware() -> dict:
    """The actual accelerators, recorded so a run's rate can be read correctly.

    The 0919b probe ran on a fallback box and its rate was later mistaken for
    the pipeline's own, because nothing in the record said what it ran on.
    """
    requested, authority, namespace = _requested_devices()
    query = "index,name,memory.total,compute_cap,driver_version"
    csv = _probe(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    devices = []
    if csv:
        for line in csv.splitlines():
            fields = [f.strip() for f in line.split(",")]
            if len(fields) == 5:
                devices.append({"index": fields[0], "name": fields[1],
                                "memory-mib": int(fields[2]) if fields[2].isdigit() else fields[2],
                                "compute-capability": fields[3], "driver": fields[4]})
    return {"device-authority": authority,
            "device-namespace": namespace,
            "requested": requested,
            "count": len(requested),
            "devices": devices,
            "visible-to-this-process": bool(devices),
            "slurm-job": os.environ.get("SLURM_JOB_ID"),
            "node": os.uname().nodename}


def _endpoint_is_local() -> bool:
    host = urlsplit(endpoint()).hostname or ""
    return host in ("localhost", "127.0.0.1", "::1", "", os.uname().nodename)


def _json_get(url: str, timeout: float = 5.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as answer:
            return json.loads(answer.read().decode())
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None


def serving() -> dict:
    """Which stack is answering, and as what model.

    A bare tag like `llama3.1:70b` does not say whether it was served by vLLM at
    bf16 or by Ollama at Q4 — and a quality baseline cannot leave that implicit.
    """
    base = endpoint()
    root = base[: -len("/v1")] if base.endswith("/v1") else base
    record = {"endpoint-is-local": _endpoint_is_local(), "stack": None,
              "stack-version": None, "served-models": [], "reachable": False}

    version = _json_get(f"{root}/version")            # vLLM
    if isinstance(version, dict) and version.get("version"):
        record["stack"] = "vllm"
        record["stack-version"] = version["version"]
        record["reachable"] = True
    else:
        ollama = _json_get(f"{root}/api/version")     # Ollama
        if isinstance(ollama, dict) and ollama.get("version"):
            record["stack"] = "ollama"
            record["stack-version"] = ollama["version"]
            record["reachable"] = True

    listed = _json_get(f"{base}/models")
    if isinstance(listed, dict):
        record["served-models"] = [entry.get("id") for entry in listed.get("data", [])
                                   if isinstance(entry, dict)]
        # vLLM reports each model's context window; it bounds which proofs fit.
        windows = [entry.get("max_model_len") for entry in listed.get("data", [])
                   if isinstance(entry, dict) and isinstance(entry.get("max_model_len"), int)]
        record["max-model-len"] = min(windows) if windows else None
        record["reachable"] = True
        if record["stack"] is None:
            record["stack"] = "openai-compatible (unidentified)"
    return record


# What the pipeline requires of whoever serves it. The compute side runs what the
# pipeline specifies; this is where the pipeline specifies it (ivan, 2026-09-19).
# The 0919b probe reached S12 against Ollama serving a 4-bit GGUF under a different
# tag, and nothing objected — so the requirement is checked, not just written down.
#
#
# Serving is machine configuration and stays here, out of the run contract. The
# contract names only the model (checkpoint, served name); this checks that the
# endpoint serves that model on a stack that pins precision, with a context long
# enough that proofs are not refused for length.
SERVING_STACK = "vllm"
SERVING_MIN_CONTEXT_TOKENS = 16384


def required_serving() -> dict:
    # Imported here, not at module level: many tools load this file on its own for
    # paths and hardware, and none of them should need the contract to do so.
    import run_contract
    return {"stack": SERVING_STACK, "min-context-tokens": SERVING_MIN_CONTEXT_TOKENS,
            **run_contract.spec()["model"], "contract": run_contract.contract_id()}

# Machine-side suggestions for a batching server. Advice only: a run on one GPU
# at concurrency 1 is slower and exactly as valid.
THROUGHPUT_ADVICE = {
    "prefix-caching": True,
    "concurrency": "32-64 per replica",
    "why": "prefill-dominated workload (5.1:1)",
}

DEVIATION_ENV = "FUTON6_ALLOW_SERVING_DEVIATION"


def serving_conformance(actual: dict | None = None) -> dict:
    """Whether the live endpoint is what the pipeline asked for, and how it differs.

    Recorded rather than raised: a deliberate experiment is legitimate, an
    unnoticed one is not. Preflight decides what to do with `conforms`.
    """
    actual = serving() if actual is None else actual
    required = required_serving()
    deviations = []

    if not actual.get("reachable"):
        deviations.append("endpoint did not answer; serving stack unverified")
    elif actual.get("stack") != required["stack"]:
        deviations.append(
            f"serving stack is {actual.get('stack')!r}, pipeline requires "
            f"{required['stack']!r} — an Ollama or unidentified endpoint "
            f"cannot batch and does not pin precision")

    served = [m for m in actual.get("served-models", []) if m]
    wanted = required["served-as"]
    if served and wanted not in served:
        deviations.append(f"endpoint serves {served}, pipeline requires {wanted!r}")
    if model() != wanted:
        deviations.append(f"MODEL is {model()!r}, pipeline requires {wanted!r}")
    window, floor = actual.get("max-model-len"), required["min-context-tokens"]
    if isinstance(window, int) and window < floor:
        deviations.append(f"endpoint context is {window} tokens, contract requires {floor}: "
                          f"longer proofs would be refused, changing which items succeed")

    return {"conforms": not deviations,
            "deviations": deviations,
            "required": required,
            "override": bool(os.environ.get(DEVIATION_ENV))}


def scale(inventory: dict | None = None) -> dict:
    """Size the run to whatever is free right now.

    One shard per GPU is Rob's convention; concurrency is what turns a shard
    from single-stream into batched, and is the knob the window depends on.
    Both are overridable, and the basis for each choice is recorded.
    """
    inventory = hardware() if inventory is None else inventory
    gpus = inventory["count"]
    local = _endpoint_is_local()

    override = os.environ.get("FUTON6_SHARDS")
    if override and override.isdigit() and int(override) > 0:
        shards, why = int(override), "FUTON6_SHARDS"
    elif local and gpus:
        shards, why = gpus, f"one shard per visible GPU ({inventory['device-authority']})"
    elif not local:
        # A remote endpoint may front any number of GPUs; the local count says nothing.
        shards, why = 1, "single shard: the endpoint is remote, local GPUs do not size it"
    else:
        shards, why = 1, "single shard: no GPU detected on this host"

    requested = os.environ.get("FUTON6_CONCURRENCY")
    if requested and requested.isdigit() and int(requested) > 0:
        concurrency, basis = int(requested), "FUTON6_CONCURRENCY"
    elif serving().get("stack") == "ollama":
        # Ollama serialises by default; more in-flight requests just queue.
        concurrency, basis = 1, "ollama serialises unless OLLAMA_NUM_PARALLEL is raised"
    else:
        concurrency, basis = 32, "default batch concurrency for a batching server"

    return {"shards": shards, "shard-basis": why,
            "concurrency-per-shard": concurrency, "concurrency-basis": basis,
            "max-in-flight": shards * concurrency}


def effective() -> dict:
    inventory = hardware()
    live = serving()
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
        "model-revision": os.environ.get("FUTON6_MODEL_REVISION"),
        # Hardware CAPABILITY is part of the run's identity; hardware IDENTITY is
        # not. A resumed run is, by construction, a different Slurm job and may be
        # on a different node - that is what resuming after an allocation ends
        # means. Pinning "slurm-job"/"node" made every multi-window run
        # unresumable: prepare() compares the whole dict, so the job id alone
        # refused a resume whose GPU model, count, driver and device list were
        # identical. The volatile fields are still RECORDED, just outside the
        # pinned comparison, so the run record still says where it ran.
        # (june, 2026-09-21 - reported upstream.)
        "hardware": {k: v for k, v in inventory.items() if k not in VOLATILE_HARDWARE_FIELDS},
        "hardware-placement": {k: inventory.get(k) for k in VOLATILE_HARDWARE_FIELDS},
        "serving": live,
        "serving-conformance": serving_conformance(live),
        "scale": scale(inventory),
    }


if __name__ == "__main__":
    if sys.argv[1:] == ["--python-argv0"]:
        for arg in python_argv():
            sys.stdout.buffer.write(arg.encode() + b"\0")
    else:
        print(json.dumps(effective(), indent=2, sort_keys=True))
