#!/usr/bin/env python3
"""PRE-GO-LIVE check: is everything this run needs actually present and working?

The companion to `replay_e2e.py`. That one asks *did the run produce sound
artifacts?*; this one asks *can the run start at all?* — and it exists because
of a specific failure that is worth stating plainly:

    LaTeXML was audited as the #1 fresh-box hazard on 2026-07-04, fixed the same
    day in `linode-postsetup-deps.sh`, and recorded READY in the readiness
    dashboard (MARK7-DEPS). On 2026-08-07 it was absent from the run host, so
    S11's `:structure` lift had been a silent no-op on every run since.

Nothing was wrong with the fix. What was missing was anything that *verified the
fix had been applied here*. A setup script is a statement of intent; a preflight
is a statement of fact. The difference is the whole of this file.

Checks are declarative and each carries the remedy, so a failure tells you what
to run rather than only what is broken. `--fix` attempts the remedies that are
safe to automate.

  python scripts/preflight.py --ids holes/math-ct-full.ids.txt
  python scripts/preflight.py --ids ... --endpoint http://localhost:8090/v1 --fix
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = []          # (name, ok, detail, remedy)


def rec(name, ok, detail, remedy=""):
    R.append((name, ok, detail, remedy))
    return ok


def sh(cmd, timeout=120):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except Exception as e:
        return 1, str(e)


# ---------------------------------------------------------------- binaries

def check_binaries(fix=False):
    for exe, remedy in [
        ("bb", "curl -sSL https://raw.githubusercontent.com/babashka/babashka/master/install | bash"),
        ("latexmlmath", "apt-get install -y latexml  |  conda install -c conda-forge latexml"),
    ]:
        have = shutil.which(exe)
        if not have and fix:
            sh(f"bash {ROOT}/scripts/linode-postsetup-deps.sh", timeout=1800)
            have = shutil.which(exe)
        rec(f"binary:{exe}", bool(have), have or "NOT FOUND", remedy)


# ---------------------------------------------------- the live bb->latexml chain

def check_structure_chain():
    """The check that would have caught the 2026-08-07 silence: not 'is latexmlmath
    installed' but 'does a formula actually come back as a :structure'."""
    script = os.path.join(ROOT, "scripts", "sfc_def_structure.bb")
    if not shutil.which("bb") or not os.path.exists(script):
        return rec("chain:formula->structure", False, "bb or sfc_def_structure.bb absent",
                   "run scripts/linode-postsetup-deps.sh")
    code, out = sh(f'printf "%s" "x = y + z" | bb {script} -', timeout=180)
    ok = code == 0 and ":structure" in out
    return rec("chain:formula->structure", ok,
               "structure returned" if ok else f"no :structure in output ({out[:80]})",
               "run scripts/linode-postsetup-deps.sh; check latexmlmath")


# ---------------------------------------------------------------- python deps

def check_python():
    missing = []
    for mod in ("edn_format", "numpy", "sentence_transformers"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    rec("python:modules", not missing, "all present" if not missing else f"missing {missing}",
        "pip install " + " ".join(missing) if missing else "")


# ---------------------------------------------------------------- gate scripts

def check_gates():
    need = ["scripts/iatc_argcheck.bb", "scripts/iatc_repair.bb", "scripts/iatc_semcheck.bb",
            "scripts/substance_gate.py", "scripts/expository_argcheck.bb",
            "scripts/clean_vocab_gate.bb", "scripts/clean_entropy_gate.py"]
    missing = [p for p in need if not os.path.exists(os.path.join(ROOT, p))]
    rec("gates:present", not missing, f"{len(need) - len(missing)}/{len(need)} gate scripts",
        f"rsync futon6 to this host (missing {missing[:3]})" if missing else "")


# ---------------------------------------------------------------- substrate

def check_substrate():
    need = ["data/warp/concept-index.json", "data/warp/def-snippets.json",
            "data/warp/defined-index.json", "data/warp/concept-usage.json",
            "data/concept-encyclopedia-ct.json",
            "../futon3/resources/sigils/patterns-index.tsv"]
    missing = [p for p in need if not os.path.exists(os.path.join(ROOT, p))]
    rec("substrate:present", not missing, f"{len(need) - len(missing)}/{len(need)} substrate files",
        "extract data/mark7-substrate.tgz -C ~/code/ (dereference symlinks)" if missing else "")


# ---------------------------------------------------------------- eprints

def check_eprints(ids_file, sample=25):
    d = os.environ.get("FUTON6_EPRINTS")
    if not d:
        return rec("eprints:resolvable", False, "FUTON6_EPRINTS unset",
                   "export FUTON6_EPRINTS=<your arXiv-math eprint dir>")
    if not os.path.isdir(d):
        return rec("eprints:resolvable", False, f"{d} is not a directory",
                   "point FUTON6_EPRINTS at the eprint store")
    if not os.path.exists(ids_file):
        return rec("eprints:resolvable", False, f"id manifest {ids_file} absent", "check --ids")
    ids = [l.strip() for l in open(ids_file) if l.strip()][:sample]
    miss = [p for p in ids
            if not any(os.path.exists(os.path.join(d, p + ext))
                       for ext in (".tar.gz", ".gz", ".tex"))]
    ok = not miss
    return rec("eprints:resolvable", ok,
               f"{len(ids) - len(miss)}/{len(ids)} sampled ids resolve in {d}",
               f"id-form mismatch? we look for <id>.tar.gz|.gz|.tex; missing e.g. {miss[:3]}"
               if miss else "")


# ---------------------------------------------------------------- model endpoint

def check_endpoint(url, model):
    if not url:
        return rec("model:endpoint", False, "no --endpoint given",
                   "pass --endpoint http://host:port/v1")
    code, out = sh(f'curl -s --max-time 10 {url}/models', timeout=30)
    ok = code == 0 and ("data" in out or "models" in out)
    detail = "reachable" if ok else f"unreachable ({out[:60]})"
    if ok and model:
        ok2 = model in out
        detail += f"; model '{model}' {'served' if ok2 else 'NOT in /models'}"
        ok = ok and ok2
    return rec("model:endpoint", ok, detail,
               "start the server, or pass the served model name as --model")


# ---------------------------------------------------------------- disk

def check_optional_inputs():
    """Which optional stage steps will SKIP, decided before the window rather than
    discovered in a log afterwards. A skip is legitimate; a skip nobody knew about
    is how a run silently produces less than its manifest promises."""
    optional = {
        "apm-structure-match (S9)": ["data/apm-proof-scope-audit.json",
                                     "data/nlab-wiring/eprint-scopes.json"],
    }
    lines = []
    for step, needs in optional.items():
        missing = [n for n in needs if not os.path.exists(os.path.join(ROOT, n))]
        lines.append(f"{step}: {'WILL SKIP (' + ', '.join(missing) + ')' if missing else 'will run'}")
    # Never a failure — an optional step is allowed to skip. This check exists so
    # the decision is visible in advance and recorded in the run directory.
    rec("optional:steps", True, "; ".join(lines),
        "if a listed step should run, supply its inputs before starting")


def check_disk(min_gb=50):
    st = os.statvfs(ROOT)
    free = st.f_bavail * st.f_frsize / 1e9
    rec("disk:free", free >= min_gb, f"{free:.0f} GB free",
        f"need >= {min_gb} GB for a full-corpus run" if free < min_gb else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="holes/math-ct-full.ids.txt")
    ap.add_argument("--endpoint", default=os.environ.get("OPENAI_BASE_URL"))
    ap.add_argument("--model", default=os.environ.get("MODEL"))
    ap.add_argument("--fix", action="store_true", help="attempt the automatable remedies")
    ap.add_argument("--min-disk-gb", type=int, default=50)
    a = ap.parse_args()

    ids = a.ids if os.path.isabs(a.ids) else os.path.join(ROOT, a.ids)
    check_binaries(a.fix)
    check_structure_chain()
    check_python()
    check_gates()
    check_substrate()
    check_eprints(ids)
    check_endpoint(a.endpoint, a.model)
    check_optional_inputs()
    check_disk(a.min_disk_gb)

    w = max(len(n) for n, _, _, _ in R)
    bad = [x for x in R if not x[1]]
    print("preflight — can this run start?\n")
    for name, ok, detail, remedy in R:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {name:<{w}}  {detail}")
        if not ok and remedy:
            print(f"         -> {remedy}")
    print(f"\n{len(R) - len(bad)}/{len(R)} checks pass")
    if bad:
        print("\n  DO NOT START. Each failure above names its remedy; --fix attempts the\n"
              "  automatable ones. A setup script having been written is not evidence that\n"
              "  it was run here — that gap is why this file exists.")
    else:
        print("\n  GO — every declared dependency is present and the live chains work.")
    return len(bad)


if __name__ == "__main__":
    sys.exit(main())
