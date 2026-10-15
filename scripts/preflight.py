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
import glob
import json
import os
import shutil
import subprocess
import sys
import re
import shlex
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import futon6_config as config
import anatomy_v0_sweep as sweep

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = []          # (name, ok, detail, remedy)


def rec(name, ok, detail, remedy=""):
    R.append((name, ok, detail, remedy))
    return ok


def sh(cmd, timeout=120):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout,
                           cwd=ROOT, env=config.child_environment())
        return p.returncode, (p.stdout + p.stderr).strip()
    except Exception as e:
        return 1, str(e)


# ---------------------------------------------------------------- binaries

def check_binaries(fix=False):
    for exe, remedy in [
        ("bb", "curl -sSL https://raw.githubusercontent.com/babashka/babashka/master/install | bash"),
        ("latexmlmath", "install LaTeXML following https://math.nist.gov/~BMiller/LaTeXML/get.html (see docs/latexml-setup.md); put latexmlmath on PATH"),
    ]:
        have = shutil.which(exe)
        if not have and fix:
            sh(f"bash {shlex.quote(ROOT + '/scripts/linode-postsetup-deps.sh')}", timeout=1800)
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
    code, out = sh(f'printf "%s" "x = y + z" | bb {shlex.quote(script)} -', timeout=180)
    ok = code == 0 and ":structure" in out
    return rec("chain:formula->structure", ok,
               "structure returned" if ok else f"no :structure in output ({out[:80]})",
               "run scripts/linode-postsetup-deps.sh; check latexmlmath")


# ------------------------------------------------- gate scripts' own interpreter

def check_gate_subprocess():
    """Run a gate the way the pipeline runs it, not the way this file imports.

    `check_python` imports edn_format into *preflight's* interpreter and passes.
    But `iatc_semcheck.bb` shells out to its own Python, and on 2026-08-07 that
    was the system interpreter, which has no edn_format — so R2d raised, the
    composer reported the graph failed, and rung-2 read as 0/98 corpus-wide when
    the true figure was 49/98. Preflight was green throughout.

    The lesson is H21's, one level in: it is not enough to check that a
    dependency is installed, or even that it is importable *here*. The check has
    to traverse the same path the caller traverses.
    """
    script = os.path.join(ROOT, "scripts", "iatc_semcheck.bb")
    if not shutil.which("bb") or not os.path.exists(script):
        return rec("chain:gate-subprocess", False, "bb or iatc_semcheck.bb absent",
                   "run scripts/linode-postsetup-deps.sh")
    graphs = sorted(glob.glob(os.path.join(ROOT, "data", "iatc-argument-graphs", "run", "*.edn")))
    graphs = [g for g in graphs if not g.endswith(".rung2.edn")]
    if not graphs:
        return rec("chain:gate-subprocess", True, "no graphs yet — nothing to exercise")
    # The semcheck composer now loads each graph's candidate from
    # <graphs-dir>/../candidates/<name>.candidate.json, the layout a run writes.
    # The committed fixtures predate that and have no candidates beside them, so
    # semcheck stopped at "candidate source-window not found" before R2d ever
    # ran, on every host (2026-10-15). Give it the run layout: the fixture graph
    # plus a candidate carrying the graph's own identity and line range. Its
    # text is blank -- this check is about R2d's interpreter, not anchors.
    import tempfile
    import edn_format
    with tempfile.TemporaryDirectory(prefix="futon6-preflight-") as tmp:
        src = graphs[0]
        name = os.path.basename(src)
        gdir = os.path.join(tmp, "graphs")
        cdir = os.path.join(tmp, "candidates")
        os.makedirs(gdir)
        os.makedirs(cdir)
        shutil.copy(src, os.path.join(gdir, name))
        g = edn_format.loads(open(src, encoding="utf-8").read())
        kw = edn_format.Keyword
        lo, hi = g[kw("source")][kw("lines")]
        candidate = {"paper-id": g[kw("paper/id")], "passage-id": g[kw("passage/id")],
                     "window-lines": [lo, hi], "source-window": "\n" * (hi - lo)}
        with open(os.path.join(cdir, name[:-len(".edn")] + ".candidate.json"), "w") as f:
            json.dump(candidate, f)
        code, out = sh(shlex.join(['bb', script, os.path.join(gdir, name)]), timeout=300)
    # PASS or FAIL are both fine; what must not happen is R2d failing to report.
    ok = "R2d" in out and "concept-coverage" in out and "R2d concept coverage failed" not in out
    return rec("chain:gate-subprocess", ok,
               "R2d reports through the composer" if ok
               else f"R2d did not report ({out[-120:]})",
               "the gate's own interpreter lacks a module — check python-bin in iatc_semcheck.bb")


# ---------------------------------------------------------------- python deps

def check_python():
    # Check the interpreter used by stages, even when preflight itself was
    # launched with another Python. Preserve actual import checks, not find_spec.
    probe = '''import importlib, json
missing = []
for mod in ("edn_format", "numpy", "sentence_transformers"):
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(mod)
print(json.dumps(missing))
'''
    result = subprocess.run([*config.python_argv(), "-c", probe], cwd=ROOT,
                            env=config.child_environment(), capture_output=True, text=True)
    if result.returncode:
        return rec("python:modules", False, "configured Python import probe failed: " + result.stderr[-300:],
                   "check FUTON6_PYTHON_CMD and install the pipeline dependencies there")
    missing = json.loads(result.stdout.splitlines()[-1])
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

def check_concept_authority():
    from concept_authority import ConceptAuthority, configured_index
    path = configured_index()
    try:
        authority = ConceptAuthority(path)
    except (OSError, ValueError, TypeError) as exc:
        return rec("substrate:concept-authority", False, str(exc),
                   "extract the updated Mark7 substrate bundle; optionally set "
                   "FUTON6_BACKGROUND_CORPUS_INDEX to its schema-2 authority index")
    return rec("substrate:concept-authority", True,
               f"{path}: {authority.meta['term-keys']} terms; Hom/End/colim resolve")


def check_substrate():
    check_concept_authority()
    warp = config.warp()
    need = [str(warp / "concept-index.json"), str(warp / "def-snippets.json"),
            str(warp / "defined-index.json"), str(warp / "concept-usage.json"),
            str(config.sibling("futon3") / "resources/sigils/patterns-index.tsv")]
    # A new subject's encyclopedia is an S2 WARP output.  Requiring it before
    # that stage makes default-on WARP impossible for precisely the subjects it
    # is meant to build.  Disabled/legacy runs still require the staged file.
    if os.environ.get("FUTON6_ENABLE_WARP") != "1":
        need.insert(-1, str(config.concept_encyclopedia()))
    missing = [p for p in need if not os.path.exists(os.path.join(ROOT, p))]
    rec("substrate:present", not missing, f"{len(need) - len(missing)}/{len(need)} substrate files",
        "extract data/mark7-ct-substrate.tgz at the configured checkout and FUTON3_ROOT" if missing else "")

    # The mining reads ONE pattern family group. cas_select filters the index by
    # FAMILY_PREFIX = "math-informal", so of 1134 library patterns exactly 45 are
    # reachable by this run; the rest (257 iiching, 64 iching, 58 p4ng, ...) are
    # noise here. Check the families the run can actually use are present, rather
    # than checking a 1355-row index exists and calling that substrate.
    fam_root = str(config.sibling("futon3") / "library")
    fams, missing_fams = {}, []
    for fam in ("math-informal", "math-informal-CT"):
        d = os.path.join(fam_root, fam)
        n = len(glob.glob(os.path.join(d, "*.flexiarg"))) if os.path.isdir(d) else 0
        fams[fam] = n
        if n == 0:
            missing_fams.append(fam)
    rec("substrate:ct-patterns", not missing_fams,
        ", ".join(f"{k}={v}" for k, v in fams.items()),
        "extract data/mark7-ct-substrate.tgz at the configured checkout and FUTON3_ROOT" if missing_fams else "")


# ---------------------------------------------------------------- eprints

def resolve_eprints():
    """Inspect exactly the path the producers use, including an explicit bad override."""
    location = config.eprints()
    if location.is_dir():
        return str(location), "FUTON6_EPRINTS" if os.environ.get("FUTON6_EPRINTS") else "discovered"
    return None, f"eprint store {location} is not a directory"


def check_eprints(ids_file, sample=25):
    d, how = resolve_eprints()
    if not d:
        return rec("eprints:resolvable", False, how,
                   "export FUTON6_EPRINTS=<your arXiv-math eprint dir>, or fetch with "
                   "scripts/fetch-arxiv-eprints.py")
    if not os.path.exists(ids_file):
        return rec("eprints:resolvable", False, f"id manifest {ids_file} absent", "check --ids")
    ids = [l.strip() for l in open(ids_file) if l.strip()][:sample]
    miss = [p for p in ids
            if not any(os.path.exists(os.path.join(d, p + ext))
                       for ext in (".tar.gz", ".gz", ".tex"))]
    ok = not miss
    return rec("eprints:resolvable", ok,
               f"{len(ids) - len(miss)}/{len(ids)} sampled ids resolve in {d} ({how})",
               f"id-form mismatch? we look for <id>.tar.gz|.gz|.tex; missing e.g. {miss[:3]}"
               if miss else "")


# ---------------------------------------------------------------- vocabulary fit

VOCAB_FLOOR_ENV = "FUTON6_VOCAB_FIT_FLOOR"


def _canon_words(text: str) -> str:
    text = re.sub(r"\$[^$]*\$|\\\(.*?\\\)", " ", text, flags=re.S)
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    return " " + " ".join(re.findall(r"[a-z]+", text.lower())) + " "


def _defined_terms(tex: str) -> set[str]:
    """Terms a lexicon paper defines: \\emph/\\textbf inside definition environments."""
    out = set()
    for m in re.finditer(r"\\begin\{(?:definition|defn|dfn)\*?\}(.*?)\\end\{(?:definition|defn|dfn)\*?\}",
                         tex, re.S):
        for term in re.findall(r"\\(?:emph|textbf)\{([^{}]{3,60})\}", m.group(1)):
            words = " ".join(re.findall(r"[a-z]+", re.sub(r"\$[^$]*\$", " ", term).lower()))
            if 1 <= len(words.split()) <= 4 and len(words) > 3:
                out.add(words)
    return out


def _paper_text(pid: str, root: str) -> str | None:
    for ext in (".tex", ".tar.gz", ".gz", ""):
        path = os.path.join(root, pid + ext)
        if os.path.isfile(path):
            try:
                files, _meta = sweep.read_eprint_files(Path(path))
            except Exception:
                return None
            return " ".join(f["text"] for f in files if f["file"].endswith(".tex"))
    return None


def check_vocabulary(ids_file, sample=200):
    """Does the concept vocabulary belong to this corpus? A classical scan.

    The staged vocabulary is a subject's (math.CT unless FUTON6_SUBJECT says
    otherwise), and outside arXiv math.CT it is demo content. On 2026-10-15 a
    corpus of analysis and topology proofs ran with it: S2 grounded nothing and
    no check said so before GPU time was spent. This counts the distinct
    vocabulary terms in a sample of the run's papers -- encyclopedia concepts
    plus terms defined in FUTON6_WARP_LEXICON_IDS papers -- and warns when too
    few papers carry at least three. Measured: math.CT papers with the CT
    vocabulary, 98% (median 35 terms); APM proofs with it, 8% (median 1); APM
    proofs with Joe's Xi as lexicon, 100% (median 11). A warning, not a
    refusal: a corpus may legitimately build its vocabulary at S2.
    """
    d, _how = resolve_eprints()
    if not d or not os.path.exists(ids_file):
        return rec("vocabulary:fit", None, "no eprints or id list to scan", "")
    vocab, sources = set(), []
    enc = config.concept_encyclopedia()
    if enc.exists():
        try:
            entries = json.loads(enc.read_text()).get("entries", [])
            vocab |= {e["concept"].lower() for e in entries if e.get("concept")}
            sources.append(f"{config.subject()} encyclopedia ({len(entries)})")
        except (ValueError, OSError):
            pass
    lex_file = os.environ.get("FUTON6_WARP_LEXICON_IDS")
    if lex_file and os.path.exists(lex_file):
        for pid in [l.strip() for l in open(lex_file) if l.strip() and not l.startswith("#")]:
            text = _paper_text(pid, d)
            if text:
                terms = _defined_terms(text)
                vocab |= terms
                sources.append(f"lexicon {pid} ({len(terms)})")
    # main() exports FUTON6_SUBJECT before the checks run, so "was it set" cannot
    # be asked here; the CT vocabulary is the demo one whoever chose it.
    building = os.environ.get("FUTON6_BUILD_LEXICON") == "1"
    if building:
        # --build-lexicon: S2 will scan the run's papers against the background
        # index. Run the same scan on the sample now and judge that vocabulary.
        import build_lexicon
        sample_ids = [l.strip() for l in open(ids_file) if l.strip()]
        sample_ids = sample_ids[::max(1, len(sample_ids) // sample)][:sample]
        cand = build_lexicon.vocabulary(config.authority())
        df, read = build_lexicon.scan(sample_ids, cand)
        built = build_lexicon.select(df, read, cand, None, 0.9)
        vocab |= {t["term"] for t in built}
        top = ", ".join(f"{m}" for m, _n in build_lexicon.msc_profile(built)[:4])
        sources.append(f"lexicon to be built at S2 ({len(built)} terms from the sample; MSC {top})")
    demo = config.subject() == "ct"
    if not vocab:
        return rec("vocabulary:fit", None, "no vocabulary to scan against",
                   "stage a concept encyclopedia, or name a concept index with FUTON6_WARP_LEXICON_IDS")
    ids = [l.strip() for l in open(ids_file) if l.strip()]
    step = max(1, len(ids) // sample)
    counts = []
    for pid in ids[::step][:sample]:
        text = _paper_text(pid, d)
        if text is not None:
            words = _canon_words(text)
            counts.append(sum(1 for v in vocab if f" {v} " in words))
    if not counts:
        return rec("vocabulary:fit", None, "no sampled paper could be read", "")
    counts.sort()
    share = sum(c >= 3 for c in counts) / len(counts)
    floor = float(os.environ.get(VOCAB_FLOOR_ENV, "0.5"))
    detail = (f"{share:.0%} of {len(counts)} sampled papers carry >=3 vocabulary terms "
              f"(median {counts[len(counts) // 2]}; floor {floor:.0%}); vocabulary: "
              + ", ".join(sources))
    if share >= floor:
        return rec("vocabulary:fit", True, detail)
    remedy = ("the vocabulary does not match these papers"
              + (" -- it is the math.CT vocabulary, demo content outside arXiv math.CT"
                 " (a custom or cross-classification collection needs its own)" if demo else "")
              + (". Even the lexicon built from these papers is thin: check the eprints, or"
                 " name a concept index with FUTON6_WARP_LEXICON_IDS" if building else
                 ". Re-run with --build-lexicon to build one from these papers with the"
                 " classical scanner (scripts/build_lexicon.py), or name the corpus's own"
                 " concept index with FUTON6_WARP_LEXICON_IDS"))
    return rec("vocabulary:fit", None, detail, remedy)


# ---------------------------------------------------------------- model endpoint

def check_endpoint(url, model):
    if not url:
        return rec("model:endpoint", False, "no --endpoint given",
                   "pass --endpoint http://host:port/v1")
    code, out = sh(shlex.join([
        "curl", "-s", "--max-time", "10", "-H",
        "Authorization: Bearer " + os.environ.get("OPENAI_API_KEY", "x"),
        url.rstrip("/") + "/models",
    ]), timeout=30)
    ok = code == 0 and ("data" in out or "models" in out)
    detail = "reachable" if ok else f"unreachable ({out[:60]})"
    if ok and model:
        ok2 = model in out
        detail += f"; model '{model}' {'served' if ok2 else 'NOT in /models'}"
        ok = ok and ok2
    return rec("model:endpoint", ok, detail,
               "start the server, or pass the served model name as --model")


# ---------------------------------------------------------------- disk

def check_disk(min_gb=50):
    st = os.statvfs(ROOT)
    free = st.f_bavail * st.f_frsize / 1e9
    rec("disk:free", free >= min_gb, f"{free:.0f} GB free",
        f"need >= {min_gb} GB for a full-corpus run" if free < min_gb else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="holes/math-ct-full.ids.txt")
    ap.add_argument("--endpoint", default=config.endpoint())
    ap.add_argument("--model", default=config.model())
    ap.add_argument("--fix", action="store_true", help="attempt the automatable remedies")
    # 50 GB is sized for a full arXiv corpus. A host running a small corpus can
    # lower it (machine configuration, like the endpoint); the stepper runs
    # preflight without arguments, so the environment is how it reaches here.
    ap.add_argument("--min-disk-gb", type=int,
                    default=int(os.environ.get("FUTON6_MIN_DISK_GB", "50")))
    a = ap.parse_args()
    os.environ.update(OPENAI_BASE_URL=a.endpoint, MODEL=a.model)
    os.environ.update(config.child_environment())
    print("host configuration: " + json.dumps(config.effective(), sort_keys=True))

    ids = a.ids if os.path.isabs(a.ids) else os.path.join(ROOT, a.ids)
    check_binaries(a.fix)
    check_structure_chain()
    check_gate_subprocess()
    check_python()
    check_gates()
    check_substrate()
    check_eprints(ids)
    check_vocabulary(ids)
    check_endpoint(a.endpoint, a.model)
    check_disk(a.min_disk_gb)

    w = max(len(n) for n, _, _, _ in R)
    # ok is True (pass), False (fail: refuses the run) or None (warn: printed,
    # does not refuse).
    bad = [x for x in R if x[1] is False]
    warned = [x for x in R if x[1] is None]
    print("preflight — can this run start?\n")
    for name, ok, detail, remedy in R:
        tag = "OK  " if ok else ("WARN" if ok is None else "FAIL")
        print(f"  [{tag}] {name:<{w}}  {detail}")
        if not ok and remedy:
            print(f"         -> {remedy}")
    print(f"\n{len(R) - len(bad) - len(warned)}/{len(R)} checks pass"
          + (f", {len(warned)} warning(s)" if warned else ""))
    for name, _ok, _detail, remedy in warned:
        print(f"\n  WARNING ({name}): {remedy}")
    if bad:
        print("\n  DO NOT START. Each failure above names its remedy; --fix attempts the\n"
              "  automatable ones. A setup script having been written is not evidence that\n"
              "  it was run here — that gap is why this file exists.")
    else:
        print("\n  GO — every declared dependency is present and the live chains work.")
    return len(bad)


if __name__ == "__main__":
    sys.exit(main())
