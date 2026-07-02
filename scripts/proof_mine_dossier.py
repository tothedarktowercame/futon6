#!/usr/bin/env python3
"""PROOF-MINE dossier assembly (one UNIT OF WORK's input) — per proof-mine-runner-spec.md.

For ONE mission, assemble the dossier that the single vLLM pass reasons over:
  mission doc (HEAD/status + body) + commits citing it + code endpoints (edits/calls)
  + live XTDB counts + its c-entries and their :outcome-refs.

This is the *full grain* the spec asks for; `mission_dossier.py` is only a value-demo
(mine × live-clock) and is left intact. Everything here is CPU-local and DEFENSIVE:
git / XTDB failures degrade to an explicit note, never an exception — the dossier is an
input to a paid GPU pass, so a missing side-channel must cost a *section*, not the run.

D10 (cost accounting): the dossier is truncated to a token budget by SECTION PRIORITY
(doc HEAD/status → cited-commit subjects → c-entries → endpoints) and every truncation is
LOGGED into `dossier["truncations"]` — no silent caps.

Usage:
  proof_mine_dossier.py M-autoclock-in            # human-readable
  proof_mine_dossier.py M-autoclock-in --json     # the dict the runner consumes
"""
import json, sys, os, re, glob, subprocess, urllib.request, urllib.parse, argparse

HOME = os.path.expanduser("~")
CODE = os.path.join(HOME, "code")
FUTON1A = os.environ.get("FUTON1A", "http://localhost:7071")
# c-entry sources: the sim overlay (canonical :c-entry shape) + the mined joint records.
C_OVERLAY = os.path.join(CODE, "futon6/data/c-vector/c-store-overlay.edn")
C_MINED = os.path.join(CODE, "futon6/data/c-vector/c-entries.openai.json")

# Rough token estimate: ~4 chars/token (English + code). Only used for the D10 budget.
CHARS_PER_TOKEN = 4


def est_tokens(s):
    return (len(s) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def mission_stem(s):
    """Normalize any mission ref to its bare stem (mirror promote_c_entries.bb mission-stem).
    'mission/M-x' / 'M-x' / '<repo>-d/mission/x' / 'x' -> 'x'."""
    if not s:
        return None
    s = str(s)
    s = re.sub(r"^.*?/mission/", "", s)   # <repo>-d/mission/x  or  mission/x
    s = re.sub(r"^(mission/)?M-", "", s)  # mission/M-x  or  M-x
    return s


def find_mission_doc(stem, repos_root=CODE):
    """Locate the mission doc on disk; return (canonical_id, doc_path) or (None, None).
    canonical id = '<repo>-d/mission/<stem>' (claude-2's contract). Excludes *-desktop-save."""
    for pat in ("%s/*/holes/missions/M-%s.md", "%s/*/holes/M-%s.md",
                "%s/*/holes/missions/%s.md", "%s/*/holes/%s.md"):
        for path in sorted(glob.glob(pat % (repos_root, stem))):
            if "desktop-save" in path:
                continue
            parts = path.split(os.sep)
            try:
                repo = parts[parts.index(os.path.basename(repos_root)) + 1] \
                    if os.path.basename(repos_root) in parts else parts[parts.index("code") + 1]
            except (ValueError, IndexError):
                repo = parts[-4] if len(parts) >= 4 else "unknown"
            return "%s-d/mission/%s" % (repo, stem), path
    return None, None


def doc_head_and_status(doc_path, head_lines=40):
    """The HEAD (title + first block) and any status/maturity line — highest-priority section."""
    try:
        with open(doc_path, errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError as e:
        return "", "", "(doc unreadable: %s)" % e
    head = "\n".join(lines[:head_lines])
    status = ""
    for ln in lines[:120]:
        if re.search(r"\b(status|maturity|stage|phase)\b\s*[:=]", ln, re.I) or "**Status" in ln:
            status = ln.strip()
            break
    body = "\n".join(lines[head_lines:])
    return head, status, body


def citing_commits(stem, doc_path, limit=40):
    """Commits whose message cites the mission stem, in the doc's repo. Defensive:
    returns ([], reason) if not a git repo / git missing / none found."""
    repo_dir = doc_path
    for _ in range(6):
        repo_dir = os.path.dirname(repo_dir)
        if os.path.isdir(os.path.join(repo_dir, ".git")):
            break
    else:
        return [], "no-git-repo"
    for needle in (stem, "M-" + stem):
        try:
            out = subprocess.run(
                ["git", "-C", repo_dir, "log", "--grep", needle, "-i",
                 "--pretty=%h %s", "-n", str(limit)],
                capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError) as e:
            return [], "git-error: %s" % e
        rows = [ln for ln in out.stdout.splitlines() if ln.strip()]
        if rows:
            return rows, None
    return [], "no-citing-commits"


def _read_overlay_c_entries(stem):
    """c-entries from the sim overlay whose outcome-ref/derived-from names this mission stem."""
    try:
        import ast
        txt = open(C_OVERLAY, errors="replace").read()
    except OSError:
        return []
    hits = []
    # Cheap structural scrape (the overlay is large EDN; avoid a full EDN parser dependency).
    # Each entity block carries :c-entry/outcome-ref {... :mission "M-x"} or :derived-from "mission/M-x".
    for m in re.finditer(r"\{:id\s+\"(scope/c-entry/[^\"]+)\".*?\}\}", txt, re.S):
        block = m.group(0)
        refs = re.findall(r':(?:mission|derived-from|referent)\s+"([^"]+)"', block)
        if any(mission_stem(r) == stem for r in refs):
            name = re.search(r':name\s+"([^"]+)"', block)
            flavour = re.search(r':c-entry/flavour\s+:(\w+)', block)
            status = re.search(r':c-entry/status\s+:(\w+)', block)
            oref = re.search(r':c-entry/outcome-ref\s+(\{[^}]*\})', block)
            hits.append({
                "id": m.group(1),
                "name": name.group(1) if name else None,
                "flavour": flavour.group(1) if flavour else None,
                "status": status.group(1) if status else None,
                "outcome_ref": oref.group(1) if oref else None,
                "source": "c-store-overlay",
            })
    return hits


def _read_mined_c_entries(stem, limit=40):
    """c-entries from the mined joint records (c_mine_joint output) that reference this mission."""
    try:
        data = json.load(open(C_MINED))
    except (OSError, ValueError):
        return []
    hits = []
    for rec in data:
        oref = rec.get("outcome_ref") or {}
        ref = oref.get("referent") or oref.get("mission")
        if ref and mission_stem(ref) == stem:
            hits.append({
                "id": rec.get("id"),
                "flavour": rec.get("flavour"),
                "status": rec.get("status"),
                "outcome_ref": oref,
                "preferred": rec.get("preferred"),
                "source": "c-mine",
            })
            if len(hits) >= limit:
                break
    return hits


def xtdb_endpoints_and_counts(canonical_id):
    """Live XTDB (:7071) read-only: code endpoints via edits/calls hyperedges on the mission
    node + a coarse count census. DEFENSIVE — any failure returns ([], {}, reason)."""
    if not canonical_id:
        return [], {}, "no-canonical-id"
    url = "%s/api/alpha/hyperedges?end=%s" % (FUTON1A, urllib.parse.quote(canonical_id))
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/edn"})
        body = urllib.request.urlopen(req, timeout=5).read().decode()
    except Exception as e:                       # noqa: BLE001 — offline is the common case
        return [], {}, "xtdb-unreachable: %s" % e
    endpoints = sorted(set(re.findall(r'"((?:code|artifact|var|fn)[:/][^"]+)"', body)))
    counts = {
        "hyperedges_mentioning": body.count("{:") ,
        "edits": len(re.findall(r'"edits?"', body)),
        "calls": len(re.findall(r'"calls?"', body)),
    }
    return endpoints, counts, None


def _budget_sections(sections, budget_tokens):
    """Truncate an ordered list of (name, text, priority) to fit budget_tokens.
    Higher-priority sections are kept whole; lower ones are trimmed or dropped. Returns
    (kept:list[(name,text)], truncations:list[str]). Logs every trim — no silent caps (D10)."""
    kept, truncations, spent = [], [], 0
    for name, text, _prio in sorted(sections, key=lambda s: -s[2]):
        tks = est_tokens(text)
        if spent + tks <= budget_tokens:
            kept.append((name, text))
            spent += tks
            continue
        room = budget_tokens - spent
        if room <= 0:
            # Drop the section entirely from the prompt text (logged, never silent) — a placeholder
            # would itself consume budget and defeat the cap.
            truncations.append("DROPPED section '%s' (%d tok, budget exhausted at %d/%d)"
                               % (name, tks, spent, budget_tokens))
            continue
        cut = text[: room * CHARS_PER_TOKEN]
        truncations.append("TRUNCATED section '%s' from %d to ~%d tok (budget %d)"
                           % (name, tks, room, budget_tokens))
        kept.append((name, cut + "\n…[truncated for budget]"))
        spent = budget_tokens
    return kept, truncations


def assemble(stem, repos_root=CODE, budget_tokens=12000):
    """Assemble ONE mission's dossier. Returns a dict; never raises for missing side-channels.
    stem may be given as 'M-x', 'mission/M-x', a canonical id, or a bare 'x'."""
    stem = mission_stem(stem)
    canonical, doc_path = find_mission_doc(stem, repos_root)
    notes = []
    if not doc_path:
        return {"mission": None, "stem": stem, "doc_found": False,
                "truncations": [], "notes": ["MISSION DOC NOT FOUND for stem '%s'" % stem],
                "commits": [], "c_entries": [], "endpoints": [], "xtdb_counts": {},
                "no_code_trail": True, "text": ""}

    head, status, body = doc_head_and_status(doc_path)
    commits, creason = citing_commits(stem, doc_path)
    if creason:
        notes.append("commits: " + creason)
    c_entries = _read_overlay_c_entries(stem) + _read_mined_c_entries(stem)
    endpoints, counts, xreason = xtdb_endpoints_and_counts(canonical)
    if xreason:
        notes.append("xtdb: " + xreason)

    no_code_trail = not commits and not endpoints

    # Build the prompt-facing text under the D10 budget, by section priority.
    commit_txt = "\n".join(commits) if commits else "(no citing commits — %s)" % (creason or "none")
    ce_txt = "\n".join(
        "- [%s/%s] %s  outcome-ref=%s" % (c.get("flavour"), c.get("status"),
                                          c.get("name") or c.get("id"), c.get("outcome_ref"))
        for c in c_entries) or "(no c-entries attached)"
    ep_txt = "\n".join(endpoints) or "(no code endpoints from XTDB)"
    sections = [
        ("DOC HEAD",  "# MISSION %s  (stem %s)\n%s" % (canonical, stem, head), 100),
        ("STATUS",    status or "(no explicit status line)", 95),
        ("COMMITS",   commit_txt, 80),
        ("C-ENTRIES", ce_txt, 70),
        ("ENDPOINTS", ep_txt, 60),
        ("DOC BODY",  body, 50),
    ]
    kept, truncations = _budget_sections(sections, budget_tokens)
    text = "\n\n".join("== %s ==\n%s" % (name, txt) for name, txt in kept)

    return {
        "mission": canonical, "stem": stem, "doc_found": True, "doc_path": doc_path,
        "status": status, "commits": commits, "c_entries": c_entries,
        "endpoints": endpoints, "xtdb_counts": counts,
        "no_code_trail": no_code_trail,
        "truncations": truncations, "notes": notes,
        "text": text,
    }


def main():
    ap = argparse.ArgumentParser(description="Assemble one mission's PROOF-MINE dossier.")
    ap.add_argument("mission", help="mission stem or ref (e.g. M-autoclock-in)")
    ap.add_argument("--budget", type=int, default=12000, help="token budget (D10; default 12k)")
    ap.add_argument("--json", action="store_true", help="emit the dossier dict as JSON")
    a = ap.parse_args()
    d = assemble(a.mission, budget_tokens=a.budget)
    if a.json:
        print(json.dumps(d, indent=2, default=str))
        return
    print("=" * 72)
    print("PROOF-MINE DOSSIER — %s" % (d["mission"] or ("(NOT FOUND) " + d["stem"])))
    print("=" * 72)
    print("doc: %s" % d.get("doc_path", "(none)"))
    print("commits: %d · c-entries: %d · endpoints: %d · no-code-trail: %s"
          % (len(d["commits"]), len(d["c_entries"]), len(d["endpoints"]), d["no_code_trail"]))
    if d["notes"]:
        print("notes: " + " | ".join(d["notes"]))
    if d["truncations"]:
        print("truncations (D10): " + " | ".join(d["truncations"]))
    print("-" * 72)
    print("dossier text (~%d tok):" % est_tokens(d["text"]))
    print(d["text"])


if __name__ == "__main__":
    main()
