#!/usr/bin/env python3
"""PROOF-MINE landing PREP — the cleaning step (D7). Deterministic, read-only on the run artifact.

The mined records carry PROSE discharge targets and `discharged_by` values (the 70B describes rather
than names ids), so the raw records are NOT safe to land. This step CLEANS + VALIDATES them into
canonical, verifiable relations, keeping the model's raw text as provenance:

  landing relation  =  <canonical mission node>  --:discharged-by-->  sha/<clean-sha>
                       props: {grade, witness (verbatim), raw_target, raw_discharged_by, source}

Why key on the MISSION node (not the model's prose target): both endpoints are then VERIFIABLE — the
mission node exists in substrate-2 and the sha exists in git — and it is the exact grain §11's join uses
(discharge evidence on canonical mission nodes). The model's finer-grained-but-unverified target is
preserved as :raw-target for traceability, never minted as an entity.

Validation (a discharge lands ONLY if ALL hold; else it is REJECTED with a reason, never minted):
  - grade == :discharged                      (only a discharge claim carries a :discharged-by)
  - a clean sha (7-40 hex) is extractable from discharged_by
  - that sha EXISTS in the mission's git repo  (git cat-file — nondestructive existence check)
  - the witness is verbatim (witness_verbatim == true)
  - the mission id is canonical (<repo>-d/mission/<stem>)

Outputs (derived; the source artifact is never modified):
  proof-mine-landing.jsonl          — validated relations, ready for land_proof_mine.bb
  proof-mine-landing-rejects.jsonl  — everything filtered out, with a reason (auditable)

  python scripts/proof_mine_land_prep.py --out data/proof-mine
"""
import argparse, json, os, re, subprocess

HOME = os.path.expanduser("~")
CODE = os.path.join(HOME, "code")
SHA_RE = re.compile(r"\b([0-9a-f]{7,40})\b")


def clean_sha(s):
    """First git-sha-shaped token in a (possibly prose) discharged_by, or None."""
    m = SHA_RE.search(str(s or ""))
    return m.group(1) if m else None


def repo_of(mission):
    """'<repo>-d/mission/<stem>' -> '<repo>' (the repo to validate the sha against)."""
    return mission.split("-d/")[0] if mission and "-d/" in mission else None


def sha_exists(repo, sha, cache):
    """True iff <sha> resolves to a commit in ~/code/<repo> (cached). Nondestructive read."""
    if not repo or not sha:
        return False
    key = (repo, sha)
    if key in cache:
        return cache[key]
    repo_dir = os.path.join(CODE, repo)
    ok = False
    if os.path.isdir(os.path.join(repo_dir, ".git")):
        try:
            ok = subprocess.run(["git", "-C", repo_dir, "cat-file", "-e", "%s^{commit}" % sha],
                                capture_output=True, timeout=10).returncode == 0
        except (OSError, subprocess.SubprocessError):
            ok = False
    cache[key] = ok
    return ok


def prep(out_dir):
    recs = [json.loads(l) for l in open(os.path.join(out_dir, "proof-mine.jsonl")) if l.strip()]
    landing, rejects, cache, seen = [], [], {}, set()
    for r in recs:
        mission = r.get("mission")
        canonical = bool(mission and "-d/mission/" in str(mission))
        for d in r.get("discharges", []):
            raw_db, raw_t = d.get("discharged_by"), d.get("target")
            base = {"mission": mission, "grade": d.get("grade"),
                    "raw_target": raw_t, "raw_discharged_by": raw_db}
            if d.get("grade") != "discharged":
                continue                                    # only :discharged carries a :discharged-by
            if not canonical:
                rejects.append({**base, "reason": "mission-not-canonical"}); continue
            sha = clean_sha(raw_db)
            if not sha:
                rejects.append({**base, "reason": "no-sha-in-discharged_by"}); continue
            if not d.get("witness_verbatim"):
                rejects.append({**base, "reason": "witness-not-verbatim"}); continue
            if not sha_exists(repo_of(mission), sha, cache):
                rejects.append({**base, "reason": "sha-not-in-git"}); continue
            key = (mission, sha)
            if key in seen:                                 # idempotent: one relation per (mission, sha)
                continue
            seen.add(key)
            landing.append({"mission": mission, "sha": sha, "grade": "discharged",
                            "witness": d.get("witness"), "raw_target": raw_t,
                            "raw_discharged_by": raw_db, "source": "proof-mine"})
    with open(os.path.join(out_dir, "proof-mine-landing.jsonl"), "w") as fh:
        for x in landing:
            fh.write(json.dumps(x) + "\n")
    with open(os.path.join(out_dir, "proof-mine-landing-rejects.jsonl"), "w") as fh:
        for x in rejects:
            fh.write(json.dumps(x) + "\n")

    import collections
    by_reason = collections.Counter(x["reason"] for x in rejects)
    print("PREP (deterministic, read-only on proof-mine.jsonl):")
    print("  landing relations : %d  (unique mission→sha, git-validated, verbatim-witnessed)" % len(landing))
    print("  rejected          : %d  %s" % (len(rejects), dict(by_reason)))
    print("  -> proof-mine-landing.jsonl  +  proof-mine-landing-rejects.jsonl")
    return landing, rejects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(CODE, "futon6/data/proof-mine"))
    prep(ap.parse_args().out)


if __name__ == "__main__":
    main()
