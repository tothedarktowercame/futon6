# Excursion: E-cold-storage — a reliable non-git storage strategy

**Date:** 2026-06-11 (stub — parked by Joe; "we'll sort this out later")
**Type:** E-prefix excursion. **Status:** :parked — note-taking only, no build.

## The tension

Data keeps ending up in git because there is nowhere better for it to go.
The house rule is *data/ is never committed*, but the rule is not enforceable
by convention alone:

- futon6 already has **tracked** files under `data/` (six
  `data/mission-scope-trees/*.json` show as modified forever; also
  `data/curriculum-proposals.edn`) — committed before the rule, now stuck in
  status output unless `git rm --cached` + a gitignore entry lands.
- Large artifacts accumulate beside the repos: `futon6/data/` is **798 MB**
  (incl. the 432 MB `nlab-wiring/pages.json` regenerated today);
  `~/code/storage/` is **195 GB** and is itself the previous round of this
  same fix (commit `338a2fa` "Move data/* and se-data outputs to
  ~/code/storage/") — a convention, not a strategy: no manifest, no
  integrity checks, no backup story, lives on the same disk as everything.

## What a strategy needs (first cut)

1. One blessed location per artifact class (regenerable extraction outputs
   vs irreplaceable corpora vs receipts), with regenerable ones documented by
   their generating command instead of backed up.
2. A manifest (what lives where, generated-by, date, size, checksum) so cold
   artifacts stay findable — the Skolem rule applies to storage too: an
   artifact nothing points to is suspect.
3. Physical media: Joe has a **2 TB thumbdrive** (not on hand 2026-06-11) —
   candidate cold tier for the irreplaceable class.
4. Repo hygiene to make the rule mechanical: gitignore `data/` wholesale +
   untrack the legacy exceptions; also `.clj-kondo/`, `.lsp/` caches.

## Disposition

Parked until Joe has the thumbdrive + an hour. Committed data isn't the end
of the world; the note exists so the fix is designed once, not improvised
per-repo.
