# Pre-cohort fold-turn quarantine

These records were removed from the live `data/fold-turns/` escrow before the
WM full-loop cohort was activated. On 2026-07-13, the strict fold-turn loader
rejected every record in this directory because its prompt pin was absent or
no longer reconstructable, or because its EDN was unreadable.

The records are retained as historical evidence. They are not eligible for
live replay and must be re-authored under the current contract rather than
repinned in place. The live directory passed with 18 accepted deposits and
zero rejected deposits after this quarantine was made.
