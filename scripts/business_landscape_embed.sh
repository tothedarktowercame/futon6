#!/bin/bash
# Regenerate data/business-landscape-embed.html from the B0 business-intel
# records (futon2/holes/labs/M-digital-nomad-patterns/b0-records.edn).
# Same encoder as the mission index (miniLM, futon3a venv) -> coordinated
# spaces; PCA 2D; house dark-field style; nomad patterns as highlight lenses.
# History: built 2026-07-27 (claude-4); pipeline stages were run inline that
# day — this script is the durable form. Stages:
#   1) bb: EDN records -> /tmp/b0.json  (id/kind/sensitivity/patterns/source/text)
#   2) futon3a venv python: miniLM encode -> SVD/PCA 2D -> /tmp/b0-embedded.json
#   3) python: emit self-contained HTML (SVG scatter, kind colours, private
#      dashed rings, per-pattern toggle buttons) -> data/business-landscape-embed.html
# The exact stage code lives in the session transcript and in
# futon2/holes/M-digital-nomad-patterns.md §B-series; reassemble here when
# this becomes load-bearing (B1). Until then, run the stages from that doc.
echo "See header comments — stages 1-3; B1 will make this a real pipeline."
