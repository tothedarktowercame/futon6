#!/usr/bin/env bash
# CLean demo pipeline — gate → embed → export → demo.
# The proof-side analogue of Rob's Lean→neo4j+pgvector indexing, run locally on
# the small APM CLean collection. See holes/clean/NEO4J-PGVECTOR-MAPPING.md.
#
#   bash scripts/clean_pipeline.sh
#
# Re-run after editing any holes/clean/*.clean.edn.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "== 1/4 gate (clean_argcheck.bb) =="
bb scripts/clean_argcheck.bb holes/clean/

echo; echo "== 2/4 embed (structure + text) =="
$PY scripts/clean_structure_embed.py

echo; echo "== 3/4 export (neo4j cypher + pgvector sql) =="
$PY scripts/clean_graph_export.py

echo; echo "== 4/4 demo html =="
$PY scripts/build_clean_demo.py

echo; echo "open: data/showcases/clean-demo/index.html"
