# mark3 eval harness summary

Run: `/home/joe/code/futon6/data/iatc-argument-graphs/loop-run-70b`
Kind: `iatc` (9 EDN artifact(s))
Papers referenced: 9

## Metrics

- grounding-%: 21.43% (6 / 28 resolved warrant edges; computable=True)
- expository-coverage-%: 0.59% (43 / 7296 lines; computable=True)
- checker-PASS-%: 100.00% (9 / 9 structural items)
- substance-PASS-%: 100.00% (9 / 9 items)
- prior-vs-posterior: 12.70% (8 / 63 posterior terms in prior)

## Checker commands

- concept: skipped (run kind is iatc, not concept)
- iatc: exit=0 pass=9/9 command=`bb /home/joe/code/futon6/scripts/iatc_argcheck.bb /home/joe/code/futon6/data/iatc-argument-graphs/loop-run-70b`
- substance: exit=0 pass=9/9 command=`/home/joe/code/futon6/.venv/bin/python /home/joe/code/futon6/scripts/substance_gate.py /home/joe/code/futon6/data/iatc-argument-graphs/loop-run-70b --kind iatc`
