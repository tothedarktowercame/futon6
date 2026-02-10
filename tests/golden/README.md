# Golden-50 Test Suite

50 physics.SE QA pairs with golden annotations for pipeline validation.

## Purpose

Ship with the superpod job. Run these 50 first as a smoke test (< 2 min)
before committing to the full 200K+ pair run (2-4 hours).

## Contents

- `golden-50.json` — 50 entries with golden NER terms, pattern tags, scope records
- Stratified: 17 easy (score >= 20), 17 medium (5-19), 16 hard (1-4)

## Usage

```bash
# After running superpod-job.py on the golden subset:
python scripts/validate-golden.py ./golden-output/

# Self-check (validates annotations internally):
python scripts/validate-golden.py --self-check
```

## Thresholds

| Check | Threshold | What it measures |
|-------|-----------|------------------|
| Entity coverage | 90% | Pipeline parsed most golden entities |
| NER recall | 50% | Pipeline finds golden NER terms |
| NER precision | 30% | Pipeline NER hits are real terms |
| Pattern agreement | 40% | Jaccard similarity with golden patterns |
| Scope recall | 30% | Pipeline finds golden scope openers |

Thresholds are deliberately conservative — the golden annotations use
classical hotword/regex methods, while the superpod LLM stage should
do better on pattern tagging.

## Generated

```
python scripts/build-golden-50.py
```

Seed: 42 (reproducible).
