# StackExchange Thread Samples

Deterministic thread extracts from local StackExchange dump files.

## Files

- `math.stackexchange.com__category-theory.jsonl` (50 threads)
- `math.stackexchange.com__mathematical-physics.jsonl` (50 threads)
- `mathoverflow.net__category-theory.jsonl` (50 threads)
- `mathoverflow.net__mathematical-physics.jsonl` (50 threads)
- `manifest.json` (metadata and counts)

Each JSONL row contains one full thread with:

- question metadata + body (HTML and text)
- all answers on the question
- comments on the question and on each answer

## Source

Generated from:

- `se-data/math.stackexchange.com/Posts.xml`
- `se-data/math.stackexchange.com/Comments.xml`
- `se-data/mathoverflow.net/Posts.xml`
- `se-data/mathoverflow.net/Comments.xml`

## Topic Tags

- Math.StackExchange:
  - category theory: `category-theory`
  - mathematical physics: `mathematical-physics`
- MathOverflow:
  - category theory: `ct.category-theory`
  - mathematical physics: `mp.mathematical-physics`

## Selection Rules

- Question must carry the topic tag.
- Question score must be `>= 0`.
- Question must have at least one answer present in `Posts.xml`.
- Final selection is deterministic via SHA-256 ranking of:
  `seed|site|topic|question-id`.

Current seed: `futon6-se-sample-v1`.

## Rebuild

```bash
./scripts/extract_se_threads.py \
  --se-root /home/joe/code/futon6/se-data \
  --output-dir data/stackexchange-samples \
  --sample-size 50 \
  --min-question-score 0 \
  --seed futon6-se-sample-v1 \
  --overwrite
```

## License Note

StackExchange network content is under CC BY-SA; preserve attribution and
license obligations in downstream use.
