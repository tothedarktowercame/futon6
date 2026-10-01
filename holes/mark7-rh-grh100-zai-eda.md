# Mark7 rh-grh top-100: Z.ai EDA findings

Read-only analysis by `zai-4` of the retrieved
`mark7rhgrh100-20260928` run. The full working report was delivered through
Agency; this note records the findings that affect the next run.

## Same-cost WEFT opportunities

1. **Reallocate the fixed S4 budget.** S4 selected 2,750 of 99,458 regions.
   Region-rich papers consequently have very low coverage; correlation between
   log region count and expository coverage was -0.856. Allocate the same 2,750
   calls with a corpus-budgeted, size-scaled cap instead of 30 per paper.
2. **Reject impossible inputs before a model call.** S3 had 266 rejections,
   dominated by missing clause spans and cyclic proposed step graphs. Both are
   deterministically detectable; saved calls can be reassigned to deferred
   expository regions.
3. **Recover already-paid completions.** Invalid JSON control characters,
   timeouts and truncation account for most S3/S4 errors. Sanitize completed JSON
   locally; route timeout/truncation retries to smaller prompt windows rather
   than repeating the same request.
4. **Repair vocabulary collapse offline.** One canonical warrant absorbed 7,365
   of 17,326 slugs (42.5%), while 5,566 of 6,799 canonicals are singletons.
   Conservative exact/lemma grouping with a bounded cluster radius should make
   recurring gaps usable without any model calls.
5. **Route by extraction quality, not markup volume.** Comprehension correlated
   strongly with rung-3 strategy score (0.90), modestly with quote agreement
   (0.36), and essentially not at all with S1 markup coverage (-0.02). Early
   quote agreement, clause-span count and thin-move density are better candidates
   for distributing a fixed call budget.

Other deterministic opportunities are noun backfill for 2,450 proofs with no
noun score and local span-search proposals for the 37,700 nodes classified as
`unclear`. These should be tested offline against this run before changing the
next production policy.
