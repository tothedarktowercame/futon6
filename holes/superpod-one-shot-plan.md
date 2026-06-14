# From goldens to scale: the one-shot superpod plan

Constraint (Joe, 2026-06-12): minimize further superpod iterations —
ideally one more run, with no do-overs. ("I'm tired of bothering Rob.")

## The reframe the walk forces

The fresh-extraction experiment already proved the decisive fact:
**the anatomy engine is CPU-deterministic and cheap.** 9,916 math.CT
papers re-extracted locally in 16.5 minutes with 10 workers (~10
papers/sec). Everything the golden-walk spec adds — macro-def sweep,
role-driven expression grammar, authored-layer harvest, binder chains,
decorator grammar, proof discourse — is more of the same: regex + table
lookup + small parsers. No GPU anywhere.

Consequence: **the deterministic anatomy lane never needs the superpod.**
Even arXiv-wide (~2.4M eprints) at 10/sec ≈ 3 days on the dev box, less
on a few Linode CPUs. We can iterate the engine locally as many times
as we like at zero Rob-cost.

The superpod is for exactly one thing: **the model lane** — embeddings
over decorated concepts and scope structures (BGE per the established
preference, hard negatives), corpus-scale concept identity/dedup for
the registry, and any learned classifiers (convention-paragraph
detection, definition-verb disambiguation). Run ONCE, after the
deterministic substrate has stabilized.

## The plan

### Phase L — local, iterate freely (no superpod)
1. **Build the engine** from golden-walk-ledger.md: (1) macro-def sweep
   → per-paper symbol table; (2) expression grammar over the role
   alphabet (latexml-math-roles.tsv + tex-plain-cseq.txt +
   RHS-transitive); (3) authored-layer harvest (\label/\ref/\cite[locus],
   designation verbs, respectively-zip, \stackrel justifications, proof
   discourse); (4) binder chains + decorator grammar; (5) satiety
   assignment. Acceptance: machine round-trip of the 9 golden graphs +
   the W1 sentence (W2 standard).
2. **Iterate on math.CT locally.** Instruments: coverage %,
   parse-incomplete rate, UNKNOWN-cseq rate, satiety distribution,
   binding-gradient floats. **Readiness criterion: hunger-field
   convergence** — when successive engine versions stop moving the
   satiety distribution (delta < epsilon), the spec has converged.
   Operator spot-walks (paper-anatomy.el) on samples as the qualitative
   gate.
3. **Registry v0** from the converged run: decorated-concept nodes,
   hunger field, canon feeding loop vs nLab (feed-canon.bb pattern).
   All local.
4. **Scale the deterministic lane** beyond CT on local/Linode CPU as
   appetite grows (math.* first). Still no superpod.

### Phase S — superpod, once
Inputs: the converged engine's OUTPUT (anatomy graphs + registry), not
raw TeX. Jobs: (a) BGE embeddings over decorated concepts + scope
neighborhoods (hard negatives from the registry's near-misses);
(b) corpus-scale concept-identity resolution (which decorated concepts
across papers are THE SAME concept); (c) optional learned lanes,
trained on golden + engine output.

### One-shot discipline (why this doesn't need a re-run)
1. **Nothing runs first at scale.** Every job dry-runs locally on
   math.CT; the superpod only widens a pipeline already validated
   end-to-end.
2. **Emit raw + intermediates.** Per-paper symbol tables, token
   streams, classified cseq lists, UNKNOWN lists ship with the run, so
   downstream changes re-derive from intermediates instead of
   re-extracting.
3. **Honesty bits are the patch mechanism.** parse-incomplete/UNKNOWN
   subsets can be re-processed LOCALLY by a later better parser —
   flagged residue never forces a corpus re-run.
4. **Model lane separated from deterministic lane.** Embedding/model
   improvements never invalidate extraction, and vice versa.
5. **In-flight loss/progress snapshots** (Rob's standing preference)
   so a sick run is caught early, not at the manifest.

## What changes vs the last superpod run
Last time the superpod did extraction AND models, so every detector fix
threatened a re-run. Now extraction is local-forever; the superpod
consumes stable substrate. The expected number of future superpod
iterations for this lane: **one** (plus at most a re-embed if the
registry schema changes fundamentally — guarded by converging it first).

## Amendment (Joe, voice, same day): Phase S doesn't need the superpod either

"Is that a need for a superpod, or is that something we could do on a
Linode with one GPU? Do we need eight GPUs and hundreds of CPUs and
20-hour runtimes? No."

Sizing confirms it. The model lane as specified is single-GPU work:
- Decorated-concept embeddings (BGE-class, short strings): math.CT ~10^5-10^6
  items → minutes-to-hours on one GPU; even math.*-wide occurrence-level
  (~10^8 snippets) is an overnight single-GPU job.
- Concept-identity resolution: ANN/FAISS over those vectors — one box,
  possibly CPU-only.
- Learned classifiers: small-model fine-tunes on golden + engine output.

**Phase S is renamed Phase G: one GPU Linode.** The superpod is
reserved for a future that may never arrive (arXiv-wide deep full-text
embeddings, large-model training) and is NOT on this lane's critical
path. Nobody needs to be bothered.

Why the original run needed scale and this doesn't: it did everything
at once (extraction + discourse wiring + GNN + full-corpus embeddings).
The concept-driven architecture is lighter — deterministic substrate
local, model lane reduced to short-string embeddings + small models.
The one-shot discipline still applies to Phase G verbatim (dry-run on
math.CT, intermediates shipped, lanes separated) — good discipline is
free even when the hardware is cheap.
