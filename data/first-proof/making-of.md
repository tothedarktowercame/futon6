# The Making of the First Proof

**89 commits. 1 day (and some infrastructure beforehand). 171 files. 282K lines. 10 problems. 3 agents.**

The first proof attempt email went out at 17:40 on Wednesday Feb 11.
By 01:15 the next morning — 7.5 hours later — we had 10 problem
solutions, a multi-agent verification system, two genuine theorems,
one open conjecture, and a clear map of what remains.

---

## Day 1 — Feb 8: Infrastructure (7 commits)

Futon6 didn't start as a proof engine. It started as a math dictionary.

```
5fecad0  Bootstrap futon6: PlanetMath loader, LaTeX enrichment, similarity module
9e78f7e  Wire up .tex file loading and enrichment pipeline tests
9a2a402  f6/P0: Add bb pattern tagger and initial PlanetMath tagging results
4edb135  f6/P7: StackExchange processing pipeline with GPU-ready embeddings
97a88fa  Fix SE pipeline: pipe-delimited tags, streaming two-pass QA builder
bb88789  f6/P7: Self-contained superpod batch job for SE processing
09fd370  NER kernel: 19K math terms from full PlanetMath (63 MSC repos) + SE tags
```

PlanetMath parsing, StackExchange ingestion, NER kernel with 19K math terms.
A superpod batch job for processing SE at scale. The bones of a mathematical
knowledge system.

At midnight: `cd4d1bc  Add CLAUDE.md: futonic development protocol`.
The development protocol that would govern everything that followed.

## Day 2 — Feb 9: Quiet (1 commit)

Just the CLAUDE.md. A rest day. The last one.

## Day 3 — Feb 10: Metatheory (12 commits)

The wiring diagram infrastructure that would later prove critical.

```
220e7ef  Scope metatheory: 7 categories, 27 types from 4,980 entity scan
a4a5e3e  Unified metatheory: 4 layers, 84 types for mathematical discourse
12bddef  Wiring diagram metatheory v3.0: connectives are wires, not annotations
0457a1b  CT validation: 313 PlanetMath entries + PM↔nLab wiring comparison
```

"Connectives are wires, not annotations." This decision — that logical
relationships between proof steps should be first-class typed edges in a
graph, not marginalia on a document — would determine the success of the
entire verification cycle two days later.

Also: the local topology analyzer (`eafae71`), treating definitions as
lambdas with arity. The beginning of seeing mathematical structure as
something computable.

## Day 4 — Feb 11: The Main Event (53 commits)

### Morning: Reverse morphogenesis (08:34–11:43)

```
498543a  Add --download to superpod job
14d9da1  Add --moist-run and Stage 6 (reverse morphogenesis S←Q←A)
e2c97b8  Add handoff: question-asking as reverse morphogenesis (← 象 香)
d2cbf46  Add stage6 first-50 reverse-morphogenesis results
321aeff  Add Codex stage6 comparison run and update repro notes
```

Still building the superpod pipeline. Stage 6: reverse morphogenesis,
generating questions from answers. This is the last moment before
everything pivots.

### Afternoon: The burst (16:53–18:28)

In 95 minutes, 10 research-level problems get first-draft solutions.

```
16:53  db9efc1  Problem 10 attempt
17:30  2f9d8e9  Problem 6 (Nelson, epsilon-light subsets)
17:36  7450422  Problem 4 (Spielman, root separation)
18:07  d499c1b  Problem 9 (quadrifocal tensor rank-1 detection)
18:10  cdd93eb  Problem 8 (Lagrangian smoothing, 4-valent vertices)
18:16  14da57e  Problem 7 (lattices with 2-torsion, rationally acyclic manifolds)
18:19  4fc3d8a  Problem 1 (Phi^4_3 measure equivalence under shifts)
18:20  e09bb94  Problem 3 (ASEP Markov chain / Macdonald polynomials)
18:24  64e443f  Problem 5 (O-slice connectivity via geometric fixed points)
18:28  35ad0c1  Problem 2 (universal test vector for Rankin-Selberg integrals)
```

~7 minutes per problem. Algebraic surgery theory, automorphic forms,
symplectic geometry, equivariant homotopy theory, stochastic processes,
random matrix theory. Generated at LLM speed.

These are the proofs that will be wrong. More precisely: these are the
proofs whose *structural gaps* will drive the entire verification system
into existence.

### Evening: First verification cycle (20:56–22:58)

```
20:56  fd50a54  Library research brief for Problems 3 & 5
21:00  9e2247c  Polya reductions for Problems 3 & 5
21:20  09bd2ab  Rewrite Problem 3 with lemma-based proof
21:21  09e23db  Problem 8 v2: symplectic direct sum forces Maslov index exactly 0
21:33  848b788  Proof-polish Codex scripts for Problems 3 and 5
21:48  6fea249  Tighten Problem 6 proof claims and fix wiring rigor
21:50  250a8c8  Proof-polish Codex scripts for Problems 1, 2, 4, 7, 8, 9
22:16  1ee91a7  Fix Problem 4 proof errors, add numerical verification
22:26  e70b702  Fix Codex-identified gaps in Problems 3 and 5
```

Codex enters. The first round of proof-polish scripts generates
verification prompts from the wiring diagrams — node by node, edge by
edge. Each proof step gets a focused prompt asking: "is this claim
justified? What's the evidence? What's the gap?"

P4 is special: it has a *checkable* claim (a numerical inequality).
Numerical verification finds that the original proof has errors but the
conjecture itself holds. This forces honesty.

### Late evening: The critic arrives (22:48–23:49)

```
22:48  502e0c0  Expand reviewer critique across all problem writeups
22:58  a4b76b4  Point-by-point response to reviewer critique (all 10 problems)
23:01  0003300  PROVE P4 superadditivity for n=3
23:16  6800c9c  Fix critical reviewer gaps in P2, P7, P8
23:49  6f9bf66  Fix second-round critical findings
23:57  3cc811a  Fix third-round blocking findings
```

The reviewer critique lands at 22:48. Three critical gaps in P2, P7, P8.
28 minutes later: fixes. 33 minutes after that: second-round fixes.
8 minutes later: third-round fixes.

Three rounds of critique in one hour. The Lakatos dialectic, compressed.

But also: the P4 n=3 proof is *actually proved* (`0003300`). Not
hand-waved, not cited — proved via a discovered identity
(Phi_3 * disc = 18 * a_2^2) and Cauchy-Schwarz. Symbolically verified
in SymPy. The first genuine theorem of the project.

### Midnight: Stress test and honesty (23:19–23:49)

```
23:19  b6b5585  P4 stress test: 0 genuine violations across 35K trials
23:39  2849a0f  P4 n>=4 research brief + scaling analysis
```

35,000 adversarial tests. Zero genuine violations. The adversarial
optimizer gets to within 10^{-14} of equality but can't break it.
The infimum is exactly 1.0 for all n.

P4 n>=4 is declared *open*. Not "proved modulo gaps." Not
"heuristically justified." Open. With a research brief listing what
would be needed for a proof.

This is the system working. Not because it proved everything, but
because it knew where to stop.

## Day 5 — Feb 12: Understanding what we built (16 commits)

### The retrospective

```
00:00  439aabe  Checkpoint note: verification dynamics
00:02  bf62e14  Wiring diagrams as critical infrastructure
```

The checkpoint note identifies the generation/verification asymmetry:
9 proofs in 60 minutes, fixes in 28 minutes, each at LLM speed.
The *verification* infrastructure (wiring diagrams with typed edges)
is what made the rapid fix cycles converge rather than spiral.

### The discovery

```
00:38  9efda5b  Free Stam inequality IS our inequality in the limit
```

Web research reveals that Voiculescu (1998) proved the free Stam
inequality: 1/Phi*(mu boxplus nu) >= 1/Phi*(mu) + 1/Phi*(nu). This
is *exactly* our P4 inequality for n -> infinity. We're not trying to
prove something new — we're trying to finitize a known theorem.

### The exploration

```
00:42  ed99dca  Three proof strategies + Codex research script
00:50  827f4c6  Strategy B: differentiation commutes exactly with ⊞_n
01:04  9e3dbb6  Strategy A: indefinite but superadditive
01:12  2f221ff  PAR: P4 n>=4 proof research session
```

Strategy B yields a beautiful exact identity — differentiation commutes
with finite free convolution — but the induction chain fails because
the key inequality goes the wrong direction.

Strategy A yields a surprise: 1/Phi_n is superadditive in cumulant
space *without being convex*. The Hessian is indefinite (100% of trials)
yet f(a+b) >= f(a) + f(b) always holds. The proof, if it exists, must
exploit structure beyond convexity.

### Meanwhile: the other agents

While this was happening, another Claude was grinding through P2, P7,
P8 reviewer fixes (`a5a4fbe`, `de3e2ac`, `65c6123`, `dce6411`,
`d230866`, `2607ab1`, `5d10418`). Codex was running library mining
(`b35f4e8`) and Strategy A research prompts (ongoing, timing out,
rerunning with longer timeouts).

Three agents. Different tasks. Coordinated via git and the wiring
diagram infrastructure. No collisions.

---

## What we built

### The artifacts

| Category | Count | Lines |
|----------|-------|-------|
| Problem solutions | 10 | 2,560 |
| Wiring diagrams | 10 | — |
| Verification scripts | 8 | 9,862 |
| Codex polish scripts | 10 | — |
| Research briefs | 3 | — |
| Checkpoint/PAR notes | 3 | — |

### The scorecard (final, after review)

| Problem | Status | Method |
|---------|--------|--------|
| P1 | **Proved** (writeup done) | Phi^4_3 measure theory |
| P2 | **Proved** (writeup done, 6 review rounds) | Rankin-Selberg / Kirillov model |
| P3 | **Proved** (writeup done) | ASEP / Macdonald polynomials |
| P4 n<=3 | **Proved** | Phi_3*disc identity + Cauchy-Schwarz |
| P4 n>=4 | Open (3 strategies, research ongoing) | Finite free Stam inequality |
| P5 | **Proved** (writeup done, with F_O-locality caveat) | Equivariant homotopy |
| P6 | Conditional (vertex-subset BSS adaptation unproved) | Nelson epsilon-light subsets |
| P7 | Partially reviewed (open findings remain) | Surgery theory / L-groups |
| P8 | **Proved** (writeup done, 6 review rounds) | Symplectic Lagrangian |
| P9 | **Proved** (writeup done) | Quadrifocal tensors |
| P10 | **Proved** (writeup done) | PCG reasoning |

**Final tally: 7 proved with clean writeups, 1 conditional, 1 partial, 1 open.**

### The system

What emerged wasn't planned. There's no commit that says "design a
Proofs-and-Refutations verification system." It grew from the wiring
diagram infrastructure (Day 3), the proof generation burst (Day 4
afternoon), the Codex polish scripts (Day 4 evening), and the reviewer
critique cycle (Day 4 night).

The key components:
1. **Wiring diagrams with typed edges** — proof structure as a graph,
   not a document. Edges typed as `assert`, `reference`, `derive`.
2. **Codex verification prompts** — generated from the wiring diagram,
   one per proof node. Each knows its predecessors and successors.
3. **Reviewer/responder cycle** — argumentative mode. Critic identifies
   gaps by node ID. Responder attempts fixes. Second critic catches
   confidence laundering.
4. **Numerical stress testing** — for problems with checkable claims
   (P4), computation forces honesty.
5. **Research briefs** — for problems where the proof hits a genuine
   wall, the system generates structured research handoffs with search
   terms, priority rankings, and MO/MSE references.

### What we learned

**Generation is cheap.** 10 research-level proof sketches in 95 minutes.
This is the MMCA observation from futon5: bit flipping is free, detecting
edge-of-chaos is hard.

**Verification infrastructure is everything.** Without wiring diagrams,
the fix cycle would produce "fix one gap, introduce two." With them, the
critic can say "edge p7-s4 -> p7-s3a is typed `assert` but should be
`assume`" and get a targeted response.

**Confidence laundering is the failure mode.** Each rewrite tends to
make claims stronger rather than more honest. The reviewer cycle catches
this, but only if the infrastructure supports targeted critique.

**Honesty is an outcome.** P4 n>=4 declared open. P7 flagged as "may
need retreat to conditional." These are better outcomes than claimed
proofs with hidden gaps.

**The interesting thing is superadditive without being convex.** The
Hessian of 1/Phi_n in cumulant space is indefinite, yet the inequality
holds. Whatever the proof is, it's not the obvious one. That's what
makes it a real problem.

## What the review cycle revealed

The scorecard above was preceded by an initial self-assessment (sent to a
colleague ~4 hours before the review cycle completed) that claimed 10/10
solved. The review process brought this to 7/10 proved — and exposed
several instructive failure modes.

### Confidence was anticorrelated with correctness

| Problem | Initial confidence | Actual outcome |
|---------|-------------------|----------------|
| P4 | **High** | Wrong (n>=4 unproved) |
| P9 | **High** | Had a critical bug (witness was rank-1) |
| P3 | **Low** | Easily fixed (minor irreducibility rewrite) |
| P5 | **Low** | Proved with modest caveat |

The two "High confidence" self-assessments were the worst failures. P4's
concavity argument had a fundamental gap for n>=4 that numerical testing
couldn't find (because the conjecture is true — the *proof* is what's
missing). P9's explicit witness used lambda=1 everywhere, which IS rank-1
(u=v=w=x=1) — the witness satisfied the forward direction, not the
converse. Meanwhile, the problems flagged "Low" needed only targeted fixes.

### Specific bugs the review caught

**P9 (Critical): The witness proved the wrong thing.** The converse argument
needed a non-rank-1 lambda, but lambda_{abgd}=1 factors as 1*1*1*1. Fix:
lambda=1 except lambda_{1234}=2, verified non-rank-1 by contradiction,
det(M)=-24 by direct computation.

**P6 (Major): Edge sparsification != vertex selection.** The proof cited
Batson-Spielman-Srivastava (2012) for a vertex-subset guarantee, but BSS
is an edge sparsification theorem. The "vertex-pruning variant follows"
claim was unproved. Fix: honest conditional framing.

**P3 (Major): Push cascades aren't adjacent transpositions.** The
irreducibility argument treated adjacent swaps as direct moves, but the
t-PushTASEP dynamics involves cascading displacements. Fix: vacancy-transport
argument (the vacancy absorbs cascades in one step; composing vacancy moves
reaches any permutation, 15-puzzle style).

### Problem descriptions got scrambled

The initial announcement to a colleague listed P6 as "Tensor decomposition
complexity" (it's epsilon-light subsets) and P10 as "Persistent cohomology /
Gromov-Hausdorff" with answer "No" (it's PCG for tensor CP, answer is a
constructive algorithm). The git log confirms P10 was always the PCG problem
from its first commit (`db9efc1`). The scrambled descriptions likely came
from the generating session confabulating problem summaries.

Solving the right problem matters. So does describing it correctly.

### The review paradox

Generation: 10 problems in 95 minutes. Review: ~4 hours to get 7 of them
right. The ratio is roughly 1:16 (generation:verification per proved result).
This matches the broader observation that LLM proof generation is cheap but
LLM proof *validation* requires adversarial structure — the wiring diagrams,
the typed edges, the node-level critique — to converge rather than
confidence-launder.

The three problems that didn't make it (P4, P6, P7) share a pattern: each
has a step that *sounds right* but relies on a theorem that doesn't quite
apply (P4: concavity that doesn't hold; P6: an edge theorem used for
vertices; P7: citations that don't close the gap). The review catches these
not by re-deriving the math but by checking whether each claim's *evidence
type* matches its *edge type* in the wiring diagram. When an `assert` edge
rests on a `reference` that doesn't exist, that's a structural tell.

---

## Timeline at a glance

```
Feb 8   ████                    Infrastructure (7)
Feb 9   █                       Protocol (1)
Feb 10  ████████████            Metatheory + wiring (12)
Feb 11  ████████████████████████████████████████████████████  The burst (53)
Feb 12  ████████████████        Understanding + research (16)
```

100+ commits across 5 days. From PlanetMath parser to 7 clean proofs,
1 open conjecture in finite free probability, and 2 honest gaps — with
a working multi-agent verification system as a side effect.

The system's real output isn't the proofs. It's the calibrated map of
what's proved, what's conditional, and what's open. That map didn't exist
after generation. It exists after review.

---

## Postscript: Agents as foragers (Feb 12, evening)

The proof work continued past the initial "89 commits" burst. P4 n=4
progressed through 24 scripts, 5 failed approaches, and a multi-agent
handoff to PHCpack. P6 was reduced to a single operator-averaging lemma.
P7 discharged its lattice-existence bottleneck via a rotation construction
and narrowed the S obligation to a computable surgery obstruction. The
scorecard updated to 8 proved (P4 n=4 closing), 1 conditional (P6), 1
partial (P7).

What emerged from the extended proof work — beyond the mathematical
results — was a pattern in how agents collaborate on research problems.
It connects back through the futon ecosystem in a way that wasn't planned.

### The ant parallel

In futon2, AIF ants explore a 2D grid. They leave pheromone trails —
positive ("food here") and negative ("dead end") — and other ants use
those trails to explore more efficiently. The colony's knowledge is an
emergent property of many agents exploring and signaling.

The proof journey followed the same structure at a higher level:

- **Positive pheromone**: handoff notes ("resultant elimination works for
  symmetry-stratified subspaces"), method wiring libraries ("D2's barrier
  technique transfers, but the atom structure differs"), proved reductions
  ("GPL-H with these 4 hypotheses implies the theorem").
- **Negative pheromone**: dead-end theorems ("SOS is structurally blocked
  by interior zeros"), exhaustion results ("6 trace-based techniques all
  hit the quadratic-vs-linear wall"), parametric tensions ("E2 needs even
  n but surgery wants odd n").

The negative pheromone is the less obvious but more valuable signal. When
Claude proved that Putinar certificates are structurally impossible for P4
(7 scripts → one theorem), that theorem prevented every subsequent agent
from re-attempting SOS. When the 6-technique exhaustion was proved for P6,
it eliminated an entire technique class. The dead-end theorems compressed
hours of exploration into transmissible facts.

### From verification to exploration

The initial agent coordination pattern was **verification dispatch**: Claude
writes a proof, Codex confirms it. This is useful but asymmetric — it
treats Codex as a reviewer, not an explorer. The return is 1 bit per
dispatch: confirmed or not.

The more productive pattern that emerged over the extended proof work is
**student dispatch** (see `futon3/library/agent/student-dispatch.flexiarg`):
give the agent the problem context, the dead ends with reasons, candidate
directions, and a structured report format. Let them discover rather than
confirm.

The P4 PHCpack run was the clearest example: Codex wasn't told "verify that
there are 12 critical points." It was told "find all critical points of this
system using homotopy continuation." It returned 12 CPs with full case
classification, discovered the mixed_volume=0 issue independently, and
provided results that updated the proof status in ways Claude hadn't
predicted. The exploration dispatch produced richer results per unit effort
than any verification dispatch.

### Connection to futon3c

The agent teaming infrastructure being built in futon3c — forum-based
coordination with detach/reattach, PAR checkpoints, and the ping-pong
protocol — now has a concrete use case beyond code coordination. The proof
journey shows what multi-agent research teaming looks like:

1. **Decompose** the problem into cases or directions (split-into-cases,
   hypothetical-proof-architecture)
2. **Dispatch** each direction to an agent with negative-knowledge transfer
   (student-dispatch)
3. **Collect** structured reports (not pass/fail but findings + surprises)
4. **Synthesize** results, update the dead-end list, re-dispatch

This is the AIF cycle — observe, orient, decide, act — applied to
research. The "orient" step is where the pheromone trail matters: the
accumulated dead-end theorems and technique-landscape maps are the
colony's shared knowledge base.

The futon5 proof bridge (`futon5-proof-bridge.md`) takes this further:
if argument structures can be embedded as tensors, then "finding a
relevant prior result" becomes a nearest-neighbor search in embedding
space — the computational equivalent of an ant following a pheromone
gradient. The gradient points toward argument shapes that have resolved
similar gaps before.

### The reflexive observation

The proof attempt set out to prove 10 mathematics problems. Along the
way it also demonstrated something about agent teaming: that multi-agent
research collaboration is structurally analogous to foraging, that
negative knowledge transfer is the key efficiency mechanism, and that
the infrastructure for representing and sharing structured findings
(wiring diagrams, typed edges, handoff notes) is the equivalent of the
pheromone trail system.

The ants in futon2 don't know they're doing category theory. The agents
in this proof attempt didn't know they were doing ant colony optimization.
But the patterns are isomorphic — and recognizing the isomorphism suggests
that the futon ecosystem's architecture (futon2's AIF agents, futon3's
pattern library, futon3c's coordination layer, futon5's tensor math,
futon6's knowledge base) may be more convergent than it appeared.
