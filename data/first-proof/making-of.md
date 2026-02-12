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

### The scorecard

| Problem | Status | Method |
|---------|--------|--------|
| P1 | In review cycle | Phi^4_3 measure theory |
| P2 | In review cycle (~2 rounds) | Rankin-Selberg / Kirillov model |
| P3 | In review cycle | ASEP / Macdonald polynomials |
| P4 n=2 | **PROVED** | Equality (disc linear in coefficients) |
| P4 n=3 | **PROVED** | Phi_3*disc identity + Cauchy-Schwarz |
| P4 n>=4 | Open (3 strategies, research ongoing) | Finite free Stam inequality |
| P5 | In review cycle | Equivariant homotopy |
| P6 | In review cycle | Nelson epsilon-light subsets |
| P7 | In review cycle (may need retreat) | Surgery theory / L-groups |
| P8 | In review cycle (~1 round) | Symplectic Lagrangian |
| P9 | In review cycle | Quadrifocal tensors |
| P10 | In review cycle | PCG reasoning |

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

---

## Timeline at a glance

```
Feb 8   ████                    Infrastructure (7)
Feb 9   █                       Protocol (1)
Feb 10  ████████████            Metatheory + wiring (12)
Feb 11  ████████████████████████████████████████████████████  The burst (53)
Feb 12  ████████████████        Understanding + research (16)
```

89 commits in 4 days. From PlanetMath parser to open conjecture in
finite free probability, with a working multi-agent verification system
as a side effect.
