# Early Closures — Lab Note (E-ground-G)

*One note, all early closures inline with diagrams, so each can be inspected properly (in
gh gfm-preview) before moving to the next. Anchored in **patterns** (the cascade) and
**substrate-2** (the sorry) — per Joe, 2026-06-10: if all we learn is "Claude fixes easy things,"
that is not interesting. Each closure shows the M-memes-arrows three states:*

| state | what it is here | source |
|---|---|---|
| **cascade** (`:correlated`) | the real pattern-language `construct_cascade(ψ)` assembles from the Pattern Library to address the hole | `futon3a holes/labs/M-memes-arrows/cascade_construct.py` (MiniLM + posteriors) |
| **sorry** (`:open`) | the hole as a **subset of substrate-2** — the scope + its neighbours | `diffsub-scopes.json` (7071 snapshot) |
| **wiring diagram** (`:constructed`) | the runnable construction that fills the hole | the actual artifact + commit |

Track-record entries: `futon6/holes/closure-ledger.edn`.

---

# Closure 01 — `recommendation-bindings/q5`

**Hole:** `recommendation-bindings/q5-futon3-src-futon3-stack-status-clj-reads-other-consumers`
**Character:** already-constructed (snapshot mislabel) · codebase-investigation · *an "easy" hole — kept deliberately as the contrast case*
**Provenance:** `futon5a/holes/missions/M-recommendation-bindings.md:447–465` @ `78d98fd`

## Stage 1 — the cascade (real pattern-language from the Library)

`construct_cascade(ψ="what does status.clj read + who consumes it", ε=0.15)` → **26 patterns**,
**C=9.791** (T=10.643 × H=0.920). High C — but look at *what* it assembled (top 8 of 26):

```mermaid
flowchart TD
    Q["ψ — what does status.clj<br/>read + who consumes it?"]
    Q --> p1["1· devmap-coherence/ifr-f3a-piti-audit · mc 0.50"]
    p1 --> p2["2· workflow-coherence/wip-cap · mc 0.27"]
    p2 --> p3["3· enrichment/ARGUMENT · mc 0.26"]
    p3 --> p4["4· transition/f0-f4-boundary · mc 0.23"]
    p4 --> p5["5· iiching/exotype-003 · mc 0.23"]
    p5 --> p6["6· stack-coherence/futon1-storage-coherence · mc 0.23"]
    p6 --> p7["7· social/ARGUMENT · mc 0.21"]
    p7 --> p8["8· futon-stack/argument · mc 0.21"]
    p8 --> tail["… 18 more, mc 0.20 → 0.15<br/>(devmap-coherence ifr-*, p4ng/*, agency/*, storage/*)"]
    classDef q fill:#fff3cd,stroke:#d39e00;
    classDef p fill:#eef,stroke:#88a;
    class Q q;
    class p1,p2,p3,p4,p5,p6,p7,p8,tail p;
```

**↑ This chain is an embedding artifact, NOT real structure.** `construct_cascade` ranks by MiniLM
cosine only (`cascade_construct.py` lines 65/94/104) and **never reads the 2,538-edge pattern
phylogeny** (`futon6/scripts/pattern_phylogeny.py`: *descent* = cross-reference toward primitives,
*co-application* = HGT roads from co-occurrence in missions). Overlaying the phylogeny on the **same**
patterns recovers the structure it discarded — **14 descent + 10 co-application edges = a semi-lattice**:

```mermaid
flowchart TD
    INV["invariants<br/>TRUNK primitive"]:::trunk
    SC["ifr-state-convergence"]:::p
    F0["ifr-f0-sati"]:::p --> SC
    F1["ifr-f1-dhammavicaya"]:::p --> F0
    F1 --> SC
    F1 --> INV
    F2["ifr-f2-viriya"]:::p --> F1
    F7["ifr-f7-upa-upekkha"]:::p --> F2
    BH["futon-bridge-health"]:::p --> F2
    ARG["ARGUMENT"]:::p --> INV
    EX["exotype-003"]:::p --> INV
    FSC["futon1-storage-coherence"]:::p --> INV
    RC["retroactive-canonicalization"]:::p --> INV
    ARG -. "co-app 16" .- INV
    INV -. "co-app 6" .- RC
    ARG -. "co-app 6" .- RC
    FLOAT["6 of 26 patterns are NOT in the phylogeny at all<br/>(argument, f0-f4-boundary, peripheral-to-core,<br/>reflective-container, readiness-windows, platform-choreography)<br/>— embedding-selected, no combination grounding"]:::float
    classDef trunk fill:#fff3cd,stroke:#d39e00;
    classDef p fill:#eef,stroke:#88a;
    classDef float fill:#fde0e0,stroke:#cc4444,stroke-dasharray:4 3;
```
*Solid = descent (toward the `invariants` trunk; the `ifr-*` chain sati ← dhammavicaya ← viriya ←
upa-upekkha); dotted = co-application roads. This is "A City is Not a Tree" — tree + cross-links.*

**Three findings:**
1. **The embedding over-selects** — 6/26 patterns have no phylogeny node, so they cannot be part of a
   real pattern-language; they are cosine-neighbours, not combiners.
2. **The cascade machinery is defective** — `construct_cascade` ignores its own combination prior (the
   phylogeny). This *is* the semilattice-vs-linear tension (E4) the campaign already named, found in the
   wild. A phylogeny-grounded cascade would be structured by construction.
3. **C ≠ meaningfulness** — the trivial hole scored *higher* C (9.79) than the meaningful `kit-outbox`
   (7.34), with blander patterns. The "Claude-fixes-easy-things" signature, measurable.

## Stage 2 — the sorry (a real subset of substrate-2)

The hole is **not** a noted gap — it is a node in substrate-2, in the recommendation-bindings MAP
cluster (72 scopes total; the relevant neighbourhood shown). Crucially its neighbours reveal the
construction context: `map-answers` (anchored — *contains the answer*) and the source file scope
(anchored — *the subject*) sit right next to the `:detached` q5.

```mermaid
flowchart TB
    subgraph MAP["recommendation-bindings · MAP phase — substrate-2 subset"]
        q5["q5 · status.clj reads + consumers<br/>state: DETACHED ← the sorry"]:::hole
        q1["q1 · detached"]:::hole
        q4["q4 · detached"]:::hole
        q7["q7 · detached"]:::hole
        q2["q2 · anchored"]:::done
        q6["q6 · anchored"]:::done
        q8["q8 · anchored"]:::done
        ans["map-answers · anchored"]:::done
        src["source/futon3-stack-status.clj · anchored"]:::src
    end
    q1 -.sibling map-item.- q5
    q4 -.sibling.- q5
    q7 -.sibling.- q5
    q2 -.sibling.- q5
    q6 -.sibling.- q5
    q8 -.sibling.- q5
    ans == contains the Q5 answer ==> q5
    src == is the subject of ==> q5
    classDef hole fill:#ffe0e0,stroke:#cc4444;
    classDef done fill:#eafbe8,stroke:#4caf50;
    classDef src fill:#e8f4ff,stroke:#4a90d9;
```

**The snapshot-stale finding, made structural:** q5 is `:detached` (open) yet `map-answers` —
its own anchored neighbour — *already holds the answer*. The substrate snapshot mislabels a closed
hole as open. **This is why `G` cannot be grounded in the snapshot; the realized-closure signal must
be doc/commit-anchored.** (This is the single most important thing closure 01 taught us.)

## Stage 3 — the wiring diagram (the construction)

What `:constructed` actually *is*: `futon3.stack.status` is a read-only aggregator — four data
sources in, one report out, four consumer surfaces, **zero `ns`-consumers**.

```mermaid
flowchart LR
    s1["vitality/latest_scan.json"]:::src --> ST
    s2["vitality/git_summary.edn"]:::src --> ST
    s3["boundary.edn (3 search paths)"]:::src --> ST
    s4["vitality/focus_profile.edn"]:::src --> ST
    ST["futon3.stack.status<br/>(read-only aggregator)"]:::agg
    ST --> c1["futon_summary.py"]:::con
    ST --> c2["futon0 stack-hud.el"]:::con
    ST --> c3["boundary_hud.el"]:::con
    c4["futon4 boundary-generate.clj"]:::con -. "generates its OWN boundary.edn" .-> s3
    classDef src fill:#e8f4ff,stroke:#4a90d9;
    classDef agg fill:#fff3cd,stroke:#d39e00;
    classDef con fill:#eafbe8,stroke:#4caf50;
```

**Critical surprise (in the construction):** `devmap_readiness.py` serializes only per-futon
aggregates — per-prototype entries do **not** exist in `boundary.edn`, so the mission's
`:serves-spine`-on-prototypes binding needs the Python emitter extended first. *The wiring diagram
makes visible that the binding target isn't in the data yet.*

## The three states, end to end

```mermaid
stateDiagram-v2
    direction LR
    correlated --> open
    open --> constructed
    note right of correlated
        CASCADE — 26 Library patterns, C=9.791,
        but GENERIC (audit/argument). Triviality signature.
    end note
    note right of open
        SORRY — substrate-2 node q5 (:detached),
        neighbours map-answers (anchored) + source.clj.
    end note
    note right of constructed
        WIRING — read-only aggregator, 4 reads → 4 consumers,
        per-prototype serialization gap.
    end note
```

## Findings from closure 01

1. **Snapshot is stale** — `:detached` q5 already had its answer in the anchored `map-answers`
   neighbour. → ground `G` from doc/commit, not the snapshot. (Confirms + explains the E-ground-G v0
   degenerate-all-detached result.)
2. **C ≠ meaningfulness** — the trivial hole's cascade scored *higher* C with blander patterns than
   the meaningful `kit-outbox` hole. The cascade machinery discriminates topicality, not quality.

## Critique surface (Joe / Fable)

- Is the cascade *for this hole* even legitimate evidence, given the hole is an already-closed
  investigation? (It is the contrast case on purpose — the *next* closure is forward + meaningful.)
- The cascade is MiniLM-embedding-driven; the generic-pattern result may be an embedding artifact,
  not a property of the hole. Worth a BGE re-embed (cf. superpod-embeddings) before trusting cascades
  as a signal.
- Is "substrate-2 subset = the diffsub-scopes snapshot neighbourhood" the right cut, or do you want
  the live 7071 hyperedge relations (the actual arrows, not just sibling scopes)?

---

> **Next:** Closure 02 will be a **forward, meaningful** hole (`kit-outbox` or
> `structure-seed-promotion/what-the-poc-does-not-ship`) — a real cascade → sorry → wiring with an
> on-topic pattern-language, to compare against this already-done easy one.
