# Hypergraph Showcase Pack

Generated: `2026-02-20T10:37:09.620272+00:00`

## Why These Cases
These examples highlight cross-scope reuse, cross-post wiring, and symbolic binders as first-class scope records feeding typed hypergraphs.

## Case 1: In the category of sets epimorphisms are surjective - Constructive Proof?

- Source: `mathoverflow.net__category-theory.jsonl`
- Thread: `mathoverflow.net:178778`
- URL: https://mathoverflow.net/questions/178778
- Stats: scopes=44, expressions=95, categorical=5, shared_q_to_a_terms=5
- Top scope types: `constrain/such-that`:21, `quant/universal`:10, `quant/existential`:6, `bind/let`:3, `bind/big-union`:1, `env/theorem`:1
- Reused scope symbols: `a`x5, `a \in A`x2, `b \in B`x2, `h : A \to B`x2, `y`x2
- Shared term-node IDs (Q↔A bridge): `term:Category`, `term:CategoryOfSets`, `term:MultivaluedFunction`, `term:Polygon`, `term:RingHomomorphism`
- Question excerpt:
```text
The statement that surjective maps are epimorphisms in the category of sets can be shown in a constructive way. What about the inverse? Is it possible to show that every epimorphism in the category of sets is surjective without reverting to a proof by contradiction / negation?
```

## Case 2: Are pointed CW complexes for which the Yoneda embedding restricted to finite CW complexes reflects isomorphisms forced to be connected?

- Source: `math.stackexchange.com__category-theory.jsonl`
- Thread: `math.stackexchange.com:3990837`
- URL: https://math.stackexchange.com/questions/3990837
- Stats: scopes=4, expressions=31, categorical=5, shared_q_to_a_terms=14
- Top scope types: `bind/let`:3, `quant/universal`:1
- Shared term-node IDs (Q↔A bridge): `term:283`, `term:Category`, `term:ConnectedPoset`, `term:ConnectedSpace`, `term:CoordinateVector`, `term:Countable`, `term:ExtendedRealNumbers`, `term:Frame`
- Question excerpt:
```text
I'm currently reading Edgar H. Brown's paper, Abstract Homotopy Theory , which proves a categorical version of the Brown Representability Theorem. I've come across a statement that I don't really how to verify (if it is in fact true). First the notation and definitions. Let $\mat
```

## Case 3: is localization of category of categories equivalent to |Cat|

- Source: `mathoverflow.net__category-theory.jsonl`
- Thread: `mathoverflow.net:10010`
- URL: https://mathoverflow.net/questions/10010
- Stats: scopes=6, expressions=36, categorical=17, shared_q_to_a_terms=14
- Top scope types: `quant/universal`:2, `constrain/such-that`:2, `assume/consider`:1, `bind/let`:1
- Shared term-node IDs (Q↔A bridge): `term:Adjoint`, `term:Category`, `term:CategoryOfSmallCategories`, `term:Class`, `term:ConcreteCategory`, `term:EquivalenceRelation`, `term:Functor`, `term:GroupAction`
- Question excerpt:
```text
It might be a stupid question. Suppose There is a category of categories,denoted by CAT,where objects are categories, morpshims are functors between categories Take multiplicative system S={category equivalences}. Then we take localization at S. Then we get localized category S^(
```

## Case 4: What is the symbol of a differential operator?

- Source: `mathoverflow.net__mathematical-physics.jsonl`
- Thread: `mathoverflow.net:3477`
- URL: https://mathoverflow.net/questions/3477
- Stats: scopes=7, expressions=75, categorical=1, shared_q_to_a_terms=37
- Top scope types: `bind/summation`:5, `constrain/where`:1, `bind/let`:1
- Reused scope symbols: `a`x2
- Shared term-node IDs (Q↔A bridge): `term:16021`, `term:2727`, `term:283`, `term:Algebra`, `term:Algebras`, `term:Canonical`, `term:ChainRule`, `term:ChangeOfBasis`
- Question excerpt:
```text
I find Wikipedia's discussion of symbols of differential operators a bit impenetrable, and Google doesn't seem to turn up useful links, so I'm hoping someone can point me to a more pedantic discussion. Background I think I understand the basic idea on $\mathbb{R}^n$ , so for read
```

## Case 5: Limit of a double integral

- Source: `mathoverflow.net__mathematical-physics.jsonl`
- Thread: `mathoverflow.net:140120`
- URL: https://mathoverflow.net/questions/140120
- Stats: scopes=29, expressions=63, categorical=1, shared_q_to_a_terms=9
- Top scope types: `bind/integral`:23, `bind/summation`:4, `quant/universal`:2
- Reused scope symbols: `q`x7, `n`x3, `t`x2
- Shared term-node IDs (Q↔A bridge): `term:313`, `term:EvenNumber`, `term:FirstOrderLanguage`, `term:FixedPointsOfNormalFunctions`, `term:HadamardMatrix`, `term:Integral`, `term:LimitOfRealNumberSequence`, `term:PolynomialRing`
- Question excerpt:
```text
What is the $\varepsilon\to 0$ limit of the following double integral $$\int\limits_{-1}^1d\tau\;\sqrt{1-\tau^2}\;\tau\int\limits_0^\infty dq\;q^2e^{iq(\tau+i\varepsilon)}\;?$$ I was asked about this integral by my friend who got it in a physics research project. In fact this is 
```

## ArXiv TeX Case
- arXiv ID: `0705.0462`
- Title: Resource modalities in game semantics
- URL: https://arxiv.org/abs/0705.0462
- TeX member: `resource_modalities.tex` in `0705.0462.tar.gz`
- Signals: environment=`lemma`, binder=`forall`
- Detected scope types: `constrain/such-that`:4, `constrain/where`:2, `assume/explicit`:1, `env/lemma`:1
- Snippet:
```tex
\ldots \xrightarrow{m_{k-1}} x_{k-1} \xrightarrow{m_k} x_k \end{equation} % Two paths are parallel when they have the same initial and final positions. % A play~(\ref{equation/play}) is \emph{alternating} when: \mathin \forall i \in \{1 , \ldots , k-1\}, \quad \quad \lambda_A(m_{i+1}) = - \lambda_A(m_i). \mathout % \paragraph{{\bf Strategy.}} % A strategy $\sigma$ of a Conway game is defined as a set of alternating plays of even length such that: % \begin{itemize} \sitem $\sigma$~contains the empty play, \sitem every nonempty play starts with an Opponent move, \sitem $\sigma$ is closed by even-length prefix: for every play~$s$, and for all moves~$m,n$, $$ s \cdot m \cdot n \in \sigma \Implie
```
