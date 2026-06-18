# IATC alternative-phrasing harvest from MO/math.SE (2026-06-17)

LLM (claude-6) pass over a 200-thread MathOverflow / math.SE sample
(`futon5/data/stackexchange-samples/*.jsonl`; ~110 sampled comments + answer openings). For each
IATC `perf/value/meta` category, the **alternative phrasings** the regex cues miss, each with a
verbatim MO/SE example. Feeds (a) the cue lexicon in `scripts/iatc_alignment_passA.py` (recall),
(b) the §3b Tier-2 exemplar bank, and (c) Pass B. This is the hand-LLM pass over the *sample*;
scaling to the full corpus is the agent-pool job.

Qualitative finding from reading the raw comments: the MO/SE **comment layer is densely
dialogical** — agreement, challenge, retraction, praise, and plausibility hedging on nearly
every thread — phrased informally. This is exactly the material stripped from published prose,
and the formal cues catch almost none of it.

## Performatives

### Agree — *dialogical (stripped on publication)*
Alt phrasings: "you're (totally/probably) right" · "good point" · "Nice!" · "X's answer is (of
course) correct" · "Yes, this is what I mean" · "that works".
- *"@AnginaSeng, you're totally right, I was just being stupid."*
- *"@MaximeRamzi Ah, good point."* · *"Nice! I think that works."* · *"Willie's answer is of course correct."*
- proposed cues: `you'?re (totally |probably )?right`, `\bgood (point|catch)\b`, `\bthat works\b`, `answer is (of course )?correct`

### Challenge — *dialogical (stripped)*
Alt phrasings: "there is something wrong with" · "but you [reversed/…]" · "this is (really)
unclear" · "Isn't it …?" · "What if …?" · "seem to be in contradiction" · "did you mean …?".
- *"There is something wrong with the definition, as w(t) cannot belong to a set of w's."*
- *"…⋆dy=dz∧dx but you reversed the orientation"* · *"…seem to be in contradiction. Did you mean …?"*
- proposed cues: `something (is )?wrong with`, `but you (reversed|forgot|missed|need)`, `did you mean`, `seem to be in contradiction`

### Retract — *dialogical (stripped; 0% in arXiv)*
Alt phrasings: "whoops/oops, my mistake" · "I'll remove my answer/comment" · "sorry for
misreading" · "my mis-interpretation" · "I was being stupid".
- *"@NajibIdrissi Oh, whoops, that was my mistake."*
- *"OK, perhaps it is my mis-interpretation, I'll remove my answer."* · *"Oh, sorry for misreading!"*
- proposed cues: `whoops|my mistake|my bad`, `i'?ll remove my (answer|comment)`, `sorry for (misreading|the confusion)`, `mis-?interpretation`

### Suggest — *survives*
Alt phrasings: "it might (also) help to" · "the simplest approach is to" · "just do X and …" ·
"I think X does a better job".
- *"It might also help to say what notation you use…"* · *"Just do as you are doing: treat the maps … and dualize the diagrams…"*
- proposed cues: `it might (also )?help to`, `simplest approach is to`, `does a better job`

### Query — *survives (open-problem)*
Alt phrasings: "how can I see …?" · "what would it mean for …?" · "Can X be defined as …?" ·
"Is there a (general) reason why …?".
- *"Can the global choice operator be defined as some sort of endofunctor?"* · *"Is there a general reason why the completion should exist?"*
- proposed cues: `how can i (see|show)`, `is there a (general )?reason why`, `can .{0,30} be defined as`

## Value

### plausible — *survives (the big miss)*
Dominant MO/SE hedge is **"I think" / "I believe" / "I'm pretty sure"** — the cues miss these entirely.
- *"I'm pretty sure that the answer is yes."* · *"I believe this is a question that has not been adequately explored."*
- proposed cues: `\bi (think|believe|suspect|guess)\b`, `i'?m (pretty |fairly )?sure`, `\bpresumably\b` (already)

### easy — *survives*
Alt phrasings: "it shouldn't be too hard to show" · "this is immediate".
- *"I think it shouldn't be too hard to show using Dugger's technology that N is full…"*
- proposed cues: `shouldn'?t be (too )?hard to (show|prove|see)`

### beautiful — *dialogical praise (stripped from papers, present in comments)*
Alt phrasings: "genius" · "Nice!" · "great" · "slick" — informal aesthetic praise.
- *"Thank you. Your construction of C is genius."* · *"great comment by the way. INCREDIBLY helpful."*
- proposed cues: `\bgenius\b`, `\bslick\b`, `\bbeautiful\b|elegant` (already) — NB "nice/great" too broad for arXiv, fine for the MO/SE baseline only.

### useful — *survives*
Already well-covered (`useful/convenient`); MO/SE adds informal "INCREDIBLY helpful", "this helps".

## Meta

### goal — *survives*
Alt phrasings: "I want to" · "it would be nice to see that …" · "what I'm after is".
- *"…Somehow it would be nice to see that (G-mod, …) is Frobenius iff …"*
- proposed cues: `it would be nice (to|if)`, `\bwhat i'?m (after|trying)\b`

### strategy — *survives*
Alt phrasings: "the approach is to" · "treat X as Y and [dualize/…]" · "let me expand on that".
- *"treat the maps α∗:X→X as unary co-operations and dualize the diagrams…"* · *"Let me expand on that."*

### auxiliary — *survives*
Alt phrasing: **"the key is X" / "the key ingredient"** (the cues only had "lemma").
- *"The key is, I believe, that CGWH is in fact cartesian closed."*
- proposed cues: `\bthe key (is|ingredient|point)\b`

### analogy — *survives*
Alt phrasing: proportional **"X is to Y as Z is to W"** + "plays the same role as".
- *"Set is to a general topos as Rel is to what structure?"*
- proposed cues: `is to .{0,40} as .{0,40} is to`, `plays the (same|analogous) role`

### generalise — *survives*
Alt phrasings: "still applies if/when" · "the same works whenever".
- *"The results of the paper still apply if W is homogeneous…"*
- proposed cues: `still (apply|applies|holds?) (if|when|whenever)`, `the same .{0,20} whenever`

### implements — *dialogical (rare even here)*
No clean MO/SE instance in the sample — consistent with its 3% arXiv rate. Carry as a known gap.

## How to use this

1. **Fold the proposed cues into `iatc_alignment_passA.py`** as a `v2`/informal cue set (kept
   separate from the strict set so the arXiv-vs-dialogue comparison stays honest), then re-run
   both corpora for a register-robust table.
2. **Add the verbatim examples to the §3b Tier-2 exemplar bank** for Pass B.
3. The **dialogical** categories (Agree/Challenge/Retract/beautiful-as-praise) are confirmed:
   common and informally-phrased in MO/SE comments, absent from arXiv prose — *stripped on
   publication*. The **plausible** miss ("I think/I believe") is the single biggest recall fix
   for the value layer.
