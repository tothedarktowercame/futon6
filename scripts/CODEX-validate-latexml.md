# Codex Task: Validate LaTeX→s-exp parser against LaTeXML

## Goal

Compare futon6's custom LaTeX→s-exp parser (`src/futon6/latex_sexp.py`) against
LaTeXML's Content MathML output. Identify:
1. Expressions where LaTeXML parses but we don't (**gaps** — we claim superset)
2. Structural disagreements (both parse, different trees)
3. Expressions where we parse but LaTeXML doesn't (**our superset**)

## Setup

```bash
cd /home/joe/code/futon6

# Install LaTeXML (Perl-based, ~100MB)
sudo apt install -y latexml

# Verify
latexmlmath --VERSION
latexmlmath --cmml=- '\frac{a}{b}'
```

## Run

```bash
# Full validation: built-in corpus (52 exprs) + thread #633512 (33 exprs)
PYTHONPATH=src python3 scripts/validate-latexml.py \
    data/first-proof/thread-633512-hypergraph.json

# Or just the built-in corpus
PYTHONPATH=src python3 scripts/validate-latexml.py --builtin-only
```

## Expected output

- `validate-latexml-report.json` — full per-expression report
- Console summary showing match/disagreement counts

## What to investigate

1. **LaTeXML-only cases**: These are our parser's gaps. For each:
   - Is the LaTeX construct one we should support?
   - If yes, add it to the parser in `src/futon6/latex_sexp.py`
   - Add a test case in `tests/test_latex_sexp.py`

2. **Structural disagreements**: For each:
   - Which parse is more semantically correct?
   - If LaTeXML's is better, adjust our parser
   - If ours is better (e.g., we use `→` where LaTeXML uses `apply`), document why

3. **Our-only cases**: These demonstrate our superset claim. Document them.

## Stretch goals

- Extract a larger corpus from `data/ct-validation/scopes.json` (1229 scope
  records with LaTeX expressions in `hx/ends`)
- Run validation on PlanetMath LaTeX expressions (from `data/ner-kernel/`)
- Improve our parser to handle any LaTeXML-only constructs found

## Files

| File | Role |
|------|------|
| `scripts/validate-latexml.py` | Validation script |
| `src/futon6/latex_sexp.py` | Our parser (modify if gaps found) |
| `tests/test_latex_sexp.py` | Parser tests (add new cases) |
| `data/first-proof/thread-633512-hypergraph.json` | Thread corpus source |
