#!/usr/bin/env python3
"""code_diff_jax_pilot.py — INSTANTIATE vertical slice for futon5 M-differentiable-code.

The `jax_refine.py` port, slot-for-slot, on real code:
  - fixed BGE scope embeddings  = measurement instrument, a CONSTANT outside jax  (O4a)
  - soft typed adjacency A[n,t]  = the optimized structural choice (softmax routing)
  - authored cosine band         = the manufactured predicate->band spec (item-zero)
  - grad(loss)(A)                = ranked edit-proposals

Two stages (different venvs — embedding is PyTorch, optimization is JAX):
  futon6/.venv/bin/python   code_diff_jax_pilot.py --embed   # BGE -> emb.npy + nodes.json
  futon5/.venv-tpg/bin/python code_diff_jax_pilot.py --jax   # loads them, runs grad step

Owner: claude-6 (E2). Escrow-clean, OUT of campaign (futon6 timeline). Baseline shape =
`:symbol` + per-node normalization, BGE-embedded (the keystone's ratified IFR pilot).
"""
from __future__ import annotations
import ast, json, os, sys, pathlib
from collections import Counter
import numpy as np

OUT = pathlib.Path("/home/joe/code/futon6/resources/differentiable-math/code-scope-probe")
SRC = pathlib.Path("/home/joe/code/futon6/src/futon6")
EMB = OUT / "emb.npy"
NODES = OUT / "nodes.json"

# authored band (gap-3 discipline: bands specified, never fit) — placeholder pending
# real wiring-claims. center 0.60 = "coherent but not a near-duplicate".
BAND_CENTER, BAND_WIDTH = 0.60, 0.12


def first_lines(s, n):
    return " ".join(l.strip() for l in (s or "").strip().splitlines()[:n] if l.strip())


def sig_of(node):
    try:
        a = node.args
        names = [ar.arg for ar in (a.posonlyargs + a.args)]
        if a.vararg: names.append("*" + a.vararg.arg)
        names += [ar.arg for ar in a.kwonlyargs]
        if a.kwarg: names.append("**" + a.kwarg.arg)
        return ", ".join(names)
    except Exception:
        return ""


def extract():
    nodes = []
    for path in sorted(SRC.glob("*.py")):
        mod = path.stem
        if mod == "__init__":
            continue
        try:
            tree = ast.parse(path.read_text())
        except Exception:
            continue
        moddoc = first_lines(ast.get_docstring(tree), 1)

        def walk(body, prefix=""):
            for n in body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qual = prefix + n.name
                    doc = first_lines(ast.get_docstring(n), 2)
                    nodes.append(dict(id=f"{mod}:{qual}", module=mod, name=qual,
                        text=f"module {mod}: {moddoc}\n{qual}({sig_of(n)})\n{doc}"))
                elif isinstance(n, ast.ClassDef):
                    cdoc = first_lines(ast.get_docstring(n), 1)
                    nodes.append(dict(id=f"{mod}:{n.name}", module=mod, name=n.name,
                        text=f"module {mod}: {moddoc}\nclass {n.name}\n{cdoc}"))
                    walk(n.body, prefix=n.name + ".")
        walk(tree.body)
    return nodes


def do_embed():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer
    nodes = extract()
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    vecs = model.encode([n["text"] for n in nodes], normalize_embeddings=True,
                        show_progress_bar=False)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(EMB, np.asarray(vecs, dtype=np.float32))
    NODES.write_text(json.dumps(nodes))
    print(f"embedded {len(nodes)} nodes -> {EMB}  shape={np.asarray(vecs).shape}")


def do_jax():
    import jax, jax.numpy as jnp
    from jax import grad
    emb = np.load(EMB)
    nodes = json.loads(NODES.read_text())
    N = emb.shape[0]
    modules = [n["module"] for n in nodes]
    E = jnp.asarray(emb)                 # FIXED observation — constant in jax (O4a)
    C = E @ E.T                          # cosine matrix (unit-normalized) — constant
    sat = jnp.exp(-((C - BAND_CENTER) / BAND_WIDTH) ** 2)   # authored band satisfaction
    diag_mask = jnp.where(jnp.eye(N, dtype=bool), -1e9, 0.0)  # no self-edges

    def loss(A):
        P = jax.nn.softmax(A + diag_mask, axis=1)   # soft adjacency: n's chosen target
        s = jnp.sum(P * sat, axis=1)                # per-node band satisfaction
        return -jnp.mean(s)

    g = grad(loss)
    A0 = jnp.zeros((N, N))

    # --- O4(b) conditioning: per-node gradient norm at uniform init ---
    G0 = g(A0)
    gnorm = np.asarray(jnp.sqrt(jnp.sum(G0 ** 2, axis=1)))
    msize = Counter(modules)
    modsize = np.array([msize[m] for m in modules], dtype=float)
    corr = float(np.corrcoef(gnorm, modsize)[0, 1])
    # derivation: d loss / d A[n,:] is proportional to the VARIANCE of node n's
    # band-satisfaction across targets. So grad-norm tracks a structural (degree-like)
    # quantity, NOT code line-count (the embedding window is bounded, so line-size never
    # enters). max/med is the conditioning health number; corr shows what drives scale.
    print(f"[O4b] grad-norm/node  min={gnorm.min():.2e}  med={np.median(gnorm):.2e}  "
          f"max={gnorm.max():.2e}  max/med={gnorm.max()/np.median(gnorm):.2f}  "
          f"(<~2 => numerically sane on this slice)")
    print(f"[O4b] corr(grad-norm, module-size) = {corr:+.3f}   "
          f"(>0 => a DEGREE-like quantity drives gradient scale; NOT line-count)")

    # --- the gradient loop: jax_refine's optimize step on code A ---
    lr = float(os.environ.get("LR", "300.0"))
    steps = int(os.environ.get("STEPS", "500"))
    A = A0
    sat_before = float(-loss(A))
    for _ in range(steps):
        A = A - lr * g(A)
    sat_after = float(-loss(A))
    print(f"[loop] mean band-satisfaction {sat_before:.4f} -> {sat_after:.4f}  "
          f"(improvement {sat_after - sat_before:+.4f})")

    # --- edit proposals: softmax-mass shift per node ---
    P0 = np.asarray(jax.nn.softmax(A0 + diag_mask, axis=1))
    P1 = np.asarray(jax.nn.softmax(A + diag_mask, axis=1))
    Cn = np.asarray(C)
    seen, probes = set(), []
    for i, n in enumerate(nodes):
        if n["module"] in ("faiss_index", "latex_sexp", "symbol_grounding") \
                and n["module"] not in seen:
            probes.append(i); seen.add(n["module"])
    print("\n--- edit proposals (target with largest softmax-mass gain after grad) ---")
    for i in probes:
        gain = P1[i] - P0[i]
        j = int(np.argmax(gain))
        print(f"[{nodes[i]['id']}]")
        print(f"   PROMOTE -> {nodes[j]['id']:42s} cos={Cn[i,j]:.3f} "
              f"band-sat={float(sat[i,j]):.3f}  mass {P0[i,j]:.4f}->{P1[i,j]:.4f}")

    summary = dict(n=N, band_center=BAND_CENTER, band_width=BAND_WIDTH,
                   sat_before=sat_before, sat_after=sat_after,
                   gradnorm_min=float(gnorm.min()), gradnorm_med=float(np.median(gnorm)),
                   gradnorm_max=float(gnorm.max()), gradnorm_corr_modsize=corr)
    (OUT / "code-diff-jax-result.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT/'code-diff-jax-result.json'}")


if __name__ == "__main__":
    if "--embed" in sys.argv:
        do_embed()
    elif "--jax" in sys.argv:
        do_jax()
    else:
        print("use --embed (futon6/.venv) then --jax (futon5/.venv-tpg)")
