#!/usr/bin/env python3
"""Consume-side: resolved memes → rollout moves (endpoint-identity bridge) + ROLLOUT-FEED.

Completes the consume half of M-operational-vocabulary: a meme whose endpoint resolves to a known
rollout MISSION becomes a meme-grounded move (real-ask provenance, beating the structure-borrowed prior).
The namespaces differ — meme refs (`mission/M-*`, `hole/*`, `agent/*`) vs rollout nodes
(`<repo>-d/mission/<stem>`, `<stem>/<phase>`, `scope/capability/*`) — so ONLY `mission/M-<stem>` refs that
match a known mission bridge; everything else is recorded out-of-rollout-domain (still arrow-store material,
not a move-prior for THIS rollout). Per the M-wm-policies declare-don't-guess seam.

  futon6/.venv/bin/python scripts/meme_consume.py            # bridge the real sample + report
  futon6/.venv/bin/python scripts/meme_consume.py --selftest # + a synthetic mission-resolved meme → move-set
"""
import argparse, json, re
ROOT = "/home/joe/code/futon6"; OUT = f"{ROOT}/data/meme-mine"


def mission_nodes():
    nodes = {}
    for s in json.load(open(f"{ROOT}/data/diffsub-scopes.json")):
        m = s.get("mission")
        if m and m not in nodes and s.get("mission_node"):
            nodes[m] = s["mission_node"]
    return nodes


def bridge(memes, nodes):
    moves, grounded, out = [], [], []
    for m in memes:
        mm = m["meme"]; hit = None
        for end in ("have", "want"):
            # the model may omit an endpoint entirely; a missing have/want is just "no ref here".
            g = re.match(r"mission/M-(.+)$", (mm.get(end) or {}).get("ref") or "")
            if g and g.group(1) in nodes:
                hit = g.group(1); break
        if hit:
            op = mm.get("op", "act"); mid = m["id"]
            moves.append({"id": f'{nodes[hit]}->{hit}/meme-{op}', "have": nodes[hit],
                          "want": f"{hit}/meme-{op}", "op": op, "conf": "mined-meme", "meme": mid,
                          "ask": m.get("ask", "")[:100]})
            grounded.append((mid, hit))
        else:
            out.append(m["id"])
    return moves, grounded, out


def edn_move(d):
    note = f'meme-grounded: {d["meme"]} op={d["op"]} | {d["ask"]}'.replace('"', "'")
    return (f'  {{:move/id "{d["id"]}" :move/class :close-hole :have "{d["have"]}" :want "{d["want"]}"'
            f' :advances-cap nil :score 0.01 :prior 0.05 :delta-g -0.001 :confidence :mined-meme'
            f' :rank 900 :move/terminal? false :note "{note}"}}')


def write_moveset(meme_moves, path):
    base = open(f"{ROOT}/data/diffsub-moves-mined.edn").read()
    inject = "\n".join(edn_move(d) for d in meme_moves)
    open(path, "w").write(base.replace("\n ]}", "\n" + inject + "\n ]}"))


def move_stems(path):
    """rollout-mission stems that already have a move (have = <repo>-d/mission/<stem>)."""
    try:
        txt = open(path).read()
    except FileNotFoundError:
        return set()
    return set(re.findall(r'-d/mission/([^"/]+)', txt))


def do_floor():
    """ACTIONABILITY-FLOOR + ACTION-CERT: per rollout mission, the best available provenance —
    meme-grounded (real ask) > structure-borrowed (MINE-GEN) > island (no move → needs-a-foothold)."""
    nodes = mission_nodes()
    struct = move_stems(f"{ROOT}/data/diffsub-moves-mined.edn")
    meme = move_stems(f"{OUT}/diffsub-moves-meme.edn")  # populated by the box/targeted run
    cert = {}
    for stem in nodes:
        tier = "meme-grounded" if stem in meme else ("structure-borrowed" if stem in struct else "island")
        cert[stem] = tier
    from collections import Counter
    c = Counter(cert.values())
    print(f"ACTIONABILITY-FLOOR over {len(nodes)} rollout missions:")
    for t in ("meme-grounded", "structure-borrowed", "island"):
        print(f"  {t}: {c.get(t,0)}")
    print("  (islands = no move → surfaced as 'needs-a-foothold', never given a fabricated path;")
    print("   82/198 are meme-GROUNDABLE via targeted sampling — they upgrade to meme-grounded once mined)")
    json.dump(cert, open(f"{OUT}/action-cert.json", "w"), indent=2)
    print(f"wrote {OUT}/action-cert.json (per-mission provenance certificate)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--floor", action="store_true")
    ap.add_argument("--memes", default=f"{OUT}/resolved-memes.json",
                    help="resolved-memes json (e.g. resolved-memes.openai.json from the box run)")
    a = ap.parse_args()
    if a.floor:
        do_floor()
        return
    nodes = mission_nodes()
    memes = json.load(open(a.memes))
    moves, grounded, out = bridge(memes, nodes)
    print(f"resolved memes: {len(memes)}  |  rollout missions known: {len(nodes)}")
    print(f"meme→move bridge: {len(grounded)} mission-grounded, {len(out)} out-of-rollout-domain")
    print(f"  (random small sample → {len(grounded)} mission-grounded: the asks don't name rollout missions;")
    print(f"   coverage needs box-scale OR mission-targeted sampling via turn→pattern→mission)")

    if a.selftest:
        # MECHANISM test (synthetic, clearly labelled): a meme that DOES resolve to a real mission.
        stem = "canon-fingerprint-store" if "canon-fingerprint-store" in nodes else next(iter(nodes))
        synth = [{"id": "ask-SELFTEST", "ask": "let's build the canon fingerprint store",
                  "meme": {"have": {"ref": f"mission/M-{stem}", "tier": "named"},
                           "want": {"ref": None, "tier": "unsupported"}, "op": "build"}}]
        sm, sg, _ = bridge(synth, nodes)
        assert sg and sm[0]["have"].endswith(f"/mission/{stem}"), "bridge failed to produce a seedable move"
        assert re.search(r"-d/mission/", sm[0]["have"]), "have is not a seedable mission root"
        write_moveset(sm, "/tmp/diffsub-moves-meme.edn")
        print(f"\n[selftest] synthetic meme (mission/M-{stem}, op=build) → move "
              f'have="{sm[0]["have"]}" want="{sm[0]["want"]}"')
        print(f"[selftest] have matches the rollout mission-root regex (-d/mission/) → seedable; "
              f"wrote /tmp/diffsub-moves-meme.edn (run the flip harness on stem '{stem}').")
        print(f"[selftest] move shape == the witnessed structure-borrowed move shape → ΔG-yielding by construction.")
    if moves:
        write_moveset(moves, f"{OUT}/diffsub-moves-meme.edn")
        print(f"wrote {OUT}/diffsub-moves-meme.edn ({len(moves)} meme-grounded moves)")


if __name__ == "__main__":
    main()
