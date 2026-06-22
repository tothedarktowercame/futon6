#!/usr/bin/env python3
"""Build the CLean structure-embedding demo for Rob.

Reads the embedder + graph-export artifacts and emits a single self-contained
HTML page (house style of the readiness pages) that shows:
  1. the pipeline (APM proof -> CLean -> {comb graph, structure embedding} -> retrieval),
     drawn as the analogue of Rob's Lean -> neo4j + pgvector;
  2. STRUCTURE vs TEXT: side-by-side cosine heatmaps + a nearest-neighbor table
     that highlights where structure finds a cross-topic twin and text misses it;
  3. per-proof: the CLean outline (method spine + typed-hole boxes) and a small
     comb diagram, plus its structure nearest-neighbors.

Usage:
  futon6/.venv/bin/python scripts/build_clean_demo.py [--dir data/showcases/clean-demo]
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/showcases/clean-demo")
    args = ap.parse_args()

    embed = json.load(open(os.path.join(args.dir, "clean-embed.json")))
    graph = json.load(open(os.path.join(args.dir, "ingest", "clean-graph.json")))

    # group steps + edges by proof
    steps, wires, discharges = {}, {}, {}
    for n in graph["nodes"]:
        if n["label"] == "Step":
            steps.setdefault(n["proof"], []).append(n)
    for e in graph["edges"]:
        if e["type"] == "WIRES":
            pid = e["from"].split("/")[0]
            wires.setdefault(pid, []).append({"from": e["from"].split("/")[1],
                                              "to": e["to"].split("/")[1],
                                              "carries": e["carries"]})
        elif e["type"] == "DISCHARGES_TO":
            pid = e["from"].split("/")[0]
            discharges.setdefault(pid, []).append({"from": e["from"].split("/")[1], "to": e["to"]})

    data = {
        "embed": embed,
        "steps": steps,
        "wires": wires,
        "discharges": discharges,
    }

    html = HTML_TEMPLATE.replace("/*__DATA__*/", json.dumps(data))
    out = os.path.join(args.dir, "index.html")
    with open(out, "w") as fh:
        fh.write(html)
    print(f"wrote {out}  ({len(embed['ids'])} proofs)")


HTML_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>CLean — structure embeddings for proofs (demo)</title>
<style>
 :root{--paper:#f7f5ef;--ink:#1a1a1a;--line:#ddd6c8;--ok:#1d7a3a;--ok-bg:#e7f6ec;
   --warn:#9a6a00;--warn-bg:#fdf3e0;--accent:#2456a6;--accent-bg:#e8f0fb;
   --hole:#a11;--hole-bg:#fbe9e9;--purple:#6d3aa8;--purple-bg:#f0e8fb;}
 *{box-sizing:border-box}
 body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.55 "Iowan Old Style",Georgia,serif;}
 header{padding:24px 32px 6px;} h1{font-size:23px;margin:0 0 4px;} .meta{color:#666;font-size:12.5px;max-width:980px;}
 .claim{margin:14px 32px;padding:12px 16px;border-left:4px solid var(--accent);background:#fff;border:1px solid var(--line);
   border-left-width:4px;border-radius:6px;font-size:13.5px;max-width:980px;}
 .claim b{color:#000}
 .flow{margin:14px 32px;padding:10px 14px;border:1px solid var(--line);background:#fff;border-radius:6px;
   font:12.5px/1.7 ui-sans-serif,system-ui;color:#333;max-width:980px;} .flow b{color:#000}
 .flow .ours{color:var(--accent);font-weight:600} .flow .robs{color:var(--purple);font-weight:600}
 main{padding:8px 24px 40px;} section{margin:22px 8px;}
 h2{font-size:16px;margin:0 0 4px;border-bottom:2px solid var(--line);padding-bottom:4px;}
 .sub{color:#777;font-size:12px;margin:0 0 12px;max-width:980px;}
 .twocol{display:flex;gap:24px;flex-wrap:wrap;}
 table.heat{border-collapse:collapse;font:11px/1 ui-sans-serif,system-ui;}
 table.heat th{font-weight:600;color:#555;padding:3px 4px;text-align:center;}
 table.heat th.row{text-align:right;padding-right:6px;}
 table.heat td{width:34px;height:30px;text-align:center;color:#1a1a1a;border:1px solid #fff;font-variant-numeric:tabular-nums;}
 .nn-table{border-collapse:collapse;font:12px/1.4 ui-sans-serif,system-ui;margin-top:4px;}
 .nn-table th,.nn-table td{border:1px solid var(--line);padding:5px 9px;text-align:left;}
 .nn-table th{background:#faf8f2;color:#555;font-weight:600;}
 .nn-table code{font-size:11px;}
 .disagree{background:var(--ok-bg);} .disagree .x{color:var(--hole);font-weight:700;}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:14px;}
 .card{border:1px solid var(--line);border-top-width:4px;background:#fff;border-radius:7px;padding:12px 14px;}
 .card.construct-exploit-discharge{border-top-color:var(--ok);}
 .card.count-invariant-obstruct{border-top-color:var(--purple);}
 .card.cover-estimate{border-top-color:var(--accent);}
 .card.contradiction-reduce{border-top-color:var(--warn);}
 .card.induct-tower{border-top-color:#0e7490;}
 .card h3{margin:0 0 1px;font-size:14px;} .card .id{color:#999;font:600 11px/1 ui-sans-serif,system-ui;}
 .macro{display:inline-block;font:600 10px/1.5 ui-sans-serif,system-ui;text-transform:uppercase;letter-spacing:.03em;
   padding:1px 7px;border-radius:10px;background:var(--accent-bg);color:var(--accent);margin-left:6px;}
 .spine{font:11.5px/1.5 "SF Mono",ui-monospace,Menlo,monospace;color:#444;margin:8px 0 6px;}
 .spine .m{background:#ece8dd;border-radius:3px;padding:1px 5px;margin:0 1px;display:inline-block;}
 svg{display:block;margin:6px 0;background:#fcfbf7;border:1px solid var(--line);border-radius:5px;width:100%;}
 .step-rect{fill:#fff;stroke:#bbb;} .step-hole{fill:var(--hole-bg);stroke:var(--hole);}
 .step-lbl{font:600 9px/1 ui-sans-serif,system-ui;fill:#222;} .step-meth{font:8px/1 ui-sans-serif,system-ui;fill:#666;}
 .wire{stroke:#888;stroke-width:1.3;fill:none;marker-end:url(#arr);}
 .boxes{margin-top:6px;} .box{font-size:12px;margin:5px 0;padding-left:10px;border-left:2px solid var(--line);}
 .box .bm{font:600 11px/1 ui-sans-serif,system-ui;color:var(--accent);}
 .box .bhole{color:var(--hole);font-size:11px;} .box .bdis{color:var(--ok);font-size:11px;}
 .nn{margin-top:8px;font-size:12px;} .nn b{color:#555;}
 .nn .pill{display:inline-block;background:#faf8f2;border:1px solid var(--line);border-radius:10px;
   padding:1px 8px;margin:2px 3px 0 0;font:11px/1.5 ui-sans-serif,system-ui;}
 footer{padding:8px 32px 70px;max-width:980px;} footer h2{font-size:15px;}
 footer li{font-size:13px;margin:4px 0;} code{background:#ece8dd;padding:1px 4px;border-radius:3px;font-size:11.5px;}
</style></head><body>
<header>
 <h1>CLean — structure embeddings for proofs</h1>
 <div class="meta">A demo of indexing proofs by their <b>compositional shape</b> (the comb of typed holes +
 the method spine) instead of their prose — the proof-side analogue of the Lean→neo4j+pgvector workflow.
 7 APM proofs, hand-lifted to CLean, gated 7/7 well-formed. futon6 · <code>holes/clean/</code></div>
</header>

<div class="claim" id="claim"></div>

<div class="flow">
 <b>Two pipelines, same shape:</b><br>
 <span class="robs">Rob (today):</span> Lean proof → parse → <span class="robs">neo4j</span> graph index +
 <span class="robs">pgvector</span> embedding index → structural retrieval.<br>
 <span class="ours">This demo:</span> APM informal proof <code>.md</code> →
 <span class="ours">CLean</span> EDN (comb of typed holes) → gate → <span class="ours">graph</span>
 (<code>load.cypher</code>) + <span class="ours">structure embedding</span> (<code>pgvector.sql</code>) → structural retrieval.<br>
 Same datatype as M-typed-holes' missions, so both ingest by one path.
</div>

<main>
 <section>
  <h2>Structure vs. text — the whole point</h2>
  <p class="sub">Cosine similarity between every pair of proofs, two ways. <b>Structure</b> (left) clusters by
   <i>proof method</i>; <b>text</b> (MiniLM, right) clusters by <i>surface vocabulary</i>. Darker = more similar.
   Watch the cross-topic twins (e.g. <code>b97J01</code> p-groups ↔ <code>t94A07</code> torus) light up on the left and vanish on the right.</p>
  <div class="twocol">
   <div><div style="font-weight:600;font-size:12px;margin-bottom:4px;color:var(--accent)">STRUCTURE embedding (33-d, comb shape)</div><div id="heat-struct"></div></div>
   <div><div style="font-weight:600;font-size:12px;margin-bottom:4px;color:var(--purple)">TEXT embedding (<span id="tmodel"></span>)</div><div id="heat-text"></div></div>
  </div>
  <h3 style="font-size:13px;margin:16px 0 2px;">Nearest neighbor by each method</h3>
  <table class="nn-table" id="nn-compare"></table>
 </section>

 <section>
  <h2>The 7 proofs as CLean</h2>
  <p class="sub">Each proof's informal method spine (the iching/CT tags) ∥ its formal comb (typed-hole boxes wired
   construct→consume). 🔴 = a typed hole (an undischarged obligation / residual sorry); 🟢 = discharged to a named theorem.</p>
  <div class="grid" id="cards"></div>
 </section>
</main>

<footer>
 <h2>How Rob consumes this</h2>
 <ul>
  <li><b>Graph-direct:</b> <code>ingest/load.cypher</code> → neo4j, <code>ingest/pgvector.sql</code> → postgres.
   The two queries at the foot of the SQL reproduce the structure-vs-text contrast in pure SQL (<code>&lt;=&gt;</code> cosine).</li>
  <li><b>Via Lean (no schema change):</b> CLean maps field-for-field to the DarkTower types
   (<code>Comb</code>/<code>TypedHole</code>/<code>BV</code>/<code>Discharge</code>); a deterministic CLean→Lean emitter
   renders each proof to a <code>ProofExample</code> his Lean→neo4j+pgvector path ingests unchanged. <i>(emitter not yet built — E-clean step 2)</i></li>
  <li><b>Scaling:</b> 7 proofs hand-lifted here; the full 462-proof APM corpus is an LLM batch pass on a Linode,
   box-typing constrained to <code>clean-method-vocab.edn</code> so the embedding space stays shared.</li>
  <li><b>Honest boundary:</b> real = the CLean files, the 7/7 gate, both embeddings, the clustering result, the cypher/SQL.
   Stand-in = we emit the cypher/SQL rather than standing up the DBs here. Not yet built = the Lean emitter + round-trip compile.</li>
 </ul>
 <p style="color:#888;font-size:12px">See <code>holes/clean/NEO4J-PGVECTOR-MAPPING.md</code> and <code>holes/excursions/E-clean.md</code>.</p>
</footer>

<script>
const DATA = /*__DATA__*/;
const E = DATA.embed, ids = E.ids;
const idx = {}; ids.forEach((d,i)=>idx[d]=i);
const titleOf = d => E.titles[idx[d]], macroOf = d => E.macros[idx[d]];

document.getElementById('tmodel').textContent = E.text_model + ', ' + E.text_dim + '-d';
document.getElementById('claim').innerHTML =
 "<b>Claim:</b> index the proof's <b>structure</b>, not its prose. The structural embedding makes "+
 "<code>b97J01</code> (p-groups, algebra) and <code>t94A07</code> (torus rotation, topology) nearest neighbors "+
 "(<b>cosine "+E.structure_sim[idx['b97J01']][idx['t94A07']]+"</b>) — they are the same proof shape "+
 "(count → invariant → obstruct). The text embedding of the same two scores only "+
 "<b>"+E.text_sim[idx['b97J01']][idx['t94A07']]+"</b> and picks a different, wrong twin. "+
 "Text embeddings provably plateau on structural similarity (EXP-3).";

// ---- heatmaps ----
function heat(elId, M, hue){
 let h = '<table class="heat"><tr><th></th>';
 ids.forEach(d=>h+='<th>'+d+'</th>'); h+='</tr>';
 for(let i=0;i<ids.length;i++){
  h+='<tr><th class="row">'+ids[i]+'</th>';
  for(let j=0;j<ids.length;j++){
   const v=M[i][j], a=(i===j)?0.12:Math.max(0,Math.min(1,v));
   const bg = i===j ? '#eee' : `hsl(${hue} 60% ${100-55*a}%)`;
   h+=`<td style="background:${bg}">${i===j?'·':v.toFixed(2)}</td>`;
  }
  h+='</tr>';
 }
 h+='</table>'; document.getElementById(elId).innerHTML=h;
}
heat('heat-struct', E.structure_sim, 214);
heat('heat-text', E.text_sim, 275);

// ---- nearest-neighbor comparison ----
let nt = '<tr><th>proof</th><th>topic</th><th>structure NN (method)</th><th>text NN (surface)</th><th></th></tr>';
ids.forEach(d=>{
 const s = E.structure_nn[d][0], t = E.text_nn[d][0];
 const disagree = s.id !== t.id;
 nt += `<tr class="${disagree?'disagree':''}">`+
   `<td><code>${d}</code></td>`+
   `<td>${titleOf(d)}</td>`+
   `<td><code>${s.id}</code> <span style="color:#888">${s.sim}</span><br><span style="font-size:10px;color:var(--accent)">${macroOf(s.id)}</span></td>`+
   `<td><code>${t.id}</code> <span style="color:#888">${t.sim}</span></td>`+
   `<td>${disagree?'<span class="x">≠</span> structure finds a cross-method twin text misses':'agree'}</td></tr>`;
});
document.getElementById('nn-compare').innerHTML = nt;

// ---- comb SVG (columns by topological depth) ----
function combSVG(pid){
 const steps = DATA.steps[pid]||[], wires = DATA.wires[pid]||[];
 const sid = s => s.id.split('/')[1];
 const byId={}; steps.forEach(s=>byId[sid(s)]=s);
 const succ={}, indeg={}; steps.forEach(s=>{succ[sid(s)]=[];indeg[sid(s)]=0;});
 wires.forEach(w=>{succ[w.from].push(w.to);indeg[w.to]++;});
 // longest-path depth
 const depth={}; steps.forEach(s=>depth[sid(s)]=0);
 let q=steps.map(sid).filter(i=>indeg[i]===0), ind=Object.assign({},indeg);
 while(q.length){const n=q.shift();(succ[n]||[]).forEach(m=>{depth[m]=Math.max(depth[m],depth[n]+1);if(--ind[m]===0)q.push(m);});}
 const cols={}; steps.forEach(s=>{const d=depth[sid(s)];(cols[d]=cols[d]||[]).push(sid(s));});
 const W=60,H=34,GX=104,GY=46,PAD=14;
 const pos={}; Object.keys(cols).forEach(d=>{cols[d].forEach((id,k)=>{pos[id]={x:PAD+d*GX,y:PAD+k*GY};});});
 const maxd=Math.max(...Object.values(depth)), maxr=Math.max(...Object.values(cols).map(c=>c.length));
 const svgW=PAD*2+maxd*GX+W, svgH=PAD*2+(maxr-1)*GY+H;
 let s='<svg viewBox="0 0 '+svgW+' '+svgH+'" height="'+(svgH)+'">';
 s+='<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#888"/></marker></defs>';
 wires.forEach(w=>{const a=pos[w.from],b=pos[w.to];if(!a||!b)return;
   s+=`<path class="wire" d="M${a.x+W},${a.y+H/2} C${a.x+W+30},${a.y+H/2} ${b.x-30},${b.y+H/2} ${b.x},${b.y+H/2}"/>`;});
 steps.forEach(st=>{const id=sid(st),p=pos[id];const hole=st.has_hole;
   s+=`<rect class="${hole?'step-hole':'step-rect'}" x="${p.x}" y="${p.y}" width="${W}" height="${H}" rx="4"/>`;
   s+=`<text class="step-lbl" x="${p.x+6}" y="${p.y+13}">${id}${hole?' ●':''}</text>`;
   s+=`<text class="step-meth" x="${p.x+6}" y="${p.y+25}">${st.method.replace(/-/g,' ').slice(0,16)}</text>`;});
 s+='</svg>'; return s;
}

// ---- per-proof cards ----
let cards='';
ids.forEach(d=>{
 const bd = E.breakdowns[idx[d]];
 const spine = bd.methods.map(m=>'<span class="m">'+m+'</span>').join(' → ');
 const steps = DATA.steps[d]||[], dis = DATA.discharges[d]||[];
 const disMap={}; dis.forEach(x=>disMap[x.from]=x.to);
 let boxes='';
 steps.forEach(st=>{const id=st.id.split('/')[1];
   boxes+='<div class="box"><span class="bm">'+id+' · '+st.method+'</span><br>'+st.text;
   if(st.has_hole) boxes+='<br><span class="bhole">🔴 hole ['+st.satiety+'/'+st.discharge+']</span>';
   if(disMap[id]) boxes+='<br><span class="bdis">🟢 discharges → '+disMap[id]+'</span>';
   boxes+='</div>';});
 const nn = E.structure_nn[d].slice(0,3).map(x=>'<span class="pill"><code>'+x.id+'</code> '+x.sim+' · '+macroOf(x.id)+'</span>').join('');
 cards += `<div class="card ${bd.macro}">`+
   `<h3><span class="id">${d}</span> ${titleOf(d)} <span class="macro">${bd.macro}</span></h3>`+
   `<div class="spine">${spine}</div>`+
   combSVG(d)+
   `<div class="boxes">${boxes}</div>`+
   `<div class="nn"><b>structure nearest:</b><br>${nn}</div>`+
   `</div>`;
});
document.getElementById('cards').innerHTML = cards;
</script>
</body></html>
"""

if __name__ == "__main__":
    main()
