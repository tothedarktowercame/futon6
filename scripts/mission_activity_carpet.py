#!/usr/bin/env python3
# mission_activity_carpet.py — mission activity and code churn laid over the embedding carpet
# (Joe, 2026-09-25: "build the material so we can visualize it or use it to augment the
# existing embedding"). Reads data/mission-activity.json (scripts/mission_activity.py) and the
# embedding positions data/mission-carpet-pos-embed.json; writes a self-contained
# data/mission-activity-embed.html. Colour, size and ring are switchable in the page; clicking
# a mission shows its doc/code sparklines and draws its coupling to other missions.
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import futon6_config as config  # noqa: E402

DATA = config.code_root() / "futon6" / "data"


def main(activity=DATA / "mission-activity.json", pos=DATA / "mission-carpet-pos-embed.json",
         out=DATA / "mission-activity-embed.html"):
    act = json.loads(Path(activity).read_text())
    P = json.loads(Path(pos).read_text())
    rows = act["missions"]
    placed = [r for r in rows if r["mission"] in P]
    unplaced = sorted(r["mission"] for r in rows if r["mission"] not in P)
    payload = {"generated": act.get("generated"), "resolution": act.get("resolution"),
               "pos": {r["mission"]: P[r["mission"]] for r in placed}, "missions": rows,
               "unplaced": unplaced}
    html = PAGE.replace("/*DATA*/null", json.dumps(payload, separators=(",", ":")))
    Path(out).write_text(html)
    print(f"wrote {out}: {len(placed)} placed on the embedding, {len(unplaced)} without a position")


PAGE = r"""<!doctype html><meta charset=utf-8><title>Mission activity — embedding carpet</title>
<style>
body{margin:0;background:#0a0c11;color:#cdd3df;font:13px sans-serif;display:flex;height:100vh}
#map{flex:1;overflow:hidden;position:relative}#side{width:520px;overflow:auto;border-left:1px solid #222;padding:10px}
h1{font-size:15px;margin:0 0 4px}p,.note{color:#8b95a7;font-size:12px;margin:2px 0}
select,button{background:#1a1e27;color:#cdd3df;border:1px solid #333;border-radius:4px;padding:2px 6px}
table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:2px 4px;text-align:left;white-space:nowrap}
th{cursor:pointer;color:#9fb0cc}tr:hover{background:#161a22}tr.sel{background:#243049}
.sp{font-family:monospace;letter-spacing:-1px;font-size:10px}a{color:#b08fd0}.null{color:#555}#info{border:1px solid #2a2f3a;padding:6px;margin:8px 0;min-height:40px}
svg text{pointer-events:none}
</style>
<div id=map><svg id=svg width=100% height=100%><g id=vp><g id=links></g><g id=nodes></g><g id=labels></g></g></svg></div>
<div id=side>
<h1>Mission activity over the embedding</h1>
<p id=meta></p>
<p>Colour <select id=col><option value=recency>days since any activity</option><option value=hot>hotspot: code churn 90d × complexity</option><option value=cls>wholeness class</option></select>
 Size <select id=size><option value=L>wholeness L</option><option value=churn>code commits 90d</option><option value=none>uniform</option></select></p>
<p class=note>Grey hollow = no data from that source (null, not zero). Ring = wholeness L. Click a mission for sparklines and coupling; scroll to zoom, drag to pan.</p>
<div id=info>Click a mission.</div>
<p>Show <select id=filt><option value=carpet>on the embedding</option><option value=all>all rows (incl. run-/leaf- ids)</option><option value=open>open (not complete/archived/closed)</option><option value=stale>alive L≥60, no activity in 90 days</option></select> <span id=cnt></span></p>
<table><thead><tr><th data-k=mission>mission</th><th data-k=days>idle d</th><th data-k=L>L</th><th data-k=c90>code 90d</th><th data-k=hot>hot</th><th>doc · code (26w)</th></tr></thead><tbody id=tb></tbody></table>
</div>
<script>
const D=/*DATA*/null;
const NS='http://www.w3.org/2000/svg',$=id=>document.getElementById(id);
const BARS=' ▁▂▃▄▅▆▇█';
const spark=a=>a?a.map(v=>v?BARS[Math.min(8,1+Math.min(7,v))]:'·').join(''):null;
const DONE=/complete|archived|closed|done|superseded|withdrawn|retired|delivered/i;
const now=Date.now();
const M=D.missions.map(r=>{
  const lastIdx=a=>{if(!a)return null;for(let i=a.length-1;i>=0;i--)if(a[i])return (a.length-1-i)*7;return null};
  const docDays=r.doc_last_commit?Math.round((now-Date.parse(r.doc_last_commit))/864e5):null;
  const codeDays=r.code?lastIdx(r.code.weekly):null;
  const ds=[docDays,codeDays].filter(x=>x!==null);
  const c=r.code;
  return {...r,days:ds.length?Math.min(...ds):null,L:r.wholeness?r.wholeness.L:null,
    c90:c?c.commits_90d:null,hot:c&&c.complexity!=null?c.commits_90d*c.complexity:null,
    open:!DONE.test(r.status_line||'')}
});
const byId=Object.fromEntries(M.map(m=>[m.mission,m]));
const maxOf=k=>Math.max(1,...M.map(m=>m[k]||0));
const mx={c90:maxOf('c90'),L:maxOf('L'),hot:maxOf('hot')};
$('meta').textContent=`generated ${D.generated||'?'} · ${Object.keys(D.pos).length} on the embedding, ${D.unplaced.length} without a position`+
  (D.resolution?` · vars resolved to files ${D.resolution.resolved}/${D.resolution.vars}`:'');
function colour(m,mode){
  if(mode==='cls'){if(!m.wholeness)return null;return{alive:'hsl(130,60%,50%)',mess:'hsl(9,70%,55%)',pipeline:'hsl(205,70%,55%)',stub:'#777'}[m.wholeness.class]||'#999'}
  if(mode==='hot'){if(m.hot==null)return null;const t=Math.sqrt(m.hot/mx.hot);return `hsl(${50-50*t},90%,${25+45*t}%)`}
  if(m.days==null)return null;const t=Math.min(1,m.days/180);return `hsl(${45+175*t},${85-55*t}%,${62-30*t}%)`
}
function radius(m,mode){
  if(mode==='none')return 9;
  const v=mode==='L'?m.L:m.c90,x=mode==='L'?mx.L:mx.c90;
  return v==null?6:8+22*Math.sqrt(v/x)
}
let sel=null;
function draw(){
  const cm=$('col').value,sm=$('size').value,g=$('nodes');g.innerHTML='';$('labels').innerHTML='';
  for(const m of M){const p=D.pos[m.mission];if(!p)continue;
    const c=colour(m,cm),r=radius(m,sm),e=document.createElementNS(NS,'circle');
    e.setAttribute('cx',p[0]);e.setAttribute('cy',p[1]);e.setAttribute('r',r);
    e.setAttribute('fill',c||'none');e.setAttribute('stroke',c?(m.L!=null?'#ffe08a':'none'):'#666');
    e.setAttribute('stroke-width',c?(m.L!=null?0.4+3*m.L/mx.L:0):1.2);e.setAttribute('opacity',m.open?0.9:0.35);
    e.style.cursor='pointer';e.onclick=()=>pick(m.mission);
    const t=document.createElementNS(NS,'title');t.textContent=`${m.mission}  idle ${m.days??'?'}d  L ${m.L??'—'}  code90 ${m.c90??'—'}`;e.appendChild(t);g.appendChild(e)}
  if(sel)pick(sel,true)
}
function pick(id,keep){
  sel=id;const m=byId[id],l=$('links');l.innerHTML='';$('labels').innerHTML='';
  const p=D.pos[id];
  if(p)for(const [o,n] of (m.coupling||[])){const q=D.pos[o];if(!q)continue;const e=document.createElementNS(NS,'line');
    e.setAttribute('x1',p[0]);e.setAttribute('y1',p[1]);e.setAttribute('x2',q[0]);e.setAttribute('y2',q[1]);
    e.setAttribute('stroke','#b08fd0');e.setAttribute('stroke-width',1+Math.log2(1+n));e.setAttribute('opacity',0.7);l.appendChild(e);label(o,q,'#b08fd0')}
  if(p)label(id,p,'#fff');
  const c=m.code;
  $('info').innerHTML=`<b>${id}</b> ${m.doc?`<span class=note>${m.doc}</span>`:''}<br>`+
   `status: ${m.status_line??'<span class=null>none</span>'} · phases ${m.lifecycle_phases??'?'}/7<br>`+
   `wholeness: ${m.wholeness?`${m.wholeness.class} L ${m.wholeness.L} (T ${m.wholeness.T} H ${m.wholeness.H})`:'<span class=null>not scored</span>'}<br>`+
   `doc  <span class=sp>${spark(m.doc_commits_weekly)??'<span class=null>no doc history</span>'}</span> last ${m.doc_last_commit??'—'}<br>`+
   `code <span class=sp>${c?(spark(c.weekly)??'<span class=null>touched vars resolve to no file</span>'):'<span class=null>no touched code</span>'}</span>`+
   (c?`<br>${c.vars_touched} vars → ${c.files_resolved} files (${c.vars_unresolved} unresolved) · commits 90d ${c.commits_90d??'—'} / all ${c.commits_all??'—'} · complexity ${c.complexity==null?'—':c.complexity.toFixed(1)}`:'')+
   `<br>coupled: ${(m.coupling||[]).map(([o,n])=>`<a href=# onclick="pick('${o}');return false">${o}</a> (${n})`).join(', ')||'<span class=null>none</span>'}`;
  for(const tr of $('tb').children)tr.classList.toggle('sel',tr.dataset.id===id)
}
function label(id,p,col){const t=document.createElementNS(NS,'text');t.setAttribute('x',p[0]+8);t.setAttribute('y',p[1]-6);
  t.setAttribute('fill',col);t.setAttribute('font-size',13/scale);t.textContent=id.replace(/^M-/,'');$('labels').appendChild(t)}
let sortK='days',sortDir=-1;
function table(){
  const f=$('filt').value;let rs=M.filter(m=>f==='all'||(f==='carpet'&&D.pos[m.mission])||(f==='open'&&m.open)||(f==='stale'&&m.open&&m.L!=null&&m.L>=60&&(m.days==null||m.days>90)));
  rs.sort((a,b)=>{const x=a[sortK],y=b[sortK];if(x==null)return 1;if(y==null)return -1;return (x<y?-1:x>y?1:0)*sortDir});
  $('cnt').textContent=rs.length+' shown';
  $('tb').innerHTML=rs.map(m=>`<tr data-id="${m.mission}"><td>${m.mission.replace(/^M-/,'')}</td><td>${m.days??'<span class=null>—</span>'}</td><td>${m.L??'<span class=null>—</span>'}</td><td>${m.c90??'<span class=null>—</span>'}</td><td>${m.hot==null?'<span class=null>—</span>':Math.round(m.hot)}</td><td class=sp>${spark(m.doc_commits_weekly)??''}<br>${m.code?(spark(m.code.weekly)??''):''}</td></tr>`).join('');
  for(const tr of $('tb').children)tr.onclick=()=>pick(tr.dataset.id)
}
document.querySelectorAll('th[data-k]').forEach(th=>th.onclick=()=>{const k=th.dataset.k;sortDir=k===sortK?-sortDir:-1;sortK=k;table()});
$('col').onchange=draw;$('size').onchange=draw;$('filt').onchange=table;
// pan/zoom
let scale=1,tx=0,ty=0;const vp=$('vp'),svg=$('svg');
function apply(){vp.setAttribute('transform',`translate(${tx},${ty}) scale(${scale})`)}
(function fit(){const ps=Object.values(D.pos);if(!ps.length)return;const xs=ps.map(p=>p[0]),ys=ps.map(p=>p[1]);
  const w=svg.clientWidth||900,h=svg.clientHeight||900,x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  scale=0.92*Math.min(w/(x1-x0||1),h/(y1-y0||1));tx=(w-scale*(x0+x1))/2;ty=(h-scale*(y0+y1))/2;apply()})();
svg.addEventListener('wheel',e=>{e.preventDefault();const k=e.deltaY<0?1.15:1/1.15,r=svg.getBoundingClientRect(),mx_=e.clientX-r.left,my=e.clientY-r.top;
  tx=mx_-(mx_-tx)*k;ty=my-(my-ty)*k;scale*=k;apply();if(sel)pick(sel,true)},{passive:false});
let drag=null;svg.onmousedown=e=>{drag=[e.clientX-tx,e.clientY-ty]};window.onmouseup=()=>drag=null;
window.onmousemove=e=>{if(drag){tx=e.clientX-drag[0];ty=e.clientY-drag[1];apply()}};
draw();table();
</script>"""

if __name__ == "__main__":
    main(*sys.argv[1:])
