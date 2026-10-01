(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('scope-data').textContent);
  const records = data.records;
  const $ = id => document.getElementById('scope-' + id);
  const positions = [...document.querySelectorAll('[data-sourcepos]')].flatMap(el => {
    const m = /^(\d+):(\d+):(\d+)-(\d+):(\d+):(\d+)$/.exec(el.dataset.sourcepos);
    return m && Number(m[1]) === data.file ? [{el, line:Number(m[2])}] : [];
  });
  const lineElements = new Map();
  positions.forEach(({el,line}) => {
    if (!lineElements.has(line)) lineElements.set(line, []);
    lineElements.get(line).push(el);
  });
  records.forEach(r => {
    r.elements = [];
    for (let line=r.lo; line<=r.hi; line++) r.elements.push(...(lineElements.get(line)||[]));
    r.search = (r.kind+' '+r.excerpt+' '+JSON.stringify(r.detail)).toLowerCase();
  });
  let filtered=[], selected=null, painted=new Set();
  function element(tag, text, parent) {
    const el=document.createElement(tag); el.textContent=text; parent.append(el); return el;
  }
  function paint() {
    painted.forEach(el=>el.classList.remove('scope-hit','scope-selected'));painted.clear();
    if ($('highlight').checked) filtered.forEach(r=>r.elements.forEach(el=>{el.classList.add('scope-hit');painted.add(el);}));
    if (selected) selected.elements.forEach(el=>{el.classList.add('scope-selected');painted.add(el);});
  }
  function jump() { if(selected?.elements.length) selected.elements[0].scrollIntoView({block:'center',behavior:'instant'}); }
  function select(r, scroll=false) {
    selected=r; $('list').value=r?.id||''; $('detail').replaceChildren();
    $('jump').disabled=!r?.elements.length;
    if(r) {
      element('h3',r.kind+' · L'+r.lo+'–'+r.hi,$('detail'));
      element('p',r.elements.length ? r.elements.length+' rendered source positions' : 'Unanchored: no rendered element starts within these source lines.',$('detail'));
      element('p','Artifact: '+r.artifact,$('detail'));
      const source=element('details','',$('detail'));source.open=true;
      element('summary','Source excerpt',source);element('pre',r.excerpt,source);
      if(r.detail.tip) element('p',r.detail.tip,$('detail'));
      if(r.detail.nodes) {
        element('p',r.detail.nodes.length+' nodes · '+(r.detail.edges||[]).length+' edges',$('detail'));
        const claims=element('details','',$('detail'));claims.open=true;
        element('summary','Argument nodes (model annotations)',claims);
        r.detail.nodes.forEach(n=>element('p',`${n.id} · ${n.kind}: ${n.gloss||n.text||''}`,claims));
        const edges=element('details','',$('detail'));edges.open=true;
        element('summary','Inference links',edges);
        (r.detail.edges||[]).forEach(e=>element('p',`${[].concat(e.premise||[]).join(', ')} → ${e.conclusion||'?'} · ${e.relation||e.kind}. Warrant: ${e.warrant?.text||e.warrant?.kind||'not recorded'}`,edges));
      }
      if(r.detail['slot-fill']) Object.entries(r.detail['slot-fill']).forEach(([key,value])=>element('p',`${key}: ${typeof value==='string'?value:JSON.stringify(value)}`,$('detail')));
      const detail=element('details','',$('detail'));
      element('summary','Recorded annotation / graph',detail);element('pre',JSON.stringify(r.detail,null,2),detail);
    }
    paint();if(scroll)jump();
  }
  function filter() {
    const query=$('search').value.toLowerCase();
    filtered=records.filter(r=>r.layer===$('layer').value && ($('kind').value==='*'||r.kind===$('kind').value) && r.search.includes(query));
    filtered.sort((a,b)=>a.lo-b.lo||a.hi-b.hi);
    $('list').replaceChildren();
    filtered.forEach(r=>{const opt=element('option',`L${r.lo}–${r.hi} · ${r.kind}${r.elements.length?'':' · unanchored'}`,$('list'));opt.value=r.id;});
    $('count').textContent=`${filtered.length} matching · ${filtered.filter(r=>!r.elements.length).length} unanchored · ${records.length} total annotations`;
    select(filtered.includes(selected)?selected:filtered[0]);
  }
  function kinds(initial=false) {
    const counts=new Map();records.filter(r=>r.layer===$('layer').value).forEach(r=>counts.set(r.kind,(counts.get(r.kind)||0)+1));
    $('kind').replaceChildren();const all=element('option','All kinds',$('kind'));all.value='*';
    [...counts].sort().forEach(([k,n])=>{const o=element('option',`${k} (${n})`,$('kind'));o.value=k;});
    if(initial && counts.has('env/proof'))$('kind').value='env/proof';filter();
  }
  $('layer').addEventListener('change',()=>kinds());$('kind').addEventListener('change',filter);$('search').addEventListener('input',filter);
  $('highlight').addEventListener('change',paint);$('jump').addEventListener('click',jump);
  $('list').addEventListener('change',()=>select(filtered.find(r=>r.id===$('list').value),true));
  for(const [name,delta] of [['prev',-1],['next',1]])$(''+name).addEventListener('click',()=>{
    if(filtered.length)select(filtered[(filtered.indexOf(selected)+delta+filtered.length)%filtered.length],true);
  });
  document.addEventListener('click',event=>{
    if(event.target.closest('#scope-panel'))return;
    const hit=event.target.closest('[data-sourcepos]');if(!hit)return;
    const matches=filtered.filter(r=>r.elements.includes(hit));
    if(matches.length)select(matches[(matches.indexOf(selected)+1)%matches.length]);
  });
  kinds(true);
})();
