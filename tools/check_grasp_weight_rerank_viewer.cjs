/* Exercise real generated data and controls; no robot, server or WebGL. */
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const page = fs.readFileSync(process.argv[2] || 'artifacts/grasp_weight_rerank/comparison.html', 'utf8');
const script = page.slice(page.lastIndexOf('<script>')+8).split('</script>')[0];
class Node {
  constructor(){this.options=[];this.value='';this.checked=false;this.innerHTML='';this.textContent='';}
  replaceChildren(...options){this.options=options;this.value=options[0]?.value||'';}
  on(){}
}
const nodes = new Map(), get = id => {if(!nodes.has(id))nodes.set(id,new Node());return nodes.get(id);};
let renders=0;
const sandbox={document:{getElementById:get},Option:function(text,value){this.text=text;this.value=String(value);},
 Plotly:{react(id,traces){for(const trace of traces){for(const axis of ['x','y','z'])assert.ok(trace[axis].every(Number.isFinite));}renders++;return Promise.resolve();},relayout(){return Promise.resolve();}}};
vm.createContext(sandbox);vm.runInContext(script,sandbox,{timeout:30000});
const keys=vm.runInContext('DATA.cases.map(c=>c.key)',sandbox);
for(const key of keys){get('case').value=key;vm.runInContext('changeCase()',sandbox);
 for(const policy of ['10','40','70']){const pick=get('pick-'+policy);for(const option of pick.options){pick.value=option.value;vm.runInContext(`drawPanel('${policy}')`,sandbox);}
  assert.ok(get('detail-'+policy).innerHTML.includes('候选')||get('detail-'+policy).textContent.includes('没有通过'));}
 for(const target of [false,true]){get('target-only').checked=target;get('prediction').checked=!target;vm.runInContext('redraw()',sandbox);}}
console.log(`Passed: ${keys.length} pools, every rank, environment/prediction toggles, ${renders} finite-geometry renders.`);
