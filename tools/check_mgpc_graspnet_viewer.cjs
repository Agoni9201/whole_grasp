/* Exercise generated page controls against real audit payload without WebGL. */
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const page = fs.readFileSync(process.argv[2] || 'artifacts/mgpc_graspnet_compare/comparison.html', 'utf8');
const script = page.slice(page.lastIndexOf('<script>') + 8).split('</script>')[0];
class Select {
  constructor() { this.options = []; this._value = ''; this.checked = true; this.textContent = ''; }
  replaceChildren(...options) { this.options = options; this._value = options.length ? options[0].value : ''; }
  add(option) { this.options.push(option); if (this.options.length === 1) this._value = option.value; }
  set value(value) { this._value = String(value); }
  get value() { return this._value; }
}
const nodes = new Map();
let rendered = 0;
const plot = { layout: {}, on() {} };
const get = id => id === 'grasp-view' ? plot : (nodes.has(id) ? nodes.get(id) : (nodes.set(id, new Select()), nodes.get(id)));
get('safety').checked = false;
get('target-only').checked = false;
const sandbox = {
  document: { getElementById: get },
  window: { addEventListener() {}, GRASP_READY: { then(callback) { callback(); return { catch() {} }; } } },
  Option: function Option(text, value) { this.text = text; this.value = String(value); },
  Plotly: {
    react(_plot, traces) {
      assert.ok(traces.length >= 4);
      for (const trace of traces) {
        for (const axis of ['x', 'y', 'z']) {
          assert.ok(trace[axis].flat().every(Number.isFinite), `Non-finite ${axis} in ${trace.name}`);
        }
      }
      rendered++;
      return Promise.resolve();
    },
    relayout() { return Promise.resolve(); },
    Plots: { resize() {} },
  },
};
vm.createContext(sandbox);
vm.runInContext(script, sandbox, { timeout: 20000 });
for (const seed of ['0', '1', '2']) {
  get('seed').value = seed;
  for (const filter of ['all', 'network_collision', 'workflow_failed', 'workflow_pass', 'target_empty']) {
    get('filter').value = filter;
    vm.runInContext('refreshOptions()', sandbox, { timeout: 20000 });
    for (const letter of ['A', 'B', 'C', 'D']) {
      const select = get(`candidate-${letter}`);
      if (select.options[0].value !== '') {
        select.value = select.options.at(-1).value;
        vm.runInContext(`selectCandidate('${letter}')`, sandbox, { timeout: 20000 });
        assert.ok(get(`detail-${letter}`).textContent.includes('raw_id'));
      }
    }
  }
}
console.log(`Viewer controls passed: 3 seeds, 5 result filters, last candidate selections; ${rendered} finite-geometry renders.`);
