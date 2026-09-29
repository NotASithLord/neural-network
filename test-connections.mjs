import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const source=fs.readFileSync(new URL('Browser/neural-gpu.js',import.meta.url),'utf8');
const expression=source.match(/const edges=(.*?);\n/)[1];
const edges=vm.runInNewContext(expression);
for(const width of [64,256]) {
 for(let l=0;l<6;l++) {
  const a=Array.from({length:l===0?2:width},(_,i)=>({i}));
  const b=Array.from({length:l===5?1:width},(_,i)=>({i}));
  const pairs=edges(a,b,l);
  assert.equal(new Set(pairs.map(p=>p[0])).size,a.length);
  assert.equal(new Set(pairs.map(p=>p[1])).size,b.length);
  assert.equal(new Set(pairs.map(([a,b])=>`${a.i}:${b.i}`)).size,pairs.length);
  assert.equal(pairs.length,l===0||l===5?a.length*b.length:width*3);
 }
}
console.log('Every neuron participates; no duplicate or out-of-layer connections');
