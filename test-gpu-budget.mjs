import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const source=fs.readFileSync(new URL('Browser/neural-gpu.js',import.meta.url),'utf8');
const match=source.match(/function gpuWork[\s\S]*?\n}/);
const adapt=vm.runInNewContext('('+match[0]+')');
for(const frame of [1000/60,1000/120]) {
 assert.equal(adapt(100,frame*.875,frame,true,8192),100);
 assert(adapt(100,1,frame,true,8192)>100);
 assert(adapt(100,40,frame,true,8192)<100);
 assert.equal(adapt(100,NaN,frame,true,8192),100);
}
assert.equal(adapt(1,100,16.7,true,8192),1);
assert.equal(adapt(8192,1,16.7,true,8192),8192);
console.log('Adaptive GPU budget: ramp, target, overload, safety cap and invalid samples passed');
