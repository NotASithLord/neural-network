// JavaScript is the browser/Canvas bridge; numerical kernels live in Swift/WASM.
(() => {
let pending;
async function load(){
  if(!pending)pending=(async()=>{
    const response=await fetch(globalThis.NEURAL_WASM_URL||'./core.wasm');
    if(!response.ok)throw Error('Swift core could not be loaded');
    let memory;
    const imports={wasi_snapshot_preview1:{random_get:(ptr,length)=>{
      if(!memory)return 52;
      const bytes=new Uint8Array(memory.buffer,ptr,length);
      for(let i=0;i<length;i+=65536)crypto.getRandomValues(bytes.subarray(i,i+65536));
      return 0;
    }}};
    const {instance}=await WebAssembly.instantiate(await response.arrayBuffer(),imports);
    memory=instance.exports.memory;
    instance.exports._initialize?.();
    return instance.exports;
  })().catch(error=>{pending=null;throw error;});
  return pending;
}
class CPU {
 constructor(){this.width=64;this.count=320;this.steps=0;this.iterations=1;this.field=null;this.loss=0;this.accuracy=0;this.history=[];this.wallMS=0;this.cpu=true;}
 async init(){this.core=await load();this.core.neural_init(this.width);this.evaluate();return this;}
 evaluate(){const c=this.core;c.neural_evaluate();this.field=new Float32Array(c.memory.buffer,c.neural_display(),4096).slice();this.loss=c.neural_loss();this.accuracy=c.neural_accuracy();}
 train(intensity){if(!this.core)return;const start=performance.now();this.core.neural_step(16*intensity);this.steps+=16*intensity;if(!this.lastEval||start-this.lastEval>300){this.evaluate();this.lastEval=start;}this.wallMS=performance.now()-start;}
 draw(...args){return NeuralGPU.prototype.draw.apply(this,args);}
 destroy(){}
}
globalThis.SwiftCore={load,CPU};
})();
