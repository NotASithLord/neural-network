(() => {
function gpuWork(current, elapsed, frameMS, timestamp, maximum) {
 if(!Number.isFinite(elapsed)||elapsed<=0)return current;
 const target=timestamp?frameMS*.875:Math.min(8,frameMS*.5);
 const ratio=elapsed/target;
 if(ratio>=.9714&&ratio<=1.0286)return current;
 const next=current*Math.max(.5,Math.min(1.25,1/ratio));
 return Math.max(1,Math.min(maximum,ratio<1?Math.ceil(next):Math.floor(next)));
}

const code=`
struct Config { width:u32, seed:u32, pad0:u32, pad1:u32 }
@group(0) @binding(0) var<uniform> p:Config;
@group(0) @binding(1) var<storage,read_write> weights:array<f32>;
@group(0) @binding(2) var<storage,read_write> acts:array<f32>;
@group(0) @binding(3) var<storage,read_write> deltas:array<f32>;
@group(0) @binding(4) var<storage,read_write> display:array<f32>;
fn ni(l:u32)->u32 {return select(p.width,2u,l==0u);}
fn no(l:u32)->u32 {return select(p.width,1u,l==5u);}
fn off(l:u32)->u32 {if(l==0u){return 0u;}return p.width*3u+(l-1u)*p.width*(p.width+1u);}
fn ai(l:u32,b:u32,i:u32)->u32 {return b*(p.width*5u+1u)+l*p.width+i;}
fn hash(v:u32)->f32 {var x=v; x=((x>>16u)^x)*73244475u;x=((x>>16u)^x)*73244475u;return f32((x>>16u)^x)/4294967295.;}
fn sample(b:u32)->vec3f {var v:vec2f;if(b>=64u){let c=b-64u;v=vec2f(f32(c%32u),f32(c/32u))/31.*2.7-1.35;}else{let k=b+p.seed*67u;let a=hash(k*3u+1u)*6.2831853;let r=sqrt(hash(k*3u+2u))*1.35;v=vec2f(cos(a),sin(a))*r;}return vec3f(v,select(0.,1.,dot(v,v)<.5625));}
fn forward(l:u32,i:u32,b:u32){if(i>=no(l)){return;}let n=ni(l);let base=off(l)+i*(n+1u);var sum=weights[base+n];for(var j=0u;j<n;j++){var a=0.;if(l==0u){a=sample(b)[j];}else{a=acts[ai(l-1u,b,j)];}sum+=weights[base+j]*a;}var a=tanh(sum);if(l==5u){a=1./(1.+exp(-clamp(sum,-25.,25.)));}acts[ai(l,b,i)]=a;}
fn backward(l:u32,i:u32,b:u32){if(i>=no(l)){return;}let a=acts[ai(l,b,i)];var delta=a-sample(b).z;if(l<5u){delta=0.;for(var j=0u;j<no(l+1u);j++){delta+=weights[off(l+1u)+j*(p.width+1u)+i]*deltas[ai(l+1u,b,j)];}delta*=1.-a*a;}deltas[ai(l,b,i)]=delta;}
${Array.from({length:6},(_,l)=>`@compute @workgroup_size(64) fn f${l}(@builtin(global_invocation_id) id:vec3u){forward(${l}u,id.x,id.y);}
@compute @workgroup_size(64) fn e${l}(@builtin(global_invocation_id) id:vec3u){forward(${l}u,id.x,id.y+64u);}
@compute @workgroup_size(64) fn b${l}(@builtin(global_invocation_id) id:vec3u){backward(${l}u,id.x,id.y);}`).join('\n')}
@compute @workgroup_size(64) fn update(@builtin(global_invocation_id) id:vec3u){let k=id.x;if(k>=off(5u)+p.width+1u){return;}var l=0u;for(var j=1u;j<6u;j++){if(k>=off(j)){l=j;}}let n=ni(l);let i=(k-off(l))/(n+1u);let j=(k-off(l))%(n+1u);var grad=0.;for(var b=0u;b<64u;b++){var a=1.;if(j<n){if(l==0u){a=sample(b)[j];}else{a=acts[ai(l-1u,b,j)];}}grad+=deltas[ai(l,b,i)]*a;}weights[k]-=.12*grad/64.;}
@compute @workgroup_size(64) fn capture(@builtin(global_invocation_id) id:vec3u){let i=id.x;if(i<1024u){display[i]=acts[ai(5u,i+64u,0u)];}if(i<p.width*5u){display[1024u+i]=acts[i];}}
`;
class NeuralGPU {
 static async create(){if(!navigator.gpu)throw Error('WebGPU is unavailable');const a=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});if(!a)throw Error('No WebGPU adapter');const timestamp=a.features.has('timestamp-query');const d=await a.requestDevice({requiredFeatures:timestamp?['timestamp-query']:[]});const n=new NeuralGPU(d,timestamp);try{await n.init();return n;}catch(e){d.destroy();throw e;}}
 constructor(device,timestamp){Object.assign(this,{device,timestamp,width:256,count:1280,frameMS:1000/60,iterations:1,steps:0,busy:false,disposed:false,field:null,loss:0,accuracy:0,lastEval:0,sampleSerial:0,gpuMS:0,wallMS:0});device.lost.then(()=>{if(!this.disposed)this.error='GPU connection lost';});}
 buffer(size,usage){return this.device.createBuffer({size,usage});}
 async init(){this.core=await SwiftCore.load();this.core.neural_init(this.width);const d=this.device,module=d.createShaderModule({code}),info=await module.getCompilationInfo();const errors=info.messages.filter(x=>x.type==='error');if(errors.length)throw Error(errors.map(x=>x.message).join('\n'));
 this.weightCount=3*this.width+4*this.width*(this.width+1)+this.width+1;
 const storage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST;this.config=this.buffer(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);this.weights=this.buffer(this.weightCount*4,storage);this.acts=this.buffer((this.count+1)*1088*4,storage);this.errors=this.buffer((this.count+1)*64*4,storage);this.output=this.buffer(16384,storage|GPUBufferUsage.COPY_SRC);this.read=this.buffer(16384,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
 const entries=[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},...[1,2,3,4].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}}))];const layout=d.createBindGroupLayout({entries});this.group=d.createBindGroup({layout,entries:[this.config,this.weights,this.acts,this.errors,this.output].map((buffer,binding)=>({binding,resource:{buffer}}))});const pipelineLayout=d.createPipelineLayout({bindGroupLayouts:[layout]});this.pipelines={};for(const entryPoint of [...Array.from({length:6},(_,i)=>['f'+i,'e'+i,'b'+i]).flat(),'update','capture'])this.pipelines[entryPoint]=await d.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}});
 d.queue.writeBuffer(this.weights,0,new Float32Array(this.core.memory.buffer,this.core.neural_weights(),this.weightCount));
 if(this.timestamp){this.query=d.createQuerySet({type:'timestamp',count:2});this.resolve=this.buffer(256,GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC);this.timeRead=this.buffer(16,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);}}
 async train(intensity=1){if(this.busy||this.disposed||this.error)return;this.busy=true;const d=this.device,start=performance.now(),evaluate=!this.field||start-this.lastEval>650,iterations=this.iterations;
 try{d.queue.writeBuffer(this.config,0,new Uint32Array([this.width,this.steps,0,0]));const e=d.createCommandEncoder(),pass=e.beginComputePass(this.timestamp?{timestampWrites:{querySet:this.query,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}}:{});pass.setBindGroup(0,this.group);const run=(name,x,y=1)=>{pass.setPipeline(this.pipelines[name]);pass.dispatchWorkgroups(x,y);};
 for(let k=0;k<iterations;k++){for(let l=0;l<6;l++)run('f'+l,l===5?1:this.width/64,64);for(let l=5;l>=0;l--)run('b'+l,l===5?1:this.width/64,64);run('update',Math.ceil(this.weightCount/64));}
 if(evaluate){for(let l=0;l<6;l++)run('e'+l,l===5?1:this.width/64,1024);run('capture',Math.ceil(Math.max(1024,this.count)/64));}pass.end();if(evaluate)e.copyBufferToBuffer(this.output,0,this.read,0,16384);if(this.timestamp){e.resolveQuerySet(this.query,0,2,this.resolve,0);e.copyBufferToBuffer(this.resolve,0,this.timeRead,0,16);}d.queue.submit([e.finish()]);await d.queue.onSubmittedWorkDone();if(this.disposed)return;this.wallMS=performance.now()-start;this.steps+=iterations;
 if(this.timestamp){await this.timeRead.mapAsync(GPUMapMode.READ);const t=new BigUint64Array(this.timeRead.getMappedRange());this.gpuMS=Number(t[1]-t[0])/1e6;this.timeRead.unmap();}
 if(evaluate){await this.read.mapAsync(GPUMapMode.READ);this.field=new Float32Array(this.read.getMappedRange().slice(0));this.read.unmap();this.lastEval=start;if(!this.field.every(Number.isFinite))throw Error('Training became unstable');new Float32Array(this.core.memory.buffer,this.core.neural_display(),4096).set(this.field);this.core.neural_measure();this.loss=this.core.neural_loss();this.accuracy=this.core.neural_accuracy();}
 if(!evaluate)this.iterations=gpuWork(iterations,this.timestamp?this.gpuMS:this.wallMS,Math.max(this.frameMS,Math.min(this.frameMS*2,this.cadenceMS||this.frameMS)),this.timestamp,128);this.sampleSerial++;if(this.timestamp){const at=performance.now();const gap=this.sampleAt?at-this.sampleAt:this.frameMS;if(gap>0&&gap<this.frameMS*3)this.cadenceMS=(this.cadenceMS||this.frameMS)*.8+gap*.2;window.dispatchEvent(new CustomEvent('neural-gpu-sample',{detail:{mode:'neural',busy:this.gpuMS,interval:this.sampleAt?at-this.sampleAt:16.7}}));this.sampleAt=at;}this.lastSample={gpuMS:this.timestamp?this.gpuMS:null,wallMS:this.wallMS,at:performance.now()};
 }catch(e){if(!this.disposed){this.error=e.message;console.error('Neural WebGPU:',e);}}finally{this.busy=false;if(!this.disposed&&!this.error&&!document.hidden&&performance.now()<this.activeUntil){clearTimeout(this.nextWork);this.nextWork=setTimeout(()=>{if(performance.now()<this.activeUntil&&!document.hidden)this.train(intensity);},Math.max(0,start+this.frameMS-performance.now()));}}}
 draw(ctx,dt,intensity){this.activeUntil=dt?performance.now()+100:0;if(dt){if(dt>=.004&&dt<.025)this.frameMS=Math.min(this.frameMS,dt*1000);this.train(intensity);}const w=ctx.canvas.width,h=ctx.canvas.height;ctx.fillStyle='#101110';ctx.fillRect(0,0,w,h);ctx.save();const scale=Math.min(w/900,h/540);ctx.translate((w-900*scale)/2,(h-540*scale)/2);ctx.scale(scale,scale);const label=(t,x,y,size=12,color='#93958a')=>{ctx.fillStyle=color;ctx.font=size+'px monospace';ctx.fillText(t,x,y);};
 this.orbit??=new NeuralView.Orbit();const width=this.width||256,cols=Math.ceil(Math.sqrt(width)),project=(x,y,z)=>this.orbit.project(x,y,z,292,265);
 const layers=Array.from({length:7},(_,l)=>Array.from({length:l===0?2:l===6?1:width},(_,i)=>({...project((l-3)*73,l===0?(i-.5)*50:l===6?0:(i%cols-(cols-1)/2)*9,l===0||l===6?0:(Math.floor(i/cols)-(Math.ceil(width/cols)-1)/2)*9),l,i})));
 // Three bijective routes per hidden-layer neuron: every source and destination
 // participates, without the aliasing stripes of fixed-index subsampling.
 const edges=(a,b,l)=>{const out=[];if(l===0||l===5){for(let i=0;i<a.length;i++)for(let j=0;j<b.length;j++)out.push([a[i],b[j]]);}else{for(let i=0;i<a.length;i++)for(let k=0;k<3;k++)out.push([a[i],b[(i*(37+l*2)+k*83+l*29)%b.length]]);}return out;};
 const mx=this.pointer?(this.pointer.x-(w-900*scale)/2)/scale:-9999,my=this.pointer?(this.pointer.y-(h-540*scale)/2)/scale:-9999;
 let focused=null,distance=8;for(const p of layers.flat()){const d=Math.hypot(p.x-mx,p.y-my);if(d<distance){focused=p;distance=d;}}
 ctx.lineWidth=.35;ctx.strokeStyle=focused?'#f2f0e904':'#f2f0e90c';ctx.beginPath();
 for(let l=0;l<6;l++)for(const [a,b] of edges(layers[l],layers[l+1],l)){ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);}ctx.stroke();
 if(focused){ctx.lineWidth=.65;ctx.strokeStyle='#f2f0e955';ctx.beginPath();for(const l of [focused.l-1,focused.l+1]){if(!layers[l])continue;for(const p of layers[l]){ctx.moveTo(focused.x,focused.y);ctx.lineTo(p.x,p.y);}}ctx.stroke();}

 layers.flat().sort((a,b)=>b.z-a.z).forEach(p=>{const hidden=p.l>0&&p.l<6,v=hidden?(this.field?.[1024+(p.l-1)*width+p.i]||0):1;ctx.globalAlpha=hidden?.35+Math.abs(v)*.65:1;ctx.fillStyle=hidden&&v>0?'#e61c1b':'#f2f0e9';ctx.beginPath();ctx.arc(p.x,p.y,(hidden?1.6:4)*p.p,0,Math.PI*2);ctx.fill();});ctx.globalAlpha=1;
 if(focused){ctx.strokeStyle='#e61c1b';ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(focused.x,focused.y,5,0,Math.PI*2);ctx.stroke();}
 label(focused?'LAYER '+(focused.l+1)+' / NEURON '+(focused.i+1)+' / '+((layers[focused.l-1]?.length||0)+(layers[focused.l+1]?.length||0))+' CONNECTIONS':'2 → '+Array(5).fill(width).join(' → ')+' → 1',28,417,11);label('7 LAYERS / '+(width*5+3).toLocaleString()+' NEURONS',28,437,10);
 const side=235;for(let i=0;i<1024;i++){const p=this.field?.[i]??.5;ctx.fillStyle='rgb('+Math.round(30+190*p)+','+Math.round(38-10*p)+','+Math.round(37-10*p)+')';ctx.fillRect(625+(i%32)*side/32,148+Math.floor(i/32)*side/32,side/32+1,side/32+1);}ctx.strokeStyle='#f2f0e97a';ctx.lineWidth=1;ctx.setLineDash([3,4]);ctx.beginPath();ctx.arc(742.5,265.5,side*.75/2.7,0,Math.PI*2);ctx.stroke();ctx.setLineDash([]);label('LEARNED BOUNDARY / TARGET',625,417,10);
 label(this.field?(this.accuracy*100).toFixed(1)+'%':'WARMING UP',28,473,25,'#f2f0e9');label('TEST GRID ACCURACY',28,497,10);label('LOSS '+this.loss.toFixed(3),300,473,21,'#f2f0e9');label((this.steps*(this.cpu?1:64)).toLocaleString()+' SAMPLES TRAINED',300,497,10);label(this.timestamp?this.gpuMS.toFixed(2)+' ms GPU':this.wallMS.toFixed(2)+(this.cpu?' ms CPU':' ms QUEUE'),620,473,21,'#f2f0e9');label(this.cpu?'SWIFT / WASM':this.iterations+' BATCHES / SUBMISSION',620,497,10);if(this.error)label('GPU STOPPED / RELOAD TO RETRY',28,525,11,'#e61c1b');ctx.restore();return `${width*5+3} neurons / 7 layers / ${this.steps*(this.cpu?1:64)} samples / ${this.cpu?'Swift WASM':'WebGPU'} training / accuracy ${(this.accuracy*100).toFixed(1)}% / loss ${this.loss.toFixed(3)}`;}
 destroy(){this.disposed=true;this.device.destroy();}
}
globalThis.NeuralGPU=NeuralGPU;
})();
