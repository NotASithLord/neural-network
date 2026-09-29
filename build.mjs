import {spawnSync} from 'node:child_process';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const root=path.dirname(fileURLToPath(import.meta.url));
const bin=process.env.SWIFT_BIN;
const sdk=process.env.WASI_SYSROOT;
if(!bin||!sdk)throw Error('Set SWIFT_BIN to a Swift 6.4 toolchain usr/bin and WASI_SYSROOT to the matching SDK WASI.sdk directory.');
function run(command,args){const r=spawnSync(command,args,{cwd:root,stdio:'inherit'});if(r.status!==0)throw Error(command+' failed');}
run(path.join(bin,'swiftc'),['-target','wasm32-unknown-none-wasm','-enable-experimental-feature','Embedded','-enable-experimental-feature','Extern','-wmo','-parse-as-library','-O','-c','Sources/Math.swift','Sources/Neural.swift','-o','core.o']);
const names=["neural_init","neural_step","neural_evaluate","neural_measure","neural_adapt","neural_weights","neural_display","neural_loss","neural_accuracy"];
run(path.join(bin,'wasm-ld'),['--no-entry','--strip-all','--export-memory','--initial-memory=2097152','--max-memory=8388608','-z','stack-size=65536',...names.map(n=>'--export='+n),'core.o','-L'+path.join(sdk,'lib/wasm32-wasip1'),'-lc','-lm',path.resolve(sdk,'../swift.xctoolchain/usr/lib/swift/clang/lib/wasip1/libclang_rt.builtins-wasm32.a'),'-o','core.wasm']);
const bytes=readFileSync(path.join(root,'core.wasm')),module=new WebAssembly.Module(bytes);
console.log('Swift core:',bytes.length,'bytes. Imports:',WebAssembly.Module.imports(module));
