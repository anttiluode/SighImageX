// Execute the shipped single-file script and exercise its actual control handlers.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const base=path.join(__dirname,'..'),elements=new Map();
const ctx=new Proxy({createImageData:(w,h)=>({data:new Uint8ClampedArray(w*h*4)})},{get:(o,k)=>o[k]??(()=>{})});
function get(id){if(!elements.has(id))elements.set(id,{value:({mode:'noise',model:'adaptive',layer:'0'})[id]??'',getContext:()=>ctx,setAttribute(){}});return elements.get(id)}
const scope=vm.createContext({document:{getElementById:get}});
const run=code=>vm.runInContext(code,scope);
run(fs.readFileSync(path.join(base,'index.html'),'utf8').match(/<script>([\s\S]+?)<\/script>/)[1]);
assert.match(get('readout').textContent,/10 \/ 10/);
get('reveal').onclick();
assert.ok(Number(get('readout').textContent.split(' ').at(-1))<1e-9);
get('phase').onclick();
assert.ok(Number(get('phaseReading').textContent.match(/pass ([\d.e+-]+)\. The/)[1])<1e-11);
get('layer').value='1';get('layer').onchange();
assert.ok(Number(get('readout').textContent.split(' ').at(-1))<1e-8);
get('reset').onclick();get('mode').value='flat';get('layer').value='0';get('mode').onchange();
assert.match(get('readout').textContent,/^1 \/ 10/);
get('inject').onclick();
assert.match(get('readout').textContent,/^2 \/ 10/);
get('model').value='fixed';get('model').onchange();
assert.match(get('intervention').textContent,/0\.00\./);
const reference=JSON.parse(fs.readFileSync(path.join(__dirname,'parity.json'),'utf8'));
const got=run(`(()=>{const img=Float64Array.from({length:P},(_,i)=>.3*Math.cos(i*.15)+.15*Math.sin(i*.7));
return runModel(img,WEIGHTS.adaptive).map(s=>[...s.gains,...[0,7,63,185,1024,4095].map(i=>s.y[i])])})()`);
for(let l=0;l<2;l++)got[l].forEach((v,i)=>assert.ok(Math.abs(v-reference[l][i])<1e-10));
console.log('Both JavaScript layers match Python; reveal, phase, flat field, checkerboard and fixed-writer controls passed.');
