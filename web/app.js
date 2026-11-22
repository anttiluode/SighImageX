const $=id=>document.getElementById(id);
function lighthouse(){
  return Float64Array.from({length:P},(_,i)=>{
    const x=i%SIZE,y=i>>6;
    let v=.15+.45*(1-y/64)+.065*Math.sin(x*.18+y*.25);
    if((x-46)**2+(y-12)**2<34)v=.95;
    if(y>32+.06*x+3*Math.sin(x*.15))v=-.14+.10*Math.cos(.6*y+.28*x);
    if(y>51+3*Math.cos(x*.18)-x*.035)v=-.62+.07*Math.sin(x*.8-y);
    if(y>18&&y<53&&Math.abs(x-23)<2+(y-18)*.10)v=(Math.floor(y/6)%2===0)?.77:-.46;
    if(y>=15&&y<=20&&Math.abs(x-23)<4)v=(x%3===0)?-.60:.92;
    if(y>=11&&y<16&&Math.abs(x-23)<(y-11)*1.3)v=-.8;
    if(y===10&&x===23)v=-.8;
    return v;
  });
}
function noise(){
  let seed=859347;
  const rand=()=>{seed=(Math.imul(1664525,seed)+1013904223)>>>0;return (seed+.5)/4294967296};
  return Float64Array.from({length:P},()=>Math.sqrt(-2*Math.log(rand()))*Math.cos(2*Math.PI*rand()));
}
const grain=noise();
let ref=lighthouse(), reveal=false, carrier=false, scrambled=false, loadVersion=0;
// Same Hermitian unit-modulus multiplier for the reference and degraded image.
const phaseKey=spectrum(grain);
function scramble(img){
  const f=spectrum(img);
  for(let i=1;i<P;i++){
    const mag=Math.hypot(phaseKey.re[i],phaseKey.im[i]);
    if(!mag)continue;
    const a=phaseKey.re[i]/mag,b=phaseKey.im[i]/mag, re=f.re[i],im=f.im[i];
    f.re[i]=a*re-b*im;f.im[i]=a*im+b*re;
  }
  return fromSpectrum(f.re,f.im,SIZE);
}
function degraded(img,mode){
  if(mode==='noise')return img.map((v,i)=>v+.31*grain[i]);
  if(mode==='blur'){
    const f=spectrum(img);
    for(let i=0;i<P;i++){
      const x=Math.min(i%SIZE,SIZE-i%SIZE), y=Math.min(i>>6,SIZE-(i>>6));
      const gain=Math.exp(-2*Math.PI**2*(x*x+y*y)/P);
      f.re[i]*=gain;f.im[i]*=gain;
    }
    return fromSpectrum(f.re,f.im,SIZE).map((v,i)=>v+.005*grain[i]);
  }
  return img.slice();
}
function raster(id,img){
  const ctx=$(id).getContext('2d'),d=ctx.createImageData(SIZE,SIZE);
  img.forEach((v,i)=>{const b=Math.max(0,Math.min(255,Math.round((v+1)*127.5)));d.data.set([b,b,b,255],4*i)});
  ctx.putImageData(d,0,0);
}
function chart(g,answer){
  const c=$('graph').getContext('2d'),W=940,H=193, X=k=>40+k*96,Y=v=>153-42*v;
  c.clearRect(0,0,W,H);c.font='15px monospace';c.textAlign='center';
  for(const v of [0,1,2,3]){c.strokeStyle='#405059';c.beginPath();c.moveTo(32,Y(v));c.lineTo(914,Y(v));c.stroke();c.fillStyle='#abbcc0';c.fillText(v,15,Y(v)+5)}
  c.fillStyle='#abbcc0';c.fillText('Low',X(0),183);c.fillText('Squared radial frequency',W/2,183);c.fillText('High',X(9),183);
  if(answer){c.beginPath();c.strokeStyle='#72dfc9';c.lineWidth=2;answer.forEach((v,i)=>i===0?c.moveTo(X(i),Y(v)):c.lineTo(X(i),Y(v)));c.stroke()}
  g.forEach((v,i)=>{c.fillStyle='#ffbd67';if(v===null){c.fillText('×',X(i),Y(0)-4);return}c.beginPath();c.arc(X(i),Y(v),4,0,Math.PI*2);c.fill()});c.lineWidth=1;
}
function mse(a,b){return a.reduce((s,v,i)=>s+(v-b[i])**2,0)/a.length}
function render(){
  const mode=$('mode').value, weights=WEIGHTS[$('model').value],l=+$('layer').value;
  const source=mode==='flat'?new Float64Array(P).fill(.24):ref;
  const degradedInput=degraded(source,mode);
  const edit=a=>carrier?a.map((v,i)=>v+(((i%SIZE+(i>>6))%2)?-.12:.12)):a;
  const raw=edit(degradedInput);
  const clean=scrambled?scramble(source):source, x=scrambled?scramble(raw):raw;
  const base=scrambled?scramble(degradedInput):degradedInput;
  const stages=runModel(x,weights),s=stages[l],y=stages[1].y;
  const estimated=readCurve(x,s.input,s.y);
  raster('source',clean);raster('input',x);raster('output',y);chart(estimated,reveal?s.gains:null);
  $('measure').textContent='Mean squared error to reference: input '+mse(x,clean).toFixed(5)+'  →  restored '+mse(y,clean).toFixed(5);
  const errors=estimated.flatMap((v,i)=>v===null?[]:[Math.abs(v-s.gains[i])]);
  $('readout').textContent=errors.length+' / 10 gains identifiable'+(reveal&&errors.length?' · maximum gain error '+Math.max(...errors).toExponential(2):'');
  $('reveal').textContent=reveal?'Hide writer curve':'Reveal writer curve';$('keylabel').hidden=!reveal;
  $('reveal').setAttribute('aria-pressed',String(reveal));$('inject').setAttribute('aria-pressed',String(carrier));
  $('inject').textContent=carrier?'Remove checkerboard':'Inject checkerboard';
  $('phase').setAttribute('aria-pressed',String(scrambled));
  $('phase').textContent=scrambled?'Restore phase':'Scramble phase';
  if(scrambled){
    const before=runModel(raw,weights)[l];
    const change=Math.max(...s.gains.map((v,i)=>Math.abs(v-before.gains[i])));
    $('phaseReading').textContent='After phase scrambling: maximum gain change at this pass '+change.toPrecision(3)+'. The first writer sees the same band energies.';
  }else $('phaseReading').textContent='Scramble phase to rearrange the picture while preserving its Fourier magnitudes.';
  if(carrier){
    const before=runModel(base,weights)[l];
    const diff=Math.max(...s.gains.map((v,i)=>Math.abs(v-before.gains[i])));
    $('intervention').textContent='Added one Fourier mode to the original input. Largest gain change at this pass: '+diff.toPrecision(3)+'.';
  }else $('intervention').textContent='Inject a checkerboard to change one input frequency and measure how far the curve moves.';
}
for(const k of ['mode','model','layer'])$(k).onchange=render;
$('reveal').onclick=()=>{reveal=!reveal;render()};
$('phase').onclick=()=>{scrambled=!scrambled;render()};
$('inject').onclick=()=>{carrier=!carrier;render()};
$('reset').onclick=()=>{loadVersion++;ref=lighthouse();$('file').value='';$('mode').value='noise';carrier=false;scrambled=false;$('msg').textContent='';render()};
$('file').onchange=async e=>{
  const file=e.target.files[0];if(!file)return;
  const version=++loadVersion;
  try{
    const bitmap=await createImageBitmap(file);
    if(version!==loadVersion){bitmap.close();return}
    const c=document.createElement('canvas');c.width=c.height=SIZE;const ctx=c.getContext('2d');
    const n=Math.min(bitmap.width,bitmap.height);
    ctx.drawImage(bitmap,(bitmap.width-n)/2,(bitmap.height-n)/2,n,n,0,0,SIZE,SIZE);bitmap.close();
    const p=ctx.getImageData(0,0,SIZE,SIZE).data;
    ref=Float64Array.from({length:P},(_,i)=>(.2126*p[4*i]+.7152*p[4*i+1]+.0722*p[4*i+2])/127.5-1);
    $('msg').textContent='Loaded '+file.name+' · center crop.';render();
  }catch{if(version===loadVersion)$('msg').textContent='Could not decode this image. Try a PNG or JPEG.'}
};
render();
