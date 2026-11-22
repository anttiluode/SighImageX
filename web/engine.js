const SIZE=64, P=SIZE*SIZE, BANDS=10, alpha=.12, bend=.65;
const grid=Array.from({length:P},(_,i)=>basis(lookupIndex(i>>6,i%SIZE,SIZE)));
const mass=new Float64Array(BANDS);
for(const [k,a,b] of grid){mass[k]+=a;mass[k+1]+=b}
function spectrum(x){return fft2(x,SIZE,false)}
function filter(f,curve){
  const re=new Float64Array(P),im=new Float64Array(P);
  grid.forEach(([k,a,b],p)=>{const g=curve[k]*a+curve[k+1]*b;re[p]=g*f.re[p];im[p]=g*f.im[p]});
  return fromSpectrum(re,im,SIZE);
}
function applyLayer(original,x,weights,override){
  const f=spectrum(x), energy=new Float64Array(BANDS);
  grid.forEach(([k,a,b],p)=>{const power=(f.re[p]**2+f.im[p]**2)/P;energy[k]+=a*power;energy[k+1]+=b*power});
  const feature=Array.from(energy,(e,k)=>Math.log1p(e/mass[k]/.01));feature.push(1);
  const gains=override??Array.from({length:BANDS},(_,k)=>{
    const t=feature.reduce((v,z,i)=>v+z*weights[i][k],0);return .05+2.95/(1+Math.exp(-t));
  });
  const z=filter(f,gains);
  return {input:x,gains,y:z.map((v,i)=>alpha*original[i]+(1-alpha)*Math.asinh(bend*v)/bend)};
}
function runModel(x,weights){
  const first=applyLayer(x,x,weights[0]);
  return [first,applyLayer(x,first.y,weights[1])];
}
// Symmetric 10x10 Jacobi eigensolve for the spectral normal equations.
// Discard directions with insufficient support. Return null on ambiguous knots.
function solveSupported(G,b){
  const n=b.length,V=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>+(i===j)));
  const scale=Math.max(...G.map((row,i)=>row[i]),1e-30);
  for(let pass=0;pass<60;pass++){
    let moves=0;
    for(let p=0;p<n;p++)for(let q=p+1;q<n;q++){
      const off=G[p][q];if(Math.abs(off)<1e-15*scale)continue;
      moves++;
      const angle=.5*Math.atan2(2*off,G[q][q]-G[p][p]),c=Math.cos(angle),s=Math.sin(angle);
      const pp=G[p][p],qq=G[q][q];
      for(let j=0;j<n;j++){
        if(j!==p&&j!==q){
          const a=G[j][p],d=G[j][q];G[j][p]=G[p][j]=c*a-s*d;G[j][q]=G[q][j]=s*a+c*d;
        }
        const a=V[j][p],d=V[j][q];V[j][p]=c*a-s*d;V[j][q]=s*a+c*d;
      }
      G[p][p]=c*c*pp-2*c*s*off+s*s*qq;G[q][q]=s*s*pp+2*c*s*off+c*c*qq;G[p][q]=G[q][p]=0;
    }
    if(!moves)break;
  }
  const x=new Float64Array(n),unknown=new Float64Array(n);
  for(let k=0;k<n;k++){
    if(G[k][k]<scale*1e-12){for(let j=0;j<n;j++)unknown[j]+=V[j][k]**2;continue}
    const coeff=b.reduce((v,t,i)=>v+t*V[i][k],0)/G[k][k];
    for(let j=0;j<n;j++)x[j]+=coeff*V[j][k];
  }
  return Array.from(x,(v,j)=>unknown[j]<1e-9?v:null);
}
function readCurve(original,input,output){
  const x=spectrum(input), z=spectrum(output.map((v,i)=>Math.sinh(bend*(v-alpha*original[i])/(1-alpha))/bend));
  const G=Array.from({length:BANDS},()=>new Float64Array(BANDS)),rhs=new Float64Array(BANDS);
  grid.forEach(([k,a,b],i)=>{
    const power=x.re[i]**2+x.im[i]**2, cross=x.re[i]*z.re[i]+x.im[i]*z.im[i];
    G[k][k]+=a*a*power;G[k+1][k+1]+=b*b*power;G[k+1][k]+=a*b*power;G[k][k+1]+=a*b*power;
    rhs[k]+=a*cross;rhs[k+1]+=b*cross;
  });
  return solveSupported(G,rhs);
}
