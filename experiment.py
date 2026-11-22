"""Run training, held-out restoration, and independent layer-boundary assays."""
import json
from pathlib import Path
import numpy as np
from model import N,K,L,B,MASS,ALPHA,BEND,FLOOR,fft,inverse,new_model,block,forward,loss_gradient,gains_from_logits
from data import batch, corrupt
from probe import recover
from controls import TwoFilter

ROOT = Path(__file__).parent
OUT = ROOT/'results'
STEPS = 240


def train(seed, adaptive):
    rng = np.random.default_rng(seed)
    w = new_model()
    m,v = np.zeros_like(w), np.zeros_like(w)
    for step in range(1, STEPS+1):
        x,y = batch(rng,4)
        _,g = loss_gradient(x,y,w)
        if not adaptive:
            g[:,:K] = 0.
        m = .9*m+.1*g
        v = .999*v+.001*g*g
        w -= .02*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
    return w


def score(x,y,w,flags):
    pred = x if w is None else forward(x,w)[0]
    return {label:float(np.mean((pred[flags==i]-y[flags==i])**2))
            for i,label in enumerate(['noise','blur'])}


def read_layers(x,w):
    inp=x.copy()
    recovered, keys, features, residuals = [], [], [], []
    for layer in w:
        out,state = block(x,inp,layer)
        rows = [recover(a,b,c) for a,b,c in zip(x,inp,out)]
        if not all(row[1].all() for row in rows):
            raise ValueError('writer fit requires full support in calibration images')
        recovered.append(np.stack([r[0] for r in rows]))
        keys.append(state[3])
        features.append(np.column_stack((np.log1p((np.abs(fft(inp))**2@B/MASS)/FLOOR),np.ones(len(x)))))
        residuals.extend(r[2] for r in rows)
        inp=out
    return recovered,keys,features,max(residuals)


def audit(w, fixed):
    cal,_=batch(np.random.default_rng(811),48)
    test,_=batch(np.random.default_rng(812),16)
    rec,_,features,_=read_layers(cal,w)
    held,keys,feats,res=read_layers(test,w)
    writer_errors, fixed_errors = [], []
    for i in range(L):
        # Infer a candidate writer from previously recovered gains only.
        g=rec[i]
        theta=np.linalg.lstsq(features[i],np.log((g-.05)/(3.-g)),rcond=1e-10)[0]
        writer_errors.append(float(np.sqrt(np.mean((gains_from_logits(feats[i]@theta)-keys[i])**2))))
        fixed_errors.append(float(np.sqrt(np.mean((g.mean(0)-keys[i])**2))))
    # Inject a checkerboard at the Nyquist corner. Evaluate the untouched Fourier
    # modes after undoing the public bend/skip. Compare the actual output with
    # the counterfactual that holds the previous EQ fixed on exactly the same edit.
    checker=(-1.)**np.indices((N,N)).sum(0)
    edited=test+.12*checker
    def other_mode_residue(model):
        _,base=block(test,test,model[0])
        active,_=block(edited,edited,model[0])
        frozen,_=block(edited,edited,model[0],frozen=base[3])
        undo=lambda y:np.sinh(BEND*(y-ALPHA*edited)/(1-ALPHA))/BEND
        df=fft(undo(active)-undo(frozen))
        df[:,(N//2)*N+N//2]=0
        return float(np.sqrt(np.mean(np.abs(df)**2)))
    dark=np.ones((1,N,N))*.2
    dark_out,_=block(dark,dark,w[0])
    _,known,_=recover(dark[0],dark[0],dark_out[0])
    # Real phase changes, equal energies. Recompute the first layer's gains.
    white=fft(np.random.default_rng(42).normal(size=test.shape))
    phase_img=inverse(fft(test)*white/np.maximum(abs(white),1e-14))
    _,p=block(phase_img,phase_img,w[0])
    return {'max_gain_error':float(np.max(abs(np.array(keys)-held))),
            'max_prebend_replay_rmse':res,'writer_test_gain_rmse':writer_errors,
            'mean_curve_test_gain_rmse':fixed_errors,
            'off_carrier_rms':{'adaptive':other_mode_residue(w),'fixed':other_mode_residue(fixed)},
            'constant_image_identified_knots':np.flatnonzero(known).tolist(),
            'phase_scramble_gain_max_change':float(np.max(abs(p[3]-keys[0])))}


def main():
    OUT.mkdir(exist_ok=True)
    x,y=batch(np.random.default_rng(9001),64)
    flags=np.repeat([0,1],len(x)//2)
    receipt={'steps':STEPS,'test_scenes':64,'test_seed':9001,'seeds':{}}
    classical=TwoFilter()
    estimate=classical.predict(x)
    receipt['two_filter']={m:float(np.mean((estimate[flags==i]-y[flags==i])**2))
                           for i,m in enumerate(['noise','blur'])}
    receipt['two_filter_threshold']=classical.threshold
    for seed in [17,29,41]:
        models={name:train(seed,adaptive) for name,adaptive in [('fixed',False),('adaptive',True)]}
        row={name:score(x,y,w,flags) for name,w in {'input':None,**models}.items()}
        # Whole-writer ablation using independent per-layer mean training-distribution gains.
        xc,_=batch(np.random.default_rng(202),32)
        frozen=new_model()
        frozen[:,:K]=0
        for i,st in enumerate(forward(xc,models['adaptive'])[1]):
            g=st[3].mean(0)
            frozen[i,-1]=np.log((g-.05)/(3.-g))
        row['mean_writer']=score(x,y,frozen,flags)
        receipt['seeds'][str(seed)]=row
        print(seed,row,flush=True)
        if seed==17:
            receipt['audit_seed17']=audit(models['adaptive'],models['fixed'])
            (OUT/'weights.json').write_text(json.dumps({k:v.tolist() for k,v in models.items()}))
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(receipt['audit_seed17'])


if __name__=='__main__':
    main()
