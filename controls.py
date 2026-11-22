"""Classical two-filter baseline and a held-out photographic check."""
import numpy as np
from model import N,B,fft,inverse
from data import batch


class TwoFilter:
    def __init__(self):
        x,y=batch(np.random.default_rng(600),64)
        f,t=fft(x),fft(y)
        self.curves=[]
        for F,T in zip(np.split(f,2),np.split(t,2)):
            gram=B.T@((abs(F)**2).sum(0)[:,None]*B)
            rhs=B.T@((F.conj()*T).real.sum(0))
            self.curves.append(np.linalg.lstsq(gram,rhs,rcond=1e-12)[0])
        # Named corruption labels fit two specialists. A scalar threshold selects
        # the expert at inference; choose threshold on new labelled image pairs.
        x,y=batch(np.random.default_rng(601),32)
        f=fft(x)
        high=self.high_energy(f)
        errors=np.array([np.mean((inverse(f*(B@g))-y)**2,axis=(1,2)) for g in self.curves])
        levels=np.unique(high)
        thresholds=np.r_[-np.inf,(levels[:-1]+levels[1:])/2,np.inf]
        self.threshold=min(thresholds,key=lambda t:np.where(high>t,errors[0],errors[1]).mean())

    @staticmethod
    def high_energy(f):
        return np.mean(abs(f[:,B[:,5:].sum(1)>.5])**2,axis=1)

    def predict(self,x):
        f=fft(x)
        g=np.array(self.curves)[(self.high_energy(f)<=self.threshold).astype(int)]
        return inverse(f*(g@B.T))
