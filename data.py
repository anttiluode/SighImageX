"""Paired synthetic restoration: same scenes, two different degradations."""
import numpy as np
from model import N, axis, radius2, fft, inverse

Y,X = np.mgrid[:N,:N]


def scenes(rng, count):
    result = []
    for _ in range(count):
        img = np.zeros((N,N))
        for _ in range(18):
            x,y = rng.uniform(0,N,2)
            dx,dy = X-x,Y-y
            a,b = rng.uniform(3,19,2)
            shape = ((dx/a)**2+(dy/b)**2<1 if rng.random()<.5
                     else (np.abs(dx)<a)&(np.abs(dy)<b))
            img[shape] = rng.uniform(-.75,.75)
        # Include fine patterns which reward preservation in the blur condition.
        kx,ky = rng.integers(2,18,2)
        img += rng.uniform(.04,.24)*np.cos(2*np.pi*(kx*X+ky*Y)/N+rng.uniform(0,2*np.pi))
        result.append(img)
    return np.stack(result)


def corrupt(clean, rng, mode):
    if mode=='noise':
        return clean+rng.normal(size=clean.shape)*rng.uniform(.20,.40,(len(clean),1,1))
    if mode=='blur':
        sigma = rng.uniform(.65,1.35,(len(clean),1))
        h = np.exp(-2*np.pi**2*sigma**2*radius2.ravel()/N**2)
        return inverse(fft(clean)*h)+rng.normal(0,.005,clean.shape)
    raise ValueError(mode)


def batch(rng, count):
    clean = scenes(rng,count)
    return np.concatenate([corrupt(clean,rng,m) for m in ('noise','blur')]), np.tile(clean,(2,1,1))
