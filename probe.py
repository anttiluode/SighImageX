"""Read boundary images and the public layer convention. No writer weights are accepted."""
import numpy as np
from model import ALPHA, BEND, B, fft, inverse


def recover(original, entering, leaving, relative_cutoff=1e-9):
    # Undo fresh-image skip and the invertible bend. Weighted real least squares
    # avoids division by tiny individual Fourier coefficients.
    target = np.sinh(BEND*(leaving-ALPHA*original)/(1-ALPHA))/BEND
    f, out = fft(entering[None])[0], fft(target[None])[0]
    design = np.vstack((f.real[:,None]*B, f.imag[:,None]*B))
    rhs = np.r_[out.real,out.imag]
    U,s,V = np.linalg.svd(design,full_matrices=False)
    keep = s > relative_cutoff*max(s[0],1e-14)
    g = V[keep].T@((U[:,keep].T@rhs)/s[keep])
    identified = (V[~keep]**2).sum(axis=0)<1e-10
    replay = inverse((f*(B@g))[None])[0]
    return g, identified, float(np.sqrt(np.mean((replay-target)**2)))
