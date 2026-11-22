"""Two differentiable Sigh blocks; all trainable parameters live in the two EQ writers."""
import numpy as np

N, K, L = 64, 10, 2
ALPHA, BEND, FLOOR = .12, .65, .01
axis = np.fft.fftfreq(N)*N
radius2 = axis[:, None]**2+axis**2
u = np.floor(255*radius2/(N*N/2))*9/255
B = np.maximum(0., 1.-np.abs(u[..., None]-np.arange(K))).reshape(N*N, K)
MASS = B.sum(0)


def fft(x):
    return np.fft.fft2(x, axes=(-2,-1), norm='ortho').reshape(len(x), -1)


def inverse(f):
    return np.fft.ifft2(f.reshape(-1,N,N), axes=(-2,-1), norm='ortho').real


def new_model():
    # Gain starts at 1 for both dynamic and fixed controls.
    w = np.zeros((L,K+1,K))
    w[:, -1] = np.log(.95/2.)
    return w


def gains_from_logits(t):
    return .05+2.95/(1+np.exp(-np.clip(t,-50,50)))


def block(x0, x, w, frozen=None):
    f = fft(x)
    energy = np.abs(f)**2 @ B / MASS
    features = np.column_stack((np.log1p(energy/FLOOR), np.ones(len(x))))
    g = gains_from_logits(features@w) if frozen is None else frozen
    transfer = g@B.T
    z = inverse(f*transfer)
    y = ALPHA*x0+(1-ALPHA)*np.arcsinh(BEND*z)/BEND
    return y, (f,energy,features,g,transfer,z)


def forward(x, w):
    y, states = x, []
    for layer in w:
        y, state = block(x,y,layer)
        states.append(state)
    return y, states


def loss_gradient(x, target, w):
    y, states = forward(x,w)
    loss = np.mean((y-target)**2)
    dy = 2*(y-target)/y.size
    grad = np.empty_like(w)
    for i in reversed(range(L)):
        f,energy,features,g,transfer,z = states[i]
        dz = dy*(1-ALPHA)/np.sqrt(1+(BEND*z)**2)
        df = fft(dz)
        dg = (df.conj()*f).real @ B
        dlogit = dg*(g-.05)*(3.-g)/2.95
        grad[i] = features.T @ dlogit
        denergy = (dlogit @ w[i,:K].T)/(energy+FLOOR)
        dy = inverse(df*transfer+2*f*((denergy/MASS)@B.T))
    return loss, grad
