"""Plot the same withheld scene under two corruptions using checkpoint 17."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import block,forward
from data import batch
from probe import recover
root=Path(__file__).parent
x,y=batch(np.random.default_rng(9001),64)
w=np.asarray(json.loads((root/'results/weights.json').read_text())['adaptive'])
fig,axes=plt.subplots(2,4,figsize=(10,4.9),width_ratios=[1,1,1,1.45],layout='constrained')
for row,i in enumerate([0,64]):
    source=x[i:i+1]
    out,states=forward(source,w)
    for ax,img in zip(axes[row,:3],[y[i],x[i],out[0]]):
        ax.imshow(img,cmap='gray',vmin=-1,vmax=1,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
    first,_=block(source,source,w[0])
    est,support,_=recover(x[i],x[i],first[0])
    a=axes[row,3]
    a.plot(states[0][3][0],c='#256657',label='Written')
    a.scatter(np.flatnonzero(support),est[support],s=28,facecolors='none',edgecolors='#ad5316',label='Recovered')
    a.axhline(1,ls=':',color='gray',lw=.9);a.set(ylim=(0,3.05),xticks=[0,3,6,9],yticks=[0,1,2,3])
    a.spines[['top','right']].set_visible(False)
    axes[row,0].set_ylabel(['Added noise','Blurred'][row])
for ax,title in zip(axes[0],['Clean reference','Input','Restored','First-pass gain']):
    ax.set_title(title)
axes[1,3].set_xlabel('EQ control (low → high)');axes[0,3].legend(loc='upper right',fontsize=8)
fig.savefig(root/'results/example.png',dpi=140)
