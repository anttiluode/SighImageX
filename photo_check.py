"""One outside-distribution photograph. Requires Matplotlib sample data and Pillow."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from matplotlib import get_data_path
from controls import TwoFilter
from model import N,forward
from data import corrupt

ROOT=Path(__file__).parent
weights={k:np.array(v) for k,v in json.loads((ROOT/'results/weights.json').read_text()).items()}
source=Path(get_data_path())/'sample_data/grace_hopper.jpg'
with Image.open(source) as im:
    a=np.asarray(im.convert('L'),dtype=float)/127.5-1
h,w=a.shape
patches=np.stack([a[i:i+N,j:j+N] for i in range(0,h-N+1,N) for j in range(0,w-N+1,N)])
receipt={'source':'Matplotlib sample_data/grace_hopper.jpg','nonoverlapping_patches':len(patches),'results':{}}
classic=TwoFilter()
rng=np.random.default_rng(3501)
for mode in ['noise','blur']:
    x=corrupt(patches,rng,mode)
    predictions={k:forward(x,v)[0] for k,v in weights.items()}
    predictions.update(input=x,two_filter=classic.predict(x))
    receipt['results'][mode]={k:float(np.mean((v-patches)**2)) for k,v in predictions.items()}
(ROOT/'results/photo.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(receipt)
