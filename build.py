from pathlib import Path
ROOT=Path(__file__).parent
def read(path):
    return (ROOT/path).read_text(encoding='utf-8')
html=read('web/template.html').replace('/*WEIGHTS*/null',read('results/weights.json'))
html=html.replace('/*ENGINE*/',read('web/fft.js')+'\n'+read('web/engine.js'))
(ROOT/'index.html').write_text(html.replace('/*APP*/',read('web/app.js')),encoding='utf-8')
