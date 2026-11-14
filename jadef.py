import numpy as np
FS=30
def _runs(x):
    d=np.diff(np.r_[0,(x>0).astype(int),0]); return np.where(d==1)[0],np.where(d==-1)[0]
def yu_ja(ib,pb,fs=FS):
    """Joint attention per Yu, Suanda & Smith (2019): both on the book, bouts > 500 ms, bridging looks away < 300 ms."""
    x=(ib*pb).astype(float).copy(); on,off=_runs(x)
    for a,b in zip(off[:-1],on[1:]):
        if b-a<int(.3*fs): x[a:b]=1
    y=np.zeros_like(x); on,off=_runs(x)
    for a,b in zip(on,off):
        if b-a>int(.5*fs): y[a:b]=1
    return y
def sustained(ib,fs=FS,sec=3):
    """Infant sustained attention: >= 3 s continuous looking (Yu et al., 2019; Ruff & Lawson, 1990)."""
    y=np.zeros_like(ib,dtype=float); on,off=_runs(ib)
    for a,b in zip(on,off):
        if b-a>=sec*fs: y[a:b]=1
    return y
