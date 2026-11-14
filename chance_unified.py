"""Circular-shift chance (parent series shifted >= 20 s, 200 shifts per dyad) for joint attention and who-led."""
import pickle,numpy as np,pandas as pd
from jadef import yu_ja,_runs,FS
S=pickle.load(open('series.pkl','rb')); c=pd.read_csv('../data/covariates.csv'); age=dict(zip(c.dyad_id,c.age_group))
def runstart(x):
    d=np.diff(np.r_[0,(x>0).astype(int)]); st=np.full(len(x),-1); cur=-1
    starts=np.where(d==1)[0]; idx=np.zeros(len(x),int); idx[starts]=1; cs=np.cumsum(idx)
    sv=np.r_[ -1, starts][cs]; sv[x<=0]=-1; return sv
def lead_p(ib,pb):
    ja=yu_ja(ib,pb); si,sp=runstart(ib),runstart(pb); on,off=_runs(ja); L=[]
    for a,b in zip(on,off):
        k=a+np.argmax((ib[a:b]*pb[a:b])>0); lag=((k-si[k])-(k-sp[k]))/FS
        if abs(lag)>.1: L.append(lag>0)
    return (np.mean(L) if len(L) else np.nan), len(L), ja.mean()
rng=np.random.default_rng(7); NSH=200; rows=[]
for d,s in S.items():
    ib,pb=s['infant_book'],s['parent_book']; n=len(ib)
    p_obs,n_obs,ja_obs=lead_p(ib,pb)
    sh=rng.integers(20*FS,n-20*FS,NSH)
    res=np.array([lead_p(ib,np.roll(pb,k)) for k in sh])
    rows.append(dict(dyad=d,age=age[d],p_inf_obs=p_obs,n_bouts=n_obs,ja_obs=ja_obs,
                     p_inf_circ=np.nanmean(res[:,0]),ja_circ=res[:,2].mean()))
    print(d,end=' ',flush=True)
R=pd.DataFrame(rows); R.to_csv('chance_unified.csv',index=False); print()
print(R.groupby('age')[['ja_obs','ja_circ','p_inf_obs','p_inf_circ']].mean().round(3))
