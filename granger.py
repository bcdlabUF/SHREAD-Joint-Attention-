"""Granger causality between parent and infant book looking.
Series: 30-Hz binary book looking, binned to 200 ms (state = looking for > half the bin).
For each dyad and direction, a logistic autoregression predicts the target's state from its own past 2 s (10 lags)
with vs. without the driver's past 2 s (10 lags). Granger strength = log-likelihood gain per bin (x1000);
LR test chi2(10). Chance = same statistic with the driver circularly shifted >= 20 s (20 shifts)."""
import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from scipy import stats
S=pickle.load(open('series.pkl','rb')); c=pd.read_csv('../data/covariates.csv'); age=dict(zip(c.dyad_id,c.age_group))
FS=30; B=6; L=10; NSH=20
def binarize(x):
    n=len(x)//B; return (x[:n*B].reshape(n,B).mean(1)>.5).astype(float)
def lagmat(x):
    return np.column_stack([x[L-k:len(x)-k] for k in range(1,L+1)])
def ll(X,y):
    if y.min()==y.max(): return 0.0
    m=LogisticRegression(C=1e4,max_iter=2000).fit(X,y); p=np.clip(m.predict_proba(X)[:,1],1e-9,1-1e-9)
    return float(np.sum(y*np.log(p)+(1-y)*np.log(1-p)))
def gc(target,driver):
    y=target[L:]; own=lagmat(target); oth=lagmat(driver)
    l0=ll(own,y); l1=ll(np.hstack([own,oth]),y); return l1-l0, len(y)
rng=np.random.default_rng(11); rows=[]
for d,s in S.items():
    ib=binarize(s['infant_book']); pb=binarize(s['parent_book']); n=len(ib)
    for tgt,drv,lab in [(ib,pb,'parent_to_infant'),(pb,ib,'infant_to_parent')]:
        g,N=gc(tgt,drv)
        sh=rng.integers(int(20*FS/B),n-int(20*FS/B),NSH)
        ch=[gc(tgt,np.roll(drv,k))[0] for k in sh]
        rows.append(dict(dyad=d,age=age[d],dir=lab,N=N,dLL=g,LR=2*g,p=stats.chi2.sf(2*g,L),gc=1000*g/N,gc_chance=1000*np.mean(ch)/N))
    print(d,end=' ',flush=True)
R=pd.DataFrame(rows); R['gc_exc']=R.gc-R.gc_chance; R.to_csv('granger_dyad.csv',index=False); print()
print(R.groupby(['dir','age'])[['gc','gc_chance','gc_exc']].mean().round(3))
print(R.groupby(['dir'])[['gc','gc_chance']].mean().round(3))
print('proportion of dyads with significant LR (p<.05):'); print(R.assign(sig=R.p<.05).groupby(['dir','age']).sig.mean().round(2))
