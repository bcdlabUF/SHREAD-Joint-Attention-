"""Stationarity checks for the Granger analysis.
1) ADF (H0: unit root) and KPSS (H0: level-stationary) tests on each 200-ms book-looking series.
2) Granger re-estimated with exogenous controls for slow changes: linear + quadratic session time and
   page-state indicators (individual, category, none, break/page turn), entered in both models."""
import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.linear_model import LogisticRegression
import pingouin as pg
S=pickle.load(open('series.pkl','rb')); c=pd.read_csv('../data/covariates.csv'); age=dict(zip(c.dyad_id,c.age_group))
l=pd.read_csv('../data/look_spans.csv'); FS=30; B=6; L=10; NSH=20
def binarize(x): n=len(x)//B; return (x[:n*B].reshape(n,B).mean(1)>.5).astype(float)
def lagmat(x): return np.column_stack([x[L-k:len(x)-k] for k in range(1,L+1)])
def ll(X,y):
    if y.min()==y.max(): return 0.0
    m=LogisticRegression(C=1e4,max_iter=3000).fit(X,y); p=np.clip(m.predict_proba(X)[:,1],1e-9,1-1e-9)
    return float(np.sum(y*np.log(p)+(1-y)*np.log(1-p)))
def gc(t,d,Z):
    y=t[L:]; own=np.hstack([lagmat(t),Z[L:]]); return ll(np.hstack([own,lagmat(d)]),y)-ll(own,y)
rng=np.random.default_rng(11); st=[]; rows=[]
for dd,s in S.items():
    n=s['n']; t0=s['t0']; cond=np.full(n,'break',dtype=object)
    for on,off,v in l[(l.dyad_id==dd)&(l.stream=='condition')][['onset_ms','offset_ms','value']].values:
        cond[max(int((on-t0)*FS/1000),0):min(int(np.ceil((off-t0)*FS/1000)),n)]=v
    nb=n//B; cb=cond[:nb*B].reshape(nb,B)[:,B//2]
    tt=np.linspace(-1,1,nb); Z=np.column_stack([tt,tt**2]+[(cb==v).astype(float) for v in ['individual','category','none']])
    ib=binarize(s['infant_book']); pb=binarize(s['parent_book'])
    for nm,x in [('infant',ib),('parent',pb)]:
        try: a=adfuller(x,autolag='AIC')[1]
        except Exception: a=np.nan
        try: k=kpss(x,regression='c',nlags='auto')[1]
        except Exception: k=np.nan
        st.append(dict(dyad=dd,age=age[dd],who=nm,adf_p=a,kpss_p=k,mean=x.mean()))
    for tgt,drv,lab in [(ib,pb,'parent_to_infant'),(pb,ib,'infant_to_parent')]:
        g=gc(tgt,drv,Z); sh=rng.integers(int(20*FS/B),nb-int(20*FS/B),NSH)
        ch=np.mean([gc(tgt,np.roll(drv,k),Z) for k in sh]); N=nb-L
        rows.append(dict(dyad=dd,age=age[dd],dir=lab,gc=1000*g/N,gc_chance=1000*ch/N))
    print(dd,end=' ',flush=True)
print()
T=pd.DataFrame(st); T.to_csv('granger_stationarity_tests.csv',index=False)
print('ADF: proportion of series rejecting a unit root (p < .05):', (T.adf_p<.05).mean().round(3), ' n =',T.adf_p.notna().sum())
print('KPSS: proportion of series rejecting level-stationarity (p < .05):', (T.kpss_p<.05).mean().round(3))
print(T.assign(adf=T.adf_p<.05,kpss=T.kpss_p<.05).groupby(['who','age'])[['adf','kpss']].mean().round(2))
R=pd.DataFrame(rows); R['gc_exc']=R.gc-R.gc_chance; R.to_csv('granger_detrended.csv',index=False)
print('\nGranger with time trend + page-state controls'); print(R.groupby(['dir','age'])[['gc','gc_chance','gc_exc']].mean().round(2))
R['ag']=R.age.astype(str); print(pg.mixed_anova(R,dv='gc_exc',within='dir',between='ag',subject='dyad').round(4))
for dr in R.dir.unique(): print(dr); print(pg.anova(R[R.dir==dr],dv='gc_exc',between='ag').round(4))
