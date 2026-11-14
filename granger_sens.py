import sys, pickle, numpy as np, pandas as pd, warnings, pingouin as pg; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from scipy import stats
S=pickle.load(open('series.pkl','rb')); c=pd.read_csv('../data/covariates.csv'); age=dict(zip(c.dyad_id,c.age_group))
FS=30
for B,L in [(6,5),(6,15),(3,20)]:
    def binarize(x):
        n=len(x)//B; return (x[:n*B].reshape(n,B).mean(1)>.5).astype(float)
    def lagmat(x): return np.column_stack([x[L-k:len(x)-k] for k in range(1,L+1)])
    def ll(X,y):
        if y.min()==y.max(): return 0.0
        p=np.clip(LogisticRegression(C=1e4,max_iter=2000).fit(X,y).predict_proba(X)[:,1],1e-9,1-1e-9); return float(np.sum(y*np.log(p)+(1-y)*np.log(1-p)))
    def gc(t,d):
        y=t[L:]; o=lagmat(t); return (ll(np.hstack([o,lagmat(d)]),y)-ll(o,y))/len(y)*1000
    rng=np.random.default_rng(11); rows=[]
    for dd,s in S.items():
        ib=binarize(s['infant_book']); pb=binarize(s['parent_book']); n=len(ib); sh=rng.integers(int(20*FS/B),n-int(20*FS/B),10)
        for t,d,lab in [(ib,pb,'P->I'),(pb,ib,'I->P')]:
            rows.append(dict(dyad=dd,age=age[dd],dir=lab,exc=gc(t,d)-np.mean([gc(t,np.roll(d,k)) for k in sh])))
    R=pd.DataFrame(rows); R['ag']=R.age.astype(str)
    mx=pg.mixed_anova(R,dv='exc',within='dir',between='ag',subject='dyad'); r=mx[mx.Source=='Interaction'].iloc[0]
    print(f'bin {B*1000//FS} ms, history {B*L/FS:.1f} s: dir x age F={r.F:.2f} p={r.p_unc:.4f}', R.groupby(['dir','age']).exc.mean().round(2).unstack().to_dict('index'))
