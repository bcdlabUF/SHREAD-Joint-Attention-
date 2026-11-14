import pickle,numpy as np,pandas as pd
from jadef import yu_ja,_runs,FS
S=pickle.load(open('series.pkl','rb')); c=pd.read_csv('../data/covariates.csv'); age=dict(zip(c.dyad_id,c.age_group))
def runstart(x):
    # index of start of run containing each frame
    st=np.full(len(x),-1); cur=-1
    for i,v in enumerate(x):
        if v>0:
            if i==0 or x[i-1]==0: cur=i
            st[i]=cur
    return st
rows=[];bouts=[]
for d,s in S.items():
    ib,pb=s['infant_book'],s['parent_book']; ja=yu_ja(ib,pb)
    si,sp=runstart(ib),runstart(pb); on,off=_runs(ja)
    for a,b in zip(on,off):
        # first frame both on within bout
        k=a+np.argmax((ib[a:b]*pb[a:b])>0)
        li=k-si[k]; lp=k-sp[k]   # how long each had already been on
        lag=(li-lp)/FS          # >0: infant there first
        lead='simul' if abs(lag)<=.1 else ('infant' if lag>0 else 'parent')
        bouts.append(dict(dyad=d,age=age[d],t=a/FS,dur=(b-a)/FS,lag=lag,lead=lead,follow=min(li,lp)/FS*0+abs(lag)))
B=pd.DataFrame(bouts); B.to_csv('lead_bouts.csv',index=False)
print(B.lead.value_counts(normalize=True).round(3)); print(B.groupby('age').lead.value_counts(normalize=True).unstack().round(3))
# per dyad
rng=np.random.default_rng(1)
for d,g in B.groupby('dyad'):
    L=g[g.lead!='simul'].lead.values; n=len(L)
    if n<3: continue
    sw=(L[1:]!=L[:-1]).sum()/(n-1); p=(L=='infant').mean()
    perm=np.mean([(lambda q:(q[1:]!=q[:-1]).mean())(rng.permutation(L)) for _ in range(1000)])
    rows.append(dict(dyad=d,age=age[d],n=n,p_inf=p,switch=sw,switch_chance=perm,follow_med=g[g.lead!='simul'].lag.abs().median(),
        pinf_lag=g[g.lead=='infant'].lag.abs().median(),ppar_lag=g[g.lead=='parent'].lag.abs().median()))
R=pd.DataFrame(rows); R.to_csv('lead_dyad.csv',index=False)
print(R.describe().round(2)); print(R.groupby('age')[['n','p_inf','switch','switch_chance','pinf_lag','ppar_lag']].mean().round(3))
from scipy import stats
print('p_inf vs .5',stats.ttest_1samp(R.p_inf,.5))
print('switch vs chance',stats.ttest_rel(R.switch,R.switch_chance), (R.switch-R.switch_chance).mean())
for v in ['p_inf','switch']:
    print(v,stats.f_oneway(*[R[R.age==a][v] for a in [6,9,12]]))
print('follow lag inf vs par',stats.wilcoxon(R.pinf_lag.dropna(),R.ppar_lag.dropna()) if R[['pinf_lag','ppar_lag']].dropna().shape[0]>5 else '')
