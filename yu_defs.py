import pandas as pd, numpy as np, pickle, warnings; warnings.filterwarnings('ignore')
import pingouin as pg, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
FS=30; S=pickle.load(open('series.pkl','rb')); l=pd.read_csv('../data/look_spans.csv')
c=pd.read_csv('../data/covariates.csv'); age=dict(zip(c.dyad_id,c.age_group)); t=pd.read_csv('talk_cat.csv')
def runs(x):
    d=np.diff(np.r_[0,x.astype(int),0]); return np.where(d==1)[0],np.where(d==-1)[0]
def bridge(x,gap):
    x=x.copy(); on,off=runs(x)
    for a,b in zip(off[:-1],on[1:]):
        if b-a<gap: x[a:b]=1
    return x
def minlen(x,L):
    y=np.zeros_like(x); on,off=runs(x)
    for a,b in zip(on,off):
        if b-a>=L: y[a:b]=1
    return y
rows=[]; ev=[]
for d,s in S.items():
    n=s['n']; ib=s['infant_book']; ja=ib*s['parent_book']; t0=s['t0']
    ja_yu=minlen(bridge(ja,int(.3*FS)),int(.5*FS))       # Yu et al. 2019: >500 ms, bridging looks away <300 ms
    sa=minlen(ib,3*FS)                                    # sustained attention: >=3 s continuous (Yu et al. 2019)
    rows.append(dict(dyad=d,age=age[d],ja_frame=ja.mean(),ja_yu=ja_yu.mean(),sa=sa.mean(),sa_share=sa.sum()/max(ib.sum(),1)))
    for _,u in t[t.dyad_id==d].iterrows():
        i=int((u.onset_ms-t0)*FS/1000)
        if 0<=i<n: ev.append(dict(dyad=d,age=age[d],kind=u['cat'],in_sa=sa[i],in_ja=ja_yu[i]))
R=pd.DataFrame(rows); R['ag']=R.age.astype(str)
print(R.groupby('age')[['ja_frame','ja_yu','sa','sa_share']].mean().round(3))
print('corr ja_frame vs ja_yu r=%.3f'%stats.pearsonr(R.ja_frame,R.ja_yu)[0])
for v in ['ja_yu','sa','sa_share']:
    a=pg.anova(R,dv=v,between='ag'); k=stats.kruskal(*[g[v] for _,g in R.groupby('age')])
    print(v,'F(%d,%d)=%.2f p=%.4f np2=%.3f | KW H=%.2f p=%.4f'%(a.ddof1[0],a.ddof2[0],a.F[0],a.iloc[0,4],a.np2[0],k.statistic,k.pvalue), pg.pairwise_tukey(R,dv=v,between='ag')[['A','B','p_tukey']].round(3).values.tolist())
E=pd.DataFrame(ev)
print('\n% of utterances produced during infant sustained attention, by type:'); print(E.groupby('kind').in_sa.mean().round(3).sort_values())
E['nm']=E.kind.isin(['individual_name','category_name']).astype(float)
m=smf.gee('in_sa~C(kind,Treatment("nonbook"))+C(age)',groups='dyad',data=E,family=sm.families.Binomial()).fit()
print({k.split('[T.')[1][:-1]:(round(np.exp(v),2),round(m.pvalues[k],4)) for k,v in m.params.items() if 'kind' in k})
R.to_csv('yu_defs_dyad.csv',index=False)
