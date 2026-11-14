import pandas as pd, numpy as np, pickle, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf, statsmodels.api as sm
FS=30
S=pickle.load(open('series.pkl','rb')); c=pd.read_csv('../data/covariates.csv'); age=dict(zip(c.dyad_id,c.age_group))
t=pd.read_csv('talk_cat.csv'); l_=pd.read_csv('../data/look_spans.csv'); E=pd.read_csv('kinds_events.csv')

def readmask(d,t0,n):
    m=np.zeros(n)
    for on,off,v in l_[(l_.dyad_id==d)&(l_.stream=='condition')][['onset_ms','offset_ms','value']].values:
        if v!='break': m[max(int((on-t0)*FS/1000),0):min(int(__import__('math').ceil((off-t0)*FS/1000)),n)]=1
    return m
rng=np.random.default_rng(5); rows=[]
for d,s in S.items():
    ib,pb,pp=s['infant_book'],s['parent_book'],s['parent_partner']; ja=__import__('jadef').yu_ja(ib,pb); n=s['n']; t0=s['t0']
    idx=[(int((u.onset_ms-t0)*FS/1000),u['cat']) for _,u in t[t.dyad_id==d].iterrows()]
    rmk=readmask(d,t0,n); jai=np.where((ja*rmk)[:n-10*FS]>0)[0]; idx+= [(i,'chance') for i in rng.choice(jai,min(200,len(jai)),replace=False)] if len(jai) else []
    for i,k in idx:
        if i<0 or i+10*FS>=n or ja[i]==0: continue
        seg=ja[i:i+10*FS]
        if not (seg==0).any(): who='none'
        else:
            e=i+np.argmax(seg==0); io,po=ib[e]==0,pb[e]==0
            who='both' if (io and po) else ('infant' if io else ('parent' if po else 'both'))
        rows.append(dict(dyad=d,age=age[d],kind=k,who=who,parent_face_3s=float(pp[i:i+3*FS].any())))
B=pd.DataFrame(rows); B.to_csv('breaker_events.csv',index=False)
print(pd.crosstab(B.kind,B.who,normalize='index').round(2).join(B.kind.value_counts()))
order=['chance','individual_name','category_name','visual_description','analogy','attention_directing','action','nonbook']
B['kind']=pd.Categorical(B.kind,order)
for y,expr in [('parent_breaks',"who=='parent'"),('infant_breaks',"who=='infant'"),('parent_face_3s',None)]:
    if expr: B[y]=B.eval(expr).astype(float)
    m=smf.gee(f"{y}~C(kind,Treatment('chance'))+C(age)",groups='dyad',data=B,family=sm.families.Binomial()).fit()
    print('\n',y,'chance rate %.2f'%B[B.kind=='chance'][y].mean())
    for k in order[1:]:
        nm=f"C(kind, Treatment('chance'))[T.{k}]"; print(f'  {k:20s} rate={B[B.kind==k][y].mean():.2f} OR={np.exp(m.params[nm]):.2f} p={m.pvalues[nm]:.3f}')
