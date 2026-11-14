import pandas as pd, numpy as np, pickle
FS=30; MAXLAG=10*FS
l=pd.read_csv('../data/look_spans.csv')
c=pd.read_csv('../data/covariates.csv')
t=pd.read_csv('../data/talk_timing.csv')
age=dict(zip(c.dyad_id,c.age_group))
series={}
for d,g in l.groupby('dyad_id'):
    cond=g[g.stream=='condition']; t0=cond.onset_ms.min(); t1=cond.offset_ms.max()
    n=int(np.ceil((t1-t0)*FS/1000))
    S={}
    for s in ['infant_book','parent_book','infant_partner','parent_partner']:
        x=np.zeros(n)
        for on,off in g[g.stream==s][['onset_ms','offset_ms']].values:
            a=int((on-t0)*FS/1000); b=int(np.ceil((off-t0)*FS/1000)); x[max(a,0):min(b,n)]=1
        S[s]=x
    sp=np.zeros(n)
    for on,off in t[t.dyad_id==d][['onset_ms','offset_ms']].values:
        a=int((on-t0)*FS/1000); b=int(np.ceil((off-t0)*FS/1000)); sp[max(a,0):min(b,n)]=1
    S['speech']=sp
    ge=np.zeros(n)
    for on in g[g.stream=='parent_gesture'].onset_ms.values:
        a=int((on-t0)*FS/1000)
        if 0<=a<n: ge[a]=1
    S['gesture']=ge
    S['t0']=t0; S['n']=n
    series[d]=S
pickle.dump(series,open('series.pkl','wb'))
# check proportions match covariates
chk=pd.DataFrame({d:{'ib':S['infant_book'].mean(),'ja':(S['infant_book']*S['parent_book']).mean()} for d,S in series.items()}).T
m=c.set_index('dyad_id')[['book_looking_prop','joint_book_prop']].join(chk)
print((m.book_looking_prop-m.ib).abs().max(), (m.joint_book_prop-m.ja).abs().max())
