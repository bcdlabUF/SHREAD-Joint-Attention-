import pandas as pd, numpy as np, pickle
FS=30
S=pickle.load(open('series.pkl','rb'))
c=pd.read_csv('../data/covariates.csv'); age=dict(zip(c.dyad_id,c.age_group))
t=pd.read_csv('talk_cat.csv')
l=pd.read_csv('../data/look_spans.csv')
rng=np.random.default_rng(21)
W3=3*FS; W5=5*FS
def ev(ja,i,n):
    on=ja[i]
    if on==0:
        return dict(state='notJA',init2=float(ja[i:i+2*FS].any()),init3=float(ja[i:i+W3].any()),init5=float(ja[i:i+W5].any()))
    seg=ja[i:]; brk=np.argmax(seg==0) if (seg==0).any() else len(seg)
    back=ja[:i+1][::-1]; el=np.argmax(back==0) if (back==0).any() else len(back)
    return dict(state='JA',maint3=float(brk>=W3),remain_s=min(brk,10*FS)/FS,elapsed_s=el/FS,total_s=(el+brk)/FS)
rows=[]
for d,s in S.items():
    n=s['n']; t0=s['t0']; ja=__import__('jadef').yu_ja(s['infant_book'],s['parent_book'])
    cond=np.full(n,'',dtype=object)
    for on,off,v in l[(l.dyad_id==d)&(l.stream=='condition')][['onset_ms','offset_ms','value']].values:
        a=int((on-t0)*FS/1000); b=int(np.ceil((off-t0)*FS/1000)); cond[max(a,0):min(b,n)]=v
    for _,u in t[t.dyad_id==d].iterrows():
        i=int((u.onset_ms-t0)*FS/1000)
        if i<0 or i+W5>=n: continue
        r=ev(ja,i,n); r.update(dyad=d,age=age[d],kind=u['cat'],cond=u.label_condition,ib_on=s['infant_book'][i],pb_on=s['parent_book'][i]); rows.append(r)
    # chance: random moments in reading pages, matched to page type
    valid=np.where(np.isin(cond,['individual','category','none']))[0]; valid=valid[valid+W5<n]
    for i in rng.choice(valid,300,replace=False):
        r=ev(ja,i,n); r.update(dyad=d,age=age[d],kind='chance',cond=cond[i],ib_on=s['infant_book'][i],pb_on=s['parent_book'][i]); rows.append(r)
E=pd.DataFrame(rows); E.to_csv('kinds_events.csv',index=False)
print(pd.crosstab(E.kind,E.state))
