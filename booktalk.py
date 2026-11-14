import pandas as pd, numpy as np, pickle, sys
exec(open('../code/talkxc.py').read().split('rows=[]; curves=[]')[0])
t=pd.read_csv('../data/talk_timing.csv')
t['cat']=t.kind
t.loc[(t.kind=='other')&(t.talk_type=='specific'),'cat']='action'
t.loc[t.cat=='other','cat']='nonbook'
# Reading time only: drop utterances whose onset falls in a page turn / break section (25 utterances)
_c=pd.read_csv('../data/look_spans.csv'); _c=_c[_c.stream=='condition']
def _where(r):
    s=_c[(_c.dyad_id==r.dyad_id)&(_c.onset_ms<=r.onset_ms)&(_c.offset_ms>r.onset_ms)]
    return 'reading' if len(s) and s.value.iloc[0] in ('individual','category','none') else 'break'
t['where']=t.apply(_where,axis=1); t=t[t['where']=='reading'].reset_index(drop=True)
t.to_csv('talk_cat.csv',index=False)
def spans(d,cats,t0,n):
    x=np.zeros(n)
    for on,off in t[(t.dyad_id==d)&(t.cat.isin(cats))][['onset_ms','offset_ms']].values:
        a=int((on-t0)*FS/1000); b=int(np.ceil((off-t0)*FS/1000)); x[max(a,0):min(b,n)]=1
    return x
BOOK=['individual_name','category_name','visual_description','analogy','attention_directing','action']
rows=[]
for d,s in S.items():
    n=s['n']; t0=s['t0']
    bk=dilate(spans(d,BOOK,t0,n),FS); nb=dilate(spans(d,['nonbook'],t0,n),FS)
    states={'book_talk':bk,'nonbook_talk':nb*(1-bk),'silent':(1-bk)*(1-nb)}
    p=s['parent_book']; i=s['infant_book']
    for st,m in states.items():
        z=mxc(p,i,m)
        ch=np.nanmean([mxc(p,np.roll(i,int(rng.integers(20*FS,n-20*FS))),m) for _ in range(20)],0)
        rows.append(dict(dyad=d,age=age[d],state=st,prop=m.mean(),z_all=np.nanmean(z),chance=np.nanmean(ch),z0=z[lags==0][0],
                         il=np.nanmean(z[lags<0]-ch[lags<0]),pl=np.nanmean(z[lags>0]-ch[lags>0])))
    print(d,end=' ',file=sys.stderr)
pd.DataFrame(rows).to_csv('booktalk_xc.csv',index=False)
