import pandas as pd, numpy as np, pickle, sys
FS=30; STEP=3; MAXLAG=10*FS
S=pickle.load(open('series.pkl','rb'))
c=pd.read_csv('../data/covariates.csv'); age=dict(zip(c.dyad_id,c.age_group))
lags=np.arange(-MAXLAG,MAXLAG+1,STEP)
rng=np.random.default_rng(11)
def dilate(x,w):
    k=np.ones(2*w+1); return (np.convolve(x,k,'same')>0).astype(float)
def mxc(p,i,m):
    n=len(p); out=np.full(len(lags),np.nan)
    for j,k in enumerate(lags):
        if k<0: a=p[-k:n]; b=i[0:n+k]; mm=m[-k:n]
        else: a=p[0:n-k]; b=i[k:n]; mm=m[0:n-k]
        a=a[mm>0]; b=b[mm>0]
        if len(a)<FS*5 or a.std()==0 or b.std()==0: continue
        out[j]=np.corrcoef(a,b)[0,1]
    return np.arctanh(np.clip(out,-.999,.999))
rows=[]; curves=[]
for d,s in S.items():
    n=s['n']; talk=dilate(s['speech'],FS)  # talking = within 1 s of an utterance
    for pair,(px,ix) in {'book':('parent_book','infant_book'),'face':('parent_partner','infant_partner')}.items():
        p=s[px]; i=s[ix]
        if i.std()==0 or p.std()==0: continue
        for state,m in {'talk':talk,'silent':1-talk}.items():
            z=mxc(p,i,m)
            sh=[mxc(p,np.roll(i,int(rng.integers(20*FS,n-20*FS))),m) for _ in range(30)] if n>40*FS*2 else []
            ch=np.nanmean(sh,0) if sh else np.full(len(lags),np.nan)
            rows.append(dict(dyad=d,age=age[d],pair=pair,state=state,prop_time=m.mean(),
                z_all=np.nanmean(z),chance=np.nanmean(ch),z0=z[lags==0][0],
                inf_lead=np.nanmean(z[lags<0]),par_lead=np.nanmean(z[lags>0]),
                inf_lead_ch=np.nanmean(ch[lags<0]),par_lead_ch=np.nanmean(ch[lags>0])))
            for L,zz,cc in zip(lags/FS,z,ch): curves.append((d,age[d],pair,state,L,zz,cc))
    print(d,file=sys.stderr,end=' ')
pd.DataFrame(rows).to_csv('talkxc_dyad.csv',index=False)
pd.DataFrame(curves,columns=['dyad','age','pair','state','lag','z','chance']).to_csv('talkxc_curves.csv',index=False)
