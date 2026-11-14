import pandas as pd, numpy as np, pickle, sys
FS=30; MAXLAG=10*FS
series=pickle.load(open('series.pkl','rb'))
c=pd.read_csv('../data/covariates.csv'); age=dict(zip(c.dyad_id,c.age_group))
def xcorr(p,i):
    # r(k)=corr(p[t], i[t+k]); k<0 => infant leads
    n=min(len(p),len(i)); p=p[:n]; i=i[:n]; out=np.full(2*MAXLAG+1,np.nan)
    for j,k in enumerate(range(-MAXLAG,MAXLAG+1)):
        a=p[max(0,-k):n-max(0,k)]; b=i[max(0,k):n-max(0,-k)] if k>=0 else i[0:n+k]
        if k<0: a=p[-k:n]; b=i[0:n+k]
        else: a=p[0:n-k]; b=i[k:n]
        if a.std()==0 or b.std()==0: continue
        out[j]=np.corrcoef(a,b)[0,1]
    return out
def circ_r(p,i):
    n=min(len(p),len(i)); p=p[:n]-p[:n].mean(); i=i[:n]-i[:n].mean()
    if p.std()==0 or i.std()==0: return None
    cc=np.fft.irfft(np.conj(np.fft.rfft(p))*np.fft.rfft(i),n)/(n*p.std()*i.std())
    return cc  # cc[s]=corr(p[t], i[t+s]) circular
lags=np.arange(-MAXLAG,MAXLAG+1)/FS
pairs={'book':('parent_book','infant_book'),'face':('parent_partner','infant_partner'),
       'speech':('speech','infant_book'),'gesture':('gesture','infant_book')}
rows=[]; summ=[]
rng=np.random.default_rng(1)
for name,(px,ix) in pairs.items():
    for d,S in series.items():
        r=xcorr(S[px],S[ix])
        if np.all(np.isnan(r)): continue
        z=np.arctanh(np.clip(r,-.999,.999))
        for L,zz in zip(lags,z): rows.append((name,d,age[d],L,zz))
        # circular-shift baseline: shifts >= 20 s from 0 (so lag window +-10 never overlaps)
        cc=circ_r(S[px],S[ix]); n=len(cc); s=np.arange(n); s=np.minimum(s,n-s)
        circ=np.arctanh(np.clip(cc[s>=20*FS],-.999,.999)).mean()
        # pseudo-dyads: same parent series with other infants of same age
        others=[o for o in series if o!=d and age[o]==age[d]]
        pz=[]
        for o in others:
            rr=xcorr(series[d][px],series[o][ix]) if False else None
            cc2=circ_r(S[px],series[o][ix])
            if cc2 is None: continue
            k=np.r_[np.arange(0,MAXLAG+1),np.arange(len(cc2)-MAXLAG,len(cc2))]
            pz.append(np.arctanh(np.clip(cc2[k],-.999,.999)).mean())
        summ.append(dict(pair=name,dyad=d,age=age[d],z_all=np.nanmean(z),
            z_inf_lead=np.nanmean(z[lags<0]),z_par_lead=np.nanmean(z[lags>0]),z0=z[MAXLAG],
            peak_lag=lags[np.nanargmax(z)],circ=circ,pseudo=np.mean(pz) if pz else np.nan))
    print(name,'done',file=sys.stderr)
pd.DataFrame(rows,columns=['pair','dyad','age','lag','z']).to_csv('xc_long.csv',index=False)
pd.DataFrame(summ).to_csv('xc_dyad.csv',index=False)
