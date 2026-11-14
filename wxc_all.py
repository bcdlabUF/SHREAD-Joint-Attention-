import pickle,numpy as np,pandas as pd
from jadef import FS
S=pickle.load(open('series.pkl','rb')); c=pd.read_csv('../data/covariates.csv'); age=dict(zip(c.dyad_id,c.age_group))
WIN,STEP,ML=20*FS,FS//2,3*FS; LAGS=np.arange(-ML,ML+1,3); TH=.30; TOL=.1
def roll(x,w):  # rolling sums of windows starting at each index
    cs=np.r_[0,np.cumsum(x)]; return cs[w:]-cs[:-w]
def wxc(ib,pb,rng=None):
    n=len(ib); starts=np.arange(ML,n-WIN-ML,STEP); M=np.full((len(starts),len(LAGS)),np.nan)
    sx=roll(ib,WIN); sxx=sx  # binary
    for j,L in enumerate(LAGS):
        y=np.roll(pb,-L)          # y[t]=pb[t+L]; L>0 -> infant leads
        sy=roll(y,WIN); sxy=roll(ib*y,WIN)
        a=starts; mx=sx[a]/WIN; my=sy[a]/WIN
        cov=sxy[a]/WIN-mx*my; vx=mx-mx**2; vy=my-my**2
        with np.errstate(invalid='ignore',divide='ignore'): M[:,j]=np.where((vx>0)&(vy>0),cov/np.sqrt(vx*vy),np.nan)
    return M
def summarize(M):
    ok=np.isfinite(M).any(1); pk=np.full(len(M),np.nan); mx=np.full(len(M),np.nan)
    pk[ok]=LAGS[np.nanargmax(M[ok],1)]/FS; mx[ok]=np.nanmax(M[ok],1)
    lead=np.where(~ok|(mx<TH),'none',np.where(pk>TOL,'infant',np.where(pk<-TOL,'parent','sync')))
    seq=lead[np.isin(lead,['infant','parent'])]
    sw=(seq[1:]!=seq[:-1]).sum() if len(seq)>1 else 0
    coupled=np.isin(lead,['infant','parent','sync']).sum()
    mins=coupled*STEP/FS/60
    return dict(p_inf=(lead=='infant').sum()/max(coupled,1),p_par=(lead=='parent').sum()/max(coupled,1),p_sync=(lead=='sync').sum()/max(coupled,1),
                coupled=coupled/len(lead),sw_min=sw/mins if mins>0 else np.nan)
rows=[]; rng=np.random.default_rng(3)
for d,s in S.items():
    ib,pb=s['infant_book'],s['parent_book']
    r=summarize(wxc(ib,pb)); r.update(dyad=d,age=age[d])
    # chance: circular shift of parent series (>=20 s), 20 shifts
    ch=[summarize(wxc(ib,np.roll(pb,rng.integers(20*FS,len(pb)-20*FS)))) for _ in range(20)]
    for k in ['sw_min','coupled','p_inf']: r[k+'_chance']=np.nanmean([x[k] for x in ch])
    rows.append(r)
R=pd.DataFrame(rows); R.to_csv('wxc_dyad.csv',index=False)
print(R.groupby('age')[['coupled','coupled_chance','p_inf','p_par','p_sync','sw_min','sw_min_chance']].mean().round(3))
print(R[['coupled','p_inf','p_par','p_sync','sw_min']].describe().round(3))

# ---- smoothed episodes: a lead must hold >= 2 s (4 consecutive windows) to count
MINRUN=4
def episodes(M):
    ok=np.isfinite(M).any(1); pk=np.full(len(M),np.nan); mx=np.full(len(M),np.nan)
    pk[ok]=LAGS[np.nanargmax(M[ok],1)]/FS; mx[ok]=np.nanmax(M[ok],1)
    lead=np.where(~ok|(mx<TH),'none',np.where(pk>TOL,'infant',np.where(pk<-TOL,'parent','sync')))
    runs=[];cur=lead[0];n=1
    for v in lead[1:]:
        if v==cur: n+=1
        else: runs.append((cur,n)); cur=v; n=1
    runs.append((cur,n))
    ep=[(v,n) for v,n in runs if v in('infant','parent') and n>=MINRUN]
    seq=[v for v,_ in ep]; sw=sum(a!=b for a,b in zip(seq[:-1],seq[1:]))
    mins=len(M)*STEP/FS/60
    durs=[n*STEP/FS for _,n in ep]
    return sw/mins, (np.mean(durs) if durs else np.nan), len(ep), (sw/(len(seq)-1) if len(seq)>1 else np.nan)
rows=[]; rng=np.random.default_rng(4)
for d,s in S.items():
    ib,pb=s['infant_book'],s['parent_book']
    sw,du,ne,sp=episodes(wxc(ib,pb))
    ch=[episodes(wxc(ib,np.roll(pb,rng.integers(20*FS,len(pb)-20*FS)))) for _ in range(20)]
    rows.append(dict(dyad=d,ep_sw_min=sw,ep_dur=du,n_ep=ne,ep_sw_min_chance=np.nanmean([x[0] for x in ch]),ep_dur_chance=np.nanmean([x[1] for x in ch]),ep_switch_prop=sp,ep_switch_prop_chance=np.nanmean([x[3] for x in ch])))
E=pd.DataFrame(rows); R=pd.read_csv('wxc_dyad.csv').merge(E,on='dyad'); R.to_csv('wxc_dyad.csv',index=False)
print(R.groupby('age')[['ep_sw_min','ep_sw_min_chance','ep_dur','ep_dur_chance','n_ep']].mean().round(2))
