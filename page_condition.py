"""Attention, joint attention, coordination, and reciprocity by page type (individual-label, category-label, no-label).
Per dyad x page type; page type is a frame mask from the 'condition' stream (page turns/breaks excluded).
Also span-level models with session time to separate page type from time on task (dyads re-read pages in different orders)."""
import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from jadef import yu_ja, sustained, _runs, FS
from sklearn.linear_model import LogisticRegression
S=pickle.load(open('series.pkl','rb')); c=pd.read_csv('../data/covariates.csv'); age=dict(zip(c.dyad_id,c.age_group))
l=pd.read_csv('../data/look_spans.csv'); LB=pd.read_csv('lead_bouts.csv')
PAGES=['individual','category','none']; rng=np.random.default_rng(7)
def cond_vec(d,n,t0):
    v=np.full(n,'break',dtype=object)
    for on,off,val in l[(l.dyad_id==d)&(l.stream=='condition')][['onset_ms','offset_ms','value']].values:
        v[max(int((on-t0)*FS/1000),0):min(int(np.ceil((off-t0)*FS/1000)),n)]=val
    return v
def xcorr_mask(i,p,m,maxlag=10*FS,step=15):
    zs=[]
    for k in range(-maxlag,maxlag+1,step):
        if k>=0: a=i[:len(i)-k]; b=p[k:]; mm=m[:len(m)-k]&m[k:]
        else: a=i[-k:]; b=p[:len(p)+k]; mm=m[-k:]&m[:len(m)+k]
        a=a[mm]; b=b[mm]
        if len(a)<FS*10 or a.std()==0 or b.std()==0: continue
        zs.append(np.arctanh(np.clip(np.corrcoef(a,b)[0,1],-.999,.999)))
    return np.mean(zs) if zs else np.nan
B=6; L=10
def binarize(x): n=len(x)//B; return (x[:n*B].reshape(n,B).mean(1)>.5).astype(float)
def lagmat(x): return np.column_stack([x[L-k:len(x)-k] for k in range(1,L+1)])
def ll(X,y):
    if y.min()==y.max(): return 0.0
    m=LogisticRegression(C=1e4,max_iter=2000).fit(X,y); p=np.clip(m.predict_proba(X)[:,1],1e-9,1-1e-9)
    return float(np.sum(y*np.log(p)+(1-y)*np.log(1-p)))
def gc(t,dr,mask):
    y=t[L:]; own=lagmat(t); oth=lagmat(dr); mm=mask[L:]
    if mm.sum()<100: return np.nan
    return 1000*(ll(np.hstack([own,oth])[mm],y[mm])-ll(own[mm],y[mm]))/mm.sum()
rows=[]; spans=[]
for d,s in S.items():
    n=s['n']; t0=s['t0']; ib=s['infant_book'].astype(float); pb=s['parent_book'].astype(float)
    cv=cond_vec(d,n,t0); ja=yu_ja(ib,pb); sa=sustained(ib)
    # chance JA: circular shifts of parent series >= 20 s
    shifts=rng.integers(20*FS,n-20*FS,50)
    jach=np.array([yu_ja(ib,np.roll(pb,k)) for k in shifts])
    lb=LB[LB.dyad==d].copy(); lb=lb[lb.lead!='simul']; lb['fr']=(lb.t*FS).astype(int).clip(0,n-1); lb['cond']=cv[lb.fr.values]
    ibb=binarize(ib); pbb=binarize(pb); cvb=cv[:len(ibb)*B].reshape(-1,B)[:,B//2]
    for pg in PAGES:
        m=cv==pg
        if m.sum()<10*FS: continue
        bl=lb[lb.cond==pg]
        mb=cvb==pg
        sh=rng.integers(int(20*FS/B),len(ibb)-int(20*FS/B),10)
        g_pi=gc(ibb,pbb,mb); g_ip=gc(pbb,ibb,mb)
        g_pi_ch=np.nanmean([gc(ibb,np.roll(pbb,k),mb) for k in sh]); g_ip_ch=np.nanmean([gc(pbb,np.roll(ibb,k),mb) for k in sh])
        cs=rng.integers(20*FS,n-20*FS,20)
        xc=xcorr_mask(ib,pb,m); xc_ch=np.nanmean([xcorr_mask(np.roll(ib,k),pb,m) for k in cs])
        rows.append(dict(dyad=d,age=age[d],page=pg,secs=m.sum()/FS,ib=ib[m].mean(),pb=pb[m].mean(),sa=sa[m].mean(),ja=ja[m].mean(),
            ja_chance=jach[:,m].mean(),xc=xc,xc_chance=xc_ch,n_bouts=len(bl),inf_led=(bl.lead=='infant').mean() if len(bl)>=3 else np.nan,
            g_pi=g_pi,g_pi_ch=g_pi_ch,g_ip=g_ip,g_ip_ch=g_ip_ch))
    # span-level for time control
    on=0
    for k in range(1,n+1):
        if k==n or cv[k]!=cv[on]:
            if cv[on] in PAGES and k-on>=3*FS:
                spans.append(dict(dyad=d,age=age[d],page=cv[on],t_mid=(on+k)/2/FS/60,secs=(k-on)/FS,ib=ib[on:k].mean(),pb=pb[on:k].mean(),ja=ja[on:k].mean(),sa=sa[on:k].mean()))
            on=k
    print(d,end=' ',flush=True)
R=pd.DataFrame(rows); R['ja_exc']=R.ja-R.ja_chance; R['xc_exc']=R.xc-R.xc_chance; R['g_pi_exc']=R.g_pi-R.g_pi_ch; R['g_ip_exc']=R.g_ip-R.g_ip_ch
R.to_csv('page_condition_dyad.csv',index=False); pd.DataFrame(spans).to_csv('page_condition_spans.csv',index=False); print('\nsaved',R.shape)
