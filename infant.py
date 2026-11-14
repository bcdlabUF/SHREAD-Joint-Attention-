import pandas as pd, numpy as np, pickle, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf, statsmodels.api as sm, pingouin as pg
from scipy import stats
FS=30; W3=3*FS; W5=5*FS
S=pickle.load(open('series.pkl','rb'))
c=pd.read_csv('../data/covariates.csv'); age=dict(zip(c.dyad_id,c.age_group))
t=pd.read_csv('talk_cat.csv'); l=pd.read_csv('../data/look_spans.csv')
BOOK=['individual_name','category_name','visual_description','analogy','attention_directing','action']
def dil(x,w): return (np.convolve(x,np.ones(2*w+1),'same')>0).astype(float)
rng=np.random.default_rng(31); ev=[]; st=[]
for d,s in S.items():
    n=s['n']; t0=s['t0']; ib=s['infant_book']; pb=s['parent_book']
    cond=np.full(n,'',dtype=object)
    for on,off,v in l[(l.dyad_id==d)&(l.stream=='condition')][['onset_ms','offset_ms','value']].values:
        cond[max(int((on-t0)*FS/1000),0):min(int(np.ceil((off-t0)*FS/1000)),n)]=v
    read=np.isin(cond,['individual','category','none'])
    def mk(cats):
        x=np.zeros(n)
        for on,off in t[(t.dyad_id==d)&(t['cat'].isin(cats))][['onset_ms','offset_ms']].values:
            x[max(int((on-t0)*FS/1000),0):min(int(np.ceil((off-t0)*FS/1000)),n)]=1
        return dil(x,FS)
    bk=mk(BOOK); nb=mk(['nonbook'])*(1-bk); sil=(1-bk)*(1-mk(['nonbook']))
    for name,m in [('book_talk',bk),('nonbook_talk',nb),('silent',sil)]:
        mm=(m>0)&read
        st.append(dict(dyad=d,age=age[d],state=name,ib=ib[mm].mean() if mm.sum()>FS*5 else np.nan))
    def score(i,k,cnd):
        if ib[i]==0:
            return dict(state='off',start2=float(ib[i:i+2*FS].any()),start3=float(ib[i:i+W3].any()),start5=float(ib[i:i+W5].any()),pb_on=pb[i])
        seg=ib[i:]; brk=np.argmax(seg==0) if (seg==0).any() else len(seg)
        back=ib[:i+1][::-1]; el=np.argmax(back==0) if (back==0).any() else len(back)
        return dict(state='on',keep3=float(brk>=W3),remain_s=min(brk,10*FS)/FS,elapsed_s=el/FS,pb_on=pb[i])
    for _,u in t[t.dyad_id==d].iterrows():
        i=int((u.onset_ms-t0)*FS/1000)
        if i<0 or i+W5>=n: continue
        r=score(i,u['cat'],u.label_condition); r.update(dyad=d,age=age[d],kind=u['cat'],cond=u.label_condition); ev.append(r)
    valid=np.where(read)[0]; valid=valid[valid+W5<n]
    for i in rng.choice(valid,300,replace=False):
        r=score(i,'chance',cond[i]); r.update(dyad=d,age=age[d],kind='chance',cond=cond[i]); ev.append(r)
E=pd.DataFrame(ev); E.to_csv('infant_events.csv',index=False)
T=pd.DataFrame(st); T.to_csv('infant_state.csv',index=False)
# --- state analysis
print(T.groupby(['age','state']).ib.mean().unstack().round(3))
X=T[T.state.isin(['book_talk','silent'])].dropna(); k=X.groupby('dyad').state.nunique(); X=X[X.dyad.isin(k[k==2].index)]; X['ag']=X.age.astype(str)
print(pg.mixed_anova(X,dv='ib',within='state',between='ag',subject='dyad')[['Source','DF1','DF2','F','p_unc','np2']].round(4))
for a,g in X.groupby('age'):
    w=g.pivot(index='dyad',columns='state',values='ib'); print(a, 'p=%.4f'%stats.ttest_rel(w.book_talk,w.silent).pvalue)
# --- event models
order=['chance','category_name','individual_name','analogy','visual_description','attention_directing','action','nonbook']
E['kind']=pd.Categorical(E.kind,order)
def run(df,y,extra,fam,cov):
    m=smf.gee(f"{y}~C(kind,Treatment('chance')){extra}",groups='dyad',data=df,family=fam,cov_struct=cov).fit()
    tr=np.exp if isinstance(fam,sm.families.Binomial) else (lambda v:v)
    out=[]
    for kk in order[1:]:
        nm=f"C(kind, Treatment('chance'))[T.{kk}]"; lo,hi=m.conf_int().loc[nm]
        out.append(dict(kind=kk,est=tr(m.params[nm]),lo=tr(lo),hi=tr(hi),p=m.pvalues[nm],n=(df.kind==kk).sum(),dyads=df[df.kind==kk].dyad.nunique(),rate=df[df.kind==kk][y].mean()))
    R=pd.DataFrame(out); print(f'\n### {y} (chance rate {df[df.kind=="chance"][y].mean():.3f})'); print(R.round(3).to_string(index=False)); return R
off=E[E.state=='off']; on=E[E.state=='on'].copy(); on['log_el']=np.log(on.elapsed_s+.1)
B=sm.families.Binomial(); X_=sm.cov_struct.Exchangeable()
r1=run(off,'start3','+C(cond)+C(age)+pb_on',B,X_); r1.to_csv('inf_start3.csv',index=False)
run(off,'start5','+C(cond)+C(age)+pb_on',B,X_)
r2=run(on,'keep3','+C(cond)+C(age)+pb_on+log_el',B,X_); r2.to_csv('inf_keep3.csv',index=False)
run(on,'remain_s','+C(cond)+C(age)+pb_on+log_el',sm.families.Gaussian(),X_)
# age-specific start for names/redirects
E2=off.copy(); E2['k2']=E2.kind.astype(str).replace({'individual_name':'name','category_name':'name'})
for a,g in E2[E2.k2.isin(['chance','name','attention_directing','visual_description','analogy'])].groupby('age'):
    m=smf.gee("start3~C(k2,Treatment('chance'))+C(cond)+pb_on",groups='dyad',data=g,family=B).fit()
    print('start3',a,{kk.split('T.')[1][:-1]:(round(np.exp(v),2),round(m.pvalues[kk],3)) for kk,v in m.params.items() if 'k2' in kk})
