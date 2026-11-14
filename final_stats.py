import pandas as pd, numpy as np, pickle, warnings, json; warnings.filterwarnings('ignore')
import pingouin as pg, scikit_posthocs as sp, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from jadef import yu_ja, sustained, _runs
FS=30; S=pickle.load(open('series.pkl','rb'))
c=pd.read_csv('../data/covariates.csv'); age=dict(zip(c.dyad_id,c.age_group))
l=pd.read_csv('../data/look_spans.csv')
print('########## A. AGE (full session)')
R=[]
for d,s in S.items():
    ib,pb=s['infant_book'],s['parent_book']; ja=yu_ja(ib,pb); sa=sustained(ib)
    n=s['n']; t0=s['t0']; read=np.zeros(n,bool)
    for on,off,v in l[(l.dyad_id==d)&(l.stream=='condition')][['onset_ms','offset_ms','value']].values:
        if v!='break': read[max(int((on-t0)*FS/1000),0):min(int(np.ceil((off-t0)*FS/1000)),n)]=True
    on,_=_runs(ja); nb=sum(read[o] for o in on)
    R.append(dict(dyad=d,age=age[d],ib=ib.mean(),pb=pb.mean(),ja=ja.mean(),sa=sa.mean(),sa_share=sa.sum()/max(ib.sum(),1),n_ep=nb,ep_min=nb/(read.sum()/FS/60)))
R=pd.DataFrame(R); R['ag']=R.age.astype(str)
R['ja_given']=R.ja/R.ib; R['ja_chance']=R.ib*R.pb; R['ja_exc']=R.ja-R.ja_chance; R['ja_ratio']=R.ja/R.ja_chance
def test(v,drop=None):
    d=R if drop is None else R[~R.dyad.isin(drop)]
    a=pg.anova(d,dv=v,between='ag'); k=stats.kruskal(*[g[v] for _,g in d.groupby('age')]); sw=stats.shapiro(smf.ols(f'{v}~C(ag)',d).fit().resid).pvalue
    tk=pg.pairwise_tukey(d,dv=v,between='ag')[['A','B','p_tukey']].round(3).values.tolist()
    dn=sp.posthoc_dunn(d,val_col=v,group_col='ag',p_adjust='bonferroni').round(3)
    print(f"{v}: M/SD {d.groupby('age')[v].agg(['mean','std']).round(3).values.tolist()} F({a.ddof1[0]},{a.ddof2[0]})={a.F[0]:.2f} p={a.iloc[0,4]:.3f} np2={a.np2[0]:.3f} SW={sw:.3f} KW H={k.statistic:.2f} p={k.pvalue:.3f} tukey={tk} dunn6-9={dn.loc['6','9']} dunn6-12={dn.loc['6','12']} dunn9-12={dn.loc['9','12']}")
for v in ['ja','sa','sa_share','ja_given','ja_exc','n_ep','ep_min']: test(v)
z=(R.ja_ratio-R.ja_ratio.mean())/R.ja_ratio.std(); test('ja_ratio',drop=list(R[abs(z)>=3].dyad))
R.to_csv('final_age.csv',index=False)
X=sm.cov_struct.Exchangeable(); B=sm.families.Binomial()
order=['chance','category_name','individual_name','analogy','visual_description','attention_directing','action','nonbook']
show=['category_name','individual_name','analogy','visual_description','attention_directing']
def run(df,y,extra,lab):
    df=df.copy(); df['kind']=pd.Categorical(df.kind,order)
    m=smf.gee(f"{y}~C(kind,Treatment('chance')){extra}",groups='dyad',data=df,family=B,cov_struct=X).fit(); rows=[]
    for k in order[1:]:
        nm=f"C(kind, Treatment('chance'))[T.{k}]"; lo,hi=np.exp(m.conf_int().loc[nm])
        rows.append(dict(kind=k,OR=np.exp(m.params[nm]),lo=lo,hi=hi,p=m.pvalues[nm],n=(df.kind==k).sum(),dy=df[df.kind==k].dyad.nunique(),rate=df[df.kind==k][y].mean()))
    T=pd.DataFrame(rows); s=T.kind.isin(show); T.loc[s,'p_holm']=multipletests(T[s].p,method='holm')[1]
    print(f'\n### {lab} chance={df[df.kind=="chance"][y].mean():.3f}'); print(T.round(3).to_string(index=False))
I=pd.read_csv('infant_events.csv'); off=I[I.state=='off']
M=pd.read_csv('maint_events.csv'); M['log_el']=np.log(M.elapsed_s+.1); M['keep3']=(M.remain_s>=3).astype(float)
K=pd.read_csv('kinds_events.csv'); nj=K[K.state=='notJA']
print('\n########## B/C. TALK (3 s)')
run(off,'start3','+C(cond)+C(age)+pb_on','GET infant within 3 s')
run(M,'keep3','+C(cond)+C(age)+pb_on+log_el','KEEP infant >=3 s')
run(nj,'init3','+C(cond)+C(age)+ib_on+pb_on','START JA (Yu) within 3 s')
E2=off.copy(); E2['k2']=E2.kind.replace({'individual_name':'name','category_name':'name'})
for a in [6,9,12]:
    g=E2[(E2.age==a)&E2.k2.isin(['chance','name'])]; m=smf.gee("start3~C(k2,Treatment('chance'))+C(cond)+pb_on",groups='dyad',data=g,family=B).fit()
    nm="C(k2, Treatment('chance'))[T.name]"; lo,hi=np.exp(m.conf_int().loc[nm]); print('names',a,round(np.exp(m.params[nm]),2),round(lo,2),round(hi,2),round(m.pvalues[nm],3))
def sub(df,kk,extra):
    g=df[df.kind.isin(['chance',kk])].copy(); g['k']=(g.kind==kk).astype(float)
    r=smf.gee(f"keep3~k{extra}",groups='dyad',data=g,family=B,cov_struct=X).fit(); lo,hi=np.exp(r.conf_int().loc['k'])
    return round(np.exp(r.params['k']),2),round(lo,2),round(hi,2),round(r.pvalues['k'],3)
for a in [6,9,12]: print('redirect',a,sub(M[M.age==a],'attention_directing','+C(cond)+pb_on+log_el'))
for pv in [1,0]: print('pb',pv,'redirect',sub(M[M.pb_on==pv],'attention_directing','+C(cond)+C(age)+log_el'),'names',sub(M[M.pb_on==pv],'individual_name','+C(cond)+C(age)+log_el'))
g=M[M.kind.isin(['chance','attention_directing'])].copy(); g['red']=(g.kind=='attention_directing').astype(float)
m=smf.gee("keep3~red*C(age)+C(cond)+pb_on+log_el",groups='dyad',data=g,family=B,cov_struct=X).fit(); print('redirect x age:',{k:round(v,3) for k,v in m.pvalues.items() if 'red:' in k})
BR=pd.read_csv('breaker_events.csv'); BR['pe']=(BR.who=='parent').astype(float); BR['kind']=pd.Categorical(BR.kind,order)
m=smf.gee("pe~C(kind,Treatment('chance'))+C(age)",groups='dyad',data=BR,family=B).fit(); nm="C(kind, Treatment('chance'))[T.individual_name]"
lo,hi=np.exp(m.conf_int().loc[nm]); print('\nparent ends JA after indiv names OR=%.2f [%.2f,%.2f] p=%.3f n=%d dyads=%d chance=%.2f'%(np.exp(m.params[nm]),lo,hi,m.pvalues[nm],(BR.kind=='individual_name').sum(),BR[BR.kind=='individual_name'].dyad.nunique(),BR[BR.kind=='chance'].pe.mean()))
