import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf, statsmodels.api as sm
E=pd.read_csv('infant_events.csv'); on=E[E.state=='on'].copy()
on['log_el']=np.log(on.elapsed_s+.1)
for s in [2,3,5]: on[f'keep{s}']=(on.remain_s>=s).astype(float)
order=['chance','category_name','individual_name','analogy','visual_description','attention_directing','action','nonbook']
on['kind']=pd.Categorical(on.kind,order)
print('events',on.kind.value_counts().to_dict())
print('median elapsed by kind',on.groupby('kind',observed=True).elapsed_s.median().round(2).to_dict())
X=sm.cov_struct.Exchangeable()
def run(df,y,fam,extra='+C(cond)+C(age)+pb_on+log_el'):
    m=smf.gee(f"{y}~C(kind,Treatment('chance')){extra}",groups='dyad',data=df,family=fam,cov_struct=X).fit()
    tr=np.exp if isinstance(fam,sm.families.Binomial) else (lambda v:v); out={}
    for kk in order[1:]:
        nm=f"C(kind, Treatment('chance'))[T.{kk}]"
        if nm in m.params: out[kk]=(round(tr(m.params[nm]),2),round(m.pvalues[nm],3))
    return out
res={}
for y in ['keep2','keep3','keep5']: res[y]=run(on,y,sm.families.Binomial())
res['remain_s']=run(on,'remain_s',sm.families.Gaussian())
R=pd.DataFrame(res); R['n']=on.kind.value_counts(); R['dyads']=on.groupby('kind',observed=True).dyad.nunique()
R['raw_keep3']=on.groupby('kind',observed=True).keep3.mean().round(3); R['raw_remain']=on.groupby('kind',observed=True).remain_s.mean().round(2)
print(R.to_string())
# moderators: parent looking at book (JA vs infant alone)
print('\n-- split by parent looking at book at onset')
for pbv,g in on.groupby('pb_on'):
    print('pb_on',pbv,'n',len(g), run(g,'keep3',sm.families.Binomial(),'+C(cond)+C(age)+log_el'))
print('\n-- by age')
for a,g in on.groupby('age'):
    print(a, run(g,'keep3',sm.families.Binomial(),'+C(cond)+pb_on+log_el'))
    print('   remain', run(g,'remain_s',sm.families.Gaussian(),'+C(cond)+pb_on+log_el'))
# kind x age interaction for redirects
g=on[on.kind.isin(['chance','attention_directing'])].copy(); g['red']=(g.kind=='attention_directing').astype(float)
m=smf.gee("keep3~red*C(age)+C(cond)+pb_on+log_el",groups='dyad',data=g,family=sm.families.Binomial(),cov_struct=X).fit()
print('redirect x age:', {k:round(v,3) for k,v in m.pvalues.items() if 'red' in k})
on.to_csv('maint_events.csv',index=False)
