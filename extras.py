import pandas as pd, numpy as np, pickle, warnings; warnings.filterwarnings('ignore')
import pingouin as pg, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
FS=30
S=pickle.load(open('series.pkl','rb')); c=pd.read_csv('../data/covariates.csv'); age=dict(zip(c.dyad_id,c.age_group))
l=pd.read_csv('../data/look_spans.csv'); t=pd.read_csv('talk_cat.csv')
rng=np.random.default_rng(9)
P=[];J=[];Q=[];TT=[]
for d,s in S.items():
    n=s['n']; t0=s['t0']; ib,pb,ip,pp=s['infant_book'],s['parent_book'],s['infant_partner'],s['parent_partner']; ja=ib*pb
    cond=np.full(n,'',dtype=object)
    for on,off,v in l[(l.dyad_id==d)&(l.stream=='condition')][['onset_ms','offset_ms','value']].values:
        cond[max(int((on-t0)*FS/1000),0):min(int(np.ceil((off-t0)*FS/1000)),n)]=v
    # 1. page type
    for v in ['individual','category','none']:
        m=cond==v
        if m.sum()>FS*10: P.append(dict(dyad=d,age=age[d],page=v,ib=ib[m].mean(),ja=ja[m].mean(),pb=pb[m].mean()))
    # 2. who initiates JA: at each JA onset, who arrived last
    read=np.isin(cond,['individual','category','none'])
    on=np.where((ja[1:]>0)&(ja[:-1]==0)&read[1:])[0]+1
    inf_last=sum(1 for i in on if ib[i-1]==0 and pb[i-1]==1); par_last=sum(1 for i in on if pb[i-1]==0 and ib[i-1]==1)
    J.append(dict(dyad=d,age=age[d],n_onsets=len(on),infant_follows=inf_last,parent_follows=par_last))
    # 3. parent talk after infant book-look onset (follow-in) vs chance
    tk=np.zeros(n); 
    for a,b in t[t.dyad_id==d][['onset_ms','offset_ms']].values:
        i=int((a-t0)*FS/1000)
        if 0<=i<n: tk[i]=1
    io=np.where((ib[1:]>0)&(ib[:-1]==0)&read[1:])[0]+1; io=io[io<n-2*FS]
    ok=np.where(read)[0]; ok=ok[ok<n-2*FS]; rnd=rng.choice(ok,500)
    f=lambda idx: np.mean([tk[i:i+2*FS].any() for i in idx]) if len(idx) else np.nan
    Q.append(dict(dyad=d,age=age[d],talk_after_look=f(io),talk_after_random=f(rnd),n_looks=len(io)))
    # 4. time on task: first vs second half of reading time
    ri=np.where(read)[0]; h=len(ri)//2
    for half,idx in [('first',ri[:h]),('second',ri[h:])]: TT.append(dict(dyad=d,age=age[d],half=half,ib=ib[idx].mean(),ja=ja[idx].mean()))
P=pd.DataFrame(P); J=pd.DataFrame(J); Q=pd.DataFrame(Q); TT=pd.DataFrame(TT)
for D_ in (P,J,Q,TT): D_['ag']=D_.age.astype(str)
print('=== 1. PAGE TYPE (individual-label / category-label / no-label pages)')
print(P.groupby(['age','page'])[['ib','ja']].mean().unstack().round(3))
k=P.groupby('dyad').page.nunique(); X=P[P.dyad.isin(k[k==3].index)]
for v in ['ib','ja']:
    print(v); print(pg.mixed_anova(X,dv=v,within='page',between='ag',subject='dyad')[['Source','DF1','DF2','F','p_unc','np2']].round(4).to_string(index=False))
    print(pg.pairwise_tests(X,dv=v,within='page',subject='dyad',padjust='holm')[['A','B','T','p_corr']].round(4).to_string(index=False))
print('\n=== 2. WHO JOINS WHOM (JA onsets where infant arrived last = infant follows parent)')
J['prop_infant_follows']=J.infant_follows/(J.infant_follows+J.parent_follows)
print(J.groupby('age')[['n_onsets','prop_infant_follows']].mean().round(3))
print('vs .5:',stats.ttest_1samp(J.prop_infant_follows.dropna(),.5)); print(pg.anova(J.dropna(),dv='prop_infant_follows',between='ag').round(4).to_string(index=False))
print(pg.pairwise_tukey(J.dropna(),dv='prop_infant_follows',between='ag')[['A','B','p_tukey']].round(4).to_string(index=False))
print('onsets per dyad by age ANOVA p=',round(pg.anova(J,dv='n_onsets',between='ag').iloc[0,4],4))
print('\n=== 3. PARENT TALK WITHIN 2 s OF INFANT LOOKING TO BOOK (follow-in responsiveness)')
Q['exc']=Q.talk_after_look-Q.talk_after_random
print(Q.groupby('age')[['talk_after_look','talk_after_random','exc']].mean().round(3))
print('exc>0:',stats.ttest_1samp(Q.exc.dropna(),0)); print(pg.anova(Q.dropna(),dv='exc',between='ag').round(4).to_string(index=False))
print('\n=== 4. TIME ON TASK (first vs second half of reading)')
print(TT.groupby(['age','half'])[['ib','ja']].mean().unstack().round(3))
for v in ['ib','ja']: print(v); print(pg.mixed_anova(TT,dv=v,within='half',between='ag',subject='dyad')[['Source','F','p_unc','np2']].round(4).to_string(index=False))
