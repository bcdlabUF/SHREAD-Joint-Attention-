import pandas as pd, numpy as np, pingouin as pg
from scipy import stats
import statsmodels.formula.api as smf
L=pd.read_csv('xc_long.csv'); L=L[L.lag!=0].copy()
L['order']=np.where(L.lag<0,'inf','par'); L['abslag']=L.lag.abs().round(3)
for pair in ['book','face']:
    x=L[L.pair==pair].dropna()
    m=smf.ols('z~C(order)*C(age)+C(dyad)+C(abslag)',x).fit()
    import statsmodels.api as sm
    a=sm.stats.anova_lm(m,typ=2)
    print(pair,'rows',len(x)); print(a.loc[['C(order)','C(order):C(age)','Residual'],['df','F','PR(>F)']])
# bout durations
l=pd.read_csv('../data/look_spans.csv'); l['dur']=(l.offset_ms-l.onset_ms)/1000
b=l[l.stream.isin(['infant_book','parent_book','infant_partner'])].groupby(['stream','dyad_id','age_group']).dur.agg(['median','mean','count']).reset_index()
for s,g in b.groupby('stream'):
    kw=stats.kruskal(*[gg['median'] for _,gg in g.groupby('age_group')])
    print(s, g.groupby('age_group')[['median','count']].median().round(2).to_dict(), 'KW p median dur',round(kw.pvalue,4))
# does mean xcorr age effect survive controlling for look duration?
d=pd.read_csv('xc_dyad_book.csv')
dm=b[b.stream=='infant_book'].set_index('dyad_id')['median']; pm=b[b.stream=='parent_book'].set_index('dyad_id')['median']
d['ib_dur']=np.log(d.dyad.map(dm)); d['pb_dur']=np.log(d.dyad.map(pm))
c=pd.read_csv('cov_derived.csv').set_index('dyad_id'); d['ibp']=d.dyad.map(c.book_looking_prop)
print(pg.ancova(d,dv='z_all',between='ag',covar=['ib_dur','pb_dur']).round(4))
print(pg.ancova(d,dv='exc_pseudo',between='ag',covar=['ib_dur','pb_dur']).round(4))
print(pg.ancova(d,dv='z_all',between='ag',covar=['ibp']).round(4))
