import pandas as pd, numpy as np, pingouin as pg, scikit_posthocs as sp
from scipy import stats
import statsmodels.formula.api as smf
c=pd.read_csv('../data/covariates.csv')
c['parent_sex']=c.parent_sex.str.upper(); c['infant_sex']=c.infant_sex.str.upper()
c['age']=c.age_group.astype(str)
print(pd.crosstab(c.age_group,c.parent_sex)); print(pd.crosstab(c.age_group,c.infant_sex)); print(pd.crosstab(c.age_group,c.book))
print(c.groupby('age_group').session_s.describe())
print('session<270s:', (c.session_s<270).sum(), ' >330s:', (c.session_s>330).sum())
c['ja_given_book']=c.joint_book_prop/c.book_looking_prop
c['ja_chance']=c.book_looking_prop*c.parent_book_prop
c['ja_excess']=c.joint_book_prop-c.ja_chance
c['ja_ratio']=c.joint_book_prop/c.ja_chance
for v in ['book_looking_prop','joint_book_prop','infant_partner_prop','parent_book_prop','parent_partner_prop','ja_given_book','ja_chance','ja_excess','ja_ratio','session_s','n_utts_kept','gestures_per_min']:
    d=c.dropna(subset=[v]); z=(d[v]-d[v].mean())/d[v].std()
    out=d[abs(z)>=3]
    dd=d[abs(z)<3]
    a=pg.anova(dd,dv=v,between='age')
    tk=pg.pairwise_tukey(dd,dv=v,between='age')
    kw=stats.kruskal(*[g[v] for _,g in dd.groupby('age')])
    dn=sp.posthoc_dunn(dd,val_col=v,group_col='age',p_adjust='bonferroni')
    m=dd.groupby('age_group')[v].agg(['mean','std']).round(3)
    sw=stats.shapiro(smf.ols(f'{v}~C(age)',dd).fit().resid).pvalue
    print(f'\n### {v}  outliers removed={len(out)} ids={list(out.dyad_id)}')
    print(m.T.to_string())
    print(f"F({a['ddof1'][0]},{a['ddof2'][0]})={a['F'][0]:.2f} p={a['p_unc'][0] if 'p_unc' in a else a['p-unc'][0]:.4f} np2={a['np2'][0]:.3f}  SW p={sw:.4f}  KW H={kw.statistic:.2f} p={kw.pvalue:.4f}")
    print(tk[['A','B','diff','p_tukey']].round(4).to_string(index=False))
    print('Dunn bonf:', dn.round(4).to_dict())
# controls
print('\n=== controls')
for v in ['book_looking_prop','joint_book_prop','infant_partner_prop','n_utts_kept','gestures_per_min']:
    m=smf.ols(f'{v}~C(infant_sex)+C(parent_sex)+C(book)',c).fit()
    print(v, m.pvalues.round(3).to_dict())
# age continuous? no age in days given
print(c.columns.tolist())
c.to_csv('cov_derived.csv',index=False)
