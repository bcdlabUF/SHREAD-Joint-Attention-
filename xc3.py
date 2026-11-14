import pandas as pd, numpy as np, pingouin as pg, scikit_posthocs as sp
from scipy import stats
import statsmodels.formula.api as smf
d=pd.read_csv('xc_dyad.csv'); d['ag']=d.age.astype(str)
for pair in ['book','face','speech','gesture']:
    x=d[d.pair==pair].copy()
    print(f'\n================ {pair}  n={len(x)}  per age={x.age.value_counts().sort_index().to_dict()}')
    x['exc_circ']=x.z_all-x.circ; x['exc_pseudo']=x.z_all-x.pseudo; x['asym']=x.z_inf_lead-x.z_par_lead
    for v in ['z_all','z0','circ','pseudo','exc_circ','exc_pseudo','asym']:
        a=pg.anova(x,dv=v,between='ag'); kw=stats.kruskal(*[g[v] for _,g in x.groupby('age')])
        m=x.groupby('age')[v].agg(['mean','sem']).round(4)
        tk=pg.pairwise_tukey(x,dv=v,between='ag')[['A','B','p_tukey']].round(3).values.tolist()
        print(f"{v:11s} M/SE {dict(zip(m.index,zip(m['mean'],m['sem'])))} F({a.ddof1[0]},{a.ddof2[0]})={a.F[0]:.2f} p={a.iloc[0,4]:.4f} KW p={kw.pvalue:.4f} tukey={tk}")
    # vs zero
    for v in ['exc_circ','exc_pseudo']:
        tt=stats.ttest_1samp(x[v],0); w=stats.wilcoxon(x[v])
        print(f'  {v} overall mean={x[v].mean():.4f} t={tt.statistic:.2f} p={tt.pvalue:.2g} wilcoxon p={w.pvalue:.2g}; by age>0 p:',
              {a:round(stats.wilcoxon(g[v]).pvalue,4) for a,g in x.groupby('age')})
    # dyad-level order x age
    lg=x.melt(id_vars=['dyad','ag'],value_vars=['z_inf_lead','z_par_lead'],var_name='order',value_name='z')
    ma=pg.mixed_anova(lg,dv='z',within='order',between='ag',subject='dyad')
    print(ma[['Source','DF1','DF2','F','p_unc','np2']].round(4).to_string(index=False))
    for a,g in x.groupby('age'):
        tt=stats.ttest_rel(g.z_inf_lead,g.z_par_lead); print(f'   age {a}: inf_lead {g.z_inf_lead.mean():.4f} par_lead {g.z_par_lead.mean():.4f} paired t p={tt.pvalue:.4f}; median peak lag={g.peak_lag.median():.2f}s')
    x.to_csv(f'xc_dyad_{pair}.csv',index=False)
