"""Summary statistics reported in the Results that are not printed by the analysis scripts.

Reads the per-dyad files written earlier in run_all.sh (in results/) and prints, in Results order:
infant attention by age, joint attention vs chance, book-looking coordination, who led joint
attention, windowed cross-correlation (coupling and lead changes), gaze following, and face looking.

Conventions (see Statistical Analyses): one-way ANOVAs with partial eta squared, Tukey HSD between
ages; Kruskal-Wallis + Bonferroni-corrected Dunn where residuals are not normal; Holm correction
for the same comparison with chance run at each age; paired t tests for observed vs chance.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, pingouin as pg, scikit_posthocs as sp
from scipy import stats
from statsmodels.stats.multitest import multipletests

AGES = [6, 9, 12]
def hdr(s): print('\n' + '=' * 78 + '\n' + s + '\n' + '=' * 78)

def age_anova(d, v, label=None, drop_outliers=False):
    d = d[['age', v]].dropna().copy(); d['ag'] = d.age.astype(str)
    note = ''
    if drop_outliers:
        z = (d[v] - d[v].mean()) / d[v].std(ddof=1); out = (z.abs() >= 3).sum()
        d = d[z.abs() < 3]; note = f' [{out} outlier(s) |z|>=3 removed]'
    a = pg.anova(d, dv=v, between='ag', detailed=False)
    p = a['p_unc' if 'p_unc' in a else 'p-unc'][0]
    m = d.groupby('age')[v].agg(['mean', 'std', 'sem', 'count']).round(3)
    sw = stats.shapiro(d[v] - d.groupby('age')[v].transform('mean')).pvalue  # residuals
    print(f'\n{label or v}{note}: F({a.ddof1[0]}, {a.ddof2[0]}) = {a.F[0]:.2f}, p = {p:.3f}, np2 = {a.np2[0]:.3f};'
          f' Shapiro-Wilk p = {sw:.3f}')
    print(m.to_string())
    tk = pg.pairwise_tukey(d, dv=v, between='ag')[['A', 'B', 'diff', 'p_tukey']].round(4)
    print('  Tukey:', tk.to_dict('records'))
    if sw < .05:
        kw = stats.kruskal(*[d[d.age == a_][v] for a_ in AGES])
        dn = sp.posthoc_dunn(d, val_col=v, group_col='age', p_adjust='bonferroni').round(4)
        print(f'  Kruskal-Wallis H(2) = {kw.statistic:.2f}, p = {kw.pvalue:.3f}; Dunn (Bonferroni):', dn.to_dict())
    return d

def vs_chance(d, obs, ch, label):
    x = d[[obs, ch, 'age']].dropna()
    t = stats.ttest_rel(x[obs], x[ch])
    print(f'\n{label}: observed M = {x[obs].mean():.3f} vs chance {x[ch].mean():.3f}, t({len(x)-1}) = {t.statistic:.2f}, p = {t.pvalue:.4f}')
    ps = []; rows = []
    for a in AGES:
        g = x[x.age == a]; tt = stats.ttest_rel(g[obs], g[ch]); ps.append(tt.pvalue)
        rows.append((a, round(g[obs].mean(), 3), round(g[ch].mean(), 3), round(tt.statistic, 2)))
    holm = multipletests(ps, method='holm')[1]
    for (a, o, c_, t_), p_, h in zip(rows, ps, holm):
        print(f'  {a:>2} mo: {o} vs {c_}, t = {t_}, p = {p_:.4f}, Holm p = {h:.4f}')
    long = pd.concat([x.assign(cond='obs', val=x[obs]), x.assign(cond='chance', val=x[ch])])
    long['subj'] = np.tile(np.arange(len(x)), 2); long['ag'] = long.age.astype(str)
    mx = pg.mixed_anova(long, dv='val', within='cond', between='ag', subject='subj')
    r = mx[mx.Source == 'Interaction'].iloc[0]
    print(f'  condition x age: F({int(r.DF1)}, {int(r.DF2)}) = {r.F:.2f}, p = {r["p_unc" if "p_unc" in r else "p-unc"]:.3f}')

FA = pd.read_csv('final_age.csv')
COV = pd.read_csv('cov_derived.csv').rename(columns={'dyad_id': 'dyad'})

hdr('INFANT ATTENTION BY AGE (Figure 3)')
COV['age'] = COV.age_group
age_anova(COV, 'book_looking_prop', 'Infant book looking (proportion of session, from covariates.csv)')
age_anova(FA, 'sa', 'Infant sustained attention (looks >= 3 s)')
age_anova(FA, 'sa_share', 'Share of book looking that was sustained')
age_anova(COV, 'parent_book_prop', 'Parent book looking (proportion of session)')

hdr('JOINT ATTENTION (Figure 4B)')
age_anova(FA, 'ja', 'Joint attention')
vs_chance(FA, 'ja', 'ja_chance', 'Joint attention vs independent-looking chance')
age_anova(FA, 'ja_exc', 'JA excess over chance')
age_anova(FA, 'ja_ratio', 'JA observed/chance ratio', drop_outliers=True)
age_anova(FA, 'ja_given', 'Share of infant book looking in JA')
age_anova(FA, 'n_ep', 'Number of JA bouts')
age_anova(FA, 'ep_min', 'JA bouts per minute of reading time')

CU = pd.read_csv('chance_unified.csv')
vs_chance(CU, 'ja_obs', 'ja_circ', 'Joint attention vs circular-shift chance (reported)')

hdr('BOOK-LOOKING COORDINATION (Figure 4C)')
XB = pd.read_csv('xc_dyad_book.csv')
vs_chance(XB, 'z_all', 'circ', 'Coordination vs circular-shift chance (reported)')
for b in ['circ', 'pseudo']:
    t = stats.ttest_rel(XB.z_all, XB[b]); print(f'z_all vs {b}: t({len(XB)-1}) = {t.statistic:.2f}, p = {t.pvalue:.4g}')
age_anova(XB, 'z_all', 'Coordination (mean Fisher z, lags -10..+10 s)')
age_anova(XB, 'exc_circ', 'Coordination minus circular-shift chance')
age_anova(XB, 'exc_pseudo', 'Coordination minus pseudo-dyad chance')
age_anova(XB, 'z0', 'Coordination at lag 0')

hdr('RECIPROCITY: WHO LED JOINT ATTENTION (Figure 4D)')
L = pd.read_csv('lead_dyad.csv')
age_anova(L, 'p_inf', 'Proportion of JA bouts led by the infant')
ps = [stats.ttest_1samp(L[L.age == a].p_inf, .5).pvalue for a in AGES]
print('  vs .50 by age (Holm):', dict(zip(AGES, np.round(multipletests(ps, method='holm')[1], 3))))
LA = L.merge(COV[['dyad', 'book_looking_prop', 'parent_book_prop']].rename(columns={'book_looking_prop': 'ib', 'parent_book_prop': 'pb'}), on='dyad'); LA['ag'] = LA.age.astype(str)
print('  partial r(p_inf, infant book looking | parent book looking) =',
      round(pg.partial_corr(LA, x='p_inf', y='ib', covar='pb').r.iloc[0], 2))
anc = pg.ancova(LA, dv='p_inf', between='ag', covar=['ib', 'pb'])
r = anc[anc.Source == 'ag'].iloc[0]; print(f'  ANCOVA (controls: infant and parent book looking): F({int(r.DF)}, {int(anc.DF.iloc[-1])}) = {r.F:.2f}, p = {r["p_unc" if "p_unc" in r else "p-unc"]:.3f}')

LC = L.merge(CU[['dyad', 'p_inf_circ']], on='dyad')
vs_chance(LC, 'p_inf', 'p_inf_circ', 'Proportion infant-led vs circular-shift chance (reported)')
LC['p_inf_exc'] = LC.p_inf - LC.p_inf_circ; age_anova(LC, 'p_inf_exc', 'Infant-led minus chance')

hdr('RECIPROCITY: WINDOWED CROSS-CORRELATION (Figure 4E)')
W = pd.read_csv('wxc_dyad.csv')
t = stats.ttest_rel(W.coupled, W.coupled_chance)
print(f'Coupled windows: {W.coupled.mean():.3f} vs chance {W.coupled_chance.mean():.3f}, t({len(W)-1}) = {t.statistic:.2f}, p = {t.pvalue:.4g}')
t = stats.ttest_rel(W.p_inf, W.p_par)
print(f'Within coupled windows: infant leads {W.p_inf.mean():.3f}, parent {W.p_par.mean():.3f}, simultaneous {W.p_sync.mean():.3f}; t = {t.statistic:.2f}, p = {t.pvalue:.3f}')
vs_chance(W, 'ep_sw_min', 'ep_sw_min_chance', 'Lead changes per minute (new leader held >= 2 s)')
W['ep_exc'] = W.ep_sw_min - W.ep_sw_min_chance
age_anova(W, 'ep_exc', 'Lead changes per minute minus chance')
x = W[['ep_switch_prop', 'ep_switch_prop_chance']].dropna()
print(f'\nProportion of successive coupled episodes with a new leader: {x.ep_switch_prop.mean():.2f} vs chance '
      f'{x.ep_switch_prop_chance.mean():.2f}, Wilcoxon p = {stats.wilcoxon(x.ep_switch_prop, x.ep_switch_prop_chance).pvalue:.3f} (n = {len(x)} dyads with >= 2 episodes)')

hdr('RECIPROCITY: GRANGER CAUSALITY (Figure 6C)')
G = pd.read_csv('granger_dyad.csv')
for dr, lab in [('parent_to_infant', 'Parent -> infant'), ('infant_to_parent', 'Infant -> parent')]:
    x = G[G.dir == dr]; t = stats.ttest_rel(x.gc, x.gc_chance)
    print(f'{lab}: {x.gc.mean():.2f} vs chance {x.gc_chance.mean():.2f}, t({len(x)-1}) = {t.statistic:.2f}, p = {t.pvalue:.4g}')
    ps = [stats.ttest_rel(x[x.age == a].gc, x[x.age == a].gc_chance).pvalue for a in AGES]
    print('  vs chance by age (Holm):', dict(zip(AGES, np.round(multipletests(ps, method='holm')[1], 4))))
    age_anova(x, 'gc_exc', f'{lab} influence above chance')
G['ag'] = G.age.astype(str)
r = pg.mixed_anova(G, dv='gc_exc', within='dir', between='ag', subject='dyad').query("Source=='Interaction'").iloc[0]
print(f'Direction x age: F({int(r.DF1)}, {int(r.DF2)}) = {r.F:.2f}, p = {r["p_unc" if "p_unc" in r else "p-unc"]:.4f}, np2 = {r.np2:.2f}')
w = G.pivot(index='dyad', columns='dir', values='gc_exc'); w['age'] = G.groupby('dyad').age.first()
ps = []; ts = []
for a in AGES:
    y = w[w.age == a]; t = stats.ttest_rel(y.infant_to_parent, y.parent_to_infant); ps.append(t.pvalue); ts.append(f'{a} mo t({len(y)-1}) = {t.statistic:.2f}')
print('Infant->parent vs parent->infant within age (Holm):', dict(zip(ts, np.round(multipletests(ps, method='holm')[1], 3))))
print('Sensitivity to bin size and history length: see log_granger_sens.txt')
