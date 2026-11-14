"""Page-type statistics (Results 'Page Type', Figure 7, Supplement S7 / Table S1).
Inputs: page_condition_dyad.csv and page_condition_spans.csv (page_condition.py); lead_by_page.csv (lead_by_page.py).
1) 3 (page type, within) x 3 (age, between) mixed ANOVAs per measure, dyads with all three page types;
   Greenhouse-Geisser correction where Mauchly's test p < .05; Friedman tests as a nonparametric check.
2) Holm-corrected paired t-tests between page types.
3) Span-level GEE (Gaussian, exchangeable, clustered by dyad, weighted by span duration):
   page type + age group, with and without session time (minutes at span midpoint); Wald chi2 for page type.
Writes page_condition_table.csv (Table S1)."""
import numpy as np, pandas as pd, pingouin as pg, warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
R = pd.read_csv('page_condition_dyad.csv'); SP = pd.read_csv('page_condition_spans.csv')
R = R.merge(pd.read_csv('lead_by_page.csv')[['dyad', 'page', 'inf_led_exc']], on=['dyad', 'page'], how='left')
P = ['individual', 'category', 'none']
M = [('ib', 'Infant book looking'), ('sa', 'Infant sustained attention'), ('pb', 'Parent book looking'),
     ('ja', 'Joint attention'), ('ja_exc', 'Joint attention above chance'), ('xc', 'Coordination (Fisher z)'),
     ('inf_led_exc', 'Bouts led by infant, above chance'), ('g_pi_exc', 'Granger, parent -> infant'), ('g_ip_exc', 'Granger, infant -> parent')]
fp = lambda p: '< .001' if p < .001 else ('= ' + f'{p:.3f}'.lstrip('0'))
rows = []
for v, lab in M:
    d = R[['dyad', 'age', 'page', v]].dropna()
    keep = d.groupby('dyad').page.nunique(); d = d[d.dyad.isin(keep[keep == 3].index)]
    a = pg.mixed_anova(data=d, dv=v, within='page', between='age', subject='dyad', correction=True)
    pt = a[a.Source == 'page'].iloc[0]; ix = a[a.Source == 'Interaction'].iloc[0]
    w = d.pivot(index='dyad', columns='page', values=v)[P]
    sph = pg.sphericity(d, dv=v, within='page', subject='dyad'); gg = sph.pval < .05; eps = pt.eps
    p_page = pt.p_GG_corr if gg else pt.p_unc
    p_int = stats.f.sf(ix.F, ix.DF1 * eps, ix.DF2 * eps) if gg else ix.p_unc
    res = w.sub(w.mean(axis=1), axis=0).values.ravel(); sw = stats.shapiro(res).pvalue
    fr = stats.friedmanchisquare(*[w[k] for k in P])
    pairs = [('individual', 'category'), ('individual', 'none'), ('category', 'none')]
    raw = [stats.ttest_rel(w[x], w[y]).pvalue for x, y in pairs]; holm = multipletests(raw, method='holm')[1]
    print(f'\n== {lab} (n = {w.shape[0]})')
    print(f'  Mauchly W = {sph.W:.3f}, p = {sph.pval:.4f}; GG epsilon = {eps:.3f}' + ('  -> Greenhouse-Geisser applied' if gg else ''))
    print(f'  page: F({pt.DF1:.0f}, {pt.DF2:.0f}) = {pt.F:.2f}, p = {p_page:.4f}, np2 = {pt.np2:.3f};  page x age F = {ix.F:.2f}, p = {p_int:.4f}')
    print(f'  Shapiro-Wilk on residuals p = {sw:.4f};  Friedman chi2(2) = {fr.statistic:.2f}, p = {fr.pvalue:.4f}')
    for (x, y), r_, h in zip(pairs, raw, holm): print(f'  {x} vs {y}: p = {r_:.4f}, Holm p = {h:.4f}')
    fmt = (lambda s: f'{s.mean():.2f} ({s.std():.2f})') if v.startswith('g_') else (lambda s: f'{s.mean():.2f} ({s.std():.2f})'.replace('0.', '.'))
    rows.append({'Measure': lab, 'n': w.shape[0], **{k: fmt(w[k]) for k in P},
                 'Page type': f'F({pt.DF1:.0f}, {pt.DF2:.0f}) = {pt.F:.2f}, p {fp(p_page)}' + (' (GG)' if gg else ''),
                 'Page x age p': f'{p_int:.3f}'.lstrip('0'), 'Friedman': f'chi2(2) = {fr.statistic:.2f}, p {fp(fr.pvalue)}',
                 'shapiro_p': round(sw, 4), 'mauchly_p': round(float(sph.pval), 4), 'gg_eps': round(eps, 3)})
pd.DataFrame(rows).to_csv('page_condition_table.csv', index=False)

# ---- separating page type from time in the session
SP = SP[SP.secs >= 3].copy(); SP['page'] = pd.Categorical(SP.page, P); SP['age'] = SP.age.astype(str)
rev = SP.groupby('dyad').page.apply(lambda s: (s != s.shift()).sum() > s.nunique()).sum()
print(f'\n== Span-level GEE: k = {len(SP)} spans, {rev} of {SP.dyad.nunique()} dyads returned to an earlier page type')
def wald(fit):
    names = [n for n in fit.params.index if n.startswith('page')]; L = np.zeros((len(names), len(fit.params)))
    for i, n in enumerate(names): L[i, list(fit.params.index).index(n)] = 1
    t = fit.wald_test(L, scalar=True); return float(t.statistic), float(t.pvalue)
for v, lab in [('ib', 'Infant book looking'), ('sa', 'Infant sustained attention'), ('pb', 'Parent book looking'), ('ja', 'Joint attention')]:
    out = []
    for f in [f'{v} ~ page + age', f'{v} ~ page + age + t_mid']:
        fit = smf.gee(f, 'dyad', SP, family=sm.families.Gaussian(), cov_struct=sm.cov_struct.Exchangeable(), weights=SP.secs).fit()
        out.append((wald(fit), fit))
    (c0, p0), _ = out[0]; (c1, p1), fit = out[1]
    print(f'  {lab}: without time chi2(2) = {c0:.2f}, p = {p0:.4f};  with time chi2(2) = {c1:.2f}, p = {p1:.4f};  '
          f'time b = {fit.params["t_mid"]:.3f}/min, p = {fit.pvalues["t_mid"]:.4f};  none vs individual b = {fit.params["page[T.none]"]:.3f}, p = {fit.pvalues["page[T.none]"]:.4f}')
