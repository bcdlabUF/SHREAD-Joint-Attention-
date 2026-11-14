"""How many times parents read the book, and whether re-reading (vs. time in session) explains page-type effects.
A 'reading' starts each time the dyad enters the individual-label pages (the book's first pages) after being elsewhere;
a reading is 'complete' when it reaches the no-label pages. Uses page_condition_spans.csv (page_condition.py).
-> page_rereading_spans.csv, page_rereading_dyad.csv; printed GEE models."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm, statsmodels.formula.api as smf
S = pd.read_csv('page_condition_spans.csv').sort_values(['dyad', 't_mid']).reset_index(drop=True)
P = ['individual', 'category', 'none']
passes = []
for d, g in S.groupby('dyad', sort=False):
    k = 0; prev = None
    for pg_ in g.page:
        if pg_ == 'individual' and prev != 'individual': k += 1
        if k == 0: k = 1                     # (a session that does not start on the first pages)
        passes.append(k); prev = pg_
S['reading'] = passes
D = S.groupby('dyad').agg(age=('age', 'first'), readings=('reading', 'max'),
                          minutes=('t_mid', 'max')).reset_index()
comp = S[S.page == 'none'].groupby('dyad').reading.nunique(); D['complete'] = D.dyad.map(comp).fillna(0).astype(int)
D.to_csv('page_rereading_dyad.csv', index=False)
print('== Times the book was read (starts of the individual-label pages)')
print(D.readings.value_counts().sort_index().to_string()); print(D.groupby('age').readings.agg(['mean', 'std', 'min', 'max']).round(2))
print('\n== Complete readings (reached the no-label pages)'); print(D.complete.value_counts().sort_index().to_string())
print(f'Read more than once: {(D.readings > 1).sum()} of {len(D)} dyads')
from scipy import stats; print('readings by age: Kruskal-Wallis H = %.2f, p = %.3f' % stats.kruskal(*[g.readings for _, g in D.groupby('age')]))
S['reread'] = (S.reading > 1).astype(int); S['rd'] = S.reading.clip(upper=3)
S.to_csv('page_rereading_spans.csv', index=False)
X = S[S.secs >= 3].copy(); X['page'] = pd.Categorical(X.page, P); X['age'] = X.age.astype(str)
def wald(fit, pre):
    nm = [n for n in fit.params.index if n.startswith(pre)]; L = np.zeros((len(nm), len(fit.params)))
    for i, n in enumerate(nm): L[i, list(fit.params.index).index(n)] = 1
    t = fit.wald_test(L, scalar=True); return float(t.statistic), float(t.pvalue), len(nm)
print('\n== Span-level GEE (Gaussian, exchangeable, clustered by dyad, weighted by duration); k = %d spans' % len(X))
for v, lab in [('ib', 'Infant book looking'), ('sa', 'Sustained attention'), ('pb', 'Parent book looking'), ('ja', 'Joint attention')]:
    print(f'\n-- {lab}')
    for f in [f'{v} ~ page + age', f'{v} ~ page + age + t_mid', f'{v} ~ page + age + reread', f'{v} ~ page + age + t_mid + reread']:
        fit = smf.gee(f, 'dyad', X, family=sm.families.Gaussian(), cov_struct=sm.cov_struct.Exchangeable(), weights=X.secs).fit()
        c, p, df = wald(fit, 'page'); extra = ''
        for term in ['t_mid', 'reread']:
            if term in fit.params: extra += f'  {term}: b = {fit.params[term]:.3f}, p = {fit.pvalues[term]:.4f}'
        print(f'  {f:<40s} page chi2({df}) = {c:6.2f}, p = {p:.4f}{extra}')
    # page-type effect within the first reading only, and within re-readings only
    for nm_, sub in [('first reading only', X[X.reread == 0]), ('re-readings only', X[X.reread == 1])]:
        sub = sub[sub.dyad.isin(sub.groupby('dyad').page.nunique().pipe(lambda s: s[s >= 2].index))]
        fit = smf.gee(f'{v} ~ page + age', 'dyad', sub, family=sm.families.Gaussian(), cov_struct=sm.cov_struct.Exchangeable(), weights=sub.secs).fit()
        c, p, df = wald(fit, 'page'); print(f'  {nm_:<40s} page chi2({df}) = {c:6.2f}, p = {p:.4f}  (k = {len(sub)}, dyads = {sub.dyad.nunique()}); '
                                            f'none - individual = {fit.params.get("page[T.none]", np.nan):.3f}')
