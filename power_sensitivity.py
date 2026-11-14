"""Sensitivity power analysis (alpha = .05, power = .80) for the age ANOVAs and the page type x age mixed ANOVAs.
Noncentral F, using G*Power's effect-size conventions for repeated-measures designs (Faul et al., 2009).
Run from results/ (reads page_condition_dyad.csv for the observed correlation across page types)."""
import numpy as np, pandas as pd
from scipy import stats, optimize
def power(lam, df1, df2, a=.05): return 1 - stats.ncf.cdf(stats.f.ppf(1 - a, df1, df2), df1, df2, lam)
def min_f(fn): return optimize.brentq(lambda f: fn(f) - .80, .01, 3)
N, k, m = 60, 3, 3
R = pd.read_csv('page_condition_dyad.csv')
rs = [R.pivot(index='dyad', columns='page', values=v).dropna().corr().values[np.triu_indices(3, 1)].mean() for v in ['ib', 'pb', 'ja', 'sa']]
rho = float(np.mean(rs)); print('mean correlation across page types: %.2f' % rho)
e2 = lambda f: f * f / (1 + f * f)
f = min_f(lambda f: power(f * f * N, k - 1, N - k)); print('one-way age ANOVA: f = %.2f, eta2 = %.2f' % (f, e2(f)))
f = min_f(lambda f: power(f * f * N * m / (1 - rho), m - 1, (N - k) * (m - 1))); print('page type (within): f = %.2f, eta_p2 = %.3f' % (f, e2(f)))
f = min_f(lambda f: power(f * f * N * m / (1 - rho), (m - 1) * (k - 1), (N - k) * (m - 1))); print('page type x age: f = %.2f, eta_p2 = %.3f' % (f, e2(f)))
