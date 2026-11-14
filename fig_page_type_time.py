"""Figure S2: page type vs. time on task (span-level GEE).
(A) Infant book looking across the session, by page type: each dot is one stretch on one page type; lines are GEE fits.
(B-D) Estimated means by page type without and with session time in the model, for infant book looking, parent book looking,
and joint visual attention. Gaussian GEE, exchangeable working correlation, clustered by dyad, weighted by stretch duration."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm, statsmodels.formula.api as smf
INK = '#0b0b0b'; INK2 = '#52514e'; GRID = '#e4e3df'
PC = {'individual': '#2a78d6', 'category': '#1f9d55', 'none': '#eb6834'}; PL = {'individual': 'Individual label', 'category': 'Category label', 'none': 'No label'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .8, 'savefig.dpi': 400})
P = ['individual', 'category', 'none']
X = pd.read_csv('page_condition_spans.csv'); X = X[X.secs >= 3].copy(); X['page'] = pd.Categorical(X.page, P); X['age'] = X.age.astype(str)
import patsy
def fit(f, v):
    r = smf.gee(f, 'dyad', X, family=sm.families.Gaussian(), cov_struct=sm.cov_struct.Exchangeable(), weights=X.secs).fit()
    r.di = patsy.dmatrices(f, X, return_type='dataframe')[1].design_info; return r
def wald(fr):
    nm = [n for n in fr.params.index if n.startswith('page')]; L = np.zeros((len(nm), len(fr.params)))
    for i, n in enumerate(nm): L[i, list(fr.params.index).index(n)] = 1
    t = fr.wald_test(L, scalar=True); return float(t.statistic), float(t.pvalue)
def emm(fr, tvalue=None):
    """estimated mean for each page type, averaged over age groups (weighted by their share of stretches), at the mean session time"""
    out = []
    for p in P:
        G = X.copy(); G['page'] = pd.Categorical([p] * len(G), P)
        if tvalue is not None: G['t_mid'] = tvalue
        dm = patsy.dmatrix(fr.di, G, return_type='dataframe')
        w = (X.secs / X.secs.sum()).values; xbar = (dm.values * w[:, None]).sum(0)
        m = xbar @ fr.params.values; se = np.sqrt(xbar @ fr.cov_params().values @ xbar)
        out.append((m, 1.96 * se))
    return out
fp = lambda p: 'p < .001' if p < .001 else 'p = ' + f'{p:.3f}'[1:]
fig, axs = plt.subplots(1, 4, figsize=(14.5, 3.9), gridspec_kw=dict(wspace=.45, width_ratios=[1.45, 1.1, 1.1, 1.1]))
# A: infant looking across the session
ax = axs[0]; f1 = fit('ib ~ page + age + t_mid', 'ib'); tm = np.linspace(0, X.t_mid.quantile(.98), 50)
for p in P:
    s = X[X.page == p]; ax.scatter(s.t_mid, s.ib, s=np.clip(s.secs, 3, 60) * .5, color=PC[p], alpha=.25, lw=0)
    rng = s.t_mid.quantile([.02, .98]).values; tt = tm[(tm >= rng[0]) & (tm <= rng[1])]
    import patsy
    G = pd.DataFrame({'page': pd.Categorical([p] * len(tt), P), 'age': X.age.mode()[0], 't_mid': tt})
    agew = X.groupby('age').secs.sum() / X.secs.sum(); yy = 0
    for a, wa in agew.items():
        G['age'] = a; yy = yy + wa * (patsy.dmatrix(f1.di, G, return_type='dataframe').values @ f1.params.values)
    ax.plot(tt, yy, color=PC[p], lw=2.2, label=PL[p] + ' pages')
ax.set_xlabel('Time in session (min)'); ax.set_ylabel('Infant book looking\n(proportion of stretch)'); ax.set_ylim(-.03, 1.25); ax.set_xlim(0, X.t_mid.quantile(.98) + .2)
ax.legend(frameon=False, fontsize=7, loc='upper right', ncol=1, bbox_to_anchor=(1.02, 1.02), handlelength=1.4)
ax.set_title(f'Infant looking across the session\n(decline: b = {f1.params["t_mid"]:.3f} per min, {fp(f1.pvalues["t_mid"])})', fontsize=9, loc='left')
# B-D: estimated means without and with session time
for ax, (v, title) in zip(axs[1:], [('ib', 'Infant book looking'), ('pb', 'Parent book looking'), ('ja', 'Joint visual attention')]):
    f0 = fit(f'{v} ~ page + age', v); ft = fit(f'{v} ~ page + age + t_mid', v)
    e0 = emm(f0); et = emm(ft, X.t_mid.mean()); c0, p0 = wald(f0); ct, pt = wald(ft)
    for j, p in enumerate(P):
        ax.errorbar(j - .14, e0[j][0], yerr=e0[j][1], fmt='o', mfc='white', mec=PC[p], color=PC[p], ms=6.5, mew=1.6, elinewidth=1.5, capsize=3)
        ax.errorbar(j + .14, et[j][0], yerr=et[j][1], fmt='o', color=PC[p], ms=6.5, mec='white', mew=.6, elinewidth=1.5, capsize=3)
    ax.set_xticks(range(3)); ax.set_xticklabels(['Individual', 'Category', 'No label'], fontsize=7.6); ax.set_xlim(-.5, 2.5)
    lo = min(m - c for m, c in e0 + et); hi = max(m + c for m, c in e0 + et); pad = (hi - lo)
    ax.set_ylim(lo - pad * .25, hi + pad * .9); ax.set_ylabel('Estimated proportion'); ax.set_title(title, fontsize=9, loc='left')
    ax.text(.03, .97, f'Page type, without time: χ²(2) = {c0:.2f}, {fp(p0)}\nPage type, with time: χ²(2) = {ct:.2f}, {fp(pt)}', transform=ax.transAxes, va='top', fontsize=6.4, color=INK2)
    ax.grid(axis='y', color=GRID, lw=.6); ax.set_axisbelow(True)
h = [plt.Line2D([], [], marker='o', ls='', mfc='white', mec=INK2, color=INK2, mew=1.4, ms=6), plt.Line2D([], [], marker='o', ls='', color=INK2, ms=6)]
axs[3].legend(h, ['Without session time', 'Controlling for session time'], frameon=False, fontsize=7, loc='lower right')
for ax, L in zip(axs, 'ABCD'): ax.text(-.2 if L == 'A' else -.3, 1.07, L, transform=ax.transAxes, fontsize=12, fontweight='bold')
fig.text(.5, -.1, f'Each continuous stretch on one page type of at least 3 s (k = {len(X)} stretches, 60 dyads) entered a GEE (Gaussian, exchangeable working correlation, clustered by dyad, weighted by stretch duration) with page type and age group, '
         'with and without session time.\nA: dots are stretches (size = duration); lines are model fits averaged over age groups. B–D: estimated means (95% CIs) averaged over age groups, at the mean session time when time is in the model.',
         ha='center', fontsize=6.8, color=INK2)
fig.savefig('../figures/FigS2_page_type_time.png', bbox_inches='tight', facecolor='white'); print('ok')
