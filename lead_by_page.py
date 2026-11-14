"""Who led joint attention within each page type, as excess over circular-shift chance
(parent series shifted >= 20 s, 200 shifts), matching the whole-session who-led analysis (chance_unified.py).
Bouts are assigned to the page type at bout onset; at least three led bouts per page type. -> lead_by_page.csv"""
import pickle, numpy as np, pandas as pd
from jadef import yu_ja, _runs, FS
S = pickle.load(open('series.pkl', 'rb')); c = pd.read_csv('../data/covariates.csv'); age = dict(zip(c.dyad_id, c.age_group))
l = pd.read_csv('../data/look_spans.csv'); PAGES = ['individual', 'category', 'none']
def cond_vec(d, n, t0):
    v = np.full(n, 'break', dtype=object)
    for on, off, val in l[(l.dyad_id == d) & (l.stream == 'condition')][['onset_ms', 'offset_ms', 'value']].values:
        v[max(int((on - t0) * FS / 1000), 0):min(int(np.ceil((off - t0) * FS / 1000)), n)] = val
    return v
def runstart(x):
    d = np.diff(np.r_[0, (x > 0).astype(int)]); starts = np.where(d == 1)[0]
    idx = np.zeros(len(x), int); idx[starts] = 1; sv = np.r_[-1, starts][np.cumsum(idx)]; sv[x <= 0] = -1; return sv
def leads(ib, pb, cv):
    """infant-led share of bouts by page type"""
    ja = yu_ja(ib, pb); si, sp = runstart(ib), runstart(pb); on, off = _runs(ja); out = {p: [] for p in PAGES}
    for a, b in zip(on, off):
        k = a + np.argmax((ib[a:b] * pb[a:b]) > 0); lag = ((k - si[k]) - (k - sp[k])) / FS
        if abs(lag) > .1 and cv[a] in out: out[cv[a]].append(lag > 0)
    return out
rng = np.random.default_rng(11); rows = []
for d, s in S.items():
    n = s['n']; ib = s['infant_book'].astype(float); pb = s['parent_book'].astype(float); cv = cond_vec(d, n, s['t0'])
    obs = leads(ib, pb, cv); ch = {p: [] for p in PAGES}
    for k in rng.integers(20 * FS, n - 20 * FS, 200):
        r = leads(ib, np.roll(pb, k), cv)
        for p in PAGES:
            if len(r[p]): ch[p].append(np.mean(r[p]))
    for p in PAGES:
        rows.append(dict(dyad=d, age=age[d], page=p, n_bouts=len(obs[p]),
                         inf_led=np.mean(obs[p]) if len(obs[p]) >= 3 else np.nan,
                         inf_led_chance=np.mean(ch[p]) if ch[p] else np.nan))
    print(d, end=' ', flush=True)
R = pd.DataFrame(rows); R['inf_led_exc'] = R.inf_led - R.inf_led_chance; R.to_csv('lead_by_page.csv', index=False)
print('\n', R.groupby(['age', 'page'])[['inf_led', 'inf_led_chance', 'inf_led_exc']].mean().round(3))
