"""Cross-correlograms of parent and infant book looking by age (lags -10 to +10 s; negative = infant looks first),
with circular-shift chance, and the age difference (12 - 6 months) at each lag."""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
INK='#0b0b0b'; INK2='#52514e'; GRID='#e4e3df'; AGEC={6:'#86b6ef',9:'#2a78d6',12:'#104281'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.edgecolor':INK2,'axes.labelcolor':INK,'xtick.color':INK2,'ytick.color':INK2,
  'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.8,'savefig.dpi':400})
L=pd.read_csv('xc_long.csv'); L=L[L.pair=='book']; X=pd.read_csv('xc_dyad_book.csv')
W=L.pivot_table(index=['dyad','age'],columns='lag',values='z').reset_index(); lags=np.array([c for c in W.columns if not isinstance(c,str)],float)
Z=W[lags].values; ages=W.age.values
def mci(M,B=2000,seed=0):
    rng=np.random.default_rng(seed); m=np.nanmean(M,0); bs=np.array([np.nanmean(M[rng.integers(0,len(M),len(M))],0) for _ in range(B)])
    return m,*np.percentile(bs,[2.5,97.5],0)
rng0=np.random.default_rng(0)
from scipy import stats as _st
ci=lambda x: x.sem()*_st.t.ppf(.975,len(x)-1)
from matplotlib.gridspec import GridSpec
fig=plt.figure(figsize=(9.0,7.6)); gs=GridSpec(2,10,wspace=2.2,hspace=.5)
axs=[fig.add_subplot(gs[0,0:4]),fig.add_subplot(gs[0,4:10]),fig.add_subplot(gs[1,1:9])]
ax=axs[0]
for j,a in enumerate([6,9,12]):
    x=X[X.age==a].z_all; ax.scatter(j+rng0.uniform(-.13,.13,len(x)),x,s=8,color=AGEC[a],alpha=.45,lw=0)
    ax.errorbar(j,x.mean(),yerr=ci(x),fmt='o',ms=6.5,color=AGEC[a],elinewidth=1.8,capsize=3,zorder=3)
    m=X[X.age==a].circ.mean(); ax.plot([j-.3,j+.3],[m,m],color=INK,lw=1.2,ls=(0,(3,2)),zorder=5)
top=X.z_all.max(); ax.plot([0,0,2,2],[top*1.04,top*1.07,top*1.07,top*1.04],color=INK2,lw=.8); ax.text(1,top*1.08,'p = .027',ha='center',va='bottom',fontsize=7.2)
ax.set_xticks([0,1,2]); ax.set_xticklabels(['6','9','12']); ax.set_xlabel('Age (months)'); ax.set_xlim(-.6,2.6); ax.set_ylim(-.1,top*1.25)
ax.set_ylabel('Mean cross-correlation (Fisher z)'); ax.grid(axis='y',color=GRID,lw=.6); ax.set_axisbelow(True)
ax.set_title('Coordination by age\n(mean across lags ±10 s)',fontsize=9,loc='left',color=INK)
ax.text(2.55,-.095,'- - chance (circular shift)',ha='right',va='bottom',fontsize=6.6,color=INK2)
ax=axs[1]
for a in [6,9,12]:
    m,lo,hi=mci(Z[ages==a],seed=a); ax.fill_between(lags,lo,hi,color=AGEC[a],alpha=.15,lw=0); ax.plot(lags,m,color=AGEC[a],lw=2,label=f'{a} months (n = {(ages==a).sum()})')
ch=X.circ.mean(); ax.axhline(ch,color=INK,lw=1.6,ls=(0,(4,2.5)),label='Chance (circular shift)',zorder=4)
ax.axvline(0,color=INK2,lw=.7)

ax.set_xlim(-10,10); ax.set_xlabel('Lag (s)'); ax.set_ylabel('Cross-correlation (Fisher z)'); ax.set_ylim(-.05,.31)
ax.text(-9.7,-.002,'← infant looks first',fontsize=7.2,color=INK2,va='bottom'); ax.text(9.7,-.002,'parent looks first →',fontsize=7.2,color=INK2,va='bottom',ha='right')
ax.legend(frameon=False,fontsize=7.3,loc='upper left')
ax.set_title('Parent–infant book looking: cross-correlation by age',fontsize=9,loc='left',color=INK)
# B: 12 - 6 difference by lag with bootstrap CI and per-lag Welch t (exploratory)
ax=axs[2]; rng=np.random.default_rng(3); A6=Z[ages==6]; A12=Z[ages==12]
d=np.nanmean(A12,0)-np.nanmean(A6,0)
bs=np.array([np.nanmean(A12[rng.integers(0,len(A12),len(A12))],0)-np.nanmean(A6[rng.integers(0,len(A6),len(A6))],0) for _ in range(2000)])
lo,hi=np.percentile(bs,[2.5,97.5],0)
ax.fill_between(lags,lo,hi,color=AGEC[12],alpha=.18,lw=0); ax.plot(lags,d,color=AGEC[12],lw=2)
ax.axhline(0,color=INK,lw=1,ls=(0,(3,2))); ax.axvline(0,color=INK2,lw=.7)
sig=lo>0
# shade contiguous lags where CI excludes 0
# colored bars along the bottom (just above the lag axis) for contiguous lags where the CI excludes 0
import matplotlib.transforms as mtr
tr=mtr.blended_transform_factory(ax.transData,ax.transAxes); inr=False
for i,s_ in enumerate(sig):
    if s_ and not inr: st=lags[i]; inr=True
    if (not s_ or i==len(sig)-1) and inr:
        ax.plot([st,lags[i]],[.025,.025],transform=tr,color='#eb6834',lw=5,solid_capstyle='butt',zorder=6); inr=False
yl=ax.get_ylim(); ax.set_ylim(yl[0]-(yl[1]-yl[0])*.08,yl[1])
ax.set_xlim(-10,10); ax.set_xlabel('Lag (s)'); ax.set_ylabel('12 months − 6 months (Fisher z)')
ax.set_title('12 vs. 6 months: difference at each lag',fontsize=9,loc='left',color=INK)
t0=stats.ttest_ind(A12[:,lags==0].ravel(),A6[:,lags==0].ravel(),equal_var=False)
ax.text(.03,.97,f'Lag 0: age effect n.s. (F(2, 57) = 1.14, p = .33)\nMean across lags: F(2, 57) = 3.64, p = .033\nOrange bars: 95% CI of the difference excludes 0',transform=ax.transAxes,fontsize=6.9,color=INK2,va='top')
for a_,l in zip(axs,'ABC'): a_.text(-.16,1.1,l,transform=a_.transAxes,fontsize=12,fontweight='bold')
fig.text(.5,-.01,'A: dots are dyads; error bars are 95% CIs; p is a Tukey comparison. B: shading is 95% bootstrap CIs across dyads.\nNegative lags: infant book looking precedes parent book looking.',ha='center',fontsize=7,color=INK2)
fig.savefig('../figures/fig_xcorr_main.png',bbox_inches='tight',facecolor='white')
print('sig lags:',lags[sig].min() if sig.any() else None, lags[sig].max() if sig.any() else None, 'frac', sig.mean())
for a in [6,9,12]: m=np.nanmean(Z[ages==a],0); print(a,'peak z',m.max().round(3),'at lag',lags[m.argmax()].round(2),'z0',m[lags==0].round(3),'z at ±5', m[np.isclose(lags,-5)].round(3), m[np.isclose(lags,5)].round(3))
