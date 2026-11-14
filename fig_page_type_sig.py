"""Figure 7: significant page-type effects only (pooled across age; no page type x age interactions):
parent book looking, joint attention, and joint attention above chance, with Holm-corrected paired comparisons."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats as st
import pingouin as pg
INK='#0b0b0b'; INK2='#52514e'; GRID='#e4e3df'
PC={'individual':'#2a78d6','category':'#1f9d55','none':'#eb6834'}; PL={'individual':'Individual\nlabel','category':'Category\nlabel','none':'No label\n(no text)'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.edgecolor':INK2,'axes.labelcolor':INK,'xtick.color':INK2,'ytick.color':INK2,
  'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.8,'savefig.dpi':400})
R=pd.read_csv('page_condition_dyad.csv'); P=['individual','category','none']
def ci(x): x=np.asarray(x); return st.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
fp=lambda p: 'p < .001' if p<.001 else 'p = '+f'{p:.3f}'[1:]
panels=[('pb','Parent looking at the book','Proportion of page time',(0,1.3)),
        ('ja','Joint attention','Proportion of page time',(0,1.3)),
        ('ja_exc','Joint attention above chance','Observed minus chance',(-.2,.42))]
rng=np.random.default_rng(3)
fig,axs=plt.subplots(1,3,figsize=(9,3.6)); fig.subplots_adjust(wspace=.42)
for ax,(v,title,yl,ylim),L in zip(axs,panels,'ABC'):
    W=R.pivot(index='dyad',columns='page',values=v)[P].dropna()
    for j,p in enumerate(P):
        ax.scatter(j+rng.uniform(-.17,.17,len(W)),W[p],s=7,color=PC[p],alpha=.3,linewidths=0,zorder=1)
        m=W[p].mean(); ax.errorbar(j,m,yerr=ci(W[p]),fmt='o',ms=8,color=PC[p],mec='white',mew=.8,elinewidth=2,capsize=3.5,zorder=4)
        ax.text(j+.2,m,f'{m:.2f}'.replace('0.','.'),va='center',fontsize=7.5,color=PC[p],fontweight='bold',zorder=5)
    if v=='ja_exc': ax.axhline(0,color=INK,lw=1,ls=(0,(3,2))); ax.text(-.45,-.008,'chance',ha='left',va='top',fontsize=6.8,color=INK2)
    Y=R[R.dyad.isin(W.index)]; pw=pg.pairwise_tests(Y,dv=v,within='page',subject='dyad',padjust='holm')
    top=W.max().max(); span=ylim[1]-ylim[0]; k=0
    for _,r in pw.sort_values('p_corr').iterrows():
        if r.p_corr<.05:
            i0,i1=sorted([P.index(r.A),P.index(r.B)]); y=top+span*(.05+.09*k); k+=1
            ax.plot([i0,i0,i1,i1],[y-span*.015,y,y,y-span*.015],color=INK,lw=.9)
            ax.text((i0+i1)/2,y+span*.008,fp(r.p_corr),ha='center',va='bottom',fontsize=7)
    ax.set_ylim(*ylim); ax.set_xlim(-.5,2.5); ax.set_xticks(range(3)); ax.set_xticklabels([PL[p] for p in P],fontsize=7.8)
    if v!='ja_exc': ax.set_yticks([0,.2,.4,.6,.8,1.0])
    ax.set_title(title,fontsize=9.5,loc='left',color=INK); ax.set_ylabel(yl); ax.grid(axis='y',color=GRID,lw=.6); ax.set_axisbelow(True)
    ax.text(-.3,1.06,L,transform=ax.transAxes,fontsize=12,fontweight='bold')
    a=pg.mixed_anova(data=Y,dv=v,within='page',between='age',subject='dyad').set_index('Source')
    ax.text(.03,.98,'Age: '+(fp(a.loc['age','p_unc']) if a.loc['age','p_unc']<.05 else 'n.s.')+' (Figure 4)' * (v=='ja'),transform=ax.transAxes,va='top',fontsize=7,color=INK2)
fig.text(.5,-.1,'Lower on no-label pages than on both labeled pages; labeled pages did not differ. No page type × age interactions (all ps ≥ .075); age effects are shown in Figure 4.\n'
         'Small dots: dyads (N = 60), pooled across age. Large dots: means with 95% CIs. Brackets: Holm-corrected paired t tests; only significant pairs shown.',
         ha='center',fontsize=7,color=INK2)
fig.savefig('../figures/Fig7_page_type.png',bbox_inches='tight',facecolor='white'); print('ok')
