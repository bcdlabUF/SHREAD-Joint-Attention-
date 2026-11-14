"""Figure S1: attention, joint attention, and coordination by page type within each age group."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats as st
import pingouin as pg
INK='#0b0b0b'; INK2='#52514e'; GRID='#e4e3df'
PC={'individual':'#2a78d6','category':'#1f9d55','none':'#eb6834'}; PL={'individual':'Individual label','category':'Category label','none':'No label'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.edgecolor':INK2,'axes.labelcolor':INK,'xtick.color':INK2,'ytick.color':INK2,
  'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.8,'savefig.dpi':400})
R=pd.read_csv('page_condition_dyad.csv'); P=['individual','category','none']; AGES=[6,9,12]; OFF={'individual':-.26,'category':0,'none':.26}
def ci(x): x=np.asarray(x); x=x[~np.isnan(x)]; return st.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
fp=lambda p: 'p < .001' if p<.001 else 'p = '+f'{p:.3f}'[1:]
panels=[('ib','Infant looking at the book','Proportion of page time',None),
        ('pb','Parent looking at the book','Proportion of page time',None),
        ('ja','Joint attention','Proportion of page time','ja_chance'),
        ('xc','Coordination','Mean cross-correlation (Fisher z)','xc_chance')]
rng=np.random.default_rng(3)
fig,axs=plt.subplots(1,4,figsize=(12,3.7)); fig.subplots_adjust(wspace=.38)
for ax,(v,title,yl,chv),L in zip(axs,panels,'ABCD'):
    W=R.pivot(index='dyad',columns='page',values=v)[P].dropna(); Y=R[R.dyad.isin(W.index)]
    for i,a in enumerate(AGES):
        Ya=Y[Y.age==a]
        for p in P:
            x=i+OFF[p]; vals=Ya[Ya.page==p][v].values
            ax.scatter(x+rng.uniform(-.07,.07,len(vals)),vals,s=6,color=PC[p],alpha=.3,linewidths=0,zorder=1)
            ax.errorbar(x,vals.mean(),yerr=ci(vals),fmt='o',ms=5.5,color=PC[p],mec='white',mew=.6,elinewidth=1.6,capsize=2.5,zorder=4)
            if chv:
                ch=Ya[Ya.page==p][chv].mean(); ax.plot([x-.11,x+.11],[ch,ch],color=INK,lw=1.1,ls=(0,(2.5,1.5)),zorder=3)
    a=pg.mixed_anova(data=Y,dv=v,within='page',between='age',subject='dyad',correction=True).set_index('Source')
    pc='p_unc'; gg=pg.sphericity(Y,dv=v,within='page',subject='dyad').pval<.05; eps=a.loc['page','eps']
    if gg:  # Greenhouse-Geisser where Mauchly's test p < .05
        a.loc['page',pc]=a.loc['page','p_GG_corr']
        a.loc['Interaction',pc]=st.f.sf(a.loc['Interaction','F'],a.loc['Interaction','DF1']*eps,a.loc['Interaction','DF2']*eps)
    ax.text(.03,.985,f'Page type: {fp(a.loc["page",pc])}\nAge: {fp(a.loc["age",pc])}\nPage type × age: {fp(a.loc["Interaction",pc])}',
            transform=ax.transAxes,va='top',fontsize=6.8,color=INK2,linespacing=1.35)
    prop=v in ('ib','pb','ja')
    if prop: ax.set_ylim(-.03,1.38); ax.set_yticks([0,.2,.4,.6,.8,1.0])
    else: ax.set_ylim(Y[v].min()-.03,Y[v].max()+.2)
    ax.set_xticks(range(3)); ax.set_xticklabels([str(a) for a in AGES]); ax.set_xlim(-.55,2.55)
    for b in (.5,1.5): ax.axvline(b,color=GRID,lw=.6,zorder=0)
    ax.set_title(title,fontsize=9,loc='left',color=INK); ax.set_xlabel('Age (months)'); ax.set_ylabel(yl); ax.grid(axis='y',color=GRID,lw=.6); ax.set_axisbelow(True)
    ax.text(-.3,1.08,L,transform=ax.transAxes,fontsize=12,fontweight='bold')
axs[0].text(.03,.80,'Page type n.s. controlling\nfor time in session',transform=axs[0].transAxes,va='top',fontsize=6.6,color=INK2,style='italic')
h=[plt.Line2D([],[],color=PC[p],marker='o',ls='',ms=6) for p in P]+[plt.Line2D([],[],color=INK,ls=(0,(2.5,1.5)),lw=1.1)]
fig.legend(h,[PL[p]+' pages' for p in P]+['Circular-shift chance'],loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.5,1.05),fontsize=7.8)
fig.text(.5,-.08,'Small dots: dyads (6 months, n = 20; 9 months, n = 23; 12 months, n = 17; one 6-month dyad missing in D). Large dots: means with 95% CIs.\n'
         'p values: 3 (page type) × 3 (age group) mixed ANOVAs (A: Greenhouse–Geisser corrected).',ha='center',fontsize=6.8,color=INK2)
fig.savefig('../figures/FigS1_page_type_by_age.png',bbox_inches='tight',facecolor='white'); print('ok')
