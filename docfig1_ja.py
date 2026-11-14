import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from scipy import stats as _st
ci=lambda x: x.sem()*_st.t.ppf(.975,len(x)-1)
from matplotlib.gridspec import GridSpec
OUT='../figures/'
INK='#0b0b0b'; INK2='#52514e'; GRID='#e4e3df'; GRAY='#a3a29c'
AGEC={6:'#86b6ef',9:'#2a78d6',12:'#104281'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.edgecolor':INK2,'axes.labelcolor':INK,'xtick.color':INK2,'ytick.color':INK2,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
rng=np.random.default_rng(0)
def bracket(ax,x1,x2,y,txt,h):
    ax.plot([x1,x1,x2,x2],[y-h,y,y,y-h],color=INK2,lw=.8); ax.text((x1+x2)/2,y+h*.4,txt,ha='center',va='bottom',fontsize=7.2,color=INK)
def letter(ax,l): ax.text(-0.18,1.1,l,transform=ax.transAxes,fontsize=12,fontweight='bold',ha='right')
def agedots(ax,df,v,brs,ylab=None,title=''):
    for j,a in enumerate([6,9,12]):
        x=df[df.age==a][v]; ax.scatter(j+rng.uniform(-.13,.13,len(x)),x,s=8,color=AGEC[a],alpha=.45,lw=0)
        ax.errorbar(j,x.mean(),yerr=ci(x),fmt='o',ms=6.5,color=AGEC[a],elinewidth=1.8,zorder=3)
    top=df[v].max()
    for n_,(a1,a2,t) in enumerate(brs): bracket(ax,a1,a2,top*(1.06+.12*n_),t,top*.03)
    ax.set_xticks([0,1,2]); ax.set_xticklabels(['6','9','12']); ax.set_xlabel('Age (months)'); ax.set_xlim(-.6,2.6); ax.set_ylim(min(0,df[v].min()*1.2),top*1.32)
    ax.set_title(title,fontsize=8.8,color=INK,loc='left'); ax.grid(axis='y',color=GRID,lw=.6); ax.set_axisbelow(True)
    if ylab: ax.set_ylabel(ylab)
# ---- Figure 1
C=pd.read_csv('cov_derived.csv').drop(columns=['age']).rename(columns={'age_group':'age'})
F=pd.read_csv('final_age.csv')[['dyad','ja','sa','n_ep','pb']].rename(columns={'dyad':'dyad_id'}); C=C.merge(F,on='dyad_id')
plt.rcParams['errorbar.capsize']=3; plt.rcParams['lines.markeredgewidth']=1.0
fig=plt.figure(figsize=(9.0,6.2)); gs=GridSpec(2,6,wspace=2.0,hspace=.62)
axs=[fig.add_subplot(gs[0,0:2]),fig.add_subplot(gs[0,2:4]),fig.add_subplot(gs[0,4:6]),fig.add_subplot(gs[1,1:3]),fig.add_subplot(gs[1,3:5])]
_b=axs[4].get_position(); axs[4].set_position([_b.x0+.05,_b.y0,_b.width,_b.height])
agedots(axs[0],C,'book_looking_prop',[(0,1,'p = .003'),(0,2,'p = .036')],'Proportion of session','Infant looking at book')
agedots(axs[1],C,'sa',[(0,1,'p = .015')],'Proportion of session','Infant sustained attention')
agedots(axs[2],C,'parent_book_prop',[],'Proportion of session','Parent looking at book')
agedots(axs[3],C,'ja',[(0,1,'p = .023')],r'$\dfrac{\mathrm{Joint\ attention\ time}}{\mathrm{Session\ time}}$','Joint visual attention')
CU=pd.read_csv('chance_unified.csv')
for j,a in enumerate([6,9,12]):
    m=CU[CU.age==a].ja_circ.mean(); axs[3].plot([j-.3,j+.3],[m,m],color=INK,lw=1.2,ls=(0,(3,2)),zorder=5)
axs[2].text(.03,.97,'Age: F(2, 57) = 1.55, p = .22',transform=axs[2].transAxes,va='top',fontsize=6.6,color=INK2)
# E: share of infant book looking spent in joint visual attention, vs. chance
FS=pd.read_csv('final_age.csv')[['dyad','ja','ib','age']].merge(pd.read_csv('chance_unified.csv')[['dyad','ja_circ']],on='dyad')
FS['share']=FS.ja/FS.ib; FS['share_ch']=FS.ja_circ/FS.ib
agedots(axs[4],FS,'share',[],r'$\dfrac{\mathrm{Joint\ attention\ time}}{\mathrm{Infant\ book\ looking\ time}}$','Joint visual attention while infant looked')
for j,a in enumerate([6,9,12]):
    m=FS[FS.age==a].share_ch.mean(); axs[4].plot([j-.3,j+.3],[m,m],color=INK,lw=1.2,ls=(0,(3,2)),zorder=5)
axs[4].text(.03,.97,'Excess over chance increased with age:\nF(2, 57) = 4.28, p = .019; 12 > 6 months, p = .022',transform=axs[4].transAxes,va='top',fontsize=6.4,color=INK2)
for k in (0,1,2,3,4): axs[k].set_ylim(0,1.18); axs[k].set_yticks([0,.2,.4,.6,.8,1.0])
axs[3].text(2.55,1.17,'- - chance (circular shift)\nabove chance at every age, p < .001',ha='right',va='top',fontsize=6.4,color=INK2)
for a,l in zip(axs,'ABCDE'): letter(a,l)
fig.text(.99,.0,'Error bars: 95% CIs. Brackets: Tukey comparisons. Dashed lines: circular-shift chance.',ha='right',fontsize=7.5,color=INK2)
fig.savefig(OUT+'fig_attention_ja_by_age.png',bbox_inches='tight',facecolor='white'); plt.close(fig)
