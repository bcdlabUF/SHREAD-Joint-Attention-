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
F=pd.read_csv('final_age.csv')[['dyad','ja','sa']].rename(columns={'dyad':'dyad_id'}); C=C.merge(F,on='dyad_id')
fig,axs=plt.subplots(1,4,figsize=(9.4,2.9),gridspec_kw=dict(wspace=.5))
agedots(axs[0],C,'book_looking_prop',[(0,1,'p = .003'),(0,2,'p = .036')],'Proportion of session','Infant looking at book')
agedots(axs[1],C,'ja',[(0,1,'p = .023')],None,'Joint attention')
agedots(axs[2],C,'sa',[(0,1,'p = .015')],None,'Infant sustained attention')
agedots(axs[3],C[C.dyad_id!=6],'infant_partner_prop',[(0,1,'p = .032')],None,'Infant looking at parent')
for a,l in zip(axs,'ABCD'): letter(a,l)
fig.text(.99,-.1,'Error bars: 95% CIs',ha='right',fontsize=7.5,color=INK2)
fig.savefig(OUT+'fig1_attention_by_age.png',bbox_inches='tight',facecolor='white'); plt.close(fig)
# ---- Figure 2
fig,axs=plt.subplots(1,2,figsize=(6.2,2.9),gridspec_kw=dict(wspace=.4))
for ax,p,ttl,ylab in [(axs[0],'book','Book looking','Coordination (mean Fisher z)'),(axs[1],'face','Face looking',None)]:
    X=pd.read_csv(f'xc_dyad_{p}.csv')
    agedots(ax,X,'z_all',[(0,2,'p = .027' if p=='book' else 'p = .023')],ylab,ttl)
    for j,a in enumerate([6,9,12]):
        m=X[X.age==a].pseudo.mean(); ax.plot([j-.3,j+.3],[m,m],color=INK,lw=1.2,ls=(0,(3,2)))
    ax.axhline(0,color=INK2,lw=.7)
for a,l in zip(axs,'AB'): letter(a,l)
fig.text(.99,-.1,'Error bars: 95% CIs',ha='right',fontsize=7.5,color=INK2)
fig.savefig(OUT+'fig2_coordination_by_age.png',bbox_inches='tight',facecolor='white'); plt.close(fig)
# ---- Figure 7
P=pd.read_csv('page_dyad.csv'); pages=[('individual','Individual\nlabels'),('category','Category\nlabels'),('none','No\nlabels')]
fig,axs=plt.subplots(1,2,figsize=(6.4,3.0),gridspec_kw=dict(wspace=.35))
for ax,(v,ttl,brs,note) in zip(axs,[('ja','Joint attention',[(1,2,'p = .013',.88),(0,2,'p = .075',.96)],None),('ib','Infant looking at book',[],'n.s. after controlling for time')]):
    W=P.pivot(index='dyad',columns='page',values=v)[[p for p,_ in pages]]
    for pi,(pg_,pl) in enumerate(pages):
        x=W[pg_]; ax.scatter(pi+rng.uniform(-.12,.12,len(x)),x,s=7,color=GRAY,alpha=.45,lw=0)
        ax.errorbar(pi,x.mean(),yerr=ci(x),fmt='o',ms=8,color='#1f9d55' if pg_!='none' else INK2,elinewidth=2,zorder=4)
    ax.plot(range(3),W.mean().values,color=INK,lw=1.2,zorder=3)
    for a1,a2,t,yb in brs: bracket(ax,a1,a2,yb,t,.015)
    if note: ax.text(1,1.0,note,ha='center',va='top',fontsize=7,color=INK2)
    ax.set_xticks(range(3)); ax.set_xticklabels([pl for _,pl in pages]); ax.set_xlim(-.4,2.4); ax.set_ylim(0,1.05)
    ax.set_title(ttl,fontsize=8.8,color=INK,loc='left'); ax.grid(axis='y',color=GRID,lw=.6); ax.set_axisbelow(True)
axs[0].set_ylabel('Proportion of page time')
for a,l in zip(axs,'AB'): letter(a,l)
fig.text(.99,-.1,'Error bars: 95% CIs',ha='right',fontsize=7.5,color=INK2)
fig.savefig(OUT+'fig7_page_type.png',bbox_inches='tight',facecolor='white'); plt.close(fig)
print('ok')
