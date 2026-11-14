import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd, pickle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy import stats as st
from jadef import yu_ja,_runs,FS
exec(open('../code/docfigs.py').read().split('# ---- Figure 1')[0])   # shared style + helpers
INF='#e8590c'; PAR='#5f3dc4'
plt.rcParams['errorbar.capsize']=3; plt.rcParams['lines.markeredgewidth']=1.0
S=pickle.load(open('series.pkl','rb')); B=pd.read_csv('lead_bouts.csv'); R=pd.read_csv('lead_dyad.csv')
D=34; s=S[D]; ib,pb=s['infant_book'],s['parent_book']; ja=yu_ja(ib,pb); b=B[B.dyad==D]
FA=pd.read_csv('final_age.csv'); CU=pd.read_csv('chance_unified.csv')

# ---------- Reciprocity figure (v8): who led, lead changes, Granger ----------
fig=plt.figure(figsize=(7.6,7.4)); gs=GridSpec(2,8,wspace=2.6,hspace=.55); WX=pd.read_csv('wxc_dyad.csv')

# E who led JA bouts
ax=fig.add_subplot(gs[0,0:4]); agedots(ax,R,'p_inf',[(0,1,'p = .022'),(0,2,'p = .030')],'Proportion infant-led','Who led joint attention')
for j,a_ in enumerate([6,9,12]):
    m=CU[CU.age==a_].p_inf_circ.mean(); ax.plot([j-.3,j+.3],[m,m],color=INK,lw=1.2,ls=(0,(3,2)),zorder=5)
for j,t in [(0,'n.s.'),(1,'p = .038'),(2,'p = .014')]: ax.text(j,.02,t,ha='center',va='bottom',fontsize=7.2,color=INK if t!='n.s.' else INK2)
ax.set_ylim(0,1.42); ax.set_yticks([0,.25,.5,.75,1]); letter(ax,'A')
ax.text(2.55,1.41,'- - chance (circular shift)',ha='right',va='top',fontsize=6.8,color=INK2)
# E lead changes per minute: dyads + mean, dashed chance (circular shift)
ax=fig.add_subplot(gs[0,4:8])
agedots(ax,WX,'ep_sw_min',[],'Lead changes per minute','Lead changes')
for j,a_ in enumerate([6,9,12]):
    m=WX[WX.age==a_].ep_sw_min_chance.mean(); ax.plot([j-.3,j+.3],[m,m],color=INK,lw=1.2,ls=(0,(3,2)),zorder=5)
top=WX.ep_sw_min.max()
for j,t in [(0,'n.s.'),(1,'p < .001'),(2,'p = .025')]: ax.text(j,top*1.04,t,ha='center',va='bottom',fontsize=7.2,color=INK if t!='n.s.' else INK2)
bracket(ax,0,1,top*1.24,'6 vs 9: p = .009',top*.03)
ax.set_ylim(0,top*1.45)
ax.text(2.55,top*.02,'- - chance (circular shift)',ha='right',va='bottom',fontsize=6.8,color=INK2); letter(ax,'B')


# C Granger causality
GR=pd.read_csv('granger_dyad.csv'); ax=fig.add_subplot(gs[1,1:7])
lab={'infant_to_parent':'Infant \u2192 parent','parent_to_infant':'Parent \u2192 infant'}
sig={('infant_to_parent',6):'***',('infant_to_parent',9):'**',('infant_to_parent',12):'*',('parent_to_infant',6):'n.s.',('parent_to_infant',9):'**',('parent_to_infant',12):'***'}
for dr,col,off in [('infant_to_parent',INF,-.16),('parent_to_infant',PAR,.16)]:
    ms=[]
    for j,a_ in enumerate([6,9,12]):
        y=GR[(GR.dir==dr)&(GR.age==a_)].gc_exc; ax.scatter(j+off+rng.uniform(-.05,.05,len(y)),y,s=6,color=col,alpha=.35,lw=0)
        ax.errorbar(j+off,y.mean(),yerr=ci(y),fmt='o',ms=6,color=col,elinewidth=1.6,zorder=4); ms.append(y.mean())
        s_=sig[(dr,a_)]; ax.text(j+off,y.mean()+ci(y)+.3,s_,ha='center',va='bottom',fontsize=8 if s_!='n.s.' else 6.6,color=col)
    ax.plot(np.arange(3)+off,ms,color=col,lw=1.4,zorder=3,label=lab[dr])
ax.axhline(0,color=INK,lw=1.2,ls=(0,(3,2)))
bracket(ax,.16,2.16,8.3,'P\u2192I, 12 vs 6: p = .002',.35)
ax.set_xticks([0,1,2]); ax.set_xticklabels(['6','9','12']); ax.set_xlabel('Age (months)'); ax.set_xlim(-.6,2.6); ax.set_ylim(-6.5,11.5); ax.yaxis.set_label_coords(-.09,.5)
ax.set_ylabel('Granger influence above chance\n(LL gain per 1,000 bins)'); ax.grid(axis='y',color=GRID,lw=.6); ax.set_axisbelow(True)
ax.set_title('Who predicts whom (Granger)',fontsize=8.8,loc='left',color=INK)
ax.legend(frameon=False,fontsize=7.4,loc='upper right',ncol=2,handlelength=1.6,bbox_to_anchor=(1.0,1.01))
ax.text(.02,.99,'Direction \u00d7 age: p = .007',transform=ax.transAxes,ha='left',va='top',fontsize=6.6,color=INK2)
letter(ax,'C')
fig.text(.5,.0,'Small dots: dyads. Error bars: 95% CIs. Dashed lines: circular-shift chance.\nWithin-age tests vs. chance are Holm-corrected (* p < .05, ** p < .01, *** p < .001).',ha='center',fontsize=6.6,color=INK2)
fig.savefig('../figures/Fig6_reciprocity.png',bbox_inches='tight',facecolor='white'); print('saved')
