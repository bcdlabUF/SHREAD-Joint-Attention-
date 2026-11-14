"""Figure 3: 90-s excerpt from one 9-month dyad (Method)."""
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
W=90
# choose 90-s excerpt where the lead visibly alternates (windowed peak lag changes sign, both leaders in A)
def wxc(a,z,win=20*FS,step=FS//2,ml=3*FS):
    lags=np.arange(-ml,ml+1,3); cen=[];M=[]
    for c0 in range(a-win//2,z-win//2+1,step):
        row=[]
        for L in lags:
            x=ib[c0:c0+win]; y=pb[c0+L:c0+L+win]
            row.append(np.corrcoef(x,y)[0,1] if c0+min(L,0)>=0 and len(y)==win and x.std()>0 and y.std()>0 else np.nan)
        M.append(row); cen.append((c0+win/2)/FS)
    return lags,np.array(M),np.array(cen)
best=None
for t0 in np.arange(10,len(ib)/FS-W-10,5):
    lags_,M_,_=wxc(int(t0*FS),int((t0+W)*FS))
    pk=np.array([lags_[np.nanargmax(r)] if np.isfinite(r).any() and np.nanmax(r)>=.3 else 0 for r in M_]); sg=np.sign(pk[pk!=0])
    g=b[(b.t>=t0)&(b.t<t0+W)&(b.lead!='simul')]
    sc=(np.diff(sg)!=0).sum()*3+min((g.lead=='infant').sum(),(g.lead=='parent').sum())-10*np.isnan(M_).mean()
    if best is None or sc>best[0]: best=(sc,t0)
t0=best[1]; a,z=int(t0*FS),int((t0+W)*FS); tt=np.arange(a,z)/FS-t0
# windowed cross-correlation (Boker et al., 2002): 20-s windows, 0.5-s steps, lags +/-3 s
win,step,ml=20*FS,FS//2,3*FS; lags=np.arange(-ml,ml+1,3)
cen=[];M=[]
for c0 in range(a-win//2,z-win//2+1,step):
    row=[]
    for L in lags:   # L>0: infant at t vs parent at t+L  -> infant leads
        x=ib[c0:c0+win]; y=pb[c0+L:c0+L+win]
        row.append(np.corrcoef(x,y)[0,1] if x.std()>0 and y.std()>0 and len(y)==win else np.nan)
    M.append(row); cen.append((c0+win/2)/FS-t0)
M=np.array(M)
FA=pd.read_csv('final_age.csv'); CU=pd.read_csv('chance_unified.csv')
# ---------- Methods figure: 90-s excerpt ----------
fig=plt.figure(figsize=(9.4,3.9)); gs=GridSpec(1,2,width_ratios=[4.2,1],wspace=.22)
# A timeline
pk=np.array([lags[np.nanargmax(r)]/FS if np.isfinite(r).any() and np.nanmax(r)>=.3 else np.nan for r in M]); cen=np.array(cen)
ax=fig.add_subplot(gs[0,0])
JAC='#2a78d6'
for y,x,col,lab in [(7.5,ib,INF,'Infant'),(6,pb,PAR,'Parent'),(4.5,ja,JAC,'Joint attention')]:
    on,off=_runs(x[a:z])
    for o,f in zip(on,off): ax.add_patch(plt.Rectangle((o/FS,y-.35),(f-o)/FS,.7,color=col,alpha=.85,lw=0))
on,off=_runs(ja[a:z])
for o,f in zip(on,off): ax.axvspan(o/FS,f/FS,ymin=.02,ymax=.98,color='#d6d5cf',zorder=0)
for _,r in b[(b.t>=t0)&(b.t<t0+W)].iterrows():
    if r.lead=='simul': continue
    col=INF if r.lead=='infant' else PAR; y0=0
    ax.plot(r.t-t0,y0,marker='^',color=col,ms=8,clip_on=False)
LCOL={'infant':INF,'parent':PAR,'sync':'#8a8984','none':'#efeeea'}
lab=np.where(np.isnan(pk),'none',np.where(pk>.1,'infant',np.where(pk<-.1,'parent','sync')))
zc=np.array([np.nanmean(np.arctanh(np.clip(r,-.999,.999))) if np.isfinite(r).any() else np.nan for r in M]); cen_=np.array(cen)
from matplotlib.colors import LinearSegmentedColormap as _LS
ccm=_LS.from_list('cc',['#ffffff','#0f8b8d']); VMAX=.6
for cc_,zz in zip(cen_,zc):
    if 0<=cc_<=W:
        if np.isnan(zz): ax.add_patch(plt.Rectangle((cc_-.25,2.65),.5,.7,facecolor='white',edgecolor='#c8c7c2',hatch='////',lw=0))
        else: ax.add_patch(plt.Rectangle((cc_-.25,2.65),.5,.7,color=ccm(np.clip(zz,0,VMAX)/VMAX),lw=0))
ax.add_patch(plt.Rectangle((0,2.65),W,.7,fill=False,edgecolor=INK2,lw=.5))
# leader strip: who leads in each coupled window (peak r >= .30); lead change = new leader held >= 2 s
LEADC={'infant':INF,'parent':PAR}
for cc_,lb in zip(cen_,lab):
    if 0<=cc_<=W and lb in LEADC: ax.add_patch(plt.Rectangle((cc_-.25,1.15),.5,.7,color=LEADC[lb],alpha=.75,lw=0))
ax.add_patch(plt.Rectangle((0,1.15),W,.7,fill=False,edgecolor=INK2,lw=.5))
seq=[(c_,l_) for c_,l_ in zip(cen_,lab) if l_ in LEADC]; cur=None; cand=None; run_=0; changes=[]
for c_,l_ in seq:
    if cur is None: cur=l_; continue
    if l_!=cur:
        if cand==l_: run_+=1
        else: cand=l_; run_=1; cstart=c_
        if run_*0.5>=2: changes.append(cstart); cur=l_; cand=None; run_=0
    else: cand=None; run_=0
for c_ in changes:
    if 0<=c_<=W: ax.plot([c_,c_],[1.0,2.0],color=INK,lw=1.4,zorder=5)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _ia
import matplotlib as _mpl
cax=_ia(ax,width='10%',height='4%',loc='lower left',bbox_to_anchor=(.0,-.25,1,1),bbox_transform=ax.transAxes,borderpad=0)
cb=fig.colorbar(_mpl.cm.ScalarMappable(norm=_mpl.colors.Normalize(0,VMAX),cmap=ccm),cax=cax,orientation='horizontal'); cb.set_ticks([0,.3,.6]); cb.ax.tick_params(labelsize=6.5,length=2); cb.set_label('')
ax.text(.115,-.235,'Coordination (Fisher z)    \u2572\u2572 not computable',transform=ax.transAxes,fontsize=7.5,color=INK2,va='bottom')
ax.set_xlim(0,W); ax.set_ylim(-.6,8.1); ax.set_yticks([0,1.5,3,4.5,6,7.5]); ax.set_yticklabels(['Who led\nJA bout','Leader\n(| = lead change)','Coordination\n(20-s window)','Joint\nattention','Parent','Infant'])
ax.set_xlabel('Time in excerpt (s)',labelpad=1); ax.spines['left'].set_visible(False); ax.tick_params(axis='y',length=0)
ax.set_title('One 9-month dyad (90-s excerpt): coordination and reciprocity',fontsize=8.8,loc='left',color=INK)
y_=-.235; X0=.62
for txt,col,dx in [('Leads:',INK2,.07),('\u25b2 \u25a0 infant',INF,.12),('\u25b2 \u25a0 parent',PAR,.12)]:
    ax.text(X0,y_,txt,transform=ax.transAxes,fontsize=7.5,color=col); X0+=dx

WX=pd.read_csv('wxc_dyad.csv')
# Granger inset: whole-session values for this dyad
GR=pd.read_csv('granger_dyad.csv'); g=GR[GR.dyad.astype(int)==int(D)]
axg=fig.add_subplot(gs[0,1])
for j,(dr,lab_,col) in enumerate([('infant_to_parent','Infant\n\u2192 parent',INF),('parent_to_infant','Parent\n\u2192 infant',PAR)]):
    r=g[g.dir==dr].iloc[0]
    axg.bar(j,r.gc,width=.55,color=col,alpha=.85); axg.plot([j-.33,j+.33],[r.gc_chance,r.gc_chance],color=INK,lw=1.3,ls=(0,(3,2)))
axg.set_xticks([0,1]); axg.set_xticklabels(['Infant\n\u2192 parent','Parent\n\u2192 infant'],fontsize=7.5); axg.set_xlim(-.6,1.6)
axg.set_ylabel('Granger influence\n(LL gain per 1,000 bins)',fontsize=7.5); axg.tick_params(labelsize=7)
axg.set_title('Who predicted whom\n(whole session)',fontsize=8.8,loc='left',color=INK)
axg.text(.5,-.2,'dashed: circular-shift chance',transform=axg.transAxes,ha='center',fontsize=7,color=INK2)
print('granger dyad',D,g[['dir','gc','gc_chance']].round(2).values.tolist(), 'lead changes in excerpt',len(changes))
fig.savefig(OUT+'fig_methods_excerpt.png',bbox_inches='tight',facecolor='white')
