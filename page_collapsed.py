import pandas as pd, numpy as np, pickle, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf, pingouin as pg
FS=30; S=pickle.load(open('series.pkl','rb')); l=pd.read_csv('../data/look_spans.csv')
rows=[]; P=[]
for d,s in S.items():
    n=s['n']; t0=s['t0']; ib=s['infant_book']; ja=__import__('jadef').yu_ja(ib,s['parent_book']); cond=np.full(n,'',dtype=object)
    for on,off,v in l[(l.dyad_id==d)&(l.stream=='condition')][['onset_ms','offset_ms','value']].values:
        cond[max(int((on-t0)*FS/1000),0):min(int(np.ceil((off-t0)*FS/1000)),n)]=v
    for v in ['individual','category','none']:
        m=cond==v
        if m.sum()>FS*10: P.append(dict(dyad=d,page=v,ib=ib[m].mean(),ja=ja[m].mean()))
    W=5*FS
    for b in range(0,n-W,W):
        vals,cnt=np.unique(cond[b:b+W],return_counts=True); v=vals[cnt.argmax()]
        if v in ('individual','category','none') and cnt.max()>.8*W: rows.append(dict(dyad=d,page=v,t=b/FS/60,ja=ja[b:b+W].mean(),ib=ib[b:b+W].mean()))
B=pd.DataFrame(rows); B['cat']=(B.page=='category').astype(float); B['ind']=(B.page=='individual').astype(float); B['lab']=B.cat+B.ind
P=pd.DataFrame(P); k=P.groupby('dyad').page.nunique(); P=P[P.dyad.isin(k[k==3].index)]; P.to_csv('page_dyad.csv',index=False)
print('dyads:',P.dyad.nunique())
for y in ['ja','ib']:
    print(f'\n##### {y}  means:',P.groupby('page')[y].mean().round(3).to_dict())
    a=pg.rm_anova(P,dv=y,within='page',subject='dyad',correction=True); print('RM ANOVA:',a[['ddof1','ddof2','F','p_unc','p_GG_corr','ng2']].round(4).to_dict('records'))
    print(pg.pairwise_tests(P,dv=y,within='page',subject='dyad',padjust='holm')[['A','B','T','p_unc','p_corr']].round(4).to_string(index=False))
    m=smf.mixedlm(f'{y}~cat+ind+t',B,groups='dyad',re_formula='~cat+ind').fit()
    print('Time-controlled: cat vs none b=%.3f p=%.4f | ind vs none b=%.3f p=%.4f | time b=%.3f/min p=%.4f'%(m.params['cat'],m.pvalues['cat'],m.params['ind'],m.pvalues['ind'],m.params['t'],m.pvalues['t']))
    m2=smf.mixedlm(f'{y}~ind+t',B[B.page!='none'],groups='dyad',re_formula='~ind').fit(); print('   individual vs category b=%.3f p=%.4f'%(m2.params['ind'],m2.pvalues['ind']))
    m3=smf.mixedlm(f'{y}~lab+t',B,groups='dyad',re_formula='~lab').fit(); print('   any label vs none b=%.3f p=%.4f'%(m3.params['lab'],m3.pvalues['lab']))
