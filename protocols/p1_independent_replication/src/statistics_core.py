"""Offline numerical core extracted from the frozen initial analysis.
No collection imports or network operations. Loop order, task order and seeds
are preserved to reproduce the archived finite Monte Carlo estimates.
"""
import collections, hashlib, math
import numpy as np
from scipy import stats
B=10000
SEED=2026102207
TASKS=('gsm8k','commonsenseqa')
def majority(a):
 c=collections.Counter(x for x in a if x is not None)
 return max(c,key=c.get) if c else None

def permuted_accuracy(a,s,gold,beta,seed):
 ix=[i for i,v in enumerate(a) if v is not None];m=majority(a)
 if not ix or len(set(a[i] for i in ix))==1 or any(s[i] is None for i in ix):return float(m==gold)
 scores=np.array([s[i]/100 for i in ix]);rng=np.random.default_rng(seed)
 # The same random permutations are used at beta=4 and 64 (paired control).
 z=np.array([rng.permutation(scores) for _ in range(200)])
 w=np.exp(beta*(z-z.max(axis=1,keepdims=True)))
 classes=list(dict.fromkeys(a[i] for i in ix))
 totals=np.column_stack([w[:,[j for j,i in enumerate(ix) if a[i]==c]].sum(axis=1) for c in classes])
 winners=np.argmax(totals,axis=1)
 return float(np.mean([classes[i]==gold for i in winners]))

def summarize(rows):
 rng=np.random.default_rng(SEED);n=len(rows);idx=rng.integers(0,n,size=(B,n));base=np.array([r['correct']['M'] for r in rows]);out={}
 for p in rows[0]['correct']:
  x=np.array([r['correct'][p] for r in rows]);d=x-base;boots=d[idx].mean(axis=1)
  v={'accuracy_percent':float(100*x.mean()),'difference_from_M_pp':float(100*d.mean()),'ci95_pp':(100*np.quantile(boots,[.025,.975])).tolist()}
  if p in rows[0]['predictions']:
   corr=int(sum(d==1));harm=int(sum(d==-1));v.update(corrections=corr,harms=harm,nonzero=corr+harm,exact_mcnemar_p=float(stats.binomtest(corr,corr+harm,.5).pvalue) if corr+harm else 1.,changed_answers=sum(r['predictions'][p]!=r['majority'] for r in rows),outside_top=sum(r['predictions'][p] is not None and r['predictions'][p] not in r['top'] for r in rows))
   assert int(d.sum())==corr-harm
  out[p]=v
 return {'n':n,'complete_confidence_pools':sum(r['complete'] for r in rows),'constant_score_pools':sum(r['score_constant'] for r in rows),'candidate_n':sum(r['candidate_n'] for r in rows),'candidate_correct':sum(r['candidate_correct'] for r in rows),'mixed_correctness_wqd_n':sum(r['wqd'] is not None for r in rows),'mean_wqd':float(np.mean([r['wqd'] for r in rows if r['wqd'] is not None])) if any(r['wqd'] is not None for r in rows) else None,'policies':out,'margin_strata':{name:{'n':len(rr),'W64_corrections':sum(r['correct']['M']==0 and r['correct']['W64']==1 for r in rr),'W64_harms':sum(r['correct']['M']==1 and r['correct']['W64']==0 for r in rr),'G64_net_vs_W64':sum(r['correct']['G64']-r['correct']['W64'] for r in rr)} for name,rr in [('gap0',[r for r in rows if r['gap']==0]),('gap1',[r for r in rows if r['gap']==1]),('gap2plus',[r for r in rows if r['gap']>=2])]}}

def pooled(rows,models):
 grouped=collections.defaultdict(list)
 for r in rows:
  if r['model'] in models:grouped[r['id']].append(r)
 ids=sorted(grouped);assert all(len(v)==len(models) for v in grouped.values())
 tasks=np.array([grouped[i][0]['task'] for i in ids]);arrays={};strata=[np.where(tasks==t)[0] for t in TASKS];rng=np.random.default_rng(SEED)
 bootidx=[rng.choice(ix,(B,len(ix)),replace=True) for ix in strata]
 def contrast_metric(d):
  d=np.asarray(d,float);means=[d[ix].mean() for ix in strata];components=[d[ix].var(ddof=1)/len(ix)/4 for ix in strata];var=sum(components);mean=float(np.mean(means))
  if var>0:
   df=var*var/sum(v*v/(len(ix)-1) for v,ix in zip(components,strata));p=float(2*stats.t.sf(abs(mean/math.sqrt(var)),df))
  else:df=None;p=1. if mean==0 else 0.
  boots=.5*(d[bootidx[0]].mean(axis=1)+d[bootidx[1]].mean(axis=1))
  return {'effect_pp':100*mean,'ci95_pp':(100*np.quantile(boots,[.025,.975])).tolist(),'p_two_sided':p,'df':df,'nonzero_distinct_questions':int(sum(d!=0))}
 h1=[];h2=[];c64=[];cg=[]
 for i in ids:
  rr=grouped[i]
  h1.append(np.mean([int(r['correct']['M']==1 and r['correct']['W64']==0)-int(r['correct']['M']==1 and r['correct']['W4']==0) for r in rr]))
  h2.append(np.mean([r['correct']['G64']-r['correct']['W64'] for r in rr]))
  c64.append(np.mean([int(r['correct']['M']==0 and r['correct']['W64']==1) for r in rr]));cg.append(np.mean([int(r['correct']['M']==0 and r['correct']['G64']==1) for r in rr]))
 primary={'H1_harm64_minus_harm4':contrast_metric(h1),'H2_accuracy_guard64_minus_64':contrast_metric(h2)}
 order=sorted(primary,key=lambda k:primary[k]['p_two_sided']);prev=0
 for j,k in enumerate(order):prev=max(prev,min(1,(2-j)*primary[k]['p_two_sided']));primary[k]['holm_p']=prev
 c64,cg=np.array(c64),np.array(cg);den=c64.sum();ret=float(cg.sum()/den) if den else None
 bden=.5*(c64[bootidx[0]].mean(axis=1)+c64[bootidx[1]].mean(axis=1));bnum=.5*(cg[bootidx[0]].mean(axis=1)+cg[bootidx[1]].mean(axis=1));ok=bden>0
 retci=np.quantile(bnum[ok]/bden[ok],[.025,.975]).tolist() if ok.any() else None
 for p in rows[0]['correct']:
  d=[np.mean([r['correct'][p]-r['correct']['M'] for r in grouped[i]]) for i in ids];arrays[p]=contrast_metric(d)
 alignment={b:contrast_metric([np.mean([r['correct']['W'+str(b)]-r['correct']['PERM'+str(b)] for r in grouped[i]]) for i in ids]) for b in (4,64)}
 return {'models':list(models),'independent_questions':len(ids),'primary':primary,'correction_retention':ret,'retention_ci95':retci,'retention_bootstrap_undefined':int(sum(~ok)),'policies_vs_M':arrays,'alignment_premium_pp':alignment}
