"""Descriptive score-quality and development-stability analyses; no network.
All inputs are archived records. No fitted parameter changes evaluation policies.
"""
from pathlib import Path
import collections,csv,json,math,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads((ROOT/p).read_text())
def write(name,data):
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def csvout(name,rows):
 p=ROOT/'tables'/name;p.parent.mkdir(exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def bootmean(a,rng,B=10000):
 a=np.asarray(a,dtype=float)
 if len(a)<2:return [None,None]
 v=a[rng.integers(0,len(a),(B,len(a)))].mean(1)
 return np.quantile(v,[.025,.975]).tolist()
def score_cell(rows,key):
 rng=np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:12],16))
 brier=[];wqd=[];bins=[[] for _ in range(10)];scored=0;missing=0;constant=0;complete=0;ties=0;pairs=0
 for r in rows:
  ix=[i for i,a in enumerate(r['answers']) if a is not None]
  valid=[i for i in ix if r['scores'][i] is not None]
  missing+=len(ix)-len(valid);scored+=len(valid)
  if valid:
   z=np.array([r['scores'][i]/100 for i in valid]);y=np.array([r['answers'][i]==r['gold'] for i in valid],dtype=float)
   brier.append(float(((z-y)**2).mean()))
   for a,b in zip(z,y):bins[min(9,int(a*10))].append((float(a),float(b),1/len(valid)))
  if ix and len(valid)==len(ix):
   complete+=1;constant+=int(np.ptp(z)==0)
   p=z[y==1];n=z[y==0]
   if len(p) and len(n):
    t=p[:,None]-n[None,:];wqd.append(float(((t>0)+.5*(t==0)).mean()));ties+=int((t==0).sum());pairs+=int(t.size)
 rel=[]
 for j,items in enumerate(bins):
  if items:
   a=np.array(items);weight=a[:,2];rel.append({'lower':j/10,'upper':(j+1)/10,'candidates':len(items),'question_mass':float(weight.sum()),'confidence':float(np.average(a[:,0],weights=weight)),'accuracy':float(np.average(a[:,1],weights=weight))})
  else:rel.append({'lower':j/10,'upper':(j+1)/10,'candidates':0,'question_mass':0,'confidence':None,'accuracy':None})
 return {'scheduled_questions':len(rows),'scored_candidates':scored,'missing_candidate_scores':missing,'brier_questions':len(brier),'brier':float(np.mean(brier)) if brier else None,'brier_ci95':bootmean(brier,rng),'wqd_questions':len(wqd),'wqd':float(np.mean(wqd)) if wqd else None,'wqd_ci95':bootmean(wqd,rng),'complete_pools':complete,'constant_score_pools':constant,'cross_correctness_score_ties':ties,'cross_correctness_pairs':pairs,'reliability':rel}
def predict(aa,ss,tau):
 c=collections.Counter(a for a in aa if a is not None);m=max(c,key=c.get) if c else None
 if not c:return None
 ix=[i for i,a in enumerate(aa) if a is not None]
 if any(ss[i] is None for i in ix):return m
 mx=max(ss[i] for i in ix); totals={a:sum(math.exp((ss[i]-mx)/tau) for i in ix if aa[i]==a) for a in c}
 return max(totals,key=totals.get)
def main():
 strata={'initial_evaluation':read('data/evaluation_rows.json'),'added_evaluation':read('data/replication_rows.json'),'added_compatibility':read('data/replication_extended_parser_rows.json')}; cells={};flat=[]
 for parser,rows in strata.items():
  for model,task in sorted({(r['model'],r['task']) for r in rows}):
   key=f'{parser}:{model}:{task}';v=score_cell([r for r in rows if r['model']==model and r['task']==task],key);cells[key]=v
   flat.append({'stratum':parser,'model':model,'task':task,**{k:v[k] for k in ['scheduled_questions','scored_candidates','missing_candidate_scores','brier_questions','brier','wqd_questions','wqd','complete_pools','constant_score_pools']},'brier_ci_low':v['brier_ci95'][0],'brier_ci_high':v['brier_ci95'][1],'wqd_ci_low':v['wqd_ci95'][0],'wqd_ci_high':v['wqd_ci95'][1]})
 write('data/confidence_quality.json',{'scope':'descriptive; equal question weights; complete-pool WQD conditional on mixed correctness; score-available candidate Brier averaged within question; ten fixed bins; question bootstrap 10000','cells':cells});csvout('confidence_quality.csv',flat)
 # Full counts and pool coverage are descriptive, never additional independent samples.
 ledger=[]
 for stratum,rows in strata.items():
  for model,task in sorted({(r['model'],r['task']) for r in rows}):
   rr=[r for r in rows if r['model']==model and r['task']==task]
   for p in ['W4','W64','G64']:
    C=sum(r['correct']['M']==0 and r['correct'][p]==1 for r in rr);H=sum(r['correct']['M']==1 and r['correct'][p]==0 for r in rr)
    D=sum(r['predictions'][p]!=r['majority'] for r in rr);J=sum(r['predictions'][p]!=r['majority'] and r['correct'][p]==0 and r['correct']['M']==0 for r in rr)
    assert D==C+H+J
    counts=[len({a for a in r['answers'] if a is not None}) for r in rr]
    ledger.append({'stratum':stratum,'model':model,'task':task,'policy':p,'n':len(rr),'C':C,'H':H,'J':J,'answer_changes':D,'empty_pools':sum(v==0 for v in counts),'single_class_pools':sum(v==1 for v in counts),'multiclass_pools':sum(v>1 for v in counts)})
 csvout('policy_counts_and_pool_coverage.csv',ledger)
 failures=collections.Counter((r['model'],r['task'],r['category']) for r in read('data/parser_failure_ledger.json'))
 csvout('parser_categories_by_model.csv',[{'model':m,'task':t,'category':c,'seats':n} for (m,t,c),n in sorted(failures.items())])
 lock=read('protocols/initial/development_lock.json');csvout('initial_development_grid.csv',[{'cell':k,'beta':b,'correct':v['correct_by_beta'][str(b)],'n':v['n'],'selected_beta':v['beta']} for k,v in sorted(lock['cells'].items()) for b in [0,1,2,4,8,16,32,64]])
 dev=read('data/cisc/development_rows.json');ev=read('data/cisc/evaluation_rows.json');locks=read('protocols/cisc/temperature_lock.json')['methods'];grid=np.logspace(-4,4,80);result={};curves=[];stability=[]
 rng=np.random.default_rng(2026102401);counts=np.array([np.bincount(i,minlength=len(dev)) for i in rng.integers(0,len(dev),(10000,len(dev)))],dtype=np.int16)
 for key,branch in [('ptrue','CISC_P'),('verbal','CISC_V')]:
  predictions=np.array([[predict(r['answers'],r[key],tau) for tau in grid] for r in dev]);truth=np.array([r['gold'] for r in dev]);correct=predictions==truth[:,None];hits=correct.sum(0)
  assert int(hits.max())==locks[key]['correct'];chosen=79-int(hits[::-1].argmax());assert abs(grid[chosen]-locks[key]['temperature'])<1e-12
  h=counts@correct.astype(np.int16);selected=79-np.argmax(h[:,::-1],axis=1);hist=np.bincount(selected,minlength=80)
  unique=np.unique(selected);ep={int(j):[predict(r['answers'],r[key],grid[j]) for r in ev] for j in unique}
  fixed=[r['prediction'][branch] for r in ev];dis={int(j):sum(a!=b for a,b in zip(ep[int(j)],fixed)) for j in unique}
  distances=np.array([dis[int(j)] for j in selected]);topweights=[];span=[];harddiff=0
  for r in ev:
   if not r['confidence_complete'][key] or len(set(r['answers']))<=1:continue
   z=np.array(r[key]);w=np.exp((z-z.max())/grid[chosen]);w/=w.sum();topweights.append(float(w.max()));span.append(float(np.ptp(z)/grid[chosen]));harddiff+=r['prediction'][branch]!=r['answers'][int(z.argmax())]
  result[key]={'temperature':float(grid[chosen]),'grid_position':chosen+1,'best_development_correct':int(hits.max()),'tied_best_grid_points':int((hits==hits.max()).sum()),'bootstrap_repeats':10000,'selected_locked_temperature_fraction':float((selected==chosen).mean()),'bootstrap_temperature_q025_q50_q975':np.quantile(grid[selected],[.025,.5,.975]).tolist(),'evaluation_decision_disagreement_median_q95':np.quantile(distances,[.5,.95]).tolist(),'evaluation_decision_disagreement_max':int(distances.max()),'evaluation_labels_used_for_stability':False,'multiclass_complete_pools':len(topweights),'multiclass_max_candidate_weight_median':float(np.median(topweights)),'multiclass_max_candidate_weight_above_0_9':int(sum(v>.9 for v in topweights)),'multiclass_score_span_over_temperature_median':float(np.median(span)),'multiclass_prediction_different_from_hard_argmax':int(harddiff)}
  curves += [{'branch':key,'grid_index':j+1,'temperature':float(t),'correct':int(hits[j]),'n':len(dev),'bootstrap_selected_count':int(hist[j])} for j,t in enumerate(grid)]
  stability += [{'branch':key,'grid_index':int(j)+1,'temperature':float(grid[j]),'bootstrap_selected_count':int(hist[j]),'evaluation_answer_disagreements':dis[int(j)]} for j in unique]
 write('data/cisc/development_stability.json',{'seed':2026102401,'scope':'diagnostic bootstrap of archived development questions; original temperature remains locked; evaluation labels not used to select or judge temperature stability','branches':result});csvout('cisc_temperature_grid.csv',curves);csvout('cisc_temperature_stability.csv',stability)
 print(json.dumps({'score_cells':len(cells),'cisc_stability':result},indent=2))
if __name__=='__main__':main()
