"""Post-completion replication analysis; never reads partial outcomes."""
from pathlib import Path
import collections,csv,hashlib,json,math
import numpy as np
from scipy import stats
import study,statistics_core as core
from collector_base import now,write_once
from run import verify,MODELS
ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads((ROOT/p).read_text())
def contrast(d,tasks):
 d=np.array(d,float);tasks=np.array(tasks);rng=np.random.default_rng(2026102402);ix=[np.flatnonzero(tasks==t) for t in study.TASKS]
 mu=float(np.mean([d[i].mean() for i in ix]));v=[float(d[i].var(ddof=1)/len(i)/4) for i in ix];V=sum(v)
 boots=sum(d[rng.choice(i,(10000,len(i)))].mean(1)/2 for i in ix)
 return {'effect_pp':100*mu,'ci95_pp':(100*np.quantile(boots,[.025,.975])).tolist(),'nonzero_questions':int(sum(d!=0)),'scope':'descriptive same-question between-batch contrast'}
def main():
 verify();assert (ROOT/'results/collection_complete.json').exists(),'No partial-outcome analysis'
 refs=read('protocols/references.json');qs=read('protocols/questions.json');rows=[]
 for q in qs:
  for model in MODELS:
   seats=[read(f"results/parsed/{q['id']}-{model}-{s}.json") for s in range(7)];aa=[s['answer'] for s in seats];ss=[s['confidence'] for s in seats];g=refs[q['id']]
   m=study.majority(aa);gap,c=study.shape(aa);ix=[i for i,a in enumerate(aa) if a is not None];complete=bool(ix) and all(ss[i] is not None for i in ix)
   pred={'M':m,**{'W'+str(b):study.select(aa,ss,b) for b in study.BETAS},'G64':study.select(aa,ss,64,guard=True),'DEV':m,'RANK4':study.select(aa,ss,4,rank=True),'RANK64':study.select(aa,ss,64,rank=True)}
   good=[i for i in ix if aa[i]==g and ss[i] is not None];bad=[i for i in ix if aa[i]!=g and ss[i] is not None]
   wqd=float(np.mean([int(ss[i]>ss[j])+.5*int(ss[i]==ss[j]) for i in good for j in bad])) if good and bad else None
   correct={p:float(a==g) for p,a in pred.items()};seed=int(hashlib.sha256((q['id']+model+':perm-v7').encode()).hexdigest()[:8],16)
   correct.update({'PERM'+str(b):core.permuted_accuracy(aa,ss,g,b,seed) for b in (4,64)})
   ess={}
   if complete:
    z=np.array([ss[i]/100 for i in ix])
    for b in study.BETAS:w=np.exp(b*(z-z.max()));w/=w.sum();ess[str(b)]=float(1/(w*w).sum())
   rows.append({'id':q['id'],'task':q['task'],'model':model,'batch':'independent_replication','answers':aa,'strict_answers':[s['strict_answer'] for s in seats],'scores':ss,'gold':g,'majority':m,'gap':gap,'top':[a for a in c if c[a]==max(c.values())] if c else [],'predictions':pred,'correct':correct,'complete':complete,'score_constant':complete and len({ss[i] for i in ix})==1,'wqd':wqd,'ess':ess,'candidate_n':len(ix),'candidate_correct':sum(aa[i]==g for i in ix)})
 assert len(rows)==1800;write_once(ROOT/'results/evaluation_rows.json',rows)
 pooled=core.pooled(rows,MODELS);cells={f'{t}:{m}':core.summarize([r for r in rows if r['task']==t and r['model']==m]) for t in study.TASKS for m in MODELS}
 counts=[]
 for t in study.TASKS:
  for m in MODELS:
   rr=[r for r in rows if r['task']==t and r['model']==m]
   for pol in ['M']+['W'+str(b) for b in study.BETAS]+['G64','DEV','RANK4','RANK64']:
    C=sum(r['correct']['M']==0 and r['correct'][pol]==1 for r in rr);H=sum(r['correct']['M']==1 and r['correct'][pol]==0 for r in rr);J=sum(r['predictions'][pol]!=r['majority'] and r['correct'][pol]==0 and r['correct']['M']==0 for r in rr);D=sum(r['predictions'][pol]!=r['majority'] for r in rr);assert D==C+H+J
    counts.append({'task':t,'model':m,'policy':pol,'n':len(rr),'correct':int(sum(r['correct'][pol] for r in rr)),'C':C,'H':H,'J':J,'changed_answers':D})
 with (ROOT/'results/policy_counts.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(counts[0]));w.writeheader();w.writerows(counts)
 old=[r for r in read('protocols/original_evaluation_rows.json') if r['model'] in MODELS];oldmap={(r['id'],r['model']):r for r in old};newmap={(r['id'],r['model']):r for r in rows}
 funs={'H1':lambda r:int(r['correct']['M']==1 and r['correct']['W64']==0)-int(r['correct']['M']==1 and r['correct']['W4']==0),'H2':lambda r:r['correct']['G64']-r['correct']['W64']};between={};ordered=sorted(qs,key=lambda q:q['id'])
 for k,fn in funs.items():
  d=[np.mean([fn(newmap[q['id'],m])-fn(oldmap[q['id'],m]) for m in MODELS]) for q in ordered];between[k]=contrast(d,[q['task'] for q in ordered])
 combined=[{**r,'model':r['model']+'_'+batch} for batch,rr in [('original',old),('replication',rows)] for r in rr];combined_summary=core.pooled(combined,[m+'_'+b for b in ['original','replication'] for m in MODELS])
 positive={(t,m) for t in study.TASKS for m in MODELS if sum(r['correct']['G64']-r['correct']['W64'] for r in rows if r['task']==t and r['model']==m)>0}
 h1=pooled['primary']['H1_harm64_minus_harm4'];h2=pooled['primary']['H2_accuracy_guard64_minus_64'];ret=pooled['correction_retention']
 criteria={'H1_independently_supported':h1['effect_pp']>0 and h1['holm_p']<.05,'H2_positive_and_adjusted_significant':h2['effect_pp']>0 and h2['holm_p']<.05,'retention_at_least_75_percent':ret is not None and ret>=.75,'positive_H2_cells':len(positive),'positive_H2_covers_each_model':{m for t,m in positive}==set(MODELS),'positive_H2_covers_both_tasks':{t for t,m in positive}==set(study.TASKS),'H2_at_least_20_nonzero_questions':h2['nonzero_distinct_questions']>=20}
 criteria['joint_gate_claim_supported']=all(criteria[k] for k in ['H1_independently_supported','H2_positive_and_adjusted_significant','retention_at_least_75_percent','positive_H2_covers_each_model','positive_H2_covers_both_tasks','H2_at_least_20_nonzero_questions']) and len(positive)>=4
 resources={};response_models=collections.Counter();hashes_checked=0
 for p in sorted((ROOT/'raw/study/requests').glob('*/result.json')):
  r=json.loads(p.read_text());k=r['provider']+':'+r['kind'];v=resources.setdefault(k,{'logical_requests':0,'transport_attempts':0,'latency_seconds':0,'usage':{},'http_status_counts':{}});v['logical_requests']+=1;v['transport_attempts']+=r['attempts'];v['latency_seconds']+=r['latency_seconds'];sk=str(r['http_status']);v['http_status_counts'][sk]=v['http_status_counts'].get(sk,0)+1;response_models[r['provider']+':'+str(r['response_model'])]+=1
  for u,n in (r['usage'] or {}).items():
   if type(n) in [int,float]:v['usage'][u]=v['usage'].get(u,0)+n
  for body in p.parent.glob('attempt_*.body'):
   meta=json.loads(body.with_suffix('.json').read_text());assert hashlib.sha256(body.read_bytes()).hexdigest()==meta['body_sha256'];hashes_checked+=1
 result={'completed_utc':now(),'primary_new_batch':pooled,'cells':cells,'decision':criteria,'original_batch':core.pooled(old,MODELS),'between_batch_new_minus_original':between,'two_batch_average_descriptive':combined_summary,'resources':resources,'response_model_counts':dict(response_models),'response_body_hashes_verified':hashes_checked,'monetary_estimates':None}
 write_once(ROOT/'results/analysis.json',result);(ROOT/'results/status.json').write_text(json.dumps({'state':'analysis_complete','utc':now(),'candidate_seats_completed':{m:4200 for m in MODELS},'decision':criteria},indent=2)+'\n');print(json.dumps({'decision':criteria,'primary':pooled['primary']},indent=2))
if __name__=='__main__':main()
