"""Offline replay of a completed independent generation batch.

No collection or network operations. The numerical definitions and seeds match
the frozen P1 analysis; score-quality diagnostics use the existing P0 routine.
"""
from pathlib import Path
import collections,csv,hashlib,json,math
import numpy as np
from verify_followup_decisions import verify_rows
from confidence_diagnostics import score_cell
import statistics_core as core
ROOT=Path(__file__).resolve().parents[1]
MODELS=['deepseek','qwen','glm'];TASKS=['gsm8k','commonsenseqa']
read=lambda p:json.loads((ROOT/p).read_text())
def close(a,b,path='root'):
 if isinstance(a,dict):
  # JSON object keys are strings; the live core uses integer beta keys.
  a={str(k):v for k,v in a.items()};b={str(k):v for k,v in b.items()}
  assert set(a)==set(b),path
  for k in a:close(a[k],b[k],path+'.'+k)
 elif isinstance(a,list):
  assert len(a)==len(b),path
  for i,(x,y) in enumerate(zip(a,b)):close(x,y,path+f'[{i}]')
 elif isinstance(a,(int,float)) and not isinstance(a,bool):assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-11),path
 else:assert a==b,path
def csvout(name,rows):
 with (ROOT/'tables'/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def contrast(d,tasks):
 d=np.array(d,float);tasks=np.array(tasks);rng=np.random.default_rng(2026102402);ix=[np.flatnonzero(tasks==t) for t in TASKS]
 boots=sum(d[rng.choice(i,(10000,len(i)))].mean(1)/2 for i in ix)
 return {'effect_pp':100*float(np.mean([d[i].mean() for i in ix])),'ci95_pp':(100*np.quantile(boots,[.025,.975])).tolist(),'nonzero_questions':int(sum(d!=0)),'scope':'descriptive same-question between-batch contrast'}
def main():
 assert (ROOT/'data/independent_replication/collection_complete.json').exists(),'No partial-outcome analysis'
 rows=read('data/independent_replication/evaluation_rows.json');stored=read('data/independent_replication/analysis.json')
 old=[r for r in read('data/evaluation_rows.json') if r['model'] in MODELS]
 assert len(rows)==len(old)==1800 and len({r['id'] for r in rows})==600
 assert {(r['id'],r['model']) for r in rows}=={(r['id'],r['model']) for r in old}
 lock=read('protocols/p1_independent_replication/protocols/original_development_lock.json')['cells']
 verify_rows(rows,lock)
 # PERM uses the identical within-pool paired permutations as the frozen run.
 for r in rows:
  seed=int(hashlib.sha256((r['id']+r['model']+':perm-v7').encode()).hexdigest()[:8],16)
  for b in [4,64]:assert r['correct']['PERM'+str(b)]==core.permuted_accuracy(r['answers'],r['scores'],r['gold'],b,seed)
 pooled=core.pooled(rows,MODELS);close(pooled,stored['primary_new_batch'])
 cells={f'{t}:{m}':core.summarize([r for r in rows if r['task']==t and r['model']==m]) for t in TASKS for m in MODELS};close(cells,stored['cells'])
 close(core.pooled(old,MODELS),stored['original_batch'])
 combined=[{**r,'model':r['model']+'_'+b} for b,rr in [('original',old),('replication',rows)] for r in rr]
 close(core.pooled(combined,[m+'_'+b for b in ['original','replication'] for m in MODELS]),stored['two_batch_average_descriptive'])
 om={(r['id'],r['model']):r for r in old};nm={(r['id'],r['model']):r for r in rows};ids=sorted({r['id'] for r in rows})
 functions={'H1':lambda r:int(r['correct']['M']==1 and r['correct']['W64']==0)-int(r['correct']['M']==1 and r['correct']['W4']==0),'H2':lambda r:r['correct']['G64']-r['correct']['W64']}
 between={k:contrast([np.mean([fn(nm[q,m])-fn(om[q,m]) for m in MODELS]) for q in ids],[nm[q,MODELS[0]]['task'] for q in ids]) for k,fn in functions.items()};close(between,stored['between_batch_new_minus_original'])
 positive={(t,m) for t in TASKS for m in MODELS if sum(functions['H2'](r) for r in rows if r['task']==t and r['model']==m)>0}
 h1=pooled['primary']['H1_harm64_minus_harm4'];h2=pooled['primary']['H2_accuracy_guard64_minus_64'];ret=pooled['correction_retention']
 decision={'H1_independently_supported':h1['effect_pp']>0 and h1['holm_p']<.05,'H2_positive_and_adjusted_significant':h2['effect_pp']>0 and h2['holm_p']<.05,'retention_at_least_75_percent':ret is not None and ret>=.75,'positive_H2_cells':len(positive),'positive_H2_covers_each_model':{m for t,m in positive}==set(MODELS),'positive_H2_covers_both_tasks':{t for t,m in positive}==set(TASKS),'H2_at_least_20_nonzero_questions':h2['nonzero_distinct_questions']>=20}
 decision['joint_gate_claim_supported']=all(decision[k] for k in ['H1_independently_supported','H2_positive_and_adjusted_significant','retention_at_least_75_percent','positive_H2_covers_each_model','positive_H2_covers_both_tasks','H2_at_least_20_nonzero_questions']) and len(positive)>=4
 close(decision,stored['decision'])
 quality={};ledger=[];variation=[];cert=[]
 for t in TASKS:
  for m in MODELS:
   rr=[r for r in rows if r['task']==t and r['model']==m];key=f'independent_generation:{m}:{t}';quality[key]=score_cell(rr,key)
   classes=[len({a for a in r['answers'] if a is not None}) for r in rr]
   for pol in ['M','W0','W1','W2','W4','W8','W16','W32','W64','G64','DEV','RANK4','RANK64']:
    C=sum(r['correct']['M']==0 and r['correct'][pol]==1 for r in rr);H=sum(r['correct']['M']==1 and r['correct'][pol]==0 for r in rr);J=sum(r['predictions'][pol]!=r['majority'] and r['correct'][pol]==r['correct']['M']==0 for r in rr);D=sum(r['predictions'][pol]!=r['majority'] for r in rr);assert D==C+H+J
    ledger.append({'task':t,'model':m,'policy':pol,'n':300,'correct':int(sum(r['correct'][pol] for r in rr)),'C':C,'H':H,'J':J,'changed_answers':D,'empty_pools':sum(v==0 for v in classes),'single_class_pools':sum(v==1 for v in classes),'multiclass_pools':sum(v>1 for v in classes),'missing_generation_seats':sum(a is None for r in rr for a in r['answers']),'missing_candidate_scores':sum(a is not None and s is None for r in rr for a,s in zip(r['answers'],r['scores']))})
    variation.append({'task':t,'model':m,'policy':pol,'n':300,'changed_answer_between_batches':sum(r['predictions'][pol]!=om[r['id'],m]['predictions'][pol] for r in rr),'original_wrong_new_correct':sum(om[r['id'],m]['correct'][pol]==0 and r['correct'][pol]==1 for r in rr),'original_correct_new_wrong':sum(om[r['id'],m]['correct'][pol]==1 and r['correct'][pol]==0 for r in rr)})
   for beta in [4,64]:
    counts=collections.Counter()
    for r in rr:
     cc=collections.Counter(a for a in r['answers'] if a is not None);nn=sorted(cc.values(),reverse=True)
     if not nn:counts['empty']+=1;continue
     if len(nn)==1:counts['single_class']+=1;continue
     counts['multiclass']+=1
     if not r['complete']:counts['incomplete_scores']+=1;continue
     z=[s/100 for a,s in zip(r['answers'],r['scores']) if a is not None]
     safe=nn[0]>nn[1] and beta*(max(z)-min(z))<math.log(nn[0]/nn[1])
     if safe:assert r['predictions']['W'+str(beta)]==r['majority']
     counts['certified_multiclass']+=safe;counts['uncertified_preserved']+=not safe and r['predictions']['W'+str(beta)]==r['majority']
    cert.append({'task':t,'model':m,'beta':beta,**dict(counts)})
 qdata={'scope':'descriptive; same complete-pool WQD, question-weighted Brier, fixed bins and question-bootstrap definitions as P0','cells':quality}
 (ROOT/'data/independent_replication/confidence_quality.json').write_text(json.dumps(qdata,indent=2)+'\n')
 csvout('independent_replication_policy_counts.csv',ledger);csvout('between_batch_decision_variation.csv',variation)
 (ROOT/'data/independent_replication/certificate_diagnostics.json').write_text(json.dumps(cert,indent=2)+'\n')
 result={'status':'verified','model_question_records':1800,'distinct_questions':600,'deterministic_readouts':'independent standard-library replay passed','frozen_primary_statistics':'recomputed and matched','decision':decision,'between_batch_contrasts':'recomputed and matched','two_batch_average':'descriptive, recomputed and matched','score_quality_cells':6,'partial_data_used':False}
 (ROOT/'data/independent_replication/offline_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
