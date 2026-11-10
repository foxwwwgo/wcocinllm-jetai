"""v7: local-only, fresh-question, matched concentration interventions.
Benchmark artifacts are immutable. No credentials are logged or packaged.
"""
from pathlib import Path
import argparse, collections, concurrent.futures, hashlib, json, math, re
import numpy as np
from collector_base import collect, load_keys, write_once, now

ROOT=Path(__file__).resolve().parents[1]
MODELS=('deepseek','qwen','glm','open_qwen14b')
TASKS=('gsm8k','commonsenseqa')
K=7
BETAS=[0,1,2,4,8,16,32,64]
GEN_INSTRUCTION='Solve the following question. Give a step-by-step explanation, then finish with exactly one line: Proposed answer: (X). Replace X with your answer. '
CONF_INSTRUCTION='Rate your confidence that your proposed answer to the original question is correct on a scale of 0 to 100. Return only one number between 0 and 100, with no explanation.'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(t):return re.sub(r'\W+','',t.casefold())
def provider(m):return read(ROOT/'protocols/providers.json')['providers'][m]
def majority(a):
 c=collections.Counter(x for x in a if x is not None)
 return max(c,key=c.get) if c else None
def shape(a):
 c=collections.Counter(x for x in a if x is not None);s=sorted(c.values(),reverse=True)
 return (s[0]-s[1] if len(s)>1 else (s[0] if s else 0)),c
def select(a,s,beta,guard=False,rank=False):
 m=majority(a);ix=[i for i,x in enumerate(a) if x is not None]
 if not ix or beta==0:return m
 if any(s[i] is None for i in ix):return m
 gap,c=shape(a)
 if guard and gap>=2:return m
 z=np.array([s[i]/100 for i in ix],float)
 if rank:
  z=np.array([(sum(z<v)+.5*(sum(z==v)-1))/(len(z)-1) for v in z]) if len(z)>1 else np.zeros(1)
 w=np.exp(beta*(z-z.max()));sums={x:float(sum(w[j] for j,i in enumerate(ix) if a[i]==x)) for x in c}
 return max(sums,key=sums.get) # first occurrence, including exact sum ties
def gen_job(q,m,seat,batch=None):
 task=q['task'];text=q['question']
 if task=='commonsenseqa':text+='\nOptions:\n'+'\n'.join('('+c['label']+'): '+c['text'] for c in q['choices'])
 instruction=GEN_INSTRUCTION+('X must be one of A, B, C, D, E.' if task=='commonsenseqa' else 'X must be a single signed integer, without units or thousands separators.')
 params={'temperature':1.0,'max_tokens':2048}
 if m=='open_qwen14b':params.update(seed=int(hashlib.sha256(f"v7:{q['id']}:{seat}".encode()).hexdigest()[:8],16)%2147483647,top_p=1,top_k=0,min_p=0,repeat_penalty=1,cache_prompt=False)
 return {'job_id':f"{q['split']}-{q['id']}-{m}-{seat}-generation",'batch':batch or q['split'],'kind':'generation','provider':m,'question_id':q['id'],'task':task,'seat':seat,'messages':[{'role':'user','content':instruction+'\n\n'+text}],'parameters':params}
def conf_job(g,r):
 params={'temperature':0.0,'max_tokens':32}
 if g['provider']=='open_qwen14b':params.update(seed=g['parameters']['seed'],top_p=1,top_k=0,min_p=0,repeat_penalty=1,cache_prompt=False)
 return {**g,'job_id':g['job_id'].replace('-generation','-confidence'),'kind':'confidence','generation_request_key':r['request_key'],'messages':g['messages']+[{'role':'assistant','content':r['raw_reply']},{'role':'user','content':CONF_INSTRUCTION}],'parameters':params}
def parse(job,r):
 if r['http_status']!=200 or r['finish_reason'] not in ('stop','eos'):return None
 text=r['raw_reply'].strip()
 if job['kind']=='confidence':
  if not re.fullmatch(r'(?:\d+(?:\.\d*)?|\.\d+)',text):return None
  z=float(text);return z if math.isfinite(z) and 0<=z<=100 else None
 pattern=r'Proposed answer:\s*\(\s*([A-E])\s*\)\.?' if job['task']=='commonsenseqa' else r'Proposed answer:\s*\(\s*([-+]?\d[\d,]*)\s*\)\.?'
 hits=re.findall(pattern,text,re.I)
 if len(hits)!=1:return None
 return hits[0].upper() if job['task']=='commonsenseqa' else str(int(hits[0].replace(',','')))
def one(q,m,s,keys,batch=None):
 out=ROOT/('raw/capability' if batch=='capability' else 'raw/study')
 j=gen_job(q,m,s,batch);r=collect(j,provider(m),keys,out);a=parse(j,r)
 if r['http_status'] in (401,402,403):raise RuntimeError('authorization/balance failure: '+m)
 row={'question_id':q['id'],'task':q['task'],'split':q['split'],'model':m,'seat':s,'answer':a,'confidence':None,'generation_key':r['request_key']}
 if a is not None:
  cj=conf_job(j,r);cr=collect(cj,provider(m),keys,out);row.update(confidence=parse(cj,cr),confidence_key=cr['request_key'])
  if cr['http_status'] in (401,402,403):raise RuntimeError('authorization/balance failure: '+m)
 write_once(ROOT/('results/capability' if batch=='capability' else 'results/parsed')/f"{q['id']}-{m}-{s}.json",row)
 return row
def capability(keys_path):
 qs=[{'id':'handmade-math','task':'gsm8k','split':'capability','question':'A box contains 7 blue balls and 4 red balls. How many balls are in the box?'},{'id':'handmade-common','task':'commonsenseqa','split':'capability','question':'What do people normally use to keep their hands warm in winter?','choices':[{'label':a,'text':t} for a,t in zip('ABCDE',['gloves','sandals','forks','screens','plates'])]}]
 write_once(ROOT/'protocols/capability_protocol.json',{'created_utc':now(),'questions':qs,'not_benchmark':True,'scope':'transport and format capability only; never tune scientific thresholds using probes'})
 keys=load_keys(keys_path)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:rows=list(ex.map(lambda z:one(*z,keys,batch='capability'),[(q,m,0) for q in qs for m in MODELS]))
 summary={'all_pass':all(r['answer'] is not None and r['confidence'] is not None for r in rows),'records':rows}
 write_once(ROOT/'results/capability_summary.json',summary);print(json.dumps(summary),flush=True)
def prepare():
 old=read(ROOT.parent/'Neurocomputing_Experiment_v4/protocols/gsm8k_selection.json')['selected']
 oldtext={norm(q['question']) for subset in old.values() for q in subset}
 oldcs=read(ROOT.parent/'Neurocomputing_Experiment_v6/protocols/questions.json')
 oldtext.update(norm(q['question']) for q in oldcs)
 # Also exclude any exact normalized question that occurs in v5 protocols.
 prior_text=' '.join(norm(p.read_text()) for p in (ROOT.parent/'Neurocomputing_Experiment_v5/protocols').glob('*.json'))
 questions=[];refs={};audit={}
 for task,file in [('gsm8k','gsm8k_test.jsonl'),('commonsenseqa','commonsenseqa_dev.jsonl')]:
  rows=[json.loads(t) for t in (ROOT/'data'/file).read_text().splitlines()];new=[];excluded=0
  for i,r in enumerate(rows):
   text=r['question'] if task=='gsm8k' else r['question']['stem']
   if norm(text) in oldtext or norm(text) in prior_text:excluded+=1;continue
   q={'id':f'gsm8k-test-{i}' if task=='gsm8k' else 'csqa-'+r['id'],'task':task,'question':text,'source_index':i}
   if task=='commonsenseqa':q['choices']=r['question']['choices']
   gold=str(int(r['answer'].split('####')[-1].strip().replace(',',''))) if task=='gsm8k' else r['answerKey']
   new.append((q,gold))
  new.sort(key=lambda v:hashlib.sha256(('ca-rc-v7-concentration-20261022:'+v[0]['id']).encode()).hexdigest())
  assert len(new)>=350
  for i,(q,gold) in enumerate(new[:350]):q['split']='development' if i<50 else 'test';questions.append(q);refs[q['id']]=gold
  audit[task]={'source_n':len(rows),'excluded_prior_questions':excluded,'remaining':len(new),'selected':350,'development':50,'test':300,'source_sha256':sha(ROOT/'data'/file)}
 assert len({norm(q['question']) for q in questions})==700
 write_once(ROOT/'protocols/questions.json',questions);write_once(ROOT/'protocols/references.json',refs);write_once(ROOT/'protocols/overlap_audit.json',audit)
 print(json.dumps(audit),flush=True)
def freeze():
 assert read(ROOT/'results/capability_summary.json')['all_pass']
 assert read(ROOT/'results/capability_gpu_summary.json')['all_pass']
 assert not list((ROOT/'raw/study').glob('requests/*/request.json'))
 files=[p for folder in ('src','protocols','data') for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.json','.jsonl','.md','.sh')]
 files+=list((ROOT/'results').glob('capability*summary.json'))
 write_once(ROOT/'protocols/freeze.json',{'frozen_utc':now(),'benchmark_requests_before_freeze':0,'files':{str(p.relative_to(ROOT)):sha(p) for p in files}})
def verify():
 for f,h in read(ROOT/'protocols/freeze.json')['files'].items():assert sha(ROOT/f)==h,('changed frozen file',f)
def run(split,keys_path):
 verify()
 if split=='test':assert (ROOT/'protocols/development_lock.json').exists()
 keys=load_keys(keys_path);qs=[q for q in read(ROOT/'protocols/questions.json') if q['split']==split]
 # Separate bounded provider pools: slow GPU work cannot occupy API workers.
 qs.sort(key=lambda q:hashlib.sha256(('schedule:'+q['id']).encode()).hexdigest())
 def run_model(m):
  jobs=iter((q,s) for q in qs for s in range(K));n=0;workers=2 if m=='glm' else 4
  with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
   pending={}
   def add():
    try:q,s=next(jobs)
    except StopIteration:return
    pending[ex.submit(one,q,m,s,keys)]=None
   for _ in range(workers):add()
   while pending:
    done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
    for fut in done:
     del pending[fut];fut.result();n+=1;add()
     if n%50==0:print(json.dumps({'split':split,'model':m,'bundles':n,'total':len(qs)*K,'utc':now()}),flush=True)
  return n
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(MODELS)) as ex:n=sum(ex.map(run_model,MODELS))
 write_once(ROOT/f'results/{split}_complete.json',{'completed_utc':now(),'bundles':n,'questions':len(qs)})
def pools(split):
 out=[]
 for q in read(ROOT/'protocols/questions.json'):
  if q['split']!=split:continue
  for m in MODELS:
   rr=[read(ROOT/f"results/parsed/{q['id']}-{m}-{i}.json") for i in range(K)]
   out.append({'id':q['id'],'task':q['task'],'model':m,'answers':[r['answer'] for r in rr],'scores':[r['confidence'] for r in rr]})
 return out
def lock():
 verify();assert (ROOT/'results/development_complete.json').exists()
 refs=read(ROOT/'protocols/references.json');rows=pools('development');locks={}
 for t in TASKS:
  for m in MODELS:
   rr=[r for r in rows if r['task']==t and r['model']==m]
   hits={str(b):sum(select(r['answers'],r['scores'],b)==refs[r['id']] for r in rr) for b in BETAS}
   best=max(BETAS,key=lambda b:(hits[str(b)],-b))
   locks[t+':'+m]={'beta':best,'correct_by_beta':hits,'n':len(rr)}
 write_once(ROOT/'protocols/development_lock.json',{'locked_utc':now(),'test_requests_before_lock':0,'freeze_sha256':sha(ROOT/'protocols/freeze.json'),'cells':locks,'rule':'max dev accuracy; ties choose smaller beta (less concentration); beta=0 permits majority fallback'})
 print(json.dumps(locks),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','capability','freeze','development','lock','test']);ap.add_argument('--keys');a=ap.parse_args()
 if a.mode in ('development','test'):run(a.mode,a.keys)
 elif a.mode=='capability':capability(a.keys)
 else:globals()[a.mode]()
