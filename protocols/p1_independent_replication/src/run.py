"""Resumable fresh-batch collection; frozen code, local API credentials only."""
from pathlib import Path
import argparse,concurrent.futures,fcntl,hashlib,json,os,shutil,threading,time
import study,run_amended
from collector_base import collect,load_keys,write_once,now
ROOT=Path(__file__).resolve().parents[1];BATCH='ida-p1-independent-20261024';MODELS=['deepseek','qwen','glm'];STOP=threading.Event();LOCK=threading.Lock();COUNTS={m:0 for m in MODELS};FAILED={m:0 for m in MODELS}
def verify():
 manifest=json.loads((ROOT/'protocols/freeze.json').read_text())
 for name,h in manifest['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,('Changed frozen file',name)
def status(state,**kw):
 with LOCK:
  p=ROOT/'results/status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps({'state':state,'batch':BATCH,'utc':now(),'candidate_seats_completed':dict(COUNTS),'candidate_seats_per_model':4200,**kw},indent=2)+'\n');os.replace(tmp,p)
def one(q,m,s,keys):
 target=ROOT/f"results/parsed/{q['id']}-{m}-{s}.json"
 if target.exists():return json.loads(target.read_text())
 if STOP.is_set():raise RuntimeError('collection stopped')
 if shutil.disk_usage(ROOT).free<512*1024**2:STOP.set();raise RuntimeError('storage below frozen reserve')
 g=study.gen_job(q,m,s,batch=BATCH);g['job_id']=BATCH+'-'+g['job_id'];g['replication_batch']=BATCH
 r=collect(g,study.provider(m),keys,ROOT/'raw/study')
 if r['http_status'] in [400,401,402,403,404,422]:STOP.set();raise RuntimeError(f'provider {m}: HTTP {r["http_status"]}; no further scheduling')
 a=run_amended.compatible_parse(g,r);row={'batch':BATCH,'question_id':q['id'],'task':q['task'],'model':m,'seat':s,'answer':a,'strict_answer':study.parse(g,r),'confidence':None,'generation_key':r['request_key'],'generation_http_status':r['http_status'],'generation_finish':r['finish_reason']}
 if a is not None:
  c=study.conf_job(g,r);cr=collect(c,study.provider(m),keys,ROOT/'raw/study')
  if cr['http_status'] in [400,401,402,403,404,422]:STOP.set();raise RuntimeError(f'provider {m}: HTTP {cr["http_status"]}; no further scheduling')
  row.update(confidence=run_amended.compatible_parse(c,cr),confidence_key=cr['request_key'],confidence_http_status=cr['http_status'],confidence_finish=cr['finish_reason'])
 write_once(target,row);return row
def model_run(m,qs,keys):
 jobs=iter((q,s) for q in qs for s in range(7));workers=2 if m=='glm' else 4
 with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
  pending={}
  def add():
   if STOP.is_set():return
   try:q,s=next(jobs)
   except StopIteration:return
   pending[pool.submit(one,q,m,s,keys)]=None
  for _ in range(workers):add()
  while pending:
   done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
   for fut in done:
    del pending[fut]
    try:r=fut.result()
    except Exception:STOP.set();raise
    COUNTS[m]+=1;bad=r['generation_http_status']!=200 or r.get('confidence_http_status',200)!=200;FAILED[m]=FAILED[m]+1 if bad else 0
    if FAILED[m]>=5:STOP.set();raise RuntimeError('five consecutive transport-failed bundles: '+m)
    if COUNTS[m]%25==0:status('collecting');print(json.dumps({'model':m,'completed_seats':COUNTS[m],'utc':now()}),flush=True)
    add()
 return COUNTS[m]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--keys',required=True);a=ap.parse_args();verify()
 with (ROOT/'results/collector.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  keys=load_keys(a.keys);assert all(keys.values()),'Missing local credential(s)'
  qs=json.loads((ROOT/'protocols/questions.json').read_text());qs.sort(key=lambda q:hashlib.sha256(('schedule:'+q['id']).encode()).hexdigest());assert len(qs)==600
  starts=ROOT/'results/starts.jsonl'
  with starts.open('a') as f:f.write(json.dumps({'utc':now(),'freeze_sha256':hashlib.sha256((ROOT/'protocols/freeze.json').read_bytes()).hexdigest(),'pid':os.getpid()})+'\n')
  status('collecting')
  try:
   with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:result=list(ex.map(lambda m:model_run(m,qs,keys),MODELS))
   assert result==[4200]*3
   write_once(ROOT/'results/collection_complete.json',{'utc':now(),'batch':BATCH,'candidate_seats':sum(result),'model_question_records':1800,'independent_questions':600})
   status('collection_complete')
  except Exception as e:
   status('stopped',reason=str(e) if isinstance(e,(RuntimeError,AssertionError)) else type(e).__name__);raise
if __name__=='__main__':main()
