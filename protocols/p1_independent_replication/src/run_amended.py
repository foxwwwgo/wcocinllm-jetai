"""Documented pre-evaluation parser compatibility amendment; originals retained."""
import argparse,concurrent.futures,hashlib,json,re
import study
from collector_base import collect,load_keys,write_once,now
ROOT=study.ROOT
original_parse=study.parse
def compatible_parse(job,r):
 value=original_parse(job,r)
 if value is not None or job['kind']!='generation':return value
 if r['http_status']!=200 or r['finish_reason'] not in ('stop','eos'):return None
 lines=[x.strip() for x in r['raw_reply'].splitlines() if x.strip()]
 if not lines:return None
 line=re.sub(r'[*`]', '',lines[-1])
 token=r'[A-E]' if job['task']=='commonsenseqa' else r'[-+]?\d[\d,]*'
 m=re.fullmatch(r'Proposed answer:\s*\(?\s*('+token+r')\s*\)?\.?',line,re.I)
 if not m:return None
 return m.group(1).upper() if job['task']=='commonsenseqa' else str(int(m.group(1).replace(',','')))
def one(q,m,s,keys,batch=None):
 j=study.gen_job(q,m,s,batch);r=collect(j,study.provider(m),keys,ROOT/'raw/study');a=compatible_parse(j,r)
 if r['http_status'] in (401,402,403):raise RuntimeError('authorization/balance failure: '+m)
 row={'question_id':q['id'],'task':q['task'],'split':q['split'],'model':m,'seat':s,'answer':a,'strict_answer':original_parse(j,r),'confidence':None,'generation_key':r['request_key'],'parser':'v7.1-explicit-final-answer-line'}
 if a is not None:
  cj=study.conf_job(j,r);cr=collect(cj,study.provider(m),keys,ROOT/'raw/study');row.update(confidence=compatible_parse(cj,cr),confidence_key=cr['request_key'])
  if cr['http_status'] in (401,402,403):raise RuntimeError('authorization/balance failure: '+m)
 write_once(ROOT/f"results/parsed_v71/{q['id']}-{m}-{s}.json",row)
 return row
def pools(split):
 rows=[]
 for q in study.read(ROOT/'protocols/questions.json'):
  if q['split']!=split:continue
  for m in study.MODELS:
   rr=[study.read(ROOT/f"results/parsed_v71/{q['id']}-{m}-{s}.json") for s in range(study.K)]
   rows.append({'id':q['id'],'task':q['task'],'model':m,'answers':[r['answer'] for r in rr],'strict_answers':[r['strict_answer'] for r in rr],'scores':[r['confidence'] for r in rr]})
 return rows
def verify():
 study.verify()
 d=study.read(ROOT/'protocols/parser_amendment.json')
 for f,h in d['files'].items():assert study.sha(ROOT/f)==h,('amendment changed',f)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['development','lock','test','analyze']);ap.add_argument('--keys');a=ap.parse_args();verify()
 study.parse=compatible_parse;study.one=one;study.pools=pools
 if a.mode in ('development','test'):study.run(a.mode,a.keys)
 elif a.mode=='lock':study.lock()
 else:
  import analyze
  analyze.main()
if __name__=='__main__':main()
