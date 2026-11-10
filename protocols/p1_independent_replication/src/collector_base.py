"""Immutable, resumable collection. Full transport bytes are durable before parsing.
No credential is stored in a request, response index, or exception summary.
"""
from pathlib import Path
import argparse,concurrent.futures,fcntl,hashlib,json,math,os,re,socket,time,urllib.request,urllib.error
from datetime import datetime,timezone
VERSION='ca-rc-collector-v4.1'
ROOT=Path(__file__).resolve().parents[1]
def now():return datetime.now(timezone.utc).isoformat()
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def write_once(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    b=data if isinstance(data,bytes) else canonical(data)+b'\n'
    try:
        with path.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    except FileExistsError:
        if path.read_bytes()!=b:raise RuntimeError('immutable artifact collision: '+str(path))
def parse_scores(content):
    # Parse final content only, never hidden reasoning. A single array may be wrapped in prose/code fences.
    hits=[]
    for m in re.finditer(r'\[[^\[\]]*\]',content or '',re.S):
        try:v=json.loads(m.group())
        except (ValueError,TypeError):continue
        if isinstance(v,list) and len(v)==7 and all(type(x) in (int,float) and math.isfinite(x) and 0<=x<=1 for x in v):hits.append([float(x) for x in v])
    return hits[0] if len(hits)==1 else None

def parse_answer(content):
    # Explicit protocol: final standalone number, optional thousands commas/fraction/percent.
    s=(content or '').strip().split('\n')[-1].strip().strip('`').strip()
    s=re.sub(r'^(?:answer\s*[:=]|####)\s*','',s,flags=re.I)
    if not re.fullmatch(r'[-+]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?%?',s):return None
    try:
        s=s.replace(',','').replace(' ','');percent=s.endswith('%');s=s.rstrip('%')
        a,b=s.split('/') if '/' in s else (s,'1');x=float(a)/float(b)/(100 if percent else 1)
        if not math.isfinite(x):return None
        x=round(x,6);return str(int(x)) if x==int(x) else str(x)
    except (ValueError,ZeroDivisionError):return None

def payload_for(job,provider):
    p={'model':provider['model'],'messages':job['messages'],'stream':False,**job['parameters']}
    p.update(provider.get('extra_body',{}));return p

def identity(job,provider):
    return {'schema':VERSION,'job':job,'endpoint':provider['endpoint'],'requested_model':provider['model'],'payload':payload_for(job,provider)}

def request_id(job,provider):return digest(identity(job,provider))

def load_keys(path):
    d=json.loads(Path(path).read_text()) if path and Path(path).is_file() else {}
    return {k:os.environ.get(k) or d.get(k) for k in ['DEEPSEEK_API_KEY','DASHSCOPE_API_KEY','GLM_API_KEY']}

def transport(endpoint,payload,key,timeout):
    headers={'Content-Type':'application/json'}
    if key:headers['Authorization']='Bearer '+key
    req=urllib.request.Request(endpoint,data=canonical(payload),headers=headers)
    try:
        with urllib.request.urlopen(req,timeout=timeout) as r:return r.status,dict(r.headers),r.read(),None
    except urllib.error.HTTPError as e:return e.code,dict(e.headers),e.read(),None
    except Exception as e:
        # The exception class suffices for retry; avoid accidental credential-bearing exception text.
        return None,{},b'',type(e).__name__

def collect(job,provider,keys,out,transport_fn=transport):
    ident=identity(job,provider);rid=digest(ident);folder=Path(out)/'requests'/rid
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        write_once(folder/'request.json',ident)
        result_path=folder/'result.json'
        if result_path.exists():return json.loads(result_path.read_text())
        key=keys.get(provider.get('key_env'))
        if provider.get('key_env') and not key:raise RuntimeError('missing credential variable '+provider['key_env'])
        attempts=[];obj=None;status=None;error=None;duration=0
        for attempt in range(1,4):
            prefix=folder/f'attempt_{attempt:02d}';metadata=Path(str(prefix)+'.json');body_path=Path(str(prefix)+'.body')
            if metadata.exists():
                m=json.loads(metadata.read_text());body=body_path.read_bytes();status=m['status'];error=m['error_class'];duration=m['latency_seconds']
                assert hashlib.sha256(body).hexdigest()==m['body_sha256']
            elif body_path.exists():
                raise RuntimeError('transport body exists without commit metadata; inspect '+str(folder))
            else:
                start=now();tic=time.monotonic();status,headers,body,error=transport_fn(provider['endpoint'],ident['payload'],key,provider.get('timeout',120));duration=time.monotonic()-tic
                # First persist untruncated provider body, before JSON/answer parsing.
                write_once(body_path,body)
                safe_headers={k:v for k,v in headers.items() if k.lower() in ['date','x-request-id','request-id','x-dashscope-request-id','retry-after','content-type']}
                m={'started_utc':start,'finished_utc':now(),'latency_seconds':duration,'status':status,'error_class':error,'response_headers':safe_headers,'body_sha256':hashlib.sha256(body).hexdigest(),'body_bytes':len(body)}
                write_once(metadata,m)
            attempts.append(m)
            if status==200:
                try:obj=json.loads(body.decode('utf-8'))
                except (ValueError,UnicodeError):error='InvalidJSONResponse'
                break
            if status not in [None,408,429,500,502,503,504]:break
            if attempt<3:time.sleep(min(2**attempt,8))
        msg={};finish=None
        if obj and obj.get('choices'):
            msg=obj['choices'][0].get('message',{});finish=obj['choices'][0].get('finish_reason')
        content=msg.get('content') or '';content=content if isinstance(content,str) else ''
        parsed=parse_scores(content) if job['kind']=='score' else parse_answer(content)
        # Length-limited or missing final responses are failures under the prespecified rule.
        ok=status==200 and parsed is not None and finish in ['stop','eos']
        result={'request_key':rid,'job_id':job['job_id'],'batch':job['batch'],'kind':job['kind'],'provider':job['provider'],'requested_model':provider['model'],'response_model':obj.get('model') if obj else None,'response_id':obj.get('id') if obj else None,'provider_request_id':obj.get('request_id') if obj else None,'system_fingerprint':obj.get('system_fingerprint') if obj else None,'usage':obj.get('usage') if obj else None,'finish_reason':finish,'http_status':status,'error_class':error,'parse_ok':ok,'parsed':parsed if ok else None,'raw_reply':content,'raw_reply_preview':content[:300],'raw_reply_truncated':False,'attempts':len(attempts),'latency_seconds':sum(a['latency_seconds'] for a in attempts),'completed_utc':now(),'identity_sha256':rid,'full_response_files':[f'attempt_{i+1:02d}.body' for i in range(len(attempts))]}
        write_once(result_path,result);return result

def run_jobs(jobs,config,keys,out,workers=4):
    ids=[request_id(j,config['providers'][j['provider']]) for j in jobs]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate logical requests; refusing ambiguous batch')
    results=[]
    def one(j):return collect(j,config['providers'][j['provider']],keys,out)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(one,j):j for j in jobs}
        for fut in concurrent.futures.as_completed(pending):
            r=fut.result();results.append(r)
            if len(results)%20==0 or len(results)==len(jobs):print(json.dumps({'completed':len(results),'total':len(jobs),'parse_ok':sum(x['parse_ok'] for x in results)},ensure_ascii=False),flush=True)
            if r['http_status'] in [401,402,403]:
                for f in pending:f.cancel()
                raise RuntimeError('provider authorization/balance failure; batch stopped without logging credentials')
    return results

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--jobs',required=True);ap.add_argument('--config',required=True);ap.add_argument('--keys');ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--limit',type=int)
    a=ap.parse_args();jobs=[json.loads(x) for x in Path(a.jobs).read_text().splitlines() if x.strip()]
    if a.limit:jobs=jobs[:a.limit]
    cfg=json.loads(Path(a.config).read_text());keys=load_keys(a.keys)
    needed={cfg['providers'][j['provider']].get('key_env') for j in jobs}-{None}
    missing=[k for k in needed if not keys.get(k)]
    if missing:raise SystemExit('Missing credentials: '+', '.join(missing))
    rr=run_jobs(jobs,cfg,keys,a.out,a.workers)
    print(json.dumps({'finished':len(rr),'valid':sum(r['parse_ok'] for r in rr)},ensure_ascii=False))
if __name__=='__main__':main()
