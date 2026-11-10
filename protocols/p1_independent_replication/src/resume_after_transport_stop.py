"""Resume unprocessed seats, retaining completed transport failures unchanged.

Operational wrapper only: the frozen per-seat collector, parser, settings,
analysis, request retry limit and new-request stopping thresholds are unchanged.
Completed cache entries are not new transport failures in a resumed session.
"""
from pathlib import Path
import concurrent.futures,hashlib,json
import run
from collector_base import now

ROOT=Path(__file__).resolve().parents[1]

def model_run(model,questions,keys):
    scheduled=[(q,s) for q in questions for s in range(7)]
    existing={}
    pending_jobs=[]
    for q,s in scheduled:
        path=ROOT/f"results/parsed/{q['id']}-{model}-{s}.json"
        if path.exists():existing[path]=hashlib.sha256(path.read_bytes()).hexdigest()
        else:pending_jobs.append((q,s))
    run.COUNTS[model]=len(existing)
    run.FAILED[model]=0
    jobs=iter(pending_jobs)
    workers=2 if model=='glm' else 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending={}
        def add():
            if run.STOP.is_set():return
            try:q,s=next(jobs)
            except StopIteration:return
            pending[pool.submit(run.one,q,model,s,keys)]=None
        for _ in range(workers):add()
        while pending:
            done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                del pending[future]
                try:row=future.result()
                except Exception:run.STOP.set();raise
                run.COUNTS[model]+=1
                bad=row['generation_http_status']!=200 or row.get('confidence_http_status',200)!=200
                run.FAILED[model]=run.FAILED[model]+1 if bad else 0
                if run.FAILED[model]>=5:
                    run.STOP.set();raise RuntimeError('five consecutive transport-failed bundles: '+model)
                if run.COUNTS[model]%25==0:
                    run.status('collecting')
                    print(json.dumps({'model':model,'completed_seats':run.COUNTS[model],'utc':now()}),flush=True)
                add()
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in existing.items())
    return run.COUNTS[model]

if __name__=='__main__':
    run.verify()
    manifest=json.loads((ROOT/'protocols/transport_resume_freeze.json').read_text())
    for name,digest in manifest['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    run.model_run=model_run
    run.main()
