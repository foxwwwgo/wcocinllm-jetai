"""Gold-free full-response parser replay and deterministic failure sampling."""
from pathlib import Path
from collections import Counter, defaultdict
import gzip, hashlib, json, re
from frozen_parsers import parse, parse_v71, parse_strict
ROOT=Path(__file__).resolve().parents[1]
SALT='ida-v10.7-review-20261023'

def category(r):
    if r['http_status']!=200 or r['finish_reason'] not in ('stop','eos'):return 'transport_or_finish'
    text=r['raw_reply'].strip()
    if not text:return 'empty'
    token=r'[A-E]' if r['task']=='commonsenseqa' else r'[-+]?\d[\d,]*'
    if len(re.findall(r'Proposed answer:\s*\(\s*('+token+r')\s*\)\.?',text,re.I))>1:return 'multiple_proposed_markers'
    if re.search(r'proposed\s+answer|final\s+answer|answer\s*:|\bX\s*=',text,re.I):return 'answer_marker_not_accepted'
    return 'no_accepted_answer_marker'

def main():
    path=ROOT/'data/parser_outputs.jsonl.gz'
    rows=[json.loads(line) for line in gzip.decompress(path.read_bytes()).decode().splitlines()]
    total=Counter();cells=defaultdict(Counter);ledger=[];failures=defaultdict(list)
    for r in rows:
        assert hashlib.sha256(r['raw_reply'].encode()).hexdigest()==r['raw_reply_sha256']
        j={'kind':'generation','task':r['task']}
        a=parse_v71(j,r);b=parse(j,r)
        assert a==r['original_answer'] and b==r['extended_answer'], (r['generation_key'],a,b)
        cls='accepted_original' if a is not None else 'accepted_compatibility_only' if b is not None else category(r)
        key=r['task']+':'+r['model'];total[cls]+=1;cells[key][cls]+=1
        d={k:r[k] for k in ['question_id','task','model','seat','generation_key','raw_reply_sha256']}
        d['category']=cls;ledger.append(d)
        if b is None:failures[key].append(r)
    sample=[]
    for cell in sorted(failures):
        chosen=sorted(failures[cell],key=lambda r:hashlib.sha256((SALT+':'+r['generation_key']).encode()).hexdigest())[:20]
        for r in chosen:
            sample.append({k:r[k] for k in ['question_id','task','model','seat','generation_key','raw_reply_sha256','raw_reply']}|{'category':category(r)})
    tests=json.loads((ROOT/'data/parser_boundary_tests.json').read_text())
    for case in tests:
        j={'kind':case.get('kind','generation'),'task':case['task']}
        rr={'http_status':case.get('http_status',200),'finish_reason':case.get('finish_reason','stop'),'raw_reply':case['text']}
        assert parse_strict(j,rr)==case['strict'],case['name']
        assert parse_v71(j,rr)==case['original'],case['name']
        assert parse(j,rr)==case['compatibility'],case['name']
    out={'status':'verified','rows':len(rows),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
         'category_counts':dict(total),'by_cell':{k:dict(v) for k,v in cells.items()},
         'boundary_tests':len(tests),'sample_n':len(sample),'selection_salt':SALT,
         'scope':'Retrospective syntax census; categories do not determine mathematical correctness. Original and compatibility outputs match frozen records exactly.'}
    (ROOT/'data/parser_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    (ROOT/'data/parser_failure_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    (ROOT/'data/parser_review_sample.json').write_text(json.dumps(sample,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
