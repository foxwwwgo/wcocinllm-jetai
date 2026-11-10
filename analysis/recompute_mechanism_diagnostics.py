"""Retrospective vote-geometry diagnostics and deterministic case selection."""
from pathlib import Path
from collections import Counter
import json,math,hashlib
R=Path(__file__).resolve().parents[1]
SALT='ida-v10.7-review-20261023'
def main():
 source=R/'data/evaluation_rows.json'
 rows=[r for r in json.loads(source.read_text()) if r['model'] in ['deepseek','qwen','glm']]
 ledger=[]
 for r in rows:
  counts=Counter(a for a in r['answers'] if a is not None); n=sorted(counts.values(),reverse=True)
  s=[v/100 for a,v in zip(r['answers'],r['scores']) if a is not None and v is not None]
  span=max(s)-min(s) if s else None
  cert={str(b):bool(r['complete'] and (len(n)==1 or len(n)>1 and n[0]>n[1] and b*span<math.log(n[0]/n[1]))) for b in [4,64]}
  for b in [4,64]:assert not cert[str(b)] or r['predictions']['W'+str(b)]==r['majority']
  ledger.append({'id':r['id'],'model':r['model'],'task':r['task'],'class_count':len(n),'counts':dict(counts),'span':span,'constant_scores':r['score_constant'],'certificate':cert,'majority':r['majority'],'gold':r['gold'],'answers':r['answers'],'scores':r['scores'],'W4':r['predictions']['W4'],'W64':r['predictions']['W64']})
 summary={'records':len(ledger),'single_class':sum(r['class_count']==1 for r in ledger),'multiple_class':sum(r['class_count']>1 for r in ledger),'constant_scores':sum(r['constant_scores'] for r in ledger),'constant_scores_multiple_class':sum(r['constant_scores'] and r['class_count']>1 for r in ledger),'by_beta':{}}
 for b in [4,64]:
  summary['by_beta'][str(b)]={'certified':sum(r['certificate'][str(b)] for r in ledger),'certified_multiple_class':sum(r['certificate'][str(b)] and r['class_count']>1 for r in ledger),'actual_overrides':sum(r['W'+str(b)]!=r['majority'] for r in ledger),'uncertified_but_preserved':sum(not r['certificate'][str(b)] and r['W'+str(b)]==r['majority'] for r in ledger)}
 strata={'correction':lambda r:r['majority']!=r['gold'] and r['W64']==r['gold'],'harm':lambda r:r['majority']==r['gold'] and r['W64']!=r['gold'],'preserved':lambda r:r['class_count']>1 and r['certificate']['64']}
 examples={}
 for category,predicate in strata.items():
  eligible=[r for r in ledger if predicate(r)]
  choice=dict(min(eligible,key=lambda r:hashlib.sha256((SALT+':'+category+':'+r['id']+':'+r['model']).encode()).hexdigest()))
  choice['eligible_records']=len(eligible)
  choice['class_weights']={}
  for b in [4,64]:
   top=max(s for a,s in zip(choice['answers'],choice['scores']) if a is not None)
   totals={c:sum(math.exp(b*(s-top)/100) for a,s in zip(choice['answers'],choice['scores']) if a==c) for c in choice['counts']}
   choice['class_weights'][str(b)]=totals
  examples[category]=choice
 result={'status':'retrospective descriptive diagnostics','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'selection_salt':SALT,'selection_rule':'Lowest SHA256(salt:category:id:model) within W64 correction, W64 harm, and multiclass W64-certified-preservation strata. No ranking by effect size.','summary':summary,'examples':examples}
 (R/'data/mechanism_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
 (R/'data/certificate_pool_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
