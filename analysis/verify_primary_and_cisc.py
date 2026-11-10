"""Offline decision replay and reported-count verification; no network access."""
from pathlib import Path
from collections import Counter, defaultdict
import json, math
from verify_followup_decisions import verify_rows

R = Path(__file__).resolve().parents[1]
def read(p): return json.loads((R/p).read_text())
def summary(rows, policies, correctness='correct'):
    result={}
    for p in policies:
        c=sum(bool(not r[correctness]['M'] and r[correctness][p]) for r in rows)
        h=sum(bool(r[correctness]['M'] and not r[correctness][p]) for r in rows)
        correct=sum(int(r[correctness][p]) for r in rows)
        assert c-h==correct-sum(int(r[correctness]['M']) for r in rows)
        result[p]={'correct':correct,'corrections':c,'harms':h,'net':c-h}
    return result

rows=read('data/evaluation_rows.json')
assert len(rows)==2400 and len({r['id'] for r in rows})==600
dev={r['task']+':'+r['model']:{'beta':0} for r in rows}
verify_rows(rows,dev)
primary=[r for r in rows if r['model'] in ['deepseek','qwen','glm']]
assert len(primary)==1800
res=summary(primary,['M','W4','W64','G64'])
assert [res[p]['correct'] for p in res]==[1677,1676,1669,1673]
assert [(res[p]['corrections'],res[p]['harms']) for p in ['W4','W64','G64']]==[(2,3),(5,13),(3,7)]
group=defaultdict(list)
for r in primary: group[r['task'],r['id']].append(r)
h1=[];h2=[]
for rr in group.values():
    assert len(rr)==3
    h1.append(sum(int(r['correct']['M'] and not r['correct']['W64'])-int(r['correct']['M'] and not r['correct']['W4']) for r in rr)/3)
    h2.append(sum(r['correct']['G64']-r['correct']['W64'] for r in rr)/3)
stored=read('data/analysis.json')['commercial_pooled']['primary']
effects={}
for name,values,nonzero in [('H1_harm64_minus_harm4',h1,9),('H2_accuracy_guard64_minus_64',h2,8)]:
    effect=100*sum(values)/len(values)
    assert math.isclose(effect,stored[name]['effect_pp'],abs_tol=1e-12)
    assert sum(x!=0 for x in values)==nonzero==stored[name]['nonzero_distinct_questions']
    effects[name]={'effect_pp':effect,'nonzero_questions':nonzero}

cisc=read('data/cisc/evaluation_rows.json')
lock=read('protocols/cisc/temperature_lock.json')['methods']
assert len(cisc)==600 and len({r['id'] for r in cisc})==600
for r in cisc:
    aa=r['answers']; counts=Counter(a for a in aa if a is not None)
    m=max(counts,key=counts.get) if counts else None
    top=[a for a in counts if counts[a]==max(counts.values())] if counts else []
    assert m==r['majority']==r['prediction']['M'] and top==r['top']
    assert r['O']==bool(m!=r['gold'] and r['gold'] in counts)
    assert r['E']==bool(len(top)>1 and r['gold'] in top)
    for branch,key in [('ptrue','CISC_P'),('verbal','CISC_V')]:
        if not counts or not r['confidence_complete'][branch]: prediction=m
        else:
            pairs=[(a,s) for a,s in zip(aa,r[branch]) if a is not None]
            largest=max(s for _,s in pairs)
            totals={a:0. for a in counts}
            for a,s in pairs: totals[a]+=math.exp((s-largest)/lock[branch]['temperature'])
            prediction=max(totals,key=totals.get)
        assert prediction==r['prediction'][key], (r['id'],key)
    if not counts or not r['confidence_complete']['ptrue']:
        predictions={p:m for p in ['T_P','W_P','A21_P']}
    else:
        totals={a:sum(s for answer,s in zip(aa,r['ptrue']) if answer==a) for a in counts}
        def lex(a): return (totals[a],counts[a],-aa.index(a))
        gated=(None if sum(counts.values())<=3 else m if len(top)==1 else
               top[0] if .5*sum(totals.values())<=1e-12 else max(totals,key=totals.get))
        predictions={'T_P':max(top,key=lex),'W_P':max(counts,key=lex),'A21_P':gated}
    for p,v in predictions.items(): assert v==r['prediction'][p],(r['id'],p)
    assert r['prediction']['A21_F0']==m
    for p,v in r['prediction'].items(): assert r['correct'][p]==(v==r['gold'])
counts=summary(cisc,['M','CISC_P','CISC_V','T_P','W_P','A21_P','A21_F0'])
assert [counts[p]['correct'] for p in ['M','CISC_P','CISC_V']]==[513,512,511]
assert [(counts[p]['corrections'],counts[p]['harms']) for p in ['CISC_P','CISC_V']]==[(8,9),(3,5)]
assert sum(r['O'] for r in cisc)==46 and sum(r['E'] for r in cisc)==0
for p in ['T_P','W_P','A21_P','A21_F0']: assert counts[p]['net']==0
result={'status':'verified','initial_records':len(rows),'primary_questions':600,'primary_counts':res,
        'primary_effects':effects,'cisc_records':600,'cisc_counts':counts,
        'cisc_unrestricted_opportunities':46,'cisc_effective_tie_opportunities':0,
        'scope':'Deterministic parsed-record replay and primary point estimates; original interval estimates are preserved separately.'}
(R/'data/reproduced_primary_and_cisc.json').write_text(json.dumps(result,indent=2)+'\n')
print('Verified initial deterministic readouts, primary effects and 600 CISC decisions across all controls.')
