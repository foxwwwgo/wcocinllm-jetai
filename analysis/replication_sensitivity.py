"""Post-hoc sensitivity diagnostics; never replaces the frozen H1/H2 analysis."""
from pathlib import Path
import collections
import itertools
import json
import math
import numpy as np
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
MODELS=('deepseek','qwen','glm')
TASKS=('gsm8k','commonsenseqa')

def holm(values):
    result={};previous=0.0
    for i,k in enumerate(sorted(values,key=values.get)):
        previous=max(previous,min(1.0,(len(values)-i)*values[k]))
        result[k]=previous
    return result

def analyze(rows):
    groups=collections.defaultdict(list)
    for r in rows:
        if r['model'] in MODELS:groups[r['id']].append(r)
    ids=sorted(groups)
    assert all({r['model'] for r in group}==set(MODELS) and len(group)==3 for group in groups.values())
    strata=[np.array([i for i,q in enumerate(ids) if groups[q][0]['task']==task]) for task in TASKS]
    rng=np.random.default_rng(2026102207)
    boot=[rng.choice(ix,size=(10000,len(ix)),replace=True) for ix in strata]
    arrays={'H1':[],'H2':[]}
    for q in ids:
        g=groups[q]
        arrays['H1'].append(sum(int(r['correct']['M']==1 and r['correct']['W64']==0)-int(r['correct']['M']==1 and r['correct']['W4']==0) for r in g)/3)
        arrays['H2'].append(sum(r['correct']['G64']-r['correct']['W64'] for r in g)/3)
    out={}
    for key,values in arrays.items():
        x=np.array(values);mean=np.mean([x[ix].mean() for ix in strata])
        v=[x[ix].var(ddof=1)/(4*len(ix)) for ix in strata];variance=sum(v)
        df=variance**2/sum(a*a/(len(ix)-1) for a,ix in zip(v,strata))
        p=float(2*stats.t.sf(abs(mean)/math.sqrt(variance),df))
        bootmean=sum(x[b].mean(axis=1) for b in boot)/2
        nonzero=x[x!=0]
        # Exact enumeration is feasible here because only 7--9 clusters change.
        assert len(nonzero)<=20
        sums=np.array([np.dot(sign,abs(nonzero)) for sign in itertools.product((-1,1),repeat=len(nonzero))])
        signp=float(np.mean(abs(sums)>=abs(nonzero.sum())-1e-12))
        out[key]={'effect_pp':float(100*mean),'ci95_pp':(100*np.quantile(bootmean,[.025,.975])).tolist(),'welch_raw_p':p,'nonzero_questions':len(nonzero),'positive_questions':int(sum(nonzero>0)),'negative_questions':int(sum(nonzero<0)),'sign_symmetry_raw_p':signp,'nonzero_question_ids':[q for q,z in zip(ids,values) if z]}
    for field,target in [('welch_raw_p','welch_holm_p'),('sign_symmetry_raw_p','sign_symmetry_holm_p')]:
        adjusted=holm({key:value[field] for key,value in out.items()})
        for key in out:out[key][target]=adjusted[key]
    return {'independent_questions':len(ids),'model_question_records':3*len(ids),'task_question_counts':{t:len(ix) for t,ix in zip(TASKS,strata)},'contrasts':out}

def main():
    original=json.loads((ROOT/'data/evaluation_rows.json').read_text())
    new=json.loads((ROOT/'data/independent_replication/evaluation_rows.json').read_text())
    bad={r['id'] for r in new if r['candidate_n']!=7 or not r['complete']}
    initial=analyze(original);full=analyze(new);complete=analyze([r for r in new if r['id'] not in bad])
    frozen=json.loads((ROOT/'data/independent_replication/analysis.json').read_text())['primary_new_batch']['primary']
    for k,fk in [('H1','H1_harm64_minus_harm4'),('H2','H2_accuracy_guard64_minus_64')]:
        assert np.allclose(full['contrasts'][k]['effect_pp'],frozen[fk]['effect_pp'],rtol=0,atol=1e-11)
        assert np.allclose(full['contrasts'][k]['ci95_pp'],frozen[fk]['ci95_pp'],rtol=0,atol=1e-11)
        assert abs(full['contrasts'][k]['welch_holm_p']-frozen[fk]['holm_p'])<1e-11
    overlap={}
    for k in ['H1','H2']:
        a=set(initial['contrasts'][k]['nonzero_question_ids']);b=set(full['contrasts'][k]['nonzero_question_ids'])
        overlap[k]={'original':len(a),'independent_batch':len(b),'intersection':len(a&b),'union':len(a|b)}
    result={'scope':'Post-hoc sensitivity only. Original frozen analyses and all-question denominators are unchanged. No new generation or outcome-dependent replacement.',
            'sign_null':'Independent symmetric signs of nonzero question-level contrasts; stronger than the mean-zero null. Exact enumeration, two-sided, Holm over H1/H2 separately in each batch.',
            'complete_question_rule':'Exclude an entire question if any hosted configuration has fewer than seven valid candidates or incomplete confidence. Retain all three models and equal task weights in remaining clusters. This changes the population and does not replace the scheduled-question analysis.',
            'original':initial,'independent_batch':full,'excluded_incomplete_questions':len(bad),'complete_question_sensitivity':complete,'between_batch_event_overlap':overlap,'frozen_estimates_verified':True}
    dest=ROOT/'data/independent_replication/posthoc_sensitivity.json'
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print('Post-hoc sensitivity reproduced: 600 full / 595 complete question clusters; frozen estimates unchanged.')

if __name__=='__main__':main()
