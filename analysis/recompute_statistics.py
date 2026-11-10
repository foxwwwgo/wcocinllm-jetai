"""Recompute intervals/tests from rows; archived estimates are comparison targets only.

Run with Python 3.11+ and requirements.txt. No network, credentials or model calls.
Initial and extension: PCG64 seed 2026102207, 10,000 resamples, task order
GSM8K then CommonsenseQA, lexically sorted question IDs, whole-question clusters.
Cells keep stored row order. CISC keeps stored row order and advances one PCG64
stream (seed 20261022) in the declared six-policy order. Linear quantiles.
"""
from pathlib import Path
import collections, hashlib, itertools, json, math, platform
import numpy as np
import scipy
from scipy import stats
from statistics_core import pooled, summarize, permuted_accuracy
from verify_followup_decisions import decisions, gap, BETAS

ROOT = Path(__file__).resolve().parents[1]
TASKS = ['gsm8k', 'commonsenseqa']
HOSTED = ['deepseek', 'qwen', 'glm']
INITIAL = HOSTED + ['open_qwen14b']
ADDED = ['granite', 'phi']
ATOL = 1e-10
checks = []

def read(name):
    return json.loads((ROOT/name).read_text())

def compare(actual, expected, location):
    """Compare every recomputed field, allowing only numerical roundoff."""
    if isinstance(actual, dict):
        for key, value in actual.items():
            stored_key=key if key in expected else str(key)
            assert stored_key in expected, (location, key)
            compare(value, expected[stored_key], location+'/'+str(key))
    elif isinstance(actual, list):
        assert len(actual) == len(expected), location
        for i, value in enumerate(actual): compare(value, expected[i], location+'/'+str(i))
    elif actual is None or isinstance(actual, (str, bool)):
        assert actual == expected, (location, actual, expected)
        checks.append(location)
    else:
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=ATOL), (location, actual, expected)
        checks.append(location)

def cells(rows, models):
    return {t+':'+m: summarize([r for r in rows if r['task']==t and r['model']==m])
            for t in TASKS for m in models}

def recompute_permutations(rows):
    for r in rows:
        seed=int(hashlib.sha256((r['id']+r['model']+':perm-v7').encode()).hexdigest()[:8],16)
        for b in [4,64]:
            value=permuted_accuracy(r['answers'],r['scores'],r['gold'],b,seed)
            compare(value,r['correct']['PERM'+str(b)],'permutation/'+r['id']+'/'+r['model']+'/'+str(b))
            r['correct']['PERM'+str(b)]=value

def strict_rows(rows):
    out=[]
    for r in rows:
        z=dict(r, answers=r['strict_answers'])
        pred={p:decisions(z,b) for p,b in [('M',0),('DEV',0)]+[('W'+str(b),b) for b in BETAS]}
        pred.update(G64=pred['M'] if gap(z)>=2 else pred['W64'])
        pred.update({f'RANK{b}':decisions(z,b,rank=True) for b in [4,64]})
        correct={p:float(a==r['gold']) for p,a in pred.items()}
        seed=int(hashlib.sha256((r['id']+r['model']+':perm-v7').encode()).hexdigest()[:8],16)
        correct.update({f'PERM{b}':permuted_accuracy(z['answers'],r['scores'],r['gold'],b,seed) for b in [4,64]})
        out.append(dict(z, majority=pred['M'], predictions=pred, correct=correct))
    return out

def cisc_statistics(rows):
    rng=np.random.default_rng(20261022)
    result={}
    for p in ['CISC_P','CISC_V','T_P','W_P','A21_P','A21_F0']:
        d=np.array([int(r['correct'][p])-int(r['correct']['M']) for r in rows])
        n=len(d); c=int(sum(d==1)); h=int(sum(d==-1))
        boots=d[rng.integers(0,n,size=(10000,n))].mean(axis=1)
        v={'n':n,'correct':sum(int(r['correct'][p]) for r in rows),
           'baseline_correct':sum(int(r['correct']['M']) for r in rows),
           'corrections':c,'harms':h,'net':c-h,'nonzero_questions':c+h,
           'difference_pp':float(100*d.mean()),'ci95_pp':(100*np.quantile(boots,[.025,.975])).tolist(),
           'mcnemar_p':float(stats.binomtest(c,c+h,.5).pvalue) if c+h else 1.}
        result[p]=v
    order=sorted(['CISC_V','T_P','W_P','A21_P'],key=lambda p:result[p]['mcnemar_p'])
    last=0
    for i,p in enumerate(order):
        last=max(last,min(1,(4-i)*result[p]['mcnemar_p']));result[p]['holm_p']=last
    return result

def sign_sensitivity(rows):
    groups=collections.defaultdict(list)
    for r in rows:
        if r['model'] in HOSTED:groups[r['id']].append(r)
    out={}
    funs={'H1':lambda r:int(r['correct']['M'] and not r['correct']['W64'])-int(r['correct']['M'] and not r['correct']['W4']),
          'H2':lambda r:r['correct']['G64']-r['correct']['W64']}
    for name,fun in funs.items():
        vals=[int(sum(fun(r) for r in g)) for g in groups.values()]
        nz=[v for v in vals if v];observed=abs(sum(nz))
        sums=[sum(s*abs(v) for s,v in zip(signs,nz)) for signs in itertools.product([-1,1],repeat=len(nz))]
        out[name]={'nonzero_questions':len(nz),'assignments':len(sums),
                   'two_sided_p':sum(abs(v)>=observed for v in sums)/len(sums)}
    last=0
    for i,name in enumerate(sorted(out,key=lambda k:out[k]['two_sided_p'])):
        last=max(last,min(1,(2-i)*out[name]['two_sided_p']));out[name]['holm_p']=last
    return out

def main():
    rows=read('data/evaluation_rows.json')
    original=read('data/replication_rows.json');extended=read('data/replication_extended_parser_rows.json')
    for rr in [rows,original,extended]:recompute_permutations(rr)
    initial={'cells':cells(rows,INITIAL),'commercial_pooled':pooled(rows,HOSTED),
             'all_models_sensitivity':pooled(rows,INITIAL)}
    complete=[r for r in rows if r['complete']]
    initial['complete_case_cells']=cells(complete,INITIAL)
    compare(initial,read('data/analysis.json'),'initial')
    print('Initial cells, cluster intervals, tests, controls and complete cases verified.',flush=True)
    extension={'cells':cells(original,ADDED),'extended_parser_sensitivity_cells':cells(extended,ADDED),
               'new_models_pooled_descriptive':pooled(original,ADDED),
               'six_models_pooled_descriptive':pooled(rows+original,INITIAL+ADDED),
               'extended_parser_sensitivity_pooled':pooled(extended,ADDED)}
    compare(extension,read('data/replication_analysis.json'),'extension')
    strict=pooled(strict_rows(rows),HOSTED)
    compare(strict,read('data/strict_parser_sensitivity.json')['commercial_pooled'],'strict')
    cisc=cisc_statistics(read('data/cisc/evaluation_rows.json'))
    compare(cisc,read('data/cisc/analysis.json')['contrasts'],'cisc')
    signs=sign_sensitivity(rows)
    compare(signs,read('data/additional_diagnostics.json')['sign_symmetry_sensitivity'],'signs')
    influence={m:pooled(rows,[z for z in HOSTED if z!=m])['primary']['H1_harm64_minus_harm4'] for m in HOSTED}
    for v in influence.values():v.pop('holm_p',None)
    sources=['evaluation_rows.json','replication_rows.json','replication_extended_parser_rows.json','cisc/evaluation_rows.json']
    out={'status':'verified','initial':initial,'extension':extension,'strict_parser_sensitivity':strict,
         'cisc':cisc,'sign_symmetry_sensitivity':signs,'leave_one_model_out_H1_descriptive':influence,
         'checked_leaf_values':len(checks),'absolute_comparison_tolerance':ATOL,
         'source_sha256':{f:hashlib.sha256((ROOT/'data'/f).read_bytes()).hexdigest() for f in sources},
         'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
         'monte_carlo':{'draws':10000,'initial_seed':2026102207,'cisc_seed':20261022,'generator':'PCG64','quantile':'linear'},
         'scope':'Fresh statistical estimates from stored per-question rows; archives used only after estimation for comparison. No original primary estimand changed. Leave-one-model-out is retrospective descriptive sensitivity; no new confirmatory family.'}
    (ROOT/'data/recomputed_statistics.json').write_text(json.dumps(out,indent=2)+'\n')
    print(f'Verified {len(checks)} statistical fields; wrote freshly recomputed estimates.',flush=True)

if __name__=='__main__':main()
