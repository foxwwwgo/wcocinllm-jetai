# Offline package copy; only input/output paths and repeatable output writing differ.
"""Post-completion independent decision replay using only Python standard library.

Does not import the collector, readout, gate, or analysis implementations. It reads
their completed records, recomputes deterministic decisions, and checks the
declared ranking and stop rule. It must run only after the development lock.
"""
from pathlib import Path
from collections import Counter
from fractions import Fraction
from datetime import datetime, timezone
import json, math

ROOT = Path(__file__).resolve().parents[1]
BETAS = [0, 1, 2, 4, 8, 16, 32, 64]
MODELS = ['deepseek', 'qwen', 'glm', 'open_qwen14b', 'granite', 'phi']
TASKS = ['gsm8k', 'commonsenseqa']
def read(p): return json.loads(p.read_text())

def decisions(row, beta, rank=False):
    pairs = [(a, s) for a, s in zip(row['answers'], row['scores']) if a is not None]
    counts = Counter(a for a, _ in pairs)
    if not counts: return None
    majority = max(counts, key=lambda c: counts[c])
    if beta == 0 or any(s is None for _, s in pairs): return majority
    scores = [s/100 for _, s in pairs]
    if rank:
        scores = [(sum(v < s for v in scores) + .5*(sum(v == s for v in scores)-1))/(len(scores)-1)
                  for s in scores] if len(scores) > 1 else [0.]
    totals = {c: 0. for c in counts}
    for (a, _), s in zip(pairs, scores): totals[a] += math.exp(beta*(s-max(scores)))
    return max(totals, key=lambda c: totals[c])

def gap(row):
    n = sorted(Counter(a for a in row['answers'] if a is not None).values(), reverse=True)
    return n[0]-(n[1] if len(n)>1 else 0) if n else 0

def gate(row, beta, h, eta):
    m, w = decisions(row, 0), decisions(row, beta)
    if m == w or gap(row) > h: return m
    def mean(c):
        z = [Fraction(str(s)) for a, s in zip(row['answers'], row['scores']) if a == c]
        return sum(z)/len(z)/100
    return w if mean(w)-mean(m) >= Fraction(str(eta)) else m

def verify_rows(rows, dev):
    for r in rows:
        m = decisions(r, 0)
        assert m == r['majority'] == r['predictions']['M']
        assert r['gap'] == gap(r)
        for beta in BETAS: assert decisions(r, beta) == r['predictions']['W'+str(beta)]
        assert (m if gap(r)>=2 else decisions(r,64)) == r['predictions']['G64']
        assert decisions(r, dev[r['task']+':'+r['model']]['beta']) == r['predictions']['DEV']
        for beta in [4,64]: assert decisions(r,beta,rank=True) == r['predictions']['RANK'+str(beta)]
        for key, answer in r['predictions'].items():
            assert r['correct'][key] == int(answer == r['gold'])
            assert answer is None or answer in r['answers']
    return len(rows)

def cell_summary(rows):
    result = {}
    for task in TASKS:
        for model in MODELS:
            rr = [r for r in rows if r['task']==task and r['model']==model]
            if not rr: continue
            measures = {}
            for policy in rr[0]['predictions']:
                c = sum(not r['correct']['M'] and r['correct'][policy] for r in rr)
                h = sum(r['correct']['M'] and not r['correct'][policy] for r in rr)
                assert c-h == sum(r['correct'][policy]-r['correct']['M'] for r in rr)
                measures[policy] = {'correct':sum(r['correct'][policy] for r in rr),'recoveries':c,'harms':h}
            result[task+':'+model] = {'n':len(rr), 'seats':7*len(rr),
                'valid_seats':sum(sum(a is not None for a in r['answers']) for r in rr),
                'empty_pools':sum(not any(a is not None for a in r['answers']) for r in rr),
                'score_failure_pools':sum(any(a is not None and s is None for a,s in zip(r['answers'],r['scores'])) for r in rr),
                'recoverable_questions':sum(not r['correct']['M'] and r['gold'] in r['answers'] for r in rr),
                'policies':measures}
    return result

def main():
    lock = read(ROOT/'data/successor_development_lock.json')
    rows = read(ROOT/'data/development_rows.json')
    assert len(rows)==1200 and len({r['id'] for r in rows})==200
    verify_rows(rows, lock['DEV'])
    candidates = []
    for beta in [4,16,64]:
        for h in [1,2]:
            for eta in [.05,.10]:
                key = f'b{beta}-h{h}-d{eta:.2f}'
                answers = [gate(r,beta,h,eta) for r in rows]
                assert all(a==r['predictions'][key] for r,a in zip(rows,answers))
                net = sum(int(a==r['gold'])-int(r['correct']['M']) for r,a in zip(rows,answers))
                base = [r for r in rows if not r['correct']['M'] and r['correct']['W'+str(beta)]]
                retained = sum(not r['correct']['M'] and a==r['gold'] for r,a in zip(rows,answers))
                for r,a in zip(rows,answers):
                    assert a in [r['majority'],r['predictions']['W'+str(beta)]]
                    if not r['correct']['M'] and a==r['gold']: assert r['correct']['W'+str(beta)]
                    if r['correct']['M'] and a!=r['gold']: assert not r['correct']['W'+str(beta)]
                rate = retained/len(base) if base else None
                tasknet = {t:sum(int(a==r['gold'])-int(r['correct']['M']) for r,a in zip(rows,answers) if r['task']==t) for t in TASKS}
                passes = net>=6 and min(tasknet.values())>=0 and len({r['id'] for r in base})>=10 and rate is not None and rate>=.75
                candidates.append({'key':key,'beta':beta,'h':h,'eta':eta,'correct':sum(a==r['gold'] for r,a in zip(rows,answers)),
                    'net':net,'overrides':sum(a!=r['majority'] for r,a in zip(rows,answers)), 'retained':retained,
                    'opportunities':len(base),'unique_opportunity_questions':len({r['id'] for r in base}),'retention':rate,'passes':passes})
    candidates.sort(key=lambda c:(-c['correct'],c['overrides'],c['beta'],c['h'],-c['eta']))
    assert [c['key'] for c in candidates]==[c['key'] for c in lock['all_candidates_ranked']]
    for computed, recorded in zip(candidates,lock['all_candidates_ranked']):
        for a,b in [('correct','correct'),('net','net_count_vs_M'),('retained','retained_recoveries'),('opportunities','base_recoveries'),('retention','retention'),('passes','passes')]: assert computed[a]==recorded[b]
    expected = 'PROCEED_TO_INDEPENDENT_PILOT' if candidates[0]['passes'] else 'STOP_CANDIDATE_FAMILY_DEVELOPMENT_SCREEN_FAILED'
    assert lock['status']==expected
    repdev = read(ROOT/'data/replication_development_lock.json')
    rep = read(ROOT/'data/replication_rows.json')
    ext = read(ROOT/'data/replication_extended_parser_rows.json')
    verify_rows(rep,repdev['cells']);verify_rows(ext,repdev['cells_extended_parser_sensitivity'])
    output = {'utc':datetime.now(timezone.utc).isoformat(),'status':'all deterministic readouts, subsets, counts, selection and stop decision verified',
              'decision':expected,'development':cell_summary(rows),'replication_original_parser':cell_summary(rep),
              'replication_compatibility_parser':cell_summary(ext),'all_candidates':candidates,
              'independent_team':False,'implementation':'Separate standard-library code path; no import of scientific decision code.'}
    p=ROOT/'data/reproduced_decision_replay.json'
    with p.open('w') as f: json.dump(output,f,indent=2)
    print(output['status']);print(expected)

if __name__=='__main__': main()
