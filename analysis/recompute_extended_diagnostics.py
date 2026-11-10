"""Descriptive certificate, scale and usable-pool analyses of existing records.

These analyses do not change the frozen design or add confirmatory hypotheses.
The nonconstant-score case rule is fixed in this script before inspecting the
chosen case. Historical examples and their original selection rule are retained.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
HOSTED = ['deepseek', 'qwen', 'glm']
BETAS = [1, 2, 4, 8, 16, 32, 64]
SALT = 'ida-nonconstant-preservation-20261024'


def read(name):
    return json.loads((ROOT / name).read_text())


def write_csv(name, rows):
    with (ROOT / 'tables' / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def geometry(r):
    counts = Counter(a for a in r['answers'] if a is not None)
    scores = [s / 100 for a, s in zip(r['answers'], r['scores'])
              if a is not None and s is not None]
    sizes = sorted(counts.values(), reverse=True)
    span = max(scores) - min(scores) if scores else None
    return counts, sizes, span


def certificate(r, beta):
    _, sizes, span = geometry(r)
    return bool(r['complete'] and sizes and
                (len(sizes) == 1 or sizes[0] > sizes[1]
                 and beta * span < math.log(sizes[0] / sizes[1])))


def select(r, beta, rank=False):
    pairs = [(a, s) for a, s in zip(r['answers'], r['scores']) if a is not None]
    counts = Counter(a for a, _ in pairs)
    if not counts:
        return None
    majority = max(counts, key=counts.get)
    if any(s is None for _, s in pairs):
        return majority
    scores = [s / 100 for _, s in pairs]
    if rank:
        ordered = sorted(scores)
        scores = [sum(i for i, v in enumerate(ordered) if v == s)
                  / ordered.count(s) / (len(scores) - 1)
                  if len(scores) > 1 else 0.0 for s in scores]
    top = max(scores)
    weights = [math.exp(beta * (s - top)) for s in scores]
    return max(counts, key=lambda c: sum(w for (a, _), w in zip(pairs, weights) if a == c))


def main():
    initial = read('data/evaluation_rows.json')
    primary = [r for r in initial if r['model'] in HOSTED]
    # Reconstruct scale controls independently from the stored decisions.
    comparisons = 0
    for r in initial:
        for beta in BETAS:
            assert select(r, beta) == r['predictions'][f'W{beta}']
            comparisons += 1
        for beta in [4, 64]:
            assert select(r, beta, rank=True) == r['predictions'][f'RANK{beta}']
            comparisons += 1
    multiclass = [r for r in primary if len(geometry(r)[0]) > 1]
    grid = []
    for beta in BETAS:
        for constant in [True, False]:
            rows = [r for r in multiclass if (geometry(r)[2] == 0) == constant]
            cert = [r for r in rows if certificate(r, beta)]
            assert all(r['predictions'][f'W{beta}'] == r['majority'] for r in cert)
            grid.append(dict(beta=beta, score_stratum='constant' if constant else 'nonconstant',
                             records=len(rows), certified=len(cert),
                             overrides=sum(r['predictions'][f'W{beta}'] != r['majority'] for r in rows)))
    eligible = [r for r in multiclass if geometry(r)[2] > 0 and certificate(r, 4)]
    key = lambda r: hashlib.sha256((SALT + ':' + r['id'] + ':' + r['model']).encode()).hexdigest()
    chosen = min(eligible, key=key)
    counts, sizes, span = geometry(chosen)
    top = max(s for a, s in zip(chosen['answers'], chosen['scores']) if a is not None)
    example = {k: chosen[k] for k in ['id', 'model', 'task', 'answers', 'scores', 'gold', 'majority']}
    example.update(counts=dict(counts), span=span, beta=4, eligible_records=len(eligible),
                   hash=key(chosen), beta_times_span=4 * span, log_vote_ratio=math.log(sizes[0] / sizes[1]),
                   class_weights={str(b): {c: sum(math.exp(b * (s - top) / 100)
                       for a, s in zip(chosen['answers'], chosen['scores']) if a == c) for c in counts}
                       for b in [4, 64]}, W4=chosen['predictions']['W4'], W64=chosen['predictions']['W64'])
    readouts = []
    for policy in ['W4', 'W64', 'RANK4', 'RANK64']:
        c = sum(r['majority'] != r['gold'] and r['predictions'][policy] == r['gold'] for r in primary)
        h = sum(r['majority'] == r['gold'] and r['predictions'][policy] != r['gold'] for r in primary)
        readouts.append(dict(policy=policy, records=len(primary), corrections=c, harms=h, net=c-h))
    identities = [dict(left=a, right=b, records=len(initial),
                       identical_decisions=sum(r['predictions'][a] == r['predictions'][b] for r in initial))
                  for a, b in [('W32', 'W64'), ('RANK4', 'W64')]]
    coverage = []
    inputs = [('evaluation_rows.json', 'initial', 'evaluation'),
              ('replication_rows.json', 'added', 'evaluation'),
              ('replication_extended_parser_rows.json', 'added', 'compatibility')]
    for filename, block, parser in inputs:
        rows = read('data/' + filename)
        for model in sorted({r['model'] for r in rows}):
            for task in ['all', 'gsm8k', 'commonsenseqa']:
                rr = [r for r in rows if r['model'] == model and (task == 'all' or r['task'] == task)]
                sizes = [len(geometry(r)[0]) for r in rr]
                coverage.append(dict(block=block, parser=parser, model=model, task=task, records=len(rr),
                    empty=sum(s == 0 for s in sizes), single_class=sum(s == 1 for s in sizes),
                    multiple_class=sum(s > 1 for s in sizes),
                    H1_net_harms=sum(int(r['majority'] == r['gold'] and r['predictions']['W64'] != r['gold'])
                                   - int(r['majority'] == r['gold'] and r['predictions']['W4'] != r['gold']) for r in rr),
                    H2_net_correct=sum(int(r['predictions']['G64'] == r['gold'])
                                     - int(r['predictions']['W64'] == r['gold']) for r in rr)))
    stats = read('data/recomputed_statistics.json')
    retention = []
    for name, r in [('primary', stats['initial']['commercial_pooled']),
                    ('initial_four', stats['initial']['all_models_sensitivity']),
                    ('added_evaluation', stats['extension']['new_models_pooled_descriptive']),
                    ('added_compatibility', stats['extension']['extended_parser_sensitivity_pooled']),
                    ('six_evaluation', stats['extension']['six_models_pooled_descriptive']),
                    ('strict_primary', stats['strict_parser_sensitivity'])]:
        ci = r['retention_ci95']
        retention.append(dict(population=name, estimate=r['correction_retention'],
                              ci95_low=ci[0], ci95_high=ci[1],
                              boundary_degenerate=bool(ci[0] == ci[1] and ci[0] in [0, 1]),
                              undefined_resamples=r['retention_bootstrap_undefined']))
    result = dict(status='Retrospective descriptive analyses; no new confirmatory hypotheses or model calls',
        primary_records=len(primary), multiclass_records=len(multiclass),
        independent_decision_comparisons=comparisons, certificate_grid=grid,
        preservation_example=example, selection_salt=SALT,
        selection_rule='Minimum SHA256(salt:id:model) among primary multiclass, nonconstant-score pools certified at beta=4; no effect-size ordering. Historical correction and harm examples are unchanged.',
        scale_readouts=readouts, observed_identities=identities,
        identity_scope='Observed records at the specified grid points only; not algebraic equivalence or a statement about continuous parameter intervals.',
        pool_composition=coverage, retention_interpretation=retention,
        retention_note='Empirical question-cluster bootstrap intervals condition on observed events. All-retained samples can produce [1,1]; this boundary-degenerate interval is descriptive and does not establish precise population retention or a guarantee. Undefined zero-opportunity resamples are reported separately. Existing frozen estimates are preserved.',
        denominator_note='All scheduled questions remain in accuracy denominators. Multiclass counts are descriptive strata, not a replacement primary population.',
        source_sha256={f: hashlib.sha256((ROOT/'data'/f).read_bytes()).hexdigest()
                       for f in [x[0] for x in inputs] + ['recomputed_statistics.json']})
    (ROOT/'data/extended_diagnostics.json').write_text(json.dumps(result, indent=2) + '\n')
    write_csv('certificate_score_strata.csv', grid)
    write_csv('rank_scale_readouts.csv', readouts)
    write_csv('pool_composition.csv', coverage)
    write_csv('retention_interpretation.csv', retention)
    print(json.dumps({'preservation_example': example, 'scale_readouts': readouts,
                      'comparisons': comparisons, 'coverage_rows': len(coverage)}, indent=2))


if __name__ == '__main__':
    main()
