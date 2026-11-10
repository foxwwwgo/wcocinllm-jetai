"""Export manuscript figure data from recomputed results; no fitting or collection."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'figure_data'
MODELS = {'deepseek': 'DeepSeek-V4.1-Flash', 'qwen': 'qwen3.8-max-0902',
          'glm': 'glm-5.2', 'open_qwen14b': 'Qwen2.5-14B-Instruct'}
TASKS = ['gsm8k', 'commonsenseqa']
BETAS = [0, 1, 2, 4, 8, 16, 32, 64]

def write(name, rows):
    with (DEST / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    DEST.mkdir(exist_ok=True)
    analysis = json.loads((ROOT / 'data/recomputed_statistics.json').read_text())['initial']
    records = json.loads((ROOT / 'data/evaluation_rows.json').read_text())
    counts = []
    for policy in ['W4', 'W64', 'G64']:
        rr = [r for r in records if r['model'] in list(MODELS)[:3]]
        c = sum(not r['correct']['M'] and r['correct'][policy] for r in rr)
        h = sum(r['correct']['M'] and not r['correct'][policy] for r in rr)
        assert len(rr) == 1800
        counts.append(dict(policy=policy, corrections=c, harms=h, net=c-h,
                           scheduled_model_question_records=len(rr)))
    write('figure_2a_recovery_harm.csv', counts)
    primary = analysis['commercial_pooled']
    metrics = [('H1_harm_W64_minus_W4', primary['primary']['H1_harm64_minus_harm4']),
               ('H2_accuracy_G64_minus_W64', primary['primary']['H2_accuracy_guard64_minus_64']),
               ('accuracy_G64_minus_M', primary['policies_vs_M']['G64'])]
    write('figure_2b_effects.csv', [dict(contrast=key, effect_pp=v['effect_pp'],
        ci95_lower_pp=v['ci95_pp'][0], ci95_upper_pp=v['ci95_pp'][1],
        independent_question_clusters=600, interval_source='recomputed question-cluster bootstrap')
        for key, v in metrics])
    concentration = []
    opportunity = []
    for task in TASKS:
        for model, label in MODELS.items():
            rr = [r for r in records if r['task'] == task and r['model'] == model]
            assert len(rr) == 300
            for beta in BETAS:
                policy = f'W{beta}'
                summary = analysis['cells'][f'{task}:{model}']['policies'][policy]
                net = sum(r['correct'][policy]-r['correct']['M'] for r in rr)
                harm = sum(r['correct']['M'] and not r['correct'][policy] for r in rr)
                assert abs(100*net/300-summary['difference_from_M_pp']) < 1e-10
                assert harm == summary['harms']
                concentration.append(dict(task=task, model=model, model_name=label,
                    beta=beta, scheduled_questions=300, accuracy_change_pp=100*net/300,
                    harms=harm, harm_percent=100*harm/300))
            present = sum(not r['correct']['M'] and r['gold'] in r['answers'] for r in rr)
            absent = sum(not r['correct']['M'] and r['gold'] not in r['answers'] for r in rr)
            opportunity.append(dict(task=task, model=model, model_name=label,
                scheduled_questions=300, majority_errors=present+absent,
                correct_class_present=present, correct_class_absent=absent))
    write('figure_3_concentration.csv', concentration)
    write('figure_4_opportunity.csv', opportunity)
    quality=json.loads((ROOT/'data/confidence_quality.json').read_text())['cells']
    score_rows=[];reliability_rows=[]
    for task in TASKS:
        for model,label in MODELS.items():
            v=quality[f'initial_evaluation:{model}:{task}']
            score_rows.append(dict(task=task,model=model,model_name=label,wqd=v['wqd'],eligible_questions=v['wqd_questions'],ci95_low=v['wqd_ci95'][0],ci95_high=v['wqd_ci95'][1],brier=v['brier'],brier_questions=v['brier_questions']))
            reliability_rows.extend(dict(task=task,model=model,model_name=label,**b) for b in v['reliability'])
    write('figure_4_score_quality.csv',score_rows)
    write('figure_4_reliability.csv',reliability_rows)
    (DEST/'figure_5_six_model_recovery_harm.csv').write_bytes(
        (ROOT/'tables/six_model_recovery_harm.csv').read_bytes())
    (DEST/'figure_5_pool_composition.csv').write_bytes(
        (ROOT/'tables/pool_composition.csv').read_bytes())
    paths = ['data/recomputed_statistics.json', 'data/evaluation_rows.json',
             'data/replication_rows.json', 'data/replication_extended_parser_rows.json',
             'tables/six_model_recovery_harm.csv', 'tables/pool_composition.csv','data/confidence_quality.json']
    manifest = {'version': '10.13', 'source_sha256': {
        name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths},
        'csv_rows': {p.name: sum(1 for _ in csv.DictReader(p.open()))
                     for p in sorted(DEST.glob('*.csv'))},
        'verification': 'Figure 2a, 3 and 4 point values recomputed from unchanged parsed rows; '
                        'Figure 2b interval estimates recomputed from question rows; '
                        'Figure 4 score diagnostics use question-level bootstrap and fixed calibration bins; '
                        'Figure 5 validated by create_six_model_overview.py; companion pool-composition strata recomputed from all scheduled records.'}
    (DEST/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest['csv_rows']))

if __name__ == '__main__':
    main()
