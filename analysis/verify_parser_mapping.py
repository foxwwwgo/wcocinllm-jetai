"""Verify the three parser names on complete stored replies, without gold labels."""
from pathlib import Path
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from frozen_parsers import parse_strict, parse_v71, parse

ROOT = Path(__file__).resolve().parents[1]
PARSERS = {'strict': parse_strict, 'evaluation': parse_v71, 'compatibility': parse}


def main():
    source = ROOT / 'data/parser_outputs.jsonl.gz'
    rows = [json.loads(line) for line in gzip.decompress(source.read_bytes()).decode().splitlines()]
    totals = Counter()
    cells = defaultdict(Counter)
    for row in rows:
        assert hashlib.sha256(row['raw_reply'].encode()).hexdigest() == row['raw_reply_sha256']
        job = {'kind': 'generation', 'task': row['task']}
        answers = {name: fn(job, row) for name, fn in PARSERS.items()}
        assert answers['evaluation'] == row['original_answer']
        assert answers['compatibility'] == row['extended_answer']
        if answers['strict'] is not None:
            assert answers['strict'] == answers['evaluation']
        if answers['evaluation'] is not None:
            assert answers['evaluation'] == answers['compatibility']
        for name, value in answers.items():
            totals[name] += value is not None
            cells[row['task'] + ':' + row['model']][name] += value is not None
    assert len(rows) == 8400
    assert dict(totals) == {'strict': 501, 'evaluation': 877, 'compatibility': 2842}
    result = {
        'status': 'verified', 'rows': len(rows),
        'functions': {name: fn.__name__ for name, fn in PARSERS.items()},
        'accepted_seats': dict(totals),
        'by_task_model': {key: dict(cells[key]) for key in sorted(cells)},
        'sha256': {
            'data/parser_outputs.jsonl.gz': hashlib.sha256(source.read_bytes()).hexdigest(),
            'analysis/frozen_parsers.py': hashlib.sha256((ROOT/'analysis/frozen_parsers.py').read_bytes()).hexdigest(),
        },
        'historical_aliases': {
            'original_answer': 'evaluation', 'original_answers': 'evaluation',
            'accepted_original': 'evaluation', 'added_original': 'evaluation',
            'replication_original': 'evaluation', 'replication_original_parser': 'evaluation',
            'strict_answers': 'strict', 'extended_answer': 'compatibility',
            'extended_answers': 'compatibility',
        },
        'scope': 'Parser-name verification on unchanged recorded outputs; no new generation, scoring, policy selection or accuracy analysis.'
    }
    (ROOT/'data/parser_mapping.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
