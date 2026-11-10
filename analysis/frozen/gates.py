"""Bounded successor family. No labels enter a deployed decision.

Candidate design only: no new development/pilot/validation data collected.
Scores are normalized to [0,1]. Fixed tie rule: first occurrence.
"""
from collections import Counter
from itertools import product
from math import exp, isfinite

CANDIDATES = tuple(product((4, 16, 64), (1, 2), (.05, .10)))

def majority(answers):
    counts = Counter(a for a in answers if a is not None)
    return max(counts, key=counts.get) if counts else None

def weighted(answers, scores, concentration):
    baseline = majority(answers)
    valid = [(a, s) for a, s in zip(answers, scores) if a is not None]
    if not valid or concentration == 0:
        return baseline
    if any(s is None or not isfinite(s) or not 0 <= s <= 1 for _, s in valid):
        return baseline
    peak = max(s for _, s in valid)
    sums = dict.fromkeys((a for a, _ in valid), 0.)
    for a, s in valid:
        sums[a] += exp(concentration * (s-peak))
    return max(sums, key=sums.get)

def gate(answers, scores, concentration, maximum_gap, minimum_confidence_gap):
    if len(answers) != len(scores):
        raise ValueError('Answers and scores must preserve the same seat positions')
    baseline = majority(answers)
    chosen = weighted(answers, scores, concentration)
    if chosen == baseline:
        return baseline
    counts = Counter(a for a in answers if a is not None)
    sizes = sorted(counts.values(), reverse=True)
    gap = sizes[0] - (sizes[1] if len(sizes)>1 else 0)
    if gap > maximum_gap:
        return baseline
    def mean_score(c):
        values = [s for a, s in zip(answers, scores) if a == c]
        return sum(values)/len(values)
    if mean_score(chosen)-mean_score(baseline) < minimum_confidence_gap:
        return baseline
    return chosen

if __name__ == '__main__':
    # Boundary/invariant checks, not a search on benchmark outcomes.
    assert len(CANDIDATES) == 12
    for parameters in CANDIDATES:
        assert gate([], [], *parameters) is None
        assert gate(['A']*7, [.7]*7, *parameters) == 'A'
        assert gate(['A']*4+['B']*3, [.5]*7, *parameters) == 'A'
        assert gate(['A']*4+['B']*3, [.2]*4+[1,None,1], *parameters) == 'A'
    assert gate(['A']*4+['B']*3, [.2]*4+[.9]*3, 64, 1, .10) == 'B'
    assert gate(['A']*5+['B']*2, [.2]*5+[.9]*2, 64, 2, .10) == 'A'
    print('51 boundary and policy-invariant assertions passed; no benchmark fitting')
