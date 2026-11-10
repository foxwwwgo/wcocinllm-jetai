"""Exact comparison of the declared decimal confidence-gap thresholds.
Inputs retain original parsed 0--100 scores. No benchmark label enters this rule.
The candidate family, weighted winner, count-gap rule and fallbacks are unchanged.
"""
from fractions import Fraction
from collections import Counter
from gates import majority,weighted,CANDIDATES
def gate100(answers,scores,concentration,maximum_gap,minimum_confidence_gap):
 if len(answers)!=len(scores):raise ValueError('Seat positions must match')
 baseline=majority(answers)
 chosen=weighted(answers,[None if s is None else s/100 for s in scores],concentration)
 if chosen==baseline:return baseline
 sizes=sorted(Counter(a for a in answers if a is not None).values(),reverse=True)
 if sizes[0]-(sizes[1] if len(sizes)>1 else 0)>maximum_gap:return baseline
 def mean(c):
  values=[Fraction(str(s)) for a,s in zip(answers,scores) if a==c]
  return sum(values)/len(values)/100
 return chosen if mean(chosen)-mean(baseline)>=Fraction(str(minimum_confidence_gap)) else baseline
if __name__=='__main__':
 for lo,hi,delta in [(90,95,.05),(65,75,.10),(29.1,34.1,.05)]:
  assert gate100(['A']*4+['B']*3,[lo]*4+[hi]*3,64,1,delta)=='B'
  assert gate100(['A']*4+['B']*3,[lo]*4+[hi-.0001]*3,64,1,delta)=='A'
 print('Six exact-threshold boundary checks passed; no benchmark outcomes read.')
