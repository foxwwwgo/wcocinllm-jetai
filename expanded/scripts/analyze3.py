import os
BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# -*- coding: utf-8 -*-
"""Conformal filtering with the score difference of the plurality cluster as the nonconformity score."""
import json, math, random, collections
exec(open(f"{BASE}/scripts/analyze2.py").read().split("def qhat")[0])
qs = sorted({r["id"] for r in ledger})

def run_margin(name, fn, alpha, K=5, seed=20261108):
    rng = random.Random(seed); qsx = qs[:]; rng.shuffle(qsx)
    folds = [set(qsx[i::K]) for i in range(K)]
    C=H=W=0; acc=0; cov_n=cov_d=0
    for f in folds:
        cal=[r for r in ledger if r["id"] not in f]; test=[r for r in ledger if r["id"] in f]
        vals=[]
        for r in cal:
            s=fn(r)
            if r["gold"] in s and r["majority"] in s:
                vals.append(-(s[r["gold"]]-s[r["majority"]]))
        q=qhat(vals,alpha)
        if q is None: continue
        for r in test:
            s=fn(r)
            if r["gold"] in s and r["majority"] in s:
                cov_n += (-(s[r["gold"]]-s[r["majority"]]) <= q); cov_d += 1
            if r["class_count"]<2 or r["majority"] not in s: continue
            alt=[(v,k) for k,v in s.items() if k!=r["majority"]]
            if not alt: continue
            smax,kmax=max(alt); smaj=s[r["majority"]]
            if (-(smax-smaj) <= q):
                acc+=1
                if kmax==r["gold"]: C+=1
                elif r["majority"]==r["gold"]: H+=1
                else: W+=1
    return dict(cov=cov_n/cov_d if cov_d else None, acc=acc,C=C,H=H,W=W,
                eta=(C/H if H else None))

print("margin-conformal: signal  alpha  cov  acc  C  H  W  eta")
for name,fn in SIG.items():
    for a in (0.05,0.10,0.20):
        r=run_margin(name,fn,a)
        print(f"{name:8s} {a:.2f}  {r['cov']:.3f}  {r['acc']:3d}  {r['C']:2d} {r['H']:2d} {r['W']:2d}  {(r['eta'] if r['eta'] else float('nan')):.2f}")

def qhat(vals, a):
    v = sorted(vals); n = len(v); k = math.ceil((n+1)*(1-a))
    return v[k-1] if k <= n else None
