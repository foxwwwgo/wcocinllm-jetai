import os
# -*- coding: utf-8 -*-
"""Conformal filtering: across 209 competition pools, 5-fold cross-fit by question, compare three signals."""
import json, math, random, collections

BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ledger = json.load(open(f"{BASE}/data/certificate_pool_ledger.json"))

def load_scores(tag):
    d = {}
    for l in open(f"{BASE}/out/scores_{tag}.jsonl"):
        r = json.loads(l)
        if r["score"] is not None:
            d[r["key"]] = r["score"]
    return d
PHI, QW = load_scores("phi"), load_scores("qwen14b")

def self_s(r):
    a = collections.defaultdict(list)
    for x, s in zip(r["answers"], r["scores"]):
        if x is None: continue
        a[x].append(s)
    return {k: sum(v)/len(v)/100.0 for k, v in a.items()}

def ext_s(r, t):
    o = {}
    for c in r["counts"]:
        v = t.get(f"{r['id']}|{r['model']}|{c}")
        if v is not None: o[c] = v/100.0
    return o

SIG = {"self": self_s, "phi3": lambda r: ext_s(r, PHI), "qwen14b": lambda r: ext_s(r, QW)}

def qhat(vals, a):
    v = sorted(vals); n = len(v); k = math.ceil((n+1)*(1-a))
    return v[k-1] if k <= n else None

qs = sorted({r["id"] for r in ledger})

def run(name, fn, alpha, rule, K=5, seed=20261108):
    rng = random.Random(seed); qsx = qs[:]; rng.shuffle(qsx)
    folds = [set(qsx[i::K]) for i in range(K)]
    C = H = W = 0; acc = 0; cov_n = cov_d = 0
    for f in folds:
        cal = [r for r in ledger if r["id"] not in f]
        test = [r for r in ledger if r["id"] in f]
        vals = []
        for r in cal:
            s = fn(r)
            if r["gold"] in s: vals.append(-s[r["gold"]])
        q = qhat(vals, alpha)
        if q is None: continue
        for r in test:
            s = fn(r)
            if r["gold"] in s:
                cov_n += (-s[r["gold"]] <= q); cov_d += 1
            if r["class_count"] < 2 or r["majority"] not in s: continue
            alt = [(v,k) for k,v in s.items() if k != r["majority"]]
            if not alt: continue
            smax, kmax = max(alt); smaj = s[r["majority"]]
            ok = (-smax <= q) and (rule == "one" or (-smaj > q))
            if not ok: continue
            acc += 1
            if kmax == r["gold"]: C += 1
            elif r["majority"] == r["gold"]: H += 1
            else: W += 1
    return dict(cov=cov_n/cov_d if cov_d else None, acc=acc, C=C, H=H, W=W,
                eta=(C/H if H else None))

print("signal  rule   alpha  cov    accepted  C   H   W   eta")
for name, fn in SIG.items():
    for rule in ("one", "two"):
        for a in (0.05, 0.10, 0.20):
            r = run(name, fn, a, rule)
            e = "inf" if r["eta"] is None else f'{r["eta"]:.2f}'
            print(f"{name:8s} {rule:4s} {a:.2f}  {r['cov']:.3f}  {r['acc']:3d}      {r['C']:2d}  {r['H']:2d}  {r['W']:2d}  {e}")
