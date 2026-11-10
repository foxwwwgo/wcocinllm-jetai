import os
# -*- coding: utf-8 -*-
import json, math, random, collections
BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ledger=json.load(open(f"{BASE}/data/certificate_pool_ledger.json"))
def ls(tag):
    d={}
    for l in open(f"{BASE}/out/scores_{tag}.jsonl"):
        r=json.loads(l)
        if r["score"] is not None: d[r["key"]]=r["score"]
    return d
PHI,QW=ls("phi"),ls("qwen14b")
def self_s(r):
    a=collections.defaultdict(list)
    for x,s in zip(r["answers"],r["scores"]):
        if x is None: continue
        a[x].append(s)
    return {k:sum(v)/len(v)/100.0 for k,v in a.items()}
def ext_s(r,t):
    o={}
    for c in r["counts"]:
        v=t.get(f"{r['id']}|{r['model']}|{c}")
        if v is not None: o[c]=v/100.0
    return o
SIG={"self":self_s,"phi3":lambda r:ext_s(r,PHI),"qwen14b":lambda r:ext_s(r,QW)}
def qhat(vals,a):
    v=sorted(vals);n=len(v);k=math.ceil((n+1)*(1-a))
    return v[k-1] if k<=n else None
qs=sorted({r["id"] for r in ledger})
def run(fn,alpha,K=5,seed=20261108):
    rng=random.Random(seed); qsx=qs[:]; rng.shuffle(qsx)
    folds=[set(qsx[i::K]) for i in range(K)]
    C=H=W=0; acc=0; cn=cd=0; qs_seen=set()
    for f in folds:
        cal=[r for r in ledger if r["id"] not in f]; test=[r for r in ledger if r["id"] in f]
        vals=[-(fn(r)[r["gold"]]-fn(r)[r["majority"]]) for r in cal
              if r["gold"] in fn(r) and r["majority"] in fn(r)]
        q=qhat(vals,alpha)
        if q is None: continue
        qs_seen.add(round(q,4))
        for r in test:
            s=fn(r)
            if r["gold"] in s and r["majority"] in s:
                cn += (-(s[r["gold"]]-s[r["majority"]])<=q); cd+=1
            if r["class_count"]<2 or r["majority"] not in s: continue
            alt=[(v,k) for k,v in s.items() if k!=r["majority"]]
            if not alt: continue
            smax,kmax=max(alt)
            if (-(smax-s[r["majority"]])<=q):
                acc+=1
                if kmax==r["gold"]: C+=1
                elif r["majority"]==r["gold"]: H+=1
                else: W+=1
    return dict(cov=cn/cd if cd else None,acc=acc,C=C,H=H,W=W,eta=(C/H if H else None),q=sorted(qs_seen))
for name,fn in SIG.items():
    for a in (0.05,0.10,0.20):
        r=run(fn,a)
        print(f"{name:8s} a={a:.2f} cov={r['cov']:.3f} acc={r['acc']:3d} C={r['C']:2d} H={r['H']:2d} W={r['W']:2d} eta={r['eta'] if r['eta'] is None else round(r['eta'],2)} q={r['q']}")
