import os
# -*- coding: utf-8 -*-
import json, math, random, collections
BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ledger=json.load(open(f"{BASE}/data/certificate_pool_ledger.json"))
def ls(tag):
    d={}
    for l in open(f"{BASE}/out/scores_{tag}.jsonl"):
        r=json.loads(l)
        if r["score"] is not None: d[r["key"]]=r["score"]/100.0
    return d
QW,PHI=ls("qwen14b"),ls("phi")
def selfs(r):
    a=collections.defaultdict(list)
    for x,s in zip(r["answers"],r["scores"]):
        if x is None: continue
        a[x].append(s)
    return {k:sum(v)/len(v)/100.0 for k,v in a.items()}
def ext(r,t): return {c:t[f"{r['id']}|{r['model']}|{c}"] for c in r["counts"] if f"{r['id']}|{r['model']}|{c}" in t}
SIG={"self":selfs,"phi":lambda r:ext(r,PHI),"qwen":lambda r:ext(r,QW)}
def qh(v,a):
    v=sorted(v);n=len(v);k=math.ceil((n+1)*(1-a)); return v[k-1] if k<=n else None
qs=sorted({r["id"] for r in ledger})
for name,fn in SIG.items():
    for a in (0.05,0.10,0.20):
        rng=random.Random(20261108); q=qs[:]; rng.shuffle(q); folds=[set(q[i::5]) for i in range(5)]
        C=H=W=0;acc=0
        for f in folds:
            cal=[r for r in ledger if r["id"] not in f]; test=[r for r in ledger if r["id"] in f]
            vals=[-fn(r)[r["gold"]] for r in cal if r["gold"] in fn(r)]
            t=qh(vals,a)
            if t is None: continue
            for r in test:
                s=fn(r)
                if r["class_count"]<2 or r["majority"] not in s: continue
                alt=[(v,k) for k,v in s.items() if k!=r["majority"]]
                if not alt: continue
                smax,kmax=max(alt)
                if smax>s[r["majority"]] and -smax<=t:
                    acc+=1
                    if kmax==r["gold"]: C+=1
                    elif r["majority"]==r["gold"]: H+=1
                    else: W+=1
        print(f"{name:5s} a={a:.2f} acc={acc:3d} C={C:2d} H={H:2d} W={W:2d} eta={(C/H if H else float('nan')):.2f}")
