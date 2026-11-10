import os
# -*- coding: utf-8 -*-
import json, random, collections, statistics
BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ledger=json.load(open(f"{BASE}/data/certificate_pool_ledger.json"))
def ls(tag):
    d={}
    try:
        for l in open(f"{BASE}/out/scores_{tag}.jsonl"):
            r=json.loads(l)
            if r["score"] is not None: d[r["key"]]=r["score"]
    except FileNotFoundError: return d
    return d
PHI,QW=ls("phi"),ls("qwen14b")
def self_s(r):
    a=collections.defaultdict(list)
    for x,s in zip(r["answers"],r["scores"]):
        if x is None: continue
        a[x].append(s)
    return {k:sum(v)/len(v)/100.0 for k,v in a.items()}
def ext_s(r,t):
    return {c:t[f"{r['id']}|{r['model']}|{c}"]/100.0 for c in r["counts"]
            if f"{r['id']}|{r['model']}|{c}" in t}
SIG={"self":self_s,"phi3":lambda r:ext_s(r,PHI),"qwen14b":lambda r:ext_s(r,QW)}
qs=sorted({r["id"] for r in ledger})
def switches(fn):
    out=[]
    for r in ledger:
        if r["class_count"]<2: continue
        s=fn(r)
        if r["majority"] not in s: continue
        alt=[(v,k) for k,v in s.items() if k!=r["majority"]]
        if not alt: continue
        smax,kmax=max(alt)
        if smax>s[r["majority"]]:
            out.append((r["id"], 1 if kmax==r["gold"] else (-1 if r["majority"]==r["gold"] else 0)))
    return out
print("signal   switches  C   H   W   eta   95%CI(eta by question cluster)")
for name,fn in SIG.items():
    sw=switches(fn)
    C=sum(1 for _,y in sw if y==1); H=sum(1 for _,y in sw if y==-1); W=sum(1 for _,y in sw if y==0)
    byq=collections.defaultdict(list)
    for q,y in sw: byq[q].append(y)
    keys=list(byq); rng=random.Random(7); est=[]
    for _ in range(4000):
        s=[rng.choice(keys) for _ in keys]
        c=sum(1 for k in s for y in byq[k] if y==1); h=sum(1 for k in s for y in byq[k] if y==-1)
        if h: est.append(c/h)
    est.sort()
    lo,hi=est[int(.025*len(est))],est[int(.975*len(est))]
    print(f"{name:8s} {len(sw):6d}  {C:3d} {H:3d} {W:3d}  {C/H:.2f}   [{lo:.2f}, {hi:.2f}]")
# oracle
tot=sum(1 for r in ledger if r["class_count"]>=2 and r["gold"] in r["counts"] and r["gold"]!=r["majority"])
har=sum(1 for r in ledger if r["class_count"]>=2 and r["gold"] in r["counts"] and r["gold"]==r["majority"])
print("ceiling: recoverable",tot,"at-risk",har)
