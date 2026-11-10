import os
# -*- coding: utf-8 -*-
import json, random, collections, math
BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ledger=json.load(open(f"{BASE}/data/certificate_pool_ledger.json"))
def load_solve(tag):
    d={}
    for l in open(f"{BASE}/out/solve_{tag}.jsonl"):
        r=json.loads(l)
        if r["answer"] is not None: d[r["id"]]=r["answer"]
    return d
QS=load_solve("qwen14b"); R1=load_solve("r1solve")
def norm(a): return None if a is None else str(a).strip().upper().replace(" ","")
def agree(tab,r,c): return norm(tab.get(r["id"]))==norm(c)
def ev(name, decide):
    sw=[]; pos=[]; neg=[]
    for r in ledger:
        if r["class_count"]<2: continue
        out=decide(r)
        if out is None: continue
        kmax,d=out
        (pos if kmax==r["gold"] else neg).append(d)
        if d>0: sw.append((r["id"],1 if kmax==r["gold"] else (-1 if r["majority"]==r["gold"] else 0)))
    C=sum(1 for _,y in sw if y==1);H=sum(1 for _,y in sw if y==-1)
    byq=collections.defaultdict(list)
    for q,y in sw: byq[q].append(y)
    keys=list(byq);rng=random.Random(7);est=[]
    for _ in range(5000):
        s=[rng.choice(keys) for _ in keys]
        c=sum(1 for k in s for y in byq[k] if y==1);h=sum(1 for k in s for y in byq[k] if y==-1)
        if h: est.append(c/h)
    est.sort()
    p=min(1.0,2*sum(math.comb(C+H,i) for i in range(0,min(C,H)+1))/2**(C+H)) if C+H else None
    print(f"{name:16s} sw={len(sw):3d} C={C:2d} H={H:2d} eta={C/H if H else float('nan'):.2f} CI=[{est[int(.025*len(est))]:.2f},{est[int(.975*len(est))]:.2f}] p={p:.3f}")
def mk(tab):
    def f(r):
        s={c for c in r["counts"]}
        if not s: return None
        alt=[c for c in s if c!=r["majority"] and agree(tab,r,c)]
        maj = agree(tab,r,r["majority"])
        if alt and not maj: return (alt[0],1.0)
        return (r["majority"],0.0)
    return f
ev("qwen solve", mk(QS))
ev("r1 solve", mk(R1))
def both(r):
    alt=[c for c in r["counts"] if c!=r["majority"] and agree(QS,r,c) and agree(R1,r,c)]
    maj = agree(QS,r,r["majority"]) or agree(R1,r,r["majority"])
    if alt and not maj: return (alt[0],1.0)
    return (r["majority"],0.0)
ev("qwen AND r1", both)
