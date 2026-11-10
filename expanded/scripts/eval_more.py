import os
# -*- coding: utf-8 -*-
import json, random, collections, math
BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ledger=json.load(open(f"{BASE}/data/certificate_pool_ledger.json"))
def ls(tag):
    d={}
    for l in open(f"{BASE}/out/scores_{tag}.jsonl"):
        r=json.loads(l)
        if r["score"] is not None: d[r["key"]]=r["score"]/100.0
    return d
QW=ls("qwen14b")
def lsolve(tag):
    d={}
    for l in open(f"{BASE}/out/solve_{tag}.jsonl"):
        r=json.loads(l)
        if r["answer"] is not None: d[r["id"]]=r["answer"]
    return d
QS=lsolve("qwen14b"); R1=lsolve("r1solve"); DSR=lsolve("dsreasoner")
def norm(a): return None if a is None else str(a).strip().upper().replace(" ","")
def ag(tab,r,c): return norm(tab.get(r["id"]))==norm(c)
def ext(r): return {c:QW[f"{r['id']}|{r['model']}|{c}"] for c in r["counts"] if f"{r['id']}|{r['model']}|{c}" in QW}
def ev(name, decide):
    sw=[]; pos=[]; neg=[]
    for r in ledger:
        if r["class_count"]<2: continue
        out=decide(r)
        if out is None: continue
        k,d=out
        (pos if k==r["gold"] else neg).append(d)
        if d>0: sw.append((r["id"],1 if k==r["gold"] else (-1 if r["majority"]==r["gold"] else 0)))
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
    print(f"{name:26s} sw={len(sw):3d} C={C:2d} H={H:2d} eta={C/H if H else float('nan'):.2f} CI=[{est[int(.025*len(est))]:.2f},{est[int(.975*len(est))]:.2f}] p={p:.3f}")
# judge ∧ r1-solve
def j_r1(r):
    s=ext(r)
    if not s or r["majority"] not in s: return None
    alt=[(v,k) for k,v in s.items() if k!=r["majority"] and ag(R1,r,k)]
    if not alt: return (r["majority"],0)
    v,k=max(alt)
    if v>s[r["majority"]] and not ag(R1,r,r["majority"]): return (k,1)
    return (r["majority"],0)
# 三者中至少两个一致
def vote2of3(r, third=R1):
    s=ext(r)
    if not s: return None
    score={}
    for c in r["counts"]:
        n=0
        if ag(QS,r,c): n+=1
        if ag(third,r,c): n+=1
        if s.get(c,0)>0.5: n+=1
        score[c]=n
    maj=score.get(r["majority"],0)
    alt=[(v,k) for k,v in score.items() if k!=r["majority"]]
    if not alt: return (r["majority"],0)
    v,k=max(alt)
    if v>=2 and v>maj: return (k,1)
    return (r["majority"],0)
ev("judge ∧ r1-solve", j_r1)
ev("2-of-3 vote (R1-Distill)", vote2of3)
ev("2-of-3 vote (DeepSeek thinking)", lambda r: vote2of3(r, DSR))
