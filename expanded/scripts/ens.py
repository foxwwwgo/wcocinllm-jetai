import os
# -*- coding: utf-8 -*-
import json, random, collections
BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ledger=json.load(open(f"{BASE}/data/certificate_pool_ledger.json"))
def ls(tag):
    d={}
    for l in open(f"{BASE}/out/scores_{tag}.jsonl"):
        r=json.loads(l)
        if r["score"] is not None: d[r["key"]]=r["score"]/100.0
    return d
QW=ls("qwen14b"); PHI=ls("phi")
solved={}
for l in open(f"{BASE}/out/solve_qwen14b.jsonl"):
    r=json.loads(l); solved[r["id"]]=r["answer"]
def norm(a):
    return None if a is None else str(a).strip().upper().replace(" ","")
def qs(r): return {c:QW[f"{r['id']}|{r['model']}|{c}"] for c in r["counts"] if f"{r['id']}|{r['model']}|{c}" in QW}
def ps(r): return {c:PHI[f"{r['id']}|{r['model']}|{c}"] for c in r["counts"] if f"{r['id']}|{r['model']}|{c}" in PHI}
def agree(r,c): return norm(solved.get(r["id"]))==norm(c)
def ev(rule):
    sw=[]
    for r in ledger:
        if r["class_count"]<2: continue
        q=qs(r)
        if r["majority"] not in q: continue
        alt=[(v,k) for k,v in q.items() if k!=r["majority"]]
        if not alt: continue
        smax,kmax=max(alt)
        if rule(r,q,ps(r),smax,kmax):
            sw.append((r["id"],1 if kmax==r["gold"] else (-1 if r["majority"]==r["gold"] else 0)))
    C=sum(1 for _,y in sw if y==1);H=sum(1 for _,y in sw if y==-1)
    byq=collections.defaultdict(list)
    for a,y in sw: byq[a].append(y)
    keys=list(byq);rng=random.Random(7);est=[]
    for _ in range(4000):
        s=[rng.choice(keys) for _ in keys]
        c=sum(1 for k in s for y in byq[k] if y==1);h=sum(1 for k in s for y in byq[k] if y==-1)
        if h: est.append(c/h)
    est.sort()
    return len(sw),C,H,(C/H if H else None),([round(est[int(.025*len(est))],2),round(est[int(.975*len(est))],2)] if est else None)
rules={
 "judge only": lambda r,q,p,s,k: s>q[r["majority"]],
 "solver only": lambda r,q,p,s,k: agree(r,k) and not agree(r,r["majority"]),
 "judge & solver": lambda r,q,p,s,k: s>q[r["majority"]] and agree(r,k) and not agree(r,r["majority"]),
 "judge & solver | phi": lambda r,q,p,s,k: s>q[r["majority"]] and agree(r,k) and not agree(r,r["majority"]) and p.get(k,0)>p.get(r["majority"],0),
 "judge+solver, gap>=0.2": lambda r,q,p,s,k: s-q[r["majority"]]>=0.20 and agree(r,k) and not agree(r,r["majority"]),
}
for n,f in rules.items():
    a,b,c,d,e=ev(f)
    print(f"{n:24s} sw={a:3d} C={b:2d} H={c:2d} eta={d if d is None else round(d,2)} CI={e}")
