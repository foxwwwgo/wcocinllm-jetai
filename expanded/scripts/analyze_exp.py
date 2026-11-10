import os
# -*- coding: utf-8 -*-
import json, random, collections, math, os, sys
BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pools=json.load(open(f"{BASE}/out/pools_expanded.json"))
def lscore(tag):
    d={}
    p=f"{BASE}/out/xscores_{tag}.jsonl"
    if os.path.exists(p):
        for l in open(p):
            r=json.loads(l)
            if r["score"] is not None: d[r["key"]]=r["score"]/100.0
    return d
def lsolve(tag):
    d={}
    p=f"{BASE}/out/xsolve_{tag}.jsonl"
    if os.path.exists(p):
        for l in open(p):
            r=json.loads(l)
            if r["answer"] is not None: d[r["id"]]=r["answer"]
    return d
QW=lscore("qwen14b"); QS=lsolve("qwen"); R1=lsolve("r1")
def norm(a): return None if a is None else str(a).strip().upper().replace(" ","")
def ag(tab,p,c): return norm(tab.get(p["qid"]))==norm(c)
def judge(p): return {c:QW[f"{p['qid']}|{p['model']}|{c}"] for c in p["counts"] if f"{p['qid']}|{p['model']}|{c}" in QW}
def auc(pos,neg):
    if not pos or not neg: return float("nan")
    w=sum(1 if a>b else .5 if a==b else 0 for a in pos for b in neg)
    return w/(len(pos)*len(neg))
def ev(name, fn, aucmode=True):
    sw=[];pos=[];neg=[]
    for p in pools:
        if len(p["counts"])<2: continue
        s=fn(p)
        if not s or p["majority"] not in s: continue
        alt=[(v,k) for k,v in s.items() if k!=p["majority"]]
        if not alt: continue
        smax,kmax=max(alt); smaj=s[p["majority"]]
        if aucmode:
            (pos if kmax==p["gold"] else neg).append(smax-smaj)
        if smax>smaj:
            sw.append((p["qid"],1 if kmax==p["gold"] else (-1 if p["majority"]==p["gold"] else 0)))
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
    ci=f"[{est[int(.025*len(est))]:.2f},{est[int(.975*len(est))]:.2f}]" if est else "-"
    print(f"{name:22s} AUC={auc(pos,neg):.3f} sw={len(sw):4d} C={C:3d} H={H:3d} eta={C/H if H else float('nan'):.2f} CI={ci} p={p:.3f}",flush=True)
def mk(fn):
    def f(p):
        s=fn(p)
        return s if s and p["majority"] in s else None
    return f
ev("self", mk(lambda p: p["scores"]))
ev("qwen judge", mk(judge))
ev("qwen solve", mk(lambda p: {c:(1.0 if ag(QS,p,c) else 0.0) for c in p["counts"]}))
ev("r1 solve", mk(lambda p: {c:(1.0 if ag(R1,p,c) else 0.0) for c in p["counts"]}))
def twoof3(p):
    j=judge(p)
    if not j: return None
    sc={}
    for c in p["counts"]:
        sc[c]=int(ag(QS,p,c))+int(ag(R1,p,c))+int(j.get(c,0)>0.5)
    return {c:float(v) for c,v in sc.items()}
ev("2-of-3", mk(twoof3), aucmode=False)
print("pools",len(pools),"competing",sum(1 for p in pools if len(p["counts"])>=2))
