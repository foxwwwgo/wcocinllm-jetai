import os
# -*- coding: utf-8 -*-
import json, random, collections
BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ledger=json.load(open(f"{BASE}/data/certificate_pool_ledger.json"))
solved={}
for l in open(f"{BASE}/out/solve_qwen14b.jsonl"):
    r=json.loads(l); solved[r["id"]]=r["answer"]
def norm(a):
    if a is None: return None
    s=str(a).strip().upper().replace(" ","")
    return s or None
def sig(r):
    m=norm(solved.get(r["id"]))
    if m is None: return {}
    return {c: (1.0 if norm(c)==m else 0.0) for c in r["counts"]}
def auc(pos,neg):
    if not pos or not neg: return float("nan")
    w=sum(1 if p>n else .5 if p==n else 0 for p in pos for n in neg)
    return w/(len(pos)*len(neg))
pos,neg,sw=[],[],[]
for r in ledger:
    if r["class_count"]<2: continue
    s=sig(r)
    if not s or r["majority"] not in s: continue
    alt=[(v,k) for k,v in s.items() if k!=r["majority"]]
    if not alt: continue
    smax,kmax=max(alt); smaj=s[r["majority"]]
    if smaj==0 and smax==1: sw.append((r["id"],1 if kmax==r["gold"] else (-1 if r["majority"]==r["gold"] else 0)))
    d=smax-smaj
    (pos if kmax==r["gold"] else neg).append(d)
C=sum(1 for _,y in sw if y==1);H=sum(1 for _,y in sw if y==-1);W=sum(1 for _,y in sw if y==0)
byq=collections.defaultdict(list)
for q,y in sw: byq[q].append(y)
keys=list(byq);rng=random.Random(7);est=[]
for _ in range(4000):
    s=[rng.choice(keys) for _ in keys]
    c=sum(1 for k in s for y in byq[k] if y==1);h=sum(1 for k in s for y in byq[k] if y==-1)
    if h: est.append(c/h)
est.sort()
print("solver-agreement: solved",len(solved),"AUC=%.3f"%auc(pos,neg))
print("switches",len(sw),"C",C,"H",H,"W",W,"eta",round(C/H,2) if H else None,
      "CI",[round(est[int(.025*len(est))],2),round(est[int(.975*len(est))],2)])
