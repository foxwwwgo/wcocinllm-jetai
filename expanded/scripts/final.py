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
QW,PHI=ls("qwen14b"),ls("phi")
solved={json.loads(l)["id"]:json.loads(l)["answer"] for l in open(f"{BASE}/out/solve_qwen14b.jsonl")}
def norm(a): return None if a is None else str(a).strip().upper().replace(" ","")
def selfs(r):
    a=collections.defaultdict(list)
    for x,s in zip(r["answers"],r["scores"]):
        if x is None: continue
        a[x].append(s)
    return {k:sum(v)/len(v)/100.0 for k,v in a.items()}
def ext(r,t): return {c:t[f"{r['id']}|{r['model']}|{c}"] for c in r["counts"] if f"{r['id']}|{r['model']}|{c}" in t}
def agree(r,c): return norm(solved.get(r["id"]))==norm(c)
def auc(pos,neg):
    if not pos or not neg: return float("nan")
    w=sum(1 if p>n else .5 if p==n else 0 for p in pos for n in neg)
    return w/(len(pos)*len(neg))
def sign_p(c,h):
    n=c+h
    if n==0: return None
    k=min(c,h)
    p=sum(math.comb(n,i) for i in range(0,k+1))/2**n*2
    return min(1.0,p)
def report(name, decide, score=None):
    sw=[];pos=[];neg=[]
    for r in ledger:
        if r["class_count"]<2: continue
        ok=decide(r)
        if ok is None: continue
        kmax, smaj, smax = ok
        if score is not None:
            d=smax-smaj
            (pos if kmax==r["gold"] else neg).append(d)
        if smaj is not None and smax>smaj if isinstance(smaj,float) else True:
            pass
    return sw

def evalrule(name, rule, aucmode=False):
    sw=[];pos=[];neg=[]
    for r in ledger:
        if r["class_count"]<2: continue
        out=rule(r)
        if out is None: continue
        kmax,d=out
        if aucmode: (pos if kmax==r["gold"] else neg).append(d)
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
    return dict(sw=len(sw),C=C,H=H,W=len(sw)-C-H,eta=(C/H if H else None),
                ci=([round(est[int(.025*len(est))],2),round(est[int(.975*len(est))],2)] if est else None),
                auc=(auc(pos,neg) if aucmode else None), p=sign_p(C,H))
def mk_sig(sfn):
    def rule(r):
        s=sfn(r)
        if not s or r["majority"] not in s: return None
        alt=[(v,k) for k,v in s.items() if k!=r["majority"]]
        if not alt: return None
        smax,kmax=max(alt)
        return kmax, smax-s[r["majority"]]
    return rule
rules=[
 ("self-confidence", mk_sig(selfs)),
 ("Phi-3-mini judge", mk_sig(lambda r: ext(r,PHI))),
 ("Qwen2.5-14B judge", mk_sig(lambda r: ext(r,QW))),
 ("Qwen2.5-14B solver", mk_sig(lambda r: {c:(1.0 if agree(r,c) else 0.0) for c in r["counts"]})),
]
print(f"{'signal':22s} {'AUC':>5s} {'sw':>4s} {'C':>3s} {'H':>3s} {'eta':>5s}  {'95%CI':>12s}  {'sign p':>6s}")
rows=[]
for n,f in rules:
    r=evalrule(n,f,aucmode=True); rows.append((n,r))
    print(f"{n:22s} {r['auc']:.3f} {r['sw']:4d} {r['C']:3d} {r['H']:3d} {r['eta']:5.2f}  {str(r['ci']):>12s}  {r['p']:.3f}")
def ens(r):
    q=ext(r,QW)
    if not q or r["majority"] not in q: return None
    alt=[(v,k) for k,v in q.items() if k!=r["majority"]]
    if not alt: return None
    smax,kmax=max(alt)
    if not (smax>q[r["majority"]] and agree(r,kmax) and not agree(r,r["majority"])): return None
    return kmax, 1.0
r=evalrule("judge&solver",ens); print(f"{'judge AND solver':22s} {'-':>5s} {r['sw']:4d} {r['C']:3d} {r['H']:3d} {r['eta']:5.2f}  {str(r['ci']):>12s}  {r['p']:.3f}")
json.dump({"signals":rows,"ensemble":r},open(f"{BASE}/out/summary.json","w"),ensure_ascii=False,indent=1,default=str)
