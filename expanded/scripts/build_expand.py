import os
# -*- coding: utf-8 -*-
"""从生成结果构建池、多数票与跨族评分任务。"""
import json, collections, glob, os
BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
Q={}
for f,k in (("gsm_q1200.json","gsm8k"),("csqa_q1200.json","csqa")):
    for r in json.load(open(f"{BASE}/data/{f}")): Q[r["id"]]=r

rows=[]
for f in glob.glob(f"{BASE}/out/gen_*_1200.jsonl"):
    for l in open(f):
        try: rows.append(json.loads(l))
        except Exception: pass
by=collections.defaultdict(list)
for r in rows:
    if r["answer"] is not None: by[(r["qid"], r["model"])].append(r)

pools=[]; tasks=[]
for (qid,model),rs in by.items():
    cnt=collections.Counter(); first={}
    for i,r in enumerate(rs):
        a=r["answer"]; cnt[a]+=1; first.setdefault(a,i)
    maj=max(cnt.items(), key=lambda kv:(kv[1],-first[kv[0]]))[0]
    conf=collections.defaultdict(list)
    for r in rs:
        if r["confidence"] is not None: conf[r["answer"]].append(r["confidence"]/100.0)
    pools.append(dict(qid=qid, model=model, task=Q[qid]["task"] if "task" in Q[qid] else ("gsm8k" if qid.startswith("gsm8k") else "csqa"),
                      gold=str(Q[qid]["gold"]), majority=maj, counts=dict(cnt), n=len(rs),
                      scores={k: sum(v)/len(v) for k,v in conf.items()}))
    for c in cnt:
        tasks.append(dict(key=f"{qid}|{model}|{c}", qid=qid, model=model, candidate=c,
                          question=Q[qid]["question"],
                          options=Q[qid].get("options"), gold=str(Q[qid]["gold"])))
json.dump(pools, open(f"{BASE}/out/pools_expanded.json","w"), ensure_ascii=False)
with open(f"{BASE}/data/tasks_expanded.jsonl","w") as f:
    for t in tasks: f.write(json.dumps(t, ensure_ascii=False)+"\n")
comp=sum(1 for p in pools if len(p["counts"])>=2)
print("pools",len(pools),"competing",comp,"score_tasks",len(tasks))
