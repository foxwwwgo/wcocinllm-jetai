import os
# -*- coding: utf-8 -*-
"""让评分模型独立解题，作为第三种信号（与候选答案是否一致）。"""
import json, re, os, sys, time, threading, requests
from concurrent.futures import ThreadPoolExecutor
BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PORT=int(sys.argv[1]); TAG=sys.argv[2]; W=int(sys.argv[3])
OUT=f"{BASE}/out/solve_{TAG}.jsonl"
tasks=[json.loads(l) for l in open(f"{BASE}/data/tasks.jsonl")]
uniq={}
for t in tasks: uniq.setdefault(t["id"], t)
done=set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: done.add(json.loads(l)["id"])
        except Exception: pass
todo=[v for k,v in uniq.items() if k not in done]
print("todo",len(todo),flush=True)
PROMPT=("Solve the problem step by step.\n\nQuestion:\n{question}\n{options}\n\n"
        "End with a line of the form: Proposed answer: (X)\n"
        "where X is a signed integer for arithmetic problems, or a single letter A-E for multiple choice.")
def one(t):
    opts=("\nOptions:\n"+t["options"]) if t.get("options") else ""
    p=PROMPT.format(question=t["question"],options=opts)
    for i in range(3):
        try:
            r=requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                json=dict(messages=[{"role":"user","content":p}],temperature=0.0,max_tokens=400),timeout=180)
            txt=r.json()["choices"][0]["message"]["content"]
            m=re.findall(r"Proposed answer:\s*\(?([^)\n]+)\)?",txt)
            return m[-1].strip() if m else None
        except Exception:
            time.sleep(1+i)
    return None
lock=threading.Lock(); f=open(OUT,"a"); n=[0]; t0=time.time()
def work(t):
    v=one(t)
    with lock:
        f.write(json.dumps({"id":t["id"],"answer":v})+"\n"); n[0]+=1
        if n[0]%50==0: print(f"{n[0]}/{len(todo)} {n[0]/(time.time()-t0):.1f}/s",flush=True)
with ThreadPoolExecutor(W) as ex: list(ex.map(work,todo))
f.close(); print("DONE",n[0],flush=True)
