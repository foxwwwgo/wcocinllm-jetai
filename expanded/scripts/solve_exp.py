import os
# -*- coding: utf-8 -*-
import json, re, os, sys, time, threading, requests
from concurrent.futures import ThreadPoolExecutor
BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PORT,TAG,W,MT=int(sys.argv[1]),sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
OUT=f"{BASE}/out/xsolve_{TAG}.jsonl"
uniq={}
for l in open(f"{BASE}/data/tasks_expanded.jsonl"):
    t=json.loads(l); uniq.setdefault(t["qid"],t)
done=set()
if os.path.exists(OUT):
    for l in open(OUT):
        try:
            r=json.loads(l)
            if r["answer"] is not None: done.add(r["qid"])
        except Exception: pass
todo=[v for k,v in uniq.items() if k not in done]
print("todo",len(todo),flush=True)
PROMPT=("Solve the problem. Think it through, then give the final answer.\n\nQuestion:\n{question}\n{options}\n\n"
        "Finish with exactly one line: Proposed answer: (X)\nX is a signed integer for arithmetic, or one letter A-E.")
def parse(txt):
    if not txt: return None
    m=re.findall(r"Proposed answer:\s*\(?([^)\n]+)\)?", txt)
    a=m[-1].strip() if m else None
    if a is None:
        m=re.findall(r"-?\d[\d,]*", txt[-200:]); a=m[-1].replace(",","") if m else None
    if a is None: return None
    a=a.replace(",","").strip().rstrip(".")
    try: return str(int(float(a)))
    except Exception: return a.upper()[:1]
def one(t):
    opts=("\nOptions:\n"+t["options"]) if t.get("options") else ""
    p=PROMPT.format(question=t["question"],options=opts)
    for i in range(3):
        try:
            r=requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                json=dict(messages=[{"role":"user","content":p}],temperature=0.0,max_tokens=MT),timeout=300).json()
            m=r["choices"][0]["message"]; txt=m.get("content") or m.get("reasoning_content") or ""
            return parse(txt)
        except Exception: time.sleep(1+i)
    return None
lock=threading.Lock(); f=open(OUT,"a"); n=[0]; t0=time.time()
def work(t):
    v=one(t)
    with lock:
        f.write(json.dumps({"id":t["qid"],"answer":v})+"\n"); f.flush(); n[0]+=1
        if n[0]%50==0: print(f"{n[0]}/{len(todo)} {n[0]/(time.time()-t0):.2f}/s",flush=True)
with ThreadPoolExecutor(W) as ex: list(ex.map(work,todo))
f.close(); print("DONE",n[0],flush=True)
