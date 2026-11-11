import os
# -*- coding: utf-8 -*-
"""Independent solving (adapted for models that emit reasoning first), resumable."""
import json, re, os, sys, time, threading, requests
from concurrent.futures import ThreadPoolExecutor
BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PORT=int(sys.argv[1]); TAG=sys.argv[2]; W=int(sys.argv[3]); MT=int(sys.argv[4]) if len(sys.argv)>4 else 1200
OUT=f"{BASE}/out/solve_{TAG}.jsonl"
tasks=[json.loads(l) for l in open(f"{BASE}/data/tasks.jsonl")]
uniq={}
for t in tasks: uniq.setdefault(t["id"], t)
done=set()
if os.path.exists(OUT):
    for l in open(OUT):
        try:
            r=json.loads(l)
            if r["answer"] is not None: done.add(r["id"])
        except Exception: pass
todo=[v for k,v in uniq.items() if k not in done]
print("todo",len(todo),flush=True)
PROMPT=("Solve the problem. Think it through, then give the final answer.\n\nQuestion:\n{question}\n{options}\n\n"
        "Finish with exactly one line: Proposed answer: (X)\n"
        "X is a signed integer for arithmetic, or a single letter A-E for multiple choice.")
def parse(txt):
    if not txt: return None
    m=re.findall(r"Proposed answer:\s*\(?([^)\n]+)\)?", txt)
    if m: return m[-1].strip()
    m=re.findall(r"(?:final answer|the answer is|answer:)\s*\**\s*\(?(-?\d+(?:\.\d+)?|[A-E])\)?", txt, re.I)
    if m: return m[-1].strip()
    m=re.findall(r"\b([A-E])\b", txt[-300:])
    if m: return m[-1]
    m=re.findall(r"-?\d+(?:\.\d+)?", txt[-300:])
    return m[-1] if m else None
def one(t):
    opts=("\nOptions:\n"+t["options"]) if t.get("options") else ""
    p=PROMPT.format(question=t["question"],options=opts)
    for i in range(3):
        try:
            r=requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                json=dict(messages=[{"role":"user","content":p}],temperature=0.0,max_tokens=MT),timeout=300)
            m=r.json()["choices"][0]["message"]
            txt=m.get("content") or ""
            if not txt.strip(): txt=m.get("reasoning_content") or ""
            return parse(txt)
        except Exception:
            time.sleep(1+i)
    return None
lock=threading.Lock(); f=open(OUT,"a"); n=[0]; t0=time.time()
def work(t):
    v=one(t)
    with lock:
        f.write(json.dumps({"id":t["id"],"answer":v})+"\n"); f.flush(); n[0]+=1
        if n[0]%25==0: print(f"{n[0]}/{len(todo)} {n[0]/(time.time()-t0):.2f}/s",flush=True)
with ThreadPoolExecutor(W) as ex: list(ex.map(work,todo))
f.close(); print("DONE",n[0],flush=True)
