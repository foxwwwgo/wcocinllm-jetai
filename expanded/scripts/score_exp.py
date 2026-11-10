import os
# -*- coding: utf-8 -*-
import json, os, re, sys, time, threading, requests
from concurrent.futures import ThreadPoolExecutor
BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PORT,TAG,W=int(sys.argv[1]),sys.argv[2],int(sys.argv[3])
OUT=f"{BASE}/out/xscores_{TAG}.jsonl"
PROMPT=("You are an evaluator for question answering. Judge whether the proposed answer is correct.\n\n"
        "Question:\n{question}\n{options}\n\nProposed answer: {candidate}\n\n"
        "On a scale from 0 to 100, what is the probability that the proposed answer is correct?\n"
        "Return only one integer between 0 and 100, with no other text.")
tasks=[json.loads(l) for l in open(f"{BASE}/data/tasks_expanded.jsonl")]
done=set()
if os.path.exists(OUT):
    for l in open(OUT):
        try:
            r=json.loads(l)
            if r["score"] is not None: done.add(r["key"])
        except Exception: pass
todo=[t for t in tasks if t["key"] not in done]
print("todo",len(todo),flush=True)
def ask(t):
    opts=("\nOptions:\n"+t["options"]) if t.get("options") else ""
    p=PROMPT.format(question=t["question"],options=opts,candidate=t["candidate"])
    for i in range(3):
        try:
            r=requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                json=dict(messages=[{"role":"user","content":p}],temperature=0.0,max_tokens=16),timeout=90)
            m=re.search(r"\d+", r.json()["choices"][0]["message"]["content"] or "")
            if m: return max(0,min(100,int(m.group())))
        except Exception: time.sleep(1+i)
    return None
lock=threading.Lock(); f=open(OUT,"a"); n=[0]; t0=time.time()
def work(t):
    v=ask(t)
    with lock:
        f.write(json.dumps({"key":t["key"],"score":v})+"\n"); f.flush(); n[0]+=1
        if n[0]%200==0: print(f"{n[0]}/{len(todo)} {n[0]/(time.time()-t0):.2f}/s",flush=True)
with ThreadPoolExecutor(W) as ex: list(ex.map(work,todo))
f.close(); print("DONE",n[0],flush=True)
