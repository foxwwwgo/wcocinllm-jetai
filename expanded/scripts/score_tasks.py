import os
BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# -*- coding: utf-8 -*-
"""用 llama-server 的 OpenAI 兼容接口给候选答案打分（0-100），可断点续跑。"""
import json, os, re, sys, time, threading
import requests
from concurrent.futures import ThreadPoolExecutor

TASKS = f"{BASE}/data/tasks.jsonl"
OUTDIR = f"{BASE}/out"
os.makedirs(OUTDIR, exist_ok=True)

PORT = int(sys.argv[1]); TAG = sys.argv[2]; WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
OUT = f"{OUTDIR}/scores_{TAG}.jsonl"

PROMPT = """You are an evaluator for question answering. Judge whether the proposed answer is correct.

Question:
{question}
{options}

Proposed answer: {candidate}

On a scale from 0 to 100, what is the probability that the proposed answer is correct?
Return only one integer between 0 and 100, with no other text."""

def ask(t, tries=3):
    q = t["question"]
    opts = t.get("options") or ""
    if opts:
        opts = "\nOptions:\n" + opts
    prompt = PROMPT.format(question=q, options=opts, candidate=t["candidate"])
    for i in range(tries):
        try:
            r = requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                              json=dict(messages=[{"role": "user", "content": prompt}],
                                        temperature=0.0, max_tokens=16),
                              timeout=90)
            txt = r.json()["choices"][0]["message"]["content"]
            m = re.search(r"\d+", txt)
            if m:
                v = max(0, min(100, int(m.group())))
                return v
        except Exception:
            time.sleep(1.0 + i)
    return None

def main():
    tasks = [json.loads(l) for l in open(TASKS)]
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try: done.add(json.loads(l)["key"])
            except Exception: pass
    todo = [t for t in tasks if t["key"] not in done]
    print(f"todo {len(todo)} / {len(tasks)}", flush=True)
    lock = threading.Lock(); f = open(OUT, "a"); n = [0]; t0 = time.time()
    def work(t):
        v = ask(t)
        with lock:
            f.write(json.dumps({"key": t["key"], "score": v}) + "\n")
            n[0] += 1
            if n[0] % 100 == 0:
                el = time.time() - t0
                print(f"{n[0]}/{len(todo)}  {n[0]/el:.1f}/s  eta {(len(todo)-n[0])/max(n[0]/el,1e-9)/60:.1f} min", flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(work, todo))
    f.close()
    print("DONE", n[0], flush=True)

main()
