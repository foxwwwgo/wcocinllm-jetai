import os
# -*- coding: utf-8 -*-
"""Expanded sample: N-path generation per question + per-path confidence. Resumable.

Usage: python gen_pipeline.py <port> <model_tag> <n_questions> <workers>
"""
import json, os, re, sys, time, threading, requests
from concurrent.futures import ThreadPoolExecutor

BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PORT, TAG, NQ, W = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
OUT = f"{BASE}/out/gen_{TAG}_{NQ}.jsonl"

def load_q():
    qs = []
    gsm = json.load(open(f"{BASE}/data/gsm_q1200.json"))
    csq = json.load(open(f"{BASE}/data/csqa_q1200.json"))
    for r in gsm[:NQ]: qs.append(dict(qid=r["id"], task="gsm8k", question=r["question"], options=None, gold=r["gold"]))
    for r in csq[:NQ]: qs.append(dict(qid=r["id"], task="csqa", question=r["question"], options=r["options"], gold=r["gold"]))
    return qs

PROMPT = ("Solve the problem step by step. Think carefully.\n\nQuestion:\n{question}\n{options}\n\n"
          "End your response with a final line of exactly this form: Proposed answer: (X)\n"
          "For arithmetic questions X is a signed integer; for multiple choice X is one letter A-E.")

CONF = ("Question:\n{question}\n{options}\n\nYour answer was:\n{answer}\n\n"
        "Rate your confidence that your proposed answer to the original question is correct, on a scale from 0 to 100.\n"
        "Return only one number between 0 and 100, with no explanation.")

def parse_ans(txt, task):
    if not txt: return None
    m = re.findall(r"Proposed answer:\s*\(?([^)\n]+)\)?", txt)
    if m: a = m[-1].strip()
    else:
        if task == "gsm8k":
            m = re.findall(r"-?\d[\d,]*", txt[-200:])
            a = m[-1].replace(",", "") if m else None
        else:
            m = re.findall(r"\b([A-E])\b", txt[-400:])
            a = m[-1] if m else None
    if a is None: return None
    a = a.replace(",", "").strip().rstrip(".")
    if task == "gsm8k":
        try: return str(int(float(a)))
        except Exception: return a
    return a.upper()[:1] if a else None

def post(prompt, mt, temp):
    r = requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                      json=dict(messages=[{"role": "user", "content": prompt}],
                                temperature=temp, max_tokens=mt), timeout=300)
    m = r.json()["choices"][0]["message"]
    return (m.get("content") or "") or (m.get("reasoning_content") or "")

def one(q, seat):
    opts = ("\nOptions:\n" + q["options"]) if q["options"] else ""
    for i in range(3):
        try:
            txt = post(PROMPT.format(question=q["question"], options=opts), 512, 1.0)
            ans = parse_ans(txt, q["task"])
            if ans is None or not txt.strip():
                time.sleep(0.5 + i); continue
            conf = None
            try:
                c = post(CONF.format(question=q["question"], options=opts, answer=ans), 8, 0.0)
                m = re.search(r"\d+", c or "")
                if m: conf = max(0, min(100, int(m.group())))
            except Exception: pass
            return dict(qid=q["qid"], task=q["task"], model=TAG, seat=seat,
                        gold=q["gold"], answer=ans, confidence=conf, raw=txt[:2000])
        except Exception:
            time.sleep(1 + i)
    return dict(qid=q["qid"], task=q["task"], model=TAG, seat=seat,
                gold=q["gold"], answer=None, confidence=None, raw=None)

def main():
    qs = load_q()
    jobs = [(q, s) for q in qs for s in range(7)]
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try:
                r = json.loads(l); done.add((r["qid"], r["seat"]))
            except Exception: pass
    todo = [j for j in jobs if (j[0]["qid"], j[1]) not in done]
    print(f"todo {len(todo)} / {len(jobs)}", flush=True)
    lock = threading.Lock(); f = open(OUT, "a"); n = [0]; t0 = time.time()
    def work(j):
        r = one(*j)
        with lock:
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush(); n[0] += 1
            if n[0] % 50 == 0:
                el = time.time() - t0
                print(f"{n[0]}/{len(todo)} {n[0]/el:.2f}/s eta {(len(todo)-n[0])/max(n[0]/el,1e-9)/3600:.2f}h", flush=True)
    with ThreadPoolExecutor(W) as ex: list(ex.map(work, todo))
    f.close(); print("DONE", n[0], flush=True)

main()
