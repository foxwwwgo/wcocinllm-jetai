import os
# -*- coding: utf-8 -*-
"""思考剥夺消融：Prompt 级长度控制（非硬截断），隔离解析率与准确率。

用法: cot_ablation.py <port> <n_questions> <paths_per_q> <workers>
"""
import json, re, sys, time, threading, requests
from concurrent.futures import ThreadPoolExecutor

BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PORT, NQ, NP, W = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
OUT = f"{BASE}/out/cot_ablation.jsonl"

QS = json.load(open(f"{BASE}/data/gsm_q1200.json"))[:NQ]

MODES = {
    "full": ("Solve this problem step by step, showing your reasoning.\n\n"
             "End your response with exactly one line of the form: Proposed answer: (X)",
             "Answer the question immediately in at most one short sentence, without working through it.\n\n"
             "End your response with exactly one line of the form: Proposed answer: (X)"),
}
# full / restricted
MODES = {"full": MODES["full"][0], "restricted": MODES["full"][1]}

PROMPT = "{instr}\n\nQuestion:\n{question}"
CONF = ("Question:\n{question}\n\nYour answer was:\n{answer}\n\n"
        "Rate your confidence that your proposed answer to the original question is correct, "
        "on a scale from 0 to 100.\nReturn only one number between 0 and 100, with no explanation.")

def post(prompt, mt, temp):
    r = requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                      json=dict(messages=[{"role": "user", "content": prompt}],
                                temperature=temp, max_tokens=mt), timeout=300)
    j = r.json()
    m = j["choices"][0]["message"]
    txt = (m.get("content") or "") or (m.get("reasoning_content") or "")
    return txt, j.get("usage", {}).get("completion_tokens")

def parse_ans(txt):
    m = re.findall(r"Proposed answer:\s*\(?([^)\n]+)\)?", txt or "")
    if m:
        a = m[-1].strip()
    else:
        m = re.findall(r"-?\d[\d,]*", (txt or "")[-200:])
        a = m[-1] if m else None
    if a is None: return None
    a = a.replace(",", "").strip().rstrip(".")
    try: return str(int(float(a)))
    except Exception: return a

def one(q, mode, k):
    for i in range(3):
        try:
            txt, nt = post(PROMPT.format(instr=MODES[mode], question=q["question"]), 900, 0.7)
            ans = parse_ans(txt)
            conf = None
            try:
                c, _ = post(CONF.format(question=q["question"], answer=txt[:1500]), 8, 0.0)
                m = re.search(r"\d+", c or "")
                if m: conf = max(0, min(100, int(m.group())))
            except Exception: pass
            return dict(qid=q["id"], mode=mode, k=k, gold=str(q["gold"]), answer=ans,
                        confidence=conf, tokens=nt, raw_len=len(txt or ""))
        except Exception:
            time.sleep(1 + i)
    return dict(qid=q["id"], mode=mode, k=k, gold=str(q["gold"]), answer=None,
                confidence=None, tokens=None, raw_len=None)

def main():
    jobs = [(q, m, k) for q in QS for m in MODES for k in range(NP)]
    done = set()
    try:
        for l in open(OUT):
            r = json.loads(l)
            if r["answer"] is not None: done.add((r["qid"], r["mode"], r["k"]))
    except FileNotFoundError: pass
    todo = [j for j in jobs if (j[0]["id"], j[1], j[2]) not in done]
    print(f"todo {len(todo)} / {len(jobs)}", flush=True)
    lock = threading.Lock(); f = open(OUT, "a"); n = [0]; t0 = time.time()
    def work(j):
        r = one(*j)
        with lock:
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush(); n[0] += 1
            if n[0] % 40 == 0:
                el = time.time() - t0
                print(f"{n[0]}/{len(todo)} {n[0]/el:.2f}/s eta {(len(todo)-n[0])/max(n[0]/el,1e-9)/60:.0f} min", flush=True)
    with ThreadPoolExecutor(W) as ex: list(ex.map(work, todo))
    f.close(); print("DONE", n[0], flush=True)

main()
