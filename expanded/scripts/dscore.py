import os
# -*- coding: utf-8 -*-
"""DeepSeek API as a cross-family independent signal: judge (scores candidate answers 0-100 probability)
and solve (solves independently).

Usage:
  python dscore.py judge tasks.jsonl <tag> [workers]
  python dscore.py solve tasks.jsonl <tag> [workers]

Outputs:
  out/scores_<tag>.jsonl   judge -> {"key":..., "score": int, "raw": last 120 chars}
  out/solve_<tag>.jsonl    solve -> {"id":..., "answer": str, "model": ...}

Environment variables:
  DS_MODEL      default deepseek-chat; can be set to deepseek-reasoner
  DS_SOLVE_MT   solve max_tokens, default 400 (reasoning models suggest 1500-4000)
The API key is read from BASE/.dskey and is not logged.
"""
import json, re, os, sys, time, threading, requests
from concurrent.futures import ThreadPoolExecutor

BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KEY = open(f"{BASE}/.dskey").read().strip()
URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = os.environ.get("DS_MODEL", "deepseek-chat")
SOLVE_MT = int(os.environ.get("DS_SOLVE_MT", 400))
JUDGE_MT = int(os.environ.get("DS_JUDGE_MT", 512))

MODE, TASKS, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
W = int(sys.argv[4]) if len(sys.argv) > 4 else 24
tasks_path = TASKS if TASKS.startswith("/") else f"{BASE}/data/{TASKS}"

JUDGE_PROMPT = (
    "You are an evaluator for question answering. Judge whether the proposed answer is correct.\n\n"
    "Question:\n{question}\n{options}\n\nProposed answer: {candidate}\n\n"
    "On a scale from 0 to 100, what is the probability that the proposed answer is correct?\n"
    "Return only one integer between 0 and 100, with no other text."
)
SOLVE_PROMPT = (
    "Solve the problem. Think it through, then give the final answer.\n\nQuestion:\n{question}\n{options}\n\n"
    "Finish with exactly one line: Proposed answer: (X)\nX is a signed integer for arithmetic, or one letter A-E."
)

OUT = f"{BASE}/out/{'scores' if MODE == 'judge' else 'solve'}_{TAG}.jsonl"
tasks = [json.loads(l) for l in open(tasks_path)]

# Dedup solving by question (old task files use id, scale-up files use qid)
if MODE == "solve":
    uniq = {}
    for t in tasks:
        qid = t.get("qid") or t["id"]
        uniq.setdefault(qid, t)
    tasks = [dict(t, key=qid) for qid, t in uniq.items()]

done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try:
            r = json.loads(l)
            v = r.get("score") if MODE == "judge" else r.get("answer")
            if v is not None:
                done.add(r["key"])
        except Exception:
            pass
todo = [t for t in tasks if t["key"] not in done]
print(f"mode={MODE} model={MODEL} total={len(tasks)} todo={len(todo)}", flush=True)


def parse_solve(txt):
    if not txt:
        return None
    m = re.findall(r"Proposed answer:\s*\(?([^)\n]+)\)?", txt)
    a = m[-1].strip() if m else None
    if a is None:
        m = re.findall(r"-?\d[\d,]*", txt[-200:])
        a = m[-1].replace(",", "") if m else None
    if a is None:
        return None
    a = a.replace(",", "").strip().rstrip(".")
    try:
        return str(int(float(a)))
    except Exception:
        return a.upper()[:1]


def call(prompt, max_tokens):
    for i in range(5):
        try:
            r = requests.post(
                URL,
                headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
                json=dict(model=MODEL, messages=[{"role": "user", "content": prompt}],
                          temperature=0.0, max_tokens=max_tokens),
                timeout=300,
            )
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 + 2 * i)
                continue
            j = r.json()
            if "choices" not in j:
                time.sleep(2 + 2 * i)
                continue
            m = j["choices"][0]["message"]
            return m.get("content") or m.get("reasoning_content") or ""
        except Exception:
            time.sleep(1 + i)
    return None


def one(t):
    """Return (value, raw tail); value is a 0-100 integer for judge, the answer parse result for solve."""
    opts = ("\nOptions:\n" + t["options"]) if t.get("options") else ""
    if MODE == "judge":
        txt = call(JUDGE_PROMPT.format(question=t["question"], options=opts, candidate=t["candidate"]), JUDGE_MT)
        if txt is None:
            return None, None
        # Take the last standalone 0-100 integer: reasoning models write step numbers first
        cands = [int(x) for x in re.findall(r"(?<![\d.])\d{1,3}(?![\d.])", txt) if int(x) <= 100]
        v = cands[-1] if cands else None
        return v, (txt.strip().replace("\n", " ")[-120:])
    txt = call(SOLVE_PROMPT.format(question=t["question"], options=opts), SOLVE_MT)
    return parse_solve(txt), None


lock = threading.Lock()
f = open(OUT, "a")
n = [0]
t0 = time.time()


def work(t):
    v, raw = one(t)
    rec = ({"key": t["key"], "score": v, "raw": raw} if MODE == "judge"
           else {"id": t["key"], "answer": v, "model": MODEL})
    with lock:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        n[0] += 1
        if n[0] % 100 == 0:
            print(f"{n[0]}/{len(todo)} {n[0]/(time.time()-t0):.2f}/s", flush=True)


with ThreadPoolExecutor(W) as ex:
    list(ex.map(work, todo))
f.close()
print("DONE", n[0], flush=True)
