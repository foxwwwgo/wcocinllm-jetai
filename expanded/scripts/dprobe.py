import os
# -*- coding: utf-8 -*-
"""Two robustness probes.

A. Solve token budget: DeepSeek at max_tokens=400 reaches only 86% on GSM8K; suspected truncation.
   Re-run the GSM8K subset at 1500 tokens to see if accuracy recovers.
B. Judge saturation: DeepSeek non-thinking mode scores 73.6% of candidates 100 under the standard prompt,
   candidate-level AUC 0.505. Try two stricter phrasings to see whether saturation disappears -- if it does,
   it is a prompt issue; if not, it is model behavior.

Usage: python dprobe.py A|B [n]
"""
import json, re, sys, time, threading, random, requests, collections
from concurrent.futures import ThreadPoolExecutor

BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KEY = open(f"{BASE}/.dskey").read().strip()
URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"
MODE = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300

led = json.load(open(f"{BASE}/data/certificate_pool_ledger.json"))
gold = {r["id"]: r["gold"] for r in led}
TASKS = [json.loads(l) for l in open(f"{BASE}/data/tasks.jsonl")]


def call(prompt, max_tokens):
    for i in range(5):
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
                              json=dict(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                        temperature=0.0, max_tokens=max_tokens), timeout=300)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 + 2 * i); continue
            j = r.json()
            if "choices" not in j:
                time.sleep(2 + 2 * i); continue
            return j["choices"][0]["message"].get("content") or ""
        except Exception:
            time.sleep(1 + i)
    return None


def parse_num(txt):
    if not txt: return None
    m = re.findall(r"Proposed answer:\s*\(?([^)\n]+)\)?", txt)
    a = m[-1].strip() if m else None
    if a is None:
        m = re.findall(r"-?\d[\d,]*", txt[-200:]); a = m[-1].replace(",", "") if m else None
    if a is None: return None
    a = a.replace(",", "").strip().rstrip(".")
    try: return str(int(float(a)))
    except Exception: return a.upper()[:1]


def run(items, fn, outfile):
    lock = threading.Lock(); f = open(outfile, "w"); n = [0]; t0 = time.time()
    def work(t):
        v = fn(t)
        with lock:
            f.write(json.dumps(v, ensure_ascii=False) + "\n"); f.flush(); n[0] += 1
            if n[0] % 50 == 0: print(f"{n[0]}/{len(items)} {n[0]/(time.time()-t0):.2f}/s", flush=True)
    with ThreadPoolExecutor(16) as ex: list(ex.map(work, items))
    f.close()


if MODE == "A":
    gsm = {}
    for t in TASKS:
        if t["id"].startswith("gsm"): gsm.setdefault(t["id"], t)
    items = list(gsm.values())
    print("GSM questions:", len(items), "max_tokens=1500", flush=True)
    P = ("Solve the problem. Think it through, then give the final answer.\n\nQuestion:\n{question}\n\n"
         "Finish with exactly one line: Proposed answer: (X)\nX is a signed integer.")
    def fn(t):
        txt = call(P.format(question=t["question"]), 1500)
        return {"id": t["id"], "answer": parse_num(txt), "chars": len(txt or "")}
    run(items, fn, f"{BASE}/out/probeA_ds_solve_1500.jsonl")
    rows = [json.loads(l) for l in open(f"{BASE}/out/probeA_ds_solve_1500.jsonl")]
    ok = sum(1 for r in rows if r["answer"] is not None
             and str(r["answer"]).strip().upper() == str(gold[r["id"]]).strip().upper())
    parsed = sum(1 for r in rows if r["answer"] is not None)
    print(f"parsed {parsed}/{len(rows)}  acc {100*ok/max(parsed,1):.1f}%  "
          f"mean chars {sum(r['chars'] for r in rows)/len(rows):.0f}")

elif MODE == "C":
    # CSQA: candidate given letter only vs letter + option text; test whether judge saturation comes from presentation
    rng = random.Random(20261108)
    pool = {}
    for t in TASKS:
        if t["id"].startswith("csqa"):
            pool.setdefault(t["key"], t)
    pool = list(pool.values())
    sub = rng.sample(pool, min(N, len(pool)))
    print("CSQA tasks:", len(sub), flush=True)

    def optmap(opts):
        d = {}
        for line in (opts or "").split("\n"):
            line = line.strip()
            if len(line) > 2 and line[1] == ".":
                d[line[0].upper()] = line[2:].strip()
        return d

    P = ("You are an evaluator for question answering. Judge whether the proposed answer is correct.\n\n"
         "Question:\n{question}\nOptions:\n{options}\n\nProposed answer: {candidate}\n\n"
         "On a scale from 0 to 100, what is the probability that the proposed answer is correct?\n"
         "Return only one integer between 0 and 100, with no other text.")

    def fn(t):
        m = optmap(t.get("options"))
        letter = str(t["candidate"]).strip().upper()[:1]
        full = f"{letter}. {m.get(letter, '')}" if letter in m else str(t["candidate"])
        txt = call(P.format(question=t["question"], options=t.get("options") or "",
                            candidate=full), 16)
        nums = re.findall(r"\d+", txt or "")
        return {"key": t["key"], "score": (max(0, min(100, int(nums[0]))) if nums else None),
                "gold": str(t["candidate"]).strip().upper() == str(gold[t["id"]]).strip().upper()}

    run(sub, fn, f"{BASE}/out/probeC_csqa_lettertext.jsonl")
    rows = [json.loads(l) for l in open(f"{BASE}/out/probeC_csqa_lettertext.jsonl")]
    vals = [r["score"] for r in rows if r["score"] is not None]
    pos = [r["score"] for r in rows if r["score"] is not None and r["gold"]]
    neg = [r["score"] for r in rows if r["score"] is not None and not r["gold"]]
    w = sum(1 if p > n else .5 if p == n else 0 for p in pos for n in neg)
    print(f"letter+text: n={len(vals)} sat100={100*sum(1 for v in vals if v==100)/len(vals):.1f}% "
          f"mean={sum(vals)/len(vals):.1f} AUC={w/(len(pos)*len(neg)):.3f}")
    std = {}
    for l in open(f"{BASE}/out/scores_deepseek.jsonl"):
        r = json.loads(l)
        if r["score"] is not None:
            std[r["key"]] = r["score"]
    s = [t for t in sub if t["key"] in std]
    vals = [std[t["key"]] for t in s]
    pos = [std[t["key"]] for t in s if str(t["candidate"]).strip().upper() == str(gold[t["id"]]).strip().upper()]
    neg = [std[t["key"]] for t in s if str(t["candidate"]).strip().upper() != str(gold[t["id"]]).strip().upper()]
    w = sum(1 if p > n else .5 if p == n else 0 for p in pos for n in neg)
    print(f"letter only(same subset): n={len(vals)} sat100={100*sum(1 for v in vals if v==100)/len(vals):.1f}% "
          f"mean={sum(vals)/len(vals):.1f} AUC={w/(len(pos)*len(neg)):.3f}")

else:
    rng = random.Random(20261108)
    pool = [t for t in TASKS if t["id"].startswith("gsm")]
    sub = rng.sample(pool, min(N, len(pool)))
    VARIANTS = {
        "strict": ("You are a strict grader. Check the solution carefully: most proposed answers contain an error. "
                   "Give the probability that the proposed answer is correct.\n\nQuestion:\n{question}\n\n"
                   "Proposed answer: {candidate}\n\nReturn only one integer between 0 and 100."),
        "spread": ("Estimate the probability that the proposed answer is correct. Use the full 0-100 range: "
                   "do not answer 100 unless you have verified every step.\n\nQuestion:\n{question}\n\n"
                   "Proposed answer: {candidate}\n\nReturn only one integer between 0 and 100."),
    }
    for name, P in VARIANTS.items():
        print("variant", name, flush=True)
        def fn(t, P=P):
            txt = call(P.format(question=t["question"], candidate=t["candidate"]), 16)
            m = re.findall(r"\d+", txt or "")
            return {"key": t["key"], "score": (max(0, min(100, int(m[0]))) if m else None),
                    "gold": t["candidate"] == gold[t["id"]]}
        run(sub, fn, f"{BASE}/out/probeB_{name}.jsonl")
        rows = [json.loads(l) for l in open(f"{BASE}/out/probeB_{name}.jsonl")]
        vals = [r["score"] for r in rows if r["score"] is not None]
        pos = [r["score"] for r in rows if r["score"] is not None and r["gold"]]
        neg = [r["score"] for r in rows if r["score"] is not None and not r["gold"]]
        w = sum(1 if p > n else .5 if p == n else 0 for p in pos for n in neg)
        a = w / (len(pos) * len(neg)) if pos and neg else None
        sat = 100 * sum(1 for v in vals if v == 100) / len(vals)
        print(f"  n={len(vals)} sat100={sat:.1f}% mean={sum(vals)/len(vals):.1f} AUC={a:.3f}")

    # Standard prompt performance on the same subset
    std = {}
    for l in open(f"{BASE}/out/scores_deepseek.jsonl"):
        r = json.loads(l)
        if r["score"] is not None: std[r["key"]] = r["score"]
    sub_rows = [t for t in sub if t["key"] in std]
    pos = [std[t["key"]] for t in sub_rows if t["candidate"] == gold[t["id"]]]
    neg = [std[t["key"]] for t in sub_rows if t["candidate"] != gold[t["id"]]]
    w = sum(1 if p > n else .5 if p == n else 0 for p in pos for n in neg)
    vals = [std[t["key"]] for t in sub_rows]
    print(f"variant standard(same subset): n={len(vals)} sat100={100*sum(1 for v in vals if v==100)/len(vals):.1f}% "
          f"mean={sum(vals)/len(vals):.1f} AUC={w/(len(pos)*len(neg)):.3f}")
