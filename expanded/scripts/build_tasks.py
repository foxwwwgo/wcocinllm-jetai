import os
BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# -*- coding: utf-8 -*-
"""Construct cross-family scoring tasks: item text + candidate answer -> to-be-scored."""
import json, os
import pandas as pd

DATA = f"{BASE}/data"
OUT  = f"{BASE}/data/tasks.jsonl"

ledger = json.load(open(f"{DATA}/certificate_pool_ledger.json"))
gsm = pd.read_parquet("/tmp/t.parquet")
csqa = pd.read_parquet(f"{DATA}/csqa_val.parquet")

def gsm_gold(s):
    return s.split("####")[-1].strip().replace(",", "")

gsm_q = {i: (gsm.iloc[i]["question"], gsm_gold(gsm.iloc[i]["answer"])) for i in range(len(gsm))}
cs_meta = {}
for _, r in csqa.iterrows():
    labels = list(r["choices"]["label"]); texts = list(r["choices"]["text"])
    opts = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
    cs_meta[r["id"]] = (r["question"], opts, r["answerKey"])

def parse_id(rid):
    if rid.startswith("gsm8k-"):
        return "gsm8k", int(rid.rsplit("-", 1)[1])
    if rid.startswith("csqa-"):
        return "csqa", rid[5:]
    raise ValueError(rid)

tasks, missing, goldbad = [], 0, 0
for rec in ledger:
    task, key = parse_id(rec["id"])
    if task == "gsm8k":
        if key not in gsm_q: missing += 1; continue
        q, gold = gsm_q[key]; opts = None
    else:
        if key not in cs_meta: missing += 1; continue
        q, opts, gold = cs_meta[key]
    if str(gold) != str(rec["gold"]): goldbad += 1
    for cls in rec["counts"]:
        tasks.append(dict(key=f"{rec['id']}|{rec['model']}|{cls}",
                          id=rec["id"], model=rec["model"], task=task,
                          question=q, options=opts, candidate=cls, gold=str(rec["gold"])))

with open(OUT, "w") as f:
    for t in tasks: f.write(json.dumps(t, ensure_ascii=False) + "\n")
print("tasks", len(tasks), "missing_q", missing, "gold_mismatch", goldbad)
