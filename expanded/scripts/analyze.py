import os
# -*- coding: utf-8 -*-
"""Compare three signals across 209 competition pools: self-confidence / Phi-3-mini / Qwen2.5-14B.
Decision unit: whether to reassign the majority answer to the highest-scoring minority cluster."""
import json, math, random, collections, statistics, sys

BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ledger = json.load(open(f"{BASE}/data/certificate_pool_ledger.json"))

def load_scores(tag):
    d = {}
    try:
        for l in open(f"{BASE}/out/scores_{tag}.jsonl"):
            r = json.loads(l)
            if r["score"] is not None:
                d[r["key"]] = r["score"]
    except FileNotFoundError:
        pass
    return d

PHI = load_scores("phi"); QW = load_scores("qwen14b")

def self_scores(rec):
    agg = collections.defaultdict(list)
    for a, s in zip(rec["answers"], rec["scores"]):
        if a is None: continue
        agg[a].append(s)
    return {k: sum(v) / len(v) / 100.0 for k, v in agg.items()}

def ext_scores(rec, table):
    out = {}
    for c in rec["counts"]:
        v = table.get(f"{rec['id']}|{rec['model']}|{c}")
        if v is not None:
            out[c] = v / 100.0
    return out

def qhat(vals, a):
    v = sorted(vals); n = len(v); k = math.ceil((n + 1) * (1 - a))
    return v[k - 1] if k <= n else float("inf")

def auc(pos, neg):
    if not pos or not neg: return float("nan")
    w = sum(1 if p > n else .5 if p == n else 0 for p in pos for n in neg)
    return w / (len(pos) * len(neg))

SIG = {"self": self_scores,
       "phi3":  lambda r: ext_scores(r, PHI),
       "qwen14b": lambda r: ext_scores(r, QW)}

print("== Signal discriminative power (209 competition pools, rewrite correct or not)")
for name, fn in SIG.items():
    pos, neg, n_ok = [], [], 0
    for r in ledger:
        if r["class_count"] < 2: continue
        s = fn(r)
        if r["majority"] not in s: continue
        alt = [(v, k) for k, v in s.items() if k != r["majority"]]
        if not alt: continue
        smax, kmax = max(alt); smaj = s[r["majority"]]
        n_ok += 1
        (pos if kmax == r["gold"] else neg).append(smax - smaj)
    print(f"  {name:8s} n={n_ok} AUC(margin)={auc(pos,neg):.3f}  correct={len(pos)} wrong={len(neg)}")

print("\n== Threshold rule: alternative cluster score > majority cluster -> reassign")
for name, fn in SIG.items():
    C = H = W = n = 0
    for r in ledger:
        if r["class_count"] < 2: continue
        s = fn(r)
        if r["majority"] not in s: continue
        alt = [(v, k) for k, v in s.items() if k != r["majority"]]
        if not alt: continue
        smax, kmax = max(alt)
        if smax > s[r["majority"]]:
            n += 1
            if kmax == r["gold"]: C += 1
            elif r["majority"] == r["gold"]: H += 1
            else: W += 1
    print(f"  {name:8s} switches={n} C={C} H={H} W={W} eta={C/H if H else float('inf'):.2f}")
