import os
# -*- coding: utf-8 -*-
"""生成器参数量阶梯：granite-2B / Phi-3.8B / Qwen-14B，同一 2,400 题、7 条路径、同一套
建池与自评置信度改判规则。

只读 gen_*_1200.jsonl，不重写 pools_expanded.json，因此不影响 §5.8 现有的两模型结果。
"""
import json, glob, os, collections, math

BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHARM = 2.0

Q = {}
for f in ("gsm_q1200.json", "csqa_q1200.json"):
    for r in json.load(open(f"{BASE}/data/{f}")):
        Q[r["id"]] = r["gold"]

MODELS = [("granite", "2B"), ("phi", "3.8B"), ("qwen14b", "14B")]


def load_model(tag):
    rows = collections.defaultdict(list)
    for l in open(f"{BASE}/out/gen_{tag}_1200.jsonl"):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r["answer"] is not None:
            rows[r["qid"]].append(r)
    return rows


def build_pools(rows):
    pools = []
    for qid, rs in rows.items():
        cnt = collections.Counter(r["answer"] for r in rs)
        if not cnt:
            continue
        first = {}
        for i, r in enumerate(rs):
            first.setdefault(r["answer"], i)
        maj = max(cnt.items(), key=lambda kv: (kv[1], -first[kv[0]]))[0]
        conf = collections.defaultdict(list)
        for r in rs:
            if r["confidence"] is not None:
                conf[r["answer"]].append(r["confidence"] / 100.0)
        pools.append(dict(qid=qid, majority=maj, counts=dict(cnt), gold=Q[qid],
                          scores={k: sum(v) / len(v) for k, v in conf.items()}))
    return pools


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    w = sum(1 if a > b else .5 if a == b else 0 for a in pos for b in neg)
    return w / (len(pos) * len(neg))


def evaluate(pools):
    comp = [p for p in pools if len(p["counts"]) >= 2]
    majok = sum(1 for p in comp if p["majority"] == p["gold"])
    C = H = sw = 0
    for p in comp:
        s = p["scores"]
        if not s or p["majority"] not in s:
            continue
        alt = [(v, k) for k, v in s.items() if k != p["majority"]]
        if not alt:
            continue
        smax, kmax = max(alt)
        if smax > s[p["majority"]]:
            sw += 1
            if kmax == p["gold"]:
                C += 1
            elif p["majority"] == p["gold"]:
                H += 1
    pos, neg = [], []
    for p in pools:
        for c, v in p["scores"].items():
            (pos if str(c).strip().upper() == str(p["gold"]).strip().upper() else neg).append(v)
    return dict(n=len(pools), comp=len(comp), majacc=100 * majok / max(len(comp), 1),
                sw=sw, C=C, H=H, eta=(C / H if H else float("nan")),
                eta_adj=(C / (CHARM * H) if H else float("nan")),
                auc=auc(pos, neg))


print(f"{'generator':10s} {'size':>5s} {'pools':>6s} {'competing':>9s} {'majority acc':>12s} "
      f"{'sw':>4s} {'C':>4s} {'H':>4s} {'eta':>6s} {'eta_adj':>7s} {'AUC':>6s}")
out = []
for tag, size in MODELS:
    p = f"{BASE}/out/gen_{tag}_1200.jsonl"
    if not os.path.exists(p):
        print(f"{tag:10s} {size:>5s}   (尚未生成)")
        continue
    r = evaluate(build_pools(load_model(tag)))
    out.append(dict(model=tag, size=size, **r))
    print(f"{tag:10s} {size:>5s} {r['n']:6d} {r['comp']:9d} {r['majacc']:11.1f}% "
          f"{r['sw']:4d} {r['C']:4d} {r['H']:4d} {r['eta']:6.2f} {r['eta_adj']:7.2f} {r['auc']:6.3f}")

json.dump(out, open(f"{BASE}/out/ladder.json", "w"), ensure_ascii=False, indent=1, default=str)
print("\nwritten out/ladder.json")
