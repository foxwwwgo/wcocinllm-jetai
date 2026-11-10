import os
# -*- coding: utf-8 -*-
"""CoT 消融的校准分析：准确率、平均置信、Brier、ECE、过度自信缺口，按题聚类自助。"""
import json, random, collections, statistics

BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
rows = [json.loads(l) for l in open(f"{BASE}/out/cot_ablation.jsonl")]
for r in rows:
    r["ok"] = 1 if (r["answer"] is not None and str(r["answer"]).strip().upper()
                    == str(r["gold"]).strip().upper()) else 0
    r["c"] = (r["confidence"] or 0) / 100.0


def ece(items, bins=10):
    tot = len(items)
    s = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sub = [r for r in items if (lo < r["c"] <= hi) or (b == 0 and r["c"] == 0)]
        if not sub:
            continue
        s += len(sub) / tot * abs(sum(r["ok"] for r in sub) / len(sub) - sum(r["c"] for r in sub) / len(sub))
    return s


def stats(items):
    n = len(items)
    acc = sum(r["ok"] for r in items) / n
    conf = sum(r["c"] for r in items) / n
    brier = sum((r["c"] - r["ok"]) ** 2 for r in items) / n
    return dict(n=n, acc=acc, conf=conf, gap=conf - acc, brier=brier, ece=ece(items))


modes = sorted({r["mode"] for r in rows})
print(f"{'mode':12s} {'n':>4s} {'acc':>6s} {'conf':>6s} {'gap':>6s} {'Brier':>6s} {'ECE':>6s} {'tokens':>7s}")
out = {}
for m in modes:
    it = [r for r in rows if r["mode"] == m]
    st = stats(it)
    st["tokens"] = statistics.mean(r["tokens"] for r in it)
    out[m] = st
    print(f"{m:12s} {st['n']:4d} {st['acc']:6.3f} {st['conf']:6.3f} {st['gap']:+6.3f} "
          f"{st['brier']:6.3f} {st['ece']:6.3f} {st['tokens']:7.0f}")

# 按题聚类自助：gap 差值、Brier 差值
by_q = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    by_q[r["qid"]][r["mode"]].append(r)
qs = [q for q in by_q if len(by_q[q]) == 2]
rng = random.Random(20261108)
dgap, dbrier, dacc = [], [], []
for _ in range(5000):
    s = [rng.choice(qs) for _ in qs]
    a = [r for q in s for r in by_q[q]["full"]]
    b = [r for q in s for r in by_q[q]["restricted"]]
    sa, sb = stats(a), stats(b)
    dgap.append(sa["gap"] - sb["gap"])
    dbrier.append(sa["brier"] - sb["brier"])
    dacc.append(sa["acc"] - sb["acc"])
for name, v in (("gap(full-restricted)", dgap), ("Brier(full-restricted)", dbrier), ("acc(full-restricted)", dacc)):
    v.sort()
    print(f"{name:24s} 点估计 {statistics.mean(v):+.3f}  95%CI [{v[int(.025*len(v))]:+.3f}, {v[int(.975*len(v))]:+.3f}]")
