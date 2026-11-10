import os
# -*- coding: utf-8 -*-
"""回答三个复核问题：
  A. α=0.10 的覆盖率点估计与区间半宽（原 1,732 条 vs 扩容 ~4,200 条），按题聚类自助
  B. N=3/5/7 路径数扫描（在扩容的 7 条路径上取前 N 条重算）
  C. 生成器参数量与自评置信度（granite 2B / phi 3.8B）
"""
import json, random, math, collections, glob

BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------- A. 覆盖率 ----------
def quant(v, a):
    v = sorted(v)
    k = math.ceil((len(v) + 1) * (1 - a))
    return v[k - 1] if k <= len(v) else None


def covered_flags(gold_score, qids, a, seed=20261108):
    """按题 5 折交叉拟合，返回每题覆盖指示（记录级展开）。"""
    uniq = sorted(set(qids))
    rng = random.Random(seed)
    q = uniq[:]
    rng.shuffle(q)
    folds = [set(q[i::5]) for i in range(5)]
    flags = collections.defaultdict(list)   # qid -> [0/1]
    for f in folds:
        cal = [s for s, qq in gold_score if qq not in f]
        t = quant([-s for s in cal], a)
        if t is None:
            continue
        for s, qq in gold_score:
            if qq in f:
                flags[qq].append(1 if -s <= t else 0)
    return flags


def cluster_ci(flags, draws=5000, seed=11):
    keys = list(flags)
    rng = random.Random(seed)
    est = []
    for _ in range(draws):
        s = [rng.choice(keys) for _ in keys]
        v = [y for k in s for y in flags[k]]
        est.append(sum(v) / len(v))
    est.sort()
    allv = [y for k in keys for y in flags[k]]
    return sum(allv) / len(allv), est[int(.025 * draws)], est[int(.975 * draws)], len(allv), len(keys)


SELF_LED = []
led = json.load(open(f"{BASE}/data/certificate_pool_ledger.json"))
for r in led:
    a = collections.defaultdict(list)
    for x, sc in zip(r["answers"], r["scores"]):
        if x is not None:
            a[x].append(sc)
    s = {k: sum(v) / len(v) / 100.0 for k, v in a.items()}
    if r["gold"] in s:
        SELF_LED.append((s[r["gold"]], r["id"]))

pools = json.load(open(f"{BASE}/out/pools_expanded.json"))
SELF_EXP = [(p["scores"][p["gold"]], p["qid"]) for p in pools if p["gold"] in p["scores"]]

QW = {}
for l in open(f"{BASE}/out/xscores_qwen14b.jsonl"):
    r = json.loads(l)
    if r["score"] is not None:
        QW[r["key"]] = r["score"] / 100.0
QW_EXP = [(QW[f"{p['qid']}|{p['model']}|{p['gold']}"], p["qid"]) for p in pools
          if f"{p['qid']}|{p['model']}|{p['gold']}" in QW]

print("== A. α=0.10 覆盖率（按题聚类自助 5000 次）")
print(f"{'数据集/信号':28s} {'n记录':>6s} {'n题':>5s} {'覆盖率':>7s} {'误覆盖':>7s} {'95%CI':>18s} {'半宽':>6s}")
for name, data in (("原 600 题 / 自评", SELF_LED), ("扩容 / 自评", SELF_EXP), ("扩容 / Qwen judge", QW_EXP)):
    fl = covered_flags(data, [q for _, q in data], 0.10)
    c, lo, hi, n, nq = cluster_ci(fl)
    print(f"{name:28s} {n:6d} {nq:5d} {c:7.3f} {1-c:7.3f} "
          f"[{lo:.3f},{hi:.3f}]".rjust(19) + f" {100*(hi-lo)/2:5.2f}pp")

# ---------- B. N 扫描 ----------
print("\n== B. N=3/5/7 路径扫描（扩容，取前 N 条路径重算）")
rows = collections.defaultdict(list)
for f in glob.glob(f"{BASE}/out/gen_*_1200.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if r["answer"] is not None:
            rows[(r["qid"], r["model"])].append(r)

Q = {}
for f in ("gsm_q1200.json", "csqa_q1200.json"):
    for r in json.load(open(f"{BASE}/data/{f}")):
        Q[r["id"]] = r["gold"]

print(f"{'N':>2s} {'池':>5s} {'竞争池':>6s} {'多数票正确':>9s} {'改判':>5s} {'C':>4s} {'H':>4s} {'eta':>6s}")
for N in (3, 5, 7):
    cnt_pool = comp = 0
    majok = 0
    C = H = sw = 0
    for (qid, model), rs in rows.items():
        rs = sorted(rs, key=lambda x: x["seat"])[:N]
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
        sc = {k: sum(v) / len(v) for k, v in conf.items()}
        cnt_pool += 1
        if len(cnt) >= 2:
            comp += 1
            if maj == Q[qid]:
                majok += 1
        if maj not in sc:
            continue
        alt = [(v, k) for k, v in sc.items() if k != maj]
        if not alt:
            continue
        smax, kmax = max(alt)
        if smax > sc[maj]:
            sw += 1
            if kmax == Q[qid]:
                C += 1
            elif maj == Q[qid]:
                H += 1
    print(f"{N:2d} {cnt_pool:5d} {comp:6d} {100*majok/max(comp,1):8.1f}% {sw:5d} {C:4d} {H:4d} "
          f"{(C/H if H else float('nan')):6.2f}")

# ---------- C. 参数量 ----------
print("\n== C. 生成器参数量与自评置信度（扩容，N=7）")
for model, size in (("granite", "2B"), ("phi", "3.8B")):
    C = H = sw = comp = majok = 0
    for (qid, m), rs in rows.items():
        if m != model:
            continue
        cnt = collections.Counter(r["answer"] for r in rs)
        if not cnt:
            continue
        first = {}
        for i, r in enumerate(rs):
            first.setdefault(r["answer"], i)
        maj = max(cnt.items(), key=lambda kv: (kv[1], -first[kv[0]]))[0]
        if len(cnt) >= 2:
            comp += 1
            if maj == Q[qid]:
                majok += 1
        conf = collections.defaultdict(list)
        for r in rs:
            if r["confidence"] is not None:
                conf[r["answer"]].append(r["confidence"] / 100.0)
        sc = {k: sum(v) / len(v) for k, v in conf.items()}
        if maj not in sc:
            continue
        alt = [(v, k) for k, v in sc.items() if k != maj]
        if not alt:
            continue
        smax, kmax = max(alt)
        if smax > sc[maj]:
            sw += 1
            if kmax == Q[qid]:
                C += 1
            elif maj == Q[qid]:
                H += 1
    print(f"  {model:8s} {size:5s} 池={len([1 for (q,m) in rows if m==model])} 竞争={comp} "
          f"多数票正确={100*majok/max(comp,1):.1f}% 改判={sw} C={C} H={H} "
          f"eta={(C/H if H else float('nan')):.2f}")
