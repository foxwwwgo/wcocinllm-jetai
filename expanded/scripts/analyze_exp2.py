import os
# -*- coding: utf-8 -*-
"""扩容数据（2,400 题 × 2 个生成模型）的信号评估，按全样本与题型分层。

修正 analyze_exp.py 的崩溃（C+H=0 时符号检验返回 None），并补上 eta_adj 与题型拆分。
"""
import json, random, collections, math, os

BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHARM = 2.0
pools = json.load(open(f"{BASE}/out/pools_expanded.json"))


def lscore(tag):
    d = {}
    p = f"{BASE}/out/xscores_{tag}.jsonl"
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l)
            if r["score"] is not None:
                d[r["key"]] = r["score"] / 100.0
    return d


def lsolve(tag):
    d = {}
    for tagp in (f"{BASE}/out/xsolve_{tag}.jsonl",):
        if os.path.exists(tagp):
            for l in open(tagp):
                r = json.loads(l)
                if r["answer"] is not None:
                    d[r["id"]] = r["answer"]
    return d


QW = lscore("qwen14b")
DSJ = {}
for _l in open(f"{BASE}/out/scores_deepseek.jsonl"):
    _r = json.loads(_l)
    if _r["score"] is not None:
        DSJ[_r["key"]] = _r["score"] / 100.0
QS = lsolve("qwen")
R1 = lsolve("r1")
DS = {}
for _l in open(f"{BASE}/out/solve_xdeepseek.jsonl"):
    _r = json.loads(_l)
    if _r.get("answer") is not None:
        DS[_r["id"]] = _r["answer"]


def norm(a):
    return None if a is None else str(a).strip().upper().replace(" ", "")


def ag(tab, p, c):
    return norm(tab.get(p["qid"])) == norm(c)


def judge(p):
    return {c: QW[f"{p['qid']}|{p['model']}|{c}"] for c in p["counts"]
            if f"{p['qid']}|{p['model']}|{c}" in QW}


def dsjudge(p):
    return {c: DSJ[f"{p['qid']}|{p['model']}|{c}"] for c in p["counts"]
            if f"{p['qid']}|{p['model']}|{c}" in DSJ}


def conj(tab1, tab2):
    def f(p):
        a1 = norm(tab1.get(p["qid"]))
        a2 = norm(tab2.get(p["qid"]))
        hits = [k for k in p["counts"] if k != p["majority"]
                and norm(k) == a1 and norm(k) == a2
                and a1 != norm(p["majority"]) and a2 != norm(p["majority"])]
        if not hits:
            return None
        return {c: (1.0 if c in hits else 0.0) for c in p["counts"]}
    return f


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    w = sum(1 if a > b else .5 if a == b else 0 for a in pos for b in neg)
    return w / (len(pos) * len(neg))


def sign_p(C, H):
    n = C + H
    if n == 0:
        return None
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(C, H) + 1)) / 2 ** n)


def ev(name, fn, sub, aucmode=True):
    sw = []
    for p in sub:
        if len(p["counts"]) < 2:
            continue
        s = fn(p)
        if not s or p["majority"] not in s:
            continue
        alt = [(v, k) for k, v in s.items() if k != p["majority"]]
        if not alt:
            continue
        smax, kmax = max(alt)
        if smax > s[p["majority"]]:
            sw.append((p["qid"], 1 if kmax == p["gold"] else (-1 if p["majority"] == p["gold"] else 0),
                       smax - s[p["majority"]], kmax == p["gold"]))
    C = sum(1 for _, y, _, _ in sw if y == 1)
    H = sum(1 for _, y, _, _ in sw if y == -1)
    pos = [d for _, _, d, g in sw if g]
    neg = [d for _, _, d, g in sw if not g]
    byq = collections.defaultdict(list)
    for q, y, _, _ in sw:
        byq[q].append(y)
    keys = list(byq)
    rng = random.Random(7)
    est = []
    for _ in range(3000):
        s = [rng.choice(keys) for _ in keys]
        c = sum(1 for k in s for y in byq[k] if y == 1)
        h = sum(1 for k in s for y in byq[k] if y == -1)
        if h:
            est.append(c / h)
    est.sort()
    p = sign_p(C, H)
    a = auc(pos, neg) if aucmode else float("nan")
    ci = f"[{est[int(.025*len(est))]:.2f},{est[int(.975*len(est))]:.2f}]" if est else "-"
    eta = C / H if H else float("nan")
    print(f"  {name:26s} AUC={a:5.3f} sw={len(sw):4d} C={C:4d} H={H:4d} "
          f"eta={eta:5.2f} eta_adj={eta/(CHARM*1.0) if H else float('nan'):5.2f} CI={ci:13s} "
          f"p={('-' if p is None else f'{p:.3f}')}")


def mk(fn):
    def f(p):
        s = fn(p)
        return s if s and p["majority"] in s else None
    return f


def solver(tab):
    return mk(lambda p: {c: (1.0 if ag(tab, p, c) else 0.0) for c in p["counts"]})


def twoof3(p):
    j = judge(p)
    if not j:
        return None
    return {c: float(int(ag(QS, p, c)) + int(ag(R1, p, c)) + int(j.get(c, 0) > 0.5)) for c in p["counts"]}


GROUPS = {
    "全样本": pools,
    "GSM8K": [p for p in pools if p["qid"].startswith("gsm")],
    "CSQA": [p for p in pools if p["qid"].startswith("csqa")],
    "granite 生成": [p for p in pools if p["model"] == "granite"],
    "phi 生成": [p for p in pools if p["model"] == "phi"],
}
for gname, sub in GROUPS.items():
    print(f"\n=== {gname}  pools={len(sub)}  competing={sum(1 for p in sub if len(p['counts'])>=2)}")
    ev("self-confidence", mk(lambda p: p["scores"]), sub)
    ev("qwen judge", mk(judge), sub)
    ev("deepseek judge", mk(dsjudge), sub)
    ev("qwen solver", solver(QS), sub, aucmode=False)
    ev("r1 solver", solver(R1), sub, aucmode=False)
    ev("deepseek solver", solver(DS), sub, aucmode=False)
    if DS:
        ev("qwen&deepseek solvers", mk(conj(QS, DS)), sub, aucmode=False)
    ev("2-of-3", mk(twoof3), sub, aucmode=False)
