import os
# -*- coding: utf-8 -*-
"""加入 DeepSeek non-thinking API 作为跨族第五信号（评判 + 独立解题）。

报告三类量：
  1) 池级判决指标（与 final.py 同规则）：改判数、纠正 C、损害 H、eta、eta_adj、聚类自助 CI、符号检验 p
  2) 候选级判别力：对全部（记录, 候选类）打分任务算 AUC（答案是否金标）
  3) 饱和率：打分为 100 的候选占比——解释为何高分评判器无法改判

eta_adj = C / (C_harm * H)，C_harm = 2.0。
"""
import json, random, collections, math

BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHARM = 2.0
ledger = json.load(open(f"{BASE}/data/certificate_pool_ledger.json"))
gold = {}
for r in ledger:
    for c in r["counts"]:
        gold[f"{r['id']}|{r['model']}|{c}"] = (c == r["gold"])


def ls(tag):
    d = {}
    for l in open(f"{BASE}/out/scores_{tag}.jsonl"):
        r = json.loads(l)
        if r["score"] is not None:
            d[r["key"]] = r["score"]
    return d


QW, PHI, DS = ls("qwen14b"), ls("phi"), ls("deepseek")
DSR = ls("dsreasoner")


def load_solve(tag):
    d = {}
    for l in open(f"{BASE}/out/solve_{tag}.jsonl"):
        r = json.loads(l)
        if r.get("answer") is not None:
            d[r["id"]] = r["answer"]
    return d


SOL_QW, SOL_DS = load_solve("qwen14b"), load_solve("deepseek1500")
SOL_DSR = load_solve("dsreasoner")


def norm(a):
    return None if a is None else str(a).strip().upper().replace(" ", "")


def selfs(r):
    a = collections.defaultdict(list)
    for x, s in zip(r["answers"], r["scores"]):
        if x is None:
            continue
        a[x].append(s)
    return {k: sum(v) / len(v) / 100.0 for k, v in a.items()}


def ext(r, t):
    return {c: t[f"{r['id']}|{r['model']}|{c}"] / 100.0 for c in r["counts"]
            if f"{r['id']}|{r['model']}|{c}" in t}


def auc(pos, neg):
    if not pos or not neg:
        return None
    w = sum(1 if p > n else .5 if p == n else 0 for p in pos for n in neg)
    return w / (len(pos) * len(neg))


def cand_auc(scored, keys):
    """候选级 AUC：分数对「该候选是否为金标」的判别力。"""
    pos = [scored[k] for k in keys if k in scored and gold.get(k)]
    neg = [scored[k] for k in keys if k in scored and not gold.get(k)]
    return auc(pos, neg)


def sign_p(c, h):
    n = c + h
    if n == 0:
        return None
    k = min(c, h)
    return min(1.0, sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2)


def evalrule(rule):
    sw = []
    for r in ledger:
        if r["class_count"] < 2:
            continue
        out = rule(r)
        if out is None:
            continue
        kmax, d = out
        if d > 0:
            sw.append((r["id"], 1 if kmax == r["gold"] else (-1 if r["majority"] == r["gold"] else 0)))
    C = sum(1 for _, y in sw if y == 1)
    H = sum(1 for _, y in sw if y == -1)
    byq = collections.defaultdict(list)
    for q, y in sw:
        byq[q].append(y)
    keys = list(byq)
    rng = random.Random(7)
    est = []
    for _ in range(5000):
        s = [rng.choice(keys) for _ in keys]
        c = sum(1 for k in s for y in byq[k] if y == 1)
        h = sum(1 for k in s for y in byq[k] if y == -1)
        if h:
            est.append(c / h)
    est.sort()
    return dict(sw=len(sw), C=C, H=H, W=len(sw) - C - H,
                eta=(C / H if H else None),
                eta_adj=(C / (CHARM * H) if H else None),
                ci=([round(est[int(.025 * len(est))], 2), round(est[int(.975 * len(est))], 2)] if est else None),
                p=sign_p(C, H))


def mk_sig(sfn):
    def rule(r):
        s = sfn(r)
        if not s or r["majority"] not in s:
            return None
        alt = [(v, k) for k, v in s.items() if k != r["majority"]]
        if not alt:
            return None
        smax, kmax = max(alt)
        return kmax, smax - s[r["majority"]]
    return rule


def mk_solver(sol):
    def rule(r):
        a = lambda c: norm(sol.get(r["id"])) == norm(c)
        s = {c: (1.0 if a(c) else 0.0) for c in r["counts"]}
        return mk_sig(lambda _: s)(r)
    return rule


def mk_conj(s1, s2):
    def rule(r):
        a1 = lambda c: norm(s1.get(r["id"])) == norm(c)
        a2 = lambda c: norm(s2.get(r["id"])) == norm(c)
        alt = [k for k in r["counts"] if k != r["majority"] and a1(k) and a2(k)
               and not a1(r["majority"]) and not a2(r["majority"])]
        return (alt[0], 1.0) if alt else None
    return rule


def mk_judge_solver(judge, sol):
    def rule(r):
        q = ext(r, judge)
        a = lambda c: norm(sol.get(r["id"])) == norm(c)
        if not q or r["majority"] not in q:
            return None
        alt = [(v, k) for k, v in q.items() if k != r["majority"]]
        if not alt:
            return None
        smax, kmax = max(alt)
        if not (smax > q[r["majority"]] and a(kmax) and not a(r["majority"])):
            return None
        return kmax, 1.0
    return rule


allkeys = list(gold)
rules = [
    ("self-confidence", mk_sig(selfs), None),
    ("Phi-3-mini judge", mk_sig(lambda r: ext(r, PHI)), PHI),
    ("Qwen2.5-14B judge", mk_sig(lambda r: ext(r, QW)), QW),
    ("Qwen2.5-14B solver", mk_solver(SOL_QW), None),
    ("DeepSeek judge (non-thinking)", mk_sig(lambda r: ext(r, DS)), DS),
    ("DeepSeek solver (non-thinking)", mk_solver(SOL_DS), None),
    ("DeepSeek judge (thinking)", mk_sig(lambda r: ext(r, DSR)), DSR),
    ("DeepSeek solver (thinking)", mk_solver(SOL_DSR), None),
    ("Qwen solver & DeepSeek solver (non-thinking)", mk_conj(SOL_QW, SOL_DS), None),
    ("Qwen solver & DeepSeek solver (thinking)", mk_conj(SOL_QW, SOL_DSR), None),
]

print(f"{'signal':46s} {'AUC*':>5s} {'candAUC':>7s} {'sat%':>5s} {'sw':>4s} {'C':>3s} {'H':>3s} "
      f"{'eta':>5s} {'eta_adj':>7s}  {'95%CI':>13s} {'p':>6s}")
rows = []
for name, rule, sc in rules:
    r = evalrule(rule)
    ca = None
    sat = None
    if sc is not None:
        sub = {k: v for k, v in sc.items()}
        ca = cand_auc(sub, allkeys)
        vals = [v for v in sub.values()]
        sat = 100.0 * sum(1 for v in vals if v == 100) / len(vals) if vals else None
    rows.append(dict(name=name, cand_auc=ca, sat=sat, **r))
    ca_s = "-" if ca is None else f"{ca:.3f}"
    sat_s = "-" if sat is None else f"{sat:.1f}"
    e = "-" if r["eta"] is None else f"{r['eta']:.2f}"
    ea = "-" if r["eta_adj"] is None else f"{r['eta_adj']:.2f}"
    p = "-" if r["p"] is None else f"{r['p']:.3f}"
    print(f"{name:46s} {'-':>5s} {ca_s:>7s} {sat_s:>5s} {r['sw']:4d} {r['C']:3d} {r['H']:3d} "
          f"{e:>5s} {ea:>7s}  {str(r['ci']):>13s} {p:>6s}")

json.dump({"charm": CHARM, "signals": rows}, open(f"{BASE}/out/deepseek_summary.json", "w"),
          ensure_ascii=False, indent=1, default=str)
print("\nwritten out/deepseek_summary.json")
