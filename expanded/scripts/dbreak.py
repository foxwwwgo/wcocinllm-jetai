import os
# -*- coding: utf-8 -*-
"""Split each signal's discriminative power and correct-fix vs harm trade-off by task type
(GSM8K arithmetic / CommonsenseQA multiple-choice).

Motivation: on the full sample the DeepSeek non-thinking judge's candidate-level AUC is only 0.505,
but 0.744 on the GSM subset and 96% of CSQA candidates are scored 100. The behavioral difference
between the two item types is a reportable result, not noise.
"""
import json, collections, math

BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHARM = 2.0
ledger = json.load(open(f"{BASE}/data/certificate_pool_ledger.json"))


def ls(tag):
    d = {}
    for l in open(f"{BASE}/out/scores_{tag}.jsonl"):
        r = json.loads(l)
        if r["score"] is not None:
            d[r["key"]] = r["score"]
    return d


def load_solve(tag):
    d = {}
    for l in open(f"{BASE}/out/solve_{tag}.jsonl"):
        r = json.loads(l)
        if r.get("answer") is not None:
            d[r["id"]] = r["answer"]
    return d


QW, PHI, DS = ls("qwen14b"), ls("phi"), ls("deepseek")
DSR = ls("dsreasoner") if __import__("os").path.exists(f"{BASE}/out/scores_dsreasoner.jsonl") else {}
SOL_QW, SOL_DS = load_solve("qwen14b"), load_solve("deepseek1500")
SOL_DSR = load_solve("dsreasoner") if __import__("os").path.exists(f"{BASE}/out/solve_dsreasoner.jsonl") else {}
gold = {r["id"]: r["gold"] for r in ledger}


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


def scored_map(t):
    return {f"{r['id']}|{r['model']}|{c}": t[f"{r['id']}|{r['model']}|{c}"]
            for r in ledger for c in r["counts"] if f"{r['id']}|{r['model']}|{c}" in t}


def auc(pos, neg):
    if not pos or not neg:
        return None
    w = sum(1 if p > n else .5 if p == n else 0 for p in pos for n in neg)
    return w / (len(pos) * len(neg))


def rule_metrics(rule, ids):
    sw = []
    for r in ledger:
        if r["id"] not in ids or r["class_count"] < 2:
            continue
        out = rule(r)
        if out is None:
            continue
        kmax, d = out
        if d > 0:
            sw.append(1 if kmax == r["gold"] else (-1 if r["majority"] == r["gold"] else 0))
    C = sum(1 for y in sw if y == 1)
    H = sum(1 for y in sw if y == -1)
    return len(sw), C, H, (C / H if H else None), (C / (CHARM * H) if H else None)


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
        a = norm(sol.get(r["id"]))
        s = {c: (1.0 if a == norm(c) else 0.0) for c in r["counts"]}
        return mk_sig(lambda _: s)(r)
    return rule


def mk_conj(s1, s2):
    def rule(r):
        a1, a2 = norm(s1.get(r["id"])), norm(s2.get(r["id"]))
        alt = [k for k in r["counts"] if k != r["majority"]
               and a1 == norm(k) and a2 == norm(k)
               and a1 != norm(r["majority"]) and a2 != norm(r["majority"])]
        return (alt[0], 1.0) if alt else None
    return rule


SIGNALS = [
    ("self-confidence", mk_sig(selfs), None),
    ("Phi-3-mini judge", mk_sig(lambda r: ext(r, PHI)), PHI),
    ("Qwen2.5-14B judge", mk_sig(lambda r: ext(r, QW)), QW),
    ("Qwen2.5-14B solver", mk_solver(SOL_QW), None),
    ("DeepSeek judge (non-thinking)", mk_sig(lambda r: ext(r, DS)), DS),
    ("DeepSeek solver (non-thinking)", mk_solver(SOL_DS), None),
    ("DeepSeek judge (thinking)", mk_sig(lambda r: ext(r, DSR)), DSR or None),
    ("DeepSeek solver (thinking)", mk_solver(SOL_DSR), None),
    ("Qwen solve & DeepSeek solve (non-thinking)", mk_conj(SOL_QW, SOL_DS), None),
    ("Qwen solve & DeepSeek solve (thinking)", mk_conj(SOL_QW, SOL_DSR), None),
]

GROUPS = {
    "ALL": lambda i: True,
    "GSM8K": lambda i: i.startswith("gsm"),
    "CSQA": lambda i: i.startswith("csqa"),
}

for gname, pred in GROUPS.items():
    ids = {r["id"] for r in ledger if pred(r["id"])}
    print(f"\n=== {gname}  (pools={len(ids)})")
    print(f"{'signal':30s} {'candAUC':>7s} {'sat%':>5s} {'sw':>4s} {'C':>3s} {'H':>3s} {'eta':>5s} {'eta_adj':>7s}")
    for name, rule, sc in SIGNALS:
        sw, C, H, eta, ea = rule_metrics(rule, ids)
        ca = sat = None
        if sc is not None:
            m = scored_map(sc)
            pos, neg, vals = [], [], []
            for r in ledger:
                if r["id"] not in ids:
                    continue
                for c in r["counts"]:
                    k = f"{r['id']}|{r['model']}|{c}"
                    if k not in m:
                        continue
                    v = m[k]
                    vals.append(v)
                    (pos if str(c).strip().upper() == str(r["gold"]).strip().upper() else neg).append(v)
            ca = auc(pos, neg)
            sat = 100 * sum(1 for v in vals if v == 100) / len(vals) if vals else None
        ca_s = "-" if ca is None else f"{ca:.3f}"
        sat_s = "-" if sat is None else f"{sat:.1f}"
        e = "-" if eta is None else f"{eta:.2f}"
        ea_s = "-" if ea is None else f"{ea:.2f}"
        print(f"{name:30s} {ca_s:>7s} {sat_s:>5s} {sw:4d} {C:3d} {H:3d} {e:>5s} {ea_s:>7s}")
