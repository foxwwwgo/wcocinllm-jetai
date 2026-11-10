# -*- coding: utf-8 -*-
"""扩容集信号对比图：纠正与损害分组条形，右侧标注 eta_adj。

灰度与色盲友好：纠正为深色实心，损害为白底斜线，不依赖色相区分。
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "out")
CHARM = 2.0

pools = json.load(open(f"{OUT}/pools_expanded.json"))


def norm(a):
    return None if a is None else str(a).strip().upper().replace(" ", "")


def load_solve(name):
    d = {}
    p = f"{OUT}/{name}"
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l)
            if r.get("answer") is not None:
                d[r["id"]] = r["answer"]
    return d


def load_scores(name):
    d = {}
    p = f"{OUT}/{name}"
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l)
            if r.get("score") is not None:
                d[r["key"]] = r["score"] / 100.0
    return d


QW = load_scores("xscores_qwen14b.jsonl")
DSJ = load_scores("scores_deepseek.jsonl")
QS = load_solve("xsolve_qwen.jsonl")
R1 = load_solve("xsolve_r1.jsonl")
DS = load_solve("solve_xdeepseek.jsonl")


def judge(p, tab):
    return {c: tab[f"{p['qid']}|{p['model']}|{c}"] for c in p["counts"]
            if f"{p['qid']}|{p['model']}|{c}" in tab}


def solve(p, tab):
    a = norm(tab.get(p["qid"]))
    return {c: (1.0 if a == norm(c) else 0.0) for c in p["counts"]}


def conj(p, t1, t2):
    a1, a2 = norm(t1.get(p["qid"])), norm(t2.get(p["qid"]))
    hits = [k for k in p["counts"] if k != p["majority"] and norm(k) == a1 and norm(k) == a2
            and a1 != norm(p["majority"]) and a2 != norm(p["majority"])]
    if not hits:
        return None
    return {c: (1.0 if c in hits else 0.0) for c in p["counts"]}


def counts(fn):
    sw = c = h = 0
    for p in pools:
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
            sw += 1
            if kmax == p["gold"]:
                c += 1
            elif p["majority"] == p["gold"]:
                h += 1
    return sw, c, h


SIGS = [
    ("Self-confidence", lambda p: p["scores"]),
    ("Qwen judge", lambda p: judge(p, QW)),
    ("DeepSeek judge", lambda p: judge(p, DSJ)),
    ("Qwen solver", lambda p: solve(p, QS)),
    ("R1-Distill-7B solver", lambda p: solve(p, R1)),
    ("DeepSeek solver", lambda p: solve(p, DS)),
    ("Qwen $\\wedge$ DeepSeek", lambda p: conj(p, QS, DS)),
]

labels, Cs, Hs, ETA = [], [], [], []
print(f"{'signal':22s} {'sw':>5s} {'C':>5s} {'H':>5s} {'eta':>6s} {'eta_adj':>7s}")
for name, fn in SIGS:
    sw, c, h = counts(fn)
    eta = c / h if h else float("nan")
    print(f"{name:22s} {sw:5d} {c:5d} {h:5d} {eta:6.2f} {c/(CHARM*h) if h else float('nan'):7.2f}")
    labels.append(name)
    Cs.append(c)
    Hs.append(h)
    ETA.append(eta / CHARM if h else float("nan"))

fig, ax = plt.subplots(figsize=(7.0, 3.0))
y = range(len(labels))
ax.barh([i - 0.19 for i in y], Cs, height=0.36, color="#1b6e66",
        edgecolor="#0f3d38", linewidth=0.6, label="Corrections")
ax.barh([i + 0.19 for i in y], Hs, height=0.36, color="#f4f4f4",
        edgecolor="#2b2b2b", linewidth=0.8, hatch="////", label="Harms")
ax.set_yticks(list(y))
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Switches on 2,544 competing pools", fontsize=8)
ax.tick_params(axis="x", labelsize=8)
for i, (c, h, e) in enumerate(zip(Cs, Hs, ETA)):
    ax.text(max(c, h) + 12, i, f"$\\eta_{{\\mathrm{{adj}}}}$ = {e:.2f}", va="center", fontsize=7.5)
ax.set_xlim(0, max(max(Cs), max(Hs)) * 1.30)
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.25, linewidth=0.5)
fig.tight_layout()
outp = os.path.join(os.path.dirname(BASE), "fig5_expanded.pdf")
fig.savefig(outp, format="pdf", bbox_inches="tight")
fig.savefig("/tmp/fig5_gray.png", dpi=200, bbox_inches="tight")
print("written", outp)
