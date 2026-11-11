import os
# -*- coding: utf-8 -*-
"""Conformal-coverage review and design effect on the scale-up set.

1) 5-fold cross-fit by question; check whether empirical coverage at alpha=0.05/0.10/0.20 reaches the nominal level
2) Same statistic (fraction of gold falling within the candidate pool), using record-level i.i.d. bootstrap vs
   question-clustered bootstrap for resampling, compare 95% interval widths, test whether the narrowing ratio
   drops from ~16% to ~9% when m goes from 3 to 2
"""
import json, random, math, collections

BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_scores(tag, prefix="scores_"):
    d = {}
    for l in open(f"{BASE}/out/{prefix}{tag}.jsonl"):
        r = json.loads(l)
        if r["score"] is not None:
            d[r["key"]] = r["score"] / 100.0
    return d


def quant(v, a):
    v = sorted(v)
    n = len(v)
    k = math.ceil((n + 1) * (1 - a))
    return v[k - 1] if k <= n else None


def coverage(pools, scoremap, a, seed=20261108):
    """Whether the conformal score of the gold candidate falls within the quantile threshold."""
    qs = sorted({p["qid"] for p in pools})
    rng = random.Random(seed)
    q = qs[:]
    rng.shuffle(q)
    folds = [set(q[i::5]) for i in range(5)]
    hits = tot = 0
    for f in folds:
        cal = [p for p in pools if p["qid"] not in f]
        test = [p for p in pools if p["qid"] in f]
        vals = []
        for p in cal:
            gk = f"{p['qid']}|{p['model']}|{p['gold']}"
            if gk in scoremap:
                vals.append(-scoremap[gk])
        t = quant(vals, a)
        if t is None:
            continue
        for p in test:
            gk = f"{p['qid']}|{p['model']}|{p['gold']}"
            if gk not in scoremap:
                continue
            tot += 1
            if -scoremap[gk] <= t:
                hits += 1
    return hits, tot


def width_compare(groups, draws=2000, seed=7):
    """Clustered bootstrap vs record-level bootstrap: 95% interval width of the same statistic."""
    rng = random.Random(seed)
    keys = list(groups)
    cl = []
    for _ in range(draws):
        s = [rng.choice(keys) for _ in keys]
        lab = [y for k in s for y in groups[k]]
        cl.append(sum(lab) / len(lab))
    recs = [y for k in keys for y in groups[k]]
    ii = []
    for _ in range(draws):
        s = [rng.choice(recs) for _ in recs]
        ii.append(sum(s) / len(s))
    cl.sort()
    ii.sort()
    w_cl = cl[int(.975 * draws)] - cl[int(.025 * draws)]
    w_ii = ii[int(.975 * draws)] - ii[int(.025 * draws)]
    return w_cl, w_ii, 1 - w_ii / w_cl


pools = json.load(open(f"{BASE}/out/pools_expanded.json"))
led = json.load(open(f"{BASE}/data/certificate_pool_ledger.json"))
QW = load_scores("qwen14b", "xscores_")

SELF = {f"{p['qid']}|{p['model']}|{p['gold']}": p["scores"][p["gold"]]
        for p in pools if p["gold"] in p["scores"]}

print("== Scale-up set conformal coverage (2,400 questions x 2 models)")
print(f"{'signal':14s} {'alpha':>6s} {'coverage':>9s} {'n':>6s}")
for name, sc in (("self", SELF), ("qwen judge", QW)):
    for a in (0.05, 0.10, 0.20):
        h, t = coverage(pools, sc, a)
        print(f"{name:14s} {a:6.2f} {h/t:9.3f} {t:6d}")

print("\n== Design effect (fraction of gold within pool)")
for label, ps in (("scale-up m=2", pools), ("orig 600 m=3", led)):
    gr = collections.defaultdict(list)
    for p in ps:
        gr[p.get("qid") or p["id"]].append(1 if p["gold"] in p["counts"] else 0)
    w_cl, w_ii, red = width_compare(gr)
    m = sum(len(v) for v in gr.values()) / len(gr)
    print(f"{label:12s} clustered width {w_cl:.4f}  record-level {w_ii:.4f}  narrowing {100*red:.1f}%  "
          f"(avg {m:.1f} per question)")
