import os
# -*- coding: utf-8 -*-
"""Design effect: for the same set of reassignment events, the ratio of eta interval widths from
clustered bootstrap vs record-level i.i.d. bootstrap.

The paper's original claim is "i.i.d. bootstrap falsely narrows the interval by ~16% (0.55 vs 0.66)".
Here we recompute the same quantity on the scale-up set (m=2) to test whether the narrowing ratio
drops as m decreases.
"""
import json, random, collections

BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def switches_from_ledger(sig="self"):
    led = json.load(open(f"{BASE}/data/certificate_pool_ledger.json"))
    sw = []
    for r in led:
        if r["class_count"] < 2:
            continue
        a = collections.defaultdict(list)
        for x, sc in zip(r["answers"], r["scores"]):
            if x is not None:
                a[x].append(sc)
        s = {k: sum(v) / len(v) / 100.0 for k, v in a.items()}
        if not s or r["majority"] not in s:
            continue
        alt = [(v, k) for k, v in s.items() if k != r["majority"]]
        if not alt:
            continue
        smax, kmax = max(alt)
        if smax > s[r["majority"]]:
            sw.append((r["id"], 1 if kmax == r["gold"] else (-1 if r["majority"] == r["gold"] else 0)))
    return sw


def switches_from_pools(sig="self"):
    pools = json.load(open(f"{BASE}/out/pools_expanded.json"))
    sw = []
    for p in pools:
        if len(p["counts"]) < 2:
            continue
        s = p["scores"]
        if not s or p["majority"] not in s:
            continue
        alt = [(v, k) for k, v in s.items() if k != p["majority"]]
        if not alt:
            continue
        smax, kmax = max(alt)
        if smax > s[p["majority"]]:
            sw.append((p["qid"], 1 if kmax == p["gold"] else (-1 if p["majority"] == p["gold"] else 0)))
    return sw


def width(sw, draws=5000, seed=11):
    byq = collections.defaultdict(list)
    for q, y in sw:
        byq[q].append(y)
    keys = list(byq)
    rng = random.Random(seed)

    def eta(labels):
        c = sum(1 for y in labels if y == 1)
        h = sum(1 for y in labels if y == -1)
        return c / h if h else None

    cl = []
    for _ in range(draws):
        s = [rng.choice(keys) for _ in keys]
        v = eta([y for k in s for y in byq[k]])
        if v is not None:
            cl.append(v)
    recs = [y for _, y in sw if y != 0]
    ii = []
    for _ in range(draws):
        s = [rng.choice(recs) for _ in recs]
        v = eta(s)
        if v is not None:
            ii.append(v)

    def w(v):
        v.sort()
        return v[int(.975 * len(v))] - v[int(.025 * len(v))], v[int(.025 * len(v))], v[int(.975 * len(v))]

    wc, lc, uc = w(cl)
    wi, li, ui = w(ii)
    C = sum(1 for _, y in sw if y == 1)
    H = sum(1 for _, y in sw if y == -1)
    return C, H, wc, (lc, uc), wi, (li, ui)


print(f"{'dataset':16s} {'C':>4s} {'H':>4s} {'eta':>5s}  clustered width  record-level width  narrowing")
for label, sw in (("orig 600q m=3", switches_from_ledger()), ("scale-up m=2", switches_from_pools())):
    C, H, wc, ci, wi, ii = width(sw)
    print(f"{label:16s} {C:4d} {H:4d} {C/H:5.2f}  {wc:8.2f} [{ci[0]:.2f},{ci[1]:.2f}]  "
          f"{wi:8.2f} [{ii[0]:.2f},{ii[1]:.2f}]  {100*(1-wi/wc):5.1f}%")
