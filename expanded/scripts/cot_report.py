import os
# -*- coding: utf-8 -*-
import json, statistics, collections
BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
rows=[json.loads(l) for l in open(f"{BASE}/out/cot_ablation.jsonl")]
by=collections.defaultdict(list)
for r in rows: by[r["mode"]].append(r)
print(f"{'mode':11s} {'n':>4s} {'parse%':>7s} {'acc(parsed)':>12s} {'meanConf(all)':>14s} {'meanConf(parsed)':>16s} {'meanTok':>8s}")
for m in ("full","restricted"):
    rs=by[m]; n=len(rs)
    parsed=[r for r in rs if r["answer"] is not None]
    corr=[r for r in parsed if r["answer"]==r["gold"]]
    conf=[r["confidence"] for r in rs if r["confidence"] is not None]
    confp=[r["confidence"] for r in parsed if r["confidence"] is not None]
    tok=[r["tokens"] for r in rs if r["tokens"]]
    print(f"{m:11s} {n:4d} {len(parsed)/max(n,1)*100:6.1f}% {len(corr)/max(len(parsed),1)*100:11.1f}% "
          f"{statistics.mean(conf):13.1f} {statistics.mean(confp):15.1f} {statistics.mean(tok):7.0f}")
