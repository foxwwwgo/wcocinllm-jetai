import os
BASE = os.environ.get("EXP_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, pandas as pd
DATA=f"{BASE}/data"
gsm=pd.read_parquet("/tmp/t.parquet"); cs=pd.read_parquet(f"{DATA}/csqa_val.parquet")
def gg(s): return s.split("####")[-1].strip().replace(",","")
G=[{"id":f"gsm8k-test-{i}","question":gsm.iloc[i]["question"],"gold":gg(gsm.iloc[i]["answer"])} for i in range(min(1200,len(gsm)))]
C=[]
for _,r in cs.head(1200).iterrows():
    lab=list(r["choices"]["label"]); txt=list(r["choices"]["text"])
    C.append({"id":"csqa-"+r["id"],"question":r["question"],
              "options":"\n".join(f"{l}. {t}" for l,t in zip(lab,txt)),"gold":r["answerKey"]})
json.dump(G,open(f"{DATA}/gsm_q1200.json","w"),ensure_ascii=False)
json.dump(C,open(f"{DATA}/csqa_q1200.json","w"),ensure_ascii=False)
print(len(G),len(C))
