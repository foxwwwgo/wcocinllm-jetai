"""Frozen parsers extracted for offline response replay; no collector imports."""
import re, math

def parse_strict(job,r):
 if r['http_status']!=200 or r['finish_reason'] not in ('stop','eos'):return None
 text=r['raw_reply'].strip()
 if job['kind']=='confidence':
  if not re.fullmatch(r'(?:\d+(?:\.\d*)?|\.\d+)',text):return None
  z=float(text);return z if math.isfinite(z) and 0<=z<=100 else None
 pattern=r'Proposed answer:\s*\(\s*([A-E])\s*\)\.?' if job['task']=='commonsenseqa' else r'Proposed answer:\s*\(\s*([-+]?\d[\d,]*)\s*\)\.?'
 hits=re.findall(pattern,text,re.I)
 if len(hits)!=1:return None
 return hits[0].upper() if job['task']=='commonsenseqa' else str(int(hits[0].replace(',','')))

def parse_v71(j,r):
 value=parse_strict(j,r)
 if value is not None or j['kind']!='generation':return value
 if r['http_status']!=200 or r['finish_reason'] not in ('stop','eos'):return None
 lines=[s.strip() for s in r['raw_reply'].splitlines() if s.strip()]
 if not lines:return None
 line=re.sub(r'[*`]', '',lines[-1]);token=r'[A-E]' if j['task']=='commonsenseqa' else r'[-+]?\d[\d,]*'
 m=re.fullmatch(r'Proposed answer:\s*\(?\s*('+token+r')\s*\)?\.?',line,re.I)
 if not m:return None
 return m.group(1).upper() if j['task']=='commonsenseqa' else str(int(m.group(1).replace(',','')))

def parse(j,r):
 value=parse_v71(j,r)
 if value is not None or j['kind']!='generation':return value
 if r['http_status']!=200 or r['finish_reason'] not in ('stop','eos'):return None
 lines=[re.sub(r'[*`]', '',s.strip()) for s in r['raw_reply'].splitlines() if s.strip()]
 if not lines:return None
 token=r'[A-E]' if j['task']=='commonsenseqa' else r'[-+]?\d[\d,]*'
 m=re.fullmatch(r'(?:Final answer|Answer):\s*\(?\s*('+token+r')\s*\)?\.?',lines[-1],re.I)
 if m is None:m=re.search(r'\(\s*X\s*=\s*('+token+r')\s*\)\.?$',lines[-1],re.I)
 if m is None:return None
 return m.group(1).upper() if j['task']=='commonsenseqa' else str(int(m.group(1).replace(',','')))

