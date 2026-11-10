"""Verify complete retained response bodies, payload identities and record linkage."""
from pathlib import Path
import gzip,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def canonical(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def main():
 coverage=json.loads((ROOT/'data/response_export_coverage.json').read_text());out={}
 for study,meta in coverage.items():
  path=ROOT/f'data/{study}_complete_responses.jsonl.gz';assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256'];seen=set();bodies=0
  with gzip.open(path,'rt') as f:
   for line in f:
    r=json.loads(line);key=r['request_key'];assert key not in seen;seen.add(key)
    assert hashlib.sha256(canonical(r['request'])).hexdigest()==key
    assert r['result']['request_key']==key
    for a in r['attempts']:
     body=a['response_body'].encode('utf-8');assert len(body)==a['metadata']['body_bytes'];assert hashlib.sha256(body).hexdigest()==a['metadata']['body_sha256'];bodies+=1
     assert not any(k.lower() in ['authorization','x-api-key','api-key'] for k in a['metadata']['response_headers'])
  assert len(seen)==meta['requests'];out[study]={'logical_requests':len(seen),'body_hashes_checked':bodies,'identity_hashes_checked':len(seen)}
 (ROOT/'data/verified_response_exports.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
