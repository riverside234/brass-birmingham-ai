"""Replay compact production-transport decisions against frozen C# rules."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import runpy
import time

ROOT=Path(__file__).resolve().parents[1]
def verify(package,out):
    os.environ['BRASS_LOCAL_PACKAGE']=str(package)
    api=runpy.run_path(str(ROOT/'tests/RustBackendLocal/test_local.py'))
    base=runpy.run_path(str(ROOT/'scripts/verify-rust-sessions.py'))
    read,write,receipt=base['read'],base['write'],base['receipt']
    index=read(base['L10']/'index.json');wanted={r['root'] for r in index if '-authored-seat-' not in r['root']}
    requests={}
    with gzip.open(base['CAPTURE']/'roots.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            if row['id'] in wanted:requests[row['id']]=row['request']
    assert set(requests)==wanted
    oracle_path=base['CAPTURE']/'exporter/RulesExporter.exe'
    assert base['sha'](oracle_path)==read(base['CAPTURE']/'capture.json')['exporter_sha256']
    oracle=runpy.run_path(str(ROOT/'scripts/capture-rust-rules.py'))['Oracle'](oracle_path)
    worker=api['Worker'](package);rows=[];coverage=Counter();out.mkdir(parents=True,exist_ok=True)
    try:
        for i,item in enumerate(index):
            request=worker.search(i,generation=i+2);start=time.monotonic();worker.send(request);response=worker.recv()
            assert response['status']=='ok',(item,response)
            data=response['data'];steps=0
            if item['root'] in requests:
                current=oracle.call(**requests[item['root']],projection=True)
                for action in data['actions']:
                    for choice in action:
                        assert choice in [c['id'] for c in current['choices']],(item,choice)
                        current=oracle.call(choice=choice,projection=True);steps+=1
            assert data['retired'] and data['response_ready_ns']<data['deadline_ns']
            coverage[item['phase']]+=1;coverage['csharp_choices']+=steps;coverage['actions']+=len(data['actions'])
            rows.append(dict(**item,response=response,csharp_choices=steps,wall_seconds=time.monotonic()-start))
            print(item['name'],data['provenance'],steps,flush=True)
    finally:worker.close();oracle.close()
    write(out/'verification.json',dict(status='passed',protocol=api['PROTOCOL'],cases=rows,coverage=dict(coverage),readiness=worker.hello,
        artifacts=[receipt(package/'build.json'),receipt(base['L10']/'index.json'),receipt(oracle_path)],
        interpretation='44 compact production replies; 39 original roots replay in C#. Five rotated-seat roots retain native evidence. Four-second allowance; not L14 matched or strength qualification.'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();verify(a.package.resolve(),a.out.resolve())
