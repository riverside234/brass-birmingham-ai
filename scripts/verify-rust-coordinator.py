"""L12: retained scalar parity, independent pooling, and C# complete-plan replay."""
import argparse
from collections import Counter
import copy
import gzip
import json
from pathlib import Path
import runpy
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BASE=runpy.run_path(str(ROOT/'scripts/qualify-rust-scaffold.py'))
read,write,sha,receipt=(BASE[k] for k in ('read','write','sha','receipt'))
L11=runpy.run_path(str(ROOT/'scripts/verify-rust-sessions.py'))
REF=runpy.run_path(str(ROOT/'scripts/reproduce-gumbel-coordinator.py'))
FIXTURES=L11['L10']
MODEL=L11['L09']/'model/manifest.json'
PRIOR=ROOT/'artifacts/rust-backend/20261025-l11-qualified-v2'


def execute(binary,body,out,name):
    source=out/(name+'-input.json');target=out/(name+'-output.json')
    source.write_text(json.dumps(body,separators=(',',':'),allow_nan=False),encoding='utf-8')
    subprocess.run([binary,source,target],check=True,timeout=600,
                   creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    return read(target)['results']


def verify(pair,sessions,out):
    out.mkdir(parents=True,exist_ok=True)
    index=read(FIXTURES/'index.json')
    sources=[receipt(FIXTURES/'index.json'),receipt(PRIOR/'qualification.json')]
    common=dict(manifest=str(MODEL),manifest_sha256=sha(MODEL))
    scalar_cases=[];pair_cases=[]
    for item in index:
        source=FIXTURES/(item['name']+'-request.json');sources.append(receipt(source))
        case=read(source)['cases'][0];case.pop('kind')
        case.update(mode=dict(kind='count'),allowance_ms=60000,
                    cooperative=dict(cap=1,pending_limit=1,max_candidate_cells=8192,flush_us=0))
        scalar_cases.append(case)
        inp=copy.deepcopy(case['input']);inp['shuffle_tape']=None
        pair_cases.append(dict(name=item['name'],input=inp,config=case['config'],mode=dict(kind='count'),allowance_ms=60000,
                               policy=dict(cap=4,pending_limit=4,max_candidate_cells=32768,flush_us=2000)))
    scalar={}
    for mode in ('recorded','native'):
        cases=scalar_cases if mode=='recorded' else [dict(c,replies=None) for c in scalar_cases]
        scalar[mode]=execute(sessions,dict(common,cases=cases),out,'singleton-'+mode)
        original=PRIOR/'release-plans'/('recorded-output.json' if mode=='recorded' else 'native-output.json')
        sources.append(receipt(original));old=read(original)['results']
        for actual,previous in zip(scalar[mode],old):
            a=actual['decisions'][0];b=previous['decisions'][0]
            for field in ('action_ids','actions','snapshot','completed','partial_completed','gumbels','plan_events','effective_k'):
                L11['equal'](a[field],b[field],path='singleton/'+field)
        print(f'{mode}: all 44 singleton roots retain L11 counts, returns and plans',flush=True)
    results=execute(pair,dict(common,cases=pair_cases),out,'paired-native')
    wanted={item['root'] for item in index if '-authored-seat-' not in item['root']}
    requests={}
    with gzip.open(L11['CAPTURE']/'roots.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row['id'] in wanted: requests[row['id']]=row['request']
    assert set(requests)==wanted
    oracle_path=L11['CAPTURE']/'exporter/RulesExporter.exe'
    assert sha(oracle_path)==read(L11['CAPTURE']/'capture.json')['exporter_sha256']
    oracle=runpy.run_path(str(ROOT/'scripts/capture-rust-rules.py'))['Oracle'](oracle_path)
    coverage=Counter();cases=[]
    try:
        for item,row in zip(index,results):
            assert row['result']['status']=='complete',(item['name'],row['result'])
            decision=row['result']['decision'];lanes=decision['lanes'];selection=decision['selection'];result=decision['result']
            assert decision['sessions']==2 and decision['threads_per_session']==1
            assert len({l['owner'] for l in lanes})==2
            searched=item['phase']!='debt' and item['candidates']>1
            for lane in lanes:
                value=lane['outcome'];n=item['simulations'] if searched else 0
                assert value['completed']==n and value['retired_pending']==0
                assert value['schedule']['max_pending']<=4
                if searched:
                    assert value['snapshot']['support_watermark']==sum(value['snapshot']['stats']['count'])==n
                    assert value['partial_completed']==0
            if searched:
                assert lanes[0]['outcome']['gumbels']==lanes[1]['outcome']['gumbels']
                pooled=selection['pool'];stats=pooled['stats']
                # The oracle accepts uniforms; invert g=-log(-log(u)).
                import math
                uniforms=[math.exp(-math.exp(-g)) for g in lanes[0]['outcome']['gumbels']]
                snapshots=[dict(count=l['outcome']['snapshot']['stats']['count'],returns=l['outcome']['snapshot']['stats']['returns'],
                                survivors=l['outcome']['snapshot']['survivors'],watermark=l['outcome']['snapshot']['support_watermark']) for l in lanes]
                expected=REF['merge'](dict(logits=stats['logits'],value=stats['value'],uniforms=uniforms,actor=item['actor'],snapshots=snapshots))
                L11['equal'](pooled['scores'],expected['scores'],tol=1e-10)
                assert pooled['ranking']==expected['ranking'] and stats['count']==expected['count']
                L11['equal'](stats['returns'],expected['returns'],tol=1e-10)
                if selection['source_session'] is not None:
                    lane=lanes[selection['source_session']]['outcome']
                    assert selection['actions'] in lane['eligible_plans']
                    assert all(2<=a['support']<=lane['snapshot']['support_watermark'] for a in result['actions'])
                else:
                    assert all(a['support']==0 for a in result['actions'])
            assert result['action_ids'] and decision['response_ready_ns']<result['budget']['deadline_ns']
            steps=0
            if item['root'] in requests:
                current=oracle.call(**requests[item['root']],projection=True)
                for action in result['actions']:
                    for step in action['action']['steps']:
                        view=step['precondition']['key']['information']['observation']['view']
                        L11['equal'](view,current['snapshot']['views'][view['seat']],path='C# permitted view')
                        assert step['choice'] in [c['id'] for c in current['choices']]
                        current=oracle.call(choice=step['choice'],projection=True);steps+=1
            coverage[selection['provenance']]+=1
            coverage['trajectories']+=sum(l['outcome']['completed'] for l in lanes)
            coverage['complete_actions']+=len(result['actions']);coverage['csharp_steps']+=steps
            coverage['batches']+=sum(l['neural']['batches'] for l in lanes)
            coverage['positions']+=sum(l['neural']['positions'] for l in lanes)
            cases.append(dict(**item,csharp_steps=steps,provenance=selection['provenance'],
                              actual_max_batch=max(l['neural']['max_batch'] for l in lanes)))
            print(item['name'],selection['provenance'],steps,flush=True)
    finally:
        oracle.close()
    assert len(cases)==44 and coverage['trajectories']==1536
    assert max(c['actual_max_batch'] for c in cases)>1
    report=dict(status='passed',identity='brass-gumbel-is-mcts/coordinator-v1',cases=cases,coverage=dict(coverage),
                singleton_roots_per_mode=44,root_matrix=[(i['era'],i['phase'],i['actor']) for i in index[:40]],
                sources=sources,binaries=[receipt(pair),receipt(sessions)],oracle=receipt(oracle_path),
                artifacts=[receipt(p) for p in out.glob('*-*.json')],
                interpretation='Count correctness with 60-second watchdog. Five rotated-seat cases retain L10 native evidence; 39 original cases replay in C#.')
    write(out/'verification.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pair',type=Path,required=True)
    parser.add_argument('--sessions',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();verify(args.pair.resolve(),args.sessions.resolve(),args.out.resolve())
