"""L11: independent plan/support extraction from sealed L10 traces plus C# replay.

The corpus is unchanged. Count runs are correctness diagnostics with a watchdog,
not per-root speed measurements or the later matched four-second benchmark.
"""
import argparse
from collections import Counter, defaultdict
import copy
import gzip
import json
from pathlib import Path
import runpy
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BASE=runpy.run_path(str(ROOT/'scripts/qualify-rust-scaffold.py'))
read,write,sha,receipt=(BASE[k] for k in ('read','write','sha','receipt'))
L10=ROOT/'artifacts/rust-backend/20261025-l10-capture'
L09=ROOT/'artifacts/rust-backend/20261025-l09-qualified-v2'
CAPTURE=ROOT/'artifacts/rust-backend/20261025-l05-capture-v2'

def turn(info):
    c=info['key']['information']['observation']['view']['context']
    return tuple(c[k] for k in ('era','round','turn','turn_owner'))+(info['actor'],c['phase']=='debt')

def support(reference,minimum=2):
    entries={}
    root=reference['nodes'][0]['info']
    for serial,t in enumerate(reference['trajectories'],1):
        actions=[];steps=[]
        for node,edge in t['path']:
            info=reference['nodes'][node]['info']
            if turn(info)!=turn(root):break
            choice=info['key']['choices'][edge]
            if choice['kind']=='liquidate':break
            steps.append(dict(precondition=info,choice=choice['id']))
            if choice['kind']=='finish':
                actions.append(dict(steps=steps));steps=[]
                key=json.dumps(actions,sort_keys=True,separators=(',',':'))
                e=entries.setdefault(key,dict(prefix=copy.deepcopy(actions),events=[]))
                e['events'].append((serial,t['edge_values'][0]))
                if len(actions)==2:break
    def select(first,watermark):
        prefix=[];chosen=[]
        for depth in (1,2):
            eligible=[]
            for e in entries.values():
                p=e['prefix']
                if len(p)!=depth or p[:-1]!=prefix or p[0]['steps'][0]['choice']!=first:continue
                events=[v for at,v in e['events'] if at<=watermark]
                if len(events)<minimum:continue
                returns=[sum(v[s] for v in events) for s in range(4)]
                ids=[step['choice'] for a in p for step in a['steps']]
                eligible.append(((-len(events),-returns[root['actor']]/len(events),ids),p,len(events),returns))
            if not eligible:break
            _,prefix,count,returns=min(eligible,key=lambda e:e[0])
            chosen.append(dict(action=prefix[-1],support=count,returns=returns))
        return chosen
    return select,sum(len(e['events']) for e in entries.values())

def equal(a,b,tol=2e-4,path=''):
    if isinstance(a,dict):
        for k,v in a.items():equal(v,b[k],tol,path+'/'+k)
    elif isinstance(a,list):
        assert len(a)==len(b),(path,len(a),len(b))
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,tol,path+'/'+str(i))
    elif isinstance(a,float):assert abs(a-b)<=tol,(path,a,b)
    else:assert a==b,(path,a,b)

def execute(binary,request,output):
    subprocess.run([str(binary),str(request),str(output)],check=True,timeout=240,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    return read(output)

def verify(binary,out):
    out.mkdir(parents=True,exist_ok=True)
    index=read(L10/'index.json')
    source=[receipt(L10/'index.json')]
    native_cases=[];taped_cases=[]
    for item in index:
        path=L10/(item['name']+'-request.json');source.append(receipt(path))
        case=read(path)['cases'][0];case.pop('kind')
        case.update(mode=dict(kind='count'),allowance_ms=60000)
        taped_cases.append(case)
        native_cases.append(dict(case,replies=None))
    manifest=L09/'model/manifest.json'
    common=dict(manifest=str(manifest),manifest_sha256=sha(manifest))
    results={}
    for label,cases in [('recorded',taped_cases),('native',native_cases)]:
        request=out/(label+'-request.json');output=out/(label+'-output.json')
        request.write_text(json.dumps(dict(common,cases=cases),separators=(',',':'),allow_nan=False),encoding='utf-8')
        results[label]=execute(binary,request,output)['results']
    wanted={x['root'] for x in index if '-authored-seat-' not in x['root']}
    requests={}
    with gzip.open(CAPTURE/'roots.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['id'] in wanted:requests[r['id']]=r['request']
    assert set(requests)==wanted
    capture=read(CAPTURE/'capture.json');oracle_exe=CAPTURE/'exporter/RulesExporter.exe'
    assert sha(oracle_exe)==capture['exporter_sha256']
    oracle=runpy.run_path(str(ROOT/'scripts/capture-rust-rules.py'))['Oracle'](oracle_exe)
    coverage=Counter();items=[]
    try:
        for n,item in enumerate(index):
            expected=read(L10/(item['name']+'-expected.json'));source.append(receipt(L10/(item['name']+'-expected.json')))
            select,events=support(expected)
            recorded=results['recorded'][n]['decisions'][0];native=results['native'][n]['decisions'][0]
            # Clock durations differ; everything affecting evidence and actions must agree.
            for field in ('action_ids','actions','provenance','snapshot','completed','partial_completed','effective_k','plan_events','gumbels'):
                equal(recorded[field],native[field],path=field)
            searched=item['phase']!='debt' and item['candidates']>1
            if searched:
                assert native['completed']==item['simulations'] and native['partial_completed']==0
                equal(expected['nodes'][0]['count'],native['snapshot']['stats']['count'])
                equal(expected['nodes'][0]['returns'],native['snapshot']['stats']['returns'])
                first=expected['selected_action'];supported=select(first,item['simulations'])
                assert events==native['plan_events']
                if supported:
                    assert native['provenance']=='search_supported'
                    equal(supported,native['actions'])
                else:
                    assert all(a['support']==0 for a in native['actions'])
                    assert native['provenance']=='bounded_policy_completion'
            else:
                assert native['completed']==native['attempted']==native['plan_events']==0
                assert native['snapshot'] is None and native['quota'] is None
                assert native['provenance']==('debt_policy' if item['phase']=='debt' else 'forced_completion')
            assert native['action_ids'] and native['response_ready_ns']<native['budget']['deadline_ns']
            assert native['budget']['reserve_ns']==1_200_000_000
            csharp_steps=0
            if item['root'] in requests:
                current=oracle.call(**requests[item['root']],projection=True)
                for action in native['actions']:
                    for j,step in enumerate(action['action']['steps']):
                        view=step['precondition']['key']['information']['observation']['view']
                        equal(view,current['snapshot']['views'][view['seat']],path='csharp-view')
                        assert step['choice'] in [c['id'] for c in current['choices']]
                        assert (step['choice'].startswith('finish|') or step['choice'].startswith('liquidate|'))==(j==len(action['action']['steps'])-1)
                        current=oracle.call(choice=step['choice'],projection=True);csharp_steps+=1
            coverage[native['provenance']]+=1;coverage['searched_trajectories']+=native['completed'];coverage['csharp_steps']+=csharp_steps
            coverage['complete_actions']+=len(native['actions'])
            items.append(dict(**item,provenance=native['provenance'],support=[a['support'] for a in native['actions']],csharp_steps=csharp_steps))
            print(item['name'],native['provenance'],native['completed'],flush=True)
    finally:oracle.close()
    assert coverage['search_supported']>0
    assert coverage['debt_policy']>0 and coverage['forced_completion']>0
    report=dict(status='passed',identity='brass-gumbel-is-mcts/session-v1',cases=items,coverage=dict(coverage),
        root_matrix=[(i['era'],i['phase'],i['actor']) for i in index[:40]],binary=receipt(binary),
        oracle=receipt(oracle_exe),sources=source,artifacts=[receipt(p) for p in out.glob('*-*.json')],
        interpretation='Scalar fixed-count plan correctness with a 60-second watchdog; not a whole-turn speed benchmark',
        rotated_fixtures='Five cyclic-seat fixtures use sealed L10 search/rules evidence; other 39 use direct C# complete-action replay')
    write(out/'verification.json',report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    verify(a.binary.resolve(),a.out.resolve())
