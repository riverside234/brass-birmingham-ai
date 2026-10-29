"""Bounded frozen-binary equivalence and focus cancellation checks."""
import copy
import argparse
import runpy
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
T=runpy.run_path(str(ROOT/'scripts/search-v2-tools.py'));read,write,receipt=(T[k] for k in ('read','write','receipt'))
p=argparse.ArgumentParser();p.add_argument('--label',default='native-integration');p.add_argument('--candidate-binary',type=Path)
p.add_argument('--focus-method',choices=['search_v2_focus_only','search_v2'],default='search_v2_focus_only');args=p.parse_args()
out=T['OUT']/args.label;out.mkdir(exist_ok=False)
study=T['OUT']/'study';protocol=read(study/'protocol.json');model=ROOT/protocol['model']['path']
candidate_binary=args.candidate_binary.resolve() if args.candidate_binary else ROOT/protocol['binary']['path']
cases=[]
for name in ('brass-003','brass-026'):
    case=read(study/'fixtures'/f'{name}.json')['cases'][0]
    case.update(name=name,mode=dict(kind='count'),allowance_ms=4000,
        policy=dict(cap=1,pending_limit=1,max_candidate_cells=8192,flush_us=0))
    case['config'].update(simulations=64,k=8,trace=False);cases.append(case)


def execute(label,binary,cases):
    source=out/(label+'-input.json');target=out/(label+'-output.json')
    write(source,dict(manifest=str(model),manifest_sha256=protocol['model']['sha256'],cases=cases))
    with (out/(label+'.log')).open('wb') as log:
        subprocess.run([str(binary),str(source),str(target)],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=25,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    return read(target)['results']


control=execute('frozen-control',ROOT/protocol['control_binary']['path'],cases)
for c in cases:c['method']='search_v2_no_switch'
candidate=execute('no-switch',candidate_binary,cases)
fields=('actions','action_ids','gumbels','effective_k','snapshot','completed','attempted',
        'partial_completed','retired_pending','plan_events','cache_hits','eligible_plans')
checks=0
for a,b in zip(control,candidate):
    assert a['result']['status']==b['result']['status']=='complete'
    x,y=(r['result']['decision'] for r in (a,b))
    for p,q in zip(x['lanes'],y['lanes']):
        for key in ('simulations','estimate','mode'):
            assert p['outcome']['quota'][key]==q['outcome']['quota'][key];checks+=1
        for key in fields:
            assert p['outcome'][key]==q['outcome'][key],(a['name'],key);checks+=1
    assert x['result']['action_ids']==y['result']['action_ids'];checks+=1
timed=copy.deepcopy(read(study/'brass-003-4s_k8-input.json')['cases'][-1])
timed.update(name='cancel-after-focus-retirement',cancel_on_focus=True,method=args.focus_method)
later=copy.deepcopy(cases[0]);later.update(name='new-turn-after-cancel',method='search_v1');later['config']['simulations']=16
cancel=execute('focus-cancel',candidate_binary,[timed,later])
first,last=cancel
assert first['cancelled'] and first['busy_after_cancel'] and first['blocked_admission']
assert not first['busy_after_retirement'] and first['result']['status']=='retired'
assert last['result']['status']=='complete';checks+=5
write(out/'result.json',dict(passed=True,checks=checks,scope='2 frozen-control count-64 comparisons; cancellation after both old owners retire; subsequent new turn',
    cancellation_method=args.focus_method,
    control_binary=receipt(ROOT/protocol['control_binary']['path']),candidate_binary=receipt(candidate_binary),
    fixtures=[receipt(p) for p in sorted(out.glob('*-input.json'))]))
print('Passed',checks,'frozen-control/cancellation checks')
