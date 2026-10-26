"""Prepare/replay L10 toy and full Brass Gumbel traces; preserve raw evidence."""
import argparse
from collections import Counter
import copy
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import runpy
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'artifacts/rust-backend/20261025-l09-qualified-v2'
spec=importlib.util.spec_from_file_location('gumbel_brass_reference',ROOT/'tests/RustBackend/gumbel_brass_reference.py')
gp=importlib.util.module_from_spec(spec);sys.modules[spec.name]=gp;spec.loader.exec_module(gp)
ref=gp.ref

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,data):Path(path).write_text(json.dumps(data,allow_nan=False,separators=(',',':')),encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(folder,file):
    with gzip.open(ROOT/'artifacts/rust-backend'/folder/file,'rt',encoding='utf-8') as f:
        for line in f:yield json.loads(line)
def config(n=32,k=8):return dict(simulations=n,k=k,seed=7,c1=50.,c2=.1,max_depth=256,max_nodes=16384,max_edges=2000000,trace=True)

def compare(a,b,path='',tol=1e-10):
    if isinstance(a,dict):
        assert isinstance(b,dict),path
        for k,v in a.items():assert k in b,(path,k);compare(v,b[k],path+'/'+k,tol)
    elif isinstance(a,list):
        assert isinstance(b,list) and len(a)==len(b),(path,len(a),len(b))
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i),tol)
    elif isinstance(a,float):
        assert isinstance(b,(int,float)) and math.isfinite(b) and math.isclose(a,b,abs_tol=tol,rel_tol=tol),(path,a,b)
    else:assert a==b,(path,a,b)

def causal_roots(reward):
    from brass_ai.learning.reward import CostLedger
    roots={}
    for r in rows('20261025-l06-capture','roots.jsonl.gz'):
        roots[(r['id'].split('-root-')[0],r['state']['context']['revision'])]=r
    found=[];ledger=None;previous=None
    for row in rows('20261025-l05-capture-v2','games.jsonl.gz'):
        if 'initial' in row:
            ledger=CostLedger(reward);current=row['initial']
        else:
            if 'choice' in row['command']:
                actor=previous['state']['context']['actor']
                choice=next(c for c in previous['choices'] if c['id']==row['command']['choice'])
                ledger.chosen({k:previous['state'][k] for k in ('context','players','slots')},choice)
            current=row['expected']
        ctx=current['state']['context']
        if not ctx['chance']:
            ledger.observe({k:current['state'][k] for k in ('context','players','slots')})
            root=roots.get((row['game'],ctx['revision']))
            if root and ctx['actor']>=0 and not ctx['terminal']:
                found.append((root,copy.deepcopy(ledger.snapshot()),copy.deepcopy(current['choices'])))
        previous=current
    return found

def prepare(out,limit):
    from brass_ai.native.protocol import NativeWorker
    out.mkdir(parents=True,exist_ok=False)
    # Use the sealed simulator, not the newly compiled search binary.
    binary=out/'reference-worker.exe'
    with zipfile.ZipFile(PRIOR/'executables.zip') as z:binary.write_bytes(z.read('release/brass-ai-worker.exe'))
    manifest=read(PRIOR/'model/manifest.json');metadata=read(PRIOR/'model/model.json');reward=metadata['model_config']['utility'] if 'model_config' in metadata else metadata['config']['utility']
    write(out/'provenance.json',dict(reference_archive_sha256=sha(PRIOR/'executables.zip'),reference_worker_sha256=sha(binary),manifest_sha256=sha(PRIOR/'model/manifest.json'),reference_script_sha256=sha(ROOT/'scripts/reproduce-gumbel-is-mcts.py'),g_py_sha256=sha(ROOT/'tests/RustBackend/gumbel_brass_reference.py')))
    selected={}
    roots=causal_roots(reward)
    for row,ledger,choices in roots:
        ctx=row['state']['context'];cell=(ctx['era'],ctx['phase'],ctx['actor'])
        if cell not in selected or len(choices)>len(selected[cell][2]):selected[cell]=(row,ledger,choices)
    # Complete missing actor/phase cells using the already qualified L08/L09
    # cyclic-seat transformation; label authored cases, never as new real games.
    def rotate(value,delta):
        if isinstance(value,list):return [rotate(v,delta) for v in value]
        if not isinstance(value,dict):return value
        result={k:rotate(v,delta) for k,v in value.items()}
        for k in ('seat','actor','turn_owner','owner'):
            if k in result and type(result[k]) is int and 0<=result[k]<4:result[k]=(result[k]+delta)%4
        if 'order' in result:result['order']=[(s+delta)%4 for s in result['order']]
        if 'players' in result:result['players'].sort(key=lambda p:p['seat'])
        if 'level' in result and 'seat' in result and 'id' in result:result['id']=str(result['seat'])+':'+value['id'].split(':',1)[1]
        return result
    for era in ('CanalEra','RailEra'):
        for phase in ('action','continuation','payment','bonus','debt'):
            source=next(v for k,v in selected.items() if k[:2]==(era,phase))
            for actor in range(4):
                if (era,phase,actor) in selected:continue
                row,ledger,choices=copy.deepcopy(source);old=row['state']['context']['actor'];delta=(actor-old)%4
                row['id']+=f'-authored-seat-{actor}'
                row['state']['context']=rotate(row['state']['context'],delta)
                row['views']=[rotate(row['views'][(s-delta)%4],delta) for s in range(4)]
                row['recall']=[row['recall'][(s-delta)%4] for s in range(4)]
                ledger['spent']=[ledger['spent'][(s-delta)%4] for s in range(4)]
                selected[(era,phase,actor)]=(row,ledger,choices)
    cases=list(selected.values())
    # Add late Rail roots to reach terminal settlement, not only first-step leaves.
    late=sorted((x for x in roots if x[0]['state']['context']['era']=='RailEra'),key=lambda x:(sum(len(h) for h in x[0]['state']['cards']['hands']),-x[0]['state']['context']['revision']))
    cases += late[:4]
    if limit:cases=cases[:limit]
    evaluator=gp.Neural(PRIOR/'model/model.onnx')
    index=[]
    with NativeWorker(binary,timeout=60) as worker:
        for i,(row,ledger,choices) in enumerate(cases):
            view=row['views'][row['state']['context']['actor']]
            remaining=[max(0.,reward['insolvency']['max_cost']-v) for v in ledger['spent']]
            ctx=dict(model_id=manifest['model_id'],reward_id=manifest['reward_id'],profile_id='gumbel-l10-neutral',style=[0.]*4,intent=[0.]*8,remaining=remaining,max_cost=reward['insolvency']['max_cost'],stable_descriptors=None)
            cfg=config(96 if i>=len(selected) else 32,1 if i>=len(selected) else 8)
            draw_count=worker.simulation(0,'prepare',view=view,recall=row['recall'][view['seat']])['draw_count']
            rng=random.Random(10890+i)
            shuffles=[[rng.randrange(bound) for bound in range(draw_count+1,1,-1)] for _ in range(cfg['simulations'])]
            inp=dict(view=view,recall=row['recall'][view['seat']],context=ctx,reward=reward,ledger=ledger,shuffle_tape=shuffles,sample_seed=0)
            evaluator.replies=[]
            expected=gp.search(worker,inp,cfg,evaluator)
            request=dict(kind='brass',config=cfg,draws=expected['draws'],input=inp,replies=evaluator.replies)
            name=f'brass-{i:03d}';write(out/(name+'-request.json'),dict(cases=[request]));write(out/(name+'-expected.json'),expected)
            index.append(dict(name=name,root=row['id'],phase=view['context']['phase'],era=view['context']['era'],actor=view['seat'],candidates=len(choices),late=i>=len(selected),simulations=cfg['simulations']))
            print(name,row['id'],view['context']['phase'],'nodes',len(expected['nodes']),'terminal',sum(t['leaf']=='terminal' for t in expected['trajectories']),flush=True)
    toy=[];expected=[]
    for players in (2,4):
        for n,k in [(0,16),(1,1),(2,16),(7,7),(23,16),(24,16),(32,16),(200,16)]:
            py=ref.search(ref.HiddenOffer(players),simulations=n,k=k,seed=7)
            toy.append(dict(kind='toy',players=players,config=config(n,k),draws=py['draws']))
            expected.append(gp.canonical(py))
    golden=read(ROOT/'tests/GumbelISMCTS/fixtures/reference-v1.json')
    cfg=config(golden['config']['simulations'],golden['config']['k']);cfg.update(golden['config'])
    toy.append(dict(kind='toy',players=golden['players'],config=cfg,draws=golden['draws']));expected.append(golden)
    write(out/'toy-request.json',dict(cases=toy));write(out/'toy-expected.json',expected)
    write(out/'index.json',index)

def toy_compare(expected,actual):
    translated=copy.deepcopy(actual)
    translated['version']=expected['version'];translated['config']=expected['config']
    players=len(expected['nodes'][0]['value'])
    for node in translated['nodes']:
        inf=node['info'];node['info']=dict(player=inf['actor'],private=inf['key']['private'],public=inf['key']['public'],actions=inf['actions'])
        node['value']=node['value'][:players];node['returns']=[r[:players] for r in node['returns']]
    for t in translated['trajectories']:t['value']=t['value'][:players]
    translated['players']=players
    compare(expected,translated)

def verify(out,binary,report,limit=0):
    checked=[];matrix=Counter();actors=Counter();phases=Counter();leaf_kinds=Counter();reused=0;max_depth=0;cost_edges=0
    subprocess.run([binary,out/'toy-request.json',report.parent/(report.stem+'-toy.json')],check=True,timeout=120)
    for e,a in zip(read(out/'toy-expected.json'),read(report.parent/(report.stem+'-toy.json'))['results']):toy_compare(e,a)
    for item in read(out/'index.json')[:limit or None]:
        name=item['name'];request=read(out/(name+'-request.json'));expected=read(out/(name+'-expected.json'))
        target=report.parent/(report.stem+'-'+name+'.json')
        subprocess.run([binary,out/(name+'-request.json'),target],check=True,timeout=180)
        actual=read(target)['results'][0];compare(expected,actual)
        # Independent per-edge ledger reconstruction catches common sign/cost errors.
        for i,node in enumerate(actual['nodes']):
            actors[node['info']['actor']]+=1
            phases[node['info']['key']['information']['observation']['view']['context']['phase']]+=1
            for edge,count in enumerate(node['count']):
                values=[t['edge_values'][j] for t in actual['trajectories'] for j,p in enumerate(t['path']) if p==[i,edge]]
                assert count==len(values)
                compare([sum(v[s] for v in values) for s in range(4)],node['returns'][edge])
                if i>0:reused+=count
        for t in actual['trajectories']:
            leaf_kinds['terminal' if t['leaf']=='terminal' else 'neural']+=1
            max_depth=max(max_depth,len(t['path'])+len(t['chance']))
            cost_edges+=sum(v!=t['value'] for v in t['edge_values'])
        matrix['/'.join(map(str,[item['era'],item['phase'],item['actor']]))]+=1
        # Replay with native Features-4 + ORT, same model/world/chance draws.
        request['cases'][0]['replies']=None
        request['manifest']=str(PRIOR/'model/manifest.json');request['manifest_sha256']=sha(PRIOR/'model/manifest.json')
        native_request=report.parent/(report.stem+'-'+name+'-native-request.json');write(native_request,request)
        native_target=report.parent/(report.stem+'-'+name+'-native.json')
        subprocess.run([binary,native_request,native_target],check=True,timeout=180)
        native=read(native_target)['results'][0]
        # Existing ORT parity tolerance; selections/counts/actions remain exact.
        compare(expected,native,tol=2e-4)
        checked.append(dict(**item,metrics=native['metrics'],neural=native['neural'],trace_sha256=sha(target),native_trace_sha256=sha(native_target)))
        print(report.stem,name,'passed',flush=True)
    write(report,dict(status='passed',identity='brass-gumbel-is-mcts/count-v1',binary_sha256=sha(binary),toy_cases=len(read(out/'toy-expected.json')),brass_cases=checked,root_matrix=matrix,node_actors=actors,node_phases=phases,leaf_kinds=leaf_kinds,nonroot_completed_edges=reused,max_trajectory_steps=max_depth,cost_adjusted_edges=cost_edges))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--limit',type=int,default=0);p.add_argument('--binary',type=Path);p.add_argument('--report',type=Path)
    a=p.parse_args()
    if a.prepare:prepare(a.fixtures,a.limit)
    else:verify(a.fixtures,a.binary,a.report,a.limit)
